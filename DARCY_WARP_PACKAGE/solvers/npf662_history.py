"""Explicit NPF/STO history driver retaining intermediate heads on the device.

Geometry, K, storage fields, cell types and prescribed heads are fixed within
one call. The caller may update physical K between calls. Source multipliers
and timestep lengths vary per step. All existing physical acceptance gates
remain mandatory; a failed step stops the history.
"""
from dataclasses import replace
from time import perf_counter

import numpy as np
import warp as wp

from .npf662_3d import case_from_model
from .npf662_newton import NPF662NewtonControls, solve_newton_device
from .npf662_kcycle import NPF662KCycleControls
from . import npf662_kernels as kernels
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference


class DeviceHistoryReference:
    """Construct the independent oracle only if device grounding is insufficient."""
    def __init__(self, *, case, workspace, ss, sy):
        self.case = case
        self.workspace = workspace
        self.ss = ss
        self.sy = sy
        self.dt = None
        self.factor = None

    def evaluate(self, *, head, **tolerances):
        previous = self.workspace.storage_previous.numpy().reshape(self.case.active.shape).copy()
        case = replace(self.case, source=self.case.source*self.factor)
        return NPF662TransientReference(case=case, previous=previous, dt=self.dt,
            ss=self.ss, sy=self.sy).evaluate(head=head, **tolerances)


def prepare_next_step(*, workspace, source, factor, dt, area):
    """Freeze the accepted previous head and recompute storage/scaling in place."""
    wp.copy(dest=workspace.storage_previous, src=workspace.head)
    wp.launch(kernel=kernels.scale_period_source, dim=workspace.n,
        inputs=[source, workspace.source, factor], device=workspace.device)
    workspace.storage_factor = area/dt
    wp.launch(kernel=kernels.storage_terms, dim=workspace.n,
        inputs=[workspace.storage_previous, workspace.storage_previous, workspace.bottom, workspace.thickness,
            workspace.types, workspace.storage_ss, workspace.storage_sy, workspace.free,
            workspace.storage_rate, workspace.previous_storage_tangent, workspace.storage_factor], device=workspace.device)
    workspace.refresh_previous_operator()
    workspace.maximum.zero_()
    wp.launch(kernel=kernels.previous_storage_scale, dim=workspace.n,
        inputs=[workspace.principal, workspace.previous_operator_tangent, workspace.scale, workspace.maximum],
        device=workspace.device)
    if workspace.maximum.numpy()[0] != 0.:
        raise ValueError('Invalid device previous-state storage tangent')
    workspace.faces_changed = False
    workspace.jv_counter.zero_()
    workspace.preconditioner_counter.zero_()
    workspace.residual_evaluations = workspace.candidate_evaluations = 0


def solve_transient_history(*, model, source_factors, dt, head_prev, ss, sy, **solver_options):
    started = perf_counter()
    if model.formulation != 'mf6_6_6_2_npf' or model._session is None:
        raise ValueError('A built mf6_6_6_2_npf model is required')
    factors = np.array(source_factors, dtype=float, copy=True)
    if factors.ndim != 1 or not factors.size or not np.isfinite(factors).all():
        raise ValueError('Source factors must be a nonempty finite vector')
    times = np.broadcast_to(np.asarray(dt, dtype=float), factors.shape).copy()
    if not np.isfinite(times).all() or np.any(times <= 0.):
        raise ValueError('Positive finite timestep lengths required')
    options = dict(solver_options)
    for name in ('transient', 'npf_device_native', 'return_info'):
        if not isinstance(options.get(name, True), (bool, np.bool_)) or not options.pop(name, True):
            raise ValueError(f'The NPF history driver requires {name}=True')
    if options.setdefault('npf_preconditioner', 'kcycle') != 'kcycle':
        raise ValueError('The NPF history driver currently requires the kcycle preconditioner')
    initial = np.asarray(head_prev, dtype=float).copy()
    source_host = model._session.rhs.copy()
    old_initial = None if model._session.initial_head is None else model._session.initial_head.copy()
    types = options.get('npf_cell_type', int(options.get('unconfined', False)))
    base = replace(case_from_model(model=model, top=options.get('top_elevation', 0.), cell_type=types),
        source=source_host, initial=initial)
    periods = []
    workspace = None
    head_readbacks = 0
    try:
        model.update_rhs(rhs=source_host*factors[0])
        model.update_initial_head(initial_head=initial)
        first_start = perf_counter()
        first_head, first_info = model.solve(transient=True, npf_device_native=True, return_info=True,
            dt=float(times[0]), head_prev=initial, ss=ss, sy=sy, **options)
        wp.synchronize_device(device=model.device)
        periods.append(dict(first_info, index=0, dt=float(times[0]), source_factor=float(factors[0]),
            public_step_seconds=perf_counter()-first_start))
        head_readbacks += 1
        workspace = model._session._npf662_workspace_holder['workspace']
        source = wp.array(data=source_host.ravel(), dtype=wp.float64, device=model.device)
        history = wp.zeros(shape=(len(factors), workspace.n), dtype=wp.float64, device=model.device)
        wp.copy(dest=history.flatten(), src=workspace.head, count=workspace.n)
        nonlinear = NPF662NewtonControls(**first_info['controls'])
        multigrid = NPF662KCycleControls(**first_info['multigrid']['controls'])
        state = model._session._npf662_workspace_holder['native_state']
        reference = DeviceHistoryReference(case=base, workspace=workspace, ss=state.ss, sy=state.sy)
        thresholds = first_info['final_acceptance_thresholds']
        for index in range(1, len(factors)):
            if not periods[-1]['converged']:
                break
            step_start = perf_counter()
            reference.dt = float(times[index])
            reference.factor = float(factors[index])
            prepare_next_step(workspace=workspace, source=source, factor=reference.factor,
                dt=reference.dt, area=base.dx*base.dy)
            head, info = solve_newton_device(model=model, case=base, reference=reference,
                max_outer=first_info['max_outer'], krylov_budget=first_info['max_krylov_per_outer'],
                restart=first_info['krylov_restart'], controls=nonlinear,
                physical_tolerance=thresholds['physical_rms_m3_day'],
                relative_tolerance=thresholds['relative_cell_imbalance'],
                npf_preconditioner='kcycle', npf_kcycle_controls=multigrid,
                npf_krylov_method=first_info['krylov_method'].lower(), device_native=True,
                prepared_workspace=workspace, device_sequence=True)
            head_readbacks += int(head is not None)
            wp.copy(dest=history.flatten(), src=workspace.head, dest_offset=index*workspace.n, count=workspace.n)
            periods.append(dict(info, index=index, dt=reference.dt, source_factor=reference.factor,
                public_step_seconds=perf_counter()-step_start))
        count = len(periods)
        # One transfer for the output history. Internal acceptance has already
        # checked every step; this download is inside the public driver timer.
        # NumPy retains the owning host Warp array through its base reference.
        # This per-call buffer is never reused, so another 200 MB host copy is
        # unnecessary even after the model/history workspace is closed.
        heads = history.numpy()[:count].reshape((count, *base.active.shape))
        last_previous = initial if count == 1 else heads[-2]
        state = model._session._npf662_workspace_holder['native_state']
        state.refresh(model=model, case=replace(base, source=source_host*factors[count-1], initial=last_previous),
            previous=last_previous, dt=float(times[count-1]), ss=ss, sy=sy)
        state.preconditioner_storage_device = workspace.previous_storage_tangent
        if workspace.predictor_checkpoint is not None:
            unused, last_dt, quadratic = workspace.predictor_checkpoint
            workspace.predictor_checkpoint = (heads[-1].copy(), last_dt, quadratic)
    except Exception:
        workspace = model._session._npf662_workspace_holder.get('workspace')
        if workspace is not None:
            workspace.close()
            model._session._npf662_workspace_holder.pop('workspace', None)
        model._session._npf662_workspace_holder.pop('native_state', None)
        raise
    finally:
        model.update_rhs(rhs=source_host)
        model.update_initial_head(initial_head=old_initial)
    wp.synchronize_device(device=model.device)
    return heads, dict(converged=len(periods) == len(factors) and all(row['converged'] for row in periods),
        execution='device_transient_history', periods=periods, requested_steps=len(factors),
        returned_steps=len(periods), accepted_steps=sum(row['converged'] for row in periods), seconds=perf_counter()-started,
        formulation='mf6_6_6_2_npf', device=model.device, precision='float64',
        numerical_seconds=sum(row['seconds'] for row in periods),
        solver_head_readbacks=head_readbacks, output_history_readbacks=1,
        fixed_inputs_within_history=['geometry', 'conductivity', 'cell_type', 'Ss', 'Sy', 'prescribed_heads', 'GHB conductance/head'],
        final_acceptance_thresholds=thresholds,
        timing='Entire public history call, including setup, all source/timestep updates, acceptance and output download')
