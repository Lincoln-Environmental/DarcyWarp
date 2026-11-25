"""Explicit NPF/STO history driver retaining intermediate heads on the device.

Geometry, K, storage fields, cell types and prescribed heads are fixed within
one call. The caller may update physical K between calls. Source multipliers,
areal recharge rates and timestep lengths can vary per step. A failed physical
acceptance check stops the history.
"""
from dataclasses import replace
from math import fsum
from time import perf_counter

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.physics.inputs_3d import floating_field
from DARCY_WARP_PACKAGE.physics.recharge_3d import surface_recharge_cells
from .npf662_3d import case_from_model
from .npf662_newton import NPF662NewtonControls, solve_newton_device
from .npf662_kcycle import NPF662KCycleControls
from . import npf662_kernels as kernels
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference, storage_previous_heads


def _history_inputs(*, dt, source_factors, recharge_rates, wel_rates, surface_shape, model):
    """Validate and align the timestep, multiplier and source schedule inputs.

    :param dt: Positive scalar duration or one duration per step.
    :param source_factors: Optional vector of volumetric source multipliers.
    :param recharge_rates: Optional scalar, horizontal field, period vector or field stack.
    :param wel_rates: Optional per-step sparse well entries replacing the built wells.
    :param surface_shape: Horizontal (ny, nx) grid dimensions.
    :param model: Built 3D model owning the active-cell grid.
    :return: Timestep vector, multiplier vector, areal rate array and owned well entries.
    """
    times = floating_field(value=dt, name="dt")
    if times.ndim > 1 or times.size == 0 or np.any(times <= 0):
        raise ValueError("dt must contain positive finite timestep lengths")
    steps = times.size
    rates = None
    if recharge_rates is not None:
        rates = floating_field(value=recharge_rates, name="recharge_rates")
        if rates.ndim > 3 or rates.size == 0:
            raise ValueError("recharge_rates must be a scalar, period vector, horizontal grid or stack")
        if rates.ndim >= 2 and rates.shape[-2:] != surface_shape:
            raise ValueError(f"recharge_rates horizontal dimensions must be {surface_shape}")
        if rates.ndim in (1, 3):
            steps = rates.shape[0]

    if source_factors is None:
        factors = np.ones(steps, dtype=np.float64)
    else:
        factors = floating_field(value=source_factors, name="source_factors")
        if factors.ndim != 1 or not factors.size:
            raise ValueError("Source factors must be a nonempty finite vector")
        steps = factors.size

    wells = None
    if wel_rates is not None:
        from DARCY_WARP_PACKAGE.physics.wel import validate_well_entries
        if isinstance(wel_rates, (str, bytes)) or not isinstance(wel_rates, (list, tuple)):
            raise ValueError("wel_rates must be None or a per-step sequence of entry values")
        wells = [validate_well_entries(entries=entries, shape=model.shape, active=model._active)
                 for entries in wel_rates]
        steps = len(wells)

    if rates is not None and rates.ndim in (1, 3) and rates.shape[0] != steps:
        raise ValueError("recharge_rates and source_factors must cover the same number of steps")
    if factors.size not in (1, steps):
        raise ValueError("source_factors must be a scalar or match the number of history steps")
    if times.size not in (1, steps):
        raise ValueError("dt must be a scalar or match the number of history steps")
    return np.broadcast_to(times, (steps,)), np.broadcast_to(factors, (steps,)), rates, wells


class HistorySources:
    """Assemble the same cell rates for the resident solver and its host reference.

    ``source_factors`` scale the built source minus every explicitly replaced
    component: a recharge schedule excludes the built recharge and a well
    schedule excludes the built entries. Replaced components enter per step.
    """

    def __init__(self, *, model, original_source, rates, factors, wells=None):
        """Prepare the static cell source and eligible recharge receivers.

        :param model: Built 3D model providing geometry and source fields.
        :param original_source: Snapshot of the complete source before this history.
        :param rates: Validated optional recharge rates in model length per time.
        :param factors: One volumetric source multiplier per step.
        :param wells: Validated optional per-step well entries replacing the built wells.
        """
        self.rates = rates
        self.factors = factors
        self.wells = wells
        self.area = model.dx * model.dy
        self.column_count = model.nx * model.ny
        self.receivers = None
        self.base = original_source
        if rates is not None:
            self.base = model._source_with_recharge(0.0)
            self.receivers = surface_recharge_cells(model._active, model._bc_mask)
        if wells is not None:
            if rates is not None:
                # Both components are replaced: the scaling base keeps the
                # raw volumetric rates alone.
                self.base = model._assemble_source(rhs=model._volumetric_source, R_field=None,
                    entries=None, active=model._active, fixed=model._bc_mask,
                    dtype=self.base.dtype)
            else:
                self.base = model._source_without_wells(dtype=self.base.dtype)

    def rate_at(self, index):
        """Return the scalar or horizontal field for one zero-based step.

        :param index: Requested step.
        """
        if self.rates.ndim in (1, 3):
            return self.rates[index]
        return self.rates

    def wells_at(self, index):
        """Return the sparse well entries replacing the built wells for one step.

        :param index: Requested step.
        """
        if self.wells is None:
            return None
        return self.wells[index]

    def host_source(self, index):
        """Form a volumetric source array for an independent solve or audit.

        :param index: Requested step.
        """
        from DARCY_WARP_PACKAGE.physics.wel import well_cell_volume_rates
        source = self.base * self.factors[index]
        if self.rates is not None:
            rate = self.rate_at(index)
            if rate.ndim == 2:
                rate = rate.ravel()[self.receivers % self.column_count]
            source.flat[self.receivers] += rate * self.area
        scheduled = well_cell_volume_rates(entries=self.wells_at(index), shape=source.shape)
        if scheduled is not None:
            source += scheduled
        return source

    def upload(self, device):
        """Upload reusable source buffers for the resident history.

        :param device: Warp device holding the Newton workspace.
        """
        self.device_base = wp.array(self.base.ravel(), dtype=wp.float64, device=device)
        self.has_scheduled_wells = self.wells is not None
        if self.has_scheduled_wells:
            self.device_wells = wp.empty(self.base.size, dtype=wp.float64, device=device)
            self.host_wells = [well_cell_volume_rates(entries=step, shape=self.base.shape)
                               for step in self.wells]
        if self.rates is None or not self.receivers.size:
            return
        self.device_receivers = wp.array(self.receivers, dtype=wp.int32, device=device)
        if self.rates.ndim < 2:
            # Uniform rates travel as kernel arguments, without period uploads.
            self.device_rates = wp.empty(1, dtype=wp.float64, device=device)
        else:
            self.device_rates = wp.array(self.rate_at(0).ravel(), dtype=wp.float64, device=device)

    def apply(self, workspace, index):
        """Prepare the complete source on device; copy only varying surface fields.

        :param workspace: Resident Newton workspace whose source field is updated.
        :param index: Requested step.
        """
        wp.launch(kernel=kernels.scale_period_source, dim=workspace.n,
            inputs=[self.device_base, workspace.source, self.factors[index]], device=workspace.device)
        if getattr(self, 'has_scheduled_wells', False):
            scheduled = self.host_wells[index]
            if scheduled is None:
                self.device_wells.zero_()
            else:
                self.device_wells.assign(scheduled.ravel())
            wp.launch(kernel=kernels.add_cell_rates, dim=workspace.n,
                inputs=[self.device_wells, workspace.source], device=workspace.device)
        if self.rates is None or not self.receivers.size:
            return
        scalar_rate = 0.0
        uniform = self.rates.ndim < 2
        if uniform:
            scalar_rate = float(self.rate_at(index))
        elif self.rates.ndim == 3:
            self.device_rates.assign(self.rate_at(index).ravel())
        wp.launch(kernel=kernels.add_surface_recharge, dim=self.receivers.size,
            inputs=[self.device_receivers, self.device_rates, self.column_count, self.area,
                    int(uniform), scalar_rate, workspace.source], device=workspace.device)


class DeviceHistoryReference:
    """Construct the independent oracle only if device grounding is insufficient."""
    def __init__(self, *, case, workspace, ss, sy, sources):
        self.case = case
        self.workspace = workspace
        self.ss = ss
        self.sy = sy
        self.dt = None
        self.sources = sources
        self.index = 0

    def evaluate(self, *, head, **tolerances):
        previous = self.workspace.storage_previous.numpy().reshape(self.case.active.shape).copy()
        case = replace(self.case, source=self.sources.host_source(self.index))
        return NPF662TransientReference(case=case, previous=previous, dt=self.dt,
            ss=self.ss, sy=self.sy).evaluate(head=head, **tolerances)


def prepare_next_step(*, workspace, sources, index, dt, area, activation_head=None):
    """Freeze the accepted previous head and recompute storage/scaling in place."""
    wp.copy(dest=workspace.storage_previous, src=workspace.head)
    sources.apply(workspace, index)
    workspace.storage_factor = area/dt
    wp.launch(kernel=kernels.storage_terms, dim=workspace.n,
        inputs=[workspace.storage_previous, workspace.storage_previous, workspace.bottom, workspace.thickness,
            workspace.types, workspace.storage_ss, workspace.storage_sy, workspace.free,
            workspace.storage_rate, workspace.previous_storage_tangent, workspace.storage_factor], device=workspace.device)
    workspace.refresh_previous_operator(activation_head=activation_head)
    workspace.maximum.zero_()
    wp.launch(kernel=kernels.previous_storage_scale, dim=workspace.n,
        inputs=[workspace.principal, workspace.previous_operator_tangent, workspace.scale, workspace.maximum],
        device=workspace.device)
    if workspace.maximum.numpy()[0] != 0.:
        raise ValueError('Invalid device previous-state restraint diagonal')
    workspace.faces_changed = False
    workspace.jv_counter.zero_()
    workspace.preconditioner_counter.zero_()
    workspace.residual_evaluations = workspace.candidate_evaluations = 0


def attach_accepted_exchange_volumes(*, period):
    """Add backward-Euler gross exchange volumes to an accepted period.

    :param period: Period diagnostics containing ``dt``, convergence and rate budgets.
    :return: The same period mapping with DRN/RIV volumes added when accepted.
    :raises ValueError: If an accepted package budget is nonfinite or malformed.
    """
    if not period.get('converged', False):
        return period
    duration = float(period['dt'])
    for package in ('drn', 'riv'):
        budget = period.get(f'{package}_budget')
        if budget is None:
            continue
        rates = tuple(float(budget[key]) for key in ('inflow', 'outflow', 'net'))
        if not np.isfinite(duration) or duration <= 0. or not np.all(np.isfinite(rates)):
            raise ValueError(f'Invalid accepted {package.upper()} rate or timestep')
        period[f'{package}_volume_budget'] = dict(
            inflow=rates[0] * duration,
            outflow=rates[1] * duration,
            net=rates[2] * duration,
        )
    return period


def total_accepted_exchange_volumes(*, periods):
    """Sum accepted gross package volumes without cancelling opposing rates.

    :param periods: Ordered per-period diagnostics.
    :return: Separate DRN and RIV gross volume totals, or None when disabled.
    """
    result = {}
    for package in ('drn', 'riv'):
        budgets = [row[f'{package}_volume_budget'] for row in periods
                   if row.get('converged', False) and f'{package}_volume_budget' in row]
        if budgets:
            result[package] = dict(
                inflow=fsum(row['inflow'] for row in budgets),
                outflow=fsum(row['outflow'] for row in budgets),
                net=fsum(row['net'] for row in budgets),
            )
    return result or None


def solve_transient_history(*, model, dt, head_prev, ss, sy,
                            source_factors=None, recharge_rates=None, wel_rates=None,
                            **solver_options):
    """Solve a series of implicit NPF/STO steps using the supplied forcing.

    :param model: Built 3D NPF model.
    :param dt: Scalar or per-step durations in model time units.
    :param head_prev: Initial head field at the start of the history.
    :param ss: Specific storage in inverse model length.
    :param sy: Dimensionless specific yield.
    :param source_factors: Optional per-step multipliers for the model's cell
        source. Explicit recharge_rates or wel_rates components are excluded
        from the scaling and replaced independently per step.
    :param recharge_rates: Optional replacement for the built areal R_field [L/T].
        Accepts a scalar, (ny, nx) grid, (steps,) rates, or (steps, ny, nx) grids.
    :param wel_rates: Optional replacement for the built sparse wells. Supply
        None or one entry value per step in ``build_from_K_fields`` entry form;
        a step value of None leaves that step without wells.
    :param solver_options: Options forwarded to the NPF solve.
    :return: Head history and per-step diagnostics.
    """
    started = perf_counter()
    if model.formulation != 'mf6_6_6_2_npf' or model._session is None:
        raise ValueError('A built mf6_6_6_2_npf model is required')
    times, factors, rates, wells = _history_inputs(
        dt=dt, source_factors=source_factors, recharge_rates=recharge_rates,
        wel_rates=wel_rates, surface_shape=(model.ny, model.nx), model=model,
    )
    options = dict(solver_options)
    for name in ('transient', 'npf_device_native', 'return_info'):
        if not isinstance(options.get(name, True), (bool, np.bool_)) or not options.pop(name, True):
            raise ValueError(f'The NPF history driver requires {name}=True')
    if options.setdefault('npf_preconditioner', 'kcycle') != 'kcycle':
        raise ValueError('The NPF history driver currently requires the kcycle preconditioner')
    source_host = model._session.rhs.copy()
    sources = HistorySources(model=model, original_source=source_host, rates=rates,
        factors=factors, wells=wells)
    old_initial = None if model._session.initial_head is None else model._session.initial_head.copy()
    types = options.get('npf_cell_type', int(options.get('unconfined', False)))
    case = case_from_model(model=model, top=options.get('top_elevation', 0.), cell_type=types)
    initial = storage_previous_heads(case=case, previous=head_prev, ss=ss, sy=sy)
    base = replace(case, source=source_host, initial=initial)
    periods = []
    workspace = None
    head_readbacks = 0
    try:
        model._session.update_rhs(sources.host_source(0))
        model.update_initial_head(initial_head=initial)
        first_start = perf_counter()
        first_head, first_info = model.solve(transient=True, npf_device_native=True, return_info=True,
            dt=float(times[0]), head_prev=initial, ss=ss, sy=sy, **options)
        wp.synchronize_device(device=model.device)
        periods.append(dict(first_info, index=0, dt=float(times[0]), source_factor=float(factors[0]),
            public_step_seconds=perf_counter()-first_start))
        attach_accepted_exchange_volumes(period=periods[-1])
        head_readbacks += 1
        workspace = model._session._npf662_workspace_holder['workspace']
        capacity = len(factors) if first_info["converged"] else 1
        if capacity > 1:
            sources.upload(model.device)
        history = wp.empty(shape=(capacity, workspace.n), dtype=wp.float64, device=model.device)
        wp.copy(dest=history.flatten(), src=workspace.head, count=workspace.n)
        nonlinear = NPF662NewtonControls(**first_info['controls'])
        multigrid = NPF662KCycleControls(**first_info['multigrid']['controls'])
        state = model._session._npf662_workspace_holder['native_state']
        reference = DeviceHistoryReference(case=base, workspace=workspace, ss=state.ss, sy=state.sy, sources=sources)
        thresholds = first_info['final_acceptance_thresholds']
        activation_watch = state.activation_watch_needed()
        for index in range(1, len(factors)):
            if not periods[-1]['converged']:
                break
            step_start = perf_counter()
            reference.dt = float(times[index])
            reference.index = index
            step_seeds = []
            activation_head = None
            if activation_watch:
                previous_host = workspace.head.numpy().reshape(base.active.shape)
                seeded_head, step_seeds = state.seed_unanchored_boundary_components(head=previous_host)
                if step_seeds:
                    workspace.head.assign(seeded_head.ravel())
                    activation_head = workspace.head
            prepare_next_step(workspace=workspace, sources=sources, index=index,
                dt=reference.dt, area=base.dx*base.dy, activation_head=activation_head)
            head, info = solve_newton_device(model=model, case=base, reference=reference,
                max_outer=first_info['max_outer'], krylov_budget=first_info['max_krylov_per_outer'],
                restart=first_info['krylov_restart'], controls=nonlinear,
                physical_tolerance=thresholds['physical_rms_m3_day'],
                relative_tolerance=thresholds['relative_cell_imbalance'],
                roundoff_acceptance=thresholds.get('roundoff_acceptance',False),
                npf_preconditioner='kcycle', npf_kcycle_controls=multigrid,
                npf_krylov_method=first_info['krylov_method'].lower(), device_native=True,
                prepared_workspace=workspace, device_sequence=True)
            head_readbacks += int(head is not None)
            wp.copy(dest=history.flatten(), src=workspace.head, dest_offset=index*workspace.n, count=workspace.n)
            periods.append(dict(info, index=index, dt=reference.dt, source_factor=float(factors[index]),
                active_set_seeds=step_seeds,
                public_step_seconds=perf_counter()-step_start))
            attach_accepted_exchange_volumes(period=periods[-1])
        count = len(periods)
        # One transfer for the output history. Internal acceptance has already
        # checked every step; this download is inside the public driver timer.
        # NumPy retains the owning host Warp array through its base reference.
        # This per-call buffer is never reused, so another 200 MB host copy is
        # unnecessary even after the model/history workspace is closed.
        if count < capacity:
            # A contiguous device view copies just the populated prefix to host.
            # On CPU the view still owns the entire history, so detach its prefix.
            prefix = history[:count].numpy()
            if history.device.is_cpu:
                prefix = prefix.copy()
        else:
            prefix = history.numpy()
        heads = prefix.reshape((count, *base.active.shape))
        last_previous = initial if count == 1 else heads[-2]
        state = model._session._npf662_workspace_holder['native_state']
        state.refresh(model=model, case=replace(base, source=sources.host_source(count-1), initial=last_previous),
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
        model._session.update_rhs(source_host)
        model.update_initial_head(initial_head=old_initial)
    wp.synchronize_device(device=model.device)
    return heads, dict(converged=len(periods) == len(factors) and all(row['converged'] for row in periods),
        execution='device_transient_history', periods=periods, requested_steps=len(factors),
        returned_steps=len(periods), accepted_steps=sum(row['converged'] for row in periods), seconds=perf_counter()-started,
        formulation='mf6_6_6_2_npf', device=model.device, precision='float64',
        numerical_seconds=sum(row['seconds'] for row in periods),
        solver_head_readbacks=head_readbacks, output_history_readbacks=1,
        accepted_exchange_volume_totals=total_accepted_exchange_volumes(periods=periods),
        fixed_inputs_within_history=['geometry', 'conductivity', 'cell_type', 'Ss', 'Sy', 'prescribed_heads', 'GHB conductance/head'],
        final_acceptance_thresholds=thresholds,
        timing='Entire public history call, including setup, all source/timestep updates, acceptance and output download')
