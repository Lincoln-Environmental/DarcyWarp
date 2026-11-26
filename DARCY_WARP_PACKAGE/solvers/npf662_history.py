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
from DARCY_WARP_PACKAGE.physics.wel import well_cell_volume_rates
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
        self.budget_receivers = surface_recharge_cells(active=model._active, bc_mask=model._bc_mask)
        self.base = original_source
        # Physical components kept separate for gross budget reporting; a
        # scheduled component drops out of the scaled base entirely.
        self.volumetric = (None if model._volumetric_source is None
                           else np.array(model._volumetric_source, copy=True))
        self.built_well_entries = model._wel_entries if wells is None else None
        self.built_wells = (None if model._wel_entries is None or wells is not None
                            else np.array(model._wel_entries.rates, copy=True))
        self.built_recharge = (None if model._recharge_rate is None or rates is not None
                               else np.array(model._recharge_rate, copy=True))
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

        from DARCY_WARP_PACKAGE.physics.wel import sparse_well_cell_rates
        self.sparse_wells = (None if wells is None else [
            sparse_well_cell_rates(entries=entries, shape=self.base.shape) for entries in wells])
        self.requires_gross = bool(wells is not None or self.built_wells is not None or
            ((rates is not None or self.built_recharge is not None) and self.volumetric is not None
             and np.any(self.volumetric != 0.)))
        self.gross_base = None
        if self.requires_gross:
            self.gross_base = (np.zeros_like(self.base) if self.volumetric is None
                               else np.abs(self.volumetric))
            if self.built_recharge is not None:
                selected = np.broadcast_to(np.abs(self.built_recharge), self.base.shape[1:]).ravel()
                self.gross_base.ravel()[self.budget_receivers] += selected[self.budget_receivers % self.column_count] * self.area
            cells, unused_signed, gross = sparse_well_cell_rates(entries=self.built_well_entries,
                shape=self.base.shape)
            self.gross_base.ravel()[cells] += gross

    def host_gross_source(self, *, index):
        """Own gross throughput of the exact period-specific physical sources.

        :param index: Requested period.
        :return: Gross source field, or None when signed magnitude is sufficient.
        """
        if not self.requires_gross:
            return None
        gross = self.gross_base * abs(self.factors[index])
        if self.rates is not None:
            rate = np.abs(self.rate_at(index))
            if rate.ndim == 2:
                rate = rate.ravel()[self.receivers % self.column_count]
            gross.ravel()[self.receivers] += rate * self.area
        if self.sparse_wells is not None:
            cells, unused_signed, rates = self.sparse_wells[index]
            gross.ravel()[cells] += rates
        return gross

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

    def component_budget(self, index):
        """Separate gross source budgets for one step's physical components.

        ``source_factors`` scale every built component that the step does not
        explicitly replace, so each gross budget carries that multiplier.

        :param index: Requested step.
        :return: Mapping with ``wel``, ``recharge`` and ``volumetric`` gross
            rate budgets in L³/T.
        """
        from DARCY_WARP_PACKAGE.physics.wel import signed_rate_budget
        factor = float(self.factors[index])
        scheduled = self.wells_at(index)
        wel = signed_rate_budget(rates=None if scheduled is None else scheduled.rates)
        if scheduled is None and self.built_wells is not None:
            wel = signed_rate_budget(rates=self.built_wells * factor)
        rate = self.rate_at(index) if self.rates is not None else self.built_recharge
        if rate is None:
            recharge = dict(inflow=0., outflow=0., net=0.)
        else:
            rates = np.broadcast_to(np.asarray(rate, dtype=np.float64), self.base.shape[1:]) * self.area
            actual_rates = rates.ravel()[self.budget_receivers % self.column_count]
            if self.rates is None:
                actual_rates = actual_rates * factor
            recharge = signed_rate_budget(rates=actual_rates)
        volumetric = signed_rate_budget(rates=None if self.volumetric is None
                                        else self.volumetric.ravel() * factor)
        return dict(wel=wel, recharge=recharge, volumetric=volumetric)

    def host_source(self, index):
        """Form a volumetric source array for an independent solve or audit.

        :param index: Requested step.
        """
        source = self.base * self.factors[index]
        if self.rates is not None:
            rate = self.rate_at(index)
            if rate.ndim == 2:
                rate = rate.ravel()[self.receivers % self.column_count]
            source.flat[self.receivers] += rate * self.area
        if self.sparse_wells is not None:
            cells, rates, unused_gross = self.sparse_wells[index]
            source.ravel()[cells] += rates
        return source

    def upload(self, device):
        """Upload persistent source fields and capacity-sized sparse WEL buffers.

        :param device: Owning Warp device.
        :return: None.
        """
        self.device_base = wp.array(self.base.ravel(), dtype=wp.float64, device=device)
        if self.requires_gross:
            self.device_gross_base = wp.array(self.gross_base.ravel(), dtype=wp.float64, device=device)
        if self.sparse_wells is not None:
            capacity = max(len(record[0]) for record in self.sparse_wells)
            self.device_well_cells = wp.empty(capacity, dtype=wp.int32, device=device)
            self.device_well_rates = wp.empty(capacity, dtype=wp.float64, device=device)
            self.device_well_gross = wp.empty(capacity, dtype=wp.float64, device=device)
        if self.rates is not None and self.receivers.size:
            self.device_receivers = wp.array(self.receivers, dtype=wp.int32, device=device)
            self.device_rates = (wp.empty(1, dtype=wp.float64, device=device) if self.rates.ndim < 2 else
                wp.array(self.rate_at(0).ravel(), dtype=wp.float64, device=device))

    def apply(self, workspace, index):
        """Update resident signed/gross forcing without dense period WEL uploads.

        :param workspace: Resident numerical workspace.
        :param index: Requested period.
        :return: None.
        """
        wp.launch(kernel=kernels.scale_period_source, dim=workspace.n,
            inputs=[self.device_base, workspace.source, self.factors[index]], device=workspace.device)
        if self.requires_gross:
            wp.launch(kernel=kernels.scale_period_source, dim=workspace.n,
                inputs=[self.device_gross_base, workspace.source_gross, abs(self.factors[index])], device=workspace.device)
        if self.rates is not None and self.receivers.size:
            uniform = self.rates.ndim < 2
            scalar_rate = float(self.rate_at(index)) if uniform else 0.
            if self.rates.ndim == 3:
                self.device_rates.assign(self.rate_at(index).ravel())
            inputs = [self.device_receivers, self.device_rates, self.column_count, self.area,
                      int(uniform), scalar_rate]
            wp.launch(kernel=kernels.add_surface_recharge, dim=self.receivers.size,
                inputs=[*inputs, workspace.source], device=workspace.device)
            if self.requires_gross:
                wp.launch(kernel=kernels.add_surface_recharge_gross, dim=self.receivers.size,
                    inputs=[*inputs, workspace.source_gross], device=workspace.device)
        if self.sparse_wells is not None:
            cells, rates, gross = self.sparse_wells[index]
            count = len(cells)
            if count:
                self.device_well_cells[:count].assign(cells.astype(np.int32))
                self.device_well_rates[:count].assign(rates)
                self.device_well_gross[:count].assign(gross)
                wp.launch(kernel=kernels.add_sparse_period_wells, dim=count,
                    inputs=[self.device_well_cells, self.device_well_rates, self.device_well_gross,
                        workspace.source, workspace.source if workspace.source_gross is None else workspace.source_gross,
                        int(workspace.source_gross is not None)], device=workspace.device)


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

    def seed_unanchored_boundary_components(self, *, head):
        """Recover an active set against the actual resident period equation.

        :param head: Numerical trial heads; physical previous heads stay frozen.
        :return: Seeded numerical heads and physical activation records.
        """
        previous = self.workspace.storage_previous.numpy().reshape(self.case.active.shape).copy()
        case = replace(self.case, source=self.sources.host_source(index=self.index), source_gross=self.sources.host_gross_source(index=self.index))
        reference = NPF662TransientReference(case=case, previous=previous, dt=self.dt,
            ss=self.ss, sy=self.sy)
        return reference.seed_unanchored_boundary_components(head=head)

    def evaluate(self, *, head, **tolerances):
        previous = self.workspace.storage_previous.numpy().reshape(self.case.active.shape).copy()
        case = replace(self.case, source=self.sources.host_source(self.index), source_gross=self.sources.host_gross_source(index=self.index))
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

    Physical source components (wells, recharge, raw volumetric rates) and the
    DRN/RIV packages each keep separate gross volumes so opposing flows never
    cancel in the reported totals.

    :param period: Period diagnostics containing ``dt``, convergence and rate budgets.
    :return: The same period mapping with accepted volumes added when accepted.
    :raises ValueError: If an accepted budget is nonfinite or malformed.
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
    budgets = period.get('source_budgets')
    if budgets is not None:
        volumes = {}
        for name, budget in budgets.items():
            rates = tuple(float(budget[key]) for key in ('inflow', 'outflow', 'net'))
            if not np.isfinite(duration) or duration <= 0. or not np.all(np.isfinite(rates)):
                raise ValueError(f'Invalid accepted {name} source rate or timestep')
            volumes[name] = dict(
                inflow=rates[0] * duration,
                outflow=rates[1] * duration,
                net=rates[2] * duration,
            )
        period['source_volume_budgets'] = volumes
    return period


def total_accepted_exchange_volumes(*, periods):
    """Sum accepted gross volumes without cancelling opposing rates.

    :param periods: Ordered per-period diagnostics.
    :return: Separate DRN, RIV and physical source gross volume totals, or
        None when no accepted volumes exist.
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
    source_names = ('wel', 'recharge', 'volumetric')
    source_budgets = [row['source_volume_budgets'] for row in periods
                      if row.get('converged', False) and 'source_volume_budgets' in row]
    if source_budgets:
        result['sources'] = {}
        for name in source_names:
            budgets = [row[name] for row in source_budgets if name in row]
            if budgets:
                result['sources'][name] = dict(
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
    if wells is not None:
        model._track_well_sources = True
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
    was_running = getattr(model, '_npf_history_running', False)
    model._npf_history_running = True
    try:
        model._npf_history_sources = (sources, 0)
        model._session.update_rhs(sources.host_source(0))
        model.update_initial_head(initial_head=initial)
        first_start = perf_counter()
        first_head, first_info = model.solve(transient=True, npf_device_native=True, return_info=True,
            dt=float(times[0]), head_prev=initial, ss=ss, sy=sy, **options)
        wp.synchronize_device(device=model.device)
        periods.append(dict(first_info, index=0, dt=float(times[0]), source_factor=float(factors[0]),
            source_budgets=sources.component_budget(0),
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
            # Freeze the accepted physical state and assemble the upcoming
            # forcing/storage before perturbing the numerical initial guess.
            prepare_next_step(workspace=workspace, sources=sources, index=index,
                dt=reference.dt, area=base.dx*base.dy)
            if activation_watch:
                previous_host = workspace.storage_previous.numpy().reshape(base.active.shape)
                step_case = replace(base, source=sources.host_source(index=index), source_gross=sources.host_gross_source(index=index))
                activation_reference = NPF662TransientReference(case=step_case,
                    previous=previous_host, dt=reference.dt, ss=state.ss, sy=state.sy)
                seeded_head, step_seeds = activation_reference.seed_unanchored_boundary_components(head=previous_host)
                if step_seeds:
                    workspace.head.assign(seeded_head.ravel())
                    workspace.refresh_previous_operator(activation_head=workspace.head)
                    workspace.maximum.zero_()
                    wp.launch(kernel=kernels.previous_storage_scale, dim=workspace.n,
                        inputs=[workspace.principal, workspace.previous_operator_tangent,
                            workspace.scale, workspace.maximum], device=workspace.device)
                    if workspace.maximum.numpy()[0] != 0.:
                        raise ValueError('Invalid seeded previous-state restraint diagonal')
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
                active_set_seeds=step_seeds, source_budgets=sources.component_budget(index),
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
        if getattr(model, '_track_well_sources', False):
            from DARCY_WARP_PACKAGE.physics.wel import WellEntries, attach_accepted_source_state, own_source_inputs
            for index, period in enumerate(periods):
                entries = sources.wells_at(index=index)
                if sources.wells is None and sources.built_well_entries is not None:
                    entries = WellEntries(cells=sources.built_well_entries.cells,
                        rates=sources.built_well_entries.rates * factors[index])
                attach_accepted_source_state(model=model, head=heads[index], info=period,
                    dt=float(times[index]), entries=entries, budgets=period['source_budgets'],
                    source_inputs=own_source_inputs(volumetric=sources.volumetric,
                        recharge=sources.rate_at(index) if sources.rates is not None else sources.built_recharge,
                        volumetric_factor=factors[index], recharge_factor=1. if sources.rates is not None else factors[index],
                        cell_area=sources.area, copy_arrays=False))
        last_previous = initial if count == 1 else heads[-2]
        state = model._session._npf662_workspace_holder['native_state']
        state.refresh(model=model, case=replace(base, source=sources.host_source(count-1), source_gross=sources.host_gross_source(index=count-1), initial=last_previous),
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
        model._npf_history_sources = None
        model._npf_history_running = was_running
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
