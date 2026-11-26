"""Explicit production 3D package qualification with matched source controls.

The archived original-forcing experiment remains separate. Regional workloads
scale abstraction to a declared fraction of actual recharge and retain gaining
and losing rivers plus independently switching multiple entries per cell.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
from time import perf_counter

import numpy as np
import warp as wp


def regional_fields(*, shape, abstraction_fraction, packages, seed, regional_controls):
    """Prepare reproducible heterogeneous fields with physically separate sources.

    :param shape: Groundwater grid shape.
    :param abstraction_fraction: Net abstraction divided by actual recharge.
    :param packages: Enable sparse DRN/RIV/GHB.
    :param seed: Existing field generator seed.
    :param regional_controls: Explicit physical stages, conductances and datum.
    :return: Builder fields, retaining physical geometry.
    """
    from regional_package_fields import heterogeneous_fields
    from DARCY_WARP_PACKAGE.physics.recharge_3d import surface_recharge_cells
    fields = heterogeneous_fields(shape=shape, seed=seed)
    receivers = surface_recharge_cells(active=fields['active'], bc_mask=fields['bc_mask'])
    recharge = np.asarray(fields['R_field']).ravel()[receivers % (shape[1] * shape[2])]
    available = float(np.sum(recharge)) * fields['dx'] * fields['dy']
    multiplier = abstraction_fraction * available / -sum(rate for cell, rate in fields['wel_entries'])
    fields['wel_entries'] = [(cell, rate * multiplier) for cell, rate in fields['wel_entries']]
    if packages:
        drains = []
        for cell, conductance, elevation in fields['drn_entries']:
            drains.extend([(cell, conductance * regional_controls['drain_first_fraction'], elevation), (cell, conductance * (1. - regional_controls['drain_first_fraction']), elevation + regional_controls['drain_second_offset'])])
        rivers = []
        for index, (cell, conductance, stage, bottom) in enumerate(fields['riv_entries']):
            bottom = regional_controls.get('riverbed_bottom_override', bottom)
            stage = regional_controls['river_middle_stage'] + regional_controls['river_stage_amplitude'] * (1. if index % 2 else -1.)
            rivers.extend([(cell, conductance * regional_controls['river_first_fraction'], stage, bottom),
                           (cell, conductance * (1. - regional_controls['river_first_fraction']), stage + regional_controls['river_second_stage_offset'], bottom + regional_controls['river_second_bottom_offset'])])
        fields['drn_entries'], fields['riv_entries'] = drains, rivers
        conductance = np.zeros(shape)
        conductance[0, 0, 1:-1:regional_controls['ghb_column_stride']] = regional_controls['ghb_conductance']
        fields.update(ghb_conductance=conductance, ghb_head=np.full(shape, regional_controls['initial_head']))
    else:
        fields['drn_entries'] = fields['riv_entries'] = None
    fields['initial_head'] = np.full(shape, regional_controls['initial_head'])
    fields['top_elevation'] = regional_controls['top_elevation']
    return fields


def boundary_controls(*, controls, variant):
    """Own explicit stages for the separate gaining/losing river comparison.

    :param controls: Baseline physical controls.
    :param variant: Baseline or gaining_losing boundary workload.
    :return: Physical control mapping for this exact numerical/native twin.
    """
    selected = dict(controls)
    if variant == 'gaining_losing':
        selected.update(river_middle_stage=controls['contrasting_river_middle_stage'],
            river_second_stage_offset=controls['contrasting_river_second_stage_offset'],
            riverbed_bottom_override=controls['contrasting_river_bottom'])
    return selected


def build_model(*, fields, device, formulation, raw_control):
    """Construct the requested actual backend with equivalent optional raw forcing.

    :param fields: Reproducible physical builder fields.
    :param device: CUDA device.
    :param formulation: Legacy direct Cartesian or matched NPF.
    :param raw_control: Replace WEL with identical volumetric RHS.
    :return: Built solver.
    """
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    from DARCY_WARP_PACKAGE.physics.wel import validate_well_entries, well_cell_volume_rates
    shape = fields['active'].shape
    options = dict(fields)
    dx, dy, dz = (options.pop(name) for name in ('dx', 'dy', 'dz'))
    initial = options.pop('initial_head')
    options.pop('top_elevation')
    if raw_control:
        entries = validate_well_entries(entries=options.pop('wel_entries'), shape=shape, active=options['active'])
        options['rhs'] = well_cell_volume_rates(entries=entries, shape=shape)
    model = WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=dx, dy=dy, dz=dz,
        solver='kcycle', implementation='fast', formulation=formulation, device=device)
    model.build_from_K_fields(**options, initial_head=initial)
    return model


def options_for(*, saved, state, precision, preconditioner, graphs, line_smoother, levels, restart=None, optimized=None, cached_lines=None, inner_forcing=None):
    """Use saved qualified controls with explicit diagnostic choices.

    :param saved: Existing qualification JSON.
    :param state: Saturated or current coefficient state.
    :param precision: Preconditioner precision.
    :param preconditioner: K-cycle or explicit diagonal.
    :param graphs: Enable qualified graph/tiled execution.
    :param line_smoother: Existing line smoother.
    :param levels: Existing coarsening depth.
    :param restart: Optional explicitly measured Krylov restart override.
    :param optimized: Enable tiled reductions and trial-state reuse independently.
    :param cached_lines: Override cached vertical line factors when comparing FP64.
    :param inner_forcing: Optional existing qualified forcing control.
    :return: Explicit public NPF solve options.
    """
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls
    options = dict(saved['solve'])
    options.pop('npf_device_native', None)
    options.update(npf_preconditioner=preconditioner, npf_krylov_method='fgmres')
    if restart is not None:
        options['krylov_restart'] = restart
    native_optimized = graphs if optimized is None else optimized
    nonlinear = dict(saved['newton'], tiled_reductions=native_optimized, reuse_trial_state=native_optimized)
    if inner_forcing is not None:
        nonlinear['inner_forcing'] = inner_forcing
    options['npf_newton_controls'] = NPF662NewtonControls(**nonlinear)
    if preconditioner == 'kcycle':
        controls = dict(saved['multigrid'], coefficient_state=state, preconditioner_precision=precision,
            use_cuda_graph=graphs, line_smoother=line_smoother, levels=levels)
        if cached_lines is not None:
            controls['cached_line_factors'] = cached_lines
        options['npf_kcycle_controls'] = NPF662KCycleControls(**controls)
    return options


def compact_info(*, info, elapsed):
    """Keep actual execution and acceptance diagnostics without grid arrays.

    :param info: Public solver metadata.
    :param elapsed: Synchronized public wall time.
    :return: JSON-compatible evidence.
    """
    names = ('converged', 'stop_reason', 'implementation', 'session_backend', 'krylov_method',
        'npf_preconditioner', 'outer_iterations', 'krylov_iterations', 'preconditioner_cycles',
        'cuda_graph_captures', 'cuda_graph_replays', 'seconds', 'public_total_seconds',
        'device_final_metrics', 'physical_acceptance', 'ghb_budget', 'drn_budget', 'riv_budget',
        'source_budgets', 'multigrid', 'n_cycles_used', 'graph_count', 'graph_replays',
        'linear_acceleration', 'preconditioner_applications', 'cuda_graph_replayed', 'r_rms_end', 'h_rms_end')
    result = {name: info.get(name) for name in names}
    result.update(public_seconds=elapsed, gpu_used_bytes=int(wp.get_mempool_used_mem_current()),
        process_gpu_high_water_bytes=int(wp.get_mempool_used_mem_high()))
    return result


def timed_solve(*, model, options, preconditioner, direct):
    """Time the actual path and fail on wrong dispatch or numerical rejection.

    :param model: Built solver.
    :param options: Public numerical options.
    :param preconditioner: Requested NPF preconditioner.
    :param direct: Expect the separate direct Cartesian path.
    :return: Heads, original metadata, compact evidence.
    """
    wp.synchronize_device(device=model.device)
    started = perf_counter()
    head, info = model.solve(**options)
    wp.synchronize_device(device=model.device)
    row = compact_info(info=info, elapsed=perf_counter() - started)
    assert info['converged'], row
    if direct:
        assert info['session_backend'] == 'fast_3d' and 'npf_preconditioner' not in info, row
        if options.get('linear_acceleration') == 'fgmres':
            assert info['linear_acceleration'] == 'fgmres' and info['preconditioner_applications'] > 0, row
            assert info['cuda_graph_replayed'], row
    else:
        assert info['npf_preconditioner'] == preconditioner and info['krylov_method'] == 'FGMRES', row
        if preconditioner == 'kcycle' and info['krylov_iterations']:
            assert info['preconditioner_cycles'] > 0, row
            if options['npf_kcycle_controls'].use_cuda_graph:
                assert info['multigrid']['graph_replays'] > 0, row
    return head, info, row


def summary(*, rows):
    """Summarize repeated public times without selecting only the fastest run."""
    values = [row['public_seconds'] for row in rows]
    return dict(median=float(np.median(values)), minimum=min(values), maximum=max(values))


def qualify_case(*, fields, configuration, repeats, saved, direct_controls, output, device, raw_control, regional_controls):
    """Separate construction, cold, identical-guess, calibration and source update costs.

    :param fields: Physical fields.
    :param configuration: Explicit numerical configuration mapping.
    :param repeats: Repeated warm measurements.
    :param saved: Existing NPF qualification controls.
    :param direct_controls: Existing direct confined controls.
    :param output: Per-case evidence directory.
    :param device: CUDA device.
    :param raw_control: Use equivalent raw WEL rates.
    :param regional_controls: Independent source update schedules and head gate.
    :return: Qualification evidence.
    """
    output.mkdir(parents=True, exist_ok=True)
    direct = configuration['name'].startswith('cartesian_')
    started = perf_counter()
    with build_model(fields=fields, device=device, formulation='legacy' if direct else 'mf6_6_6_2_npf', raw_control=raw_control) as model:
        construction = perf_counter() - started
        options = dict(direct_controls) if direct else options_for(saved=saved, **{key: value for key, value in configuration.items() if key != 'name'})
        if configuration['name'] == 'cartesian_fgmres':
            from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
            options.update(linear_acceleration='fgmres', linear_kcycle_controls=NPF662KCycleControls(**saved['multigrid']))
        if not direct:
            options['top_elevation'] = fields['top_elevation']
        preconditioner = configuration.get('preconditioner')
        head, info, cold = timed_solve(model=model, options=options, preconditioner=preconditioner, direct=direct)
        reference = head.copy()
        snapshots = [accepted_snapshot(model=model, head=head, info=info, fields=fields)]
        identical, calibration, update_costs = [], [], []
        for repeat in range(repeats):
            model.update_initial_head(initial_head=fields['initial_head'])
            head, info, row = timed_solve(model=model, options=options, preconditioner=preconditioner, direct=direct)
            row['head_difference_from_cold_m'] = float(np.max(np.abs(head - reference)))
            assert row['head_difference_from_cold_m'] <= regional_controls['head_gate_m'], row
            identical.append(row)
        base_wells = fields['wel_entries']
        for repeat in range(repeats):
            model.update_initial_head(initial_head=head)
            factor = regional_controls['calibration_wel_factors'][repeat]
            started = perf_counter()
            wells = [(cell, rate * factor) for cell, rate in base_wells]
            if raw_control:
                from DARCY_WARP_PACKAGE.physics.wel import validate_well_entries, well_cell_volume_rates
                entries = validate_well_entries(entries=wells, shape=model.shape, active=model._active)
                model.update_rhs(rhs=well_cell_volume_rates(entries=entries, shape=model.shape))
            else:
                model.update_wel(entries=wells)
            wel_seconds = perf_counter() - started
            started = perf_counter()
            model.update_R_in_place(R_field=np.asarray(fields['R_field']) * regional_controls['calibration_recharge_factors'][repeat])
            recharge_seconds = perf_counter() - started
            stage_seconds = None
            if fields['riv_entries'] is not None:
                started = perf_counter()
                model.update_riv_entries(entries=[(cell, c, stage + regional_controls['calibration_stage_offsets'][repeat], bottom)
                    for cell, c, stage, bottom in fields['riv_entries']])
                stage_seconds = perf_counter() - started
            head, info, row = timed_solve(model=model, options=options, preconditioner=preconditioner, direct=direct)
            snapshots.append(accepted_snapshot(model=model, head=head, info=info, fields=fields))
            calibration.append(row)
            update_costs.append(dict(wel_or_raw_seconds=wel_seconds, recharge_seconds=recharge_seconds, river_stage_seconds=stage_seconds))
        started = perf_counter()
        if fields['drn_entries'] is not None:
            for package in ('drn', 'riv'):
                model.accepted_boundary_entry_rates(package=package, head=head, info=info)
                getattr(model, f'accepted_{package}_discharge')(head=head, info=info)
        if not raw_control:
            model.accepted_wel_rates(head=head, info=info)
        extraction = perf_counter() - started
        arrays = {f'{name}_{index}': value for index, snapshot in enumerate(snapshots) for name, value in snapshot.items()}
        np.savez_compressed(output / 'accepted_heads.npz', **arrays)
        if fields['riv_entries'] is not None and fields['riv_entries'][0][2] < regional_controls['contrasting_river_middle_stage'] + regional_controls['river_stage_amplitude']:
            assert cold['riv_budget']['inflow'] > 0. and cold['riv_budget']['outflow'] > 0., cold
        result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), configuration=configuration, shape=list(model.shape), construction_seconds=construction,
            cold=cold, identical_initial=identical, identical_summary=summary(rows=identical),
            warm_start_calibration=calibration, calibration_summary=summary(rows=calibration),
            source_updates=update_costs, accepted_output_extraction_seconds=extraction,
            forcing=dict(abstraction=model.wel_budget() if not raw_control else None), raw_control=raw_control)
        (output / 'result.json').write_text(json.dumps(result, indent=2))
        print(model.shape, configuration['name'], 'raw' if raw_control else 'WEL', result['identical_summary'], flush=True)
        return result


def run_matrix(*, grids, configurations, variants, repeats, saved, direct_controls, output, device, abstraction_fraction, seed, regional_controls):
    """Run the requested explicit matrix, retaining every failure as failed evidence."""
    output.mkdir(parents=True, exist_ok=True)
    report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        device=device, repeats=repeats, abstraction_fraction=abstraction_fraction, rows=[], failures=[])
    previous = output / 'qualification.json'
    if previous.exists():
        stored = json.loads(previous.read_text())
        if stored['commit'] == report['commit']:
            report = stored
        else:
            previous.rename(output / f"qualification_{stored['commit'][:8]}.json")
    for label, shape in grids.items():
        for variant, packages, raw in variants:
            physical_controls = boundary_controls(controls=regional_controls, variant=variant)
            fields = regional_fields(shape=shape, abstraction_fraction=abstraction_fraction, packages=packages, seed=seed, regional_controls=physical_controls)
            for configuration in configurations:
                if configuration['name'].startswith('cartesian_') and packages:
                    continue
                directory = output / label / variant / configuration['name']
                try:
                    row = qualify_case(fields=fields, configuration=configuration, repeats=repeats, saved=saved,
                        direct_controls=direct_controls, output=directory, device=device, raw_control=raw, regional_controls=regional_controls)
                    report['rows'].append(dict(grid=label, variant=variant, result=row))
                except Exception as error:
                    failure = dict(grid=label, variant=variant, configuration=configuration, error=repr(error))
                    report['failures'].append(failure)
                    print('FAILED', failure, flush=True)
                (output / 'qualification.json').write_text(json.dumps(report, indent=2))
    if report['failures']:
        raise AssertionError('Production qualification contains failed solves; see qualification.json')
    return report



def accepted_snapshot(*, model, head, info, fields):
    """Extract owned accepted outputs outside the timed numerical solve.

    :param model: Model owning this accepted equation.
    :param head: Accepted groundwater heads.
    :param info: Associated accepted metadata.
    :param fields: Physical boundary definitions.
    :return: Host arrays for independent native comparison.
    """
    arrays = dict(head=np.array(head, copy=True))
    if fields['drn_entries'] is not None:
        for package in ('drn', 'riv'):
            arrays[package] = model.accepted_boundary_entry_rates(package=package, head=head, info=info)
    return arrays


def qualify_history(*, fields, saved, controls, repeats, output, device):
    """Measure actual resident NPF/STO histories from the same physical IC.

    :param fields: Fixed geometry and package definitions.
    :param saved: Existing qualified NPF controls.
    :param controls: Physical periods and independent package schedules.
    :param repeats: Warm repeated history measurements.
    :param output: Evidence directory.
    :param device: CUDA device.
    :return: Public and numerical history measurements with accepted budgets.
    """
    output.mkdir(parents=True, exist_ok=True)
    steps = controls['history_steps']
    well_factors = np.resize(controls['history_wel_pattern'], steps)
    recharge_factors = np.resize(controls['history_recharge_pattern'], steps)
    wells = [[(cell, rate * factor) for cell, rate in fields['wel_entries']] for factor in well_factors]
    recharge = np.asarray([fields['R_field'] * factor for factor in recharge_factors])
    options = options_for(saved=saved, state='current', precision='float32', preconditioner='kcycle',
        graphs=True, line_smoother='jacobi', levels=saved['multigrid']['levels'])
    options.update(top_elevation=fields['top_elevation'])
    rows = []
    with build_model(fields=fields, device=device, formulation='mf6_6_6_2_npf', raw_control=False) as model:
        for repeat in range(repeats + 1):
            wp.synchronize_device(device=device)
            started = perf_counter()
            heads, info = model.solve_transient_npf(dt=[controls['history_dt_days']] * steps,
                head_prev=fields['initial_head'], ss=controls['specific_storage_per_m'], sy=controls['specific_yield'],
                wel_rates=wells, recharge_rates=recharge, **options)
            wp.synchronize_device(device=device)
            elapsed = perf_counter() - started
            assert info['converged'] and info['accepted_steps'] == steps, info
            assert info['execution'] == 'device_transient_history', info
            assert info['output_history_readbacks'] == 1, info
            for period in info['periods']:
                assert period['converged'] and period['npf_preconditioner'] == 'kcycle'
                assert period['krylov_method'] == 'FGMRES'
                if period['krylov_iterations']:
                    assert period['preconditioner_cycles'] > 0 and period['multigrid']['graph_replays'] > 0
            rows.append(dict(public_seconds=elapsed, numerical_seconds=info['numerical_seconds'],
                solver_head_readbacks=info['solver_head_readbacks'], output_history_readbacks=info['output_history_readbacks'],
                gpu_used_bytes=int(wp.get_mempool_used_mem_current()),
                process_gpu_high_water_bytes=int(wp.get_mempool_used_mem_high()),
                periods=[compact_info(info=period, elapsed=period['public_step_seconds']) for period in info['periods']],
                accepted_exchange_volume_totals=info['accepted_exchange_volume_totals']))
        started = perf_counter()
        arrays = {f'{name}_{index}': value for index, period in enumerate(info['periods'])
            for name, value in accepted_snapshot(model=model, head=heads[index], info=period, fields=fields).items()}
        extraction = perf_counter() - started
        np.savez_compressed(output / 'accepted_heads.npz', **arrays)
    result = dict(cold=rows[0], warm=rows[1:], warm_summary=summary(rows=rows[1:]),
        extraction_seconds=extraction, controls=controls, configuration=dict(solve=saved))
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    print(fields['active'].shape, 'resident history', result['warm_summary'], flush=True)
    return result


if __name__ == "__main__":
    os.environ.setdefault('DARCY_FLOAT', 'float64')
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / 'working_tests/npf_ghb_five_level_controls.json').read_text())
    from sanity_3d_controls import read_profiles
    profile = read_profiles(root=root)['steady']['fast']
    direct_controls = {key: value for key, value in profile.items() if key in
        ('max_cycles', 'max_levels', 'nu_pre', 'nu_post', 'nu_coarse', 'omega', 'rel_tol', 'abs_tol_min', 'check_every_no', 'smoother', 'cheby_lambda_min', 'cheby_lambda_max', 'dh_rms_tol', 'dh_max_tol')}
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', choices=('100k', '1M', 'all'), default='all')
    parser.add_argument('--mode', choices=('steady', 'history', 'all'), default='all')
    parser.add_argument('--configuration', default='fast')
    parser.add_argument('--variant', choices=('all', 'raw_control', 'wel_only', 'packages', 'gaining_losing'), default='all')
    arguments = parser.parse_args()
    grids = {'100k': (10, 112, 90), '1M': (5, 450, 450)}
    grids = {name: shape for name, shape in grids.items() if arguments.grid in ('all', name)}
    configurations = [dict(name='cartesian_fast'), dict(name='cartesian_fgmres'),
        dict(name='saturated_fp64', state='saturated', precision='float64', preconditioner='kcycle', graphs=True, line_smoother='jacobi', levels=5),
        dict(name='current_fp64', state='current', precision='float64', preconditioner='kcycle', graphs=True, line_smoother='jacobi', levels=5),
        dict(name='qualified_current_fp32', state='current', precision='float32', preconditioner='kcycle', graphs=True, line_smoother='jacobi', levels=5),
        dict(name='diagonal_restart48', restart=48, state='current', precision='float64', preconditioner='diagonal', graphs=True, line_smoother='jacobi', levels=5),
        dict(name='diagonal_fgmres', state='current', precision='float64', preconditioner='diagonal', graphs=True, line_smoother='jacobi', levels=5)]
    alternatives = [
        dict(name='saturated_fp32', state='saturated', precision='float32', preconditioner='kcycle', graphs=True, line_smoother='jacobi', levels=5),
        dict(name='current_fp32_red_black', state='current', precision='float32', preconditioner='kcycle', graphs=True, line_smoother='red_black', levels=5),
        dict(name='current_fp32_three_levels', state='current', precision='float32', preconditioner='kcycle', graphs=True, line_smoother='jacobi', levels=3),
        dict(name='current_fp32_eager', state='current', precision='float32', preconditioner='kcycle', graphs=False, optimized=False, line_smoother='jacobi', levels=5),
        dict(name='current_fp32_mg_eager', state='current', precision='float32', preconditioner='kcycle', graphs=False, optimized=True, line_smoother='jacobi', levels=5),
        dict(name='current_fp64_uncached', state='current', precision='float64', preconditioner='kcycle', graphs=True, line_smoother='jacobi', levels=5, cached_lines=False)]
    fast_names = {item['name'] for item in configurations if not item['name'].startswith('diagonal') and item['name'] != 'cartesian_fast'}
    configurations += alternatives
    alternative_names = {item['name'] for item in alternatives}
    variants = [('raw_control', False, True), ('wel_only', False, False), ('packages', True, False), ('gaining_losing', True, False)]
    variants = [item for item in variants if arguments.variant in ('all', item[0])]
    repeats = 3
    output = root / 'benchmark_results/package_audit_repair/production_3d'
    device = 'cuda:0'
    regional_controls = json.loads((root / 'working_tests/wel_regional_qualification_controls.json').read_text())
    abstraction_fraction = regional_controls['abstraction_fraction']
    seed = regional_controls['seed']
    wp.init()
    configurations = [item for item in configurations if (arguments.configuration == 'all' and not item['name'].startswith('diagonal') and item['name'] != 'cartesian_fast') or arguments.configuration == item['name'] or (arguments.configuration == 'fast' and item['name'] in fast_names) or (arguments.configuration == 'alternatives' and item['name'] in alternative_names)]
    if not configurations:
        raise ValueError('Unknown or empty numerical configuration selection')
    if arguments.mode in ('steady', 'all'):
        run_matrix(grids=grids, configurations=configurations, variants=variants, repeats=repeats,
        saved=saved, direct_controls=direct_controls, output=output, device=device,
        abstraction_fraction=abstraction_fraction, seed=seed, regional_controls=regional_controls)

    if arguments.mode in ('history', 'all'):
        for label, shape in grids.items():
            fields = regional_fields(shape=shape, abstraction_fraction=abstraction_fraction, packages=True, seed=seed, regional_controls=regional_controls)
            qualify_history(fields=fields, saved=saved, controls=regional_controls, repeats=repeats, output=output / label / 'history', device=device)
