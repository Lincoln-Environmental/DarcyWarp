"""Measure maintained 2D WEL paths using the saved production solver controls."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp


def fields_for_grid(*, shape, controls):
    """Build identical physical sources for all maintained 2D paths.

    :param shape: Horizontal grid shape.
    :param controls: Shared physical qualification controls.
    :return: Model fields and explicit volumetric wells.
    """
    from qualify_package_kcycle import regional_fields
    fields = regional_fields(shape=(1, *shape), abstraction_fraction=controls['abstraction_fraction'],
        packages=False, seed=controls['seed'], regional_controls=controls)
    return dict(T_field=fields['kx_field'][0] * fields['dz'], R_field=fields['R_field'],
        active=fields['active'][0], bc_mask=fields['bc_mask'][0], bc_values=fields['bc_values'][0],
        wel_entries=[(cell[1:], rate) for cell, rate in fields['wel_entries']]), fields['dx']


def make_model(*, fields, dx, device, source_control):
    """Build WEL or its algebraically equivalent prescribed depth-source control.

    :param fields: Two-dimensional model fields.
    :param dx: Cell width.
    :param device: CUDA device.
    :param source_control: Replace WEL by identical combined depth forcing.
    :return: Built production model.
    """
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    from DARCY_WARP_PACKAGE.physics.wel import validate_well_entries, well_cell_volume_rates
    options = dict(fields)
    if source_control:
        entries = validate_well_entries(entries=options.pop('wel_entries'), shape=fields['active'].shape, active=fields['active'])
        options['R_field'] = options['R_field'] + well_cell_volume_rates(entries=entries, shape=fields['active'].shape) / (dx * dx)
    shape = fields['active'].shape
    model = WarpDarcySolver(nx=shape[1], ny=shape[0], dx=dx, solver_type='kcycle', device=device)
    model.build_from_fields(**options)
    model._qualification_recharge_f64 = np.array(options['R_field'], dtype=np.float64, copy=True)
    return model


def capture_info(*, info, elapsed):
    """Retain actual backend, graphs, numerical work and accepted physical budgets.

    :param info: Public solve metadata.
    :param elapsed: Synchronized public runtime.
    :return: Compact JSON-safe diagnostics.
    """
    keys = ('converged', 'solver_backend', 'implementation', 'production_precision_path',
        'n_cycles_used', 'outer_iterations', 'cuda_graph_built_this_call', 'cuda_graph_reused',
        'graph_count', 'graph_fallback_count', 'device_side_picard_fast_path_active',
        'transient_face_graphs', 'transient_face_kcycle_graph_count', 'transient_face_refresh_graph_count',
        'seconds', 'residual_norm', 'head_residual_rms', 'physical_period_balance',
        'source_budgets', 'source_volume_budgets')
    return dict(public_seconds=elapsed, **{key: info.get(key) for key in keys},
        gpu_used_bytes=int(wp.get_mempool_used_mem_current()),
        process_gpu_high_water_bytes=int(wp.get_mempool_used_mem_high()))


def run_confined(*, model, fields, initial, options, mixed, config):
    """Solve through the real FP64 fast or FP64-master/FP32 hierarchy API.

    :param model: Built model.
    :param fields: Pure recharge and fixed heads.
    :param initial: Numerical starting heads.
    :param options: Existing fast controls.
    :param mixed: Select production mixed precision.
    :param config: Existing mixed precision controls.
    :return: Accepted head and diagnostics.
    """
    if mixed:
        from DARCY_WARP_PACKAGE.solvers.mixed_fast import solve_mixed_fast
        recharge = getattr(model, '_qualification_recharge_f64', None)
        if recharge is None:
            recharge = np.empty(model._recharge_input_host.shape, dtype=np.float64)
            model._qualification_recharge_f64 = recharge
        head, info = solve_mixed_fast(model=model, initial_head_f64=initial,
            bc_values_f64=fields['bc_values'], gh_head_f64=None,
            R_f64=recharge, config=config)
        assert info['production_precision_path'] and info['converged'], info
        assert model._mixed_fast_workspace._correction_graph is not None
    else:
        head, info = model.solve(implementation='fast', initial_head=initial, **options)
        assert info['solver_backend'] == 'confined_kcycle' and info['implementation'] == 'fast' and info['converged'], info
        assert info['cuda_graph_built_this_call'] or info['cuda_graph_reused'], info
    return head, info


def qualify_steady(*, fields, dx, controls, profiles, mixed, repeats, source_control, device, directory):
    """Separate construction, cold execution, identical guesses and warm calibration.

    :param fields: Physical model inputs.
    :param dx: Cell width.
    :param controls: Source schedules and original initial head.
    :param profiles: Existing saved solver controls.
    :param mixed: Select the maintained mixed precision API.
    :param repeats: Number of warm repeats.
    :param source_control: Run equivalent combined source without WEL.
    :param device: CUDA device.
    :param directory: Evidence destination.
    :return: Repeated solve evidence.
    """
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import MixedFastConfig
    from qualify_package_kcycle import summary
    config = MixedFastConfig()
    initial = np.full(fields['active'].shape, controls['initial_head'])
    directory.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with make_model(fields=fields, dx=dx, device=device, source_control=source_control) as model:
        construction = perf_counter() - started
        rows, calibration, updates = [], [], []
        for repeat in range(repeats + 1):
            wp.synchronize_device(device=device)
            started = perf_counter()
            head, info = run_confined(model=model, fields=fields, initial=initial,
                options=profiles['steady']['fast'], mixed=mixed, config=config)
            wp.synchronize_device(device=device)
            rows.append(capture_info(info=info, elapsed=perf_counter() - started))
        cold_head = head.copy()
        if not source_control:
            for index in range(repeats):
                started = perf_counter()
                model.update_wel_in_place(wel_entries=[(cell, rate * controls['calibration_wel_factors'][index]) for cell, rate in fields['wel_entries']])
                wp.synchronize_device(device=device)
                wel_seconds = perf_counter() - started
                started = perf_counter()
                changed_recharge = fields['R_field'] * controls['calibration_recharge_factors'][index]
                model.update_R_in_place(R_field=changed_recharge)
                np.copyto(model._qualification_recharge_f64, changed_recharge)
                wp.synchronize_device(device=device)
                recharge_seconds = perf_counter() - started
                started = perf_counter()
                head, info = run_confined(model=model, fields=fields, initial=head,
                    options=profiles['steady']['fast'], mixed=mixed, config=config)
                wp.synchronize_device(device=device)
                calibration.append(capture_info(info=info, elapsed=perf_counter() - started))
                updates.append(dict(wel_seconds=wel_seconds, recharge_seconds=recharge_seconds))
        np.savez_compressed(directory / 'heads.npz', cold=cold_head, calibration=head)
    result = dict(construction_seconds=construction, cold=rows[0], warm=rows[1:],
        warm_summary=summary(rows=rows[1:]), calibration=calibration, source_updates=updates,
        controls=asdict(config) if mixed else profiles['steady']['fast'], source_control=source_control)
    (directory / 'result.json').write_text(json.dumps(result, indent=2))
    print(fields['active'].shape, 'mixed' if mixed else 'fast confined', source_control, result['warm_summary'], flush=True)
    return result


def qualify_transient(*, fields, dx, controls, profiles, repeats, unconfined, device, directory):
    """Measure fixed physical timesteps and pumping/recharge schedules.

    :param fields: Physical model inputs.
    :param dx: Cell width.
    :param controls: Physical storage and period schedule.
    :param profiles: Existing qualified controls.
    :param repeats: Number of repeated histories.
    :param unconfined: Select device-native Picard rather than confined fast.
    :param device: CUDA device.
    :param directory: Evidence destination.
    :return: Repeated history evidence and accepted package volumes.
    """
    from transient_replay_settings import default_solve_controls
    from qualify_package_kcycle import summary
    directory.mkdir(parents=True, exist_ok=True)
    initial = np.full(fields['active'].shape, controls['initial_head'])
    steps, dt = controls['history_steps'], controls['history_dt_days']
    well_factors = np.resize(controls['history_wel_pattern'], steps)
    recharge_factors = np.resize(controls['history_recharge_pattern'], steps)
    schedules = [[(cell, rate * factor) for cell, rate in fields['wel_entries']] for factor in well_factors]
    recharge = [fields['R_field'] * factor for factor in recharge_factors]
    rows = []
    with make_model(fields=fields, dx=dx, device=device, source_control=False) as model:
        for repeat in range(repeats + 1):
            wp.synchronize_device(device=device)
            started = perf_counter()
            if unconfined:
                options = default_solve_controls()
                options.update(controls['runtime_2d'])
                # The maintained Picard driver accepts uniform scalar recharge schedules.
                assert np.all(fields['R_field'] == fields['R_field'].flat[0])
                heads, info = model.solve_transient_2d_unconfined(solver='unconfined_picard_kcycle',
                    initial_head=initial, recharge_rates=np.asarray(recharge_factors) * fields['R_field'].flat[0],
                    wel_rates=schedules, k_field=fields['T_field'] / controls['initial_head'],
                    zbot_field=np.zeros_like(initial), ztop_field=np.full_like(initial, controls['unconfined_top']),
                    sy=controls['unconfined_sy'], ss=controls['specific_storage_per_m'], dt=dt,
                    active=fields['active'], bc_mask=fields['bc_mask'], bc_values=fields['bc_values'], solve_controls=options)
                periods = info['period_infos']
                assert len(periods) == steps and all(period['converged'] for period in periods), info
                assert all(period['device_side_picard_fast_path_active'] for period in periods)
                assert any(period['transient_face_kcycle_graph_count'] > 0 for period in periods)
            else:
                previous, history, periods = initial, [], []
                options = dict(profiles['transient'])
                for index in range(steps):
                    model.update_wel_in_place(wel_entries=schedules[index])
                    model.update_R_in_place(R_field=recharge[index])
                    head, period = model.solve(implementation='fast', initial_head=previous,
                        transient=True, dt=dt, head_prev=previous,
                        storage_coeff=controls['specific_storage_per_m'] * controls['confined_thickness'], **options)
                    assert period['converged'] and period['implementation'] == 'fast_transient_face_f64', period
                    assert period['graph_count'] > 0 and period['graph_fallback_count'] == 0
                    previous = head.copy()
                    history.append(previous)
                    periods.append(period)
                heads = np.asarray(history)
            wp.synchronize_device(device=device)
            rows.append(dict(public_seconds=perf_counter() - started,
                gpu_used_bytes=int(wp.get_mempool_used_mem_current()),
                process_gpu_high_water_bytes=int(wp.get_mempool_used_mem_high()),
                periods=[capture_info(info=period, elapsed=period.get('seconds', 0.)) for period in periods]))
        np.savez_compressed(directory / 'heads.npz', heads=heads)
    result = dict(cold=rows[0], warm=rows[1:], warm_summary=summary(rows=rows[1:]),
        steps=steps, dt_days=dt, unconfined=unconfined)
    (directory / 'result.json').write_text(json.dumps(result, indent=2))
    print(fields['active'].shape, 'device Picard' if unconfined else 'fast transient', result['warm_summary'], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('steady', 'mixed', 'transient', 'picard'), required=True)
    parser.add_argument('--grid', choices=('100k', '1M', 'all'), default='all')
    arguments = parser.parse_args()
    os.environ['DARCY_FLOAT'] = 'float32' if arguments.mode == 'mixed' else 'float64'
    root = Path(__file__).resolve().parents[1]
    from sanity_3d_controls import read_profiles
    profiles = read_profiles(root=root)
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import MixedFastConfig
    spectrum = MixedFastConfig()
    profiles['steady']['fast'].update(cheby_lambda_min=spectrum.cheby_lambda_min, cheby_lambda_max=spectrum.cheby_lambda_max)
    profiles['transient'] = dict(profiles['steady']['fast'])
    controls = json.loads((root / 'working_tests/wel_regional_qualification_controls.json').read_text())
    controls.update(runtime_2d=json.loads((root / 'working_tests/2d_transient_runtime_controls.json').read_text())['runtime']['solve_controls'],
        confined_thickness=10., unconfined_top=100., unconfined_sy=.15)
    shapes = {'100k': (320, 320), '1M': (1000, 1000)}
    repeats = 3
    device = 'cuda:0'
    output = root / 'benchmark_results/package_audit_repair/production_2d'
    wp.init()
    for label, shape in shapes.items():
        if arguments.grid not in ('all', label):
            continue
        fields, dx = fields_for_grid(shape=shape, controls=controls)
        if arguments.mode in ('steady', 'mixed'):
            for source_control in (True, False):
                qualify_steady(fields=fields, dx=dx, controls=controls, profiles=profiles,
                    mixed=arguments.mode == 'mixed', repeats=repeats, source_control=source_control,
                    device=device, directory=output / label / arguments.mode / ('source_control' if source_control else 'wel'))
        else:
            if arguments.mode == 'picard':
                fields['R_field'] = np.full(shape, np.mean(fields['R_field']))
            qualify_transient(fields=fields, dx=dx, controls=controls, profiles=profiles, repeats=repeats,
                unconfined=arguments.mode == 'picard', device=device, directory=output / label / arguments.mode)
