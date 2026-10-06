"""Parity, forcing and rollback gates for the persistent production period API.

Small cases use roundoff bounds. Large-grid acceptance reports original
replay variability and preserves canonical solver/MF6 acceptance controls.
"""

from pathlib import Path
import importlib.util
import subprocess
import tempfile
import json
import time

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
from DARCY_WARP_PACKAGE.solvers.transient_config import production_secant_sy_settings
from DARCY_WARP_PACKAGE.solvers.transient_unconfined import (
    solve_transient_unconfined_backend, TransientUnconfinedSession,
)
from working_tests.transient_artifacts import (
    load_transient_artifact, spatial_fields_from_artifact, select_artifact_warm_start,
)
from working_tests.transient_replay_metrics import compare_transient
from working_tests.transient_replay_reporting import evaluate_head_accuracy
from working_tests.transient_replay_mass_balance import compute_replay_mass_balance, annotate_mass_balance_classification


def load_oracle(*, oracle_path: Path):
    spec = importlib.util.spec_from_file_location('DARCY_WARP_PACKAGE.solvers._period_oracle', oracle_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.solve_transient_unconfined_backend


def new_model(*, fields, device):
    return WarpDarcySolver(nx=fields['nx'], ny=fields['ny'], dx=fields['dx'], device=device,
                           solver_type='kcycle', use_ghb=False, diag_preconditioner_backend='device')


def period_physical_quantities(*, fields, head, head_prev, recharge, sy, ss, dt, min_sat):
    """Evaluate the native storage and boundary budget for a forcing gate."""
    from DARCY_WARP_PACKAGE.physics.budgets_2d import (
        compute_mass_balance_budget, add_exact_storage_to_budget,
    )
    from DARCY_WARP_PACKAGE.physics.storage_2d import exact_unconfined_storage_terms
    flow_sat = np.clip(head-fields['bottom'], min_sat,
                       np.maximum(fields['top']-fields['bottom'], min_sat))
    transmissivity = fields['k']*flow_sat
    transmissivity[fields['active'] == 0] = 0.
    budget = compute_mass_balance_budget(T_field=transmissivity, R_field=recharge,
        head=head, active=fields['active'], bc_mask=fields['bc_mask'],
        bc_values=fields['bc_values'], dx=fields['dx'], case='unconfined_picard_kcycle')
    storage_flux = exact_unconfined_storage_terms(head_new=head, head_old=head_prev,
        bottom=fields['bottom'], top=fields['top'], specific_yield=sy,
        specific_storage=ss, dt=dt)[0]*fields['dx']**2
    storage_flux[(fields['active'] == 0) | (fields['bc_mask'] != 0)] = 0.
    budget = add_exact_storage_to_budget(budget=budget, storage_flux=storage_flux)
    return storage_flux*dt, {name: (float(value) if name in {'percent_discrepancy', 'imbalance_fraction'}
                                 else float(value)*dt) for name, value in
        dict(budget.iloc[0]).items() if isinstance(value, (int, float, np.number))}


def run_gate(*, artifact_path: Path, oracle_path: Path, output_path: Path, device: str, mixed=False, repeat_original=False, production_acceptance=False):
    wp.init()
    artifact = load_transient_artifact(path=artifact_path)
    fields = spatial_fields_from_artifact(artifact=artifact)
    settings = production_secant_sy_settings()
    if mixed:
        settings['solve_controls']['transient_mixed_precision_enabled'] = True
    initial, _ = select_artifact_warm_start(artifact=artifact, spatial=fields,
                                           warm_start_mode=settings['warm_start_mode'])
    parameters = dict(initial_head=initial, recharge_rates=artifact['recharge_rates'],
        k_field=fields['k'], zbot_field=fields['bottom'], ztop_field=fields['top'],
        sy=float(artifact['sy']), ss=float(artifact['ss']), dt=float(artifact['dt_days']),
        active=fields['active'], bc_mask=fields['bc_mask'], bc_values=fields['bc_values'],
        solve_controls=settings['solve_controls'], save_diagnostics=True, return_info=True)
    oracle = load_oracle(oracle_path=oracle_path)
    outputs = []
    for function in (oracle, oracle if repeat_original else solve_transient_unconfined_backend):
        model = new_model(fields=fields, device=device)
        wp.synchronize_device(device)
        start = time.perf_counter()
        heads, info = function(model=model, **parameters)
        wp.synchronize_device(device)
        outputs.append((heads, info, time.perf_counter()-start))
    old, new = outputs
    native_acceptance = []
    for heads, result, wall in outputs:
        result['warm_start_head'] = initial
        result['unconfined_storage_mode'] = 'mf6_convertible_secant_sy'
        comparison = compare_transient(warp_result=result,
            mf6_heads_per_period=artifact['heads_per_period'],
            mf6_heads_final=artifact['heads_final'], active=fields['active'])
        accuracy = evaluate_head_accuracy(comparison=comparison)
        budget = compute_replay_mass_balance(spatial=fields, recharge_rates=artifact['recharge_rates'],
            sy=parameters['sy'], ss=parameters['ss'], dt=parameters['dt'], formulation='unconfined',
            unconfined_storage_mode='mf6_convertible_secant_sy', warp_result=result, min_sat=.1)
        annotate_mass_balance_classification(budget)
        assert accuracy['passed'], accuracy
        assert budget['mass_balance_passed'], budget
        assert all(row['strict_picard_convergence_passed'] for row in result['period_infos'])
        native_acceptance.append(dict(head_accuracy=accuracy,
            mass_balance_passed=budget['mass_balance_passed'], cumulative=budget['cumulative'],
            maximum_discrepancy_percent=budget['max_abs_percent_discrepancy']))
    output_path.with_suffix('.diagnosis.json').write_text(json.dumps(dict(
        per_period_max_difference=[float(np.max(np.abs(before-after))) for before, after in zip(old[0],new[0])],
        old=[{key: row.get(key) for key in ('outer_iterations','total_inner_kcycles',
             'startup_inner_kcycles','adaptive_dt_substep_count','adaptive_dt_retry_count',
             'final_head_residual_rms','strict_picard_convergence_passed')}
             for row in old[1]['period_infos']],
        new=[{key: row.get(key) for key in ('outer_iterations','total_inner_kcycles',
             'startup_inner_kcycles','adaptive_dt_substep_count','adaptive_dt_retry_count',
             'final_head_residual_rms','strict_picard_convergence_passed')}
             for row in new[1]['period_infos']]),indent=2))
    parity_tolerance = settings['solve_controls']['hclose'] if production_acceptance else 1.e-10
    np.testing.assert_allclose(old[0], new[0], rtol=0., atol=parity_tolerance)
    diagnostic_keys = ('outer_iterations', 'strict_picard_convergence_passed',
        'practical_picard_acceptance_passed', 'total_inner_kcycles', 'startup_inner_kcycles',
        'adaptive_dt_substep_count', 'adaptive_dt_retry_count', 'adaptive_dt_substep_dts',
        'final_flow_residual_rms', 'final_head_residual_rms')
    for before, after in zip(old[1]['period_infos'], new[1]['period_infos']):
        if production_acceptance:
            assert before['strict_picard_convergence_passed'] == after['strict_picard_convergence_passed']
            assert before.get('adaptive_dt_substep_dts') == after.get('adaptive_dt_substep_dts')
            continue
        for key in diagnostic_keys:
            if key.startswith('final_'):
                np.testing.assert_allclose(before.get(key), after.get(key), rtol=1.e-7, atol=1.e-10)
            else:
                assert before.get(key) == after.get(key), (key, before.get(key), after.get(key))
    for key in old[1]:
        if production_acceptance:
            continue
        if key.endswith('_per_period') and isinstance(old[1][key], np.ndarray):
            np.testing.assert_allclose(old[1][key], new[1][key], rtol=1.e-9, atol=1.e-8, err_msg=key)
    # Uniform spatial forcing reproduces the first original production period.
    model = new_model(fields=fields, device=device)
    session = TransientUnconfinedSession(model=model, **{name: parameters[name] for name in (
        'k_field', 'zbot_field', 'ztop_field', 'sy', 'ss', 'active', 'bc_mask', 'bc_values',
        'solve_controls')})
    forcing = np.full(initial.shape, float(artifact['recharge_rates'][0]))
    forcing[(fields['active'] == 0) | (fields['bc_mask'] != 0)] = 0.
    head, info = session.advance(head_prev=initial, recharge_rate_m_per_day=forcing,
                                 dt=parameters['dt'])
    np.testing.assert_allclose(head, old[0][0], rtol=0., atol=parity_tolerance)
    for key in diagnostic_keys:
        if not production_acceptance and not key.startswith('final_'):
            assert info.get(key) == old[1]['period_infos'][0].get(key), key
    original_storage, original_budget = period_physical_quantities(fields=fields,
        head=old[0][0], head_prev=initial, recharge=forcing,
        sy=parameters['sy'], ss=parameters['ss'], dt=parameters['dt'], min_sat=.1)
    spatial_storage, spatial_budget = period_physical_quantities(fields=fields,
        head=head, head_prev=initial, recharge=forcing,
        sy=parameters['sy'], ss=parameters['ss'], dt=parameters['dt'], min_sat=.1)
    if not production_acceptance:
        np.testing.assert_allclose(original_storage, spatial_storage, rtol=1.e-7, atol=1.e-8)
        for name, value in original_budget.items():
            np.testing.assert_allclose(value, spatial_budget[name], rtol=1.e-7, atol=1.e-8,
                                       err_msg=name)
    pointers = {name: value.ptr for name, value in session.workspace['buffers'].items()}
    hierarchy = id(model.mg_levels)
    graph_ids = {key: id(graph) for key, graph in session.workspace['kcycle_graphs'].items()}
    # Provisional trial discarded; alternative from accepted initial must
    # equal a fresh session. Different recharge and dt exercise graph keys.
    alternate = forcing*0.7
    alt_head, alt_info = session.advance(head_prev=initial, recharge_rate_m_per_day=alternate,
                                        dt=parameters['dt']*0.5)
    fresh_model = new_model(fields=fields, device=device)
    fresh = TransientUnconfinedSession(model=fresh_model, **{name: parameters[name] for name in (
        'k_field', 'zbot_field', 'ztop_field', 'sy', 'ss', 'active', 'bc_mask', 'bc_values',
        'solve_controls')})
    fresh_head, fresh_info = fresh.advance(head_prev=initial, recharge_rate_m_per_day=alternate,
                                          dt=parameters['dt']*0.5)
    np.testing.assert_allclose(alt_head, fresh_head, rtol=0., atol=parity_tolerance)
    for key in diagnostic_keys:
        if not production_acceptance and not key.startswith('final_'):
            assert alt_info.get(key) == fresh_info.get(key), key
    assert pointers == {name: value.ptr for name, value in session.workspace['buffers'].items()}
    assert hierarchy == id(model.mg_levels)
    assert all(id(session.workspace['kcycle_graphs'][key]) == value for key, value in graph_ids.items())
    report = dict(artifact=str(artifact_path), device=device, periods=len(old[0]),
        head_max_difference_m=float(np.max(np.abs(old[0]-new[0]))),
        acceptance_exact=True, work_exact=all(a.get('total_inner_kcycles') == b.get('total_inner_kcycles')
            for a,b in zip(old[1]['period_infos'],new[1]['period_infos'])),
        native_head_parity_tolerance_m=parity_tolerance, native_replay_acceptance=native_acceptance,
        uniform_spatial_forcing_max_difference_m=float(np.max(np.abs(head-old[0][0]))),
        uniform_spatial_storage_max_difference_m3=float(np.max(np.abs(spatial_storage-original_storage))),
        uniform_spatial_total_storage_difference_m3=float(np.sum(spatial_storage-original_storage)),
        uniform_spatial_budget_difference={name: spatial_budget[name]-value
            for name, value in original_budget.items()},
        rollback_alternate_max_difference_m=float(np.max(np.abs(alt_head-fresh_head))),
        persistent_buffers=True, persistent_hierarchy=True,
        persistent_graphs=True, setup_count=session.setup_count,
        original_wall_s=old[2], refactored_wall_s=new[2],
        original_period_diagnostics=[{name: row.get(name) for name in diagnostic_keys}
                                     for row in old[1]['period_infos']],
        refactored_period_diagnostics=[{name: row.get(name) for name in diagnostic_keys}
                                       for row in new[1]['period_infos']])
    output_path.write_text(json.dumps(report, indent=2, default=str)+'\n')


def validate_revision(*, repository: Path, oracle_revision: str, artifact_path: Path,
                      output_path: Path, device: str, mixed: bool,
                      repeat_original: bool, production_acceptance: bool) -> None:
    """Read Git's original production source without changing the checkout."""
    relative_source = 'DARCY_WARP_PACKAGE/solvers/transient_unconfined.py'
    original = subprocess.run(['git', 'show', f'{oracle_revision}:{relative_source}'],
        cwd=repository, check=True, text=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory() as directory:
        oracle_path = Path(directory)/'transient_oracle.py'
        oracle_path.write_text(original)
        run_gate(artifact_path=artifact_path, oracle_path=oracle_path,
            output_path=output_path, device=device, mixed=mixed,
            repeat_original=repeat_original, production_acceptance=production_acceptance)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Original replay versus persistent period API.')
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--oracle-revision', default='5163cb1')
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--repeat-original', action='store_true')
    parser.add_argument('--production-acceptance', action='store_true',
        help='Use unchanged replay acceptance and hclose for the nondeterministic large-grid comparison.')
    options = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    validate_revision(repository=repository, oracle_revision=options.oracle_revision,
        artifact_path=options.artifact, output_path=options.output, device=options.device,
        mixed=options.mixed, repeat_original=options.repeat_original,
        production_acceptance=options.production_acceptance)
