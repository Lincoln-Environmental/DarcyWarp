"""Archive compact, auditable results of the device-history qualification."""
from pathlib import Path
import hashlib
import json
import statistics


def summarize_values(*, values):
    return dict(minimum=min(values), median=statistics.median(values), maximum=max(values))


def summarize_qualification(*, path):
    data = json.loads(path.read_text())
    if not data['passed']:
        raise ValueError(f'Qualification failed: {path}')
    cases = []
    for case in data['cases']:
        repeats = [run for run in case['runs'] if not run['warmup']]
        steps = [step for run in repeats for step in run['periods']]
        all_steps = [step for run in case['runs'] for step in run['periods']]
        if not all(step['accepted'] for step in all_steps):
            raise ValueError(f'Unaccepted endpoint: {case["case"]}')
        cases.append(dict(case=case['case'], case_sha256=case['case_sha256'],
            public_seconds=[run['public_solve_seconds'] for run in repeats],
            numerical_seconds=[sum(step['backend_seconds'] for step in run['periods']) for run in repeats],
            krylov=[sum(step['krylov'] for step in run['periods']) for run in repeats],
            outer=[sum(step['outer_iterations'] for step in run['periods']) for run in repeats],
            audited_endpoints_including_warmup=len(all_steps),
            relative_imbalance=summarize_values(values=[step['physical']['relative_cell_imbalance'] for step in steps]),
            head_residual_rms=summarize_values(values=[step['physical']['head_residual_rms'] for step in steps]),
            head_max_error_m=max(step['head_max_m'] for step in all_steps),
            flux_relative_l2_max=max(step['flux_relative_l2'] for step in all_steps),
            history_readbacks=[run['history_readbacks'] for run in repeats]))
    return dict(raw_path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        revision=data['revision'], source_sha256=data['npf_source_sha256'],
        controls=data['settings'], summary=data['summary'], cases=cases)


def summarize(*, qualifications, calibration_path, manifest_path, output, verification):
    groups = {name:summarize_qualification(path=path) for name,path in qualifications.items()}
    first = next(iter(groups.values()))['source_sha256']
    if any(group['source_sha256'] != first for group in groups.values()):
        raise ValueError('Qualification source fingerprints differ')
    reference = json.loads(manifest_path.read_text())
    calibration = json.loads(calibration_path.read_text())
    if not calibration['passed']:
        raise ValueError('Calibration qualification failed')
    calibration_rows = []
    for row in calibration['cases']:
        history = row['device_history']
        calibration_rows.append(dict(k_factor=row['k_factor'],
            host_device_head_max_m=row['worst_head_difference_m'],
            history_step_api_head_max_m=history['worst_step_api_head_difference_m'],
            history_public_seconds=history['public_seconds'],
            history_numerical_seconds=history['numerical_seconds'],
            history_extra_audit_seconds=history['extra_audit_seconds'],
            audited_history_endpoints=len(history['independent_physical']),
            runs={name:{key:value for key,value in run.items() if key != 'steps'}
                for name,run in row['runs'].items()}))
    report = dict(status='One-second strict target achieved; development qualification',
        device='NVIDIA RTX 4070 Ti SUPER, 16 GiB',
        physics='500x500x5 upper-layer-convertible NPF/STO; 20 weekly steps; original initial heads; no SS presolve',
        precision='FP64 physics, heads, FGMRES and acceptance; FP32 multigrid correction',
        timing='Entire synchronized public history API including internal setup, source/timestep updates, acceptance and output download. External model construction and independent benchmark audits separate. MF6 is one saved engine execution.',
        qualifications=groups,
        verification=verification,
        mf6_relative_imbalance=summarize_values(values=[step['physical']['relative_cell_imbalance']
            for step in reference['native']['periods']]),
        calibration=dict(raw_path=str(calibration_path), sha256=hashlib.sha256(calibration_path.read_bytes()).hexdigest(),
            case_sha256=calibration['case_sha256'], controls=calibration['controls'],
            timing='One run per changed K, reused models, each from original heads. History API runs after the individual-step comparison has prepared the current K. K update and first changed-K work are reported separately in step-API results; no MF6 speedup claimed for changed K.',
            cases=calibration_rows))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+'\n')
    for name,group in groups.items():
        print(name, json.dumps(group['summary']))
    print('calibration', json.dumps(calibration_rows))


if __name__ == "__main__":
    root = Path('benchmark_results/gpu_native')
    qualifications = {
        'strict': root/'stretch_history_strict_qualification/results.json',
        'mf6_accuracy': root/'stretch_history_mf6_qualification/results.json',
        'confined': root/'stretch_history_confined_qualification/results.json',
    }
    calibration_path = root/'stretch_history_calibration/results.json'
    manifest_path = Path('/tmp/darcywarp-gpu-main-20261111/release_evidence/realistic_reference_cache/3ce0d90e2320da4f03725c60760dd2a32bb83af68748756a7484de0d16b00617/manifest.json')
    output = root/'history_qualification.json'
    verification = dict(tests_passed=1697, skipped=4, native_mf6_deselected=52,
        skip_reason='CPU parameterizations of CUDA-only graph/reduction checks',
        tests_command='MPLBACKEND=Agg DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/run_3d_gpu_tests.py',
        documentation='Sphinx -W HTML build passed',
        native_references='Fingerprint-verified saved MF6 results; no fresh MF6 execution in this checkpoint')
    summarize(qualifications=qualifications, calibration_path=calibration_path,
        manifest_path=manifest_path, output=output, verification=verification)
