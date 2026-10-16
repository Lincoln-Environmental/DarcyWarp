"""Publish already measured local evidence; never run or change a solver."""
from datetime import datetime, timezone
from pathlib import Path
import shutil
import csv

from analyze_npf662_baseline import read_json, write_report
from benchmark_npf662_baseline import digest_file, snapshot_sources, write_json


def publish(*, baseline, analysis_directory, output, native_directories, test_logs, documentation, cold_directory):
    output.mkdir(parents=True, exist_ok=False)
    complete = read_json(path=analysis_directory/'complete_results.json')
    result = complete['primary']; analysis = complete['analysis']
    cold = read_json(path=cold_directory/'results.json')
    if not all(row['accepted'] for row in cold.values()):
        raise AssertionError('Cold process failed physical acceptance')
    for name, record in result['variants'].items():
        if not (record['mf6_accepted'] and len(record['warm_repeats']) == result['settings']['repeats']
                and all(row['accepted'] for row in record['warm_repeats'])):
            raise AssertionError(f'No fully accepted baseline for {name}')
        nam = (native_directories[name]/'npf662.nam').read_text().upper()
        if 'RCH6' not in nam or 'WEL6' in nam:
            raise AssertionError('Published native reference must use RCH without WEL')
    write_report(baseline=baseline, output=output, result=result,
        audits=analysis['native_audits'], profiles=analysis['launch_profiles'], linearity=analysis['linearity'],
        resident=analysis['resident_workspace'])
    report_path = output/'REPORT.rst'
    original_report = report_path.read_text()
    heading = (
        'Final canonical case\n=====================\n\n'
        'Shape (nz,ny,nx)=(5,250,250); 100 m square cells; 25 km square domain.\n'
        'Recharge is 150 mm/year (365 days), supplied by native RCH6, with IRCH=1\n'
        'and FIXED_CELL. No WEL package occurs in either published native deck.\n'
        'The only CHD boundary is the west column at head 0 m. Layer 0 is\n'
        'convertible in the unconfined variant; all other layers are confined.\n'
        'No conductance saturation floor or governing equation was changed.\n\n'
        'At the final recharge and cell size, the convertible variant reaches\n'
        'about 82.904 m maximum head; 59,054/62,500 columns are above model top.\n'
        'It is mostly in the fully saturated regime. Surface discharge/runoff\n'
        'is outside the specified single-coast/no-flow model.\n\n'
        'summary.csv is the consolidated performance table. complete_results.json\n'
        'contains all settings, measurements, counters and comparisons. Native\n'
        'HEAD/CBC/GRB/decks/iteration CSVs are in native/confined and native/unconfined.\n'
        'Exact input NPZs and direct input arrays remain in ../run_001/<variant>.\n'
        'Decoded native arrays remain in ../analysis_003/reference_arrays/<variant>.\n'
        'The source/runtime/hardware metadata and all failures remain archived.\n\n'
        'Validation: 6 focused regressions and 16 existing targeted regressions passed.\n'
        'The backend metadata retains a legacy WEL_SOURCE name for its volumetric\n'
        'RHS interface; the actual native package is RCH6, verified from each NAM\n'
        'file and the returned recharge budget.\n\n'
    )
    report_path.write_text(data=heading+original_report)
    with report_path.open(mode='a') as stream:
        stream.write('\nTrue fresh-process cold end-to-end\n----------------------------------\n\n')
        for name, row in cold.items():
            stream.write(f"{name}: {row['fresh_process_wall_seconds']:.6f} seconds.\n")
        stream.write('\nIncludes interpreter/imports, CUDA context startup, private empty-cache JIT, model/workspace/graphs, solve and independent physical acceptance, through process exit. One cold process per variant; these are not timing medians. The earlier cold column is the separately measured component total.\n')
    for name in ('summary.csv', 'complete_results.json', 'work_and_time_breakdown.png'):
        shutil.copy2(src=analysis_directory/name, dst=output/name)
    complete['cold_process'] = cold
    write_json(path=output/'complete_results.json', value=complete)
    with (output/'summary.csv').open() as stream:
        rows = list(csv.DictReader(f=stream))
    for name, row in cold.items():
        rows.append(dict(experiment=f'{name}_fresh_process_cold', backend='device_newton_gmres_diagonal_cold_process',
            accepted=row['accepted'], seconds=row['fresh_process_wall_seconds'], end_to_end_seconds=row['fresh_process_wall_seconds'],
            iterations=row['info']['krylov_iterations'], peak_mib=row['memory']['device_sampled_peak_mib']))
    columns = sorted(set().union(*(row.keys() for row in rows)))
    with (output/'summary.csv').open(mode='w', newline='') as stream:
        writer = csv.DictWriter(f=stream, fieldnames=columns); writer.writeheader(); writer.writerows(rows)
    shutil.copy2(src=cold_directory/'executed_driver.py', dst=output/'executed_cold_process_driver.py')
    shutil.copy2(src=cold_directory/'results.json', dst=output/'cold_process.json')
    for name in ('newton_residual_history.png', 'kcycle_residual_history.png', 'kcycle_levels_time_memory.png'):
        shutil.copy2(src=baseline/name, dst=output/name)
    for name, directory in native_directories.items():
        shutil.copytree(src=directory, dst=output/'native'/name)
    executable = Path(result['metadata']['executable'])
    if digest_file(path=executable) != result['metadata']['mf6_sha256']:
        raise AssertionError('Native executable changed since measurement')
    shutil.copy2(src=executable, dst=output/'mf6_6.6.2')
    shutil.copy2(src=documentation, dst=output/documentation.name)
    for index, path in enumerate(test_logs):
        shutil.copy2(src=path, dst=output/f'validation_{index}.log')
    sources = [Path(__file__), Path('working_tests/analyze_npf662_baseline.py'),
        Path('working_tests/benchmark_npf662_baseline.py'), Path('working_tests/measure_npf662_cold_process.py'), documentation]
    source_hashes = snapshot_sources(root=output, paths=sources)
    write_json(path=output/'publication.json', value=dict(completed_utc=datetime.now(tz=timezone.utc).isoformat(),
        source_sha256=source_hashes, measurements=str(analysis_directory), primary_measurements=str(baseline),
        actual_solver_source_snapshots='See primary and analysis source_snapshot directories; publication snapshots are rendering/current reproduction code.',
        all_native_and_Newton_repeats_accepted=True,
        failed_linear_configurations=sum(not row['accepted'] for row in result['kcycle']),
        successful_linear_configurations=sum(row['accepted'] for row in result['kcycle'])))
    manifest = {str(path.relative_to(output)): dict(bytes=path.stat().st_size, sha256=digest_file(path=path))
        for path in output.rglob(pattern='*') if path.is_file()}
    write_json(path=output/'manifest.json', value=manifest)
    for relative, entry in manifest.items():
        if digest_file(path=output/relative) != entry['sha256']:
            raise AssertionError(f'Publication integrity check failed: {relative}')
    print(f'Published and verified {len(manifest)} files: {output}', flush=True)


if __name__ == "__main__":
    baseline = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/run_001')
    analysis_directory = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/analysis_003')
    output = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/final')
    native_directories = dict(
        confined=Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/analysis_002/native_refined/confined'),
        unconfined=analysis_directory/'native_refined'/'unconfined')
    test_logs = [Path('/tmp/darcy_benchmark_final_tests.log'), Path('/tmp/darcy_baseline_final_existing_tests.log')]
    documentation = Path('docs/NPF662_250_BASELINE.rst')
    cold_directory = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/cold_process_001')
    publish(baseline=baseline, analysis_directory=analysis_directory, output=output,
        native_directories=native_directories, test_logs=test_logs, documentation=documentation, cold_directory=cold_directory)
