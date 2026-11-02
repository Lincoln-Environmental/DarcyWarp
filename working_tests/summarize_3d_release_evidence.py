# SPDX-License-Identifier: AGPL-3.0-or-later
"""Index retained test results, scientific comparisons and source identities."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
import xml.etree.ElementTree as ET


def test_results(*, xml_path, log_path):
    root = ET.parse(source=xml_path).getroot()
    cases = list(root.iter('testcase'))
    failures = [case for case in cases if case.find('failure') is not None]
    errors = [case for case in cases if case.find('error') is not None]
    skipped = [case for case in cases if case.find('skipped') is not None]
    warnings = re.findall(pattern=r'(\d+) warnings?', string=log_path.read_text())
    return dict(tests=len(cases), passed=len(cases)-len(failures)-len(errors)-len(skipped),
        failed=len(failures), errors=len(errors), skipped=len(skipped),
        warnings=int(warnings[-1]) if warnings else 0,
        unsuccessful=[dict(module=case.attrib['classname'], test=case.attrib['name']) for case in failures+errors])


def summarize(*, evidence, project, test_paths, executable, source_patterns, ci_groups):
    evidence = evidence.resolve()
    runs = {}
    for stem in test_paths:
        xml_path = evidence/(stem+'.xml')
        if xml_path.is_file():
            runs[stem] = test_results(xml_path=xml_path, log_path=evidence/(stem+'.log'))
    for group in ci_groups:
        runs['ci_'+group] = test_results(xml_path=evidence/'ci'/(group+'.xml'),
                                        log_path=evidence/('ci_'+group+'.log'))
    gate = json.loads(s=(evidence/'independent_confined_verified/gate.json').read_text())
    cases = gate['cases']
    comparison = dict(cases=len(cases), maximum_warp_scipy_head_error_m=max(
        row['devices']['cpu']['head_max_difference_m'] for row in cases), maximum_mf6_scipy_head_error_m=max(
        row['mf6_head_max_difference_m'] for row in cases), maximum_supplied_head_flux_error_m3_day=max(
        row['same_head_native_flux_max_error'] for row in cases), maximum_warp_relative_cell_imbalance=max(
        row['devices']['cpu']['independent']['relative_cell_imbalance'] for row in cases))
    sources = {}
    for pattern in source_patterns:
        for path in project.glob(pattern=pattern):
            sources[str(path.relative_to(project))] = hashlib.sha256(string=path.read_bytes()).hexdigest()
    runtime = dict(python=platform.python_version(), packages={name:importlib.metadata.version(distribution_name=name)
        for name in ('warp-lang', 'numpy', 'scipy', 'pytest', 'flopy', 'pandas')},
        platform=platform.platform(), mf6_version=subprocess.check_output(args=[str(executable), '-v'], text=True).strip(),
        mf6_sha256=hashlib.sha256(string=executable.read_bytes()).hexdigest(),
        revision=subprocess.check_output(args=['git','rev-parse','HEAD'], text=True).strip(),
        source_sha256=sources, cuda_release_gate='Deferred at user request; CPU-only validation')
    full_suite = runs.get('stage_c_uncapped_cpu', runs.get('stage_c_full_cpu'))
    result = dict(runtime=runtime, tests=runs, independent_confined=comparison,
                  full_suite_is_green=bool(full_suite and not (full_suite['failed'] or full_suite['errors'])))
    (evidence/'validation_summary.json').write_text(data=json.dumps(obj=result, indent=2)+'\n')
    manifest = {str(path.relative_to(project)):hashlib.sha256(string=path.read_bytes()).hexdigest()
                for path in evidence.rglob(pattern='*') if path.is_file() and path.name not in ('SHA256SUMS.json', 'summary.log')
                and not any(part.endswith('_initial') or part.endswith('_pre_cpu_request') for part in path.parts)
                and 'independent_confined' not in path.parts and 'terrain_roots' not in path.parts}
    (evidence/'SHA256SUMS.json').write_text(data=json.dumps(obj=manifest, indent=2)+'\n')
    return result


if __name__ == "__main__":
    evidence = Path('release_evidence')
    project = Path.cwd()
    executable = Path('/bin/modflow/mf6')
    test_paths = ('stage_a_before', 'stage_a_baseline_final_tests', 'stage_a_final_baseline', 'reference_controller_before',
                  'stage_a_after', 'stage_a_extended', 'stage_a_final_cpu', 'stage_b_cpu',
                  'stage_b_cpu_recheck', 'stage_c_full_cpu', 'stage_c_uncapped_cpu', 'fas_baseline', 'final_contract',
                  'final_acceptance_recheck', 'final_npf_dispatch', 'final_focused', 'final_budget_mask_validation')
    ci_groups = ('contract', 'npf', 'terrain', '2d')
    source_patterns = ('DARCY_WARP_PACKAGE/**/*.py', 'working_tests/*3d_release*.py',
                       'working_tests/profile_npf662_reference_cache.py', 'examples/*.py', 'tests/*.py')
    result = summarize(evidence=evidence, project=project, test_paths=test_paths,
                       executable=executable, source_patterns=source_patterns, ci_groups=ci_groups)
    print(json.dumps(obj=result['tests'], indent=2))
