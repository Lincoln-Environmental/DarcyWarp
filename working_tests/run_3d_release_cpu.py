# SPDX-License-Identifier: AGPL-3.0-or-later
"""Practical CPU CI groups; use pytest tests/ separately for the full suite."""
from pathlib import Path
import json
import os
import platform
import subprocess
import sys
import importlib.metadata


def run_group(*, group, groups, output):
    output.mkdir(parents=True, exist_ok=True)
    if group not in groups:
        raise ValueError(f'Unknown CPU CI group: {group}')
    versions = {name: importlib.metadata.version(distribution_name=name) for name in
                ('warp-lang', 'numpy', 'scipy', 'pytest', 'flopy')}
    identity = dict(python=platform.python_version(), packages=versions,
        commit=subprocess.check_output(args=['git', 'rev-parse', 'HEAD'], text=True).strip(),
        cpu_only=os.environ.get('CUDA_VISIBLE_DEVICES') == '-1', tests=groups[group])
    (output/f'{group}-runtime.json').write_text(json.dumps(identity, indent=2)+'\n')
    result = subprocess.run(args=[sys.executable, '-m', 'pytest', '-q', *groups[group],
        f'--junitxml={output/group}.xml'], check=False)
    return result.returncode


if __name__ == "__main__":
    group = os.environ.get('DARCYWARP_RELEASE_GROUP', 'contract')
    output = Path('release_evidence/ci')
    groups = dict(
        contract=['tests/test_3d_release_contract.py', 'tests/test_3d_session.py',
                  'tests/test_npf662_acceptance.py', 'tests/test_terrain_runtime.py',
                  'tests/test_3d_operator_analytical.py', 'tests/test_fast_3d_kcycle.py'],
        npf=['tests/test_npf662_reference_controller.py', 'tests/test_npf662_acceptance.py',
             'tests/test_npf662_kcycle.py', 'tests/test_3d_release_contract.py'],
        terrain=['tests/test_3d_terrain_reference.py', 'tests/test_terrain_contact_stencils.py',
                 'tests/test_terrain_component_validation.py', 'tests/test_terrain_newton_accounting.py'],
        **{'2d': ['tests/test_2d_transient.py', 'tests/test_nonlinear_operator_2d.py',
                  'tests/test_solver_registry_2d.py', 'tests/test_sparse_system_consistency.py']})
    status = run_group(group=group, groups=groups, output=output)
    sys.exit(status)
