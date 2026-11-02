# SPDX-License-Identifier: AGPL-3.0-or-later
"""Small independent SciPy/Warp CPU/pinned-MF6 checks and retained terrain roots."""
from pathlib import Path
import numpy as np
from validate_npf662_confined import run_confined_matrix
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase, atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls, solve_cpu


def verify_roots(*, fixture_directory, output, controls, seed, perturbation_scale):
    case = TerrainCase.load(directory=fixture_directory)
    perturbation = np.random.default_rng(seed=seed).normal(size=case.shape)*perturbation_scale
    perturbation[case.arrays['fixed'] != 0] = 0.
    output.mkdir(parents=True, exist_ok=True)
    rows = {}
    heads = []
    with np.load(file=fixture_directory/'seeds.npz') as seeds:
        for label in ('first', 'second'):
            head, info = solve_cpu(case=case, controls=controls, initial=seeds[label]+perturbation)
            if not info['converged']:
                raise AssertionError(info)
            np.savez_compressed(file=output/(label+'.npz'), head=head)
            rows[label] = info
            heads.append(head)
    rows['maximum_branch_separation_m'] = float(np.max(np.abs(heads[0]-heads[1])))
    atomic_json(path=output/'results.json', value=rows)
    return rows


if __name__ == "__main__":
    root = Path('validation_runs/3d_release/confined')
    variants = ('x', 'y', 'z', 'constant', 'layered', 'heterogeneous', 'masked', 'disconnected')
    shape = (3, 4, 5)
    spacing = (2., 3., 4.)
    end_heads = (8., 9.)
    contrast = (-3., 3.)
    source = .01234567891234567
    seed = 1966
    layer_kz = (2., .0001, 8.)
    devices = ('cpu',)
    executable = Path('/bin/modflow/mf6')
    fixture_directory = Path('tests/fixtures/terrain_multiple_roots')
    terrain_output = Path('validation_runs/3d_release/terrain_roots')
    controls = CPUControls(max_outer=30, head_tol=1e-10, flow_tol=1e-12, relative_tol=1e-11,
                           fallback=False, recovery_growth=1.)
    perturbation_scale = 1e-4
    run_confined_matrix(root=root, variants=variants, shape=shape, spacing=spacing, end_heads=end_heads,
        contrast=contrast, source=source, seed=seed, layer_kz=layer_kz, devices=devices, executable=executable)
    roots = verify_roots(fixture_directory=fixture_directory, output=terrain_output, controls=controls,
                        seed=seed, perturbation_scale=perturbation_scale)
    print('Terrain branch separation [m]:', roots['maximum_branch_separation_m'])
