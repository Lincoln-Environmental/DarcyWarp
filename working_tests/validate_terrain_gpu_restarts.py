"""Post-fix cold comparisons and explicitly labelled GPU cross-start checks."""
from pathlib import Path
import json
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from validate_terrain_gpu_baselines import run_gpu


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001)]
    amplitude=.001
    seed=538
    rng=np.random.default_rng(seed=seed)
    cross_cap=500
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        for method in ('picard','newton'):
            path=root/'gpu_after'/case.case_id/method
            if not path.exists():
                run_gpu(case=case,directory=path,method=method,outer_cap=case.controls['outer_cap'])
            other='picard' if method=='newton' else 'newton'
            source=root/'gpu_before'/case.case_id/other
            if not json.loads((source/'summary.json').read_text())['info']['converged']:
                source=root/'gpu_before'/case.case_id/'picard_reference'
            initial=np.load(source/'head.npz')['head']
            noise=rng.normal(size=initial.shape)
            noise[case.arrays['fixed']!=0]=0.
            initial=initial+amplitude*noise
            path=root/'gpu_crossstarts'/case.case_id/(method+'_from_'+other)
            if not path.exists():
                run_gpu(case=case,directory=path,method=method,outer_cap=cross_cap,initial=initial)
