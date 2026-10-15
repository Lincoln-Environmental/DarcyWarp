"""Restartable fresh campaign. Immutable run keys include source/runtime/inputs.

Delivered historical experiments remain immutable evidence. This entry point
creates new runs and refuses stale or interrupted outputs; choose a fresh root
when the implementation/runtime/settings change. All controls are below.
"""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import canonical_case,TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6
from validate_terrain_cpu_runs import run_cpu
from validate_terrain_gpu_baselines import run_gpu


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_reproduction')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    dx=20.
    executable=Path('/bin/modflow/mf6')
    cpu_controls=CPUControls(max_outer=6000)
    gpu_cap=2000
    reference_cap=8000
    mf6_cap=3000
    for size,k in cases:
        path=root/'cases'/f'terrain_{size}_k{k:g}'
        if not path.exists():
            case=canonical_case(size=size,basement_k=k,dx=dx)
            case.save(directory=path)
        case=TerrainCase.load(directory=path)
        run_cpu(case=case,directory=root/'cpu'/case.case_id/'newton',controls=cpu_controls)
        for method in ('picard','newton'):
            head,result=run_gpu(case=case,directory=root/'gpu'/case.case_id/method,method=method,outer_cap=gpu_cap)
            if method=='picard' and not result['info']['converged']:
                run_gpu(case=case,directory=root/'gpu'/case.case_id/'picard_reference',method=method,outer_cap=reference_cap)
        run_mf6(case=case,directory=root/'mf6'/case.case_id,executable=executable,xt3d=True,convertible=True,outer_cap=mf6_cap)
