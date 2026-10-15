"""Additional CPU-discovered branches and larger explicit reduction controls."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_cpu_runs import run_cpu
from validate_terrain_gpu_baselines import run_gpu
from validate_terrain_branches import constrained_diagnostics


def extra_branch_checks(*,case,root,output,controls,seed,amplitude,radius,cpu_method="newton"):
    output.mkdir(parents=True,exist_ok=True)
    cold=root/'cpu_final'/case.case_id/cpu_method/'head.npz'
    if not cold.exists():cold=root/'cpu_hybrid'/case.case_id/cpu_method/'head.npz'
    if cold.exists():
        h=np.load(cold)['head']
        rng=np.random.default_rng(seed=seed)
        perturb=rng.normal(size=h.shape)*amplitude
        perturb[case.arrays['fixed']!=0]=0.
        for label,start in [('cpu_root',h),('cpu_root_perturbed',h+perturb)]:
            path=output/label
            if not path.exists():run_cpu(case=case,directory=path,controls=controls,initial=start)
        atomic_json(path=output/'cpu_root_diagnostics.json',value=constrained_diagnostics(case=case,head=h))
        for method in ('picard','newton'):
            path=output/('gpu_'+method+'_from_cpu')
            if not path.exists():run_gpu(case=case,directory=path,method=method,outer_cap=2000,initial=h+perturb)
    a=np.load(root/'branches'/case.case_id/'picard_perturb_0'/'head.npz')['head']
    b=np.load(root/'branches'/case.case_id/'newton_perturb_0'/'head.npz')['head']
    if cpu_method=='picard':b=np.load(output/'cpu_root'/'head.npz')['head']
    cell=tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(a-b)),a.shape))
    unused,j,i=cell
    arrays={key:value.copy() for key,value in case.arrays.items()}
    arrays['fixed'][:]=1
    arrays['fixed'][:,max(0,j-radius):j+radius+1,max(0,i-radius):i+radius+1]=0
    arrays['fixed'][case.arrays['fixed']!=0]=1
    arrays['values']=a.copy();arrays['initial']=a.copy()
    reduced=TerrainCase(case_id=case.case_id+f'_radius{radius}',arrays=arrays,physical=case.physical.copy(),controls=case.controls.copy(),contacts=())
    if not (output/'reduced_case').exists():reduced.save(directory=output/'reduced_case')
    for label,start in [('a',a),('b',b)]:
        path=output/('reduced_'+label)
        if not path.exists():run_cpu(case=reduced,directory=path,controls=controls,initial=start)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001)]
    controls=CPUControls(max_outer=100,head_tol=1e-10,flow_tol=1e-11,relative_tol=1e-11,fallback=False,recovery_growth=1.)
    seed=1966
    amplitude=1e-4
    radius=8
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        extra_branch_checks(case=case,root=root,output=root/'extra_branches'/case.case_id,
            controls=controls,seed=seed,amplitude=amplitude,radius=radius)
