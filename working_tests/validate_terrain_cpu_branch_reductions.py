"""Explicit local Dirichlet reductions of CPU-specific additional branches."""
from pathlib import Path
import json
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_cpu_runs import run_cpu
from validate_terrain_branches import constrained_diagnostics
from verify_mf6_tpfa_and_precision import longdouble_residual


def reduce_cpu_branch(*,case,root,radius,controls):
    a=np.load(root/'branches'/case.case_id/'picard_perturb_0'/'head.npz')['head']
    b=np.load(root/'extra_branches'/case.case_id/'cpu_root'/'head.npz')['head']
    cell=tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(a-b)),a.shape))
    unused,j,i=cell
    arrays={key:value.copy() for key,value in case.arrays.items()}
    arrays['fixed'][:]=1
    arrays['fixed'][:,max(0,j-radius):j+radius+1,max(0,i-radius):i+radius+1]=0
    arrays['fixed'][case.arrays['fixed']!=0]=1
    arrays['values']=a.copy();arrays['initial']=a.copy()
    reduced=TerrainCase(case_id=case.case_id+'_cpu_local',arrays=arrays,physical=case.physical.copy(),controls=case.controls.copy(),contacts=())
    output=root/'cpu_branch_reductions'/case.case_id/f'radius_{radius}'
    output.mkdir(parents=True,exist_ok=True)
    if not (output/'case').exists():reduced.save(directory=output/'case')
    heads={};results={}
    for label,initial in [('gpu',a),('cpu',b)]:
        h,result=run_cpu(case=reduced,directory=output/label,controls=controls,initial=initial)
        heads[label]=h
        results[label]=dict(result=result,precision=longdouble_residual(case=reduced,head=h),
            diagnostics=constrained_diagnostics(case=reduced,head=h))
    atomic_json(path=output/'comparison.json',value=dict(hotspot=cell,free_unknowns=int(np.sum(arrays['fixed']==0)),
        boundary='all exterior unknowns fixed to polished GPU Picard head; original global geometry/stencils retained',
        maximum_head_separation=float(np.max(np.abs(heads['gpu']-heads['cpu']))),branches=results))


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(64,1e-4),(64,.1),(64,.001)]
    radii=[3,8]
    controls=CPUControls(max_outer=100,head_tol=1e-10,flow_tol=1e-11,relative_tol=1e-11,fallback=False,recovery_growth=1.)
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        for radius in radii:
            reduce_cpu_branch(case=case,root=root,radius=radius,controls=controls)
