"""Analytic counterexample: three roots of the unchanged discrete laws.

A distinct flat control, not the original terrain problem or a proposed fix.
"""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls,CPUReference
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6
from validate_terrain_cpu_runs import run_cpu
from validate_terrain_gpu_baselines import run_gpu


def scalar_case(*,thickness,high_head,low_head,source,kx,kz,floor,initial):
    shape=(2,1,2)
    interfaces=np.broadcast_to(np.array([thickness,0.,-thickness])[:,None,None],(3,1,2)).copy()
    fixed=np.ones(shape,dtype=int);fixed[0,0,0]=0
    values=np.full(shape,low_head);values[0,0,1]=high_head
    h=values.copy();h[0,0,0]=initial
    q=np.zeros(shape);q[0,0,0]=source
    return TerrainCase(case_id='flat_scalar_three_roots',arrays=dict(interfaces=interfaces,active=np.ones((1,2),dtype=bool),
        fixed=fixed,values=values,initial=h,source=q,kx=np.full(shape,kx),ky=np.full(shape,.6*kx),kz=np.full(shape,kz),
        thickness=np.full(shape,thickness)),physical=dict(dx=1.,dy=1.,floor=floor,steady=True,storage=0),
        controls=dict(head_tol=1e-6,flow_tol=1e-12,relative_tol=1e-8,relaxation=.7,outer_cap=2000,krylov_budget=2000,
            restart=24,max_levels=6,min_coarse_n=16),contacts=())


def analytic_roots(*,high_head,low_head,source,kx,kz,thickness,floor):
    vertical=kz/thickness
    coefficients=[2*kx*high_head+vertical,-2*kx*high_head**2+vertical*(high_head-low_head)-source,
        -vertical*low_head*high_head-source*high_head]
    wet=np.sort(np.roots(coefficients))
    floor_c=2*kx*floor*high_head/(floor+high_head)
    dry=(source+floor_c*high_head+vertical*low_head)/(floor_c+vertical)
    return [float(dry),float(wet[0]),float(wet[1])]


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16/scalar_roots')
    thickness=4.
    high_head=2.
    low_head=-1.
    source=.8
    kx=1.
    kz=4.
    floor=.01
    initial=.5
    perturbation=.001
    executable=Path('/bin/modflow/mf6')
    case=scalar_case(thickness=thickness,high_head=high_head,low_head=low_head,source=source,kx=kx,kz=kz,floor=floor,initial=initial)
    root.mkdir(parents=True,exist_ok=True)
    if not (root/'case').exists():case.save(directory=root/'case')
    roots=analytic_roots(high_head=high_head,low_head=low_head,source=source,kx=kx,kz=kz,thickness=thickness,floor=floor)
    rows=[]
    ref=CPUReference(case=case)
    for index,value in enumerate(roots):
        h=case.arrays['values'].copy();h[0,0,0]=value
        op,residual,budget=ref.evaluate(head=h)
        derivative=float(ref.jacobian(operator=op,head=h)[0,0])
        start=h.copy();start[0,0,0]+=perturbation
        cpu,ci=run_cpu(case=case,directory=root/f'root_{index}_cpu',controls=CPUControls(max_outer=30,fallback=False,recovery_growth=1.),initial=start)
        gpu,gi=run_gpu(case=case,directory=root/f'root_{index}_gpu',method='newton',outer_cap=30,initial=start)
        rows.append(dict(root=value,analytic_residual=float(residual[0,0,0]),jacobian=derivative,
            floor_active=value<=floor,cpu_head=float(cpu[0,0,0]),cpu_converged=ci['converged'],
            gpu_head=float(gpu[0,0,0]),gpu_converged=gi['info']['converged']))
    mf=run_mf6(case=case,directory=root/'mf6',executable=executable,xt3d=False,convertible=True,drains=False)
    atomic_json(path=root/'analytic.json',value=dict(roots=rows,wet_polynomial=[5.,-5.8,.4],
        equation='4*h/(h+2)*(h-2) + (h+1) = 0.8 for floor<h<4; replace h in conductance by floor on dry branch',
        mf6_head=float(np.load(root/'mf6'/'extracted.npz')['head'][0,0,0]),mf6_native=mf['native_success'],
        mf6_balance=mf['independent_conservation'],mf6_upstream_expected=(source+3.)/3.,
        interpretation='Analytic nonuniqueness of this declared control only. Two smooth wet roots have inactive floor. It does not prove original-case uniqueness, nonuniqueness, physical stability, or that one formulation is preferred.'))
