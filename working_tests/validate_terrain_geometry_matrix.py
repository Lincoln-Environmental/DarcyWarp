"""Analytic physical-geometry controls and fixed-physical-domain refinement."""
from pathlib import Path
import numpy as np
from scipy.sparse.linalg import spsolve
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls,CPUReference,solve_cpu
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D,TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
from validate_terrain_gpu_baselines import run_gpu
from validate_terrain_cpu_runs import run_cpu


def geometry_case(*,size,length,slope,variable,regime):
    spacing=length/size
    y,x=np.meshgrid((np.arange(size)+.5)*spacing,(np.arange(size)+.5)*spacing,indexing='ij')
    top=10.+slope*x+.5*slope*y
    s=x/length
    t1=2.+(.2*np.sin(2*np.pi*s) if variable else np.zeros_like(s))
    t2=3.+(.3*np.cos(2*np.pi*y/length) if variable else np.zeros_like(s))
    t3=np.full_like(s,4.)
    z=np.stack([top,top-t1,top-t1-t2,top-t1-t2-t3])
    grid=TerrainGrid3D(z_interfaces=z,dx=spacing,dy=spacing)
    exact=grid.elevations.copy() if regime=='vertical_affine' else np.full(grid.shape,100.)
    fixed=np.zeros(grid.shape,dtype=int)
    fixed[0]=fixed[-1]=1
    fixed[:,:,0]=fixed[:,:,-1]=1
    fixed[:,0]=fixed[:,-1]=1
    initial=exact+.01
    initial[fixed!=0]=exact[fixed!=0]
    k=np.ones(grid.shape)
    arrays=dict(interfaces=z,active=np.ones((size,size),dtype=bool),fixed=fixed,values=exact,
        initial=initial,source=np.zeros(grid.shape),kx=k,ky=.6*k,kz=.15*k,
        thickness=grid.thickness,volumes=grid.volumes)
    return TerrainCase(case_id=f'geometry_n{size}_slope{slope:g}_variable{variable}_{regime}',arrays=arrays,
        physical=dict(dx=spacing,dy=spacing,floor=.01,steady=True,storage=0,physical_extent=[length,length]),
        controls=dict(head_tol=1e-6,flow_tol=1e-12,relative_tol=1e-8,relaxation=.7,outer_cap=2000,
            krylov_budget=2000,restart=24,max_levels=6,min_coarse_n=4),contacts=())


def run_geometry(*,case,root,executable):
    directory=root/case.case_id
    directory.mkdir(parents=True,exist_ok=True)
    if not (directory/'case').exists():case.save(directory=directory/'case')
    case=TerrainCase.load(directory=directory/'case')
    exact=case.arrays['values']
    op=TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=case.grid()),kx=case.arrays['kx'],
        ky=case.arrays['ky'],kz=case.arrays['kz'],head=exact,contacts=())
    qx,qy,qz,unused=op.fluxes(head=exact)
    target=.15*case.physical['dx']*case.physical['dy'] if 'vertical_affine' in case.case_id else 0.
    analytic=dict(qx_max=float(np.max(np.abs(qx))),qy_max=float(np.max(np.abs(qy))),
        qz_error_max=float(np.max(np.abs(qz-target))),physical_target_downward_m3_day=target)
    if not (directory/'cpu').exists():
        run_cpu(case=case,directory=directory/'cpu',controls=CPUControls(max_outer=100))
    if not (directory/'gpu').exists():
        run_gpu(case=case,directory=directory/'gpu',method='newton',outer_cap=100)
    for xt3d in (False,True):
        label='mf6_xt3d' if xt3d else 'mf6_tpfa'
        if not (directory/label).exists():
            run_mf6(case=case,directory=directory/label,executable=executable,xt3d=xt3d,convertible=False,drains=False)
        path=directory/label/'extracted.npz'
        if path.exists():
            data=np.load(path)
            free=case.arrays['fixed']==0
            analytic[label]=dict(head_rms_error=float(np.sqrt(np.mean((data['head'][free]-exact[free])**2))),
                layer_transfer_total=data['qz'].sum(axis=(1,2)).tolist(),
                target_transfer_total=target*case.shape[1]*case.shape[2],
                interpretation='MF6 confined, DarcyWarp h=z has uniform f=.5; vertical physical target unaffected by horizontal effective K')
    atomic_json(path=directory/'analytic.json',value=analytic)
    return analytic


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16/geometry')
    executable=Path('/bin/modflow/mf6')
    length=320.
    sizes=[8,16,32]
    slope=.1
    controls=[(8,0.,False,'constant'),(8,0.,False,'vertical_affine'),
        (8,slope,False,'vertical_affine'),(8,slope,True,'constant')]
    rows={}
    for size,control_slope,variable,regime in controls:
        case=geometry_case(size=size,length=length,slope=control_slope,variable=variable,regime=regime)
        rows[case.case_id]=run_geometry(case=case,root=root,executable=executable)
    for size in sizes:
        case=geometry_case(size=size,length=length,slope=slope,variable=True,regime='vertical_affine')
        rows[case.case_id]=run_geometry(case=case,root=root,executable=executable)
    atomic_json(path=root/'matrix.json',value=rows)
