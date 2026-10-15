"""Small saturation/floor/contact complexity controls, distinct from target cases."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
from validate_terrain_cpu_runs import run_cpu
from validate_terrain_gpu_baselines import run_gpu
from validate_terrain_mf6_feedback import run_feedback


def physics_control(*,size,dx,regime,contact,floor,slope):
    y,x=np.mgrid[:size,:size]
    top=10.+slope*x*dx
    z=np.stack([top,top-1.,top-4.,top-10.,top-30.])
    grid=TerrainGrid3D(z_interfaces=z,dx=dx,dy=dx)
    shape=grid.shape
    boundary=9.5 if regime=='wet' else 2.
    fixed=np.zeros(shape,dtype=int)
    fixed[:,:,0]=1;fixed[:,:,-1]=1
    values=np.full(shape,boundary)
    source=np.zeros(shape);source[0]=.0001*dx*dx
    source[fixed!=0]=0.
    k=np.broadcast_to(np.array([2.,8.,.5,.1])[:,None,None],shape).copy()
    contacts=()
    if contact:
        # west top cell made free so it owns one physical outlet subface.
        fixed[0,size//2,0]=0
        hb=9. if regime=='wet' else 2.
        elevation=float(grid.elevations[0,size//2,0]-.5*slope*dx)
        contacts=(TerrainContact3D(cell=(0,size//2,0),direction='west',area=.25*dx,
            distance=dx/2,elevation=elevation,head=hb,kind='discharge_only'),)
    return TerrainCase(case_id=f'control_{regime}_contact{contact}_floor{floor:g}_slope{slope:g}',
        arrays=dict(interfaces=z,active=np.ones((size,size),dtype=bool),fixed=fixed,values=values,
            initial=np.broadcast_to(top-.5,shape).copy(),source=source,kx=k,ky=.6*k,kz=.15*k,thickness=grid.thickness),
        physical=dict(dx=dx,dy=dx,floor=floor,steady=True,storage=0),
        controls=dict(head_tol=1e-6,flow_tol=1e-12,relative_tol=1e-8,relaxation=.7,
            outer_cap=2000,krylov_budget=2000,restart=24,max_levels=6,min_coarse_n=4),contacts=contacts)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16/physics_controls')
    executable=Path('/bin/modflow/mf6')
    size=8
    dx=20.
    settings=[('wet',False,.01,0.),('wet',False,.01,.02),('dry',False,.01,0.),
        ('dry',True,.01,0.),('dry',True,.001,0.),('wet',True,.01,0.)]
    cpu_controls=CPUControls(max_outer=1000)
    for regime,contact,floor,slope in settings:
        case=physics_control(size=size,dx=dx,regime=regime,contact=contact,floor=floor,slope=slope)
        path=root/case.case_id
        path.mkdir(parents=True,exist_ok=True)
        if not (path/'case').exists():case.save(directory=path/'case')
        run_cpu(case=case,directory=path/'cpu',controls=cpu_controls)
        run_gpu(case=case,directory=path/'gpu',method='newton',outer_cap=1000)
        if contact and not (path/'mf6_coupled').exists():
            run_feedback(case=case,directory=path/'mf6_coupled',executable=executable,max_feedback=30,
                contact_tolerance=1e-8,relaxation=.5,xt3d=True)
