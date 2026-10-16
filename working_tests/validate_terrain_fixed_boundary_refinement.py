"""Fixed domain AND physical boundary surfaces; no moving fixed-cell centres."""
from pathlib import Path
import json
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_geometry_matrix import geometry_case
from validate_terrain_cpu_runs import run_cpu
from validate_terrain_gpu_baselines import run_gpu


def fixed_boundary_case(*,size,length,slope):
    case=geometry_case(size=size,length=length,slope=slope,variable=True,regime='vertical_affine')
    case.case_id=f'fixed_boundaries_refinement_{size}'
    case.arrays['fixed'][:]=0
    case.arrays['initial']=case.arrays['values']+.01
    g=case.grid();m=TerrainMetrics3D(grid=g)
    contacts=[]
    for direction,k,interface in [('top',0,0),('bottom',case.shape[0]-1,case.shape[0])]:
        for j,i in np.ndindex((size,size)):
            area=g.dx*g.dy*np.sqrt(1+m.zx_interfaces[interface,j,i]**2+m.zy_interfaces[interface,j,i]**2)
            contacts.append(TerrainContact3D(cell=(k,j,i),direction=direction,area=float(area),
                distance=float(g.thickness[k,j,i]/2),elevation=float(g.z_interfaces[interface,j,i]),
                head=float(g.z_interfaces[interface,j,i]),kind='head'))
    case.contacts=tuple(contacts)
    return case


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16/fixed_boundary_refinement_v2')
    sizes=[8,16,32]
    length=320.
    slope=.1
    executable=Path('/bin/modflow/mf6')
    rows={}
    for size in sizes:
        case=fixed_boundary_case(size=size,length=length,slope=slope)
        path=root/case.case_id
        path.mkdir(parents=True,exist_ok=True)
        if not (path/'case').exists():case.save(directory=path/'case')
        run_cpu(case=case,directory=path/'cpu',controls=CPUControls(max_outer=50))
        run_gpu(case=case,directory=path/'gpu',method='newton',outer_cap=50)
        rows[case.case_id]={}
        for xt3d in (False,True):
            label='mf6_xt3d' if xt3d else 'mf6_tpfa'
            result=run_mf6(case=case,directory=path/label,executable=executable,xt3d=xt3d,
                convertible=False,drains=True,contact_mode='native_head_contacts')
            data=np.load(path/label/'extracted.npz')
            rows[case.case_id][label]=dict(native=result['native_success'],conservation=result['independent_conservation'],
                head_rms=float(np.sqrt(np.mean((data['head']-case.arrays['values'])**2))),
                qx_max=float(np.max(np.abs(data['qx']))),qy_max=float(np.max(np.abs(data['qy']))),
                qz_error_max=float(np.max(np.abs(data['qz']-.15*(length/size)**2))),
                boundary_surface='same analytic top/bottom physical surfaces, lateral physical x/y planes no-flow; top/bottom physical Kz projected-area GHB on MF6 stepped grid')
        atomic_json(path=root/'matrix.json',value=rows)
