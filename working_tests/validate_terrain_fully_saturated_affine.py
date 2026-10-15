"""Nonzero fully saturated affine control: physical and discrete face areas."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference,CPUControls
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6
from validate_terrain_geometry_matrix import geometry_case
from validate_terrain_cpu_runs import run_cpu
from validate_terrain_gpu_baselines import run_gpu


def run_affine_control(*,size,length,slope,gradient_x,gradient_y,gradient_z,datum,root,executable):
    case=geometry_case(size=size,length=length,slope=slope,variable=True,regime='vertical_affine')
    g=case.grid()
    x=(np.arange(size)+.5)*g.dx
    y=(np.arange(size)+.5)*g.dy
    exact=datum+gradient_x*x[None,None,:]+gradient_y*y[None,:,None]+gradient_z*g.elevations
    case.case_id=f'fully_saturated_affine_xy_{size}'
    case.arrays['values']=exact
    case.arrays['initial']=exact+.01
    case.arrays['initial'][case.arrays['fixed']!=0]=exact[case.arrays['fixed']!=0]
    path=root/case.case_id;path.mkdir(parents=True,exist_ok=True)
    if not (path/'case').exists():case.save(directory=path/'case')
    ref=CPUReference(case=case)
    op,r,b=ref.evaluate(head=exact)
    q=op.fluxes(head=exact)
    physical_qx=-gradient_x*g.dy*.5*(g.thickness[:,:,:-1]+g.thickness[:,:,1:])
    physical_qy=-.6*gradient_y*g.dx*.5*(g.thickness[:,:-1,:]+g.thickness[:,1:,:])
    expected_qz=g.dx*g.dy*(.15*gradient_z-gradient_x*ref.metrics.zx_interfaces[1:-1]-.6*gradient_y*ref.metrics.zy_interfaces[1:-1])
    physical_div=op.divergence(fluxes=(physical_qx,physical_qy,expected_qz,q[3]))
    result=dict(geometry='uniform physical K; affine physical h; analytic source is zero',
        exact_head_Darcy_residual_rms=b['cell_residual_rms'],
        true_midface_area_flux_error_max=float(np.max(np.abs(q[0]-physical_qx))),
        true_y_midface_area_flux_error_max=float(np.max(np.abs(q[1]-physical_qy))),
        vertical_metric_flux_error_max=float(np.max(np.abs(q[2]-expected_qz))),
        physical_affine_balance_rms=float(np.sqrt(np.mean(physical_div[ref.free]**2))),
        interpretation='Darcy horizontal half-cell harmonic thickness is not the arithmetic midpoint physical face thickness. Existing affine tests assert the harmonic formula, not exact physical shared-face area.')
    for method in ('cpu','gpu'):
        if method=='cpu':head,info=run_cpu(case=case,directory=path/method,controls=CPUControls(max_outer=50))
        else:head,info=run_gpu(case=case,directory=path/method,method='newton',outer_cap=50)
        result[method]=dict(head_rms_error=float(np.sqrt(np.mean((head[ref.free]-exact[ref.free])**2))))
    native=run_mf6(case=case,directory=path/'mf6',executable=executable,xt3d=True,convertible=False,drains=False)
    data=np.load(path/'mf6'/'extracted.npz')
    result['mf6']=dict(native=native['native_success'],conservation=native['independent_conservation'],
        head_rms_error=float(np.sqrt(np.mean((data['head'][ref.free]-exact[ref.free])**2))),
        interpretation='MF6 vertical connection normal is Cartesian vertical; does not claim geological-interface-normal flux identity')
    atomic_json(path=path/'analytic.json',value=result)
    return result


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16/fully_saturated_affine_xy')
    sizes=[8,16,32]
    length=320.
    slope=.1
    gradient_x=.01
    gradient_y=.006
    gradient_z=.02
    datum=200.
    executable=Path('/bin/modflow/mf6')
    rows={}
    for size in sizes:
        rows[str(size)]=run_affine_control(size=size,length=length,slope=slope,gradient_x=gradient_x,
            gradient_y=gradient_y,gradient_z=gradient_z,datum=datum,root=root,executable=executable)
    atomic_json(path=root/'matrix.json',value=rows)
