"""Version-specific angle-default experiment and saved connection geometry."""
from pathlib import Path
import numpy as np
import flopy
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6


def decode_saturation(*,path,shape):
    budget=flopy.utils.CellBudgetFile(path,precision='double')
    records=budget.get_data(text='DATA-SAT')[-1]
    sat=np.zeros(shape)
    for r in records:sat.ravel()[int(r['node'])-1]=float(r['sat'])
    budget.close()
    return sat[:,::-1,:]


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    executable=Path('/bin/modflow/mf6')
    case=TerrainCase.load(directory=root/'fixed_boundary_refinement_v2'/'fixed_boundaries_refinement_8'/'case')
    path=root/'angle_default'/'xt3d_no_angle2'
    result=run_mf6(case=case,directory=path,executable=executable,xt3d=True,convertible=False,
        drains=True,use_connection_elevations=False)
    data=np.load(path/'extracted.npz')
    atomic_json(path=root/'angle_default'/'comparison.json',value=dict(native_success=result['native_success'],
        independent_conservation=result['independent_conservation'],head_rms_error=float(np.sqrt(np.mean((data['head']-case.arrays['values'])**2))),
        qx_max=float(np.max(np.abs(data['qx']))),qy_max=float(np.max(np.abs(data['qy']))),
        physical_tensor_unchanged=True,only_difference='ANGLE2/ANGLE3 omitted rather than explicitly zero; XT3D nozee flag'))
    cases=[(32,1e-4),(64,1e-4),(64,.1)]
    rows={}
    output=root/'geometry_mapping'
    output.mkdir(parents=True,exist_ok=True)
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        directory=root/'mf6_feedback_xt3d'/case.case_id
        last=sorted(directory.glob('feedback_*/terrain.cbc'))[-1]
        sat=decode_saturation(path=last,shape=case.shape)
        g=case.grid()
        horizontal_z=g.bottoms+.5*sat*g.thickness
        vertical_z=g.elevations
        np.savez_compressed(output/(case.case_id+'.npz'),horizontal_connection_centres=horizontal_z,
            vertical_connection_centres=vertical_z,saturation=sat)
        rows[case.case_id]=dict(max_connection_centroid_offset=float(np.max(np.abs(horizontal_z-vertical_z))),
            meaning='MF6 DIS 6.6.2 source-confirmed head-dependent horizontal vector endpoint; vertical vector uses full-thickness midpoint. This is not a new physical-head mapping.',
            source='Dis.f90 connection_vector')
    atomic_json(path=output/'summary.json',value=rows)
