"""Durable small live examples of repaired validation, no historical campaign."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_compare import evaluate_saved_head
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
from execute_terrain_validation_programme import run_native_contact_experiments


def make_example(*,shape,masked):
    nz,ny,nx=shape
    z=np.broadcast_to(np.arange(nz+1,dtype=float)[::-1,None,None],(nz+1,ny,nx)).copy()
    active=np.ones((ny,nx),dtype=bool)
    if masked:active[0,1]=False
    fixed=np.zeros(shape,dtype=int);fixed[:,:,0]=1;fixed[:,:,-1]=1;fixed[:,~active]=0
    head=np.broadcast_to(np.linspace(2.,4.,nx),shape).copy()
    if masked:head[:]=4.
    k=np.ones(shape)
    return TerrainCase(case_id='stage0_masked' if masked else 'stage0_prescribed_heads',
        arrays=dict(interfaces=z,active=active,fixed=fixed,values=head,initial=head.copy(),source=np.zeros(shape),kx=k,ky=k.copy(),kz=k.copy()),
        physical=dict(dx=1.,dy=1.,floor=.01,steady=True,storage=0),controls={},contacts=())


def run_examples(*,root,executable):
    root.mkdir(parents=True,exist_ok=False)
    case=make_example(shape=(2,2,3),masked=False)
    case.save(directory=root/'boundary_case')
    result,unused=evaluate_saved_head(case=case,head=case.arrays['values']+10.,output=root/'shifted_heads.npz')
    if result['independent_accepted'] or result['prescribed_head_error_max_m']!=10.:raise AssertionError(result)
    atomic_json(path=root/'shifted_acceptance.json',value=result)
    case=make_example(shape=(2,3,5),masked=True)
    case.arrays['source'][0,0,3]=.01234567891234567
    case.save(directory=root/'masked_case')
    result=run_mf6(case=case,directory=root/'masked_mf6',executable=executable,xt3d=False,convertible=False,
        drains=False,contact_mode='none',head_tol=1e-12,residual_tol=1e-12)
    if not result['independent_conservation']:raise AssertionError(result)
    case=make_example(shape=(2,3,3),masked=False)
    case.case_id='stage0_original_drn_rejection'
    case.arrays['fixed'][:,:,0]=0
    case.arrays['values'][:]=4.;case.arrays['initial'][:]=4.
    case.contacts=(TerrainContact3D(cell=(0,1,0),direction='west',area=1.,distance=.5,elevation=1.5,head=0.,kind='discharge_only'),)
    case.save(directory=root/'drn_case')
    result=run_native_contact_experiments(case=case,directory=root/'native_contacts',executable=executable,outer_cap=1000)
    if result['original_convertible_drn']['native_success']:raise AssertionError('Expected original below-bottom DRN rejection')
    if not result['confined_contact_control']['native_success']:raise AssertionError(result)
    atomic_json(path=root/'contact_identities.json',value=result)


if __name__ == "__main__":
    root=Path('validation_artifacts/sevenpoint_v1/stage0/examples')
    executable=Path('/bin/modflow/mf6')
    run_examples(root=root,executable=executable)
