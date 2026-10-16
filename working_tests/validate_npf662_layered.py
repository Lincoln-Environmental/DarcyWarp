"""Local DIS geometry matrix using the existing four-method evidence runner."""
from dataclasses import replace
from pathlib import Path
import hashlib
import numpy as np

from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from validate_npf662_confined import make_case
from validate_npf662_unconfined import run_case, source_snapshot


def make_layered_control(*, variation, shape, seed, unconfined=False, initial='mixed'):
    base=make_case(shape=shape,spacing=(2.,3.,2.),variant='heterogeneous',end_heads=(8.,9.),
        contrast=(-2.,2.),source=.003,seed=seed,layer_kz=(1.,1.,1.))
    k,j,i=np.indices(shape)
    thickness=np.broadcast_to(2.+.15*k,shape).copy()
    surface=np.full(shape[1:],10.)
    if variation in ('x','independent','masked'):thickness+=.05*i;surface+=.1*i[0]
    if variation in ('y','independent','masked'):thickness+=.08*j;surface+=.12*j[0]
    if variation=='masked':base.active[:,shape[1]//2,shape[2]//2]=0
    interfaces=np.concatenate([surface[None],surface[None]-np.cumsum(thickness,axis=0)],axis=0)
    types=np.zeros(shape,dtype=int)
    if unconfined:
        types[:]=1;types[1]=0
        if initial=='wet':head=interfaces[:-1]+1.
        elif initial=='dry':head=interfaces[1:]-.2
        else:head=np.where((i+j+k)%2,interfaces[1:]-.2,base.values)
    else:head=base.initial.copy()
    return NPF662LayeredCase(kx=base.kx,ky=base.ky,kz=base.kz,active=base.active,fixed=base.fixed,
        values=base.values,source=base.source,initial=head,dx=base.dx,dy=base.dy,
        z_interfaces=interfaces,cell_type=types)


def make_layered_dry_source(*,initial):
    shape=(1,1,2);one=np.ones(shape)
    return NPF662LayeredCase(kx=one,ky=.6*one,kz=.15*one,active=one.astype(int),
        fixed=np.array([0,1]).reshape(shape),values=np.zeros(shape),source=np.array([1.,0.]).reshape(shape),
        initial=np.full(shape,initial),dx=1.,dy=1.,
        z_interfaces=np.array([[[2.,4.]],[[0.,0.]]]),cell_type=1)


def run_layered_matrix(*,root,phase,shape,seed,devices,executable,max_outer,krylov_budget):
    root.mkdir(parents=True,exist_ok=False)
    cases=[]
    if phase=='confined':
        for variation in ('uniform','x','y','independent','masked'):
            cases.append((variation,make_layered_control(variation=variation,shape=shape,seed=seed)))
    elif phase=='unconfined':
        for initial in ('wet','dry','mixed'):
            cases.append(('heterogeneous_'+initial,make_layered_control(variation='independent',shape=shape,
                seed=seed,unconfined=True,initial=initial)))
        cases.append(('masked_mixed',make_layered_control(variation='masked',shape=shape,seed=seed,unconfined=True)))
        for initial in (-.2,1.5):
            cases.append((f'dry_source_initial_{initial:g}',make_layered_dry_source(initial=initial)))
    else:raise ValueError('Choose explicit confined/unconfined phase')
    rows=[]
    for case_id,case in cases:
        magnitude=max(1.,float(np.max(np.abs(case.initial))),float(np.max(np.abs(case.values[case.fixed!=0]))))
        row=run_case(case=case,case_id=case_id,root=root,devices=devices,executable=executable,
            max_outer=max_outer,krylov_budget=krylov_budget,seed=seed,
            native_head_tolerance_m=max(1e-14,32.*np.finfo(float).eps*magnitude))
        rows.append(row)
    result=dict(phase=phase,passed=all(row['passed'] for row in rows),cases=rows,
        source_sha256=source_snapshot(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic_json(path=root/'gate.json',value=result)
    if not result['passed']:raise AssertionError('Layered gate failed; retain all native/numerical evidence')
    return result


def run_geology_case(*,root,shape,dx,devices,executable,max_outer,krylov_budget,seed):
    from shanz_terrain_case_study import build_case
    root.mkdir(parents=True,exist_ok=False)
    ny,nx=shape
    old=build_case(nx=nx,ny=ny,dx=dx)
    case=NPF662LayeredCase(kx=old['k'],ky=.6*old['k'],kz=.15*old['k'],
        active=np.broadcast_to(old['active'],old['k'].shape).astype(int).copy(),fixed=old['fixed'],
        values=old['values'],source=old['source'],initial=old['initial'],dx=dx,dy=dx,
        z_interfaces=old['interfaces'],cell_type=1)
    # Intentional new boundary formulation: no terrain contact is relabelled.
    changes=dict(identity='synthetic_varying_geology_DIS_CHD_WEL_no_contacts',same_as_historical_terrain=False,
        preserved='Exact existing builder interfaces,K,coastal CHD,WEL source and cold initial head',
        changed='Conventional DIS/NPF law instead of transformed terrain; all three west terrain discharge contacts omitted and those exterior faces no-flow; no terrain floor',
        omitted_contacts=[dict(cell=contact.cell,head=contact.head,area=contact.area,distance=contact.distance,
            elevation=contact.elevation,kind=contact.kind) for contact in old['contacts']])
    atomic_json(path=root/'boundary_mapping.json',value=changes)
    magnitude=max(1.,float(np.max(np.abs(case.initial))),float(np.max(np.abs(case.values))))
    row=run_case(case=case,case_id='varying_geology',root=root,devices=devices,executable=executable,
        max_outer=max_outer,krylov_budget=krylov_budget,seed=seed,
        native_head_tolerance_m=max(1e-14,32.*np.finfo(float).eps*magnitude))
    result=dict(passed=row['passed'],case=row,source_sha256=source_snapshot(),boundary_mapping=changes)
    atomic_json(path=root/'gate.json',value=result)
    return result


if __name__ == "__main__":
    root=Path('validation_artifacts/npf662_layered_v1/confined/experiments')
    phase='confined'
    shape=(3,4,5)
    seed=2231
    devices=['cpu','cuda:0']
    executable=Path('/bin/modflow/mf6')
    max_outer=200
    krylov_budget=2000
    gate=run_layered_matrix(root=root,phase=phase,shape=shape,seed=seed,devices=devices,
        executable=executable,max_outer=max_outer,krylov_budget=krylov_budget)
