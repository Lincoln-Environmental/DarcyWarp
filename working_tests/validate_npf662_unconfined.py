"""Local staged matched-Newton controls, then an explicitly flat catchment."""
from pathlib import Path
import hashlib
import subprocess
from time import perf_counter
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662Reference,save_npf662_case
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
from validate_npf662_confined import make_gpu_model
from validate_npf662_dry_start import make_dry_case
from validate_terrain_scalar_roots import scalar_case


def make_section(*,shape,initial,seed,heterogeneous,masked):
    rng=np.random.default_rng(seed=seed)
    k=10.**rng.uniform(low=-2.,high=2.,size=shape) if heterogeneous else np.ones(shape)
    active=np.ones(shape,dtype=int)
    if masked:active[:,shape[1]//2,shape[2]//2]=0
    fixed=np.zeros(shape,dtype=int);fixed[:,:,0]=1;fixed[:,:,-1]=1
    values=np.broadcast_to(np.linspace(2.8,3.2,shape[2]),shape).copy()
    bottoms=np.broadcast_to((4.-2.*np.arange(1,shape[0]+1))[:,None,None],shape)
    head=np.full(shape,5.) if initial=='wet' else bottoms-.2
    if initial=='mixed':head=np.where(np.indices(shape).sum(axis=0)%2,head,values)
    source=np.zeros(shape);source[0,1,1]=.004
    types=np.ones(shape,dtype=int);types[1]=0
    return NPF662Case(kx=k,ky=.6*k,kz=.15*k,active=active,fixed=fixed,values=values,source=source,
        initial=head,dx=2.,dy=3.,dz=2.,top=4.,cell_type=types)


def make_vertical_case():
    shape=(3,1,1);one=np.ones(shape)
    values=np.array([3.,-2.,-7.]).reshape(shape);fixed=np.array([1,0,1]).reshape(shape)
    return NPF662Case(kx=one,ky=.6*one,kz=.15*one,active=one.astype(int),fixed=fixed,values=values,
        source=np.zeros(shape),initial=np.array([3.,-4.2,-7.]).reshape(shape),dx=1.,dy=1.,dz=4.,top=4.,cell_type=1)


def make_scalar_case(*,initial):
    old=scalar_case(thickness=4.,high_head=2.,low_head=-1.,source=.8,kx=1.,kz=4.,floor=.01,initial=initial)
    a=old.arrays
    return NPF662Case(kx=a['kx'],ky=a['ky'],kz=a['kz'],active=old.grid().active.astype(int),fixed=a['fixed'],
        values=a['values'],source=a['source'],initial=a['initial'],dx=1.,dy=1.,dz=4.,top=4.,cell_type=1)


def run_case(*,case,case_id,root,devices,executable,max_outer,krylov_budget,seed,supplied_check=True,native_head_tolerance_m=1e-14):
    directory=root/case_id;directory.mkdir(parents=True,exist_ok=False)
    save_npf662_case(case=case,path=directory/'inputs.npz')
    ref=NPF662Reference(case=case)
    start=perf_counter()
    cpu,ci,cf=ref.solve(max_outer=max_outer)
    ci['seconds']=perf_counter()-start
    atomic_json(path=directory/'scipy_cpu.json',value=finite_diagnostics(value=ci))
    np.savez_compressed(directory/'scipy_cpu.npz',head=cpu,**cf)
    row=dict(case_id=case_id,input_hash=case.digest(),cpu=ci,devices={})
    passed=ci['independent_accepted']
    for device in devices:
        with make_gpu_model(case=case,device=device) as model:
            head,info=model.solve(unconfined=bool(np.any(case.cell_types)),
                top_elevation=None if case.geometry=='structured_dis_layered' else case.top,npf_cell_type=case.cell_types,
                max_outer=max_outer,max_cycles=krylov_budget)
        independent,fields=ref.evaluate(head=head)
        atomic_json(path=directory/('warp_'+device.replace(':','_')+'.json'),value=finite_diagnostics(value=info))
        error=float(np.max(np.abs((head-cpu)[ref.free]))) if ref.free.any() else 0.
        np.savez_compressed(directory/('warp_'+device.replace(':','_')+'.npz'),head=head,**fields)
        row['devices'][device]=dict(info=info,independent=independent,head_max_difference_m=error,
            face_max_difference=float(np.max(np.abs(fields['flux']-cf['flux']))) if cf['flux'].size else 0.)
        passed=bool(passed and info['converged'] and independent['independent_accepted'] and error<=1e-7)
    native,mi,mf=run_reference_mf6(case=case,directory=directory/'mf6',executable=executable,head_tolerance_m=native_head_tolerance_m)
    row['mf6']=mi
    if native is not None:
        error=float(np.max(np.abs((native-cpu)[ref.free]))) if ref.free.any() else 0.
        unused,at=ref.evaluate(head=native)
        row['mf6_head_max_difference_m']=error
        row['mf6_same_head_flux_max_difference']=float(np.max(np.abs(mf['flux_native']-at['flux'])))
        parity=np.allclose(mf['flux_native'],at['flux'],rtol=3e-12,atol=2e-11)
        passed=bool(passed and mi['native_success'] and mi['independent_conservation'] and
            mi['independent']['independent_accepted'] and error<=1e-7 and parity and mi['saturation_max_error']<=1e-12)
    else:passed=False
    if supplied_check:
        head=case.bottoms+.35*case.thicknesses+.02*np.random.default_rng(seed=seed).random(size=case.active.shape)
        head[case.fixed!=0]=case.values[case.fixed!=0]
        native,si,sf=run_reference_mf6(case=case,directory=directory/'mf6_supplied',executable=executable,prescribed_state=head,head_tolerance_m=native_head_tolerance_m)
        unused,at=ref.evaluate(head=head)
        row['supplied_native_success']=si['native_success']
        if sf is None:
            passed=False
            row['supplied_failure']=si['stop_reason']
        else:
            row['supplied_flux_max_difference']=float(np.max(np.abs(sf['flux_native']-at['flux'])))
            row['supplied_residual_max_difference']=float(np.max(np.abs(sf['native_residual']-at['residual'])))
            passed=bool(passed and si['native_success'] and np.allclose(sf['flux_native'],at['flux'],rtol=3e-12,atol=2e-11)
                and np.allclose(sf['native_residual'],at['residual'],rtol=3e-12,atol=2e-11) and si['saturation_max_error']<=1e-12)
    row['passed']=passed
    atomic_json(path=directory/'comparison.json',value=finite_diagnostics(value=row))
    print(case_id,passed,ci.get('outers'),{d:r['info'].get('outer_iterations') for d,r in row['devices'].items()},flush=True)
    return row


def verify_derivatives(*,case,root,seed,steps):
    root.mkdir(parents=True,exist_ok=False)
    ref=NPF662Reference(case=case);rng=np.random.default_rng(seed=seed)
    head=case.bottoms+.4*case.dz+.05*rng.random(size=case.active.shape)
    head[case.fixed!=0]=case.values[case.fixed!=0]
    v=rng.normal(size=case.active.shape);v[~ref.free]=0.
    J=ref.jacobian(head=head);jv=(J@v.ravel()).reshape(case.active.shape)
    rows=[]
    for epsilon in steps:
        unused,plus=ref.evaluate(head=head+epsilon*v);unused,minus=ref.evaluate(head=head-epsilon*v)
        fd=-(plus['residual']-minus['residual'])/(2.*epsilon)
        rows.append(dict(epsilon=epsilon,relative_l2=float(np.linalg.norm((fd-jv)[ref.free])/np.linalg.norm(jv[ref.free]))))
    if min(row['relative_l2'] for row in rows)>2e-7:raise AssertionError(rows)
    atomic_json(path=root/'differences.json',value=dict(steps=rows,max_columns=int(np.max(np.diff(J.indptr)))))
    np.savez_compressed(root/'directions.npz',head=head,direction=v,jv=jv)


def run_small_matrix(*,root,devices,executable,max_outer,krylov_budget,seed):
    root.mkdir(parents=True,exist_ok=False)
    entries=[('scalar_upstream',make_scalar_case(initial=-.2)),('horizontal_dry_source',make_dry_case(initial=-.2,source=1.)),
        ('vertical_partial',make_vertical_case())]
    for initial in ('wet','dry','mixed'):
        entries.append(('heterogeneous_'+initial,make_section(shape=(3,4,7),initial=initial,seed=seed,heterogeneous=True,masked=False)))
    entries.append(('masked_mixed',make_section(shape=(3,4,7),initial='mixed',seed=seed,heterogeneous=True,masked=True)))
    verify_derivatives(case=entries[-1][1],root=root/'derivatives',seed=seed,steps=[1e-3,1e-4,1e-5,1e-6,1e-7])
    rows=[]
    for case_id,case in entries:
        rows.append(run_case(case=case,case_id=case_id,root=root,devices=devices,executable=executable,
            max_outer=max_outer,krylov_budget=krylov_budget,seed=seed))
    gate=dict(stage=2,scope='Small declared controls only',passed=all(row['passed'] for row in rows),cases=rows)
    gate['source_sha256']=source_snapshot()
    atomic_json(path=root/'gate.json',value=finite_diagnostics(value=gate))
    if not gate['passed']:raise AssertionError('Small matched-unconfined gate failed; larger runs prohibited')
    return gate


def run_existing_catchment_remap(*,root,shape,dx,dz,seed,devices,executable,max_outer,krylov_budget):
    from run_3d_unconfined_warp_vs_mf6 import build_simple_unconfined_multilayer_case
    root.mkdir(parents=True,exist_ok=False)
    nz,ny,nx=shape
    old=build_simple_unconfined_multilayer_case(nx=nx,ny=ny,nlay=nz,dx=dx,layer_thickness=dz,
        transmissivity=3000.,recharge=1e-4,initial_saturated_thickness=dz*.5,
        heterogeneous_t=True,seed=seed,workspace=root/'builder')
    top=float(old.bc_values_3d[old.bc_mask_3d!=0].min())+dz-1.
    k=np.broadcast_to(old.hk_2d,shape).copy()
    # A distinct flat-geology case. Preserve all original K, sources, masks,
    # prescribed heads and the exact builder's cold head. No demo main runs.
    case=NPF662Case(kx=k,ky=k.copy(),kz=k.copy(),active=old.active_3d,fixed=old.bc_mask_3d,
        values=old.bc_values_3d,source=old.rhs_3d,initial=old.initial_head_3d,
        dx=dx,dy=dx,dz=dz,top=top,cell_type=1)
    np.savez_compressed(root/'original_geometry.npz',top=old.top_3d,bottom=old.bottom_3d,
        source=old.rhs_3d,head=old.initial_head_3d,active=old.active_3d,fixed=old.bc_mask_3d,values=old.bc_values_3d)
    notes=dict(identity='legacy_catchment_flat_geometry_CHD_WEL',same_as_original_terrain=False,
        changed='All interfaces replaced by flat contiguous uniform-dz planes; WEL source ownership follows contract reservoir convention',
        preserved='Original builder K in all axes, masks, original volumetric source, coastal prescribed heads and cold initial field',
        top=top,dz=dz,shape=shape,original_top_min=float(old.top_3d.min()),original_top_max=float(old.top_3d.max()))
    atomic_json(path=root/'geometry_mapping.json',value=notes)
    magnitude=max(1.,float(np.max(np.abs(case.initial[case.active!=0]))),float(np.max(np.abs(case.values[case.fixed!=0]))))
    native_head_tolerance_m=max(1e-14,32.*np.finfo(float).eps*magnitude)
    notes.update(native_head_tolerance_m=native_head_tolerance_m,
        iteration_tolerance_policy='32 machine eps times maximum active initial/prescribed head magnitude; independent gates unchanged')
    atomic_json(path=root/'geometry_mapping.json',value=notes)
    row=run_case(case=case,case_id='flat_catchment',root=root,devices=devices,executable=executable,
        max_outer=max_outer,krylov_budget=krylov_budget,seed=seed,supplied_check=False,native_head_tolerance_m=native_head_tolerance_m)
    atomic_json(path=root/'gate.json',value=finite_diagnostics(value=dict(stage=2,passed=row['passed'],case=row,geometry=notes,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256=source_snapshot())))


def source_snapshot():
    paths=list(Path('DARCY_WARP_PACKAGE/solvers').glob('npf662*.py'))
    paths+=list(Path('DARCY_WARP_PACKAGE/validation').glob('npf662*.py'))
    paths+=[Path('DARCY_WARP_PACKAGE/warped_darcy_3d.py'),Path('DARCY_WARP_PACKAGE/solvers/session_3d.py'),
        Path('DARCY_WARP_PACKAGE/physics/npf662_layers.py'),
        Path('working_tests/validate_npf662_unconfined.py')]
    return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


if __name__ == "__main__":
    root=Path('validation_artifacts/sevenpoint_v1/stage2/final_experiments')
    devices=['cpu','cuda:0']
    executable=Path('/bin/modflow/mf6')
    max_outer=200
    krylov_budget=2000
    seed=1937
    catchment_shape=(4,24,32)
    catchment_dx=20.
    catchment_dz=20.
    small=run_small_matrix(root=root/'small',devices=devices,executable=executable,max_outer=max_outer,krylov_budget=krylov_budget,seed=seed)
    run_existing_catchment_remap(root=root/'catchment',shape=catchment_shape,dx=catchment_dx,dz=catchment_dz,
        seed=seed,devices=devices,executable=executable,max_outer=max_outer,krylov_budget=krylov_budget)
