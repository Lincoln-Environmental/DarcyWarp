"""Local confined parity gate; exact inputs, native files and failures retained."""
from pathlib import Path
import hashlib
import subprocess
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def make_case(*,shape,spacing,variant,end_heads,contrast,source,seed,layer_kz):
    rng=np.random.default_rng(seed=seed)
    active=np.ones(shape,dtype=int)
    if variant=='masked':active[:,shape[1]//2,shape[2]//2]=0
    if variant=='disconnected':active[:,shape[1]//2,:]=0
    k=np.ones(shape)
    if variant in ('heterogeneous','masked'):k=10.**rng.uniform(low=contrast[0],high=contrast[1],size=shape)
    fixed=np.zeros(shape,dtype=int)
    axis={'x':2,'y':1,'z':0}.get(variant,2)
    left=[slice(None)]*3;right=left.copy();left[axis]=0;right[axis]=-1
    fixed[tuple(left)]=1;fixed[tuple(right)]=1;fixed[active==0]=0
    profile=np.linspace(end_heads[0],end_heads[1],shape[axis])
    if variant=='constant':profile[:]=end_heads[0]
    dimensions=[1,1,1];dimensions[axis]=shape[axis]
    values=np.broadcast_to(profile.reshape(dimensions),shape).copy()
    kz=.15*k
    if variant=='layered':
        kz=np.broadcast_to(np.asarray(layer_kz)[:,None,None],shape).copy()
        fixed[:]=0;fixed[0]=1;fixed[-1]=1
        resistance=.5*spacing[2]/np.asarray(layer_kz[:-1])+.5*spacing[2]/np.asarray(layer_kz[1:])
        profile=end_heads[0]+(end_heads[1]-end_heads[0])*np.concatenate([[0.],np.cumsum(resistance)])/resistance.sum()
        values=np.broadcast_to(profile[:,None,None],shape).copy()
    q=np.zeros(shape)
    if variant in ('heterogeneous','masked'):q[1,1,2]=source
    return NPF662Case(kx=k,ky=.6*k,kz=kz,active=active,fixed=fixed,values=values,source=q,
        initial=np.full(shape,end_heads[0]),dx=spacing[0],dy=spacing[1],dz=spacing[2])


def make_gpu_model(*,case,device):
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    nz,ny,nx=case.active.shape
    model=WarpDarcySolver3D(nx=nx,ny=ny,nz=nz,dx=case.dx,dy=case.dy,dz=case.dz,
        device=device,implementation='fast',formulation='mf6_6_6_2_npf')
    model.build_from_K_fields(kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,active=case.active,
        bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
    return model


def run_confined_matrix(*,root,variants,shape,spacing,end_heads,contrast,source,seed,layer_kz,devices,executable):
    root.mkdir(parents=True,exist_ok=False)
    rows=[]
    for variant in variants:
        directory=root/variant;directory.mkdir()
        case=make_case(shape=shape,spacing=spacing,variant=variant,end_heads=end_heads,
            contrast=contrast,source=source,seed=seed,layer_kz=layer_kz)
        reference=NPF662Reference(case=case)
        np.savez_compressed(directory/'inputs.npz',**{n:np.asarray(getattr(case,n)) for n in case.__dataclass_fields__})
        head,checks,fields=reference.solve()
        np.savez_compressed(directory/'scipy_cpu.npz',head=head,**fields)
        result=dict(case=variant,input_hash=case.digest(),cpu=checks,devices={})
        for device in devices:
            with make_gpu_model(case=case,device=device) as model:
                solved,info=model.solve(max_cycles=2000)
            independent,physical=reference.evaluate(head=solved)
            np.savez_compressed(directory/('warp_'+device.replace(':','_')+'.npz'),head=solved,**physical)
            error=float(np.max(np.abs((solved-head)[reference.free]))) if reference.free.any() else 0.
            result['devices'][device]=dict(info=info,independent=independent,head_max_difference_m=error)
            if not independent['independent_accepted'] or error>1e-7:raise AssertionError(result)
        native,mi,mf=run_reference_mf6(case=case,directory=directory/'mf6',executable=executable)
        unused,same_head=reference.evaluate(head=native)
        np.testing.assert_allclose(mf['flux_native'],same_head['flux'],rtol=3e-12,atol=2e-11)
        result['mf6']=mi
        result['mf6_head_max_difference_m']=float(np.max(np.abs((native-head)[reference.free])))
        if not mi['independent_conservation'] or result['mf6_head_max_difference_m']>1e-7:raise AssertionError(result)
        supplied=case.values.copy()
        supplied[reference.free]+=.1*np.random.default_rng(seed=seed).normal(size=int(reference.free.sum()))
        unused,si,sf=run_reference_mf6(case=case,directory=directory/'mf6_supplied',executable=executable,prescribed_state=supplied)
        unused,expected=reference.evaluate(head=supplied)
        np.testing.assert_allclose(sf['flux_native'],expected['flux'],rtol=3e-12,atol=2e-11)
        np.testing.assert_allclose(sf['native_residual'],expected['residual'],rtol=3e-12,atol=2e-11)
        result['same_head_native_flux_max_error']=float(np.max(np.abs(sf['flux_native']-expected['flux'])))
        result['analytical_head_max_error_m']=float(np.max(np.abs((head-case.values)[reference.free]))) if variant not in ('heterogeneous','masked') else None
        if result['analytical_head_max_error_m'] is not None and result['analytical_head_max_error_m']>1e-7:raise AssertionError(result)
        atomic_json(path=directory/'comparison.json',value=result)
        rows.append(result)
        print(variant,'CPU/CUDA/MF6 passed',flush=True)
    atomic_json(path=root/'gate.json',value=dict(stage=1,passed=True,cases=rows,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        source_sha256={name:hashlib.sha256(Path(name).read_bytes()).hexdigest() for name in
            ('DARCY_WARP_PACKAGE/validation/npf662_reference.py','DARCY_WARP_PACKAGE/validation/npf662_mf6.py',
             'DARCY_WARP_PACKAGE/solvers/npf662_3d.py','DARCY_WARP_PACKAGE/warped_darcy_3d.py')}))


if __name__ == "__main__":
    root=Path('validation_artifacts/sevenpoint_v1/stage1/final_experiments')
    variants=['constant','x','y','z','layered','heterogeneous','masked','disconnected']
    shape=(3,5,7)
    spacing=(2.,3.,4.)
    end_heads=(8.,9.)
    contrast=(-3.,3.)
    source=.01234567891234567
    seed=1966
    layer_kz=(2.,.0001,8.)
    devices=['cpu','cuda:0']
    executable=Path('/bin/modflow/mf6')
    run_confined_matrix(root=root,variants=variants,shape=shape,spacing=spacing,end_heads=end_heads,
        contrast=contrast,source=source,seed=seed,layer_kz=layer_kz,devices=devices,executable=executable)
