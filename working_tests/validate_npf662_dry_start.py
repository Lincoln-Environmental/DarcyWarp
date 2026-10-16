"""Discriminating zero-derivative dry-start experiment, not changed physics."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
from validate_npf662_confined import make_gpu_model


def make_dry_case(*,initial,source):
    shape=(1,1,2);one=np.ones(shape)
    fixed=np.array([0,1]).reshape(shape);values=np.zeros(shape);q=np.zeros(shape);q.ravel()[0]=source
    head=np.full(shape,initial);head[fixed!=0]=0.
    return NPF662Case(kx=one,ky=.6*one,kz=.15*one,active=one.astype(int),fixed=fixed,values=values,
        source=q,initial=head,dx=1.,dy=1.,dz=4.,top=4.,cell_type=1)


def run_dry_start(*,root,initial,source,max_outer,krylov_budget,devices,executable):
    root.mkdir(parents=True,exist_ok=False)
    case=make_dry_case(initial=initial,source=source)
    np.savez_compressed(root/'inputs.npz',**{n:np.asarray(getattr(case,n)) for n in case.__dataclass_fields__})
    reference=NPF662Reference(case=case)
    h,result,fields=reference.solve(max_outer=max_outer)
    np.savez_compressed(root/'scipy_cpu.npz',head=h,**fields)
    rows={'scipy_cpu':result}
    for device in devices:
        with make_gpu_model(case=case,device=device) as model:
            h,info=model.solve(unconfined=True,top_elevation=case.top,max_outer=max_outer,max_cycles=krylov_budget)
        np.savez_compressed(root/('warp_'+device.replace(':','_')+'.npz'),head=h)
        rows['warp_'+device]=info
    h,result,fields=run_reference_mf6(case=case,directory=root/'mf6',executable=executable)
    rows['mf6']=result
    atomic_json(path=root/'comparison.json',value=finite_diagnostics(value=rows))
    print({key:value.get('converged',value.get('independent_accepted',value.get('native_success'))) for key,value in rows.items()})


if __name__ == "__main__":
    root=Path('validation_artifacts/sevenpoint_v1/stage2/dry_initial')
    initial=-.2
    source=1.
    max_outer=3
    krylov_budget=24
    devices=['cpu','cuda:0']
    executable=Path('/bin/modflow/mf6')
    run_dry_start(root=root,initial=initial,source=source,max_outer=max_outer,krylov_budget=krylov_budget,devices=devices,executable=executable)
