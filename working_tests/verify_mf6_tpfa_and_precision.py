"""Independent MF6 TPFA nonlinear reevaluation and extended-precision root checks."""
from pathlib import Path
import json
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference
from DARCY_WARP_PACKAGE.validation.mf6_tpfa_reference import tpfa_fluxes
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D


def longdouble_residual(*,case,head):
    ref=CPUReference(case=case)
    h=head.astype(np.longdouble)
    a=case.arrays
    op=TerrainReferenceOperator3D(metrics=ref.metrics,kx=a['kx'].astype(np.longdouble),ky=a['ky'].astype(np.longdouble),
        kz=a['kz'].astype(np.longdouble),head=h,min_sat=case.physical['floor'],contacts=case.contacts,trusted_k=True)
    hx=(ref.metrics.gx@h.ravel()).reshape(h.shape)
    hy=(ref.metrics.gy@h.ravel()).reshape(h.shape)
    hz=(ref.metrics.gz@h.ravel()).reshape(h.shape)
    cx,cy,cz=op.coefficients
    qx=cx[0]*(h[:,:,:-1]-h[:,:,1:])-cx[1]*hz[:,:,:-1]-cx[2]*hz[:,:,1:]
    qy=cy[0]*(h[:,:-1]-h[:,1:])-cy[1]*hz[:,:-1]-cy[2]*hz[:,1:]
    qz=cz[0]*(h[:-1]-h[1:])-cz[1]*hx[:-1]-cz[2]*hx[1:]-cz[3]*hy[:-1]-cz[4]*hy[1:]-cz[5]*hz[:-1]-cz[6]*hz[1:]
    divergence=np.zeros_like(h)
    divergence[:,:,:-1]+=qx;divergence[:,:,1:]-=qx
    divergence[:,:-1]+=qy;divergence[:,1:]-=qy
    divergence[:-1]+=qz;divergence[1:]-=qz
    for record in op.contact_records:
        cell,c,bz,bx,by,hb,discharge=record
        q=c*(h.ravel()[cell]-hb)-bz*hz.ravel()[cell]-bx*hx.ravel()[cell]-by*hy.ravel()[cell]
        divergence.ravel()[cell]+=max(q,0) if discharge else q
    residual=np.where(ref.free,a['source'].astype(np.longdouble)-divergence,0.)
    return dict(mantissa_bits=int(np.finfo(np.longdouble).nmant),physical_rms=float(np.sqrt(np.mean(residual[ref.free]**2))),
        physical_max=float(np.max(np.abs(residual))),residual_l1=float(np.sum(np.abs(residual))))


def verify_tpfa(*,case,directory):
    last=sorted(directory.glob('feedback_*/extracted.npz'))[-1]
    data=np.load(last)
    head=data['head']
    calculated=tpfa_fluxes(case=case,head=head)
    parity={key:float(np.max(np.abs(q-data[key]))) for key,q in zip(('qx','qy','qz'),calculated)}
    ref=CPUReference(case=case)
    op,unused,unused=ref.evaluate(head=head)
    contacts=op.fluxes(head=head)[3]
    residual=np.where(ref.free,case.arrays['source']-op.divergence(fluxes=(*calculated,contacts)),0.)
    denominator=2*np.abs(case.arrays['source'][ref.free]).sum()
    result=dict(flux_difference_max=parity,physical_rms=float(np.sqrt(np.mean(residual[ref.free]**2))),
        physical_max=float(np.max(np.abs(residual))),relative_L1_source_scale=float(np.sum(np.abs(residual))/max(denominator,np.finfo(float).tiny)),
        fully_recomputed=True,method='source-derived MF66.6.2 TPFA upstream/sQuadratic saturation plus exact independently evaluated terrain outlets')
    # This verifies equations at a supplied head without running MF6 or advancing it.
    if max(parity.values())>2e-7:raise AssertionError(result)
    return result


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    rows={}
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        row=dict(tpfa=verify_tpfa(case=case,directory=root/'mf6_feedback_v3'/case.case_id),precision={})
        for path in (root/'branches'/case.case_id).glob('*/head.npz'):
            if path.parent.name in ('picard_perturb_0','newton_perturb_0'):
                row['precision'][path.parent.name]=longdouble_residual(case=case,head=np.load(path)['head'])
        cold=root/'cpu_hybrid'/case.case_id/'newton'/'head.npz'
        if cold.exists():row['precision']['cpu_cold']=longdouble_residual(case=case,head=np.load(cold)['head'])
        rows[case.case_id]=row
    atomic_json(path=root/'mf6_tpfa_and_precision.json',value=rows)
