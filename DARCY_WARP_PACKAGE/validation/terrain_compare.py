"""Offline physical diagnostics from saved canonical inputs and head fields."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference,CPUControls,accepted
from DARCY_WARP_PACKAGE.physics.terrain_shanz_diagnostics import water_table_profile


def evaluate_saved_head(*,case,head,output=None):
    if head.shape!=case.shape or not np.all(np.isfinite(head)):
        return dict(independent_accepted=False,status='invalid_or_sentinel_head'),None
    ref=CPUReference(case=case)
    op,r,budget=ref.evaluate(head=head)
    qx,qy,qz,contacts=op.fluxes(head=head)
    g=ref.grid
    sat=np.clip(head-g.bottoms,0,g.thickness)
    floor=head-g.bottoms<=np.minimum(case.physical['floor'],g.thickness)
    wt,kind,depth=water_table_profile(head=head,z_interfaces=case.arrays['interfaces'],surface=case.arrays['interfaces'][0])
    fields=dict(head=head,residual=r,saturated_thickness=sat,floor=floor,dry=sat==0,wet=sat>0,
        qx=qx,qy=qy,qz=qz,contacts=contacts,gates=contacts>0,water_table=wt,water_table_kind=kind,
        water_table_depth=depth)
    layers=[]
    for k in range(case.shape[0]):
        free=ref.free[k]
        layers.append(dict(layer=k,head_min=float(head[k][free].min()),head_max=float(head[k][free].max()),
            dry_cells=int(np.sum((sat[k]==0)&free)),floor_cells=int(np.sum(floor[k]&free)),
            thin_cells=int(np.sum((sat[k]>0)&(sat[k]<.1)&free)),residual_l1=float(np.sum(np.abs(r[k]))),
            residual_squared=float(np.sum(r[k]**2)),saturated_volume=float(np.sum(sat[k])*g.dx*g.dy)))
    leakage=[dict(net=float(q.sum()),downward=float(np.maximum(q,0).sum()),upward=float(np.maximum(-q,0).sum()),
        gross=float(np.abs(q).sum())) for q in qz]
    free=ref.free
    coastal_in=coastal_out=0.
    for axis,q in zip((2,1,0),(qx,qy,qz)):
        l,u=[slice(None)]*3,[slice(None)]*3
        l[axis],u[axis]=slice(None,-1),slice(1,None)
        l,u=tuple(l),tuple(u)
        exchange=np.concatenate([q[free[l]&~free[u]],-q[free[u]&~free[l]]])
        coastal_in+=float(np.maximum(-exchange,0).sum())
        coastal_out+=float(np.maximum(exchange,0).sum())
    order=np.argsort(np.abs(r).ravel())[-10:][::-1]
    worst=[dict(cell=[int(i) for i in np.unravel_index(index,head.shape)],residual=float(r.ravel()[index]),
        head_minus_bottom=float((head-g.bottoms).ravel()[index]),floor=bool(floor.ravel()[index])) for index in order]
    result=dict(independent_accepted=accepted(head=head,budget=budget,controls=CPUControls()),budget=budget,
        residual_l1=float(np.sum(np.abs(r))),layers=layers,leakage=leakage,
        coastal_free_to_fixed_inflow=coastal_in,coastal_free_to_fixed_outflow=coastal_out,
        contacts=contacts.tolist(),gate_states=(contacts>0).tolist(),worst_cells=worst,
        water_table_class_counts={key:int(np.sum(kind==key)) for key in np.unique(kind)},
        interpretation='DarcyWarp equations evaluated offline; not native MF6 residual for MF6 heads')
    if output is not None:
        np.savez_compressed(output,**fields)
    return result,fields


def compare_fields(*,case,left,right):
    free=case.grid().active.astype(bool)&(case.arrays['fixed']==0)
    difference=left['head']-right['head']
    wet_l,wet_r=left['wet'],right['wet']
    regimes=dict(wet_both=free&wet_l&wet_r,dry_both=free&~wet_l&~wet_r,
        wet_dry_mismatch=free&(wet_l!=wet_r),thin_either=free&((left['saturated_thickness']<.1)|(right['saturated_thickness']<.1)))
    result=dict(head_max=float(np.max(np.abs(difference[free]))),head_rms=float(np.sqrt(np.mean(difference[free]**2))),
        floor_switches=int(np.sum((left['floor']!=right['floor'])&free)),
        saturated_thickness_max=float(np.max(np.abs(left['saturated_thickness']-right['saturated_thickness']))),
        regimes={key:dict(count=int(mask.sum()),rms=float(np.sqrt(np.mean(difference[mask]**2))) if mask.any() else None,
            maximum=float(np.max(np.abs(difference[mask]))) if mask.any() else None) for key,mask in regimes.items()})
    for name in ('qx','qy','qz','contacts'):
        diff=left[name]-right[name]
        result[name]=dict(max_abs=float(np.max(np.abs(diff))) if diff.size else 0.,
            l1=float(np.sum(np.abs(diff))),net_difference=float(np.sum(diff)),
            relative_l1=float(np.sum(np.abs(diff))/max(np.sum(np.abs(right[name])),np.finfo(float).tiny)),
            denominator='L1 of reference flux; tiny denominator uses float64.tiny')
    wtmask=(left['water_table_kind']=='water_table')&(right['water_table_kind']=='water_table')
    dwt=left['water_table']-right['water_table']
    result['water_table']=dict(shared_phreatic_columns=int(wtmask.sum()),
        classification_mismatches=int(np.sum(left['water_table_kind']!=right['water_table_kind'])),
        max_abs=float(np.max(np.abs(dwt[wtmask]))) if wtmask.any() else None,
        rms=float(np.sqrt(np.mean(dwt[wtmask]**2))) if wtmask.any() else None)
    result['layerwise_head_rms']=[float(np.sqrt(np.mean(difference[k][free[k]]**2))) for k in range(case.shape[0])]
    return result


def native_flux_fields(*,case,extracted,darcy_fields):
    """Map native stratigraphic intercell flows; surfaces differ from terrain.

    Heads share physical centre locations. Interlayer totals compare transfer
    between corresponding geological units, not Cartesian vertical discharge.
    """
    result=darcy_fields.copy()
    for key in ('qx','qy','qz'):
        result[key]=extracted[key]
    return result
