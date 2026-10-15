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
    water_table_layer=np.full(head.shape[1:],-1,dtype=np.int32)
    for k in range(head.shape[0]):
        selected=(water_table_layer<0)&(kind=='water_table')&(head[k]>g.bottoms[k])
        water_table_layer[selected]=k
    fields=dict(water_table_layer=water_table_layer,head=head,residual=r,saturated_thickness=sat,floor=floor,dry=sat==0,wet=sat>0,
        qx=qx,qy=qy,qz=qz,contacts=contacts,gates=contacts>0,water_table=wt,water_table_kind=np.asarray(kind,dtype=str),
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
        wet_dry_mismatch=free&(wet_l!=wet_r),thin_either=free&(((left['saturated_thickness']>0)&(left['saturated_thickness']<.1))|((right['saturated_thickness']>0)&(right['saturated_thickness']<.1))))
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
    result['water_table']=dict(selected_layer_changes=int(np.sum(left['water_table_layer']!=right['water_table_layer'])),shared_phreatic_columns=int(wtmask.sum()),
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
    if 'saturation' in extracted:
        native_sat=extracted['saturation']
        result['native_saturation']=native_sat
        result['head_implied_wet']=result['wet']
        result['wet']=native_sat>0
        result['dry']=native_sat==0
        result['saturated_thickness']=native_sat*case.grid().thickness
        incompatible=np.any((native_sat>0)&(~result['head_implied_wet']),axis=0)
        result['water_table_kind']=np.asarray(result['water_table_kind'],dtype='U32').copy()
        result['water_table_kind'][incompatible]='native_confined_control'
        result['water_table']=result['water_table'].copy()
        result['water_table'][incompatible]=np.nan
        result['water_table_depth']=result['water_table_depth'].copy()
        result['water_table_depth'][incompatible]=np.nan
        result['water_table_layer']=result['water_table_layer'].copy()
        result['water_table_layer'][incompatible]=-1
    return result


def native_saved_balance(*,case,fields,contact_fluxes,return_fields=False):
    """Rebuild each native internal face once; conserve free cells independently."""
    ref=CPUReference(case=case)
    shape=case.shape
    divergence=np.zeros(shape)
    for axis,name in ((2,'qx'),(1,'qy'),(0,'qz')):
        l,u=[slice(None)]*3,[slice(None)]*3
        l[axis],u[axis]=slice(None,-1),slice(1,None)
        l,u=tuple(l),tuple(u)
        divergence[l]+=fields[name];divergence[u]-=fields[name]
    for c,q in zip(case.contacts,contact_fluxes):divergence[c.cell]+=q
    residual=np.where(ref.free,fields['source_realised']-divergence,0.)
    inflow=outflow=0.
    for axis,name in ((2,'qx'),(1,'qy'),(0,'qz')):
        l,u=[slice(None)]*3,[slice(None)]*3
        l[axis],u[axis]=slice(None,-1),slice(1,None)
        l,u=tuple(l),tuple(u)
        q=fields[name]
        exterior=np.concatenate([q[ref.free[l]&~ref.free[u]],-q[ref.free[u]&~ref.free[l]]])
        inflow+=float(np.maximum(-exterior,0.).sum());outflow+=float(np.maximum(exterior,0.).sum())
    throughput=float(np.abs(fields['source_realised'][ref.free]).sum())+inflow+outflow+float(np.abs(contact_fluxes).sum())
    rms=float(np.sqrt(np.mean(residual[ref.free]**2)))
    l1=float(np.abs(residual).sum())
    relative=l1/max(throughput,np.finfo(float).tiny)
    result=dict(cell_residual_rms=rms,cell_residual_l1=l1,cell_residual_max=float(np.abs(residual).max()),
        gross_external_throughput=throughput,signed_imbalance=float(residual.sum()),relative_cell_imbalance=relative,
        coastal_inflow=inflow,coastal_outflow=outflow,contacts=np.asarray(contact_fluxes).tolist(),
        source_error_max=float(np.max(np.abs(fields['source_realised']-np.where(ref.free,case.arrays['source'],0.)))),
        conservation=bool(rms<=1e-12 or relative<=1e-8))

    return (result,residual) if return_fields else result
