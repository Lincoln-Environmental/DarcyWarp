"""Independent final-head terrain flux and exact stored-volume balance."""
import numpy as np


def compute_terrain_budget_3d(*, operator, head, source, fixed, storage_change,
                             storage_diagonal, return_fields=False):
    free = operator.active & ~np.asarray(fixed, dtype=bool)
    fluxes = operator.fluxes(head=head)
    divergence = operator.divergence(fluxes=fluxes)
    residual = np.where(free, source-divergence-storage_change, 0)
    inflow, outflow = 0.0, 0.0
    for axis, flux in zip((2,1,0),fluxes[:3]):
        left,right = [slice(None)]*3,[slice(None)]*3
        left[axis],right[axis] = slice(None,-1),slice(1,None)
        left,right = tuple(left),tuple(right)
        boundary = free[left] & (np.asarray(fixed)[right] != 0)
        reverse = free[right] & (np.asarray(fixed)[left] != 0)
        exchange = np.concatenate((flux[boundary],-flux[reverse]))
        inflow += float(np.sum(np.maximum(-exchange,0)))
        outflow += float(np.sum(np.maximum(exchange,0)))
    for q,record in zip(fluxes[3],operator.contact_records):
        if free.ravel()[record[0]]:
            inflow += max(-float(q),0)
            outflow += max(float(q),0)
    uptake = float(np.sum(storage_change[free]))
    throughput = float(np.sum(np.abs(source[free])) + inflow + outflow + np.sum(np.abs(storage_change[free])))
    denominator = max(throughput,np.finfo(np.float64).tiny)
    imbalance = float(np.sum(residual))
    diagonal = sum(operator.principal_faces()) + storage_diagonal
    for index,record in enumerate(operator.contact_records):
        if not record[6] or fluxes[3][index] > 0:
            diagonal.ravel()[record[0]] += record[1]
    equivalent = np.divide(residual,diagonal,out=np.zeros_like(residual),where=diagonal > 0)
    equivalent[(diagonal <= 0) & (residual != 0)] = np.inf
    result = dict(cell_residual_max=float(np.max(np.abs(residual[free]))) if free.any() else 0,
        cell_residual_rms=float(np.sqrt(np.mean(residual[free]**2))) if free.any() else 0,
        head_residual_rms=float(np.sqrt(np.mean(equivalent[free]**2))) if free.any() else 0,
        source_total=float(np.sum(source[free])),boundary_inflow=inflow,boundary_outflow=-outflow,boundary_net=inflow-outflow,
        storage_uptake=uptake,imbalance=imbalance,domain_imbalance=imbalance,relative_imbalance=abs(imbalance)/denominator,
        relative_cell_imbalance=float(np.sum(np.abs(residual)))/denominator,
        throughput=throughput,physical_diagonal_scaling="seven-point principal part",
        coastal_subface_outflows=fluxes[3].tolist())
    if return_fields:
        result.update(cell_residual_field=residual,storage_change_field=storage_change,
                      face_fluxes=fluxes)
    return result


def terrain_dry_diagnostics_3d(*, operator, head, source, fixed, budget):
    """Final-head constitutive sensitivity, not evidence of Richards accuracy."""
    from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
    free = operator.active & (np.asarray(fixed) == 0)
    true = TerrainReferenceOperator3D(metrics=operator.metrics,kx=operator.kx,
        ky=operator.ky,kz=operator.kz,head=head,min_sat=0,contacts=operator.contacts)
    actual_flux, true_flux = operator.fluxes(head=head),true.fluxes(head=head)
    excess = [0.0,0.0,0.0]
    dry = operator.active & (np.broadcast_to(operator.true_sat,operator.shape) == 0)
    dry_vertical = 0.0
    for index,axis in enumerate((2,1,0)):
        left,right = [slice(None)]*3,[slice(None)]*3
        left[axis],right[axis] = slice(None,-1),slice(1,None)
        left,right = tuple(left),tuple(right)
        valid = operator.active[left]&operator.active[right]&(free[left]|free[right])
        excess[index] = float(np.sum(np.abs(actual_flux[index]-true_flux[index])[valid]))
        if axis == 0:
            dry_vertical = float(np.sum(np.abs(actual_flux[index])[valid&(dry[left]|dry[right])]))
    horizontal = excess[0]+excess[1]
    recharge = float(np.sum(np.maximum(source[free],0)))
    abstraction = float(np.sum(np.maximum(-source[free],0)))
    return dict(horizontal_floor_excess_flow=horizontal,vertical_floor_excess_flow=excess[2],
        vertical_dry_interface_flow=dry_vertical,n_dry_cells=int(np.sum(dry & free)),
        horizontal_floor_to_recharge=horizontal/recharge if recharge > 0 else None,
        horizontal_floor_to_abstraction=horizontal/abstraction if abstraction > 0 else None,
        horizontal_floor_to_budget_throughput=horizontal/budget["throughput"] if budget["throughput"] > 0 else None,
        vertical_floor_to_recharge=excess[2]/recharge if recharge > 0 else None,
        vertical_floor_to_abstraction=excess[2]/abstraction if abstraction > 0 else None,
        vertical_floor_to_budget_throughput=excess[2]/budget["throughput"] if budget["throughput"] > 0 else None,
        vertical_dry_to_budget_throughput=dry_vertical/budget["throughput"] if budget["throughput"] > 0 else None,
        interpretation="same-head floor sensitivity; vertical means terrain-layer-normal flux, includes metric projections; physical Kz remains saturation-independent, not unsaturated flow")
