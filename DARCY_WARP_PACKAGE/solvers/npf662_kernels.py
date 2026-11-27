"""FP64 endpoint-only physical NPF662 faces, derivatives and safeguarded updates."""
import warp as wp


NPFRowMetrics = wp.types.vector(length=9, dtype=wp.float64)


@wp.func
def saturation_pair(head:wp.float64,bottom:wp.float64,dz:wp.float64,cell_type:int):
    if cell_type==0:
        return wp.vec2d(wp.float64(1.),wp.float64(0.))
    t=wp.clamp((head-bottom)/dz,wp.float64(0.),wp.float64(1.))
    eps=wp.float64(1e-6)
    a=wp.float64(1.)/(wp.float64(1.)-eps)
    sat=wp.float64(1.);derivative=wp.float64(0.)
    if t<eps:
        sat=a*wp.float64(.5)*t*t/eps
        derivative=a*t/eps/dz
    elif t<wp.float64(1.)-eps:
        sat=a*t+wp.float64(.5)*(wp.float64(1.)-a)
        derivative=a/dz
    elif t<wp.float64(1.):
        sat=wp.float64(1.)-a*wp.float64(.5)*(wp.float64(1.)-t)*(wp.float64(1.)-t)/eps
        derivative=a*(wp.float64(1.)-t)/eps/dz
    return wp.vec2d(sat,derivative)


@wp.kernel
def physical_faces(head:wp.array(dtype=wp.float64),bottom:wp.array(dtype=wp.float64),
    types:wp.array(dtype=wp.int32),right:wp.array2d(dtype=wp.int32),c0:wp.array2d(dtype=wp.float64),
    flux:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    dl:wp.array2d(dtype=wp.float64),dr:wp.array2d(dtype=wp.float64),thickness:wp.array(dtype=wp.float64),linearize:int):
    i,axis=wp.tid()
    j=right[i,axis]
    C=c0[i,axis]
    left_derivative=wp.float64(0.);right_derivative=wp.float64(0.);q=wp.float64(0.)
    if j>=0 and C>wp.float64(0.):
        donor=j
        if head[i]>head[j] or (head[i]==head[j] and axis==1):donor=i
        variable=axis!=2 and (types[i]!=0 or types[j]!=0)
        extra=wp.float64(0.)
        if variable:
            pair=saturation_pair(head[donor],bottom[donor],thickness[donor],types[donor])
            extra=C*pair[1]*(head[i]-head[j])
            C=C*pair[0]
        q=C*(head[i]-head[j])
        left_derivative=C;right_derivative=-C
        if donor==i:left_derivative+=extra
        else:right_derivative+=extra
    flux[i,axis]=q
    conductance[i,axis]=C
    if linearize!=0:
        dl[i,axis]=left_derivative;dr[i,axis]=right_derivative


@wp.kernel
def storage_terms(head:wp.array(dtype=wp.float64),previous:wp.array(dtype=wp.float64),
    bottom:wp.array(dtype=wp.float64),thickness:wp.array(dtype=wp.float64),types:wp.array(dtype=wp.int32),
    ss:wp.array(dtype=wp.float64),sy:wp.array(dtype=wp.float64),free:wp.array(dtype=wp.int32),
    rate:wp.array(dtype=wp.float64),tangent:wp.array(dtype=wp.float64),factor:wp.float64):
    i=wp.tid()
    rate[i]=wp.float64(0.);tangent[i]=wp.float64(0.)
    if free[i]==0:return
    pair=saturation_pair(head[i],bottom[i],thickness[i],types[i])
    old=saturation_pair(previous[i],bottom[i],thickness[i],types[i])
    change=pair[0]-old[0]
    yield_coefficient=wp.float64(0.)
    if types[i]!=0:yield_coefficient=sy[i]
    elastic=ss[i]*thickness[i]*(pair[0]*(head[i]-previous[i])+
        change*(previous[i]-bottom[i]-wp.float64(.5)*thickness[i]*(pair[0]+old[0])))
    rate[i]=factor*(elastic+yield_coefficient*thickness[i]*change)
    tangent[i]=factor*(ss[i]*thickness[i]*(pair[0]+pair[1]*(head[i]-bottom[i]-thickness[i]*pair[0]))+
        yield_coefficient*thickness[i]*pair[1])


@wp.kernel
def roundoff_rows(head:wp.array(dtype=wp.float64), source:wp.array(dtype=wp.float64),
    residual:wp.array(dtype=wp.float64), free:wp.array(dtype=wp.int32),
    left:wp.array2d(dtype=wp.int32), right:wp.array2d(dtype=wp.int32),
    conductance:wp.array2d(dtype=wp.float64), ghb_conductance:wp.array(dtype=wp.float64),
    ghb_head:wp.array(dtype=wp.float64), storage:wp.array(dtype=wp.float64),
    tangent:wp.array(dtype=wp.float64), previous:wp.array(dtype=wp.float64),
    previous_tangent:wp.array(dtype=wp.float64), types:wp.array(dtype=wp.int32),
    bottom:wp.array(dtype=wp.float64), thickness:wp.array(dtype=wp.float64),
    partials:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if free[i]==0:return
    scale=wp.abs(source[i])+wp.abs(storage[i])
    coordinates=wp.abs(head[i])+wp.abs(previous[i])
    # Convertible STO subtracts layer-referenced saturations. Small absolute
    # heads do not imply small operands in that subtraction.
    if types[i]!=0:coordinates+=wp.float64(2.)*(wp.abs(bottom[i])+thickness[i])
    scale+=(wp.abs(tangent[i])+wp.abs(previous_tangent[i]))*coordinates
    for axis in range(3):
        j=right[i,axis]
        if j>=0:scale+=conductance[i,axis]*(wp.abs(head[i])+wp.abs(head[j]))
        j=left[i,axis]
        if j>=0:scale+=conductance[j,axis]*(wp.abs(head[i])+wp.abs(head[j]))
    if ghb_conductance[i]>wp.float64(0.):
        scale+=ghb_conductance[i]*(wp.abs(head[i])+wp.abs(ghb_head[i]))
    allowance=wp.float64(7.105427357601002e-15)*scale  # 32 * FP64 eps
    ratio=wp.float64(0.)
    if not wp.isfinite(allowance) or allowance<wp.float64(0.) or not wp.isfinite(residual[i]):
        ratio=wp.float64(wp.inf)
    elif allowance>wp.float64(0.):ratio=wp.abs(residual[i])/allowance
    elif residual[i]!=wp.float64(0.):ratio=wp.float64(wp.inf)
    wp.atomic_max(partials,i/256,ratio)


@wp.kernel
def roundoff_rows_drn_riv(head:wp.array(dtype=wp.float64), source:wp.array(dtype=wp.float64),
    residual:wp.array(dtype=wp.float64), free:wp.array(dtype=wp.int32),
    left:wp.array2d(dtype=wp.int32), right:wp.array2d(dtype=wp.int32),
    conductance:wp.array2d(dtype=wp.float64), ghb_conductance:wp.array(dtype=wp.float64),
    ghb_head:wp.array(dtype=wp.float64), storage:wp.array(dtype=wp.float64),
    tangent:wp.array(dtype=wp.float64), previous:wp.array(dtype=wp.float64),
    previous_tangent:wp.array(dtype=wp.float64), types:wp.array(dtype=wp.int32),
    bottom:wp.array(dtype=wp.float64), thickness:wp.array(dtype=wp.float64),
    drn_offsets:wp.array(dtype=wp.int32), drn_c:wp.array(dtype=wp.float64),
    drn_elevation:wp.array(dtype=wp.float64),
    riv_offsets:wp.array(dtype=wp.int32), riv_c:wp.array(dtype=wp.float64),
    riv_stage:wp.array(dtype=wp.float64), riv_bottom:wp.array(dtype=wp.float64),
    partials:wp.array(dtype=wp.float64)):
    """Reduce equation-row rounding bounds including every CSR package entry.

    :param head: Current groundwater heads.
    :param source: Cell source rates.
    :param residual: Independently evaluated physical residuals.
    :param free: Free-cell indicators.
    :param left: Neighbor indices on negative coordinate faces.
    :param right: Neighbor indices on positive coordinate faces.
    :param conductance: Integrated intercell conductances.
    :param ghb_conductance: Integrated GHB conductances.
    :param ghb_head: GHB reference heads.
    :param storage: Physical storage residual contribution.
    :param tangent: Current storage tangent.
    :param previous: Previous accepted heads.
    :param previous_tangent: Previous-state restraint diagonal.
    :param types: Cell formulation codes.
    :param bottom: Cell bottoms.
    :param thickness: Cell thicknesses.
    :param drn_offsets: CSR drain entry offsets per cell.
    :param drn_c: Integrated drain entry conductances.
    :param drn_elevation: Drain entry elevations.
    :param riv_offsets: CSR river entry offsets per cell.
    :param riv_c: Integrated river entry conductances.
    :param riv_stage: River entry stages.
    :param riv_bottom: River entry bed bottoms.
    :param partials: Per-block maximum scaled row errors, updated in place.
    :return: None; this Warp kernel updates ``partials`` in place.
    """
    i=wp.tid()
    if free[i]==0:return
    scale=wp.abs(source[i])+wp.abs(storage[i])
    coordinates=wp.abs(head[i])+wp.abs(previous[i])
    if types[i]!=0:coordinates+=wp.float64(2.)*(wp.abs(bottom[i])+thickness[i])
    scale+=(wp.abs(tangent[i])+wp.abs(previous_tangent[i]))*coordinates
    for axis in range(3):
        j=right[i,axis]
        if j>=0:scale+=conductance[i,axis]*(wp.abs(head[i])+wp.abs(head[j]))
        j=left[i,axis]
        if j>=0:scale+=conductance[j,axis]*(wp.abs(head[i])+wp.abs(head[j]))
    if ghb_conductance[i]>wp.float64(0.):
        scale+=ghb_conductance[i]*(wp.abs(head[i])+wp.abs(ghb_head[i]))
    for entry in range(drn_offsets[i],drn_offsets[i+1]):
        if drn_c[entry]>wp.float64(0.) and head[i]>drn_elevation[entry]:
            scale+=drn_c[entry]*(wp.abs(head[i])+wp.abs(drn_elevation[entry]))
    for entry in range(riv_offsets[i],riv_offsets[i+1]):
        if riv_c[entry]>wp.float64(0.):
            reference_head=wp.max(head[i],riv_bottom[entry])
            scale+=riv_c[entry]*(wp.abs(reference_head)+wp.abs(riv_stage[entry]))
    allowance=wp.float64(7.105427357601002e-15)*scale
    ratio=wp.float64(0.)
    if not wp.isfinite(allowance) or allowance<wp.float64(0.) or not wp.isfinite(residual[i]):
        ratio=wp.float64(wp.inf)
    elif allowance>wp.float64(0.):ratio=wp.abs(residual[i])/allowance
    elif residual[i]!=wp.float64(0.):ratio=wp.float64(wp.inf)
    wp.atomic_max(partials,i/256,ratio)


@wp.kernel
def combine_roundoff(partials:wp.array(dtype=wp.float64), maximum:wp.array(dtype=wp.float64)):
    i=wp.tid()
    wp.atomic_max(maximum,0,partials[i])


@wp.func
def physical_row_metrics_base(i:int,head:wp.array(dtype=wp.float64),source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),fixed:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),
    right:wp.array2d(dtype=wp.int32),flux:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64),residual:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),
    storage:wp.array(dtype=wp.float64),storage_tangent:wp.array(dtype=wp.float64),
    q_ghb:wp.float64,q_drn:wp.float64,q_riv:wp.float64,boundary_c:wp.float64):
    """Assemble one physical row with per-package gross exchange accounting.

    The residual uses the signed boundary sum while inflow and outflow keep
    every package contribution separate, so opposing exchanges cannot mask
    local equation errors in the throughput metrics.

    :param q_ghb: Signed GHB exchange, positive into the aquifer.
    :param q_drn: Signed DRN exchange, positive into the aquifer.
    :param q_riv: Signed RIV exchange, positive into the aquifer.
    :param boundary_c: Combined active boundary tangent for the row diagonal.
    """
    metrics = NPFRowMetrics(wp.float64(0.))
    invalid=not wp.isfinite(head[i]) or ((free[i]!=0 or fixed[i]!=0) and wp.abs(head[i])>=wp.float64(1e20)) or not wp.isfinite(scale[i]) or scale[i]<=wp.float64(0.)
    for axis in range(3):
        if not wp.isfinite(flux[i,axis]) or not wp.isfinite(conductance[i,axis]):invalid=True
    if invalid:metrics[6]=wp.float64(1.)
    if free[i]==0:
        residual[i]=wp.float64(0.);rhs[i]=wp.float64(0.)
        return metrics
    divergence=wp.float64(0.);diagonal=wp.float64(0.);inflow=wp.float64(0.);outflow=wp.float64(0.)
    for axis in range(3):
        j=right[i,axis]
        if j>=0:
            q=flux[i,axis];divergence+=q;diagonal+=conductance[i,axis]
            if fixed[j]!=0:
                inflow+=wp.max(-q,wp.float64(0.));outflow+=wp.max(q,wp.float64(0.))
        j=left[i,axis]
        if j>=0:
            q=-flux[j,axis];divergence+=q;diagonal+=conductance[j,axis]
            if fixed[j]!=0:
                inflow+=wp.max(-q,wp.float64(0.));outflow+=wp.max(q,wp.float64(0.))
    diagonal+=storage_tangent[i]+boundary_c
    r=source[i]-divergence-storage[i]-q_ghb-q_drn-q_riv
    inflow+=wp.max(-q_ghb,wp.float64(0.))+wp.max(-q_drn,wp.float64(0.))+wp.max(-q_riv,wp.float64(0.))
    outflow+=wp.max(q_ghb,wp.float64(0.))+wp.max(q_drn,wp.float64(0.))+wp.max(q_riv,wp.float64(0.))
    equivalent=wp.float64(0.)
    if diagonal>wp.float64(0.):equivalent=r/diagonal
    elif r!=wp.float64(0.):equivalent=wp.float64(wp.inf)
    residual[i]=r;rhs[i]=r/scale[i]
    metrics[0]=r*r;metrics[1]=wp.abs(r)
    metrics[2]=equivalent*equivalent
    metrics[3]=wp.abs(source[i])+inflow+outflow+wp.abs(storage[i])
    metrics[4]=inflow;metrics[5]=outflow
    if not wp.isfinite(r) or not wp.isfinite(diagonal):metrics[6]+=wp.float64(1.)
    metrics[7]=rhs[i]*rhs[i];metrics[8]=r
    return metrics


@wp.func
def physical_row_metrics(i:int,head:wp.array(dtype=wp.float64),source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),fixed:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),
    right:wp.array2d(dtype=wp.int32),flux:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64),residual:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),
    storage:wp.array(dtype=wp.float64),storage_tangent:wp.array(dtype=wp.float64)):
    return physical_row_metrics_base(i,head,source,free,fixed,left,right,flux,conductance,
        scale,residual,rhs,storage,storage_tangent,wp.float64(0.),wp.float64(0.),wp.float64(0.),wp.float64(0.))


@wp.func
def ghb_flow_at(i:int,head:wp.array(dtype=wp.float64),conductance:wp.array(dtype=wp.float64),
    boundary_head:wp.array(dtype=wp.float64)):
    value=wp.float64(0.)
    if conductance[i]>wp.float64(0.):value=conductance[i]*(head[i]-boundary_head[i])
    return value


@wp.func
def drn_flow_pair(head: wp.float64, conductance: wp.float64, elevation: wp.float64):
    """Evaluate drain discharge and its active-branch derivative.

    :param head: Groundwater head.
    :param conductance: Integrated drain conductance, in L²/T.
    :param elevation: Drain elevation.
    :return: ``(discharge, derivative)`` with discharge positive out of the aquifer.
    """
    flow = wp.float64(0.0)
    tangent = wp.float64(0.0)
    if conductance > wp.float64(0.0) and head > elevation:
        flow = conductance * (head - elevation)
        tangent = conductance
    return wp.vec2d(flow, tangent)


@wp.func
def riv_flow_pair(head: wp.float64, conductance: wp.float64, stage: wp.float64,
    bottom: wp.float64):
    """Evaluate river exchange and its active-branch derivative.

    :param head: Groundwater head.
    :param conductance: Integrated river conductance, in L²/T.
    :param stage: River stage.
    :param bottom: Riverbed bottom elevation.
    :return: ``(discharge, derivative)`` with discharge positive out of the aquifer.
    """
    flow = wp.float64(0.0)
    tangent = wp.float64(0.0)
    if conductance > wp.float64(0.0):
        reference_head = wp.max(head, bottom)
        flow = conductance * (reference_head - stage)
        if head > bottom:
            tangent = conductance
    return wp.vec2d(flow, tangent)


@wp.kernel
def check_local_balance(head: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
                        fixed: wp.array(dtype=wp.int32), source: wp.array(dtype=wp.float64),
                        left: wp.array2d(dtype=wp.int32), right: wp.array2d(dtype=wp.int32),
                        flux: wp.array2d(dtype=wp.float64), conductance: wp.array2d(dtype=wp.float64),
                        storage: wp.array(dtype=wp.float64), tangent: wp.array(dtype=wp.float64),
                        residual: wp.array(dtype=wp.float64), has_ghb: int,
                        ghb_c: wp.array(dtype=wp.float64), ghb_head: wp.array(dtype=wp.float64),
                        component_anchor: wp.array(dtype=wp.float64), has_component_anchor: int,
                        absolute_tolerance: wp.float64, relative_tolerance: wp.float64,
                        failed: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if free[i] == 0:
        return
    throughput = wp.abs(source[i]) + wp.abs(storage[i])
    anchor = tangent[i]
    roundoff = wp.float64(0.0)
    eps = wp.float64(2.220446049250313e-16)
    if has_ghb != 0:
        anchor += ghb_c[i]
        throughput += wp.abs(ghb_flow_at(i, head, ghb_c, ghb_head))
    for axis in range(3):
        for side in range(2):
            j = right[i, axis]
            owner = i
            if side == 1:
                j = left[i, axis]
                owner = j
            if j >= 0:
                c = conductance[owner, axis]
                q = flux[owner, axis]
                throughput += wp.abs(q)
                if fixed[j] != 0:
                    anchor += c
                if q != wp.float64(0.0):
                    roundoff += (eps * wp.abs(head[i]) + eps * wp.abs(head[j])) * c
    if has_component_anchor != 0 and component_anchor[i] > wp.float64(0.0):
        anchor = component_anchor[i]
    allowance = wp.max(absolute_tolerance, relative_tolerance * throughput)
    if anchor > wp.float64(0.0):
        allowance = anchor * wp.float64(1.0e-6)
    allowance += roundoff
    if not wp.isfinite(allowance) or wp.abs(residual[i]) > allowance:
        wp.atomic_max(failed, 0, wp.float64(1.0))


@wp.kernel
def physical_rows_ghb(head:wp.array(dtype=wp.float64),source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),fixed:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),
    right:wp.array2d(dtype=wp.int32),flux:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64),residual:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),
    metrics:wp.array2d(dtype=wp.float64),storage:wp.array(dtype=wp.float64),storage_tangent:wp.array(dtype=wp.float64),
    block_size:int,ghb_conductance:wp.array(dtype=wp.float64),ghb_head:wp.array(dtype=wp.float64)):
    i=wp.tid()
    q=ghb_flow_at(i,head,ghb_conductance,ghb_head)
    values=physical_row_metrics_base(i,head,source,free,fixed,left,right,flux,conductance,
        scale,residual,rhs,storage,storage_tangent,q,wp.float64(0.),wp.float64(0.),ghb_conductance[i])
    for metric in range(9):
        if values[metric]!=wp.float64(0.):wp.atomic_add(metrics,i/block_size,metric,values[metric])


@wp.func
def cell_boundary_flow(head: wp.float64, offsets: wp.array(dtype=wp.int32),
    c: wp.array(dtype=wp.float64), first_level: wp.array(dtype=wp.float64),
    second_level: wp.array(dtype=wp.float64), third_level: wp.array(dtype=wp.float64),
    package: int, cell: int, entry_rate: wp.array(dtype=wp.float64)) -> wp.vec4d:
    """Sum every CSR entry's own flux for one cell.

    :param head: Current groundwater head at the cell.
    :param offsets: CSR entry offsets of shape (n + 1).
    :param c: Integrated entry conductances.
    :param first_level: Drain elevations (unused for RIV).
    :param second_level: River stages (unused for DRN).
    :param third_level: Riverbed bottoms (unused for DRN).
    :param package: Zero for DRN, one for RIV.
    :param cell: Owning cell index.
    :param entry_rate: Per-entry discharge output, positive out of the aquifer.
    :return: Signed flow, tangent, gross inflow and gross outflow for the entries.
    """
    flow = wp.float64(0.0)
    tangent = wp.float64(0.0)
    inflow = wp.float64(0.0)
    outflow = wp.float64(0.0)
    for entry in range(offsets[cell], offsets[cell + 1]):
        if package == 0:
            pair = drn_flow_pair(head, c[entry], first_level[entry])
        else:
            pair = riv_flow_pair(head, c[entry], second_level[entry], third_level[entry])
        flow = flow + pair[0]
        tangent = tangent + pair[1]
        entry_rate[entry] = pair[0]
        inflow += wp.max(-pair[0], wp.float64(0.0))
        outflow += wp.max(pair[0], wp.float64(0.0))
    return wp.vec4d(flow, tangent, inflow, outflow)


@wp.kernel
def physical_rows_drn_riv(head:wp.array(dtype=wp.float64), source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32), fixed:wp.array(dtype=wp.int32),
    left:wp.array2d(dtype=wp.int32), right:wp.array2d(dtype=wp.int32),
    flux:wp.array2d(dtype=wp.float64), conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64), residual:wp.array(dtype=wp.float64), rhs:wp.array(dtype=wp.float64),
    metrics:wp.array2d(dtype=wp.float64), storage:wp.array(dtype=wp.float64),
    storage_tangent:wp.array(dtype=wp.float64), block_size:int,
    ghb_c:wp.array(dtype=wp.float64), ghb_head:wp.array(dtype=wp.float64),
    drn_offsets:wp.array(dtype=wp.int32), drn_c:wp.array(dtype=wp.float64),
    drn_elevation:wp.array(dtype=wp.float64),
    riv_offsets:wp.array(dtype=wp.int32), riv_c:wp.array(dtype=wp.float64),
    riv_stage:wp.array(dtype=wp.float64), riv_bottom:wp.array(dtype=wp.float64),
    drn_entry_rate:wp.array(dtype=wp.float64), riv_entry_rate:wp.array(dtype=wp.float64),
    boundary_throughput:wp.array(dtype=wp.float64),
    boundary_tangent:wp.array(dtype=wp.float64), operator_tangent:wp.array(dtype=wp.float64)):
    """Evaluate physical rows with CSR DRN/RIV entries per cell.

    Each entry's physical flux is evaluated independently before the cell
    contributions sum, so multiple reaches or drains in one cell keep the
    exact piecewise-linear package law.

    :param head: Current or trial groundwater heads.
    :param source: Cell source rates.
    :param free: Free-cell indicators.
    :param fixed: Prescribed-head indicators.
    :param left: Neighbor indices on negative coordinate faces.
    :param right: Neighbor indices on positive coordinate faces.
    :param flux: Physical intercell fluxes.
    :param conductance: Integrated intercell conductances.
    :param scale: Fixed equation row scales.
    :param residual: Physical residual output.
    :param rhs: Assembled right-hand-side values.
    :param metrics: Per-block physical solver metrics.
    :param storage: Storage residual contribution.
    :param storage_tangent: Storage tangent contribution.
    :param block_size: Number of rows per metric block.
    :param ghb_c: Integrated GHB conductances.
    :param ghb_head: GHB reference heads.
    :param drn_offsets: CSR drain entry offsets per cell.
    :param drn_c: Integrated drain entry conductances.
    :param drn_elevation: Drain entry elevations.
    :param riv_offsets: CSR river entry offsets per cell.
    :param riv_c: Integrated river entry conductances.
    :param riv_stage: River entry stages.
    :param riv_bottom: River entry bed bottoms.
    :param drn_entry_rate: Per-entry drain discharge output.
    :param riv_entry_rate: Per-entry river discharge output.
    :param boundary_throughput: Combined GHB/DRN/RIV discharge output.
    :param boundary_tangent: Combined boundary tangent output.
    :param operator_tangent: Storage plus boundary diagonal output.
    :return: None; outputs are written in place.
    """
    i = wp.tid()
    drain = cell_boundary_flow(head[i], drn_offsets, drn_c, drn_elevation, drn_elevation,
        drn_elevation, 0, i, drn_entry_rate)
    river = cell_boundary_flow(head[i], riv_offsets, riv_c, riv_stage, riv_stage,
        riv_bottom, 1, i, riv_entry_rate)
    ghb_flow = ghb_flow_at(i, head, ghb_c, ghb_head)
    combined_flow = ghb_flow + drain[0] + river[0]
    combined_tangent = ghb_c[i] + drain[1] + river[1]
    boundary_throughput[i] = wp.abs(ghb_flow) + drain[2] + drain[3] + river[2] + river[3]
    boundary_tangent[i] = combined_tangent
    operator_tangent[i] = storage_tangent[i] + combined_tangent
    values = physical_row_metrics_base(i, head, source, free, fixed, left, right, flux, conductance,
        scale, residual, rhs, storage, storage_tangent,
        ghb_flow, drain[0], river[0], combined_tangent)
    if free[i] != 0:
        extra_in = drain[2] + river[2] - wp.max(-drain[0], wp.float64(0.0)) - wp.max(-river[0], wp.float64(0.0))
        extra_out = drain[3] + river[3] - wp.max(drain[0], wp.float64(0.0)) - wp.max(river[0], wp.float64(0.0))
        values[3] += extra_in + extra_out
        values[4] += extra_in
        values[5] += extra_out
    for metric in range(9):
        if values[metric] != wp.float64(0.0):
            wp.atomic_add(metrics, i / block_size, metric, values[metric])


@wp.kernel
def physical_rows_drn_riv_tiled(head:wp.array(dtype=wp.float64), source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32), fixed:wp.array(dtype=wp.int32),
    left:wp.array2d(dtype=wp.int32), right:wp.array2d(dtype=wp.int32),
    flux:wp.array2d(dtype=wp.float64), conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64), residual:wp.array(dtype=wp.float64), rhs:wp.array(dtype=wp.float64),
    metrics:wp.array(dtype=wp.float64), storage:wp.array(dtype=wp.float64),
    storage_tangent:wp.array(dtype=wp.float64), block_size:int,
    ghb_c:wp.array(dtype=wp.float64), ghb_head:wp.array(dtype=wp.float64),
    drn_offsets:wp.array(dtype=wp.int32), drn_c:wp.array(dtype=wp.float64),
    drn_elevation:wp.array(dtype=wp.float64),
    riv_offsets:wp.array(dtype=wp.int32), riv_c:wp.array(dtype=wp.float64),
    riv_stage:wp.array(dtype=wp.float64), riv_bottom:wp.array(dtype=wp.float64),
    drn_entry_rate:wp.array(dtype=wp.float64), riv_entry_rate:wp.array(dtype=wp.float64),
    boundary_throughput:wp.array(dtype=wp.float64),
    boundary_tangent:wp.array(dtype=wp.float64), operator_tangent:wp.array(dtype=wp.float64)):
    """Tiled equivalent of :func:`physical_rows_drn_riv`.

    :param head: Current or trial groundwater heads.
    :param source: Cell source rates.
    :param free: Free-cell indicators.
    :param fixed: Prescribed-head indicators.
    :param left: Neighbor indices on negative coordinate faces.
    :param right: Neighbor indices on positive coordinate faces.
    :param flux: Physical intercell fluxes.
    :param conductance: Integrated intercell conductances.
    :param scale: Fixed equation row scales.
    :param residual: Physical residual output.
    :param rhs: Assembled right-hand-side values.
    :param metrics: Per-block physical solver metrics.
    :param storage: Storage residual contribution.
    :param storage_tangent: Storage tangent contribution.
    :param block_size: Number of rows per metric block.
    :param ghb_c: Integrated GHB conductances.
    :param ghb_head: GHB reference heads.
    :param drn_offsets: CSR drain entry offsets per cell.
    :param drn_c: Integrated drain entry conductances.
    :param drn_elevation: Drain entry elevations.
    :param riv_offsets: CSR river entry offsets per cell.
    :param riv_c: Integrated river entry conductances.
    :param riv_stage: River entry stages.
    :param riv_bottom: River entry bed bottoms.
    :param drn_entry_rate: Per-entry drain discharge output.
    :param riv_entry_rate: Per-entry river discharge output.
    :param boundary_throughput: Combined GHB/DRN/RIV discharge output.
    :param boundary_tangent: Combined boundary tangent output.
    :param operator_tangent: Storage plus boundary diagonal output.
    :return: None; outputs are written in place.
    """
    i = wp.tid()
    drain = cell_boundary_flow(head[i], drn_offsets, drn_c, drn_elevation, drn_elevation,
        drn_elevation, 0, i, drn_entry_rate)
    river = cell_boundary_flow(head[i], riv_offsets, riv_c, riv_stage, riv_stage,
        riv_bottom, 1, i, riv_entry_rate)
    ghb_flow = ghb_flow_at(i, head, ghb_c, ghb_head)
    combined_flow = ghb_flow + drain[0] + river[0]
    combined_tangent = ghb_c[i] + drain[1] + river[1]
    boundary_throughput[i] = wp.abs(ghb_flow) + drain[2] + drain[3] + river[2] + river[3]
    boundary_tangent[i] = combined_tangent
    operator_tangent[i] = storage_tangent[i] + combined_tangent
    values = physical_row_metrics_base(i, head, source, free, fixed, left, right, flux, conductance,
        scale, residual, rhs, storage, storage_tangent,
        ghb_flow, drain[0], river[0], combined_tangent)
    if free[i] != 0:
        extra_in = drain[2] + river[2] - wp.max(-drain[0], wp.float64(0.0)) - wp.max(-river[0], wp.float64(0.0))
        extra_out = drain[3] + river[3] - wp.max(drain[0], wp.float64(0.0)) - wp.max(river[0], wp.float64(0.0))
        values[3] += extra_in + extra_out
        values[4] += extra_in
        values[5] += extra_out
    for metric in range(9):
        wp.tile_store(metrics, wp.tile_sum(wp.tile(values[metric])), offset=(i / block_size) * 9 + metric)


@wp.kernel
def check_local_balance_drn_riv(head: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32), source: wp.array(dtype=wp.float64),
    left: wp.array2d(dtype=wp.int32), right: wp.array2d(dtype=wp.int32),
    flux: wp.array2d(dtype=wp.float64), conductance: wp.array2d(dtype=wp.float64),
    storage: wp.array(dtype=wp.float64), storage_tangent: wp.array(dtype=wp.float64),
    residual: wp.array(dtype=wp.float64), boundary_throughput: wp.array(dtype=wp.float64),
    boundary_tangent: wp.array(dtype=wp.float64), component_anchor: wp.array(dtype=wp.float64), absolute_tolerance: wp.float64,
    relative_tolerance: wp.float64, failed: wp.array(dtype=wp.float64)):
    """Check independently assembled cell balances including package flow.

    :param head: Accepted or trial groundwater heads.
    :param free: Free-cell indicators.
    :param fixed: Prescribed-head indicators.
    :param source: Cell source rates.
    :param left: Neighbor indices on negative coordinate faces.
    :param right: Neighbor indices on positive coordinate faces.
    :param flux: Physical intercell fluxes.
    :param conductance: Integrated intercell conductances.
    :param storage: Physical storage residual contribution.
    :param storage_tangent: Storage tangent contribution.
    :param residual: Independently evaluated physical residuals.
    :param boundary_throughput: Gross accepted DRN/RIV/GHB entry throughput.
    :param boundary_tangent: Combined active boundary tangent.
    :param absolute_tolerance: Absolute cell balance tolerance.
    :param relative_tolerance: Throughput-relative balance tolerance.
    :param failed: Scalar failure flag, updated in place.
    :return: None; ``failed`` is set when any free-cell balance fails.
    """
    i = wp.tid()
    if free[i] == 0:
        return
    throughput = wp.abs(source[i]) + wp.abs(storage[i]) + boundary_throughput[i]
    anchor = storage_tangent[i] + boundary_tangent[i]
    roundoff = wp.float64(0.0)
    eps = wp.float64(2.220446049250313e-16)
    for axis in range(3):
        for side in range(2):
            j = right[i, axis]
            owner = i
            if side == 1:
                j = left[i, axis]
                owner = j
            if j >= 0:
                c = conductance[owner, axis]
                q = flux[owner, axis]
                throughput += wp.abs(q)
                if fixed[j] != 0:
                    anchor += c
                if q != wp.float64(0.0):
                    roundoff += (eps * wp.abs(head[i]) + eps * wp.abs(head[j])) * c
    allowance = wp.max(absolute_tolerance, relative_tolerance * throughput)
    if component_anchor[i] > wp.float64(0.0):
        anchor = component_anchor[i]
    if anchor > wp.float64(0.0):
        allowance = anchor * wp.float64(1.0e-6)
    allowance += roundoff
    if not wp.isfinite(allowance) or wp.abs(residual[i]) > allowance:
        wp.atomic_max(failed, 0, wp.float64(1.0))


@wp.kernel
def sum_component_restraint(labels: wp.array(dtype=wp.int32), weights: wp.array(dtype=wp.float64),
    storage: wp.array(dtype=wp.float64), boundary: wp.array(dtype=wp.float64),
    sums: wp.array(dtype=wp.float64)):
    """Average actual external derivatives within each physical component."""
    i = wp.tid()
    wp.atomic_add(sums, labels[i], weights[i] * (storage[i] + boundary[i]))


@wp.kernel
def share_component_restraint(labels: wp.array(dtype=wp.int32), sums: wp.array(dtype=wp.float64),
    anchor: wp.array(dtype=wp.float64)):
    """Broadcast each component's common-head restraint to its local checks."""
    i = wp.tid()
    anchor[i] = sums[labels[i]]


@wp.kernel
def detect_connectivity_change(conductance: wp.array2d(dtype=wp.float64),
    right: wp.array2d(dtype=wp.int32), cached: wp.array2d(dtype=wp.int32),
    changed: wp.array(dtype=wp.int32)):
    """Update the conducting-face mask and reduce any change to one scalar."""
    i, axis = wp.tid()
    current = wp.int32(0)
    if right[i, axis] >= 0 and conductance[i, axis] > wp.float64(0.):
        current = wp.int32(1)
    if cached[i, axis] != current:
        cached[i, axis] = current
        wp.atomic_max(changed, 0, wp.int32(1))


@wp.kernel
def drn_riv_budget(head: wp.array(dtype=wp.float64),
    drn_offsets: wp.array(dtype=wp.int32), drn_c: wp.array(dtype=wp.float64),
    drn_elevation: wp.array(dtype=wp.float64),
    riv_offsets: wp.array(dtype=wp.int32), riv_c: wp.array(dtype=wp.float64),
    riv_stage: wp.array(dtype=wp.float64), riv_bottom: wp.array(dtype=wp.float64),
    drn_entry_rate: wp.array(dtype=wp.float64), riv_entry_rate: wp.array(dtype=wp.float64),
    drn_rate: wp.array(dtype=wp.float64), riv_rate: wp.array(dtype=wp.float64),
    totals: wp.array(dtype=wp.float64)):
    """Evaluate accepted entry rates, cell sums and gross exchange totals.

    Every entry keeps its own discharge so multiple reaches or drains in one
    cell retain separate physical budgets.

    :param head: Accepted groundwater heads.
    :param drn_offsets: CSR drain entry offsets per cell.
    :param drn_c: Integrated drain entry conductances.
    :param drn_elevation: Drain entry elevations.
    :param riv_offsets: CSR river entry offsets per cell.
    :param riv_c: Integrated river entry conductances.
    :param riv_stage: River entry stages.
    :param riv_bottom: River entry bed bottoms.
    :param drn_entry_rate: Per-entry drain discharge output.
    :param riv_entry_rate: Per-entry river discharge output.
    :param drn_rate: Per-cell drain discharge output, positive out of aquifer.
    :param riv_rate: Per-cell river discharge output, positive out of aquifer.
    :param totals: Gross inflow/outflow totals, accumulated in place.
    :return: None; outputs are written in place.
    """
    i = wp.tid()
    drain = cell_boundary_flow(head[i], drn_offsets, drn_c, drn_elevation, drn_elevation,
        drn_elevation, 0, i, drn_entry_rate)
    river = cell_boundary_flow(head[i], riv_offsets, riv_c, riv_stage, riv_stage,
        riv_bottom, 1, i, riv_entry_rate)
    drn_rate[i] = drain[0]
    riv_rate[i] = river[0]
    wp.atomic_add(totals, 0, drain[2])
    wp.atomic_add(totals, 1, drain[3])
    wp.atomic_add(totals, 2, river[2])
    wp.atomic_add(totals, 3, river[3])


@wp.kernel
def physical_rows_ghb_tiled(head:wp.array(dtype=wp.float64),source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),fixed:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),
    right:wp.array2d(dtype=wp.int32),flux:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64),residual:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),
    metrics:wp.array(dtype=wp.float64),storage:wp.array(dtype=wp.float64),storage_tangent:wp.array(dtype=wp.float64),
    block_size:int,ghb_conductance:wp.array(dtype=wp.float64),ghb_head:wp.array(dtype=wp.float64)):
    i=wp.tid()
    q=ghb_flow_at(i,head,ghb_conductance,ghb_head)
    values=physical_row_metrics_base(i,head,source,free,fixed,left,right,flux,conductance,
        scale,residual,rhs,storage,storage_tangent,q,wp.float64(0.),wp.float64(0.),ghb_conductance[i])
    for metric in range(9):
        wp.tile_store(metrics,wp.tile_sum(wp.tile(values[metric])),offset=(i/block_size)*9+metric)


@wp.kernel
def ghb_diagonal(storage:wp.array(dtype=wp.float64),conductance:wp.array(dtype=wp.float64),
    result:wp.array(dtype=wp.float64)):
    i=wp.tid()
    result[i]=storage[i]+conductance[i]


@wp.kernel
def previous_restraint_diagonal(storage_tangent:wp.array(dtype=wp.float64),
    activation_tangent:wp.array(dtype=wp.float64), ghb_c:wp.array(dtype=wp.float64), drn_offsets:wp.array(dtype=wp.int32),
    drn_c:wp.array(dtype=wp.float64), drn_elevation:wp.array(dtype=wp.float64),
    riv_offsets:wp.array(dtype=wp.int32), riv_c:wp.array(dtype=wp.float64),
    riv_stage:wp.array(dtype=wp.float64), riv_bottom:wp.array(dtype=wp.float64),
    previous:wp.array(dtype=wp.float64), has_ghb:int, has_entries:int,
    result:wp.array(dtype=wp.float64)):
    """Combine every previous-state restraint into one preconditioner diagonal.

    Systems constrained only by an active drain or a connected river must keep
    a nonsingular approximate preconditioner without touching the physical
    residual or Newton Jacobian.

    :param storage_tangent: Physical storage tangent at the previous state.
    :param activation_tangent: Actual storage tangent at the numerical activation
        guess; its maximum with the previous tangent safeguards only the
        approximate preconditioner and never enters the physical residual.
    :param drn_offsets: Authoritative CSR drain offsets by groundwater cell.
    :param riv_offsets: Authoritative CSR river offsets by groundwater cell.
    :param ghb_c: Integrated GHB conductances.
    :param drn_c: Integrated drain conductances.
    :param drn_elevation: Drain elevations.
    :param riv_c: Integrated river conductances.
    :param riv_stage: River stages.
    :param riv_bottom: Riverbed bottoms.
    :param previous: Previous accepted heads used to test branch activation.
    :param has_ghb: One when GHB arrays carry real fields.
    :param has_entries: One when CSR offsets are allocated.
    :param result: Combined previous-state restraint diagonal, written in place.
    :return: None; ``result`` is modified in place.
    """
    i=wp.tid()
    tangent = wp.max(storage_tangent[i], activation_tangent[i])
    if has_ghb != 0:
        tangent = tangent + ghb_c[i]
    if has_entries != 0:
        for entry in range(drn_offsets[i], drn_offsets[i + 1]):
            drain = drn_flow_pair(previous[i], drn_c[entry], drn_elevation[entry])
            tangent = tangent + drain[1]
        for entry in range(riv_offsets[i], riv_offsets[i + 1]):
            river = riv_flow_pair(previous[i], riv_c[entry], riv_stage[entry], riv_bottom[entry])
            tangent = tangent + river[1]
    result[i] = tangent


@wp.kernel
def add_cell_rates(rates:wp.array(dtype=wp.float64),source:wp.array(dtype=wp.float64)):
    """Add one dense volumetric rate field into the resident source.

    :param rates: Signed per-cell volumetric rates [L³/T] to add.
    :param source: Resident cell source, updated in place.
    :return: None; ``source`` is modified in place.
    """
    i=wp.tid()
    source[i]=source[i]+rates[i]


@wp.kernel
def chd_budget(fixed: wp.array(dtype=wp.int32), source: wp.array(dtype=wp.float64),
               left: wp.array2d(dtype=wp.int32),
               right: wp.array2d(dtype=wp.int32), flux: wp.array2d(dtype=wp.float64),
               totals: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if fixed[i] != 0:
        rate = -source[i]
        for axis in range(3):
            if right[i, axis] >= 0:
                rate += flux[i, axis]
            j = left[i, axis]
            if j >= 0:
                rate -= flux[j, axis]
        wp.atomic_add(totals, 0, wp.max(rate, wp.float64(0.0)))
        wp.atomic_add(totals, 1, wp.max(-rate, wp.float64(0.0)))


@wp.kernel
def ghb_budget(head:wp.array(dtype=wp.float64),conductance:wp.array(dtype=wp.float64),
    boundary_head:wp.array(dtype=wp.float64),result:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if conductance[i]>wp.float64(0.):
        q=ghb_flow_at(i,head,conductance,boundary_head)
        wp.atomic_add(result,0,wp.max(-q,wp.float64(0.)))
        wp.atomic_add(result,1,wp.max(q,wp.float64(0.)))


@wp.kernel
def physical_rows(head:wp.array(dtype=wp.float64),source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),fixed:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),
    right:wp.array2d(dtype=wp.int32),flux:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64),residual:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),
    metrics:wp.array2d(dtype=wp.float64),storage:wp.array(dtype=wp.float64),storage_tangent:wp.array(dtype=wp.float64),block_size:int):
    i=wp.tid()
    values=physical_row_metrics(i,head,source,free,fixed,left,right,flux,conductance,scale,residual,rhs,storage,storage_tangent)
    for metric in range(9):
        if values[metric] != wp.float64(0.):
            wp.atomic_add(metrics,i/block_size,metric,values[metric])


@wp.kernel
def physical_rows_tiled(head:wp.array(dtype=wp.float64),source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),fixed:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),
    right:wp.array2d(dtype=wp.int32),flux:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64),residual:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),
    metrics:wp.array(dtype=wp.float64),storage:wp.array(dtype=wp.float64),storage_tangent:wp.array(dtype=wp.float64),block_size:int):
    i=wp.tid()
    values=physical_row_metrics(i,head,source,free,fixed,left,right,flux,conductance,scale,residual,rhs,storage,storage_tangent)
    for metric in range(9):
        wp.tile_store(metrics,wp.tile_sum(wp.tile(values[metric])),offset=(i/block_size)*9+metric)


@wp.kernel
def combine_scalar_tiled(partials:wp.array(dtype=wp.float64),out:wp.array(dtype=wp.float64),count:int):
    lane=wp.tid()
    value=wp.float64(0.)
    for index in range(lane,count,256):
        value+=partials[index]
    wp.tile_store(out,wp.tile_sum(wp.tile(value)),offset=0)


@wp.kernel
def combine_metrics_tiled(partials:wp.array2d(dtype=wp.float64),out:wp.array(dtype=wp.float64),count:int):
    thread=wp.tid()
    metric=thread/256
    lane=thread%256
    value=wp.float64(0.)
    for index in range(lane,count,256):
        value+=partials[index,metric]
    wp.tile_store(out,wp.tile_sum(wp.tile(value)),offset=metric)


@wp.kernel
def combine_physical_metrics(partials:wp.array2d(dtype=wp.float64),
    metrics:wp.array(dtype=wp.float64),count:int):
    metric=wp.tid()
    value=wp.float64(0.)
    for block in range(count):
        value+=partials[block,metric]
    metrics[metric]=value


@wp.kernel
def final_storage_certificate(head:wp.array(dtype=wp.float64),prescribed:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),fixed:wp.array(dtype=wp.int32),storage_tangent:wp.array(dtype=wp.float64),
    dl:wp.array2d(dtype=wp.float64),dr:wp.array2d(dtype=wp.float64),out:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if fixed[i]!=0:
        error=wp.abs(head[i]-prescribed[i])
        if not wp.isfinite(error):error=wp.float64(wp.inf)
        wp.atomic_max(out,0,error)
    if free[i]!=0 and storage_tangent[i]<=wp.float64(0.):
        wp.atomic_add(out,1,wp.float64(1.))
    invalid=not wp.isfinite(storage_tangent[i]) or storage_tangent[i]<wp.float64(0.)
    for axis in range(3):
        if not wp.isfinite(dl[i,axis]) or not wp.isfinite(dr[i,axis]):invalid=True
    if invalid:wp.atomic_add(out,2,wp.float64(1.))


@wp.kernel
def jacobian_action(x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float64),z:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),right:wp.array2d(dtype=wp.int32),
    dl:wp.array2d(dtype=wp.float64),dr:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    c0:wp.array2d(dtype=wp.float64),scale:wp.array(dtype=wp.float64),mode:wp.array(dtype=wp.int32),counter:wp.array(dtype=wp.int64),
    alpha:wp.float64,beta:wp.float64,storage_tangent:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if i==0 and mode[0]==1:wp.atomic_add(counter,0,wp.int64(1))
    value=x[i]
    if free[i]!=0:
        value=wp.float64(0.)
        for axis in range(3):
            j=right[i,axis]
            if j>=0:
                a=dl[i,axis];b=dr[i,axis]
                if mode[0]==0:a=conductance[i,axis];b=-a
                if mode[0]==2:a=c0[i,axis];b=-a
                value+=a*x[i]
                if free[j]!=0:value+=b*x[j]
            j=left[i,axis]
            if j>=0:
                a=dl[j,axis];b=dr[j,axis]
                if mode[0]==0:a=conductance[j,axis];b=-a
                if mode[0]==2:a=c0[j,axis];b=-a
                value-=b*x[i]
                if free[j]!=0:value-=a*x[j]
        value=(value+storage_tangent[i]*x[i])/scale[i]
    old=wp.float64(0.)
    if beta!=wp.float64(0.):old=beta*y[i]
    z[i]=alpha*value+old


@wp.kernel
def diagonal_action(x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float64),z:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),right:wp.array2d(dtype=wp.int32),
    dl:wp.array2d(dtype=wp.float64),dr:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    c0:wp.array2d(dtype=wp.float64),scale:wp.array(dtype=wp.float64),mode:wp.array(dtype=wp.int32),alpha:wp.float64,beta:wp.float64,
    storage_tangent:wp.array(dtype=wp.float64)):
    i=wp.tid();diag=wp.float64(0.)
    if free[i]!=0:
        for axis in range(3):
            if right[i,axis]>=0:
                if mode[0]==2:diag+=c0[i,axis]
                elif mode[0]==0:diag+=conductance[i,axis]
                else:diag+=dl[i,axis]
            j=left[i,axis]
            if j>=0:
                if mode[0]==2:diag+=c0[j,axis]
                elif mode[0]==0:diag+=conductance[j,axis]
                else:diag-=dr[j,axis]
        diag+=storage_tangent[i]
    if diag==wp.float64(0.):diag=scale[i]
    old=wp.float64(0.)
    if beta!=wp.float64(0.):old=beta*y[i]
    z[i]=alpha*x[i]*scale[i]/wp.abs(diag)+old


@wp.func
def has_vertical_connection(cell: int, left: wp.array2d(dtype=wp.int32),
                            c0: wp.array2d(dtype=wp.float64)) -> bool:
    """Default NPF vertical faces retain full conductance even for dry cells."""
    if c0[cell, 2] > wp.float64(0.):
        return True
    above = left[cell, 2]
    if above >= 0:
        return c0[above, 2] > wp.float64(0.)
    return False


@wp.kernel
def candidate_head(head:wp.array(dtype=wp.float64),delta:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),trial:wp.array(dtype=wp.float64),alpha:wp.float64,
    bottom:wp.array(dtype=wp.float64),types:wp.array(dtype=wp.int32),damp_drying:int,
    right:wp.array2d(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),
    c0:wp.array2d(dtype=wp.float64),source:wp.array(dtype=wp.float64)):
    i=wp.tid()
    trial[i]=head[i]
    if free[i]!=0:
        candidate=head[i]+alpha*delta[i]
        if types[i] > 0 and candidate <= bottom[i]:
            if damp_drying != 0:
                # MF6 follows the column down to its lowest connected bottom.
                relaxation_bottom = bottom[i]
                below = right[i, 2]
                while below >= 0 and bottom[below] < relaxation_bottom:
                    relaxation_bottom = bottom[below]
                    below = right[below, 2]
                if candidate < relaxation_bottom:
                    candidate = relaxation_bottom + wp.float64(0.1) * (head[i] - relaxation_bottom)
            if (candidate == bottom[i] and head[i] > bottom[i] and source[i] > wp.float64(0.)
                    and not has_vertical_connection(i, left, c0)):
                # An exact bottom head closes a previously wet recharge outlet.
                # The one-tenth step can itself round to bottom.
                candidate = bottom[i] + wp.float64(0.1) * (head[i] - bottom[i])
                if candidate == bottom[i]:
                    candidate = head[i]
        trial[i]=candidate


@wp.kernel
def shift_linear_diagonal(physical: wp.array(dtype=wp.float64),
                          scale: wp.array(dtype=wp.float64), shift: wp.float64,
                          typical_scale: wp.float64, linear: wp.array(dtype=wp.float64)):
    i = wp.tid()
    weight = wp.float64(1.)
    if typical_scale > wp.float64(0.):
        weight = wp.max(weight, scale[i] / typical_scale)
    linear[i] = physical[i] + shift * weight


@wp.kernel
def project_dry_trial(head: wp.array(dtype=wp.float64), bottom: wp.array(dtype=wp.float64),
                      types: wp.array(dtype=wp.int32), free: wp.array(dtype=wp.int32),
                      source: wp.array(dtype=wp.float64), ghb: wp.array(dtype=wp.float64),
                      head_tolerance: wp.float64, trial: wp.array(dtype=wp.float64),
                      left: wp.array2d(dtype=wp.int32), c0: wp.array2d(dtype=wp.float64)):
    i = wp.tid()
    trial[i] = head[i]
    if (free[i] != 0 and types[i] > 0 and source[i] == wp.float64(0.)
            and ghb[i] == wp.float64(0.) and not has_vertical_connection(i,left,c0)):
        if wp.abs(head[i] - bottom[i]) <= head_tolerance:
            trial[i] = bottom[i]


@wp.kernel
def backtrack_head(head: wp.array(dtype=wp.float64), trial: wp.array(dtype=wp.float64),
                   factor: wp.float64, maximum: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if factor != wp.float64(1.):
        trial[i] = head[i] + factor * (trial[i] - head[i])
    wp.atomic_max(maximum, 0, wp.abs(trial[i] - head[i]))


@wp.kernel
def hold_disconnected_dry_head(head: wp.array(dtype=wp.float64), bottom: wp.array(dtype=wp.float64),
                               types: wp.array(dtype=wp.int32), free: wp.array(dtype=wp.int32),
                               source: wp.array(dtype=wp.float64), residual: wp.array(dtype=wp.float64),
                               tangent: wp.array(dtype=wp.float64), left: wp.array2d(dtype=wp.int32),
                               dl: wp.array2d(dtype=wp.float64), dr: wp.array2d(dtype=wp.float64),
                               delta: wp.array(dtype=wp.float64)):
    i = wp.tid()
    if (delta[i] == wp.float64(0.) or free[i] == 0 or types[i] == 0 or head[i] != bottom[i]
            or source[i] != wp.float64(0.) or residual[i] != wp.float64(0.) or tangent[i] != wp.float64(0.)):
        return
    for axis in range(3):
        if dl[i, axis] != wp.float64(0.) or dr[i, axis] != wp.float64(0.):
            return
        neighbour = left[i, axis]
        if neighbour >= 0 and (dl[neighbour, axis] != wp.float64(0.) or dr[neighbour, axis] != wp.float64(0.)):
            return
    # The physical equation and its Jacobian row and column vanish here.
    # Keep the dry head until forcing or a neighbouring face reconnects it.
    delta[i] = wp.float64(0.)


@wp.kernel
def correction_max(delta:wp.array(dtype=wp.float64),maximum:wp.array(dtype=wp.float64)):
    i=wp.tid()
    wp.atomic_max(maximum,0,wp.abs(delta[i]))


@wp.kernel
def impossible_linear_rows(rhs:wp.array(dtype=wp.float64),free:wp.array(dtype=wp.int32),
    left:wp.array2d(dtype=wp.int32),right:wp.array2d(dtype=wp.int32),dl:wp.array2d(dtype=wp.float64),
    dr:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),c0:wp.array2d(dtype=wp.float64),
    mode:int,count:wp.array(dtype=wp.float64),storage_tangent:wp.array(dtype=wp.float64)):
    i=wp.tid()
    if free[i]==0 or rhs[i]==wp.float64(0.):return
    diag=wp.float64(0.);off=wp.float64(0.)
    for axis in range(3):
        j=right[i,axis]
        if j>=0:
            a=dl[i,axis];b=dr[i,axis]
            if mode==0:a=conductance[i,axis];b=-a
            if mode==2:a=c0[i,axis];b=-a
            diag+=a
            if free[j]!=0:off+=wp.abs(b)
        j=left[i,axis]
        if j>=0:
            a=dl[j,axis];b=dr[j,axis]
            if mode==0:a=conductance[j,axis];b=-a
            if mode==2:a=c0[j,axis];b=-a
            diag-=b
            if free[j]!=0:off+=wp.abs(a)
    diag+=storage_tangent[i]
    if wp.abs(diag)+off==wp.float64(0.):wp.atomic_add(count,0,wp.float64(1.))
@wp.kernel
def previous_storage_scale(principal: wp.array(dtype=wp.float64),
    storage: wp.array(dtype=wp.float64), scale: wp.array(dtype=wp.float64),
    invalid: wp.array(dtype=wp.float64)):
    i = wp.tid()
    value = principal[i]+storage[i]
    if not wp.isfinite(storage[i]) or storage[i] < wp.float64(0.):
        wp.atomic_max(invalid, 0, wp.float64(1.))
    scale[i] = wp.float64(1.)
    if value > wp.float64(0.):
        scale[i] = value


@wp.kernel
def store_head_increment(head: wp.array(dtype=wp.float64), previous: wp.array(dtype=wp.float64),
    free: wp.array(dtype=wp.int32), increment: wp.array(dtype=wp.float64)):
    i = wp.tid()
    increment[i] = wp.float64(0.)
    if free[i] != 0:
        increment[i] = head[i]-previous[i]


@wp.kernel
def predict_timestep_head(head: wp.array(dtype=wp.float64), increment: wp.array(dtype=wp.float64),
    older: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float64),
    ratio: wp.float64, quadratic: int):
    i = wp.tid()
    out[i] = head[i]
    if free[i] != 0:
        change = ratio*increment[i]
        if quadratic != 0:
            change = wp.float64(2.)*increment[i]-older[i]
        out[i] = head[i]+change


@wp.kernel
def scale_period_source(base: wp.array(dtype=wp.float64), source: wp.array(dtype=wp.float64), factor: wp.float64):
    i = wp.tid()
    source[i] = base[i]*factor


@wp.kernel
def add_surface_recharge(receivers: wp.array(dtype=wp.int32), rates: wp.array(dtype=wp.float64),
    column_count: int, area: wp.float64, uniform: int, scalar_rate: wp.float64,
    source: wp.array(dtype=wp.float64)):
    """Add areal recharge to the single eligible source cell in each column."""
    cell = receivers[wp.tid()]
    rate = scalar_rate
    if uniform == 0:
        rate = rates[cell % column_count]
    source[cell] += rate * area


@wp.kernel
def add_gross_source_metrics(source: wp.array(dtype=wp.float64),
    gross: wp.array(dtype=wp.float64), free: wp.array(dtype=wp.int32),
    partials: wp.array2d(dtype=wp.float64), block_size: int):
    """Add physical source throughput omitted by signed source cancellation.

    :param source: Signed numerical source field.
    :param gross: Independently assembled gross physical source field.
    :param free: Free groundwater cells.
    :param partials: Existing per-block physical metrics, modified in place.
    :param block_size: Cells per partial metric block.
    :return: None; the signed residual and head gates remain unchanged.
    """
    i = wp.tid()
    if free[i] != 0:
        correction = gross[i] - wp.abs(source[i])
        if correction > wp.float64(0.0):
            wp.atomic_add(partials, i / block_size, 3, correction)


@wp.kernel
def add_sparse_period_wells(cells: wp.array(dtype=wp.int32), rates: wp.array(dtype=wp.float64),
    gross_rates: wp.array(dtype=wp.float64), source: wp.array(dtype=wp.float64),
    source_gross: wp.array(dtype=wp.float64), has_gross: int):
    """Add unique sparse cell rates without full-grid WEL uploads or atomics.

    :param cells: Unique flat groundwater cell indices.
    :param rates: Stably accumulated signed WEL rates.
    :param gross_rates: Gross per-cell WEL throughput.
    :param source: Resident signed source, updated in place.
    :param source_gross: Optional resident physical throughput field.
    :param has_gross: One when the gross field is present.
    :return: None.
    """
    entry = wp.tid()
    cell = cells[entry]
    source[cell] += rates[entry]
    if has_gross != 0:
        source_gross[cell] += gross_rates[entry]


@wp.kernel
def gross_source_correction(source: wp.array(dtype=wp.float64), gross: wp.array(dtype=wp.float64),
    free: wp.array(dtype=wp.int32), correction: wp.array(dtype=wp.float64)):
    """Retain gross source reporting separately from equation acceptance."""
    i = wp.tid()
    if free[i] != 0:
        wp.atomic_add(correction, 0, gross[i] - wp.abs(source[i]))


@wp.kernel
def add_surface_recharge_gross(receivers: wp.array(dtype=wp.int32),
    rates: wp.array(dtype=wp.float64), column_count: int, area: wp.float64,
    uniform: int, scalar_rate: wp.float64, source_gross: wp.array(dtype=wp.float64)):
    """Add absolute surface recharge with the existing receiving-cell convention.

    :param receivers: Actual flat recharge receiving cells.
    :param rates: Optional nonuniform depth field.
    :param column_count: Horizontal cell count.
    :param area: Horizontal cell area.
    :param uniform: One for scalar recharge.
    :param scalar_rate: Uniform signed depth rate.
    :param source_gross: Resident gross physical source field.
    :return: None.
    """
    index = wp.tid()
    cell = receivers[index]
    rate = scalar_rate
    if uniform == 0:
        rate = rates[cell % column_count]
    source_gross[cell] += wp.abs(rate) * area
