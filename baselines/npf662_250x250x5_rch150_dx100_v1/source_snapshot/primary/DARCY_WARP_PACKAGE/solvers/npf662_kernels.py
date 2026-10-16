"""FP64 endpoint-only physical NPF662 faces, derivatives and safeguarded updates."""
import warp as wp


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
def physical_rows(head:wp.array(dtype=wp.float64),source:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),fixed:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),
    right:wp.array2d(dtype=wp.int32),flux:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    scale:wp.array(dtype=wp.float64),residual:wp.array(dtype=wp.float64),rhs:wp.array(dtype=wp.float64),
    metrics:wp.array(dtype=wp.float64)):
    i=wp.tid()
    invalid=not wp.isfinite(head[i]) or not wp.isfinite(scale[i]) or scale[i]<=wp.float64(0.)
    for axis in range(3):
        if not wp.isfinite(flux[i,axis]) or not wp.isfinite(conductance[i,axis]):invalid=True
    if invalid:wp.atomic_add(metrics,6,wp.float64(1.))
    if free[i]==0:
        residual[i]=wp.float64(0.);rhs[i]=wp.float64(0.)
        return
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
    r=source[i]-divergence
    equivalent=wp.float64(0.)
    if diagonal>wp.float64(0.):equivalent=r/diagonal
    elif r!=wp.float64(0.):equivalent=wp.float64(wp.inf)
    residual[i]=r;rhs[i]=r/scale[i]
    wp.atomic_add(metrics,0,r*r);wp.atomic_add(metrics,1,wp.abs(r))
    wp.atomic_add(metrics,2,equivalent*equivalent)
    wp.atomic_add(metrics,3,wp.abs(source[i])+inflow+outflow)
    wp.atomic_add(metrics,4,inflow);wp.atomic_add(metrics,5,outflow)
    if not wp.isfinite(r) or not wp.isfinite(diagonal):wp.atomic_add(metrics,6,wp.float64(1.))
    wp.atomic_add(metrics,7,rhs[i]*rhs[i]);wp.atomic_add(metrics,8,r)


@wp.kernel
def jacobian_action(x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float64),z:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),right:wp.array2d(dtype=wp.int32),
    dl:wp.array2d(dtype=wp.float64),dr:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    c0:wp.array2d(dtype=wp.float64),scale:wp.array(dtype=wp.float64),mode:wp.array(dtype=wp.int32),counter:wp.array(dtype=wp.int64),
    alpha:wp.float64,beta:wp.float64):
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
        value=value/scale[i]
    old=wp.float64(0.)
    if beta!=wp.float64(0.):old=beta*y[i]
    z[i]=alpha*value+old


@wp.kernel
def diagonal_action(x:wp.array(dtype=wp.float64),y:wp.array(dtype=wp.float64),z:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),left:wp.array2d(dtype=wp.int32),right:wp.array2d(dtype=wp.int32),
    dl:wp.array2d(dtype=wp.float64),dr:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),
    c0:wp.array2d(dtype=wp.float64),scale:wp.array(dtype=wp.float64),mode:wp.array(dtype=wp.int32),alpha:wp.float64,beta:wp.float64):
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
    if diag==wp.float64(0.):diag=scale[i]
    old=wp.float64(0.)
    if beta!=wp.float64(0.):old=beta*y[i]
    z[i]=alpha*x[i]*scale[i]/wp.abs(diag)+old


@wp.kernel
def candidate_head(head:wp.array(dtype=wp.float64),delta:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),trial:wp.array(dtype=wp.float64),alpha:wp.float64):
    i=wp.tid()
    trial[i]=head[i]
    if free[i]!=0:trial[i]=head[i]+alpha*delta[i]


@wp.kernel
def correction_max(delta:wp.array(dtype=wp.float64),maximum:wp.array(dtype=wp.float64)):
    i=wp.tid()
    wp.atomic_max(maximum,0,wp.abs(delta[i]))


@wp.kernel
def impossible_linear_rows(rhs:wp.array(dtype=wp.float64),free:wp.array(dtype=wp.int32),
    left:wp.array2d(dtype=wp.int32),right:wp.array2d(dtype=wp.int32),dl:wp.array2d(dtype=wp.float64),
    dr:wp.array2d(dtype=wp.float64),conductance:wp.array2d(dtype=wp.float64),c0:wp.array2d(dtype=wp.float64),
    mode:int,count:wp.array(dtype=wp.float64)):
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
    if wp.abs(diag)+off==wp.float64(0.):wp.atomic_add(count,0,wp.float64(1.))
