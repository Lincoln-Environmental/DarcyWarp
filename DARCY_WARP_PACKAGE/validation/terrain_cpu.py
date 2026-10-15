"""Independent sparse NumPy/SciPy steady nonlinear terrain reference.

No Warp import, GPU controller, packed coefficients, or GPU derivatives.
J is D(divergence), so J delta = source-divergence. Fixed heads are eliminated.
"""
from dataclasses import dataclass
from time import perf_counter
import warnings

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D, harmonic_resistance
from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d


@dataclass(frozen=True)
class CPUControls:
    method: str = 'newton'
    max_outer: int = 6000
    head_tol: float = 1e-6
    flow_tol: float = 1e-12
    relative_tol: float = 1e-8
    relaxation: float = 1.
    max_change: float = 50.
    line_search_steps: int = 20
    recovery_growth: float = 1.10
    fallback: bool = True
    physical_merit: bool = True
    adequate_alpha: float = .125
    adequate_merit_ratio: float = .8
    retry_contraction: float = .5


class CPUReference:
    def __init__(self, *, case):
        case.validate()
        self.case=case
        self.grid=case.grid()
        self.metrics=TerrainMetrics3D(grid=self.grid)
        self.free=(self.grid.active!=0)&(case.arrays['fixed']==0)
        self.ids=np.arange(np.prod(case.shape)).reshape(case.shape)
        self.evaluations=0
        self.selectors=[]
        n=self.ids.size
        for axis in (2,1,0):
            left,right=[slice(None)]*3,[slice(None)]*3
            left[axis],right[axis]=slice(None,-1),slice(1,None)
            left,right=tuple(left),tuple(right)
            l,r=self.ids[left].ravel(),self.ids[right].ravel()
            rows=np.arange(l.size)
            L=sparse.csr_matrix((np.ones(l.size),(rows,l)),shape=(l.size,n))
            U=sparse.csr_matrix((np.ones(l.size),(rows,r)),shape=(l.size,n))
            self.selectors.append((left,right,L,U))

    def evaluate(self, *, head):
        a=self.case.arrays
        op=TerrainReferenceOperator3D(metrics=self.metrics,kx=a['kx'],ky=a['ky'],kz=a['kz'],
            head=head,min_sat=self.case.physical['floor'],contacts=self.case.contacts,trusted_k=True)
        fluxes=op.fluxes(head=head)
        residual=np.where(self.free,a['source']-op.divergence(fluxes=fluxes),0.)
        budget=compute_terrain_budget_3d(operator=op,head=head,source=a['source'],fixed=a['fixed'],
            storage_change=np.zeros_like(head),storage_diagonal=np.zeros_like(head))
        self.evaluations+=1
        return op,residual,budget

    def jacobian(self, *, operator, head):
        """Frozen flux-gradient matrix plus endpoint coefficient derivatives.

        Face coefficient dependencies are only endpoint heads; the frozen
        gradient matrix contains the full multipoint gradient sparsity.
        """
        op=operator
        g,m=self.grid,self.metrics
        floor=np.minimum(self.case.physical['floor'],g.thickness)
        df=((head-g.bottoms>floor)&(head<g.tops))/g.thickness
        dex,dey=op.kx*df,op.ky*df
        gradients=[(G@head.ravel()).reshape(head.shape) for G in (m.gx,m.gy,m.gz)]
        hx,hy,hz=gradients
        J,unused=op.sparse_matrix(head_for_contacts=head)
        area=g.dx*g.dy
        for index,(left,right,L,U) in enumerate(self.selectors):
            if index<2:
                K=op.kx if index==0 else op.ky
                a=op.ex*g.thickness if index==0 else op.ey*g.thickness
                da=K*df*g.thickness
                dl=dr=(g.dx if index==0 else g.dy)/2
                width=g.dy if index==0 else g.dx
                slope=m.zxf if index==0 else m.zyf
                C,wl,wr=harmonic_resistance(left=a[left],right=a[right],dl=dl,dr=dr)
                dCl=wl**2*da[left]/dl
                dCr=wr**2*da[right]/dr
                dwl=-wl*wr*da[left]/a[left]
                dwr=wl*wr*da[right]/a[right]
                jump=head[left]-head[right]
                vl=width*(dCl*jump-(dwl*a[left]+wl*da[left])*slope*hz[left]+dwl*a[right]*slope*hz[right])
                vr=width*(dCr*jump-dwr*a[left]*slope*hz[left]-(-dwr*a[right]+wr*da[right])*slope*hz[right])
            else:
                sx,sy=m.zx_interfaces[1:-1],m.zy_interfaces[1:-1]
                a=op.ex[left]*sx**2+op.ey[left]*sy**2+op.kz[left]
                b=op.ex[right]*sx**2+op.ey[right]*sy**2+op.kz[right]
                da=dex[left]*sx**2+dey[left]*sy**2
                db=dex[right]*sx**2+dey[right]*sy**2
                dl,dr=g.thickness[left]/2,g.thickness[right]/2
                C,wl,wr=harmonic_resistance(left=a,right=b,dl=dl,dr=dr)
                dCl,dCr=wl**2*da/dl,wr**2*db/dr
                dwl,dwr=-wl*wr*da/a,wl*wr*db/b
                tl=(op.ex[left]*sx*hx[left]+op.ey[left]*sy*hy[left]
                    +(op.ex[left]*sx*(m.zx[left]-sx)+op.ey[left]*sy*(m.zy[left]-sy))*hz[left])
                tr=(op.ex[right]*sx*hx[right]+op.ey[right]*sy*hy[right]
                    +(op.ex[right]*sx*(m.zx[right]-sx)+op.ey[right]*sy*(m.zy[right]-sy))*hz[right])
                dtl=(dex[left]*sx*hx[left]+dey[left]*sy*hy[left]
                    +(dex[left]*sx*(m.zx[left]-sx)+dey[left]*sy*(m.zy[left]-sy))*hz[left])
                dtr=(dex[right]*sx*hx[right]+dey[right]*sy*hy[right]
                    +(dex[right]*sx*(m.zx[right]-sx)+dey[right]*sy*(m.zy[right]-sy))*hz[right])
                jump=head[left]-head[right]
                vl=area*(dCl*jump-dwl*tl-wl*dtl+dwl*tr)
                vr=area*(dCr*jump-dwr*tl+dwr*tr-wr*dtr)
            valid=op.active[left]&op.active[right]
            extra=sparse.diags((vl*valid).ravel())@L+sparse.diags((vr*valid).ravel())@U
            J+=(L-U).T@extra
        boundary=op.fluxes(head=head)[3]
        contact_diag=np.zeros(head.size)
        for c,record,q in zip(op.contacts,op.contact_records,boundary):
            cell,conductance,bz,bx,by,hb,discharge=record
            if discharge and q<=0:
                continue
            if c.direction not in ('top','bottom'):
                dc=0. if c.conductance is not None else (dex[c.cell] if c.direction in ('west','east') else dey[c.cell])*c.area/c.distance
                derivative=dc*(head.ravel()[cell]-hb)
                if conductance>0:
                    derivative-=dc/conductance*(bz*hz.ravel()[cell]+bx*hx.ravel()[cell]+by*hy.ravel()[cell])
            else:
                sign=-1 if c.direction=='top' else 1
                k,j,i=c.cell
                interface=k+int(sign>0)
                sx,sy=m.zx_interfaces[interface,j,i],m.zy_interfaces[interface,j,i]
                normal=op.ex[c.cell]*sx*sx+op.ey[c.cell]*sy*sy+op.kz[c.cell]
                dn=dex[c.cell]*sx*sx+dey[c.cell]*sy*sy
                scale=c.area/np.sqrt(1+sx*sx+sy*sy) if c.conductance is None else conductance*c.distance/normal
                ds=0. if c.conductance is None else -scale*dn/normal
                dc=dn*scale/c.distance if c.conductance is None else 0.
                defect=op.ex[c.cell]*sx*(m.zx[c.cell]-sx)+op.ey[c.cell]*sy*(m.zy[c.cell]-sy)
                dd=dex[c.cell]*sx*(m.zx[c.cell]-sx)+dey[c.cell]*sy*(m.zy[c.cell]-sy)
                dbz=sign*(ds*defect+scale*dd)
                dbx=sign*sx*(dex[c.cell]*scale+op.ex[c.cell]*ds)-dc*c.x_offset
                dby=sign*sy*(dey[c.cell]*scale+op.ey[c.cell]*ds)-dc*c.y_offset
                derivative=dc*(head.ravel()[cell]-hb)-dbz*hz.ravel()[cell]-dbx*hx.ravel()[cell]-dby*hy.ravel()[cell]
            contact_diag[cell]+=derivative
        return (J+sparse.diags(contact_diag)).tocsr()


    def check_components(self, *, head):
        """Reject ungrounded/unsuppliable steady hydraulic components."""
        from scipy.sparse.csgraph import connected_components
        n=self.ids.size
        graph=sparse.csr_matrix((n,n))
        for left,right,L,U in self.selectors:
            valid=self.grid.active[left].ravel().astype(bool)&self.grid.active[right].ravel().astype(bool)
            graph+=L.T@sparse.diags(valid.astype(float))@U
        graph=graph+graph.T
        count,labels=connected_components(csgraph=graph,directed=False)
        a=self.case.arrays
        seeds=0
        for component in range(count):
            mask=(labels.reshape(self.case.shape)==component)&self.grid.active.astype(bool)
            if not mask.any() or np.any(a['fixed'][mask]):
                continue
            contacts=[c for c in self.case.contacts if mask[c.cell] and c.conductance!=0]
            if any(c.kind=='head' for c in contacts):
                continue
            net=float(a['source'][mask].sum())
            magnitude=float(np.abs(a['source'][mask]).sum())
            if not contacts or abs(net)<=64*np.finfo(float).eps*magnitude:
                raise ValueError('steady component has undetermined datum')
            if net<0:
                raise ValueError('negative source cannot be supplied by discharge-only component')
            op,unused,unused=self.evaluate(head=head)
            if not any(q>0 for c,q in zip(op.contacts,op.fluxes(head=head)[3]) if mask[c.cell]):
                # Explicit numerical seeding of a valid positive-source branch.
                # No equation, source, datum constraint, or old storage added.
                lift=max(float(np.max(self.grid.tops[mask]-head[mask])),
                    max(c.head-head[c.cell] for c in contacts))+1.
                head[mask]+=lift
                seeds+=1
        return seeds


def accepted(*, head, budget, controls):
    return bool(np.all(np.isfinite(head)) and budget['head_residual_rms']<=controls.head_tol and
        (budget['cell_residual_rms']<=controls.flow_tol or budget['relative_cell_imbalance']<=controls.relative_tol))


def merit(*, budget, head_scale, flow_scale):
    return (budget['head_residual_rms']/head_scale)**2+(budget['cell_residual_rms']/flow_scale)**2


def solve_cpu(*, case, controls, initial=None):
    """Independent direct Newton/Picard with dual-merit backtracking/recovery.

    Newton first tries exact local linearization, then independently computed
    Picard if necessary. Every trial rebuilds nonlinear coefficients/gates.
    No GPU algorithm, convergence flag, or saved result is consulted.
    """
    if controls.method not in ('newton','picard'):
        raise ValueError('unknown CPU method')
    ref=CPUReference(case=case)
    a=case.arrays
    head=a['initial'].copy() if initial is None else np.array(initial,copy=True)
    head[a['fixed']!=0]=a['values'][a['fixed']!=0]
    seeds=ref.check_components(head=head)
    history=[]
    start=perf_counter()
    reason='outer_budget'
    factors=0
    retry_flow=np.inf
    for outer in range(controls.max_outer):
        op,r,budget=ref.evaluate(head=head)
        if accepted(head=head,budget=budget,controls=controls):
            reason='converged'
            break
        hscale=np.inf if controls.physical_merit else max(budget['head_residual_rms'],controls.head_tol)
        qscale=max(budget['cell_residual_rms'],controls.flow_tol)
        old=merit(budget=budget,head_scale=hscale,flow_scale=qscale)
        attempt_newton=controls.method=='newton' and budget['cell_residual_rms']<=retry_flow
        methods=['newton','picard'] if attempt_newton and controls.fallback else [controls.method if attempt_newton else 'picard']
        if controls.method=='newton' and not controls.fallback:
            methods=['newton']
        found=False
        trials=[]
        saved_newton=None
        for method in methods:
            matrix=ref.jacobian(operator=op,head=head) if method=='newton' else op.sparse_matrix(head_for_contacts=head)[0]
            f=ref.free.ravel()
            A=matrix[f][:,f].tocsc()
            rhs=r.ravel()[f]
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('error')
                    factor=splu(A=A)
                correction=factor.solve(rhs=rhs)
                factors+=1
            except (RuntimeError, Warning) as error:
                trials.append(dict(method=method,factorization_failure=str(error)))
                continue
            if not np.all(np.isfinite(correction)):
                trials.append(dict(method=method,failure='nonfinite_correction'))
                continue
            linear=rhs-A@correction
            normA=float(np.max(np.asarray(abs(A).sum(axis=1))))
            backward=float(np.max(np.abs(linear))/max(normA*np.max(np.abs(correction))+np.max(np.abs(rhs)),np.finfo(float).tiny))
            delta=np.zeros_like(head)
            delta.ravel()[f]=correction if method=='newton' else np.clip(correction,-controls.max_change,controls.max_change)
            first_alpha=min(controls.relaxation,controls.max_change/max(float(np.max(np.abs(correction))),np.finfo(float).tiny)) if method=='newton' else controls.relaxation
            best=None
            for attempt in range(controls.line_search_steps):
                alpha=first_alpha*2.**(-attempt)
                candidate=head+alpha*delta
                unused,unused_r,trial=ref.evaluate(head=candidate)
                value=merit(budget=trial,head_scale=hscale,flow_scale=qscale)
                trials.append(dict(method=method,alpha=alpha,merit=value,flow=trial['cell_residual_rms'],
                    head=trial['head_residual_rms'],linear_relative=float(np.linalg.norm(linear)/max(np.linalg.norm(rhs),np.finfo(float).tiny)),backward_error=backward))
                if np.isfinite(value) and (value<old or accepted(head=candidate,budget=trial,controls=controls)):
                    best=(candidate,trial,alpha,'descent')
                    break
            if best is None and controls.recovery_growth>1:
                viable=[t for t in trials if t.get('method')==method and t.get('merit',np.inf)<=controls.recovery_growth**2*old]
                if viable:
                    alpha=viable[0]['alpha']
                    candidate=head+alpha*delta
                    unused,unused_r,trial=ref.evaluate(head=candidate)
                    best=(candidate,trial,alpha,'bounded_recovery')
            if best is not None and method=='newton' and controls.fallback:
                candidate,trial,alpha,mode=best
                if alpha<controls.adequate_alpha or merit(budget=trial,head_scale=hscale,flow_scale=qscale)>controls.adequate_merit_ratio*old:
                    saved_newton=best
                    retry_flow=controls.retry_contraction*qscale
                    best=None
            if best is None and method=='picard' and saved_newton is not None:
                best=saved_newton
                method='newton_retained'
            if best is None and method=='newton':
                retry_flow=controls.retry_contraction*qscale
            if best is not None:
                if method=='newton':
                    retry_flow=np.inf
                head,budget,alpha,mode=best
                found=True
                break
        history.append(dict(outer=outer+1,accepted=found,direction=method,trials=trials,
            budget=budget,alpha=alpha if found else 0.,acceptance=mode if found else 'rejected'))
        if (outer+1)%100==0:
            print(case.case_id,controls.method,'progress',outer+1,budget['cell_residual_rms'],flush=True)
        if not found:
            reason='line_search_or_factorization_failed'
            break
    unused,unused_r,budget=ref.evaluate(head=head)
    passed=accepted(head=head,budget=budget,controls=controls)
    if passed:
        reason='converged'
    return head,dict(converged=passed,stop_reason=reason,outers=len(history),seconds=perf_counter()-start,
        factorizations=factors,discharge_component_seeds=seeds,residual_evaluations=ref.evaluations,budget=budget,history=history)
