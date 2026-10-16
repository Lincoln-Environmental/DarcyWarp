"""Independent NumPy/SciPy reference for contract NPF662 Cartesian v1.

No Warp imports or production conductance arrays. Outward flux has positive
incidence on its first endpoint. The sparse Jacobian is D(div q)=-D(R).
"""
from dataclasses import dataclass
import hashlib
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from DARCY_WARP_PACKAGE.validation.head_checks import inspect_heads
from DARCY_WARP_PACKAGE.validation.npf662_metrics import final_tolerances, stable_rms, stable_sum


def serialized_case_fields(*,case):
    names=tuple(case.__dataclass_fields__)
    if getattr(case,'_legacy_implicit_cell_type',False) and np.ndim(case.cell_type)==0 and case.cell_type==0:
        return tuple(name for name in names if name!='cell_type')
    return names


@dataclass
class NPF662Case:
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    active: np.ndarray
    fixed: np.ndarray
    values: np.ndarray
    source: np.ndarray
    initial: np.ndarray
    dx: float
    dy: float
    dz: float
    top: float = 0.
    cell_type: np.ndarray | int = 0

    def validate(self):
        shape=self.active.shape
        if len(shape)!=3 or min(shape)<1:raise ValueError('Expected nonempty (nz,ny,nx)')
        for name in ('kx','ky','kz','fixed','values','source','initial'):
            value=np.asarray(getattr(self,name))
            if value.shape!=shape or not np.all(np.isfinite(value)):raise ValueError(f'Invalid {name}')
        spacing=(self.dx,self.dy) if self.geometry=='structured_dis_layered' else (self.dx,self.dy,self.dz)
        if any(np.ndim(x)!=0 or not np.isfinite(x) or x<=0 for x in spacing):raise ValueError('Positive finite spacing required')
        if self.geometry!='structured_dis_layered' and (np.ndim(self.top)!=0 or not np.isfinite(self.top)):
            raise ValueError('Finite flat scalar top required; variable elevations are unsupported')
        for mask in (self.active,self.fixed):
            if not np.all((mask==0)|(mask==1)):raise ValueError('Binary masks required')
        active=self.active.astype(bool)
        if not active.any():raise ValueError('At least one active cell required')
        if np.any(self.fixed.astype(bool)&~active):raise ValueError('Fixed cells must be active')
        if np.any(self.source[~active]!=0):raise ValueError('Inactive source would be lost')
        for k in (self.kx,self.ky,self.kz):
            if np.any(k[active]<=0):raise ValueError('Positive physical Cartesian K required')
        types=np.asarray(self.cell_type)
        if types.ndim!=0 and types.shape!=shape:raise ValueError('cell_type must be scalar or full-grid 0/1 array')
        if np.any((types!=0)&(types!=1)):raise ValueError('Only ICELLTYPE 0/1 supported')
        if np.any(active&(self.fixed!=0)&(self.cell_types!=0)&(self.values<self.bottoms)):
            raise ValueError('Convertible CHD head below cell bottom is invalid for MF6 6.6.2; prescribed heads are not repaired')

    @property
    def cell_types(self):
        return np.broadcast_to(self.cell_type,self.active.shape)

    @property
    def bottoms(self):
        return np.broadcast_to((self.top-self.dz*np.arange(1,self.active.shape[0]+1))[:,None,None],self.active.shape)

    @property
    def tops(self):
        return np.broadcast_to((self.top-self.dz*np.arange(self.active.shape[0]))[:,None,None],self.active.shape)

    @property
    def thicknesses(self):
        return np.full(self.active.shape,self.dz,dtype=float)

    @property
    def geometry(self):
        return 'orthogonal_cartesian_uniform_dz'

    @property
    def contract(self):
        return 'NPF662_CARTESIAN_V1'

    def digest(self):
        digest=hashlib.sha256()
        for name in serialized_case_fields(case=self):
            value=np.ascontiguousarray(getattr(self,name))
            digest.update(name.encode());digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
        return digest.hexdigest()


@dataclass
class NPF662LayeredCase:
    """Separate schema2: no meaningless scalar dz/top in the canonical inputs."""
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    active: np.ndarray
    fixed: np.ndarray
    values: np.ndarray
    source: np.ndarray
    initial: np.ndarray
    dx: float
    dy: float
    z_interfaces: np.ndarray
    cell_type: np.ndarray | int = 0

    def validate(self):
        from DARCY_WARP_PACKAGE.physics.npf662_layers import layered_geometry
        layered_geometry(z_interfaces=self.z_interfaces,shape=self.active.shape,dx=self.dx,dy=self.dy)
        NPF662Case.validate(self)

    @property
    def cell_types(self):
        return np.broadcast_to(self.cell_type,self.active.shape)

    @property
    def bottoms(self):
        return np.asarray(self.z_interfaces,dtype=np.float64)[1:]

    @property
    def tops(self):
        return np.asarray(self.z_interfaces,dtype=np.float64)[:-1]

    @property
    def thicknesses(self):
        return self.tops-self.bottoms

    @property
    def geometry(self):
        return 'structured_dis_layered'

    @property
    def contract(self):
        return 'NPF662_LAYERED_DIS_V2'

    def digest(self):
        digest=hashlib.sha256(b'NPF662_LAYERED_DIS_V2')
        for name in self.__dataclass_fields__:
            value=np.ascontiguousarray(getattr(self,name))
            digest.update(name.encode());digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
        return digest.hexdigest()


def save_npf662_case(*,case,path):
    arrays={name:np.asarray(getattr(case,name)) for name in serialized_case_fields(case=case)}
    if isinstance(case,NPF662LayeredCase):arrays['schema_version']=np.asarray(2)
    np.savez_compressed(path,**arrays)


def load_npf662_case(*,path):
    with np.load(path,allow_pickle=False) as saved:
        schema=int(saved['schema_version']) if 'schema_version' in saved else 1
        if schema not in (1,2):raise ValueError('Unsupported NPF input schema')
        cls=NPF662Case if schema==1 else NPF662LayeredCase
        missing=set(cls.__dataclass_fields__)-set(saved.files)
        implicit=bool(schema==1 and missing=={'cell_type'})
        if missing and not implicit:raise ValueError(f'Missing NPF input fields: {sorted(missing)}')
        case=cls(**{name:saved[name].copy() for name in cls.__dataclass_fields__ if name in saved})
        if implicit:case._legacy_implicit_cell_type=True
    case.validate()
    return case


class NPF662Reference:
    def __init__(self,*,case):
        case.validate()
        self.case=case
        self.shape=case.active.shape
        self.free=case.active.astype(bool)&~case.fixed.astype(bool)
        ids=np.arange(case.active.size).reshape(self.shape)
        left=[];right=[];conductances=[];axes=[]
        for axis,k,d,width in ((2,case.kx,case.dx,case.dy),
            (1,case.ky,case.dy,case.dx),(0,case.kz,1.,case.dx*case.dy)):
            l,u=[slice(None)]*3,[slice(None)]*3
            l[axis],u[axis]=slice(None,-1),slice(1,None)
            l,u=tuple(l),tuple(u)
            mask=case.active[l].astype(bool)&case.active[u].astype(bool)
            a,b=ids[l][mask],ids[u][mask]
            if case.geometry=='structured_dis_layered':
                if axis==0:width=np.longdouble(case.dx)*case.dy
                kl,kr=np.asarray(k[l][mask],dtype=np.longdouble),np.asarray(k[u][mask],dtype=np.longdouble)
                tl,tr=np.asarray(case.thicknesses[l][mask],dtype=np.longdouble),np.asarray(case.thicknesses[u][mask],dtype=np.longdouble)
                resistance=(.5*d/kl/tl+.5*d/kr/tr) if axis else (.5*tl/kl+.5*tr/kr)
                with np.errstate(over='ignore',under='ignore'):C=np.asarray(np.longdouble(width)/resistance,dtype=float)
            else:
                distance=case.dz if axis==0 else d
                area=width if axis==0 else width*case.dz
                C=area/(.5*distance/k[l][mask]+.5*distance/k[u][mask])
            if np.any(~np.isfinite(C)) or np.any(C<=0):raise ValueError('Unrepresentable physical conductance')
            left.append(a);right.append(b);conductances.append(C);axes.append(np.full(C.size,axis))
        self.left=np.concatenate(left);self.right=np.concatenate(right)
        self.c0=np.concatenate(conductances);self.axes=np.concatenate(axes)
        n=ids.size;m=self.left.size
        self.incidence=sparse.coo_matrix((np.concatenate([np.ones(m),-np.ones(m)]),
            (np.concatenate([self.left,self.right]),np.tile(np.arange(m),2))),shape=(n,m)).tocsr()
        self.matrix=(self.incidence@sparse.diags(self.c0)@self.incidence.T).tocsr()
        indices=np.arange(m)
        self.select_left=sparse.csr_matrix((np.ones(m),(indices,self.left)),shape=(m,n))
        self.select_right=sparse.csr_matrix((np.ones(m),(indices,self.right)),shape=(m,n))

    def face_terms(self,*,head):
        c=self.case
        h=np.where(c.active,head,0.).ravel()
        saturation,derivative=quadratic_saturation_derivative(head=head,bottom=c.bottoms,thickness=c.thicknesses,cell_type=c.cell_types)
        choose_left=(h[self.left]>h[self.right])|((h[self.left]==h[self.right])&(self.axes==1))
        donor=np.where(choose_left,self.left,self.right)
        horizontal=self.axes!=0
        variable=horizontal&((c.cell_types.ravel()[self.left]!=0)|(c.cell_types.ravel()[self.right]!=0))
        conductance=self.c0*np.where(variable,saturation.ravel()[donor],1.)
        extra=np.where(variable,self.c0*derivative.ravel()[donor]*(h[self.left]-h[self.right]),0.)
        dl=conductance+np.where(choose_left,extra,0.)
        dr=-conductance+np.where(~choose_left,extra,0.)
        return conductance,dl,dr,saturation

    def jacobian(self,*,head):
        unused,dl,dr,unused_saturation=self.face_terms(head=head)
        return (self.incidence@(self.select_left.multiply(dl[:,None])+self.select_right.multiply(dr[:,None]))).tocsr()

    def check_components(self):
        n=self.case.active.size
        graph=sparse.coo_matrix((np.ones(self.left.size),(self.left,self.right)),shape=(n,n)).tocsr()
        unused,labels=connected_components(csgraph=graph,directed=False)
        for label in np.unique(labels[self.case.active.ravel()!=0]):
            mask=(labels==label)&(self.case.active.ravel()!=0)
            if not np.any(self.case.fixed.ravel()[mask]):
                raise ValueError(f'Ungrounded active component; net source={self.case.source.ravel()[mask].sum():.17g}')

    def evaluate(self,*,head,physical_tolerance=1e-12,relative_tolerance=1e-8,
                 head_tolerance_m=1e-6,boundary_tolerance_m=1e-10):
        c=self.case
        thresholds=final_tolerances(physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance,
            head_tolerance_m=head_tolerance_m,boundary_tolerance_m=boundary_tolerance_m)
        checks=inspect_heads(head=head,active=c.active,fixed=c.fixed,values=c.values,
            boundary_tolerance_m=boundary_tolerance_m)
        h=np.where(c.active,head,0.).ravel()
        conductance,dl,dr,saturation=self.face_terms(head=head)
        flux=conductance*(h[self.left]-h[self.right])
        divergence=(self.incidence@flux).reshape(self.shape)
        # Recover a representable signed sum if sparse accumulation overflowed
        # before cancellation. Do not repair unrepresentable positive diagonals.
        for index in np.flatnonzero(~np.isfinite(divergence.ravel())):
            row=self.incidence.getrow(int(index))
            divergence.ravel()[index]=stable_sum(values=row.data*flux[row.indices])
        residual=np.where(self.free,c.source-divergence,0.)
        diagonal=np.asarray(abs(self.incidence)@conductance).reshape(self.shape)
        scaled=np.divide(residual,diagonal,out=np.zeros_like(residual),where=diagonal>0)
        scaled[(diagonal<=0)&(residual!=0)]=np.inf
        boundary=self.free.ravel()[self.left]&c.fixed.ravel()[self.right].astype(bool)
        reverse=self.free.ravel()[self.right]&c.fixed.ravel()[self.left].astype(bool)
        exchange=np.concatenate([flux[boundary],-flux[reverse]])
        inflow=stable_sum(values=np.maximum(-exchange,0.));outflow=stable_sum(values=np.maximum(exchange,0.))
        throughput=stable_sum(values=np.concatenate([np.abs(c.source[self.free]),np.abs(exchange)]))
        rms=stable_rms(values=residual[self.free]);hrms=stable_rms(values=scaled[self.free])
        l1=stable_sum(values=np.abs(residual[self.free]));relative=l1/max(throughput,np.finfo(float).tiny)
        active=c.active!=0
        metrics_valid=all(np.all(np.isfinite(value)) for value in
            (flux,conductance,dl,dr,saturation[active],diagonal[active],divergence[active],residual[self.free],scaled[self.free]))
        metrics_valid=bool(metrics_valid and np.all(np.isfinite([inflow,outflow,throughput,rms,hrms,l1,relative])))
        edges=conductance>0.
        graph=sparse.coo_matrix((np.ones(int(edges.sum())),(self.left[edges],self.right[edges])),shape=self.matrix.shape).tocsr()
        unused,labels=connected_components(csgraph=graph,directed=False)
        active_labels=set(labels[c.active.ravel()!=0])
        grounded_labels=set(labels[(c.active.ravel()!=0)&(c.fixed.ravel()!=0)])
        ungrounded=len(active_labels-grounded_labels)
        result=dict(**checks,head_residual_rms=hrms,physical_rms=rms,
            physical_max=float(np.abs(residual).max()),residual_l1=l1,relative_cell_imbalance=relative,
            gross_throughput=throughput,signed_imbalance=stable_sum(values=residual[self.free]),boundary_inflow=inflow,boundary_outflow=outflow,
            fixed_cell_source_total=stable_sum(values=c.source[c.fixed!=0]),free_cells=int(self.free.sum()),
            fixed_reservoir_required_inflow=stable_sum(values=(divergence-c.source)[c.fixed!=0]),
            ungrounded_current_components=ungrounded,physical_metrics_valid=metrics_valid,
            final_acceptance_thresholds=thresholds)
        result['physical_metrics_valid']=bool(metrics_valid and np.all(np.isfinite([
            result['signed_imbalance'],result['fixed_cell_source_total'],result['fixed_reservoir_required_inflow']])))
        result['independent_accepted']=bool(checks['input_valid'] and checks['boundary_condition_valid'] and
            result['physical_metrics_valid'] and ungrounded==0 and hrms<=head_tolerance_m and
            (rms<=physical_tolerance or relative<=relative_tolerance))
        return result,dict(flux=flux,residual=residual,fixed_reservoir_required_inflow=np.where(c.fixed,divergence-c.source,0.),
            principal_diagonal=diagonal,saturation=saturation,
            conductance=conductance,dflux_left=dl,dflux_right=dr)

    def solve(self,*,max_outer=200):
        self.check_components()
        if np.any(self.case.cell_types[self.case.active!=0]):
            return self.solve_newton(max_outer=max_outer)
        c=self.case;f=self.free.ravel();fixed=c.fixed.ravel()!=0
        head=np.where(c.fixed,c.values,c.initial).astype(float).ravel().copy()
        linear_max=backward=None
        if f.any():
            rhs=c.source.ravel()[f]-self.matrix[f][:,fixed]@head[fixed]
            matrix=self.matrix[f][:,f].tocsc()
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                head[f]=spsolve(A=matrix,b=rhs)
            error=rhs-matrix@head[f]
            linear_max=float(np.max(np.abs(error)))
            norm=float(np.max(np.asarray(abs(matrix).sum(axis=1))))
            backward=linear_max/max(norm*np.max(np.abs(head[f]))+np.max(np.abs(rhs)),np.finfo(float).tiny)
        head=head.reshape(self.shape)
        result,fields=self.evaluate(head=head)
        result.update(linear_residual_max=linear_max,linear_backward_error=backward)
        return head,result,fields

    def solve_newton(self,*,max_outer):
        c=self.case;f=self.free.ravel()
        head=np.where(c.fixed,c.values,c.initial).astype(float).copy()
        history=[];reason='outer_budget'
        for outer in range(max_outer):
            checks,fields=self.evaluate(head=head)
            if checks['independent_accepted']:
                reason='converged';break
            rhs=fields['residual'].ravel()[f]
            old=checks['physical_rms']
            found=False;trials=[]
            for mode in ('newton','full_saturated_safeguard'):
                A=(self.jacobian(head=head) if mode=='newton' else self.matrix)[f][:,f].tocsc()
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('error')
                        correction=spsolve(A=A,b=rhs)
                except (RuntimeError,Warning) as error:
                    trials.append(dict(direction=mode,factorization_failure=str(error)));continue
                if not np.all(np.isfinite(correction)):
                    trials.append(dict(direction=mode,nonfinite_correction=True));continue
                delta=np.zeros_like(head);delta.ravel()[f]=correction
                alpha=min(1.,50./max(float(np.max(np.abs(correction))),np.finfo(float).tiny))
                for attempt in range(30):
                    trial=head+alpha*delta
                    check,trial_fields=self.evaluate(head=trial)
                    if check['input_valid'] and (check['physical_rms']<old or check['independent_accepted']):
                        head=trial;found=True;break
                    alpha*=.5
                linear=float(np.max(np.abs(A@correction-rhs)))
                trials.append(dict(direction=mode,alpha=alpha,accepted=found,linear_residual_max=linear,
                    physical_rms=check['physical_rms']))
                if found:break
            history.append(dict(outer=outer+1,accepted=found,attempts=trials))
            if not found:
                reason='line_search_failed';break
        checks,fields=self.evaluate(head=head)
        checks.update(stop_reason='converged' if checks['independent_accepted'] else reason,history=history,outers=len(history))
        return head,checks,fields


def quadratic_saturation_derivative(*,head,bottom,thickness,cell_type):
    t=np.clip((np.asarray(head)-bottom)/thickness,0.,1.)
    eps=1e-6
    a=1./(1.-eps)
    saturation=np.where(t<eps,a*.5*t*t/eps,np.where(t<1.-eps,a*t+.5*(1.-a),
        np.where(t<1.,1.-a*.5*(1.-t)**2/eps,1.)))
    derivative=np.where(t<eps,a*t/eps,np.where(t<1.-eps,a,np.where(t<1.,a*(1.-t)/eps,0.)))/thickness
    return np.where(cell_type!=0,saturation,1.),np.where(cell_type!=0,derivative,0.)
