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

    def validate(self):
        shape=self.active.shape
        if len(shape)!=3 or min(shape)<1:raise ValueError('Expected nonempty (nz,ny,nx)')
        for name in ('kx','ky','kz','fixed','values','source','initial'):
            value=np.asarray(getattr(self,name))
            if value.shape!=shape or not np.all(np.isfinite(value)):raise ValueError(f'Invalid {name}')
        if any(not np.isfinite(x) or x<=0 for x in (self.dx,self.dy,self.dz)):raise ValueError('Positive finite spacing required')
        if not np.isfinite(self.top):raise ValueError('Finite top required')
        for mask in (self.active,self.fixed):
            if not np.all((mask==0)|(mask==1)):raise ValueError('Binary masks required')
        active=self.active.astype(bool)
        if not active.any():raise ValueError('At least one active cell required')
        if np.any(self.fixed.astype(bool)&~active):raise ValueError('Fixed cells must be active')
        if np.any(self.source[~active]!=0):raise ValueError('Inactive source would be lost')
        for k in (self.kx,self.ky,self.kz):
            if np.any(k[active]<=0):raise ValueError('Positive physical Cartesian K required')

    def digest(self):
        digest=hashlib.sha256()
        for name in self.__dataclass_fields__:
            value=np.ascontiguousarray(getattr(self,name))
            digest.update(name.encode());digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
        return digest.hexdigest()


class NPF662Reference:
    def __init__(self,*,case):
        case.validate()
        self.case=case
        self.shape=case.active.shape
        self.free=case.active.astype(bool)&~case.fixed.astype(bool)
        ids=np.arange(case.active.size).reshape(self.shape)
        left=[];right=[];conductances=[];axes=[]
        for axis,k,d,area in ((2,case.kx,case.dx,case.dy*case.dz),
            (1,case.ky,case.dy,case.dx*case.dz),(0,case.kz,case.dz,case.dx*case.dy)):
            l,u=[slice(None)]*3,[slice(None)]*3
            l[axis],u[axis]=slice(None,-1),slice(1,None)
            l,u=tuple(l),tuple(u)
            mask=case.active[l].astype(bool)&case.active[u].astype(bool)
            a,b=ids[l][mask],ids[u][mask]
            C=area/(.5*d/k[l][mask]+.5*d/k[u][mask])
            if np.any(~np.isfinite(C)) or np.any(C<=0):raise ValueError('Unrepresentable physical conductance')
            left.append(a);right.append(b);conductances.append(C);axes.append(np.full(C.size,axis))
        self.left=np.concatenate(left);self.right=np.concatenate(right)
        self.c0=np.concatenate(conductances);self.axes=np.concatenate(axes)
        n=ids.size;m=self.left.size
        self.incidence=sparse.coo_matrix((np.concatenate([np.ones(m),-np.ones(m)]),
            (np.concatenate([self.left,self.right]),np.tile(np.arange(m),2))),shape=(n,m)).tocsr()
        self.matrix=(self.incidence@sparse.diags(self.c0)@self.incidence.T).tocsr()

    def check_components(self):
        n=self.case.active.size
        graph=sparse.coo_matrix((np.ones(self.left.size),(self.left,self.right)),shape=(n,n)).tocsr()
        unused,labels=connected_components(csgraph=graph,directed=False)
        for label in np.unique(labels[self.case.active.ravel()!=0]):
            mask=(labels==label)&(self.case.active.ravel()!=0)
            if not np.any(self.case.fixed.ravel()[mask]):
                raise ValueError(f'Ungrounded active component; net source={self.case.source.ravel()[mask].sum():.17g}')

    def evaluate(self,*,head):
        c=self.case
        checks=inspect_heads(head=head,active=c.active,fixed=c.fixed,values=c.values)
        h=np.where(c.active,head,0.).ravel()
        flux=self.c0*(h[self.left]-h[self.right])
        divergence=(self.incidence@flux).reshape(self.shape)
        residual=np.where(self.free,c.source-divergence,0.)
        diagonal=self.matrix.diagonal().reshape(self.shape)
        scaled=np.divide(residual,diagonal,out=np.zeros_like(residual),where=diagonal>0)
        scaled[(diagonal<=0)&(residual!=0)]=np.inf
        boundary=self.free.ravel()[self.left]&c.fixed.ravel()[self.right].astype(bool)
        reverse=self.free.ravel()[self.right]&c.fixed.ravel()[self.left].astype(bool)
        exchange=np.concatenate([flux[boundary],-flux[reverse]])
        inflow=float(np.maximum(-exchange,0.).sum());outflow=float(np.maximum(exchange,0.).sum())
        throughput=float(np.abs(c.source[self.free]).sum())+inflow+outflow
        rms=float(np.sqrt(np.mean(residual[self.free]**2))) if self.free.any() else 0.
        hrms=float(np.sqrt(np.mean(scaled[self.free]**2))) if self.free.any() else 0.
        l1=float(np.abs(residual).sum());relative=l1/max(throughput,np.finfo(float).tiny)
        result=dict(**checks,head_residual_rms=hrms,physical_rms=rms,
            physical_max=float(np.abs(residual).max()),residual_l1=l1,relative_cell_imbalance=relative,
            gross_throughput=throughput,signed_imbalance=float(residual.sum()),boundary_inflow=inflow,boundary_outflow=outflow,
            fixed_cell_source_total=float(c.source[c.fixed!=0].sum()),free_cells=int(self.free.sum()))
        result['independent_accepted']=bool(checks['input_valid'] and checks['boundary_condition_valid'] and
            np.all(np.isfinite(flux)) and hrms<=1e-6 and (rms<=1e-12 or relative<=1e-8))
        return result,dict(flux=flux,residual=residual,principal_diagonal=diagonal)

    def solve(self):
        self.check_components()
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
