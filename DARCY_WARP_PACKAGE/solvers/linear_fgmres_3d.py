"""FP64 Cartesian defect correction using the qualified NPF flexible K-cycle.

The Cartesian physical operator and independent budget remain authoritative.
Only the linear iteration/preconditioner are shared with the NPF backend.
"""
from dataclasses import asdict

import numpy as np
import warp as wp

from .face_kernels_3d_f64 import apply_A_faces_3d_f64_kernel
from .npf662_fgmres import RestartedFGMRES3D
from .npf662_kcycle import NPF662KCycleControls, NPF662KCycleWorkspace


@wp.kernel
def scaled_vector(value:wp.array(dtype=wp.float64),scale:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),y:wp.array(dtype=wp.float64),out:wp.array(dtype=wp.float64),
    alpha:wp.float64,beta:wp.float64):
    i=wp.tid()
    result=wp.float64(0.)
    if free[i]!=0:
        result=alpha*value[i]/scale[i]+beta*y[i]
    out[i]=result


@wp.kernel
def add_increment(head:wp.array(dtype=wp.float64),delta:wp.array(dtype=wp.float64),
    free:wp.array(dtype=wp.int32),partials:wp.array2d(dtype=wp.float64),block_size:int):
    i=wp.tid()
    if free[i]!=0:
        old=head[i]
        head[i]=old+delta[i]
        change=head[i]-old
        wp.atomic_add(partials,i/block_size,0,change*change)
        wp.atomic_max(partials,i/block_size,1,wp.abs(change))


class CartesianFGMRESWorkspace:
    def __init__(self,*,fine,active,fixed,device,restart):
        self.device=device
        self.fine=fine
        self.shape=fine['dim']
        self.n=fine['n_cells']
        self.restart=restart
        self.free=wp.array(data=((active!=0)&(fixed==0)).astype(np.int32).ravel(),dtype=wp.int32,device=device)
        self.scale=fine['diag'].flatten()
        self.rhs=wp.zeros(shape=self.n,dtype=wp.float64,device=device)
        self.delta=wp.zeros_like(src=self.rhs)
        self.product=wp.zeros_like(src=self.rhs)
        self.block_size=256
        self.updates=wp.zeros(shape=((self.n+self.block_size-1)//self.block_size,2),dtype=wp.float64,device=device)
        self.device_arnoldi=bool(wp.get_device(ident=device).is_cuda)
        self.device_tiled=self.device_arnoldi
        self.flexible=RestartedFGMRES3D(workspace=self,restart=restart)
        self.multigrid=None
        self.closed=False

    def matvec(self,x,y,z,alpha,beta):
        fine=self.fine
        wp.launch(kernel=apply_A_faces_3d_f64_kernel,dim=self.n,
            inputs=[x.reshape(shape=self.shape),*fine['faces'],fine['storage'],fine['active'],fine['bc_mask'],
                self.product.reshape(shape=self.shape),fine['nx'],fine['ny'],fine['nz']],device=self.device)
        wp.launch(kernel=scaled_vector,dim=self.n,
            inputs=[self.product,self.scale,self.free,y,z,alpha,beta],device=self.device)

    def prepare_rhs(self):
        wp.launch(kernel=scaled_vector,dim=self.n,
            inputs=[self.fine['residual'].flatten(),self.scale,self.free,self.rhs,self.rhs,1.,0.],device=self.device)

    def update_head(self,*,free_count):
        self.updates.zero_()
        wp.launch(kernel=add_increment,dim=self.n,
            inputs=[self.fine['x'].flatten(),self.delta,self.free,self.updates,self.block_size],device=self.device)
        values=self.updates.numpy()
        return float(np.sqrt(np.sum(values[:,0])/free_count)),float(np.max(values[:,1]))

    def close(self):
        if self.closed:return
        self.flexible.close()
        if self.multigrid is not None:self.multigrid.close()
        self.multigrid=self.flexible=self.fine=None
        self.free=self.scale=self.rhs=self.delta=self.product=self.updates=None
        self.closed=True


def run_linear_fgmres(*,levels,record,device,residual_norm,head_residual_norm,
    initial_rms,tolerance,rel_tol,dh_rms_tol,dh_max_tol,budget,restart,controls,
    n_free,transient,dt_used,isolated,n_isolated,restore_values,return_info):
    fine=levels[0]
    active,fixed,unused,storage,*faces=record['operator_snapshot']
    workspace=record.get('linear_fgmres')
    reused=workspace is not None and not workspace.closed and workspace.restart==restart
    if not reused:
        if workspace is not None:workspace.close()
        workspace=CartesianFGMRESWorkspace(fine=fine,active=active,fixed=fixed,device=device,restart=restart)
        record['linear_fgmres']=workspace
    if controls is None:
        controls=NPF662KCycleControls(levels=len(levels),cached_line_factors=True,device_hierarchy_build=True)
    if not isinstance(controls,NPF662KCycleControls):
        raise ValueError('linear_kcycle_controls must be NPF662KCycleControls')
    controls.validate()
    mg=workspace.multigrid
    values=dict(faces=tuple(faces),active=active,fixed=fixed,device=device,controls=controls,storage_diag=storage)
    if mg is None or not mg.compatible(**values):
        if mg is not None and mg.can_refresh(faces=tuple(faces),active=active,fixed=fixed,device=device,controls=controls):
            mg.refresh(faces=tuple(faces),active=active,fixed=fixed,storage_diag=storage)
        else:
            if mg is not None:mg.close()
            mg=NPF662KCycleWorkspace(**values)
            workspace.multigrid=mg
    applications=mg.applications
    graph_replays=mg.graph_replays
    head_tolerance=float(dh_rms_tol) if dh_rms_tol is not None else max(float(rel_tol),np.finfo(float).eps)
    rms=initial_rms
    hrms=head_residual_norm()
    used=0
    history=[]
    dh_rms=dh_max=0.
    converged=bool(rms<=tolerance and hrms<=head_tolerance)
    while not converged and used<budget:
        workspace.prepare_rhs()
        # Each correction is solved in its own scale, so FP64 refinement
        # continues after the initial relative linear target is met.
        target=max(0.001*hrms*np.sqrt(n_free),np.finfo(float).tiny)
        count,error,unused=workspace.flexible.solve(preconditioner=mg,max_iterations=budget-used,absolute_tolerance=target)
        used+=count
        dh_rms,dh_max=workspace.update_head(free_count=n_free)
        rms=float(np.sqrt(max(residual_norm(fine,fine['x'],fine['residual']),0.)/n_free))
        hrms=head_residual_norm()
        physical_passed=bool(np.isfinite(rms) and np.isfinite(hrms) and rms<=tolerance and hrms<=head_tolerance)
        dh_passed=(dh_rms_tol is None or dh_rms<=dh_rms_tol) and (dh_max_tol is None or dh_max<=dh_max_tol)
        converged=bool(physical_passed and dh_passed)
        history.append(dict(**workspace.flexible.last_info,physical_rms=rms,head_residual_rms=hrms,
            head_update_rms=dh_rms,head_update_max=dh_max))
        if count==0 or not np.isfinite(error) or (dh_max==0. and not physical_passed):
            break
    head=fine['x'].numpy().copy()
    if n_isolated and isolated is not None:head[isolated]=restore_values[isolated]
    info=dict(solver_type='fgmres_kcycle_7point_3d',implementation='fast',fast_operator='fp64_face_arrays',
        linear_acceleration='fgmres',converged=converged,n_cycles_used=used,krylov_iterations=used,
        linear_budget_units='total FGMRES iterations across FP64 defect corrections',max_cycles=budget,
        n_levels=len(mg.levels),level_shapes=[level['dim'] for level in mg.levels],
        r_rms0=initial_rms,r_rms_end=rms,h_rms_end=hrms,tol_abs=tolerance,head_residual_tolerance=head_tolerance,
        dh_rms_lastcheck=dh_rms,dh_max_lastcheck=dh_max,workspace_reused=bool(record.get('reused',False)),
        linear_workspace_reused=reused,linear_history=history,linear_kcycle_controls=asdict(controls),
        preconditioner_applications=mg.applications-applications,cuda_graph_replayed=mg.graph_replays>graph_replays,
        cuda_graph_capture_failed=mg.graph_failed,transient=bool(transient),
        transient_formulation='confined' if transient else 'steady',dt=float(dt_used) if transient else float('nan'),
        n_isolated_cells_excluded=n_isolated)
    return (head,info) if return_info else head
