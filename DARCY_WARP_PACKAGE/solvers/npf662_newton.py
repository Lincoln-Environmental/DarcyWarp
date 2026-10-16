"""Device-resident endpoint Newton/GMRES for the pinned Cartesian contract."""
from dataclasses import asdict,dataclass
from time import perf_counter
import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator
from DARCY_WARP_PACKAGE.solvers.terrain_gmres_3d import TerrainGMRES3D
from DARCY_WARP_PACKAGE.solvers import npf662_kernels as kernels
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics


@dataclass(frozen=True)
class NPF662NewtonControls:
    inner_forcing: float = .01
    newton_budget_fraction: float = .75
    line_search_steps: int = 20
    max_head_change_m: float = 50.
    recovery_growth: float = 1.10

    def validate(self):
        if not 0.<self.inner_forcing<1. or not 0.<self.newton_budget_fraction<=1.:
            raise ValueError('Newton forcing/budget fraction must lie in (0,1]')
        if self.line_search_steps<1 or not np.isfinite(self.max_head_change_m) or self.max_head_change_m<=0.:
            raise ValueError('Positive finite step/search controls required')
        if not 1.<=self.recovery_growth<=1.10:raise ValueError('Recovery growth must be between1 and1.10')


class NPF662Workspace:
    def __init__(self,*,model,case,restart,krylov_budget):
        self.device=model.device;self.shape=case.active.shape;self.n=case.active.size;self.closed=False
        self.restart=restart;self.active_snapshot=case.active.copy();self.fixed_snapshot=case.fixed.copy()
        self.free_count=int(np.sum((case.active!=0)&(case.fixed==0)))
        for name in ('head','trial','source','bottom','thickness','scale','rhs','trial_rhs','residual','trial_residual','delta'):
            setattr(self,name,wp.zeros(shape=self.n,dtype=wp.float64,device=self.device))
        for name in ('c0','q','c','trial_q','trial_c','dl','dr'):
            setattr(self,name,wp.zeros(shape=(self.n,3),dtype=wp.float64,device=self.device))
        for name in ('free','fixed','types'):
            setattr(self,name,wp.zeros(shape=self.n,dtype=wp.int32,device=self.device))
        ids=np.arange(self.n).reshape(self.shape)
        left=np.full((self.n,3),-1,dtype=np.int32);right=left.copy()
        for a,axis in enumerate((2,1,0)):
            l,u=[slice(None)]*3,[slice(None)]*3;l[axis],u[axis]=slice(None,-1),slice(1,None);l,u=tuple(l),tuple(u)
            mask=(case.active[l]!=0)&(case.active[u]!=0)
            i,j=ids[l][mask],ids[u][mask]
            right[i,a]=j;left[j,a]=i
        self.left=wp.array(data=left,dtype=wp.int32,device=self.device)
        self.right=wp.array(data=right,dtype=wp.int32,device=self.device)
        self.metrics=wp.zeros(shape=9,dtype=wp.float64,device=self.device)
        self.maximum=wp.zeros(shape=1,dtype=wp.float64,device=self.device)
        self.mode=wp.ones(shape=1,dtype=wp.int32,device=self.device)
        self.jv_counter=wp.zeros(shape=1,dtype=wp.int64,device=self.device)
        self.residual_evaluations=0;self.candidate_evaluations=0
        self.multigrid=None;self.flexible=None
        self.load(model=model,case=case)
        self.evaluate(trial=False,linearize=True)
        operator=LinearOperator(shape=(self.n,self.n),dtype=wp.float64,device=wp.get_device(ident=self.device),matvec=self.matvec)
        preconditioner=LinearOperator(shape=(self.n,self.n),dtype=wp.float64,device=wp.get_device(ident=self.device),matvec=self.precondition)
        self.krylov=TerrainGMRES3D(A=operator,b=self.rhs,x=self.delta,M=preconditioner,restart=restart,
            maxiter=max(krylov_budget,1),tol=0.,atol=1e-12,check_every=restart)

    def compatible(self,*,model,case,restart):
        return not self.closed and self.device==model.device and self.shape==case.active.shape and self.restart==restart and np.array_equal(
            self.active_snapshot,case.active) and np.array_equal(self.fixed_snapshot,case.fixed)

    def load(self,*,model,case):
        self.source.assign(case.source.ravel())
        self.free.assign(((case.active!=0)&(case.fixed==0)).astype(np.int32).ravel())
        self.fixed.assign(case.fixed.astype(np.int32).ravel());self.types.assign(case.cell_types.astype(np.int32).ravel())
        self.bottom.assign(np.asarray(case.bottoms).ravel())
        self.thickness.assign(np.asarray(case.thicknesses).ravel())
        self.head.assign(np.where(case.active,np.where(case.fixed,case.values,case.initial),0.).ravel())
        # Production C0 comes from the Cartesian session, not the CPU oracle.
        faces=model._session.faces
        self.c0.assign(np.stack([faces[0].ravel(),faces[2].ravel(),faces[4].ravel()],axis=1))
        principal=sum(faces).ravel()
        self.scale.assign(np.where(principal>0.,principal,1.))
        self.jv_counter.zero_();self.residual_evaluations=0;self.candidate_evaluations=0

    def evaluate(self,*,trial,linearize=False):
        head=self.trial if trial else self.head
        q=self.trial_q if trial else self.q;c=self.trial_c if trial else self.c
        r=self.trial_residual if trial else self.residual;rhs=self.trial_rhs if trial else self.rhs
        wp.launch(kernel=kernels.physical_faces,dim=(self.n,3),inputs=[head,self.bottom,self.types,self.right,
            self.c0,q,c,self.dl,self.dr,self.thickness,int(linearize)],device=self.device)
        self.metrics.zero_()
        wp.launch(kernel=kernels.physical_rows,dim=self.n,inputs=[head,self.source,self.free,self.fixed,self.left,
            self.right,q,c,self.scale,r,rhs,self.metrics],device=self.device)
        self.residual_evaluations+=1
        if trial:self.candidate_evaluations+=1
        m=self.metrics.numpy()
        count=max(self.free_count,1)
        return dict(physical_rms=float(np.sqrt(m[0]/count)),head_residual_rms=float(np.sqrt(m[2]/count)),
            relative_cell_imbalance=float(m[1]/max(m[3],np.finfo(float).tiny)),gross_throughput=float(m[3]),
            residual_l1=float(m[1]),signed_imbalance=float(m[8]),boundary_inflow=float(m[4]),boundary_outflow=float(m[5]),
            scaled_l2=float(np.sqrt(m[7])),finite=bool(m[6]==0 and np.all(np.isfinite(m[[0,1,3,4,5,7,8]]))))

    def matvec(self,x,y,z,alpha,beta):
        wp.launch(kernel=kernels.jacobian_action,dim=self.n,inputs=[x,y,z,self.free,self.left,self.right,
            self.dl,self.dr,self.c,self.c0,self.scale,self.mode,self.jv_counter,alpha,beta],device=self.device)

    def precondition(self,x,y,z,alpha,beta):
        wp.launch(kernel=kernels.diagonal_action,dim=self.n,inputs=[x,y,z,self.free,self.left,self.right,
            self.dl,self.dr,self.c,self.c0,self.scale,self.mode,alpha,beta],device=self.device)

    def candidate(self,*,alpha):
        wp.launch(kernel=kernels.candidate_head,dim=self.n,inputs=[self.head,self.delta,self.free,self.trial,alpha],device=self.device)
        return self.evaluate(trial=True)

    def close(self):
        """Break the bound-matvec/Krylov cycle and release owned device buffers."""
        if self.closed:return
        if self.flexible is not None:self.flexible.close()
        if self.multigrid is not None:self.multigrid.close()
        self.flexible=None;self.multigrid=None
        self.closed=True;self.krylov=None
        for name in ('head','trial','source','bottom','thickness','scale','rhs','trial_rhs','residual','trial_residual','delta',
            'c0','q','c','trial_q','trial_c','dl','dr','free','fixed','types','left','right','metrics','maximum','mode','jv_counter'):
            setattr(self,name,None)


def device_accepted(*,metrics,physical_tolerance=1e-12,relative_tolerance=1e-8):
    valid=all(np.isfinite(metrics[name]) and metrics[name]>=0. for name in
        ('head_residual_rms','physical_rms','relative_cell_imbalance','gross_throughput'))
    return metrics['finite'] and valid and metrics['head_residual_rms']<=1e-6 and (
        metrics['physical_rms']<=physical_tolerance or metrics['relative_cell_imbalance']<=relative_tolerance)


def residual_merit(*,candidate,current):
    if not candidate['finite']:return np.inf
    flow=candidate['physical_rms']/max(current['physical_rms'],1e-12)
    head=0. if not np.isfinite(current['head_residual_rms']) else candidate['head_residual_rms']/max(current['head_residual_rms'],1e-6)
    return max(flow,head)**2


def search_direction(*,workspace,current,controls,physical_tolerance=1e-12,relative_tolerance=1e-8):
    workspace.maximum.zero_()
    wp.launch(kernel=kernels.correction_max,dim=workspace.n,inputs=[workspace.delta,workspace.maximum],device=workspace.device)
    maximum=float(workspace.maximum.numpy()[0])
    if maximum==0. or not np.isfinite(maximum):
        return dict(accepted=False,alpha=0.,mode='zero_or_nonfinite_direction',metrics=current,trials=[],raw_max=maximum)
    first=min(1.,controls.max_head_change_m/max(maximum,np.finfo(float).tiny))
    old=residual_merit(candidate=current,current=current)
    rows=[];best=None;best_value=np.inf
    for attempt in range(controls.line_search_steps):
        alpha=first*2.**(-attempt)
        trial=workspace.candidate(alpha=alpha)
        value=residual_merit(candidate=trial,current=current)
        rows.append(dict(alpha=alpha,merit=value if np.isfinite(value) else None,metrics=trial))
        if value<best_value:best=(alpha,trial);best_value=value
        if device_accepted(metrics=trial,physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance) or value<old:
            return dict(accepted=True,alpha=alpha,mode='dual_descent',metrics=trial,trials=rows,raw_max=maximum)
    if best is not None and best_value<=controls.recovery_growth**2*old:
        alpha,unused=best
        trial=workspace.candidate(alpha=alpha)
        return dict(accepted=True,alpha=alpha,mode='bounded_recovery',metrics=trial,trials=rows,raw_max=maximum)
    return dict(accepted=False,alpha=0.,mode='rejected',metrics=current,trials=rows,raw_max=maximum)


def solve_newton_device(*,model,case,reference,max_outer,krylov_budget,restart,controls,
                        physical_tolerance=1e-12,relative_tolerance=1e-8,
                        npf_preconditioner='diagonal',npf_kcycle_controls=None):
    controls.validate()
    if max_outer<0 or krylov_budget<0 or restart<2:raise ValueError('Nonnegative budgets and restart>=2 required')
    holder=model._session._npf662_workspace_holder
    workspace=holder.get('workspace')
    reused=workspace is not None and workspace.compatible(model=model,case=case,restart=restart)
    if not reused:
        if workspace is not None:workspace.close()
        workspace=NPF662Workspace(model=model,case=case,restart=restart,krylov_budget=krylov_budget)
        holder['workspace']=workspace
    else:workspace.load(model=model,case=case)
    mg_reused=False;mg_seconds=0.;linear_history=[]
    if npf_preconditioner=='kcycle':
        from .npf662_kcycle import NPF662KCycleControls,NPF662KCycleWorkspace
        from .npf662_fgmres import RestartedFGMRES3D
        mg_controls=NPF662KCycleControls() if npf_kcycle_controls is None else npf_kcycle_controls
        if not isinstance(mg_controls,NPF662KCycleControls):raise ValueError('npf_kcycle_controls must be NPF662KCycleControls')
        mg_controls.validate()
        mg_reused=workspace.multigrid is not None and workspace.multigrid.compatible(
            faces=model._session.faces,active=case.active,fixed=case.fixed,device=model.device,controls=mg_controls)
        if not mg_reused:
            if workspace.multigrid is not None:workspace.multigrid.close()
            workspace.multigrid=NPF662KCycleWorkspace(faces=model._session.faces,active=case.active,
                fixed=case.fixed,device=model.device,controls=mg_controls)
            mg_seconds=workspace.multigrid.construction_seconds
        if workspace.flexible is None:workspace.flexible=RestartedFGMRES3D(workspace=workspace,restart=restart)
        mg_start=(workspace.multigrid.applications,workspace.multigrid.cycles,workspace.multigrid.graph_replays)
    elif npf_preconditioner!='diagonal':raise ValueError('npf_preconditioner must be diagonal or kcycle')
    if wp.get_device(ident=model.device).is_cuda:wp.synchronize_device(device=model.device)
    start=perf_counter();total=0;fallbacks=0;saturated=0;backtracks=0;history=[];reason='outer_budget'
    current=workspace.evaluate(trial=False,linearize=True)
    for outer in range(max_outer):
        if device_accepted(metrics=current,physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance):reason='converged';break
        remaining=krylov_budget;accepted=False;attempts=[]
        for mode in (1,0,2):
            if remaining<=0:reason='krylov_budget';break
            direction={1:'newton',0:'picard',2:'full_saturated_safeguard'}[mode]
            if mode!=1:fallbacks+=1
            workspace.maximum.zero_()
            wp.launch(kernel=kernels.impossible_linear_rows,dim=workspace.n,inputs=[workspace.rhs,workspace.free,
                workspace.left,workspace.right,workspace.dl,workspace.dr,workspace.c,workspace.c0,mode,workspace.maximum],device=workspace.device)
            impossible=int(workspace.maximum.numpy()[0])
            if impossible:
                attempts.append(dict(direction=direction,krylov=0,remaining=remaining,stopping='zero_row_nonzero_rhs',zero_rows=impossible))
                continue
            if mode==2:saturated+=1
            allowance=min(remaining,max(1,int(controls.newton_budget_fraction*krylov_budget))) if mode==1 else max(1,remaining//2) if mode==0 else remaining
            workspace.mode.fill_(mode);workspace.delta.zero_()
            if npf_preconditioner=='diagonal':
                workspace.krylov._maxiter=allowance
                workspace.krylov._atol=max(controls.inner_forcing*current['scaled_l2'],np.finfo(float).tiny)
                used,error,target=workspace.krylov()
            else:
                used,error,target=workspace.flexible.solve(preconditioner=workspace.multigrid,max_iterations=allowance,
                    absolute_tolerance=max(controls.inner_forcing*current['scaled_l2'],np.finfo(float).tiny))
                linear_history.append(dict(outer=outer+1,direction=direction,**workspace.flexible.last_info))
            total+=int(used);remaining-=int(used)
            search=search_direction(workspace=workspace,current=current,controls=controls,
                physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance)
            backtracks+=max(len(search['trials'])-1,0)
            attempts.append(dict(direction=direction,krylov=int(used),remaining=remaining,
                achieved_scaled_l2=float(error),requested_scaled_l2=float(target),target_reached=bool(error<=target),
                stopping='target' if error<=target else 'budget_or_breakdown',search=search))
            if search['accepted']:
                wp.copy(src=workspace.trial,dest=workspace.head)
                current=workspace.evaluate(trial=False,linearize=True)
                accepted=True;break
        history.append(dict(outer=outer+1,accepted=accepted,krylov=krylov_budget-remaining,attempts=attempts,metrics=current))
        if not accepted:
            if reason!='krylov_budget':reason='line_search_failed'
            break
    workspace.mode.fill_(1)
    current=workspace.evaluate(trial=False,linearize=True)
    if wp.get_device(ident=model.device).is_cuda:wp.synchronize_device(device=model.device)
    seconds=perf_counter()-start
    head=workspace.head.numpy().reshape(case.active.shape)
    checks,unused=reference.evaluate(head=head,physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance)
    device_passed=device_accepted(metrics=current,physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance)
    converged=bool(checks['independent_accepted'] and device_passed)
    if not device_passed and checks['independent_accepted']:reason='device_acceptance_failed'
    if not converged and reason=='converged':reason='independent_acceptance_failed'
    if checks['ungrounded_current_components']>0:reason='ungrounded_saturation_components'
    result=dict(converged=converged,stop_reason='converged' if converged else reason,outer_iterations=len(history),
        krylov_iterations=total,jacobian_vector_products=int(workspace.jv_counter.numpy()[0]),
        nonlinear_residual_evaluations=workspace.residual_evaluations,candidate_evaluations=workspace.candidate_evaluations,
        nonlinear_backtracks=backtracks,fallback_count=fallbacks,outer_history=history,independent_acceptance=checks,
        full_saturated_safeguard_solves=saturated,
        device_final_metrics=current,workspace_reused=reused,seconds=seconds,controls=asdict(controls),
        linear_budget_units='Krylov iterations shared across Newton and Picard per outer',
        max_krylov_per_outer=krylov_budget,max_outer=max_outer,krylov_restart=restart,
        scaling='fixed full-saturated principal diagonal; S*J and S*R',preconditioner='fixed diagonal within each GMRES solve')
    result['krylov_method']='GMRES' if npf_preconditioner=='diagonal' else 'FGMRES'
    result['npf_preconditioner']=npf_preconditioner
    if npf_preconditioner=='kcycle':
        mg=workspace.multigrid
        result.update(preconditioner='full-saturated A0 K-cycle; physical RHS D0*v; homogeneous corrections',
            linear_history=linear_history,krylov_restarts=sum(row['restarts'] for row in linear_history),
            multigrid=dict(controls=asdict(mg.controls),smoother='vertical_line',
                dimensions=[list(level['dim']) for level in mg.levels],actual_levels=len(mg.levels),
                coarsening_factors=[[level['coarsen_y'],level['coarsen_x']] for level in mg.levels[:-1]],
                construction_seconds=mg_seconds,hierarchy_reused=mg_reused,
                allocated_device_bytes=mg.allocated_bytes(),
                applications=mg.applications-mg_start[0],cycles=mg.cycles-mg_start[1],
                graph_captures=int(not mg_reused)*mg.graph_captures,graph_replays=mg.graph_replays-mg_start[2],
                graph_capture_failed=mg.graph_failed,graph_error=mg.graph_error,
                inner_scalar_readbacks=0,
                scalar_readbacks=sum(row['scalar_readbacks'] for row in linear_history),
                synchronization_seconds=sum(row['synchronization_seconds'] for row in linear_history),
                device_apply_seconds=sum(row['multigrid_device_seconds'] for row in linear_history),
                preconditioner_enqueue_seconds=sum(row['preconditioner_enqueue_seconds'] for row in linear_history)))
    return head,finite_diagnostics(value=result)
