"""Device-resident endpoint Newton/GMRES for the pinned Cartesian contract."""
from dataclasses import asdict,dataclass
from time import perf_counter
import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator
from DARCY_WARP_PACKAGE.solvers.terrain_gmres_3d import TerrainGMRES3D
from DARCY_WARP_PACKAGE.solvers import npf662_kernels as kernels
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
from DARCY_WARP_PACKAGE.solvers.work_counters_3d import increment_work
from DARCY_WARP_PACKAGE.physics.inputs_3d import positive_integer
from DARCY_WARP_PACKAGE.physics.ghb_3d import validate_ghb_fields


@dataclass(frozen=True)
class NPF662NewtonControls:
    inner_forcing: float = .01
    newton_budget_fraction: float = .75
    line_search_steps: int = 20
    max_head_change_m: float = 50.
    recovery_growth: float = 1.10
    adaptive_forcing: bool = False
    forcing_min: float = 1e-4
    forcing_max: float = .3
    forcing_power: float = 1.5
    timestep_predictor: bool = False
    timestep_predictor_order: int = 1
    tiled_reductions: bool = False
    reuse_trial_state: bool = False

    def validate(self):
        if not isinstance(self.line_search_steps, (int, np.integer)):
            raise ValueError('line_search_steps must be a positive integer')
        positive_integer(value=self.line_search_steps, name='line_search_steps')
        if not 0.<self.inner_forcing<1. or not 0.<self.newton_budget_fraction<=1.:
            raise ValueError('Newton forcing/budget fraction must lie in (0,1]')
        if self.line_search_steps<1 or not np.isfinite(self.max_head_change_m) or self.max_head_change_m<=0.:
            raise ValueError('Positive finite step/search controls required')
        if not 1.<=self.recovery_growth<=1.10:raise ValueError('Recovery growth must be between1 and1.10')
        if not 0. < self.forcing_min <= self.forcing_max < 1. or not np.isfinite(self.forcing_power) or self.forcing_power <= 0.:
            raise ValueError('Adaptive forcing bounds/power must be positive, with bounds below one')
        if any(not isinstance(value, (bool, np.bool_)) for value in (self.adaptive_forcing, self.timestep_predictor, self.tiled_reductions, self.reuse_trial_state)):
            raise ValueError('Adaptive forcing and timestep predictor switches must be Boolean')
        if isinstance(self.timestep_predictor_order, (bool, np.bool_)) or self.timestep_predictor_order not in (1, 2):
            raise ValueError('Timestep predictor order must be 1 or 2')


class NPF662Workspace:
    def __init__(self,*,model,case,restart,krylov_budget,reference=None):
        self.device=model.device;self.shape=case.active.shape;self.n=case.active.size;self.closed=False
        self.restart=restart;self.active_snapshot=case.active.copy();self.fixed_snapshot=case.fixed.copy()
        self.free_count=int(np.sum((case.active!=0)&(case.fixed==0)))
        for name in ('head','prescribed','trial','source','bottom','thickness','scale','rhs','trial_rhs','residual','trial_residual','delta',
            'storage_previous','storage_ss','storage_sy','storage_rate','storage_tangent','trial_storage_rate','trial_storage_tangent',
            'principal','previous_storage_tangent','predictor_increment','predictor_older_increment'):
            setattr(self,name,wp.zeros(shape=self.n,dtype=wp.float64,device=self.device))
        for name in ('c0','q','c','trial_q','trial_c','dl','dr','trial_dl','trial_dr'):
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
        self.metric_block_size=256
        self.metric_partials=wp.zeros(shape=((self.n+self.metric_block_size-1)//self.metric_block_size,9),
            dtype=wp.float64,device=self.device)
        self.certificate=wp.zeros(shape=3,dtype=wp.float64,device=self.device)
        self.maximum=wp.zeros(shape=1,dtype=wp.float64,device=self.device)
        self.mode=wp.ones(shape=1,dtype=wp.int32,device=self.device)
        self.jv_counter=wp.zeros(shape=1,dtype=wp.int64,device=self.device)
        self.preconditioner_counter=wp.zeros(shape=1,dtype=wp.int64,device=self.device)
        self.residual_evaluations=0;self.candidate_evaluations=0
        self.multigrid=None;self.flexible=None;self.krylov=None;self.diagonal_flexible=None
        self.commit_graph=None;self.commit_graph_failed=False;self.reuse_trial_state=False
        self.ghb_conductance = self.ghb_head = self.ghb_totals = None
        self.ghb_operator = self.trial_ghb_operator = self.previous_ghb_operator = None
        self.roundoff_acceptance = False
        self.roundoff_partials = self.roundoff_zero = None
        self.device_native = bool(getattr(reference, 'device_storage_state', False))
        self.load(model=model,case=case,reference=reference)
        self.evaluate(trial=False,linearize=True)

    def select_krylov(self,*,method,krylov_budget):
        """Own only the selected basis; share all hydraulic/operator buffers."""
        if self.closed:raise RuntimeError('NPF workspace is closed')
        if method not in ('gmres','fgmres'):raise ValueError('Krylov method must be gmres or fgmres')
        if method=='fgmres':
            self.krylov=None  # Also releases its captured graphs and bound operators.
            reused=self.flexible is not None
            if not reused:
                from .npf662_fgmres import RestartedFGMRES3D
                self.flexible=RestartedFGMRES3D(workspace=self,restart=self.restart)
        else:
            if self.flexible is not None:self.flexible.close()
            self.flexible=None
            reused=self.krylov is not None
            if not reused:
                operator=LinearOperator(shape=(self.n,self.n),dtype=wp.float64,device=wp.get_device(ident=self.device),matvec=self.matvec)
                preconditioner=LinearOperator(shape=(self.n,self.n),dtype=wp.float64,device=wp.get_device(ident=self.device),matvec=self.precondition)
                self.krylov=TerrainGMRES3D(A=operator,b=self.rhs,x=self.delta,M=preconditioner,restart=self.restart,
                    maxiter=max(krylov_budget,1),tol=0.,atol=1e-12,check_every=self.restart)
        return reused

    def compatible(self,*,model,case,restart):
        return not self.closed and self.device==model.device and self.shape==case.active.shape and self.restart==restart and np.array_equal(
            self.active_snapshot,case.active) and np.array_equal(self.fixed_snapshot,case.fixed)

    def load(self,*,model,case,reference=None):
        native = getattr(reference, 'device_storage_state', False)
        static_reused = native and getattr(self, '_native_state', None) is reference
        conductance, boundary_head = validate_ghb_fields(conductance=case.ghb_conductance,
            head=case.ghb_head, active=case.active, fixed=case.fixed)
        if conductance is not None and self.ghb_conductance is None:
            for name in ('ghb_conductance', 'ghb_head', 'ghb_operator', 'trial_ghb_operator', 'previous_ghb_operator'):
                setattr(self, name, wp.zeros(shape=self.n, dtype=wp.float64, device=self.device))
            self.ghb_totals = wp.zeros(shape=2, dtype=wp.float64, device=self.device)
            # Existing captured matvec/commit graphs reference the old diagonal.
            self.krylov = None
            if self.flexible is not None:self.flexible.close()
            self.flexible = None
            self.commit_graph = None
            self.commit_graph_failed = False
        if self.ghb_conductance is not None:
            if conductance is None:
                self.ghb_conductance.zero_();self.ghb_head.zero_()
            else:
                self.ghb_conductance.assign(conductance.ravel())
                self.ghb_head.assign(boundary_head.ravel())
        self.source.assign(case.source.ravel())
        self.prescribed.assign(case.values.ravel())
        if not static_reused:
            self.free.assign(((case.active!=0)&(case.fixed==0)).astype(np.int32).ravel())
            self.fixed.assign(case.fixed.astype(np.int32).ravel())
            self.bottom.assign(np.asarray(case.bottoms).ravel())
            self.thickness.assign(np.asarray(case.thicknesses).ravel())
        if not static_reused or not np.array_equal(self._types_snapshot, case.cell_type):
            self.types.assign(case.cell_types.astype(np.int32).ravel())
            self._types_snapshot = np.asarray(case.cell_type).copy()
        self.head.assign(np.where(case.active,np.where(case.fixed,case.values,case.initial),0.).ravel())
        # Production C0 comes from the Cartesian session, not the CPU oracle.
        faces=model._session.faces
        self.faces_changed = not static_reused or any(not np.array_equal(old, new)
            for old, new in zip(self._faces_snapshot, faces))
        if self.faces_changed:
            self.c0.assign(np.stack([faces[0].ravel(),faces[2].ravel(),faces[4].ravel()],axis=1))
            self.principal.assign(sum(faces).ravel())
            self._faces_snapshot = tuple(face.copy() for face in faces) if native else None
        self._native_state = reference if native else None
        self.transient=reference is not None and hasattr(reference,'previous')
        self.storage_factor=0.
        if self.transient:
            self.storage_previous.assign(reference.previous.ravel())
            for name in ('ss', 'sy'):
                scalar = getattr(reference, name+'_scalar', None)
                buffer = getattr(self, 'storage_'+name)
                if scalar is None:
                    buffer.assign(getattr(reference, name).ravel())
                else:
                    buffer.fill_(scalar)
            self.storage_factor=case.dx*case.dy/reference.dt
            if native:
                wp.launch(kernel=kernels.storage_terms,dim=self.n,
                    inputs=[self.storage_previous,self.storage_previous,self.bottom,self.thickness,
                        self.types,self.storage_ss,self.storage_sy,self.free,self.storage_rate,
                        self.previous_storage_tangent,self.storage_factor],device=self.device)
                reference.preconditioner_storage_device = self.previous_storage_tangent
            else:
                self.previous_storage_tangent.assign(reference.preconditioner_storage.ravel())
        else:
            for name in ('storage_rate','storage_tangent','trial_storage_rate','trial_storage_tangent'):
                getattr(self,name).zero_()
            self.previous_storage_tangent.zero_()
        self.refresh_previous_operator()
        self.maximum.zero_()
        wp.launch(kernel=kernels.previous_storage_scale,dim=self.n,
            inputs=[self.principal,self.previous_operator_tangent,self.scale,self.maximum],device=self.device)
        if native and self.maximum.numpy()[0] != 0.:
            raise ValueError('Invalid device previous-state storage tangent')
        self.jv_counter.zero_();self.preconditioner_counter.zero_();self.residual_evaluations=0;self.candidate_evaluations=0

    @property
    def operator_tangent(self):
        return self.storage_tangent if self.ghb_conductance is None else self.ghb_operator

    @property
    def previous_operator_tangent(self):
        return self.previous_storage_tangent if self.ghb_conductance is None else self.previous_ghb_operator

    def refresh_previous_operator(self):
        if self.ghb_conductance is not None:
            wp.launch(kernel=kernels.ghb_diagonal, dim=self.n,
                inputs=[self.previous_storage_tangent,self.ghb_conductance,self.previous_ghb_operator],device=self.device)

    def boundary_budget(self):
        if self.ghb_conductance is None:
            return dict(ghb_inflow=0.,ghb_outflow=0.,ghb_net=0.)
        self.ghb_totals.zero_()
        wp.launch(kernel=kernels.ghb_budget,dim=self.n,
            inputs=[self.head,self.ghb_conductance,self.ghb_head,self.ghb_totals],device=self.device)
        inflow,outflow=self.ghb_totals.numpy()
        return dict(ghb_inflow=float(inflow),ghb_outflow=float(outflow),ghb_net=float(inflow-outflow))

    def evaluate(self,*,trial,linearize=False):
        head=self.trial if trial else self.head
        q=self.trial_q if trial else self.q;c=self.trial_c if trial else self.c
        r=self.trial_residual if trial else self.residual;rhs=self.trial_rhs if trial else self.rhs
        storage=self.trial_storage_rate if trial else self.storage_rate
        tangent=self.trial_storage_tangent if trial else self.storage_tangent
        cache_trial=trial and self.reuse_trial_state
        dl=self.trial_dl if cache_trial else self.dl
        dr=self.trial_dr if cache_trial else self.dr
        if self.transient:
            wp.launch(kernel=kernels.storage_terms,dim=self.n,inputs=[head,self.storage_previous,self.bottom,
                self.thickness,self.types,self.storage_ss,self.storage_sy,self.free,storage,tangent,self.storage_factor],device=self.device)
        if self.ghb_conductance is not None:
            wp.launch(kernel=kernels.ghb_diagonal,dim=self.n,
                inputs=[tangent,self.ghb_conductance,self.trial_ghb_operator if trial else self.ghb_operator],device=self.device)
        wp.launch(kernel=kernels.physical_faces,dim=(self.n,3),inputs=[head,self.bottom,self.types,self.right,
            self.c0,q,c,dl,dr,self.thickness,int(linearize or cache_trial)],device=self.device)
        self.metric_partials.zero_()
        block_size=self.metric_block_size if (getattr(self,'device_native',False) or
            getattr(self,'device_tiled',False)) else self.n
        tiled = getattr(self,'device_tiled',False)
        row_kernel=kernels.physical_rows_tiled if tiled else kernels.physical_rows
        row_inputs=[head,self.source,self.free,self.fixed,self.left,self.right,q,c,self.scale,r,rhs,
            self.metric_partials.flatten() if tiled else self.metric_partials,storage,tangent,block_size]
        if self.ghb_conductance is not None:
            row_kernel=kernels.physical_rows_ghb_tiled if tiled else kernels.physical_rows_ghb
            row_inputs.extend((self.ghb_conductance,self.ghb_head))
        wp.launch(kernel=row_kernel,dim=self.n,inputs=row_inputs,
            device=self.device,block_dim=self.metric_block_size)
        wp.launch(kernel=kernels.combine_metrics_tiled if tiled else kernels.combine_physical_metrics,dim=2304 if tiled else 9,
            inputs=[self.metric_partials,self.metrics,(self.n+block_size-1)//block_size],device=self.device,block_dim=256)
        self.residual_evaluations+=1
        if trial:self.candidate_evaluations+=1
        m=self.metrics.numpy()
        count=max(self.free_count,1)
        result=dict(physical_rms=float(np.sqrt(m[0]/count)),head_residual_rms=float(np.sqrt(m[2]/count)),
            relative_cell_imbalance=float(m[1]/max(m[3],np.finfo(float).tiny)),gross_throughput=float(m[3]),
            residual_l1=float(m[1]),signed_imbalance=float(m[8]),boundary_inflow=float(m[4]),boundary_outflow=float(m[5]),
            scaled_l2=float(np.sqrt(m[7])),finite=bool(m[6]==0 and np.all(np.isfinite(m[[0,1,3,4,5,7,8]]))))
        if self.roundoff_acceptance:
            result.update(roundoff_mode=True, roundoff_accepted=False, roundoff_bound_ratio_max=None)
            if (result['finite'] and result['head_residual_rms'] <= 1e-12 and
                    (result['physical_rms'] <= 1e-12 or result['relative_cell_imbalance'] <= 1e-8)):
                if self.roundoff_partials is None:
                    self.roundoff_partials=wp.zeros(shape=(self.n+255)//256,dtype=wp.float64,device=self.device)
                    self.roundoff_zero=wp.zeros(shape=self.n,dtype=wp.float64,device=self.device)
                self.roundoff_partials.zero_();self.maximum.zero_()
                wp.launch(kernel=kernels.roundoff_rows,dim=self.n,inputs=[head,self.source,r,self.free,
                    self.left,self.right,c,self.roundoff_zero if self.ghb_conductance is None else self.ghb_conductance,
                    self.roundoff_zero if self.ghb_head is None else self.ghb_head,storage,tangent,
                    self.storage_previous,self.previous_storage_tangent,self.types,self.bottom,self.thickness,
                    self.roundoff_partials],device=self.device)
                wp.launch(kernel=kernels.combine_roundoff,dim=self.roundoff_partials.size,
                    inputs=[self.roundoff_partials,self.maximum],device=self.device)
                ratio=float(self.maximum.numpy()[0])
                result.update(roundoff_bound_ratio_max=ratio,roundoff_accepted=bool(np.isfinite(ratio) and ratio<=1.))
        return result

    def matvec(self,x,y,z,alpha,beta):
        wp.launch(kernel=kernels.jacobian_action,dim=self.n,inputs=[x,y,z,self.free,self.left,self.right,
            self.dl,self.dr,self.c,self.c0,self.scale,self.mode,self.jv_counter,alpha,beta,self.operator_tangent],device=self.device)

    def precondition(self,x,y,z,alpha,beta):
        wp.launch(kernel=increment_work, dim=1, inputs=[self.preconditioner_counter, 0], device=self.device)
        wp.launch(kernel=kernels.diagonal_action,dim=self.n,inputs=[x,y,z,self.free,self.left,self.right,
            self.dl,self.dr,self.c,self.c0,self.scale,self.mode,alpha,beta,self.operator_tangent],device=self.device)

    def candidate(self,*,alpha):
        wp.launch(kernel=kernels.candidate_head,dim=self.n,inputs=[self.head,self.delta,self.free,self.trial,alpha],device=self.device)
        return self.evaluate(trial=True)

    def _enqueue_commit(self):
        for current,trial in (('head','trial'),('q','trial_q'),('c','trial_c'),('dl','trial_dl'),
                ('dr','trial_dr'),('rhs','trial_rhs'),('residual','trial_residual'),
                ('storage_rate','trial_storage_rate'),('storage_tangent','trial_storage_tangent')):
            wp.copy(dest=getattr(self,current),src=getattr(self,trial))
        if self.ghb_conductance is not None:
            wp.copy(dest=self.ghb_operator,src=self.trial_ghb_operator)

    def accept_trial(self,*,metrics):
        """Commit the evaluated state into stable buffers referenced by graphs."""
        if not self.reuse_trial_state:
            wp.copy(dest=self.head,src=self.trial)
            return self.evaluate(trial=False,linearize=True)
        if wp.get_device(ident=self.device).is_cuda and not self.commit_graph_failed:
            if self.commit_graph is None:
                try:
                    with wp.ScopedCapture(device=self.device) as capture:
                        self._enqueue_commit()
                    if capture.graph is None:raise RuntimeError('Empty trial commit graph')
                    self.commit_graph=capture.graph
                except Exception:
                    self.commit_graph_failed=True
            if not self.commit_graph_failed:
                wp.capture_launch(graph=self.commit_graph)
            else:self._enqueue_commit()
        else:self._enqueue_commit()
        return metrics

    def final_certificate(self):
        self.certificate.zero_()
        wp.launch(kernel=kernels.final_storage_certificate,dim=self.n,
            inputs=[self.head,self.prescribed,self.free,self.fixed,self.storage_tangent,
                    self.dl,self.dr,self.certificate],device=self.device)
        values=self.certificate.numpy()
        return dict(prescribed_head_error_max_m=float(values[0]),
            all_free_cells_storage_anchored=bool(values[1]==0),
            derivatives_and_storage_finite=bool(values[2]==0 and np.all(np.isfinite(values))))

    def close(self):
        """Break the bound-matvec/Krylov cycle and release owned device buffers."""
        if self.closed:return
        if self.flexible is not None:self.flexible.close()
        if self.multigrid is not None:self.multigrid.close()
        if self.diagonal_flexible is not None:self.diagonal_flexible.close()
        self.flexible=None;self.multigrid=None;self.diagonal_flexible=None
        self.closed=True;self.krylov=None
        self.commit_graph=None
        self._native_state = self._faces_snapshot = self._types_snapshot = None
        self.principal = self.previous_storage_tangent = None
        self.ghb_conductance = self.ghb_head = self.ghb_totals = None
        self.ghb_operator = self.trial_ghb_operator = self.previous_ghb_operator = None
        self.predictor_increment = self.predictor_older_increment = self.predictor_checkpoint = None
        self.roundoff_partials = self.roundoff_zero = None
        for name in ('head','prescribed','trial','source','bottom','thickness','scale','rhs','trial_rhs','residual','trial_residual','delta',
            'c0','q','c','trial_q','trial_c','dl','dr','trial_dl','trial_dr','free','fixed','types','left','right','metrics','maximum','mode','jv_counter','preconditioner_counter',
            'storage_previous','storage_ss','storage_sy','storage_rate','storage_tangent','trial_storage_rate','trial_storage_tangent','certificate','metric_partials'):
            setattr(self,name,None)


def device_accepted(*,metrics,physical_tolerance=1e-12,relative_tolerance=1e-8):
    valid=all(np.isfinite(metrics[name]) and metrics[name]>=0. for name in
        ('head_residual_rms','physical_rms','relative_cell_imbalance','gross_throughput'))
    if metrics.get('roundoff_mode',False):
        return bool(metrics['finite'] and valid and metrics['head_residual_rms']<=1e-12 and
            (metrics['physical_rms']<=1e-12 or metrics['relative_cell_imbalance']<=1e-8) and metrics['roundoff_accepted'])
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
                        npf_preconditioner='diagonal',npf_kcycle_controls=None,npf_krylov_method=None,
                        device_native=False,prepared_workspace=None,device_sequence=False,roundoff_acceptance=False):
    controls.validate()
    setup_start=perf_counter()
    if max_outer<0 or krylov_budget<0 or restart<2:raise ValueError('Nonnegative budgets and restart>=2 required')
    method=('fgmres' if npf_preconditioner=='kcycle' else 'gmres') if npf_krylov_method is None else npf_krylov_method
    if method not in ('gmres','fgmres'):raise ValueError('npf_krylov_method must be gmres or fgmres')
    if npf_preconditioner=='kcycle' and method!='fgmres':raise ValueError('K-cycle requires flexible GMRES')
    holder=model._session._npf662_workspace_holder
    workspace=holder.get('workspace')
    if device_sequence and (prepared_workspace is None or not device_native):
        raise ValueError('A device sequence requires its prepared native workspace')
    if prepared_workspace is not None:
        if workspace is not prepared_workspace or workspace.closed or workspace.restart != restart:
            raise ValueError('Prepared NPF workspace is no longer current')
        reused=True
    else:
        reused=workspace is not None and workspace.compatible(model=model,case=case,restart=restart)
    if not reused:
        if workspace is not None:workspace.close()
        workspace=NPF662Workspace(model=model,case=case,restart=restart,krylov_budget=krylov_budget,reference=reference)
        holder['workspace']=workspace
    elif prepared_workspace is None:workspace.load(model=model,case=case,reference=reference)
    krylov_reused=workspace.select_krylov(method=method,krylov_budget=krylov_budget)
    tiled = bool(controls.tiled_reductions and wp.get_device(ident=model.device).is_cuda)
    if tiled != getattr(workspace,'device_tiled',False) and workspace.flexible is not None:
        workspace.flexible.arnoldi_graphs.clear()
    workspace.device_tiled = tiled
    workspace.device_arnoldi=device_native or tiled
    workspace.device_native=device_native
    workspace.reuse_trial_state=controls.reuse_trial_state
    workspace.roundoff_acceptance=roundoff_acceptance
    blocks_start=getattr(workspace.krylov, 'restart_blocks', 0) if method=='gmres' else 0
    graph_start=(getattr(workspace.krylov, 'graph_captures', 0), getattr(workspace.krylov, 'graph_replays', 0))
    mg_reused=False;mg_seconds=0.;linear_history=[];mg_values_refreshed=False
    if npf_preconditioner=='kcycle':
        from .npf662_kcycle import NPF662KCycleControls,NPF662KCycleWorkspace
        mg_controls=NPF662KCycleControls() if npf_kcycle_controls is None else npf_kcycle_controls
        if not isinstance(mg_controls,NPF662KCycleControls):raise ValueError('npf_kcycle_controls must be NPF662KCycleControls')
        mg_controls.validate()
        direct_refresh = (device_native and workspace.multigrid is not None
            and not workspace.multigrid.closed and workspace.multigrid.controls == mg_controls
            and not workspace.faces_changed)
        storage_diagonal = None if direct_refresh else getattr(reference,'preconditioner_storage',None)
        if not direct_refresh and workspace.ghb_conductance is not None:
            storage_diagonal=workspace.previous_operator_tangent.numpy().reshape(case.active.shape)
        mg_reused=direct_refresh or (workspace.multigrid is not None and workspace.multigrid.compatible(
            faces=model._session.faces,active=case.active,fixed=case.fixed,device=model.device,controls=mg_controls,
            storage_diag=storage_diagonal))
        if direct_refresh:
            refresh_start=perf_counter()
            wp.copy(dest=workspace.multigrid.levels[0]['storage'].flatten(), src=workspace.previous_operator_tangent)
            # Current-coefficient preconditioning refreshes the complete
            # hierarchy immediately before each Krylov solve. Avoid preparing
            # an intermediate previous-state hierarchy that is never applied.
            if mg_controls.coefficient_state != 'current':
                workspace.multigrid.refresh_storage_device()
            workspace.multigrid.value_refresh_count += 1
            # Host storage snapshot is deliberately unavailable after a device
            # refresh. A later host path must compare or refresh real values.
            workspace.multigrid.snapshots = (*workspace.multigrid.snapshots[:-1], None)
            mg_seconds=perf_counter()-refresh_start
            mg_values_refreshed=True
        if not mg_reused and device_native and workspace.multigrid is not None and workspace.multigrid.can_refresh(
                faces=model._session.faces,active=case.active,fixed=case.fixed,device=model.device,controls=mg_controls):
            refresh_start=perf_counter()
            workspace.multigrid.refresh(faces=model._session.faces,active=case.active,fixed=case.fixed,
                storage_diag=storage_diagonal)
            mg_seconds=perf_counter()-refresh_start
            mg_reused=True;mg_values_refreshed=True
        if not mg_reused:
            if workspace.multigrid is not None:workspace.multigrid.close()
            workspace.multigrid=NPF662KCycleWorkspace(faces=model._session.faces,active=case.active,
                fixed=case.fixed,device=model.device,controls=mg_controls,storage_diag=storage_diagonal)
            mg_seconds=workspace.multigrid.construction_seconds
        mg_start=(workspace.multigrid.applications,workspace.multigrid.cycles,workspace.multigrid.graph_replays)
    elif npf_preconditioner!='diagonal':raise ValueError('npf_preconditioner must be diagonal or kcycle')
    if method=='fgmres':
        if npf_preconditioner=='kcycle':flex_preconditioner=workspace.multigrid
        else:
            if workspace.diagonal_flexible is None:
                from .npf662_fgmres import DiagonalFGMRESPreconditioner
                workspace.diagonal_flexible=DiagonalFGMRESPreconditioner(workspace=workspace)
            flex_preconditioner=workspace.diagonal_flexible
        applications_start=flex_preconditioner.applications
    if wp.get_device(ident=model.device).is_cuda:wp.synchronize_device(device=model.device)
    start=perf_counter();total=0;fallbacks=0;saturated=0;backtracks=0;history=[];reason='outer_budget'
    setup_seconds=start-setup_start
    current=workspace.evaluate(trial=False,linearize=True)
    predictor_used=False
    predictor_continuation=False
    checkpoint=getattr(workspace,'predictor_checkpoint',None)
    if controls.timestep_predictor and workspace.transient and checkpoint is not None:
        last_head,last_dt,quadratic_available=checkpoint
        if device_sequence or (np.array_equal(reference.previous,last_head) and np.array_equal(case.initial,reference.previous)):
            predictor_continuation=True
            quadratic=controls.timestep_predictor_order == 2 and quadratic_available and reference.dt == last_dt
            wp.launch(kernel=kernels.predict_timestep_head,dim=workspace.n,
                inputs=[workspace.head,workspace.predictor_increment,workspace.predictor_older_increment,
                    workspace.free,workspace.trial,reference.dt/last_dt,int(quadratic)],
                device=workspace.device)
            predicted=workspace.evaluate(trial=True)
            if predicted['finite'] and predicted['scaled_l2'] < current['scaled_l2'] and residual_merit(candidate=predicted,current=current)<1.:
                current=workspace.accept_trial(metrics=predicted)
                predictor_used=True
    forcing_previous=current['scaled_l2']
    for outer in range(max_outer):
        if device_accepted(metrics=current,physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance):reason='converged';break
        forcing=controls.inner_forcing
        if controls.adaptive_forcing and outer:
            forcing=float(np.clip((current['scaled_l2']/max(forcing_previous,np.finfo(float).tiny))**controls.forcing_power,
                controls.forcing_min,controls.forcing_max))
        forcing_previous=current['scaled_l2']
        remaining=krylov_budget;accepted=False;attempts=[]
        for mode in (1,0,2):
            if remaining<=0:reason='krylov_budget';break
            direction={1:'newton',0:'picard',2:'full_saturated_safeguard'}[mode]
            if mode!=1:fallbacks+=1
            workspace.maximum.zero_()
            wp.launch(kernel=kernels.impossible_linear_rows,dim=workspace.n,inputs=[workspace.rhs,workspace.free,
                workspace.left,workspace.right,workspace.dl,workspace.dr,workspace.c,workspace.c0,mode,workspace.maximum,
                workspace.operator_tangent],device=workspace.device)
            impossible=int(workspace.maximum.numpy()[0])
            if impossible:
                attempts.append(dict(direction=direction,krylov=0,remaining=remaining,stopping='zero_row_nonzero_rhs',zero_rows=impossible))
                continue
            if mode==2:saturated+=1
            allowance=min(remaining,max(1,int(controls.newton_budget_fraction*krylov_budget))) if mode==1 else max(1,remaining//2) if mode==0 else remaining
            workspace.mode.fill_(mode);workspace.delta.zero_()
            if npf_preconditioner == 'kcycle' and workspace.multigrid.controls.coefficient_state == 'current':
                workspace.multigrid.refresh_current_operator(workspace=workspace, mode=mode)
            if method=='gmres':
                workspace.krylov._maxiter=allowance
                workspace.krylov._atol=max(forcing*current['scaled_l2'],np.finfo(float).tiny)
                used,error,target=workspace.krylov()
            else:
                used,error,target=workspace.flexible.solve(preconditioner=flex_preconditioner,max_iterations=allowance,
                    absolute_tolerance=max(forcing*current['scaled_l2'],np.finfo(float).tiny))
                linear_history.append(dict(outer=outer+1,direction=direction,**workspace.flexible.last_info))
            total+=int(used);remaining-=int(used)
            search=search_direction(workspace=workspace,current=current,controls=controls,
                physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance)
            backtracks+=max(len(search['trials'])-1,0)
            attempts.append(dict(direction=direction,krylov=int(used),remaining=remaining,
                achieved_scaled_l2=float(error),requested_scaled_l2=float(target),target_reached=bool(error<=target),
                stopping='target' if error<=target else 'budget_or_breakdown',search=search))
            if search['accepted']:
                current=workspace.accept_trial(metrics=search['metrics'])
                accepted=True;break
        history.append(dict(outer=outer+1,accepted=accepted,krylov=krylov_budget-remaining,attempts=attempts,metrics=current,inner_forcing=forcing))
        if not accepted:
            if reason!='krylov_budget':reason='line_search_failed'
            break
    workspace.mode.fill_(1)
    # No head/source/storage/scaling mutation follows the accepted evaluation.
    # Rejected trials have separate flux, derivative and storage buffers.
    if not controls.reuse_trial_state:current=workspace.evaluate(trial=False,linearize=True)
    if wp.get_device(ident=model.device).is_cuda:wp.synchronize_device(device=model.device)
    seconds=perf_counter()-start
    head=None if device_sequence else workspace.head.numpy().reshape(case.active.shape).copy()
    device_passed=device_accepted(metrics=current,physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance)
    certificate=workspace.final_certificate() if device_native else None
    independent_performed=not device_native or not certificate['all_free_cells_storage_anchored']
    if independent_performed:
        if head is None:
            head=workspace.head.numpy().reshape(case.active.shape).copy()
        checks,unused=reference.evaluate(head=head,physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance,
            **(dict(roundoff_acceptance=True) if roundoff_acceptance else {}))
        accepted=checks['independent_accepted']
    else:
        accepted=bool(device_passed and certificate['derivatives_and_storage_finite']
            and certificate['prescribed_head_error_max_m']<=1e-10)
        checks=dict(**current,**certificate,ungrounded_current_components=0,
            accepted=accepted,acceptance_source='device physical reductions and positive storage grounding certificate')
    converged=bool(accepted and device_passed)
    requested_met=bool(current['finite'] and current['head_residual_rms']<=1e-6 and
        (current['physical_rms']<=physical_tolerance or current['relative_cell_imbalance']<=relative_tolerance))
    if roundoff_acceptance:
        if independent_performed:requested_met=bool(requested_met and checks['requested_thresholds_met'])
        checks.update(requested_thresholds_met=requested_met,precision_limited=bool(converged and not requested_met))
    workspace.predictor_checkpoint = None
    if controls.timestep_predictor and workspace.transient and converged:
        wp.copy(dest=workspace.predictor_older_increment,src=workspace.predictor_increment)
        wp.launch(kernel=kernels.store_head_increment,dim=workspace.n,
            inputs=[workspace.head,workspace.storage_previous,workspace.free,workspace.predictor_increment],device=workspace.device)
        workspace.predictor_checkpoint = (None if device_sequence else head.copy(), reference.dt,
            bool(predictor_continuation and reference.dt == checkpoint[1]))
    if not device_passed and accepted:reason='device_acceptance_failed'
    if not converged and reason=='converged':reason='independent_acceptance_failed' if independent_performed else 'device_certificate_failed'
    if checks['ungrounded_current_components']>0:reason='ungrounded_saturation_components'
    result=dict(converged=converged,stop_reason='converged' if converged else reason,outer_iterations=len(history),
        krylov_iterations=total,jacobian_vector_products=int(workspace.jv_counter.numpy()[0]),
        nonlinear_residual_evaluations=workspace.residual_evaluations,candidate_evaluations=workspace.candidate_evaluations,
        nonlinear_backtracks=backtracks,fallback_count=fallbacks,outer_history=history,
        independent_acceptance=checks if independent_performed else None,physical_acceptance=checks,
        independent_audit_performed=independent_performed,device_native=device_native,tiled_reductions=tiled,
        full_saturated_safeguard_solves=saturated,timestep_predictor_used=predictor_used,
        device_final_metrics=current,workspace_reused=reused,seconds=seconds,setup_seconds=setup_seconds,controls=asdict(controls),
        linear_budget_units='Krylov iterations shared across Newton and Picard per outer',
        max_krylov_per_outer=krylov_budget,max_outer=max_outer,krylov_restart=restart,
        scaling='fixed full-saturated principal diagonal; S*J and S*R',preconditioner='fixed diagonal within each GMRES solve')
    result['krylov_method']=method.upper()
    if roundoff_acceptance:
        result.update(roundoff_accepted=converged,requested_thresholds_met=requested_met,
            precision_limited=bool(converged and not requested_met),
            roundoff_bound_ratio_max=current.get('roundoff_bound_ratio_max'))
    result['ghb_budget']=workspace.boundary_budget()
    result['krylov_workspace_reused']=krylov_reused
    result['npf_preconditioner']=npf_preconditioner
    result.update(accepted_updates=sum(row['accepted'] for row in history),
        rejected_updates=sum(not row['accepted'] for row in history),
        rejected_direction_attempts=sum(not attempt.get('search', {}).get('accepted', False) for row in history for attempt in row['attempts']))
    if method=='gmres':
        result.update(krylov_restart_blocks=getattr(workspace.krylov, 'restart_blocks', 0)-blocks_start,
            preconditioner_applications=int(workspace.preconditioner_counter.numpy()[0]), preconditioner_cycles=0,
            cuda_graph_captures=getattr(workspace.krylov, 'graph_captures', 0)-graph_start[0],
            cuda_graph_replays=getattr(workspace.krylov, 'graph_replays', 0)-graph_start[1])
    if method=='fgmres':
        result.update(linear_history=linear_history,krylov_restarts=sum(row['restarts'] for row in linear_history),
            krylov_restart_blocks=sum(row['blocks'] for row in linear_history),
            preconditioner_applications=flex_preconditioner.applications-applications_start,
            preconditioner_cycles=workspace.multigrid.cycles-mg_start[1] if npf_preconditioner=='kcycle' else 0,
            krylov_synchronization_seconds=sum(row['synchronization_seconds'] for row in linear_history),
            krylov_scalar_readbacks=sum(row['scalar_readbacks'] for row in linear_history))
    if npf_preconditioner=='kcycle':
        mg=workspace.multigrid
        result.update(preconditioner='full-saturated A0 K-cycle; physical RHS D0*v; homogeneous corrections',
            linear_history=linear_history,krylov_restarts=sum(row['restarts'] for row in linear_history),
            multigrid=dict(controls=asdict(mg.controls),smoother='vertical_line',
                coefficient_state=mg.controls.coefficient_state,precision=mg.controls.preconditioner_precision,
                dimensions=[list(level['dim']) for level in mg.levels],actual_levels=len(mg.levels),
                coarsening_factors=[[level['coarsen_y'],level['coarsen_x']] for level in mg.levels[:-1]],
                construction_seconds=mg_seconds,hierarchy_reused=mg_reused,
                coefficients_refreshed=mg_values_refreshed,value_refresh_count=mg.value_refresh_count,
                device_storage_refresh_count=mg.device_storage_refresh_count,
                refresh_graph_captures=len(mg.refresh_graphs),refresh_graph_replays=mg.refresh_graph_replays,
                refresh_graph_fallback=mg.refresh_graph_failed,
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
