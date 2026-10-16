"""Safeguarded steady terrain Newton--Krylov with a Picard MG preconditioner.

R=source-div(q); A=-J_R=D(div(q)), b=R. Newton GMRES uses the
unscaled physical equations (S=I), so its forcing controls physical error.
Final head-equivalent norms still use the original principal diagonal.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator

from DARCY_WARP_PACKAGE.solvers.terrain_device_3d import DeviceTerrainPicard
from DARCY_WARP_PACKAGE.solvers.terrain_device_kernels import terrain_scale_divide_kernel
from DARCY_WARP_PACKAGE.solvers.terrain_gmres_3d import TerrainGMRES3D
from DARCY_WARP_PACKAGE.solvers.terrain_kernels_3d import terrain_gradients_kernel, terrain_faces_kernel
from DARCY_WARP_PACKAGE.solvers.terrain_newton_kernels import (
    project_direction, coefficient_direction, add_face_direction, contact_direction, jacobian_divergence)


@dataclass(frozen=True)
class TerrainNewtonControls:
    """Globalization/forcing choices; none changes the physical acceptance bar."""
    physical_forcing: float = .05
    krylov_fraction: float = .75
    adequate_relaxation: float = .5
    adequate_merit_ratio: float = .25
    retry_contraction: float = .5

    def validate(self):
        for name in ('physical_forcing', 'krylov_fraction', 'adequate_relaxation',
                     'adequate_merit_ratio', 'retry_contraction'):
            value = getattr(self, name)
            if not np.isfinite(value) or not 0 < value < 1:
                raise ValueError(f"{name} must be finite and strictly between zero and one")


class TerrainJacobian3D:
    """Analytic matrix-free derivative with private scratch; no residual calls."""
    def __init__(self, *, driver):
        self.driver = driver
        s = driver.state
        self.device = driver.device
        for name in ('v', 'gx', 'gy', 'gz', 'base_gx', 'base_gy', 'base_gz', 'qx', 'qy', 'qz', 'contact'):
            setattr(self, name, wp.zeros(shape=s.n, dtype=wp.float64, device=self.device))
        self.linear_scale = wp.ones(shape=s.n, dtype=wp.float64, device=self.device)
        self.precondition_input = wp.zeros(shape=s.n, dtype=wp.float64, device=self.device)
        self.dc = wp.zeros(shape=(s.n, 13), dtype=wp.float64, device=self.device)
        self.counter = wp.zeros(shape=1, dtype=wp.int64, device=self.device)
        self.operator = LinearOperator(shape=(s.n, s.n), dtype=wp.float64, device=wp.get_device(ident=self.device), matvec=self.matvec)

    def prepare(self):
        """Called after full accepted-state evaluation, before any Krylov solve."""
        d = self.driver
        s = d.state
        w = d.workspace
        wp.launch(kernel=terrain_gradients_kernel, dim=s.n,
            inputs=[s.head_rel, w.horizontal, w.depths, self.base_gx, self.base_gy, self.base_gz,
                    s.nx, s.ny, s.nz], device=self.device)

    def precondition(self, x, y, z, alpha, beta):
        s = self.driver.state
        wp.launch(kernel=terrain_scale_divide_kernel, dim=s.n,
            inputs=[x, s.scale, self.precondition_input], device=self.device)
        self.driver.workspace.precondition(x=self.precondition_input, y=y, z=z, alpha=alpha, beta=beta)

    def matvec(self, x, y, z, alpha, beta):
        d = self.driver
        s = d.state
        w = d.workspace
        wp.launch(kernel=project_direction, dim=s.n, inputs=[x, s.free, self.v], device=self.device)
        wp.launch(kernel=coefficient_direction, dim=s.n,
            inputs=[s.head, self.v, s.bottoms, s.fraction, s.kx, s.ky, s.kz, s.thickness,
                s.zx_interfaces, s.zy_interfaces, s.zx_centre, s.zy_centre, s.zxf, s.zyf,
                s.active, self.dc, d.min_sat, d.dx, d.dy, s.nx, s.ny, s.nz], device=self.device)
        wp.launch(kernel=terrain_gradients_kernel, dim=s.n,
            inputs=[self.v, w.horizontal, w.depths, self.gx, self.gy, self.gz,
                    s.nx, s.ny, s.nz], device=self.device)
        wp.launch(kernel=terrain_faces_kernel, dim=s.n,
            inputs=[self.v, self.gx, self.gy, self.gz, s.coefficients, self.qx, self.qy, self.qz,
                    s.nx, s.ny, s.nz], device=self.device)
        wp.launch(kernel=add_face_direction, dim=s.n,
            inputs=[s.head_rel, self.base_gx, self.base_gy, self.base_gz, self.dc,
                    self.qx, self.qy, self.qz, s.nx, s.ny, s.nz], device=self.device)
        self.contact.zero_()
        wp.launch(kernel=contact_direction, dim=s.sub_cell.shape[0],
            inputs=[s.sub_cell, s.sub_open, s.sub_c, s.sub_bz, s.sub_bx, s.sub_by, s.sub_hb,
                *s.contact_refresh, s.head, s.head_rel, s.bottoms, s.thickness, s.fraction,
                self.v, self.base_gx, self.base_gy, self.base_gz, self.gx, self.gy, self.gz,
                self.contact, d.min_sat], device=self.device)
        wp.launch(kernel=jacobian_divergence, dim=s.n,
            inputs=[x, y, self.qx, self.qy, self.qz, self.contact, self.linear_scale, s.free, z,
                    float(alpha), float(beta), self.counter, s.nx, s.ny], device=self.device)


class DeviceTerrainNewton(DeviceTerrainPicard):
    """Full-step Newton trial; unchanged dual merit/recovery; Picard fallback.

    Each Newton attempt may use at most 3/4 of the outer budget, reserving
    at least one restart (when possible) for fallback. Unused work remains
    available to Picard. A weak model/linear solve triggers Picard; Newton
    retries after physical RMS halves, rather than on a grid-specific schedule. Neither line search nor fallback gets a new budget.
    """
    def __init__(self, *, newton_controls=None, **kwargs):
        super().__init__(**kwargs)
        if self.transient or not self.unconfined:
            raise ValueError("terrain Newton supports steady unconfined solves only")
        self.newton_controls = TerrainNewtonControls() if newton_controls is None else newton_controls
        self.newton_controls.validate()
        self.jacobian = TerrainJacobian3D(driver=self)
        self.newton_krylov = TerrainGMRES3D(A=self.jacobian.operator, b=self.state.rhs,
            x=self.state.delta, M=LinearOperator(shape=(self.state.n,self.state.n),dtype=wp.float64,device=wp.get_device(ident=self.device),matvec=self.jacobian.precondition), restart=self.requested_restart,
            maxiter=max(self.krylov_budget, 1), tol=0., atol=self.abs_tol_min, check_every=self.restart)
        self.active_krylov = self.newton_krylov
        self.saved_newton_head = wp.zeros_like(src=self.state.head)
        self.saved_newton_delta = wp.zeros_like(src=self.state.delta)

    def krylov_block(self, *, budget, target_rms):
        krylov = self.active_krylov
        krylov._maxiter = int(budget)
        krylov._tol = 0.0
        krylov._atol = max(target_rms*self.sqrt_free, np.finfo(float).tiny)
        used, error, tolerance = krylov()
        return used, error/self.sqrt_free, tolerance/self.sqrt_free

    def search(self, *, current, newton):
        initial = 1.0 if newton or self.outer_limit_val == 1 else self.unconfined_relax
        update = self.backtrack(current=current, initial_relaxation=initial)
        if not update['accepted']:
            trials = update['trials']
            update = self.backtrack(current=current, allow_growth=True, initial_relaxation=initial)
            update['trials'] = trials+update['trials']
        return update

    def run(self, max_outer):
        s = self.state
        history = []
        self._eta = self._eta_residual = None
        previous_dh = np.inf
        total_krylov = 0
        stop_reason = 'outer_budget'
        converged = False
        start = perf_counter()
        c = self.newton_controls
        original_budget = self.krylov_budget
        picard_reads_before = getattr(self.workspace.krylov, 'norm_readbacks', 0)
        retry_flow = np.inf
        for outer in range(max_outer):
            self.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
            self.physical_residual(head=s.head_rel)
            combined = self.dual_check(head_buffer=s.head_rel, candidate_buffer=s.head_rel)
            current = self.residual_metrics(combined=combined)
            if self.physically_converged(metrics=current):
                converged = True
                stop_reason = 'converged'
                break
            if not current['finite']:
                stop_reason = 'nonfinite_residual'
                break
            wp.launch(kernel=terrain_scale_divide_kernel, dim=s.n, inputs=[s.residual, s.scale, s.rhs], device=self.device)
            wp.copy(src=s.rhs, dest=s.rhs_original)
            attempt_newton = current["flow"] <= retry_flow
            wp.copy(src=s.residual, dest=s.rhs)
            if attempt_newton:
                self.jacobian.prepare()
            s.delta.zero_()
            self.active_krylov = self.newton_krylov
            reserve = min(self.restart, original_budget)
            self.krylov_budget = min(max(original_budget-reserve, 0), int(c.krylov_fraction*original_budget))
            target = self.inner_target(head_rms=current['head'], previous_dh_rms=previous_dh)
            if attempt_newton:
                outcome = self.run_inner(initial_rms=current['flow'], target_rms=c.physical_forcing*current['flow'])
            else:
                outcome = dict(cycles=0,restart_blocks=0,reached=False,usable=False,retried=False,
                    stagnated=False,diverged=False,head_rms=current['flow'],target_rms=c.physical_forcing*current['flow'])
            newton_work = outcome['cycles']
            update = dict(accepted=False, metrics=current, relaxation=0., dh_rms=0., dh_max=0.,
                proposed_max=0., clipped_max=0., trials=[], acceptance_mode='rejected')
            if outcome['usable'] or outcome['reached']:
                update = self.search(current=current, newton=True)
            newton_update = update
            newton_outcome = outcome.copy()
            newton_trials = update['trials']
            # Prefer safeguarded Picard when status or severe damping exposes
            # a weak Newton direction. Retain Newton if Picard also fails.
            old_merit, new_merit = self.residual_merit(current=current, candidate=update['metrics'])
            fallback = (not self.physically_converged(metrics=update['metrics']) and
                (not outcome['reached'] or not update['accepted'] or update['relaxation'] < c.adequate_relaxation
                 or new_merit > c.adequate_merit_ratio*old_merit))
            if attempt_newton:
                retry_flow = c.retry_contraction*current['flow'] if fallback else np.inf
            if fallback and update['accepted']:
                wp.copy(src=s.head, dest=self.saved_newton_head)
                wp.copy(src=s.delta, dest=self.saved_newton_delta)
                wp.copy(src=s.starting, dest=s.head)
                self.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
                self.physical_residual(head=s.head_rel)
            method = 'newton'
            picard_work = 0
            picard_blocks = 0
            picard = None
            if fallback and newton_work < original_budget:
                # Rejected search restores the complete accepted operator/hierarchy.
                # No search was run for an unusable direction, so it is still accepted.
                wp.copy(src=s.rhs_original, dest=s.rhs)
                s.delta.zero_()
                self.active_krylov = self.workspace.krylov
                self.krylov_budget = original_budget-newton_work
                picard = self.run_inner(initial_rms=current['head'], target_rms=target)
                picard_work = picard['cycles']
                picard_blocks = picard['restart_blocks']
                if picard['usable'] or picard['reached']:
                    update = self.search(current=current, newton=False)
                    if not update['accepted'] and picard['cycles'] < self.krylov_budget:
                        previous_trials = update['trials']
                        wp.copy(src=s.rhs_original, dest=s.rhs)
                        picard = self.refine_inner(outcome=picard, initial_rms=current['head'],
                            target_rms=min(.1*picard['head_rms'], self.controls.residual_floor))
                        if picard['usable']:
                            update = self.search(current=current, newton=False)
                            update['trials'] = previous_trials+update['trials']
                        picard_work = picard['cycles']
                        picard_blocks = picard['restart_blocks']
                    update['trials'] = newton_trials+update['trials']
                outcome = picard
                method = 'picard_fallback'
                if newton_update['accepted']:
                    if not (picard['usable'] or picard['reached']) or not update['accepted']:
                        all_trials = update['trials']
                        update = newton_update
                        update['trials'] = all_trials
                        wp.copy(src=self.saved_newton_head, dest=s.head)
                        wp.copy(src=self.saved_newton_delta, dest=s.delta)
                        self.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
                        self.physical_residual(head=s.head_rel)
                        outcome = newton_outcome
                        method = 'newton'
            used = newton_work+picard_work
            total_krylov += used
            self.krylov_budget = original_budget
            history.append(dict(outer_iteration=outer+1, direction=method, newton_attempted=attempt_newton,
                krylov_iterations=used, newton_krylov_iterations=newton_work, picard_krylov_iterations=picard_work,
                restart_blocks=newton_outcome['restart_blocks']+picard_blocks,
                inner_reached=outcome['reached'], inner_usable=outcome['usable'], inner_retried=outcome['retried'],
                inner_stagnated=outcome['stagnated'], inner_diverged=outcome['diverged'],
                inner_residual_rms=outcome['head_rms'], inner_target_rms=outcome['target_rms'], forcing_target_rms=outcome['target_rms'],
                inner_residual_units='m3/day' if method == 'newton' else 'm',
                newton_inner=newton_outcome if attempt_newton else None, picard_inner=picard,
                picard_forcing_target_rms=target,
                initial_head_residual_rms=current['head'], initial_physical_residual_rms=current['flow'],
                head_residual_rms=update['metrics']['head'], physical_residual_rms=update['metrics']['flow'],
                relative_cell_imbalance=update['metrics']['relative'],
                proposed_head_change_max=update['proposed_max'], clipped_head_change_max=update['clipped_max'],
                accepted_head_change_rms=update['dh_rms'], accepted_head_change_max=update['dh_max'],
                relaxation=update['relaxation'], accepted=update['accepted'], acceptance_mode=update['acceptance_mode'],
                fallback_attempted=fallback and picard_work > 0,
                backtracking_trials=update['trials'], backtracking_evaluations=len(update['trials']),
                backtracks=len(update['trials'])-int(update['accepted'])))
            previous_dh = update['dh_rms']
            if self.physically_converged(metrics=update['metrics']):
                converged = True
                stop_reason = 'converged'
                break
            if not update['accepted']:
                stop_reason = 'line_search_failed' if outcome['usable'] else 'linear_solve_failed'
                break
        # Leave canonical accepted-state fields even after a zero-budget or
        # unusable linear solve (whose working RHS was a correction residual).
        self.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
        self.physical_residual(head=s.head_rel)
        norm_readbacks = (getattr(self.newton_krylov, 'norm_readbacks', 0)
            +getattr(self.workspace.krylov, 'norm_readbacks', 0)-picard_reads_before)
        return dict(converged=converged, outer_iterations=len(history), krylov_iterations=total_krylov,
            seconds=perf_counter()-start, host_scalar_reads=self.host_scalar_reads+norm_readbacks+1,
            host_outer_reductions=self.host_scalar_reads,
            gmres_norm_readbacks=norm_readbacks,
            rejected=sum(not row['accepted'] for row in history), operator_evaluations=self.operator_evaluations,
            stop_reason=stop_reason, history=history, jacobian_vector_products=int(self.jacobian.counter.numpy()[0]),
            fallback_count=sum(row['fallback_attempted'] for row in history),
            candidate_evaluations=sum(row['backtracking_evaluations'] for row in history))

    def close(self):
        """Release graph callbacks before the solve-local state is closed."""
        self.newton_krylov._cycle_graph = None
        self.newton_krylov._A = None
        self.newton_krylov._M = None
        self.jacobian.driver = None
        self.active_krylov = None
