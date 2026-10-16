"""Device-resident 3D terrain Picard driver (2D fast-unconfined contract).

The nonlinear outer loop keeps the total head, transformed coefficients,
secant storage, RHS, corrections and residual on the device: per outer
iteration it refreshes saturation, the packed 13-column face operator and
the storage diagonals with the terrain device kernels, freezes the
discharge-only gates at the accepted head, extracts the seven-point
principal part into the multigrid level-0 buffers, refreshes head-dependent contact conductances and rebuilds the row-scaled
RHS, and runs restarted GMRES in CORRECTION FORM (A·delta = b − A·h with
homogeneous prescribed-head constraints) under an adaptive inexact inner
controller modelled on the production 2D driver:

* fixed-work restart-cycle blocks with dual-residual contraction
  classification, rollback of diverging blocks and stagnation detection;
* Eisenstat-Walker-style forcing targets that loosen early and tighten
  as the nonlinear residual contracts;
* acceptance of usable inexact corrections, so a stiff frozen system
  cannot burn its whole Krylov budget on every outer iteration (the
  measured host-path stagnation mechanism).

Damped clipped updates and snapshot rollback of rejected corrections run
on device; only a handful of scalars cross to the host per outer
iteration. Final acceptance is independently reconstructed on the host
from the completed physical state by the session, never from the frozen
inner residual.

The host-driven path in terrain_fast_3d.TerrainOperatorSession3D.solve
remains the reference implementation and the default; this driver is
selected explicitly with device_picard=True.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.solvers.terrain_device_kernels import (
    terrain_saturation_kernel, terrain_face_coefficients_kernel,
    terrain_secant_storage_kernel, terrain_rhs_kernel, terrain_relaxed_update_kernel,
    terrain_dual_check_kernel, terrain_combine_partials_kernel,
    terrain_relative_head_kernel, terrain_extract_principal_kernel,
    terrain_subface_gate_kernel, terrain_subface_scatter_kernel,
    terrain_coarse_faces_kernel, terrain_coarse_shifts_kernel,
    terrain_scale_divide_kernel,
    terrain_contact_refresh_kernel,
    terrain_contact_flux_kernel, terrain_physical_residual_kernel, BLOCK_SIZE)
from DARCY_WARP_PACKAGE.solvers.terrain_kernels_3d import (
    terrain_gradients_kernel, terrain_faces_kernel)


@dataclass
class DevicePicardControls:
    """Adaptive inner and nonlinear controls (2D production defaults)."""
    initial_block_cycles: int = 2          # restart cycles per inner block
    min_block_cycles: int = 1
    max_block_cycles: int = 4
    eta_initial: float = 0.05
    eta_min: float = 0.005
    eta_max: float = 0.10
    eta_gamma: float = 0.25
    eta_power: float = 1.5
    good_contraction_ratio: float = 0.40
    weak_contraction_ratio: float = 0.90
    stall_contraction_ratio: float = 0.9995
    divergence_ratio: float = 1.10
    stall_patience: int = 3
    nonlinear_growth_limit: float = 1.10  # recovery only after descent search fails
    max_head_change_per_outer: float = 50.0
    residual_floor: float = 1e-12


class DeviceTerrainState:
    """Persistent device buffers for the terrain Picard loop."""
    def __init__(self, *, session, device, workspace):
        shape = session.grid.shape
        nz, ny, nx = shape
        self.shape = shape
        self.nz, self.ny, self.nx = nz, ny, nx
        self.n = int(np.prod(shape))
        self.device = device
        grid = session.grid
        metrics = session.metrics

        def f64(array):
            return wp.array(np.ascontiguousarray(array, dtype=np.float64).ravel(),
                dtype=wp.float64, device=device)

        # static geometry (refreshed only on geometry revision)
        self.bottoms = f64(grid.bottoms)
        self.thickness = f64(grid.thickness)
        self.zx_interfaces = f64(metrics.zx_interfaces)
        self.zy_interfaces = f64(metrics.zy_interfaces)
        self.zx_centre = f64(metrics.zx)
        self.zy_centre = f64(metrics.zy)
        zxf_full = np.zeros(shape); zxf_full[:, :, :-1] = metrics.zxf
        zyf_full = np.zeros(shape); zyf_full[:, :-1, :] = metrics.zyf
        self.zxf = f64(zxf_full)
        self.zyf = f64(zyf_full)
        self.active = workspace.active
        self.fixed = workspace.fixed
        # adopt the workspace buffers the Krylov matvec reads
        self.coefficients = workspace.coefficients
        self.storage = workspace.storage
        self.boundary = workspace.boundary
        self.scale = workspace.scale
        self.delta = workspace.x
        self.rhs = workspace.b
        self.kx = f64(session.kx)
        self.ky = f64(session.ky)
        self.kz = f64(session.kz)
        # dynamic nonlinear fields
        self.head = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.previous = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.candidate = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.head_rel = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.candidate_rel = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.delta_snapshot = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.rhs_original = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.rhs_raw = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.fraction = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.true_sat = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.storage_diag = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.storage_change = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.residual = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.bc_values = f64(session.bc_values)
        self.open_col0 = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.contact_net = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.contact_gross = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.throughput = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.contact_rhs = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.starting = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.free = wp.zeros(self.n, dtype=wp.int32, device=device)
        blocks = (self.n + BLOCK_SIZE - 1)//BLOCK_SIZE
        self.partials = wp.zeros((blocks, 8), dtype=wp.float64, device=device)
        self.blocks = blocks
        self.combined = wp.zeros(8, dtype=wp.float64, device=device)

    def upload_static(self, *, source, sy, ss, free, subfaces, contact_refresh=None):
        self.source = wp.array(np.ascontiguousarray(source, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.sy = wp.array(np.ascontiguousarray(sy, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.ss = wp.array(np.ascontiguousarray(ss, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.free.assign(np.ascontiguousarray(free.ravel(), dtype=np.int32))
        self.n_free = int(np.count_nonzero(free))
        # Per-SUBFACE contact rows: gating is evaluated independently for
        # each subface and only the open ones are aggregated per outer.
        self.sub_cell = wp.array(np.ascontiguousarray(subfaces[:, 0], dtype=np.int32),
            dtype=wp.int32, device=self.device)
        self.sub_c = wp.array(np.ascontiguousarray(subfaces[:, 1], dtype=np.float64),
            dtype=wp.float64, device=self.device)
        self.sub_bz = wp.array(np.ascontiguousarray(subfaces[:, 2], dtype=np.float64),
            dtype=wp.float64, device=self.device)
        self.sub_bx = wp.array(np.ascontiguousarray(subfaces[:, 3], dtype=np.float64),
            dtype=wp.float64, device=self.device)
        self.sub_by = wp.array(np.ascontiguousarray(subfaces[:, 4], dtype=np.float64),
            dtype=wp.float64, device=self.device)
        self.sub_hb = wp.array(np.ascontiguousarray(subfaces[:, 5], dtype=np.float64),
            dtype=wp.float64, device=self.device)
        self.sub_discharge = wp.array(np.ascontiguousarray(subfaces[:, 6], dtype=np.int32),
            dtype=wp.int32, device=self.device)
        self.sub_open = wp.zeros(subfaces.shape[0], dtype=wp.int32, device=self.device)
        self.contact_refresh = None
        if contact_refresh is not None:
            self.contact_refresh = tuple(wp.array(np.ascontiguousarray(array),
                dtype=wp.float64, device=self.device) for array in contact_refresh)

    def update_head(self, host_head):
        self.head.assign(np.ascontiguousarray(host_head, dtype=np.float64).ravel())

    def update_previous(self, host_previous):
        self.previous.assign(np.ascontiguousarray(host_previous, dtype=np.float64).ravel())

    def refresh_geometry(self, session):
        grid = session.grid
        metrics = session.metrics
        self.bottoms.assign(np.ascontiguousarray(grid.bottoms, dtype=np.float64).ravel())
        self.thickness.assign(np.ascontiguousarray(grid.thickness, dtype=np.float64).ravel())
        self.zx_interfaces.assign(np.ascontiguousarray(metrics.zx_interfaces, dtype=np.float64).ravel())
        self.zy_interfaces.assign(np.ascontiguousarray(metrics.zy_interfaces, dtype=np.float64).ravel())
        self.zx_centre.assign(np.ascontiguousarray(metrics.zx, dtype=np.float64).ravel())
        self.zy_centre.assign(np.ascontiguousarray(metrics.zy, dtype=np.float64).ravel())
        shape = self.shape
        zxf_full = np.zeros(shape); zxf_full[:, :, :-1] = metrics.zxf
        zyf_full = np.zeros(shape); zyf_full[:, :-1, :] = metrics.zyf
        self.zxf.assign(np.ascontiguousarray(zxf_full, dtype=np.float64).ravel())
        self.zyf.assign(np.ascontiguousarray(zyf_full, dtype=np.float64).ravel())

    def close(self):
        for name in ("head", "previous", "candidate", "head_rel", "candidate_rel",
                     "delta_snapshot", "rhs_original", "rhs_raw", "fraction", "true_sat",
                     "storage_diag", "storage_change", "residual", "bc_values",
                     "free", "partials", "combined", "bottoms", "thickness",
                     "zx_interfaces", "zy_interfaces", "zx_centre", "zy_centre",
                     "zxf", "zyf", "kx", "ky", "kz", "source", "sy", "ss",
                     "open_col0", "contact_rhs", "contact_net", "contact_gross", "throughput",
                     "contact_refresh", "starting", "sub_cell", "sub_c",
                     "sub_bz", "sub_bx", "sub_by", "sub_hb", "sub_discharge",
                     "sub_open"):
            setattr(self, name, None)


class DeviceTerrainPicard:
    """One solve invocation of the device-resident Picard loop."""
    def __init__(self, *, session, state, workspace, device, controls, datum,
                 restart, krylov_budget, unconfined, transient, dt,
                 abs_tol_min, head_tolerance, budget_relative_tol, rel_tol,
                 min_sat, outer_limit, unconfined_relax):
        self.session = session
        self.state = state
        self.workspace = workspace
        self.device = device
        self.controls = controls
        self.datum = float(datum)
        self.min_sat = float(min_sat)
        self.outer_limit_val = int(outer_limit)
        self.unconfined_relax = float(unconfined_relax)
        self.requested_restart = int(restart)
        # Warp caps the allocated restart at its initial maxiter. Count
        # the actual blocks used by this persistent Krylov workspace.
        self.restart = int(workspace.krylov._restart)
        self.krylov_budget = krylov_budget
        self.unconfined = unconfined
        self.transient = transient
        self.dt = dt
        self.abs_tol_min = abs_tol_min
        self.head_tolerance = head_tolerance
        self.budget_relative_tol = budget_relative_tol
        self.rel_tol = rel_tol
        self.n_free = state.n_free
        self.sqrt_free = float(np.sqrt(max(self.n_free, 1)))
        self.host_scalar_reads = 0
        self.operator_evaluations = 0
        self.dx = float(session.grid.dx)
        self.dy = float(session.grid.dy)

    # ---- device-side operator evaluation --------------------------------
    def evaluate(self, head_buffer, rel_buffer):
        """Full nonlinear evaluation at the given head: saturation,
        transformed coefficients, secant storage, per-subface discharge
        gates with aggregation, level-0 extraction, the FULL coarse
        hierarchy refresh, and the row-scaled RHS. The accepted head_rel
        buffer is never touched when evaluating a candidate."""
        s = self.state
        self.operator_evaluations += 1
        wp.launch(terrain_saturation_kernel, dim=s.n,
            inputs=[head_buffer, s.bottoms, s.thickness, wp.float64(self.min_sat),
                    s.true_sat, s.fraction], device=self.device)
        wp.launch(terrain_face_coefficients_kernel, dim=s.n,
            inputs=[s.fraction, s.kx, s.ky, s.kz, s.thickness,
                    s.zx_interfaces, s.zy_interfaces, s.zx_centre, s.zy_centre,
                    s.zxf, s.zyf, s.active, s.coefficients,
                    wp.float64(self.dx), wp.float64(self.dy), s.nx, s.ny, s.nz],
            device=self.device)
        if self.transient and self.unconfined:
            wp.launch(terrain_secant_storage_kernel, dim=s.n,
                inputs=[head_buffer, s.previous, s.bottoms, s.thickness, s.sy, s.ss,
                        s.free, wp.float64(self.dx), wp.float64(self.dy),
                        wp.float64(self.dt), s.storage_diag, s.storage_change],
                device=self.device)
        else:
            s.storage_diag.zero_()
            s.storage_change.zero_()
        # The adopted Krylov matvec reads workspace.storage: keep it
        # identical to the freshly computed diagonals.
        wp.copy(src=s.storage_diag, dest=s.storage)
        wp.launch(terrain_relative_head_kernel, dim=s.n,
            inputs=[head_buffer, rel_buffer, wp.float64(self.datum), s.active],
            device=self.device)
        wp.launch(terrain_gradients_kernel, dim=s.n,
            inputs=[rel_buffer, self.workspace.horizontal, self.workspace.depths,
                    self.workspace.gx, self.workspace.gy, self.workspace.gz,
                    s.nx, s.ny, s.nz], device=self.device)
        # Refresh the constitutive contact rows before evaluating their gates.
        if s.contact_refresh is not None:
            wp.launch(terrain_contact_refresh_kernel, dim=s.sub_open.shape[0],
                inputs=[s.sub_cell, s.fraction, *s.contact_refresh,
                        s.sub_c, s.sub_bz, s.sub_bx, s.sub_by], device=self.device)
        # Per-subface independent gating, then aggregation of the open set.
        s.boundary.zero_()
        s.contact_rhs.zero_()
        s.open_col0.zero_()
        wp.launch(terrain_subface_gate_kernel, dim=s.sub_open.shape[0],
            inputs=[s.sub_cell, s.sub_c, s.sub_bz, s.sub_bx, s.sub_by, s.sub_hb,
                    s.sub_discharge, self.workspace.gx, self.workspace.gy,
                    self.workspace.gz, rel_buffer, s.sub_open], device=self.device)
        wp.launch(terrain_subface_scatter_kernel, dim=s.sub_open.shape[0],
            inputs=[s.sub_cell, s.sub_c, s.sub_bz, s.sub_bx, s.sub_by, s.sub_hb,
                    s.sub_open, s.boundary, s.contact_rhs, s.open_col0],
            device=self.device)
        # Level-0 principal part with the OPEN contact conductances riding
        # on the storage slot (preconditioner approximation, as on host).
        level0 = self.workspace.levels[0]
        wp.launch(terrain_extract_principal_kernel, dim=s.n,
            inputs=[s.coefficients, s.active, s.fixed, s.storage_diag, s.open_col0,
                    level0["faces"][0], level0["faces"][1], level0["faces"][2],
                    level0["faces"][3], level0["faces"][4], level0["faces"][5],
                    level0["diag"], level0["storage"], s.scale, s.nx, s.ny, s.nz],
            device=self.device)
        # Full coarse refresh (2D fast-path contract): every coarse level
        # recomputed from its fine level whenever saturation changes.
        for level_index in range(len(self.workspace.levels)-1):
            fine = self.workspace.levels[level_index]
            coarse = self.workspace.levels[level_index+1]
            for array in (*coarse["faces"], coarse["storage"]):
                array.zero_()
            wp.launch(terrain_coarse_faces_kernel, dim=fine["dim"],
                inputs=[fine["faces"][0], fine["faces"][1], fine["faces"][2],
                        fine["faces"][3], fine["faces"][4], fine["faces"][5],
                        fine["storage"], fine["active"], fine["bc_mask"],
                        coarse["faces"][0], coarse["faces"][1], coarse["faces"][2],
                        coarse["faces"][3], coarse["faces"][4], coarse["faces"][5],
                        coarse["storage"], coarse["active"], coarse["bc_mask"],
                        fine["coarsen_y"], fine["coarsen_x"],
                        fine["nx"], fine["ny"], fine["nz"]], device=self.device)
            wp.launch(terrain_coarse_shifts_kernel, dim=coarse["dim"],
                inputs=[coarse["faces"][0], coarse["faces"][1], coarse["faces"][2],
                        coarse["faces"][3], coarse["faces"][4], coarse["faces"][5],
                        coarse["storage"], coarse["active"], coarse["bc_mask"],
                        coarse["diag"]], device=self.device)
        wp.launch(terrain_rhs_kernel, dim=s.n,
            inputs=[s.rhs, s.source, s.storage_diag, s.previous,
                    wp.float64(self.datum), s.fixed, s.bc_values,
                    s.contact_rhs, s.boundary, s.active, s.scale],
            device=self.device)

    def dual_check(self, head_buffer, candidate_buffer):
        s = self.state
        s.partials.zero_()
        wp.launch(terrain_dual_check_kernel, dim=s.n,
            inputs=[head_buffer, candidate_buffer, s.residual, s.throughput,
                    s.scale, s.free, s.delta, s.partials, BLOCK_SIZE], device=self.device)
        wp.launch(terrain_combine_partials_kernel, dim=1,
            inputs=[s.partials, s.combined, s.blocks], device=self.device)
        self.host_scalar_reads += 1
        return s.combined.numpy()

    def physical_residual(self, *, head):
        """Evaluate original forcing minus face fluxes and exact storage.

        This never reconstructs forcing from a rounded storage RHS and
        never uses GMRES's frozen-system residual for nonlinear acceptance.
        evaluate() must first refresh the state and gates at this head.
        """
        s = self.state
        w = self.workspace
        wp.launch(terrain_gradients_kernel, dim=s.n,
            inputs=[head, w.horizontal, w.depths, w.gx, w.gy, w.gz,
                    s.nx, s.ny, s.nz], device=self.device)
        wp.launch(terrain_faces_kernel, dim=s.n,
            inputs=[head, w.gx, w.gy, w.gz, s.coefficients, w.qx, w.qy, w.qz,
                    s.nx, s.ny, s.nz], device=self.device)
        s.contact_net.zero_()
        s.contact_gross.zero_()
        wp.launch(terrain_contact_flux_kernel, dim=s.sub_open.shape[0],
            inputs=[s.sub_cell, s.sub_c, s.sub_bz, s.sub_bx, s.sub_by, s.sub_hb,
                    s.sub_open, head, w.gx, w.gy, w.gz, s.contact_net,
                    s.contact_gross], device=self.device)
        wp.launch(terrain_physical_residual_kernel, dim=s.n,
            inputs=[s.source, s.storage_change, s.free, s.fixed, w.qx, w.qy,
                    w.qz, s.contact_net, s.contact_gross, s.residual,
                    s.throughput, s.nx, s.ny, s.nz], device=self.device)

    def residual_metrics(self, *, combined):
        n = max(self.n_free, 1)
        return dict(head=float(np.sqrt(combined[3]/n)),
            flow=float(np.sqrt(combined[2]/n)),
            relative=float(combined[6]/max(combined[4], np.finfo(float).tiny)),
            finite=bool(combined[5] == 0 and np.all(np.isfinite(combined))))

    def physically_converged(self, *, metrics):
        return metrics["finite"] and metrics["head"] <= self.head_tolerance and (
            metrics["flow"] <= self.abs_tol_min
            or metrics["relative"] <= self.budget_relative_tol)

    def candidate_improves(self, *, current, candidate):
        """Decrease a balanced merit of BOTH independently evaluated norms.

        Freeze the normalization at the accepted state for every candidate.
        Requiring separate descent of both norms can reject a useful Picard
        correction when one norm grows slightly and the other drops sharply.
        Final physical convergence still uses the unchanged absolute gates.
        """
        if not candidate["finite"]:
            return False
        old, new = self.residual_merit(current=current, candidate=candidate)
        return new < old or self.physically_converged(metrics=candidate)

    def residual_merit(self, *, current, candidate):
        head_scale = max(current["head"], self.head_tolerance)
        flow_scale = max(current["flow"], self.abs_tol_min, np.finfo(float).tiny)
        old = (current["head"]/head_scale)**2+(current["flow"]/flow_scale)**2
        head_ratio = candidate["head"]/head_scale
        flow_ratio = candidate["flow"]/flow_scale
        safe_ratio = 0.5*np.sqrt(np.finfo(float).max)
        if max(abs(head_ratio), abs(flow_ratio)) > safe_ratio:
            return old, np.inf
        new = head_ratio**2+flow_ratio**2
        return old, new

    # ---- adaptive inner controller -------------------------------------
    def inner_target(self, *, head_rms, previous_dh_rms):
        c = self.controls
        if self._eta is None:
            eta = c.eta_initial
        else:
            ratio = min(head_rms/max(self._eta_residual, c.residual_floor), 1e3)
            eta = float(np.clip(c.eta_gamma*ratio**c.eta_power, c.eta_min, c.eta_max))
        self._eta = eta
        self._eta_residual = head_rms
        target = min(eta*head_rms, 1e-2,
            0.10*max(self.head_tolerance, previous_dh_rms))
        # Do not ask for a tolerance larger than an already tiny residual.
        return min(max(target, min(c.residual_floor, 0.1*head_rms)), 0.5*head_rms)

    def krylov_block(self, *, budget, target_rms):
        """The only RMS/L2 conversion boundary; budget is Krylov iterations."""
        krylov = self.workspace.krylov
        krylov._maxiter = int(budget)
        krylov._tol = 0.0
        krylov._atol = max(target_rms*self.sqrt_free, np.finfo(float).tiny)
        used, error_l2, tolerance_l2 = krylov()
        return used, error_l2/self.sqrt_free, tolerance_l2/self.sqrt_free

    def run_inner(self, *, initial_rms, target_rms):
        s = self.state
        c = self.controls
        used_total = 0
        blocks = 0
        block_cycles = c.initial_block_cycles
        patience = 0
        error = initial_rms
        reached = error <= target_rms
        stagnated = diverged = retried = False
        while used_total < self.krylov_budget and not reached:
            wp.copy(src=s.delta, dest=s.delta_snapshot)
            budget = min(block_cycles*self.restart, self.krylov_budget-used_total)
            used, trial, tolerance = self.krylov_block(budget=budget, target_rms=target_rms)
            used_total += used
            blocks += (used+self.restart-1)//self.restart
            rho = trial/max(error, np.finfo(float).tiny)
            if not np.isfinite(trial) or rho > c.divergence_ratio:
                wp.copy(src=s.delta_snapshot, dest=s.delta)
                diverged = True
                break
            # Retain the best correction, including on a stagnating block.
            if trial > error:
                wp.copy(src=s.delta_snapshot, dest=s.delta)
            else:
                error = trial
            reached = error <= tolerance
            if reached or used == 0:
                break
            per_restart = rho**(self.restart/max(used, 1))
            patience = patience+1 if per_restart >= c.stall_contraction_ratio else 0
            if patience >= c.stall_patience:
                stagnated = True
                break
            if rho <= c.good_contraction_ratio:
                block_cycles = min(2*block_cycles, c.max_block_cycles)
            elif rho > c.weak_contraction_ratio:
                block_cycles = c.min_block_cycles
        outcome = dict(cycles=used_total, restart_blocks=blocks, reached=reached,
            stagnated=stagnated, diverged=diverged, retried=retried,
            usable=bool(np.isfinite(error) and error < initial_rms), head_rms=error,
            target_rms=target_rms)
        # Exhaust the remaining budget before declaring an unusable solve.
        if not outcome["usable"] and not reached:
            outcome = self.refine_inner(outcome=outcome, initial_rms=initial_rms,
                target_rms=target_rms)
        return outcome

    def refine_inner(self, *, outcome, initial_rms, target_rms):
        """Retry from the best correction within this outer iteration's budget.
        Missing a target never discards a useful finite correction.
        """
        remaining = self.krylov_budget-outcome["cycles"]
        if remaining <= 0:
            return outcome
        s = self.state
        wp.copy(src=s.delta, dest=s.delta_snapshot)
        used, trial, tolerance = self.krylov_block(budget=remaining, target_rms=target_rms)
        outcome["cycles"] += used
        outcome["restart_blocks"] += (used+self.restart-1)//self.restart
        outcome["retried"] = True
        outcome["target_rms"] = target_rms
        if np.isfinite(trial) and trial <= outcome["head_rms"]:
            outcome["head_rms"] = trial
        else:
            wp.copy(src=s.delta_snapshot, dest=s.delta)
        outcome["reached"] = outcome["head_rms"] <= tolerance
        outcome["usable"] = bool(np.isfinite(outcome["head_rms"])
            and outcome["head_rms"] < initial_rms)
        return outcome

    def backtrack(self, *, current, allow_growth=False, initial_relaxation=None):
        s = self.state
        relaxation = self.unconfined_relax if self.outer_limit_val > 1 else 1.0
        if initial_relaxation is not None:
            relaxation = initial_relaxation
        wp.copy(src=s.head, dest=s.starting)
        proposed_max = clipped_max = 0.0
        trials = []
        for attempt in range(12):
            wp.copy(src=s.starting, dest=s.candidate)
            wp.launch(terrain_relaxed_update_kernel, dim=s.n,
                inputs=[s.candidate, s.delta, wp.float64(relaxation),
                        wp.float64(self.controls.max_head_change_per_outer),
                        s.fixed, s.bc_values, s.active], device=self.device)
            self.evaluate(head_buffer=s.candidate, rel_buffer=s.candidate_rel)
            self.physical_residual(head=s.candidate_rel)
            combined = self.dual_check(head_buffer=s.head_rel, candidate_buffer=s.candidate_rel)
            trial = self.residual_metrics(combined=combined)
            proposed_max = max(proposed_max, float(combined[7]))
            clipped_max = max(clipped_max, float(combined[1])/relaxation)
            trials.append(dict(relaxation=relaxation, head_residual_rms=trial["head"],
                physical_residual_rms=trial["flow"], finite=trial["finite"]))
            improves = self.candidate_improves(current=current, candidate=trial)
            old_merit, new_merit = self.residual_merit(current=current, candidate=trial)
            recoverable = (allow_growth and trial["finite"] and combined[1] > 0
                and new_merit <= self.controls.nonlinear_growth_limit**2*old_merit)
            if improves or recoverable:
                wp.copy(src=s.candidate, dest=s.head)
                return dict(accepted=True, metrics=trial, relaxation=relaxation,
                    dh_rms=float(np.sqrt(combined[0]/max(self.n_free, 1))),
                    dh_max=float(combined[1]), proposed_max=proposed_max, clipped_max=clipped_max,
                    backtracks=attempt, trials=trials,
                    acceptance_mode="merit_decrease" if improves else "bounded_recovery")
            relaxation *= 0.5
        # Candidate evaluation overwrote coefficients/gates/hierarchy/RHS.
        # Restore every derived field before a retry or returning failure.
        self.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
        self.physical_residual(head=s.head_rel)
        return dict(accepted=False, metrics=current, relaxation=0.0, dh_rms=0.0,
            dh_max=0.0, proposed_max=proposed_max, clipped_max=clipped_max, backtracks=12, trials=trials,
            acceptance_mode="rejected")

    # ---- main loop ------------------------------------------------------
    def run(self, max_outer):
        s = self.state
        self._eta = self._eta_residual = None
        history = []
        total_krylov = rejected = 0
        previous_dh_rms = np.inf
        converged = False
        stop_reason = "outer_budget"
        start = perf_counter()
        norm_reads_before = getattr(self.workspace.krylov, "norm_readbacks", 0)
        current = dict(head=np.inf, flow=np.inf, relative=np.inf, finite=False)
        for outer in range(max_outer):
            self.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
            self.physical_residual(head=s.head_rel)
            combined = self.dual_check(head_buffer=s.head_rel, candidate_buffer=s.head_rel)
            current = self.residual_metrics(combined=combined)
            if self.physically_converged(metrics=current):
                converged = True
                stop_reason = "converged"
                break
            if not current["finite"]:
                stop_reason = "nonfinite_residual"
                break
            wp.launch(terrain_scale_divide_kernel, dim=s.n,
                inputs=[s.residual, s.scale, s.rhs], device=self.device)
            # Candidate evaluations overwrite rhs; keep the correction system
            # for a possible full-effort retry on this same frozen operator.
            wp.copy(src=s.rhs, dest=s.rhs_original)
            s.delta.zero_()
            target = self.inner_target(head_rms=current["head"],
                previous_dh_rms=previous_dh_rms)
            outcome = self.run_inner(initial_rms=current["head"], target_rms=target)
            update = dict(accepted=False, metrics=current, relaxation=0.0,
                dh_rms=0.0, dh_max=0.0, proposed_max=0.0, clipped_max=0.0, backtracks=0, trials=[],
                acceptance_mode="rejected")
            if outcome["usable"] or outcome["reached"]:
                update = self.backtrack(current=current)
                if not update["accepted"]:
                    # Picard directions need not descend the true nonlinear
                    # residual. Prefer descent, then allow a bounded recovery
                    # relative to THIS accepted state's physical dual merit.
                    # This changes globalization, never final tolerances.
                    descent_trials = update["trials"]
                    update = self.backtrack(current=current, allow_growth=True)
                    update["trials"] = descent_trials+update["trials"]
                if not update["accepted"] and outcome["cycles"] < self.krylov_budget:
                    # backtrack restored the accepted operator and hierarchy.
                    previous_trials = update["trials"]
                    wp.copy(src=s.rhs_original, dest=s.rhs)
                    outcome = self.refine_inner(outcome=outcome,
                        initial_rms=current["head"],
                        target_rms=min(0.1*outcome["head_rms"], self.controls.residual_floor))
                    if outcome["usable"]:
                        update = self.backtrack(current=current)
                        if not update["accepted"]:
                            retry_trials = update["trials"]
                            update = self.backtrack(current=current, allow_growth=True)
                            update["trials"] = retry_trials+update["trials"]
                        update["trials"] = previous_trials+update["trials"]
            total_krylov += outcome["cycles"]
            rejected += int(not update["accepted"])
            history.append(dict(outer_iteration=outer+1,
                krylov_iterations=outcome["cycles"], restart_blocks=outcome["restart_blocks"],
                inner_reached=outcome["reached"], inner_stagnated=outcome["stagnated"],
                inner_diverged=outcome["diverged"], inner_usable=outcome["usable"],
                inner_retried=outcome["retried"], inner_residual_rms=outcome["head_rms"],
                inner_target_rms=outcome["target_rms"], forcing_target_rms=target,
                initial_head_residual_rms=current["head"],
                initial_physical_residual_rms=current["flow"],
                proposed_head_change_max=update["proposed_max"],
                clipped_head_change_max=update["clipped_max"],
                accepted_head_change_max=update["dh_max"],
                accepted_head_change_rms=update["dh_rms"],
                head_residual_rms=update["metrics"]["head"],
                physical_residual_rms=update["metrics"]["flow"],
                relative_cell_imbalance=update["metrics"]["relative"],
                accepted=update["accepted"], acceptance_mode=update["acceptance_mode"],
                backtracks=len(update["trials"])-int(update["accepted"]),
                backtracking_evaluations=len(update["trials"]),
                backtracking_trials=update["trials"],
                relaxation=update["relaxation"]))
            current = update["metrics"]
            previous_dh_rms = update["dh_rms"]
            if self.physically_converged(metrics=current):
                converged = True
                stop_reason = "converged"
                break
            if not update["accepted"]:
                stop_reason = "line_search_failed" if outcome["usable"] else "linear_solve_failed"
                break
        norm_readbacks = getattr(self.workspace.krylov, "norm_readbacks", 0)-norm_reads_before
        return dict(converged=converged, outer_iterations=len(history),
            krylov_iterations=total_krylov, seconds=perf_counter()-start,
            host_scalar_reads=self.host_scalar_reads+norm_readbacks,
            host_outer_reductions=self.host_scalar_reads, gmres_norm_readbacks=norm_readbacks,
            rejected=rejected,
            operator_evaluations=self.operator_evaluations, stop_reason=stop_reason,
            history=history, final_head_residual=current["head"])
