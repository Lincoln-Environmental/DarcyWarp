"""Device-resident 3D terrain Picard driver (2D fast-unconfined contract).

The nonlinear outer loop keeps the total head, transformed coefficients,
secant storage, RHS, corrections and residual on the device: per outer
iteration it refreshes saturation, the packed 13-column face operator and
the storage diagonals with the terrain device kernels, freezes the
discharge-only gates at the accepted head, extracts the seven-point
principal part into the multigrid level-0 buffers, rebuilds the row-scaled
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
    terrain_copy_kernel, terrain_dual_check_kernel, terrain_combine_partials_kernel,
    terrain_relative_head_kernel, terrain_extract_principal_kernel,
    terrain_subface_gate_kernel, terrain_subface_scatter_kernel,
    terrain_coarse_faces_kernel, terrain_coarse_shifts_kernel,
    terrain_scale_divide_kernel,
    terrain_scale_multiply_kernel, BLOCK_SIZE)
from DARCY_WARP_PACKAGE.solvers.terrain_kernels_3d import (terrain_combine_kernel,
    terrain_gradients_kernel)


@dataclass
class DevicePicardControls:
    """Adaptive inner and nonlinear controls (2D production defaults)."""
    initial_block_cycles: int = 2          # restart cycles per inner block
    min_block_cycles: int = 1
    max_block_cycles: int = 4
    min_total_cycles: int = 1
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
    minimum_usable_reduction: float = 0.10
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
        self.contact_rhs = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.starting = wp.zeros(self.n, dtype=wp.float64, device=device)
        self.free = wp.zeros(self.n, dtype=wp.int32, device=device)
        blocks = (self.n + BLOCK_SIZE - 1)//BLOCK_SIZE
        self.partials = wp.zeros((blocks, 7), dtype=wp.float64, device=device)
        self.blocks = blocks
        self.combined = wp.zeros(7, dtype=wp.float64, device=device)

    def upload_static(self, *, source, sy, ss, free, subfaces):
        self.source = wp.array(np.ascontiguousarray(source, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.sy = wp.array(np.ascontiguousarray(sy, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.ss = wp.array(np.ascontiguousarray(ss, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.free.assign(np.ascontiguousarray(free.ravel(), dtype=np.int32))
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
                     "open_col0", "contact_rhs", "starting", "sub_cell", "sub_c",
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
        self.restart = restart
        self.krylov_budget = krylov_budget
        self.unconfined = unconfined
        self.transient = transient
        self.dt = dt
        self.abs_tol_min = abs_tol_min
        self.head_tolerance = head_tolerance
        self.budget_relative_tol = budget_relative_tol
        self.rel_tol = rel_tol
        self.n_free = max(int(np.count_nonzero((session.grid.active != 0))), 1)
        self.host_scalar_reads = 0
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
            inputs=[s.coefficients, s.active, s.storage_diag, s.open_col0,
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
            inputs=[head_buffer, candidate_buffer, s.residual, self.workspace.b,
                    s.scale, s.free, s.partials, BLOCK_SIZE], device=self.device)
        wp.launch(terrain_combine_partials_kernel, dim=1,
            inputs=[s.partials, s.combined, s.blocks], device=self.device)
        self.host_scalar_reads += 1
        return s.combined.numpy()

    def matvec_residual(self, x_buffer, rhs_buffer):
        s = self.state
        self.workspace.apply_operator(x=x_buffer, y=rhs_buffer, z=s.residual,
            alpha=-1.0, beta=1.0, row_scaled=False)
        return s.residual

    # ---- adaptive inner controller -------------------------------------
    def inner_target(self, head_rms, flow_rms, rhs_rms, scale_max, previous_dh_rms):
        c = self.controls
        if self._eta is None:
            eta = c.eta_initial
        else:
            ratio = min(head_rms/max(self._eta_residual, c.residual_floor), 1e3)
            eta = float(np.clip(c.eta_gamma*ratio**c.eta_power, c.eta_min, c.eta_max))
        self._eta = eta
        self._eta_residual = max(head_rms, c.residual_floor)
        target = float(np.clip(eta*self._eta_residual, c.residual_floor, 1e-2))
        picard_bound = 0.10*max(self.head_tolerance, previous_dh_rms)
        target = min(target, picard_bound)
        return target, max(c.residual_floor, eta*max(flow_rms, c.residual_floor))

    def run_inner(self, initial, targets):
        """Adaptive inexact inner solve on the frozen correction system.

        Contraction is measured from GMRES's own restart-check norms (the
        row-scaled correction residual), so no extra device reductions are
        needed inside the block loop; the outer dual check provides the
        physical acceptance quantities."""
        s = self.state
        c = self.controls
        krylov = self.workspace.krylov
        head_target, flow_target_scaled = targets
        total_cycles = 0
        block_cycles = c.initial_block_cycles
        stall_patience = 0
        stagnated = False
        diverged = False
        reached = False
        usable = True
        prev_error = max(initial[0], c.residual_floor)
        while total_cycles < self.krylov_budget:
            wp.launch(terrain_copy_kernel, dim=s.n,
                inputs=[s.delta], outputs=[s.delta_snapshot], device=self.device)
            cycles = min(block_cycles, self.krylov_budget-total_cycles)
            krylov._maxiter = cycles*self.restart
            used, error, tolerance = krylov()
            total_cycles += used
            if not np.isfinite(error):
                wp.launch(terrain_copy_kernel, dim=s.n,
                    inputs=[s.delta_snapshot], outputs=[s.delta], device=self.device)
                diverged = True
                usable = False
                break
            if error <= tolerance and total_cycles >= c.min_total_cycles:
                reached = True
                break
            rho = error/max(prev_error, c.residual_floor)
            per_cycle = rho**(1.0/max(used/max(self.restart, 1), 1.0))
            if rho > c.divergence_ratio:
                wp.launch(terrain_copy_kernel, dim=s.n,
                    inputs=[s.delta_snapshot], outputs=[s.delta], device=self.device)
                diverged = True
                usable = False
                break
            if per_cycle >= c.stall_contraction_ratio:
                stall_patience += 1
                if stall_patience >= c.stall_patience:
                    stagnated = True
                    usable = prev_error/max(initial[0], c.residual_floor) \
                        <= c.minimum_usable_reduction
                    break
            else:
                stall_patience = 0
            if rho <= c.good_contraction_ratio:
                block_cycles = min(2*max(block_cycles, c.min_block_cycles),
                                   c.max_block_cycles)
            elif rho > c.weak_contraction_ratio:
                block_cycles = c.min_block_cycles
            prev_error = error
        if not reached and not stagnated and not diverged:
            # Budget exhausted: keep the correction only if it reduced the
            # correction residual substantially (a usable inexact step).
            usable = prev_error/max(initial[0], c.residual_floor) \
                <= c.minimum_usable_reduction
        return dict(cycles=total_cycles, reached=reached, stagnated=stagnated,
            diverged=diverged, usable=usable, head_rms=prev_error, flow_rms=0.0,
            dh_max=float(np.sqrt(max(s.combined.numpy()[1] if hasattr(s, "combined") else 0.0, 0.0))))

    # ---- main loop ------------------------------------------------------
    def run(self, max_outer):
        s = self.state
        c = self.controls
        self._eta = None
        self._eta_residual = None
        self._retries = 0
        history = []
        total_krylov = 0
        converged = False
        accepted_head_residual = np.inf
        previous_dh_rms = np.inf
        rejected = 0
        start = perf_counter()
        for outer in range(max_outer):
            self.evaluate(s.head, s.head_rel)
            # Raw physical residual at the accepted head (units m^3/day),
            # then the row-scaled correction RHS for the Krylov system.
            wp.launch(terrain_scale_multiply_kernel, dim=s.n,
                inputs=[s.rhs, s.scale, s.rhs_raw], device=self.device)
            self.matvec_residual(s.head_rel, s.rhs_raw)
            wp.launch(terrain_scale_divide_kernel, dim=s.n,
                inputs=[s.residual, s.scale, s.rhs], device=self.device)
            s.delta.zero_()
            wp.launch(terrain_copy_kernel, dim=s.n,
                inputs=[s.head_rel], outputs=[s.candidate_rel], device=self.device)
            combined = self.dual_check(s.head_rel, s.candidate_rel)
            n = max(self.n_free, 1)
            # Channel semantics: c2 = squared residual in the units the dual
            # check was fed (RAW here), c3 = head-equivalent (raw/scale)^2.
            flow_rms = float(np.sqrt(max(combined[2], 0.0)/n))
            head_residual = float(np.sqrt(max(combined[3], 0.0)/n))
            rhs_rms = float(np.sqrt(max(combined[4], 0.0)/n))
            scale_max = float(combined[6]) + 1e-300
            if float(combined[5]) > 0:
                break
            if head_residual <= self.head_tolerance and (
                    flow_rms <= self.abs_tol_min*np.sqrt(n)
                    or flow_rms/max(rhs_rms, 1e-300) <= self.budget_relative_tol):
                converged = True
                accepted_head_residual = head_residual
                break
            targets = self.inner_target(head_residual, flow_rms, rhs_rms, scale_max,
                                        previous_dh_rms)
            krylov = self.workspace.krylov
            krylov._tol = 0.0
            krylov._atol = max(targets[0]*np.sqrt(self.n_free), 1e-300)
            outcome = self.run_inner((head_residual, targets[1]), targets)
            total_krylov += outcome["cycles"]
            if not outcome["usable"] and not outcome["reached"]:
                if self._retries < 2:
                    # 2D retry-first policy: one full-effort retry with a
                    # tight tolerance before declaring a genuine failure.
                    self._retries += 1
                    s.delta.zero_()
                    krylov._maxiter = self.krylov_budget
                    krylov._tol = 0.0
                    krylov._atol = max(self.abs_tol_min*np.sqrt(self.n_free), 1e-300)
                    used, error, tolerance = krylov()
                    total_krylov += used
                    outcome = dict(cycles=used, reached=error <= tolerance,
                        stagnated=False, diverged=not np.isfinite(error),
                        usable=np.isfinite(error) and error <= 1.0,
                        head_rms=error, flow_rms=0.0, dh_max=0.0)
                    history.append(dict(outer_iteration=outer+1,
                        krylov_iterations=used, inner_reached=outcome["reached"],
                        inner_stagnated=False, inner_diverged=outcome["diverged"],
                        inner_usable=outcome["usable"], inner_retried=True,
                        proposed_head_change_max=0.0, accepted_head_change_max=0.0,
                        head_residual_rms=head_residual, relaxation=0.0))
                    if not outcome["usable"] or not outcome["reached"]:
                        accepted_head_residual = head_residual
                        break
                else:
                    # A frozen-operator correction that no damping can use
                    # is a genuine linear/nonlinear failure (active-set kink
                    # or preconditioner stagnation), not progress: stop and
                    # report instead of applying rejected proposals forever.
                    history.append(dict(outer_iteration=outer+1,
                        krylov_iterations=0,
                        inner_reached=False, inner_stagnated=outcome["stagnated"],
                        inner_diverged=outcome["diverged"], inner_usable=False,
                        inner_retried=True,
                        proposed_head_change_max=0.0, accepted_head_change_max=0.0,
                        head_residual_rms=head_residual, relaxation=0.0))
                    accepted_head_residual = head_residual
                    break
            # Damped nonlinear update with the FULL operator, storage, gates
            # and RHS re-evaluated at every candidate head; the accepted
            # state (head, operator, gates, RHS) is preserved untouched
            # until a candidate passes the physical residual test.
            relaxation = self.unconfined_relax if self.outer_limit_val > 1 else 1.0
            wp.launch(terrain_copy_kernel, dim=s.n,
                inputs=[s.head], outputs=[s.starting], device=self.device)
            accepted = False
            trial_head_residual = np.inf
            trial_flow_rms = 0.0
            proposed_max = 0.0
            for backtrack in range(12):
                wp.launch(terrain_copy_kernel, dim=s.n,
                    inputs=[s.starting], outputs=[s.candidate], device=self.device)
                wp.launch(terrain_relaxed_update_kernel, dim=s.n,
                    inputs=[s.candidate, s.delta, wp.float64(relaxation),
                            wp.float64(c.max_head_change_per_outer), s.fixed,
                            s.bc_values, s.active], device=self.device)
                # Everything at the CANDIDATE: coefficients, storage, gates,
                # aggregated boundary, full hierarchy, RHS. The accepted
                # head_rel is left untouched for the dh comparison.
                self.evaluate(s.candidate, s.candidate_rel)
                wp.launch(terrain_scale_multiply_kernel, dim=s.n,
                    inputs=[s.rhs, s.scale, s.rhs_raw], device=self.device)
                self.matvec_residual(s.candidate_rel, s.rhs_raw)
                combined = self.dual_check(s.head_rel, s.candidate_rel)
                trial_head_residual = float(np.sqrt(max(combined[3], 0.0)/n))
                trial_flow_rms = float(np.sqrt(max(combined[2], 0.0)/n))
                proposed_max = float(combined[1])/max(relaxation, 1e-300)
                if trial_head_residual <= accepted_head_residual \
                        or trial_head_residual <= self.head_tolerance:
                    accepted = True
                    break
                relaxation *= 0.5
            if not accepted:
                rejected += 1
                # Roll the accepted state back in full: head, operator,
                # gates, aggregated boundary, hierarchy and RHS.
                wp.launch(terrain_copy_kernel, dim=s.n,
                    inputs=[s.starting], outputs=[s.head], device=self.device)
                self.evaluate(s.head, s.head_rel)
                trial_head_residual = accepted_head_residual
                accepted_dh_rms = 0.0
                accepted_dh_max = 0.0
            else:
                accepted_head_residual = min(accepted_head_residual, trial_head_residual)
                # Commit the accepted candidate as the new nonlinear state
                # (operator/gates/RHS already evaluated at it).
                wp.launch(terrain_copy_kernel, dim=s.n,
                    inputs=[s.candidate], outputs=[s.head], device=self.device)
                accepted_dh_rms = float(np.sqrt(max(combined[0], 0.0))/n)
                accepted_dh_max = float(combined[1])
            previous_dh_rms = accepted_dh_rms
            history.append(dict(outer_iteration=outer+1,
                krylov_iterations=outcome["cycles"],
                inner_reached=outcome["reached"], inner_stagnated=outcome["stagnated"],
                inner_diverged=outcome["diverged"], inner_usable=outcome["usable"],
                proposed_head_change_max=proposed_max,
                accepted_head_change_max=accepted_dh_max,
                accepted_head_change_rms=accepted_dh_rms,
                head_residual_rms=trial_head_residual,
                relaxation=float(relaxation)))
            if accepted and trial_head_residual <= self.head_tolerance and (
                    trial_flow_rms <= self.abs_tol_min*np.sqrt(n)):
                converged = True
                accepted_head_residual = trial_head_residual
                break
        seconds = perf_counter()-start
        return dict(converged=converged, outer_iterations=len(history),
            krylov_iterations=total_krylov, seconds=seconds,
            host_scalar_reads=self.host_scalar_reads, rejected=rejected,
            history=history, final_head_residual=accepted_head_residual)
