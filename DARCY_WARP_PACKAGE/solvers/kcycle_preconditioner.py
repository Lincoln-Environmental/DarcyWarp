# SPDX-License-Identifier: AGPL-3.0-only
"""Fixed-work use of the shared geometric K-cycle as a Newton preconditioner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from time import perf_counter

import warp as wp

from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT
from .multigrid_kcycle import solve_kcycle_device_buffers


@dataclass(slots=True)
class FrozenLevelCoefficients:
    transmissivity: Any
    storage_diagonal: Any
    diagonal_inverse: Any


class KCyclePreconditionerWorkspace2D:
    """Persistent coefficients separate from the trusted Picard hierarchy state."""

    def __init__(self, *, levels: Any, device: str):
        self.device = str(device)
        self.signature = tuple((int(level.ny), int(level.nx)) for level in levels)
        self.coefficients = []
        for ny, nx in self.signature:
            shape = (ny, nx)
            self.coefficients.append(
                FrozenLevelCoefficients(
                    transmissivity=wp.zeros(shape, dtype=WP_FLOAT, device=device),
                    storage_diagonal=wp.zeros(shape, dtype=WP_FLOAT, device=device),
                    diagonal_inverse=wp.zeros(shape, dtype=WP_FLOAT, device=device),
                )
            )
        ny0, nx0 = self.signature[0]
        self.graph_cache = {}
        self.zero_boundary_values = wp.zeros((ny0, nx0), dtype=WP_FLOAT, device=device)
        self.closed = False

    def compatible(self, *, levels: Any, device: str) -> bool:
        signature = tuple((int(level.ny), int(level.nx)) for level in levels)
        return not self.closed and self.device == str(device) and signature == self.signature

    def close(self) -> None:
        if self.closed:
            return
        self.graph_cache.clear()
        self.coefficients = []
        self.zero_boundary_values = None
        self.closed = True


class FixedWorkKCyclePreconditioner2D:
    """Borrow hierarchy topology/kernels while isolating frozen coefficients.

    Each application performs a configured, fixed number of cycles.  It never
    checks convergence and the shared K-cycle's fixed-work path performs no
    device-to-host scalar reads.  Level coefficient references are installed
    only for the duration of a call and restored in ``finally``.
    """

    def __init__(
        self,
        *,
        model: Any,
        levels: Any,
        workspace: KCyclePreconditionerWorkspace2D,
        n_cycles: int,
        nu_pre: int,
        nu_post: int,
        nu_coarse: int,
        smoother: str,
        omega: float,
        cheby_lambda_min: float,
        cheby_lambda_max: float,
        coarse_residual_restriction: str = "average",
        implementation: str = "classic",
        profile_timing: bool = False,
    ):
        self.model = model
        self.levels = levels
        self.workspace = workspace
        self.n_cycles = max(1, int(n_cycles))
        self.implementation = str(implementation).strip().lower()
        if self.implementation not in ('classic', 'fast'):
            raise ValueError('K-cycle preconditioner implementation must be classic or fast')
        if self.implementation == 'fast' and coarse_residual_restriction != 'average':
            raise ValueError('fast face preconditioner supports average residual restriction only')
        self.face_levels = None
        self.graph_cache = workspace.graph_cache
        self.controls = {
            "coarse_residual_restriction": coarse_residual_restriction,
            "max_cycles": self.n_cycles,
            "nu_pre": int(nu_pre),
            "nu_post": int(nu_post),
            "nu_coarse": int(nu_coarse),
            "smoother": str(smoother),
            "omega": float(omega),
            "cheby_lambda_min": float(cheby_lambda_min),
            "cheby_lambda_max": float(cheby_lambda_max),
        }
        self.profile_timing = bool(profile_timing)
        self.timing_events = {'setup': [], 'apply': []}
        self.submission_wall_s = {'setup': 0.0, 'apply': 0.0}
        self.timing_device = wp.get_device(model.device_str)
        self.applications = 0
        self.cycles = 0

    def _start_timing(self, *, kind):
        if not self.profile_timing:
            return None
        start = end = None
        if self.timing_device.is_cuda:
            start = wp.Event(device=self.timing_device, enable_timing=True)
            end = wp.Event(device=self.timing_device, enable_timing=True)
            wp.record_event(start)
        return kind, perf_counter(), start, end

    def _end_timing(self, *, token):
        if token is None:
            return
        kind, started, start, end = token
        if end is not None:
            wp.record_event(end)
            self.timing_events[kind].append((start, end))
        self.submission_wall_s[kind] += perf_counter() - started

    def timing_report(self):
        if not self.profile_timing:
            return {}
        # Resolve events once after the solve; no per-application host barrier.
        gpu = {kind: sum(wp.get_event_elapsed_time(start, end) / 1000.0
                        for start, end in events)
               for kind, events in self.timing_events.items()}
        return {'preconditioner_setup_wall_s': self.submission_wall_s['setup'],
                'preconditioner_apply_wall_s': self.submission_wall_s['apply'],
                'preconditioner_setup_gpu_s': gpu['setup'] if self.timing_device.is_cuda else None,
                'preconditioner_apply_gpu_s': gpu['apply'] if self.timing_device.is_cuda else None,
                'preconditioner_timing_method': 'deferred CUDA events and CPU submission time'}

    @property
    def fine_diagonal_inverse(self) -> Any:
        return self.workspace.coefficients[0].diagonal_inverse

    def freeze(self, *, transmissivity: Any, storage_diagonal: Any) -> None:
        """Refresh all frozen Picard coefficients in-place on the device."""
        timing = self._start_timing(kind='setup')
        wp.copy(self.workspace.coefficients[0].transmissivity, transmissivity)
        wp.copy(self.workspace.coefficients[0].storage_diagonal, storage_diagonal)
        kernel_module = __import__("DARCY_WARP_PACKAGE.warped_darcy", fromlist=["coarsen_transient_operator_level_kernel"])
        coarsen_kernel = kernel_module.coarsen_transient_operator_level_kernel

        for level_index in range(1, len(self.levels)):
            fine_level = self.levels[level_index - 1]
            coarse_level = self.levels[level_index]
            fine_coeff = self.workspace.coefficients[level_index - 1]
            coarse_coeff = self.workspace.coefficients[level_index]
            wp.launch(
                kernel=coarsen_kernel,
                dim=(int(coarse_level.ny), int(coarse_level.nx)),
                inputs=[
                    fine_coeff.transmissivity,
                    fine_coeff.storage_diagonal,
                    fine_level.active_wp,
                    coarse_level.active_wp,
                    coarse_level.bc_mask_wp,
                    int(fine_level.nx),
                    int(fine_level.ny),
                    int(coarse_level.nx),
                    int(coarse_level.ny),
                    coarse_coeff.transmissivity,
                    coarse_coeff.storage_diagonal,
                ],
                device=self.model.device_str,
            )

        for level, coeff in zip(self.levels, self.workspace.coefficients):
            self.model._update_diag_preconditioner_device(
                T_wp=coeff.transmissivity,
                active_wp=level.active_wp,
                bc_mask_wp=level.bc_mask_wp,
                gh_mask_wp=level.gh_mask_wp,
                ghb_factor_wp=level.ghb_factor_wp,
                M_inv_wp=coeff.diagonal_inverse,
                nx=int(level.nx),
                ny=int(level.ny),
                use_ghb=bool(self.model.use_ghb),
                storage_diag_wp=coeff.storage_diagonal,
            )

        if self.implementation == 'fast':
            from .face_transient_f64 import ensure_transient_face_levels, refresh_transient_face_levels
            snapshots = [(level.T_wp, level.storage_diag_wp, level.M_inv_wp) for level in self.levels]
            try:
                for level, coeff in zip(self.levels, self.workspace.coefficients):
                    level.T_wp = coeff.transmissivity
                    level.storage_diag_wp = coeff.storage_diagonal
                    level.M_inv_wp = coeff.diagonal_inverse
                self.face_levels = ensure_transient_face_levels(model=self.model, levels=self.levels)
                refresh_transient_face_levels(model=self.model, levels=self.levels, face_levels=self.face_levels)
            finally:
                for level, snapshot in zip(self.levels, snapshots):
                    level.T_wp, level.storage_diag_wp, level.M_inv_wp = snapshot

        self._end_timing(token=timing)

    def apply(self, rhs: Any, out: Any) -> None:
        """Apply fixed K-cycle work without allocations, checks, or host reads."""
        timing = self._start_timing(kind='apply')
        out.fill_(WP_FLOAT(0.0))
        snapshots = []
        level0 = self.levels[0]
        level0_front = (
            level0.x_wp,
            level0.b_wp,
            level0.T_wp,
            level0.storage_diag_wp,
            level0.M_inv_wp,
            level0.bc_values_wp,
        )
        for level, coeff in zip(self.levels, self.workspace.coefficients):
            snapshots.append((level.T_wp, level.storage_diag_wp, level.M_inv_wp))
            level.T_wp = coeff.transmissivity
            level.storage_diag_wp = coeff.storage_diagonal
            level.M_inv_wp = coeff.diagonal_inverse
        try:
            if self.implementation == 'fast':
                from .face_transient_f64 import solve_kcycle_face_transient_device_buffers
                solve_kcycle_face_transient_device_buffers(model=self.model, x_wp=out, rhs_wp=rhs,
                    T_wp=self.workspace.coefficients[0].transmissivity,
                    storage_diag_wp=self.workspace.coefficients[0].storage_diagonal,
                    active_wp=level0.active_wp, bc_mask_wp=level0.bc_mask_wp,
                    bc_values_wp=self.workspace.zero_boundary_values, levels=self.levels,
                    face_levels=self.face_levels, solve_controls=self.controls,
                    return_scalar_info=False, graph_cache=self.graph_cache)
            else:
                solve_kcycle_device_buffers(
                model=self.model,
                x_wp=out,
                rhs_wp=rhs,
                T_wp=self.workspace.coefficients[0].transmissivity,
                storage_diag_wp=self.workspace.coefficients[0].storage_diagonal,
                active_wp=level0.active_wp,
                bc_mask_wp=level0.bc_mask_wp,
                bc_values_wp=self.workspace.zero_boundary_values,
                levels=self.levels,
                solve_controls=self.controls,
                return_scalar_info=False,
                fixed_work_no_scalar_reads=True,
                )
        finally:
            for level, snapshot in zip(self.levels, snapshots):
                level.T_wp, level.storage_diag_wp, level.M_inv_wp = snapshot
            (
                level0.x_wp,
                level0.b_wp,
                level0.T_wp,
                level0.storage_diag_wp,
                level0.M_inv_wp,
                level0.bc_values_wp,
            ) = level0_front
        self._end_timing(token=timing)
        self.applications += 1
        self.cycles += self.n_cycles


__all__ = ["FixedWorkKCyclePreconditioner2D", "KCyclePreconditionerWorkspace2D"]
