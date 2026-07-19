# SPDX-License-Identifier: AGPL-3.0-only
"""Persistent nonlinear level state for the experimental 2D FAS backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D, from_arrays
from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT
from .fas_hierarchy import FASPhysicalLevel2D
from . import fas_kernels as _k


class NonlinearLevelOperator2D:
    """Authoritative Stage-1 operator plus a level-specific FAS forcing."""

    def __init__(self, *, physical: FASPhysicalLevel2D, transient: bool, dt: float | None, min_sat: float, device: str):
        # The level operator collapses sy/ss to scalars (the Stage-1 operator
        # context is scalar-valued); spatially varying fields would be
        # silently misrepresented, so reject them loudly. The public
        # solve_unconfined_fas entry only accepts scalar sy/ss, so the public
        # path always satisfies this.
        active_values = physical.sy[physical.active != 0]
        if active_values.size and not np.allclose(active_values, active_values[0], rtol=1e-12, atol=0.0):
            raise ValueError(
                "NonlinearLevelOperator2D requires spatially uniform sy; "
                "the FAS backend currently supports scalar sy/ss only."
            )
        sy = float(active_values[0]) if active_values.size else 0.0
        ss_values = physical.ss[physical.active != 0]
        if ss_values.size and not np.allclose(ss_values, ss_values[0], rtol=1e-12, atol=0.0):
            raise ValueError(
                "NonlinearLevelOperator2D requires spatially uniform ss; "
                "the FAS backend currently supports scalar sy/ss only."
            )
        ss = float(ss_values[0]) if ss_values.size else 0.0
        context = from_arrays(
            nx=physical.nx,
            ny=physical.ny,
            dx=physical.dx,
            K=physical.conductivity,
            zbot=physical.bottom,
            ztop=physical.top if physical.has_top else None,
            active=physical.active,
            dirichlet_mask=physical.dirichlet_mask,
            dirichlet_values=physical.dirichlet_values,
            R_field=physical.source_rate,
            ghb_mask=physical.ghb_mask,
            ghb_external_head=physical.ghb_external_head,
            ghb_factor=physical.ghb_factor,
            sy=sy,
            ss=ss,
            head_prev=physical.previous_head if transient else None,
            dt=dt if transient else None,
            transient=transient,
            min_sat=min_sat,
            device=device,
        )
        self.operator = NonlinearOperator2D(context)
        self.physical = physical
        self.device = str(device)
        self.dim = physical.shape
        self.nx = physical.nx
        self.ny = physical.ny
        self.kernel_launches = 0

    def evaluate(
        self,
        *,
        head: Any,
        state: "FASLevelState",
        physical_residual: Any | None = None,
        defect: Any | None = None,
    ) -> None:
        residual_out = state.physical_residual if physical_residual is None else physical_residual
        defect_out = state.defect if defect is None else defect
        self.operator.residual_device(head, out=residual_out, reduce=False)
        wp.launch(
            kernel=_k.fas_defect_kernel,
            dim=self.dim,
            inputs=[
                residual_out,
                state.physical_forcing,
                state.forcing,
                self.operator.active_device,
                self.operator.dirichlet_mask_device,
                defect_out,
                self.nx,
                self.ny,
            ],
            device=self.device,
        )
        self.kernel_launches += 2

    def refresh_frozen_diagonal(self, *, head: Any, state: "FASLevelState") -> None:
        transmissivity, storage = self.operator.freeze_picard_device(head)
        wp.launch(
            kernel=_k.fas_frozen_diagonal_kernel,
            dim=self.dim,
            inputs=[
                transmissivity,
                storage,
                self.operator.active_device,
                self.operator.dirichlet_mask_device,
                self.operator.ghb_mask_device,
                self.operator.ghb_factor_device,
                state.diagonal,
                self.nx,
                self.ny,
            ],
            device=self.device,
        )
        self.kernel_launches += 2

    def close(self) -> None:
        self.operator.close()


@dataclass(slots=True)
class FASLevelState:
    physical: FASPhysicalLevel2D
    nonlinear_operator: NonlinearLevelOperator2D
    head: Any
    head_initial: Any
    head_cycle_start: Any
    physical_residual: Any
    defect: Any
    forcing: Any
    physical_forcing: Any
    tau: Any
    restricted_defect: Any
    restricted_forcing: Any
    correction: Any
    prolonged_correction: Any
    candidate: Any
    candidate_residual: Any
    candidate_defect: Any
    diagonal: Any
    sum_sq: Any
    max_abs: Any
    change_sq: Any
    change_max: Any
    finite_flag: Any
    n_free: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.physical.shape

    def close(self) -> None:
        self.nonlinear_operator.close()
        for name in (
            "head", "head_initial", "head_cycle_start", "physical_residual", "defect",
            "forcing", "physical_forcing", "tau", "restricted_defect",
            "restricted_forcing", "correction", "prolonged_correction", "candidate",
            "candidate_residual", "candidate_defect", "diagonal", "sum_sq", "max_abs",
            "change_sq", "change_max", "finite_flag",
        ):
            setattr(self, name, None)


class FASWorkspace:
    """Model-owned FAS levels and counters, reusable across compatible solves."""

    def __init__(
        self,
        *,
        physical_levels: list[FASPhysicalLevel2D],
        transient: bool,
        dt: float | None,
        min_sat: float,
        device: str,
        max_levels: int | None = None,
        min_coarse_cells: int | None = None,
    ):
        self.device = str(device)
        self.signature = tuple(level.shape for level in physical_levels)
        self.transient = bool(transient)
        self.dt = None if dt is None else float(dt)
        self.min_sat = float(min_sat)
        self.max_levels = None if max_levels is None else int(max_levels)
        self.min_coarse_cells = None if min_coarse_cells is None else int(min_coarse_cells)
        self.levels = [
            self._allocate_level(
                physical=physical,
                transient=transient,
                dt=dt,
                min_sat=min_sat,
                device=device,
            )
            for physical in physical_levels
        ]
        # Static (structural) inputs captured from the fine level.  Only these
        # decide whether the workspace may be reused; timestep-dependent state
        # (previous head, source field, dt, Sy, Ss) is refreshed in place by
        # refresh_timestep instead of forcing a rebuild.  Note the per-level
        # ``physical`` host mirrors keep their build-time dynamic fields — the
        # device arrays are authoritative after a refresh.
        fine = physical_levels[0]
        self.static_inputs = {
            "dx": float(fine.dx),
            "has_top": bool(fine.has_top),
            "conductivity": np.array(fine.conductivity, dtype=np.float64, copy=True),
            "bottom": np.array(fine.bottom, dtype=np.float64, copy=True),
            "top": np.array(fine.top, dtype=np.float64, copy=True) if fine.has_top else None,
            "active": np.array(fine.active, dtype=np.int32, copy=True),
            "dirichlet_mask": np.array(fine.dirichlet_mask, dtype=np.int32, copy=True),
            "dirichlet_values": np.array(fine.dirichlet_values, dtype=np.float64, copy=True),
            "ghb_mask": np.array(fine.ghb_mask, dtype=np.int32, copy=True),
            "ghb_factor": np.array(fine.ghb_factor, dtype=np.float64, copy=True),
            "ghb_external_head": np.array(fine.ghb_external_head, dtype=np.float64, copy=True),
            "sy": np.array(fine.sy, dtype=np.float64, copy=True),
            "ss": np.array(fine.ss, dtype=np.float64, copy=True),
        }
        self._forcing_stage = wp.zeros(fine.shape, dtype=WP_FLOAT, device="cpu")
        self.refresh_count = 0
        self.closed = False
        self.reset_counters()

    @staticmethod
    def _allocate_level(
        *,
        physical: FASPhysicalLevel2D,
        transient: bool,
        dt: float | None,
        min_sat: float,
        device: str,
    ) -> FASLevelState:
        shape = physical.shape
        operator = NonlinearLevelOperator2D(
            physical=physical,
            transient=transient,
            dt=dt,
            min_sat=min_sat,
            device=device,
        )
        free = (physical.active != 0) & (physical.dirichlet_mask == 0)
        physical_forcing_host = np.zeros(shape, dtype=np.float64)
        physical_forcing_host[free] = physical.source_rate[free] * physical.area
        physical_forcing = wp.array(physical_forcing_host, dtype=WP_FLOAT, device=device)
        forcing = wp.array(physical_forcing_host, dtype=WP_FLOAT, device=device)

        def field():
            return wp.zeros(shape, dtype=WP_FLOAT, device=device)

        return FASLevelState(
            physical=physical,
            nonlinear_operator=operator,
            head=field(),
            head_initial=field(),
            head_cycle_start=field(),
            physical_residual=field(),
            defect=field(),
            forcing=forcing,
            physical_forcing=physical_forcing,
            tau=field(),
            restricted_defect=field(),
            restricted_forcing=field(),
            correction=field(),
            prolonged_correction=field(),
            candidate=field(),
            candidate_residual=field(),
            candidate_defect=field(),
            diagonal=field(),
            sum_sq=wp.zeros(1, dtype=wp.float64, device=device),
            max_abs=wp.zeros(1, dtype=wp.float64, device=device),
            change_sq=wp.zeros(1, dtype=wp.float64, device=device),
            change_max=wp.zeros(1, dtype=wp.float64, device=device),
            finite_flag=wp.zeros(1, dtype=wp.int32, device=device),
            n_free=int(np.count_nonzero(free)),
        )

    def static_compatible(
        self,
        *,
        fine_physical: FASPhysicalLevel2D,
        transient: bool,
        min_sat: float,
        device: str,
        max_levels: int | None = None,
        min_coarse_cells: int | None = None,
    ) -> bool:
        """True when every structural (non-timestep) input is unchanged.

        Only static data decides reuse: grid/hierarchy signature, dx,
        conductivity, geometry, active and prescribed masks, prescribed-head
        and GHB fields, and the hierarchy build controls.  Previous head,
        source field, dt, Sy and Ss are timestep state and are handled by
        :meth:`refresh_timestep`, never by a rebuild.
        """
        if self.closed or self.transient != bool(transient):
            return False
        if self.min_sat != float(min_sat) or self.device != str(device):
            return False
        if self.max_levels is not None and max_levels is not None and self.max_levels != int(max_levels):
            return False
        if (
            self.min_coarse_cells is not None
            and min_coarse_cells is not None
            and self.min_coarse_cells != int(min_coarse_cells)
        ):
            return False
        if self.signature[0] != tuple(fine_physical.shape):
            return False
        static = self.static_inputs
        if float(static["dx"]) != float(fine_physical.dx):
            return False
        if bool(static["has_top"]) != bool(fine_physical.has_top):
            return False
        names = (
            "conductivity", "bottom", "active", "dirichlet_mask",
            "dirichlet_values", "ghb_mask", "ghb_factor", "ghb_external_head",
        )
        for name in names:
            if not np.array_equal(static[name], getattr(fine_physical, name)):
                return False
        if static["has_top"] and not np.array_equal(static["top"], fine_physical.top):
            return False
        return True

    def refresh_timestep(
        self,
        *,
        previous_head: Any | None = None,
        dt: float | None = None,
        source_rate: Any | None = None,
        sy: float | None = None,
        ss: float | None = None,
    ) -> None:
        """Refresh all timestep-dependent state for a new FAS timestep.

        Uploads the previous accepted head and restricts it down the
        hierarchy, rebuilds level forcing from the current source field,
        updates dt/Sy/Ss on every level operator, and resets all cycle
        scratch (tau, defect, restricted, correction, candidate and coarse
        initial-approximation arrays) so nothing carries between timesteps.
        Static hierarchy data (topology, transfers, conductivity, geometry,
        masks, persistent buffers) is preserved.
        """
        if self.closed:
            raise RuntimeError("cannot refresh a closed FASWorkspace.")
        if not self.transient and (previous_head is not None or dt is not None):
            raise ValueError("previous_head/dt refresh requires a transient FASWorkspace.")
        dt_f: float | None = None
        if dt is not None:
            dt_f = float(dt)
            if not np.isfinite(dt_f) or dt_f <= 0.0:
                raise ValueError("dt must be positive and finite.")
            self.dt = dt_f
        sy_f = None if sy is None else float(sy)
        ss_f = None if ss is None else float(ss)
        if sy_f is not None:
            self.static_inputs["sy"][...] = sy_f
        if ss_f is not None:
            self.static_inputs["ss"][...] = ss_f

        fine = self.levels[0]
        fine_op = fine.nonlinear_operator.operator
        fine_op.update_transient_state(
            head_prev=previous_head,
            dt=dt_f,
            source_rate=source_rate,
            sy=sy_f,
            ss=ss_f,
        )
        if source_rate is not None:
            # Fine physical forcing: source * area on free cells.
            physical = fine.physical
            source_host = np.asarray(source_rate, dtype=np.float64)
            if source_host.shape != physical.shape:
                raise ValueError(f"source_rate must have shape {physical.shape}, got {source_host.shape}.")
            free = (physical.active != 0) & (physical.dirichlet_mask == 0)
            stage = self._forcing_stage.numpy()
            stage[...] = 0.0
            stage[free] = source_host[free] * float(physical.area)
            wp.copy(fine.physical_forcing, self._forcing_stage)

        # Coarse levels: restrict previous head and physical forcing down the
        # hierarchy with the same kernels the V-cycle transfer path uses.
        for level_index in range(1, len(self.levels)):
            parent = self.levels[level_index - 1]
            level = self.levels[level_index]
            parent_op = parent.nonlinear_operator.operator
            level_op = level.nonlinear_operator.operator
            level_op.update_transient_state(dt=dt_f, sy=sy_f, ss=ss_f)
            if previous_head is not None:
                wp.launch(
                    kernel=_k.fas_restrict_head_kernel,
                    dim=level.shape,
                    inputs=[
                        parent_op.head_prev_device,
                        parent_op.active_device,
                        level_op.active_device,
                        level_op.dirichlet_mask_device,
                        level_op.dirichlet_values_device,
                        level_op.head_prev_device,
                        parent.physical.nx,
                        parent.physical.ny,
                        level.physical.nx,
                        level.physical.ny,
                    ],
                    device=self.device,
                )
            if source_rate is not None:
                wp.launch(
                    kernel=_k.fas_restrict_integrated_kernel,
                    dim=level.shape,
                    inputs=[
                        parent.physical_forcing,
                        parent_op.active_device,
                        parent_op.dirichlet_mask_device,
                        level_op.active_device,
                        level_op.dirichlet_mask_device,
                        level.physical_forcing,
                        parent.physical.nx,
                        parent.physical.ny,
                        level.physical.nx,
                        level.physical.ny,
                    ],
                    device=self.device,
                )

        # Timestep-local scratch must not carry across timesteps.
        for level_index, level in enumerate(self.levels):
            for name in (
                "physical_residual", "defect", "tau", "restricted_defect",
                "restricted_forcing", "correction", "prolonged_correction",
                "candidate", "candidate_residual", "candidate_defect",
                "head_initial", "head_cycle_start",
            ):
                getattr(level, name).fill_(WP_FLOAT(0.0))
            if level_index > 0:
                level.head.fill_(WP_FLOAT(0.0))
        self.refresh_count += 1

    def reset_counters(self) -> None:
        self.kernel_launches = 0
        self.transfer_launches = 0
        self.synchronizations = 0
        self.pre_sweeps = [0 for _ in self.levels]
        self.post_sweeps = [0 for _ in self.levels]
        self.coarse_sweeps = [0 for _ in self.levels]
        self.smoothing_history: list[dict[str, Any]] = []
        for level in self.levels:
            level.nonlinear_operator.kernel_launches = 0

    def close(self) -> None:
        if self.closed:
            return
        for level in self.levels:
            level.close()
        self.levels = []
        self.closed = True


__all__ = ["FASLevelState", "FASWorkspace", "NonlinearLevelOperator2D"]
