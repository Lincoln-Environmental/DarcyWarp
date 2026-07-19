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
    ):
        self.device = str(device)
        self.signature = tuple(level.shape for level in physical_levels)
        self.transient = bool(transient)
        self.dt = None if dt is None else float(dt)
        self.min_sat = float(min_sat)
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

    def compatible(
        self,
        *,
        physical_levels: list[FASPhysicalLevel2D],
        transient: bool,
        dt: float | None,
        min_sat: float,
        device: str,
    ) -> bool:
        basic = (
            not self.closed
            and self.signature == tuple(level.shape for level in physical_levels)
            and self.transient == bool(transient)
            and self.dt == (None if dt is None else float(dt))
            and self.min_sat == float(min_sat)
            and self.device == str(device)
        )
        if not basic:
            return False
        names = (
            "conductivity", "top", "bottom", "active", "active_fraction",
            "dirichlet_mask", "dirichlet_values", "source_rate", "ghb_mask",
            "ghb_factor", "ghb_external_head", "sy", "ss", "previous_head",
        )
        return all(
            np.array_equal(getattr(old.physical, name), getattr(new, name))
            for old, new in zip(self.levels, physical_levels)
            for name in names
        )

    def reset_counters(self) -> None:
        self.kernel_launches = 0
        self.transfer_launches = 0
        self.synchronizations = 0
        self.pre_sweeps = [0 for _ in self.levels]
        self.post_sweeps = [0 for _ in self.levels]
        self.coarse_sweeps = [0 for _ in self.levels]
        self.smoothing_history: list[dict[str, Any]] = []

    def close(self) -> None:
        if self.closed:
            return
        for level in self.levels:
            level.close()
        self.levels = []
        self.closed = True


__all__ = ["FASLevelState", "FASWorkspace", "NonlinearLevelOperator2D"]
