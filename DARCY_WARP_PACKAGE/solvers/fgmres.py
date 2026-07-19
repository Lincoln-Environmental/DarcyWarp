# SPDX-License-Identifier: AGPL-3.0-only
"""Restarted matrix-free GPU FGMRES for the experimental Newton backend."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.nonlinear.kernels import WP_FLOAT
from . import newton_kernels as _k


@dataclass(slots=True)
class FGMRESResult:
    converged: bool
    iterations: int
    restarts: int
    final_residual: float
    breakdown: bool
    breakdown_reason: str | None
    residual_history: list[float]
    reduction_count: int


class FGMRESWorkspace2D:
    """Persistent full-grid vectors owned by the Darcy model resource owner."""

    def __init__(self, *, shape: tuple[int, int], restart: int, device: str):
        if int(restart) < 2:
            raise ValueError("FGMRES restart must be at least 2.")
        self.shape = tuple(int(v) for v in shape)
        self.restart = int(restart)
        self.device = str(device)
        self.basis = [wp.zeros(self.shape, dtype=WP_FLOAT, device=device) for _ in range(self.restart + 1)]
        self.preconditioned_basis = [
            wp.zeros(self.shape, dtype=WP_FLOAT, device=device) for _ in range(self.restart)
        ]
        self.rhs = wp.zeros(self.shape, dtype=WP_FLOAT, device=device)
        self.solution = wp.zeros(self.shape, dtype=WP_FLOAT, device=device)
        self.residual = wp.zeros(self.shape, dtype=WP_FLOAT, device=device)
        self.work = wp.zeros(self.shape, dtype=WP_FLOAT, device=device)
        self.jacobian_solution = wp.zeros(self.shape, dtype=WP_FLOAT, device=device)
        self.scalar = wp.zeros(1, dtype=wp.float64, device=device)
        self.closed = False

    def compatible(self, *, shape: tuple[int, int], restart: int, device: str) -> bool:
        return (
            not self.closed
            and self.shape == tuple(shape)
            and self.restart == int(restart)
            and self.device == str(device)
        )

    def close(self) -> None:
        if self.closed:
            return
        self.basis = []
        self.preconditioned_basis = []
        self.rhs = None
        self.solution = None
        self.residual = None
        self.work = None
        self.jacobian_solution = None
        self.scalar = None
        self.closed = True


class RestartedFGMRES2D:
    """Flexible right-preconditioned Arnoldi iteration on Warp arrays.

    Full-grid storage is allocated only by :class:`FGMRESWorkspace2D`.  The
    Hessenberg matrix and Givens rotations are intentionally host-resident and
    bounded by ``restart``; scalar dot/norm reads are the explicit GPU
    synchronizations required by this first production implementation.
    """

    def __init__(
        self,
        *,
        workspace: FGMRESWorkspace2D,
        active: Any,
        prescribed: Any,
        nx: int,
        ny: int,
        device: str,
    ):
        self.ws = workspace
        self.active = active
        self.prescribed = prescribed
        self.nx = int(nx)
        self.ny = int(ny)
        self.dim = (self.ny, self.nx)
        self.device = str(device)
        self.reduction_count = 0

    def _zero_scalar(self) -> None:
        self.ws.scalar.fill_(wp.float64(0.0))

    def _dot(self, x: Any, y: Any) -> float:
        self._zero_scalar()
        wp.launch(
            kernel=_k.masked_dot_kernel,
            dim=self.dim,
            inputs=[x, y, self.active, self.prescribed, self.ws.scalar, self.nx, self.ny],
            device=self.device,
        )
        self.reduction_count += 1
        return float(self.ws.scalar.numpy()[0])

    def _norm(self, x: Any) -> float:
        return float(np.sqrt(max(self._dot(x, x), 0.0)))

    def _copy_scaled(self, src: Any, dst: Any, scale: float) -> None:
        wp.launch(
            kernel=_k.masked_copy_kernel,
            dim=self.dim,
            inputs=[src, self.active, self.prescribed, dst, float(scale), self.nx, self.ny],
            device=self.device,
        )

    def _axpy(self, y: Any, x: Any, alpha: float) -> None:
        wp.launch(
            kernel=_k.masked_axpy_kernel,
            dim=self.dim,
            inputs=[y, x, self.active, self.prescribed, float(alpha), self.nx, self.ny],
            device=self.device,
        )

    def solve(
        self,
        *,
        rhs: Any,
        apply_jacobian: Callable[[Any, Any], None],
        apply_preconditioner: Callable[[Any, Any], None],
        relative_tolerance: float,
        absolute_tolerance: float,
        max_iterations: int,
        breakdown_tolerance: float = 1.0e-30,
    ) -> FGMRESResult:
        restart = self.ws.restart
        max_iterations_i = max(1, int(max_iterations))
        rtol = max(0.0, float(relative_tolerance))
        atol = max(0.0, float(absolute_tolerance))
        self.reduction_count = 0

        self._copy_scaled(rhs, self.ws.rhs, 1.0)
        self.ws.solution.fill_(WP_FLOAT(0.0))
        self._copy_scaled(self.ws.rhs, self.ws.residual, 1.0)
        beta0 = self._norm(self.ws.residual)
        tolerance = max(atol, rtol * beta0)
        history = [beta0]
        if not np.isfinite(beta0):
            return FGMRESResult(False, 0, 0, beta0, True, "nonfinite_initial_residual", history, self.reduction_count)
        if beta0 <= tolerance:
            return FGMRESResult(True, 0, 0, beta0, False, None, history, self.reduction_count)

        total_iterations = 0
        restart_count = 0
        breakdown = False
        breakdown_reason: str | None = None
        beta = beta0

        while total_iterations < max_iterations_i:
            self._copy_scaled(self.ws.residual, self.ws.basis[0], 1.0 / beta)
            H = np.zeros((restart + 1, restart), dtype=np.float64)
            cs = np.zeros(restart, dtype=np.float64)
            sn = np.zeros(restart, dtype=np.float64)
            g = np.zeros(restart + 1, dtype=np.float64)
            g[0] = beta
            used = 0
            estimated_converged = False

            for column in range(restart):
                if total_iterations >= max_iterations_i:
                    break
                apply_preconditioner(self.ws.basis[column], self.ws.preconditioned_basis[column])
                apply_jacobian(self.ws.preconditioned_basis[column], self.ws.work)

                for row in range(column + 1):
                    H[row, column] = self._dot(self.ws.work, self.ws.basis[row])
                    self._axpy(self.ws.work, self.ws.basis[row], -H[row, column])

                H[column + 1, column] = self._norm(self.ws.work)
                if not np.isfinite(H[column + 1, column]):
                    breakdown = True
                    breakdown_reason = "nonfinite_arnoldi_norm"
                    break
                if H[column + 1, column] > float(breakdown_tolerance):
                    self._copy_scaled(
                        self.ws.work,
                        self.ws.basis[column + 1],
                        1.0 / H[column + 1, column],
                    )

                for row in range(column):
                    h0 = H[row, column]
                    h1 = H[row + 1, column]
                    H[row, column] = cs[row] * h0 + sn[row] * h1
                    H[row + 1, column] = -sn[row] * h0 + cs[row] * h1

                denom = float(np.hypot(H[column, column], H[column + 1, column]))
                if denom <= float(breakdown_tolerance):
                    breakdown = True
                    breakdown_reason = "happy_breakdown" if abs(g[column]) <= tolerance else "arnoldi_breakdown"
                    cs[column] = 1.0
                    sn[column] = 0.0
                else:
                    cs[column] = H[column, column] / denom
                    sn[column] = H[column + 1, column] / denom
                H[column, column] = cs[column] * H[column, column] + sn[column] * H[column + 1, column]
                H[column + 1, column] = 0.0
                g[column + 1] = -sn[column] * g[column]
                g[column] = cs[column] * g[column]

                total_iterations += 1
                used = column + 1
                estimate = abs(g[column + 1])
                history.append(float(estimate))
                if estimate <= tolerance:
                    estimated_converged = True
                    break
                if breakdown:
                    break

            if used == 0:
                break
            try:
                y = np.linalg.solve(H[:used, :used], g[:used])
            except np.linalg.LinAlgError:
                breakdown = True
                breakdown_reason = "singular_hessenberg"
                break
            for index in range(used):
                self._axpy(self.ws.solution, self.ws.preconditioned_basis[index], float(y[index]))

            apply_jacobian(self.ws.solution, self.ws.jacobian_solution)
            self._copy_scaled(self.ws.rhs, self.ws.residual, 1.0)
            self._axpy(self.ws.residual, self.ws.jacobian_solution, -1.0)
            beta = self._norm(self.ws.residual)
            history.append(beta)
            if not np.isfinite(beta):
                breakdown = True
                breakdown_reason = "nonfinite_true_linear_residual"
                break
            if beta <= tolerance:
                return FGMRESResult(
                    True, total_iterations, restart_count, beta,
                    breakdown and breakdown_reason != "happy_breakdown",
                    breakdown_reason, history, self.reduction_count,
                )
            if breakdown and not estimated_converged:
                break
            restart_count += 1

        return FGMRESResult(
            False, total_iterations, restart_count, beta, breakdown,
            breakdown_reason, history, self.reduction_count,
        )


__all__ = ["FGMRESResult", "FGMRESWorkspace2D", "RestartedFGMRES2D"]
