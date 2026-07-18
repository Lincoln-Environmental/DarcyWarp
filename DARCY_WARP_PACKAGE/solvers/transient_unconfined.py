# SPDX-License-Identifier: AGPL-3.0-only
"""Production multi-period unconfined Picard/K-cycle driver boundary."""

from __future__ import annotations

from typing import Any

from .context import SolverContext
from .registry import select_backend


def solve_transient_unconfined(
    context: SolverContext,
    *,
    solver: str | None = "unconfined_picard_kcycle",
    **kwargs: Any,
):
    """Run the model-owned transient driver through its compatibility bridge.

    The bridge is deliberately narrow: all period, adaptive-dt, and diagnostic
    behaviour remains byte-for-byte in the existing implementation until the
    single-step Picard and K-cycle bodies have been extracted.
    """
    if context.formulation != "unconfined":
        raise ValueError("transient unconfined backend requires formulation='unconfined'.")
    backend = select_backend(
        solver=solver,
        formulation=context.formulation,
        transient=True,
        default="unconfined_picard_kcycle",
    )
    if backend.name != "unconfined_picard_kcycle":
        raise ValueError("transient unconfined flow requires unconfined_picard_kcycle.")
    if context.run_transient is None:
        raise RuntimeError("transient unconfined backend is not available.")
    result = context.run_transient(**kwargs)
    if kwargs.get("return_info", True):
        heads, info = result
        info_out = dict(info)
        info_out["solver_backend"] = backend.name
        last_info = info_out.get("last_info")
        if isinstance(last_info, dict):
            info_out["last_info"] = dict(last_info, solver_backend=backend.name)
        period_infos = info_out.get("period_infos")
        if isinstance(period_infos, list):
            info_out["period_infos"] = [
                dict(period_info, solver_backend=backend.name)
                if isinstance(period_info, dict)
                else period_info
                for period_info in period_infos
            ]
        return heads, info_out
    return result
