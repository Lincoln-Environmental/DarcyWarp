# SPDX-License-Identifier: AGPL-3.0-only
"""
Factory for creating the 2D Warp Darcy solver.

This module provides a single switch point so callers do not need to know
which concrete solver class to instantiate.  This branch is 2D-only; the
experimental 3D implementation lives on ``dev``.
"""

from __future__ import annotations

from typing import Any


def create_solver(
    *,
    dim: int,
    solver: str = "kcycle",
    nx: int,
    ny: int,
    dx: float,
    dy: float | None = None,
    device: str = "cuda:0",
    **kwargs: Any,
):
    """
    Create the 2D Warp Darcy solver.

    Parameters
    ----------
    dim
        Spatial dimension. Only ``2`` is supported; any other value raises
        ``ValueError`` (the experimental 3D solver lives on the ``dev``
        branch).
    solver
        Solver preference. Accepts canonical backend names
        ``'confined_pcg'``, ``'confined_kcycle'``, and
        ``'unconfined_picard_kcycle'``. The experimental
        ``'unconfined_semismooth_newton_kcycle'`` and ``'unconfined_fas'``
        backends are explicit-only. Legacy aliases ``'pcg'``, ``'kcycle'``,
        ``'multigrid'``, and ``'mg'`` remain available. An explicitly
        requested unconfined backend is preserved and becomes the default
        backend for later ``solve()`` calls; formulation is still selected
        at ``solve`` time.
    nx, ny
        Number of columns and rows.
    dx
        Cell size in the x direction.
    dy
        Cell size in the y direction. The current 2D solver uses square
        cells, so omit this argument or pass the same value as ``dx``.
    device
        Warp device string, e.g. ``'cuda:0'``.
    **kwargs
        Extra arguments forwarded to the solver constructor.

    Returns
    -------
    WarpDarcySolver
        A solver instance exposing ``build_*`` and ``solve`` methods.
    """
    if dim != 2:
        raise ValueError(
            "DarcyWarp main is 2D-only: dim must be 2 (the experimental 3D "
            "solver lives on the 'dev' branch)."
        )
    if "nz" in kwargs or "dz" in kwargs:
        raise ValueError("nz/dz are 3D-only arguments; main is a 2D-only branch.")
    if dy is not None and float(dy) != float(dx):
        raise ValueError(
            "DarcyWarp's 2D solver currently requires square cells: "
            "omit dy or pass dy equal to dx."
        )

    # Validate the backend name before importing any solver module, so
    # callers get clear ValueError messages even if warp is not installed.
    from DARCY_WARP_PACKAGE.solver_capabilities import ALIASES, CAPABILITIES

    solver_norm = str(solver).strip().lower()
    solver_norm = ALIASES.get(solver_norm, solver_norm)
    if solver_norm not in CAPABILITIES:
        raise ValueError(
            "2D solver must be a supported backend: " + ", ".join(CAPABILITIES)
        )

    from DARCY_WARP_PACKAGE.model import WarpDarcySolver

    # Preserve explicitly requested unconfined backends (Picard, semismooth
    # Newton, FAS) as the solve-time default. Legacy aliases and the confined
    # default keep the generic 'pcg'/'kcycle' constructor options, so a
    # 'kcycle' solver can still solve unconfined problems through the
    # formulation-aware default in canonical_solver_name.
    if solver_norm.startswith("unconfined_"):
        solver_type = solver_norm
    elif solver_norm == "confined_pcg":
        solver_type = "pcg"
    else:
        solver_type = "kcycle"

    return WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=dx,
        device=device,
        solver_type=solver_type,
        **kwargs,
    )
