# SPDX-License-Identifier: AGPL-3.0-only
"""
Factory for creating 2D or 3D Warp Darcy solvers.

This module provides a single switch point so callers do not need to know
which concrete solver class to instantiate.
"""

from __future__ import annotations

from typing import Any


def create_solver(
    *,
    dim: int,
    solver: str = "kcycle",
    nx: int,
    ny: int,
    nz: int | None = None,
    dx: float,
    dy: float | None = None,
    dz: float | None = None,
    device: str = "cuda:0",
    **kwargs: Any,
):
    """
    Create a 2D or 3D Warp Darcy solver.

    Parameters
    ----------
    dim
        Spatial dimension: 2 or 3.
    solver
        Solver preference. 2D accepts canonical backend names
        ``'confined_pcg'``, ``'confined_kcycle'``, and
        ``'unconfined_picard_kcycle'`` plus legacy ``'pcg'``, ``'kcycle'``,
        ``'multigrid'``, and ``'mg'``.  Formulation is selected at ``solve``
        time.  3D accepts ``'kcycle'`` or ``'chebyshev'``.
    nx, ny
        Number of columns and rows.
    nz
        Number of layers (required when ``dim=3``).
    dx
        Cell size in the x direction.
    dy
        Cell size in the y direction. Defaults to ``dx`` for 2D and 3D.
    dz
        Cell size in the z direction. Required when ``dim=3``; ignored for 2D.
    device
        Warp device string, e.g. ``'cuda:0'``.
    **kwargs
        Extra arguments forwarded to the solver constructor.

    Returns
    -------
    WarpDarcySolver or WarpDarcySolver3D
        A solver instance exposing ``build_*`` and ``solve`` methods.
    """
    if dim not in (2, 3):
        raise ValueError("dim must be 2 or 3")

    solver_norm = str(solver).strip().lower()

    # Validate dimension-specific arguments before importing any solver module,
    # so callers get clear ValueError messages even if warp is not installed.
    if dim == 2:
        solver_aliases = {
            "pcg": "confined_pcg",
            "kcycle": "confined_kcycle",
            "multigrid": "confined_kcycle",
            "mg": "confined_kcycle",
            "picard": "unconfined_picard_kcycle",
            "picard_kcycle": "unconfined_picard_kcycle",
        }
        solver_norm = solver_aliases.get(solver_norm, solver_norm)
        if solver_norm not in {
            "confined_pcg",
            "confined_kcycle",
            "unconfined_picard_kcycle",
        }:
            raise ValueError(
                "2D solver must be a supported backend: confined_pcg, "
                "confined_kcycle, or unconfined_picard_kcycle"
            )
    else:
        if solver_norm not in {"kcycle", "chebyshev"}:
            raise ValueError("3D solver must be 'kcycle' or 'chebyshev'")
        if nz is None:
            raise ValueError("nz is required for 3D solver")
        if dz is None:
            raise ValueError("dz is required for 3D solver")

    if dim == 2:
        from DARCY_WARP_PACKAGE.model import WarpDarcySolver

        solver_type = "pcg" if solver_norm == "confined_pcg" else "kcycle"
        return WarpDarcySolver(
            nx=nx,
            ny=ny,
            dx=dx,
            device=device,
            solver_type=solver_type,
            **kwargs,
        )

    # dim == 3
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    return WarpDarcySolver3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=dx,
        dy=dy,
        dz=dz,
        device=device,
        solver=solver_norm,
        **kwargs,
    )
