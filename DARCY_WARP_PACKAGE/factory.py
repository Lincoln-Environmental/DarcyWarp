# SPDX-License-Identifier: AGPL-3.0-or-later
"""
Factory for creating structured 2D or 3D Warp Darcy solvers.

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
    dx: float,
    dy: float | None = None,
    device: str = "cuda:0",
    **kwargs: Any,
):
    """Create a structured groundwater model on a Warp CPU or CUDA device.

    :param dim: Number of spatial dimensions, either 2 or 3.
    :param solver: Solver backend. In 2D, ``kcycle`` selects the default
        backend for the formulation supplied at solve time. An explicit
        unconfined backend name, such as ``unconfined_picard_kcycle``,
        becomes the default for later solves. In 3D, choose ``kcycle``
        or ``chebyshev``. See :doc:`/capabilities` for supported combinations.
    :param nx: Number of grid columns along the x axis.
    :param ny: Number of grid rows along the y axis.
    :param dx: Cell width along the x axis, in model length units.
    :param dy: Cell width along the y axis, in the same units as ``dx``.
        Defaults to ``dx``. The 2D solver requires square cells.
    :param device: Warp execution device, for example ``"cpu"`` or ``"cuda:0"``.
    :param kwargs: Additional keyword arguments passed to the selected model
        constructor. A 3D model requires ``nz`` (layer count) and ``dz``
        (vertical cell thickness, in model length units).
    :returns: A ``WarpDarcySolver`` for 2D or a ``WarpDarcySolver3D`` for 3D.
        Use its build methods to supply the groundwater fields before solving.
    :raises ValueError: If the dimension or backend is unsupported, a required
        3D argument is missing, or the grid arguments conflict with the dimension.
    """
    if dim == 3:
        solver_norm = str(solver).strip().lower()
        if solver_norm not in {"kcycle", "chebyshev"}:
            raise ValueError("3D solver must be 'kcycle' or 'chebyshev'")
        for argument in ("nz", "dz"):
            if kwargs.get(argument) is None:
                raise ValueError(f"{argument} is required for the 3D solver")

        from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

        return WarpDarcySolver3D(
            nx=nx, ny=ny, dx=dx, dy=dy, device=device, solver=solver_norm, **kwargs
        )
    if dim != 2:
        raise ValueError("dim must be 2 or 3")
    if "nz" in kwargs or "dz" in kwargs:
        raise ValueError("nz/dz are 3D-only arguments.")
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
