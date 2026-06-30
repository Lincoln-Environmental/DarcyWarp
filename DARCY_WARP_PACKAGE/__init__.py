# SPDX-License-Identifier: AGPL-3.0-only
"""
DarcyWarp — GPU-accelerated Darcy flow solvers and benchmarks.
"""

from __future__ import annotations

# Lazy exports: importing the package must not require optional dependencies
# such as warp or MODFLOW 6 to be installed until a solver is actually created.

__all__ = [
    "create_solver",
    "WarpDarcySolver",
    "WarpDarcySolver3D",
]


def __getattr__(name: str):
    if name == "create_solver":
        from DARCY_WARP_PACKAGE.factory import create_solver

        return create_solver
    if name == "WarpDarcySolver":
        from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

        return WarpDarcySolver
    if name == "WarpDarcySolver3D":
        from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

        return WarpDarcySolver3D
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
