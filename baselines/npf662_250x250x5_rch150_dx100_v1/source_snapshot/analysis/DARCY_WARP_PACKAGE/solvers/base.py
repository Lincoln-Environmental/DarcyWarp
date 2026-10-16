# SPDX-License-Identifier: AGPL-3.0-only
"""Stable, dependency-explicit interfaces for 2D solver backends.

The model owns field arrays, Warp allocations, the multigrid hierarchy, and
their lifetime.  A backend receives only the operations it is allowed to use;
this keeps future nonlinear methods independent from model construction.
"""

from __future__ import annotations

from typing import Any, Protocol

from .context import SolverContext


class SolverBackend(Protocol):
    """A named numerical backend selected by :mod:`.registry`."""

    name: str

    def solve(self, context: "SolverContext", **kwargs: Any) -> Any:
        """Run the backend using model-owned resources exposed by *context*."""


__all__ = ["SolverBackend", "SolverContext"]
