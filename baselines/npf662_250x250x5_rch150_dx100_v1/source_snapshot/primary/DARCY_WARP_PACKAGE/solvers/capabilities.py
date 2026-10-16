# SPDX-License-Identifier: AGPL-3.0-only
"""Solver-package re-export of the import-light capability metadata."""

from DARCY_WARP_PACKAGE.solver_capabilities import (
    ALIASES,
    CAPABILITIES,
    SolverCapabilities,
    canonical_name,
)

__all__ = ["ALIASES", "CAPABILITIES", "SolverCapabilities", "canonical_name"]
