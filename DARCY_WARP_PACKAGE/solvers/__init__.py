# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit 2D DarcyWarp solver backends and their selection registry."""

from .context import (
    ConvergenceControls,
    MultigridHierarchy,
    SolverContext,
    SolverWorkspace,
)
from .resources import SolverResourceOwner
from .capabilities import CAPABILITIES, SolverCapabilities
from .regression import assert_diagnostic_schema_and_values, compare_heads, normalize_diagnostics
from .registry import available_backends, canonical_solver_name, select_backend, solve_selected

__all__ = [
    "SolverContext",
    "ConvergenceControls",
    "MultigridHierarchy",
    "SolverResourceOwner",
    "SolverWorkspace",
    "SolverCapabilities",
    "CAPABILITIES",
    "assert_diagnostic_schema_and_values",
    "compare_heads",
    "normalize_diagnostics",
    "available_backends",
    "canonical_solver_name",
    "select_backend",
    "solve_selected",
]
