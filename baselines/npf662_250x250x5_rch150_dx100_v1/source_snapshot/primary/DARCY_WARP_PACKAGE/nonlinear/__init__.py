# SPDX-License-Identifier: AGPL-3.0-only
"""Authoritative 2D unconfined nonlinear operator (stage 1).

Neutral, backend-independent infrastructure that evaluates the true nonlinear
DarcyWarp equation directly from hydraulic head.  No solver backend is
registered here; this module is consumed later by semismooth-Newton and FAS
backends.  The trusted ``unconfined_picard_kcycle`` backend is unchanged.
"""

from .context import (
    NonlinearBoundaryFields,
    NonlinearFlowFields,
    NonlinearGrid,
    NonlinearOperatorContext2D,
    NonlinearSourceField,
    NonlinearStorageFields,
    from_arrays,
    from_unconfined_solve_inputs,
)
from .operator import (
    FrozenPicardOperator2D,
    NonlinearOperator2D,
    ResidualNorms,
    StorageTerms2D,
)

__all__ = [
    "NonlinearOperatorContext2D",
    "NonlinearGrid",
    "NonlinearFlowFields",
    "NonlinearBoundaryFields",
    "NonlinearSourceField",
    "NonlinearStorageFields",
    "NonlinearOperator2D",
    "StorageTerms2D",
    "FrozenPicardOperator2D",
    "ResidualNorms",
    "from_arrays",
    "from_unconfined_solve_inputs",
]
