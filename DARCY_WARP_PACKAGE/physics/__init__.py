# SPDX-License-Identifier: AGPL-3.0-only
"""Shared 2D groundwater operator, storage, and budget definitions."""

from .operator_data import (
    BoundaryFields,
    GridSpec,
    OperatorFields,
    StorageState,
    compute_ghb_factor_from_raw_fields,
)
from .storage_2d import (
    exact_unconfined_storage_terms,
    secant_specific_storage_coeff,
    secant_specific_yield_coeff,
    specific_storage_potential,
)

__all__ = [
    "BoundaryFields",
    "GridSpec",
    "OperatorFields",
    "StorageState",
    "compute_ghb_factor_from_raw_fields",
    "exact_unconfined_storage_terms",
    "secant_specific_storage_coeff",
    "secant_specific_yield_coeff",
    "specific_storage_potential",
]
