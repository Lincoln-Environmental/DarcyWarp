# SPDX-License-Identifier: AGPL-3.0-only
"""Typed borrowing context passed from the model to 2D solver backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from DARCY_WARP_PACKAGE.physics.operator_data import (
    BoundaryFields,
    GridSpec,
    OperatorFields,
    StorageState,
)


@dataclass(frozen=True, slots=True)
class MultigridHierarchy:
    """Borrowed hierarchy and persistent multigrid work storage."""

    levels: Any
    work: Any
    coarsening_diagnostics: Any


@dataclass(frozen=True, slots=True)
class SolverWorkspace:
    """Borrowed PCG and K-cycle workspace references owned by the model."""

    pcg_buffers: Mapping[str, Any]
    cuda_graph: Any
    transient_replay_counters: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class ConvergenceControls:
    """Immutable formulation/mode information shared by a backend call."""

    formulation: str
    transient: bool


@dataclass(frozen=True, slots=True)
class SolverContext:
    """Zero-copy solver input and explicit model ownership.

    Fields are borrowed model-owned references. ``model`` exists only for the
    small set of established assembly/resource methods that must remain owned
    by the model; solver implementations never release or replace it.
    """

    grid: GridSpec
    fields: OperatorFields
    boundaries: BoundaryFields
    storage: StorageState
    hierarchy: MultigridHierarchy
    workspace: SolverWorkspace
    convergence: ConvergenceControls
    model: Any

    @property
    def formulation(self) -> str:
        return self.convergence.formulation

    @property
    def transient(self) -> bool:
        return self.convergence.transient

    @property
    def device(self) -> str:
        return self.grid.device
