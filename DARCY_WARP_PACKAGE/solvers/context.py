# SPDX-License-Identifier: AGPL-3.0-only
"""Typed borrowing context passed from the model to 2D solver backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

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
    """Zero-copy solver input and backend implementation hooks.

    Fields are borrowed model-owned references.  The hooks name the active
    implementation without granting resource ownership to a backend.  They are
    deliberately narrow compatibility seams while algorithm bodies move out of
    the model container.
    """

    grid: GridSpec
    fields: OperatorFields
    boundaries: BoundaryFields
    storage: StorageState
    hierarchy: MultigridHierarchy
    workspace: SolverWorkspace
    convergence: ConvergenceControls
    run_pcg: Callable[..., Any]
    run_kcycle: Callable[..., Any]
    run_transient: Callable[..., Any] | None = None

    @property
    def formulation(self) -> str:
        return self.convergence.formulation

    @property
    def transient(self) -> bool:
        return self.convergence.transient

    @property
    def device(self) -> str:
        return self.grid.device
