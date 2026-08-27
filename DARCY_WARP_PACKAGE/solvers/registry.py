# SPDX-License-Identifier: AGPL-3.0-only
"""2D solver selection, aliases, and formulation validation."""

from __future__ import annotations

import warnings
from typing import Any

from .base import SolverBackend, SolverContext
from .multigrid_kcycle import ConfinedKCycleBackend
from .pcg import ConfinedPCGBackend
from .picard_unconfined import UnconfinedPicardKCycleBackend
from .semismooth_newton import UnconfinedSemismoothNewtonKCycleBackend
from .fas import UnconfinedFASBackend
from .capabilities import CAPABILITIES, canonical_name


_BACKENDS: dict[str, SolverBackend] = {
    "confined_pcg": ConfinedPCGBackend(),
    "confined_kcycle": ConfinedKCycleBackend(),
    "unconfined_picard_kcycle": UnconfinedPicardKCycleBackend(),
    "unconfined_semismooth_newton_kcycle": UnconfinedSemismoothNewtonKCycleBackend(),
    "unconfined_fas": UnconfinedFASBackend(),
}


def available_backends() -> tuple[str, ...]:
    """Return canonical backend names in stable presentation order."""
    return tuple(_BACKENDS)


def canonical_solver_name(solver: str | None, *, formulation: str, default: str) -> str:
    """Resolve compatibility aliases and choose the formulation-safe default."""
    return canonical_name(solver, formulation=formulation, default=default)


def select_backend(
    *,
    solver: str | None,
    formulation: str,
    transient: bool,
    default: str,
) -> SolverBackend:
    """Validate the requested formulation/mode and return its backend."""
    name = canonical_solver_name(solver, formulation=formulation, default=default)
    capability = CAPABILITIES[name]
    if capability.experimental:
        warnings.warn(
            f"solver={name!r} is experimental and not validated for production runs.",
            stacklevel=2,
        )
    if transient and not capability.supports_transient:
        raise NotImplementedError(
            f"solver={name!r} does not support transient storage; select a "
            "backend whose capability metadata declares supports_transient=True."
        )
    return _BACKENDS[name]


def solve_selected(context: SolverContext, *, solver: str | None, default: str, **kwargs: Any):
    """Select and invoke a backend without exposing model implementation details."""
    backend = select_backend(
        solver=solver,
        formulation=context.formulation,
        transient=context.transient,
        default=default,
    )
    return backend.solve(context, **kwargs)
