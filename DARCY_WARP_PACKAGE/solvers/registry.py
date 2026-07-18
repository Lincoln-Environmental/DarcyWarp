# SPDX-License-Identifier: AGPL-3.0-only
"""2D solver selection, aliases, and formulation validation."""

from __future__ import annotations

from typing import Any

from .base import SolverBackend, SolverContext
from .multigrid_kcycle import ConfinedKCycleBackend
from .pcg import ConfinedPCGBackend
from .picard_unconfined import UnconfinedPicardKCycleBackend


_BACKENDS: dict[str, SolverBackend] = {
    "confined_pcg": ConfinedPCGBackend(),
    "confined_kcycle": ConfinedKCycleBackend(),
    "unconfined_picard_kcycle": UnconfinedPicardKCycleBackend(),
}

_ALIASES = {
    "pcg": "confined_pcg",
    "kcycle": "confined_kcycle",
    "multigrid": "confined_kcycle",
    "mg": "confined_kcycle",
    "picard": "unconfined_picard_kcycle",
    "picard_kcycle": "unconfined_picard_kcycle",
}


def available_backends() -> tuple[str, ...]:
    """Return canonical backend names in stable presentation order."""
    return tuple(_BACKENDS)


def canonical_solver_name(solver: str | None, *, formulation: str, default: str) -> str:
    """Resolve compatibility aliases and choose the formulation-safe default."""
    requested = default if solver is None else str(solver)
    name = str(requested).strip().lower()
    # ``kcycle`` historically meant the Picard/K-cycle path for unconfined
    # calls.  Retain that interpretation while exposing the explicit name.
    if formulation == "unconfined" and name in {"kcycle", "multigrid", "mg"}:
        name = "unconfined_picard_kcycle"
    else:
        name = _ALIASES.get(name, name)
    if name not in _BACKENDS:
        choices = ", ".join(available_backends())
        raise ValueError(f"unknown 2D solver backend {requested!r}; choose one of: {choices}.")
    if formulation == "confined" and name == "unconfined_picard_kcycle":
        raise ValueError(
            "solver='unconfined_picard_kcycle' requires formulation='unconfined'."
        )
    if formulation == "unconfined" and name != "unconfined_picard_kcycle":
        raise ValueError(
            "unconfined flow currently requires solver='unconfined_picard_kcycle' "
            "(legacy alias: 'kcycle')."
        )
    return name


def select_backend(
    *,
    solver: str | None,
    formulation: str,
    transient: bool,
    default: str,
) -> SolverBackend:
    """Validate the requested formulation/mode and return its backend."""
    name = canonical_solver_name(solver, formulation=formulation, default=default)
    if transient and name == "confined_pcg":
        raise NotImplementedError(
            "Transient storage is implemented for solver='kcycle' only; use "
            "'confined_kcycle' or 'unconfined_picard_kcycle'."
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
