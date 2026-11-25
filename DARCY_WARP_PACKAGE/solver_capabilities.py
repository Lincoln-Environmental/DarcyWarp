# SPDX-License-Identifier: AGPL-3.0-or-later
"""Import-light 2D solver capability metadata shared by factory and registry."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SolverCapabilities:
    """Formulation and runtime support declared for one solver backend.

    :param formulations: Supported flow formulation names.
    :param supports_transient: Whether storage can be included.
    :param experimental: Whether production qualification is incomplete.
    :param production_default: Whether this is the default production backend.
    :param supports_production_period_driver: Whether multi-period dispatch is supported.
    """

    formulations: frozenset[str]
    supports_transient: bool
    experimental: bool = False
    production_default: bool = False
    supports_production_period_driver: bool = False


CAPABILITIES: dict[str, SolverCapabilities] = {
    "confined_pcg": SolverCapabilities(frozenset({"confined"}), False),
    "confined_kcycle": SolverCapabilities(
        frozenset({"confined"}),
        True,
        production_default=True,
    ),
    "unconfined_picard_kcycle": SolverCapabilities(
        frozenset({"unconfined"}),
        True,
        production_default=True,
        supports_production_period_driver=True,
    ),
    "unconfined_semismooth_newton_kcycle": SolverCapabilities(
        frozenset({"unconfined"}),
        True,
        supports_production_period_driver=True,
    ),
    "unconfined_fas": SolverCapabilities(
        frozenset({"unconfined"}),
        True,
        experimental=True,
        supports_production_period_driver=True,
    ),
}

ALIASES: dict[str, str] = {
    "pcg": "confined_pcg",
    "kcycle": "confined_kcycle",
    "multigrid": "confined_kcycle",
    "mg": "confined_kcycle",
    "picard": "unconfined_picard_kcycle",
    "picard_kcycle": "unconfined_picard_kcycle",
}


def canonical_name(solver: str | None, *, formulation: str, default: str) -> str:
    """Resolve and validate a user-facing solver name.

    :param solver: Backend name or alias, or None to use default.
    :param formulation: Requested confined or unconfined formulation.
    :param default: Backend name used when solver is omitted.
    :returns: Canonical compatible solver name.
    :raises ValueError: If the name or requested formulation is unsupported.
    """
    requested = default if solver is None else str(solver)
    name = str(requested).strip().lower()
    if formulation == "unconfined" and name in {"kcycle", "multigrid", "mg"}:
        name = "unconfined_picard_kcycle"
    else:
        name = ALIASES.get(name, name)
    if name not in CAPABILITIES:
        choices = ", ".join(CAPABILITIES)
        raise ValueError(f"unknown 2D solver backend {requested!r}; choose one of: {choices}.")
    capability = CAPABILITIES[name]
    if formulation not in capability.formulations:
        if formulation == "confined" and name.startswith("unconfined_"):
            raise ValueError(f"solver={name!r} requires formulation='unconfined'.")
        if formulation == "unconfined":
            raise ValueError(
                "unconfined flow currently requires an unconfined backend "
                "(production default: 'unconfined_picard_kcycle'; legacy alias: 'kcycle')."
            )
        raise ValueError(f"solver={name!r} does not support formulation={formulation!r}.")
    return name


__all__ = ["ALIASES", "CAPABILITIES", "SolverCapabilities", "canonical_name"]
