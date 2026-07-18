# SPDX-License-Identifier: AGPL-3.0-only
"""Unconfined Picard iteration using the shared geometric K-cycle backend."""

from __future__ import annotations

from typing import Any

from .base import SolverContext


class UnconfinedPicardKCycleBackend:
    """Trusted Picard implementation; linear work is supplied by K-cycle."""

    name = "unconfined_picard_kcycle"

    def solve(self, context: SolverContext, **kwargs: Any):
        kwargs["unconfined"] = True
        return context.run_kcycle(**kwargs)
