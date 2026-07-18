# SPDX-License-Identifier: AGPL-3.0-only
"""Explicit ownership metadata for model-managed Warp resources."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class SolverResourceOwner:
    """Model-owned resource inventory exposed read-only to backend contexts.

    Backends receive references through ``SolverContext`` and must never call
    ``release``.  The model invokes it from its idempotent ``close`` path after
    it has broken graph references and dropped all hierarchy/work references.
    """

    device: str
    hierarchy: Any = None
    work: Any = None
    cuda_graph: Any = None
    closed: bool = False

    def refresh(self, *, hierarchy: Any, work: Any, cuda_graph: Any) -> None:
        if self.closed:
            raise RuntimeError("cannot refresh resources after close()")
        self.hierarchy = hierarchy
        self.work = work
        self.cuda_graph = cuda_graph

    def release(self) -> None:
        """Drop ownership references; allocation release remains model-driven."""
        self.cuda_graph = None
        self.hierarchy = None
        self.work = None
        self.closed = True
