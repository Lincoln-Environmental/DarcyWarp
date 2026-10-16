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
    experimental_workspaces: dict[str, Any] | None = None
    closed: bool = False

    def refresh(self, *, hierarchy: Any, work: Any, cuda_graph: Any) -> None:
        if self.closed:
            raise RuntimeError("cannot refresh resources after close()")
        self.hierarchy = hierarchy
        self.work = work
        self.cuda_graph = cuda_graph
        if self.experimental_workspaces is None:
            self.experimental_workspaces = {}

    def get_experimental_workspace(self, name: str) -> Any:
        if self.experimental_workspaces is None:
            self.experimental_workspaces = {}
        return self.experimental_workspaces.get(str(name))

    def set_experimental_workspace(self, name: str, workspace: Any) -> None:
        if self.closed:
            raise RuntimeError("cannot attach a workspace after close()")
        if self.experimental_workspaces is None:
            self.experimental_workspaces = {}
        previous = self.experimental_workspaces.get(str(name))
        if previous is not None and previous is not workspace and hasattr(previous, "close"):
            previous.close()
        self.experimental_workspaces[str(name)] = workspace

    def release(self) -> None:
        """Drop ownership references; allocation release remains model-driven."""
        if self.experimental_workspaces is not None:
            for workspace in self.experimental_workspaces.values():
                if hasattr(workspace, "close"):
                    workspace.close()
            self.experimental_workspaces.clear()
        self.cuda_graph = None
        self.hierarchy = None
        self.work = None
        self.experimental_workspaces = None
        self.closed = True
