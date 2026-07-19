# SPDX-License-Identifier: AGPL-3.0-only
"""Cached fine-grid operator state for the experimental semismooth-Newton backend.

``NewtonOperatorWorkspace2D`` is the Newton analogue of the FAS
:class:`FASWorkspace` reuse/refresh split.  It owns a single persistent
:class:`DARCY_WARP_PACKAGE.nonlinear.NonlinearOperator2D` so the backend no
longer rebuilds the operator (and its ~29 device allocations plus ~12 field
uploads) on every solve.  Reuse is decided by structural inputs only; per
timestep state (previous accepted head, dt, source field, Sy, Ss) is refreshed
in place through :meth:`NonlinearOperator2D.update_transient_state` without any
reallocation.

Lifetime is managed by :class:`solvers.resources.SolverResourceOwner`:
``set_experimental_workspace`` closes a previous incompatible workspace, and
``release`` (model close) closes this one.  The backend therefore never closes
the operator itself between solves.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from DARCY_WARP_PACKAGE.nonlinear import NonlinearOperator2D


def _capture_static(context: Any) -> dict[str, Any]:
    """Snapshot the structural inputs that decide workspace reuse.

    Timestep-dependent state (previous head, source field, dt, Sy, Ss) is
    intentionally excluded; it is handled by :meth:`refresh`.
    """
    flow = context.flow
    bnd = context.boundaries
    grid = context.grid
    return {
        "shape": (int(grid.ny), int(grid.nx)),
        "dx": float(grid.dx),
        "has_top": flow.ztop is not None,
        "K": np.array(flow.K, dtype=np.float64, copy=True),
        "zbot": np.array(flow.zbot, dtype=np.float64, copy=True),
        "ztop": None if flow.ztop is None else np.array(flow.ztop, dtype=np.float64, copy=True),
        "active": np.array(bnd.active, dtype=np.int32, copy=True),
        "dirichlet_mask": np.array(bnd.dirichlet_mask, dtype=np.int32, copy=True),
        "dirichlet_values": np.array(bnd.dirichlet_values, dtype=np.float64, copy=True),
        "ghb_mask": np.array(bnd.ghb_mask, dtype=np.int32, copy=True),
        "ghb_factor": np.array(bnd.ghb_factor, dtype=np.float64, copy=True),
        "ghb_external_head": np.array(bnd.ghb_external_head, dtype=np.float64, copy=True),
    }


def _static_equal(a: dict[str, Any], b: dict[str, Any]) -> bool:
    if a["shape"] != b["shape"] or a["has_top"] != b["has_top"]:
        return False
    if float(a["dx"]) != float(b["dx"]):
        return False
    names = (
        "K", "zbot", "active", "dirichlet_mask", "dirichlet_values",
        "ghb_mask", "ghb_factor", "ghb_external_head",
    )
    for name in names:
        if not np.array_equal(a[name], b[name]):
            return False
    if a["has_top"] and not np.array_equal(a["ztop"], b["ztop"]):
        return False
    return True


class NewtonOperatorWorkspace2D:
    """Model-owned fine operator, reusable across compatible Newton solves."""

    def __init__(
        self,
        *,
        context: Any,
        transient: bool,
        min_sat: float,
        device: str,
    ):
        self.operator = NonlinearOperator2D(context)
        self.transient = bool(transient)
        self.min_sat = float(min_sat)
        self.device = str(device)
        self.static_inputs = _capture_static(context)
        self.refresh_count = 0
        self.closed = False

    def compatible(
        self,
        *,
        context: Any,
        transient: bool,
        min_sat: float,
        device: str,
    ) -> bool:
        """True when every structural input is unchanged.

        Only static data decides reuse (grid shape, transient regime, min_sat,
        device, conductivity, geometry, active and prescribed masks, prescribed
        and GHB fields).  Previous head, source field, dt, Sy and Ss are
        timestep state handled by :meth:`refresh`.
        """
        if self.closed or self.transient != bool(transient):
            return False
        if self.min_sat != float(min_sat) or self.device != str(device):
            return False
        return _static_equal(self.static_inputs, _capture_static(context))

    def refresh(
        self,
        *,
        head_prev: Any | None = None,
        dt: float | None = None,
        source_rate: Any | None = None,
        sy: float | None = None,
        ss: float | None = None,
    ) -> None:
        """Re-point the cached operator at a new timestep without reallocation.

        Delegates to :meth:`NonlinearOperator2D.update_transient_state`, which
        overwrites the persistent device mirrors (previous accepted head, source
        field) and per-launch scalars (dt, Sy, Ss) that the kernels read.
        """
        if self.closed:
            raise RuntimeError("cannot refresh a closed NewtonOperatorWorkspace2D.")
        if not self.transient and (head_prev is not None or dt is not None):
            raise ValueError("head_prev/dt refresh requires a transient workspace.")
        self.operator.update_transient_state(
            head_prev=head_prev,
            dt=dt,
            source_rate=source_rate,
            sy=sy,
            ss=ss,
        )
        self.refresh_count += 1

    def close(self) -> None:
        if self.closed:
            return
        if self.operator is not None:
            self.operator.close()
        self.operator = None
        self.closed = True


__all__ = ["NewtonOperatorWorkspace2D"]
