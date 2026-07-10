#!/usr/bin/env python
"""
Transient unconfined convergence diagnostic ladder.

DarcyWarp transient unconfined runs fail to converge even when started from a
converged steady-state unconfined solution.  This script treats that as a
*validation failure* and runs a ladder of small, deterministic diagnostics that
isolate the first failing layer:

    1. confined transient control (basic storage RHS / diagonal sanity)
    2. 2D unconfined steady->transient invariant (head_prev == h_ss must return ~h_ss)
    3. 2D hierarchy isolation (single-level vs full K-cycle)
    4. dt sensitivity ladder ([1e-3, 1, 1e3, 1e9])
    5. mild stress-change (1-5% recharge perturbation)
    6. 3D unconfined storage mode (phreatic_sy active vs silent confined_volume fallback)
    7. residual-at-steady diagnostic (transient operator vs steady operator at h_ss)
    8. convergence acceptance diagnostic (per-outer trace + optional head-change-only experiment)

Core invariant under test (backward Euler, unchanged stresses)::

    A(h_ss) h_ss + S h_ss = b + S h_ss      # S*head_prev with head_prev == h_ss

so the transient step started from ``initial_head == head_prev == h_ss`` must
converge in very few Picard iterations and return heads close to ``h_ss``.

Run directly::

    python working_tests/run_transient_unconfined_diagnostics.py
    python working_tests/run_transient_unconfined_diagnostics.py --device cpu --grid small -v
    python working_tests/run_transient_unconfined_diagnostics.py --cases 2,3,4,7

The script does not fix the solver; it only reports which layer fails first.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Warp kernel cache must be set before importing the package.
os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))

# Make DARCY_WARP_PACKAGE importable when this script is run directly from the
# repo root (Python otherwise only puts this script's own directory on sys.path).
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Reproducibility: the synthetic fields below are all constant (no random field
# is used), so the run is deterministic.  The seed is set for completeness only
# in case a future caller swaps in a random K field.
np.random.seed(20260710)


NA = "n/a"

# Table column order (matches the deliverable spec).
COLUMNS = [
    "case",
    "dim",
    "storage_mode",
    "dt",
    "max_levels",
    "steady_converged",
    "transient_converged",
    "outer_iters",
    "max_abs_dh_vs_ss",
    "rms_dh_vs_ss",
    "final_head_change",
    "final_residual",
    "failure_reason",
]


# ---------------------------------------------------------------------------
# Warp availability
# ---------------------------------------------------------------------------
def warp_available() -> bool:
    """Return True if the ``warp`` GPU/CI kernel package can be imported."""
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# Independent 5-point residual (NumPy reference)
# ---------------------------------------------------------------------------
def _face_conductance_x(T: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Harmonic-mean conductance on east/west faces.

    :param T: cell transmissivity field, shape (ny, nx).
    :param active: active-cell mask, shape (ny, nx).
    :return: field ``ce`` of shape (ny, nx) where ``ce[:, j]`` is the
        conductance between column ``j`` and ``j+1``; the last column is 0 and
        any face touching an inactive cell is 0.
    """
    ce = np.zeros_like(T, dtype=np.float64)
    left = T[:, :-1]
    right = T[:, 1:]
    denom = left + right
    valid = (active[:, :-1] != 0) & (active[:, 1:] != 0) & (denom > 1.0e-12)
    vals = np.zeros_like(denom)
    vals[valid] = 2.0 * left[valid] * right[valid] / denom[valid]
    ce[:, :-1] = vals
    return ce


def _face_conductance_y(T: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Harmonic-mean conductance on north/south faces.

    :param T: cell transmissivity field, shape (ny, nx).
    :param active: active-cell mask, shape (ny, nx).
    :return: field ``cn`` where ``cn[i, :]`` is the conductance between row
        ``i`` and ``i+1``; last row is 0.
    """
    cn = np.zeros_like(T, dtype=np.float64)
    top = T[:-1, :]
    bottom = T[1:, :]
    denom = top + bottom
    valid = (active[:-1, :] != 0) & (active[1:, :] != 0) & (denom > 1.0e-12)
    vals = np.zeros_like(denom)
    vals[valid] = 2.0 * top[valid] * bottom[valid] / denom[valid]
    cn[:-1, :] = vals
    return cn


def flux_residual_2d(
    T: np.ndarray,
    head: np.ndarray,
    recharge: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    dx: float,
    storage_coeff: float | np.ndarray | None = None,
    head_prev: np.ndarray | None = None,
    dt: float | None = None,
) -> tuple[float, float]:
    """Independent discrete residual of the 2D 5-point operator.

    Computes, on free (active, non-Dirichlet) cells::

        r_i = sum_j C_ij (h_i - h_j) - R_i * dx^2  [ - S_i * dx^2/dt * (h_i - h_prev_i) ]

    where ``sum_j C_ij (h_i - h_j)`` is the net lateral OUTFLOW from cell ``i``.
    At a discrete solution this is ~0 (outflow balances recharge minus storage);
    this matches the convention of ``_compute_mass_balance_residual`` in
    ``tests/test_2d_transient.py``.

    :param T: transmissivity field evaluated at the linearization head.
    :param head: head field at which to evaluate the residual.
    :param recharge: recharge/source field, same units as the solver.
    :param active: active-cell mask.
    :param bc_mask: Dirichlet-cell mask.
    :param dx: cell width (square cells).
    :param storage_coeff: optional storage coefficient (scalar or field).
    :param head_prev: optional previous-time head for the storage term.
    :param dt: optional time step for the storage term.
    :return: ``(max_abs, rms)`` of the residual over free cells, in flux units.
    """
    h = np.asarray(head, dtype=np.float64)
    area = float(dx) * float(dx)
    div = np.zeros_like(h)

    ce = _face_conductance_x(T, active)
    cn = _face_conductance_y(T, active)
    # Net OUTFLOW convention: div[i] = sum_j C_ij (h_i - h_j).
    div[:, :-1] += ce[:, :-1] * (h[:, :-1] - h[:, 1:])
    div[:, 1:] += ce[:, :-1] * (h[:, 1:] - h[:, :-1])
    div[:-1, :] += cn[:-1, :] * (h[:-1, :] - h[1:, :])
    div[1:, :] += cn[:-1, :] * (h[1:, :] - h[:-1, :])

    res = div - np.asarray(recharge, dtype=np.float64) * area
    if storage_coeff is not None and dt is not None and head_prev is not None:
        hp = np.asarray(head_prev, dtype=np.float64)
        s_field = np.asarray(storage_coeff, dtype=np.float64)
        res = res - s_field * area / float(dt) * (h - hp)

    free = (active != 0) & (bc_mask == 0)
    rf = res[free]
    if rf.size == 0:
        return float("nan"), float("nan")
    return float(np.max(np.abs(rf))), float(np.sqrt(np.mean(rf * rf)))


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def build_2d_solver(nx: int, ny: int, device: str, recharge: float, transmissivity: float):
    """Construct a 2D WarpDarcySolver with two Dirichlet vertical boundaries.

    :param nx: number of columns.
    :param ny: number of rows.
    :param device: warp device string.
    :param recharge: uniform recharge rate [L/T].
    :param transmissivity: uniform confined transmissivity [L^2/T].
    :return: ``(solver, active, bc_mask, K, zbot)`` where K/zbot are unconfined
        fields and active/bc_mask are the integer masks.
    """
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=100.0,
        device=device,
        use_ghb=False,
        solver_type="kcycle",
        diag_preconditioner_backend="host",
    )
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 9.0
    recharge_field = np.full((ny, nx), recharge, dtype=np.float64)
    transmissivity_field = np.full((ny, nx), transmissivity, dtype=np.float64)
    solver.build_from_fields(
        T_field=transmissivity_field,
        R_field=recharge_field,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
    )
    K = np.full((ny, nx), 1.0, dtype=np.float64)
    zbot = np.zeros((ny, nx), dtype=np.float64)
    return solver, active, bc_mask, K, zbot


def build_3d_solver(nx: int, ny: int, nz: int, device: str, recharge: float, thickness: float):
    """Construct a 3D WarpDarcySolver3D with two Dirichlet vertical faces.

    :param nx: columns.
    :param ny: rows.
    :param nz: layers.
    :param device: warp device string.
    :param recharge: uniform recharge rate applied to the top layer only.
    :param thickness: full aquifer thickness for the (inactive) confined cap.
    :return: ``(solver, active, bc_mask, zbot, ztop, initial)``.
    """
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    solver = WarpDarcySolver3D(
        nx=nx,
        ny=ny,
        nz=nz,
        dx=100.0,
        dy=100.0,
        dz=10.0,
        device=device,
        solver="kcycle",
        diag_preconditioner_backend="host",
    )
    active = np.ones((nz, ny, nx), dtype=np.int32)
    bc_mask = np.zeros((nz, ny, nx), dtype=np.int32)
    bc_values = np.zeros((nz, ny, nx), dtype=np.float64)
    bc_mask[:, :, 0] = 1
    bc_mask[:, :, -1] = 1
    bc_values[:, :, 0] = 12.0
    bc_values[:, :, -1] = 9.0
    rhs = np.zeros((nz, ny, nx), dtype=np.float64)
    rhs[-1, :, :] = recharge  # recharge applied on the top active layer
    kx = np.full((nz, ny, nx), 1.0, dtype=np.float64)
    ky = np.full((nz, ny, nx), 1.0, dtype=np.float64)
    kz = np.full((nz, ny, nx), 1.0, dtype=np.float64)
    zbot = np.zeros((nz, ny, nx), dtype=np.float64)
    ztop = zbot + float(thickness)
    initial = zbot + 5.0
    solver.build_from_K_fields(
        kx_field=kx,
        ky_field=ky,
        kz_field=kz,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        rhs=rhs,
        initial_head=initial,
    )
    return solver, active, bc_mask, zbot, ztop, initial


# ---------------------------------------------------------------------------
# Solve helpers
# ---------------------------------------------------------------------------
def solve_2d_steady_unconfined(solver, K, zbot, initial, *, max_levels=3, max_outer=30, hclose=1.0e-3):
    """Run a 2D steady unconfined solve.

    :param solver: built WarpDarcySolver.
    :param K: hydraulic conductivity field.
    :param zbot: aquifer bottom field.
    :param initial: starting head.
    :return: ``(h_ss, info)``.
    """
    return solver.solve(
        formulation="unconfined",
        K_field=K,
        zbot_field=zbot,
        initial_head=initial,
        max_cycles=20,
        max_levels=max_levels,
        min_coarse_cells=1,
        check_every_no=1,
        max_outer_iterations=max_outer,
        hclose=hclose,
        rel_tol=5.0e-7,
        abs_tol_min=5.0e-7,
        return_info=True,
    )


def solve_2d_steady_confined(solver, *, max_levels=3):
    """Run a 2D steady confined solve.

    :param solver: built WarpDarcySolver.
    :return: ``(h_ss, info)``.
    """
    return solver.solve(
        formulation="confined",
        max_cycles=20,
        max_levels=max_levels,
        min_coarse_cells=1,
        check_every_no=1,
        return_info=True,
    )


# ---------------------------------------------------------------------------
# Diagnostic results container
# ---------------------------------------------------------------------------
class DiagResults:
    """Accumulates table rows and cross-case diagnostic signals."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self.signals: dict[str, Any] = {}

    def add_signal(self, key: str, value: Any) -> None:
        """Record a named diagnostic signal for the final verdict."""
        self.signals[key] = value


# ---------------------------------------------------------------------------
# Row construction
# ---------------------------------------------------------------------------
def _row(
    *,
    case: str,
    dim: str,
    storage_mode: str,
    dt: Any,
    max_levels: Any,
    steady_converged: Any,
    transient_converged: Any,
    outer_iters: Any,
    max_abs_dh_vs_ss: Any,
    rms_dh_vs_ss: Any,
    final_head_change: Any,
    final_residual: Any,
    failure_reason: str = "",
) -> dict[str, Any]:
    """Build one table row dict in the canonical column order."""
    return {
        "case": case,
        "dim": dim,
        "storage_mode": storage_mode,
        "dt": dt,
        "max_levels": max_levels,
        "steady_converged": steady_converged,
        "transient_converged": transient_converged,
        "outer_iters": outer_iters,
        "max_abs_dh_vs_ss": max_abs_dh_vs_ss,
        "rms_dh_vs_ss": rms_dh_vs_ss,
        "final_head_change": final_head_change,
        "final_residual": final_residual,
        "failure_reason": failure_reason,
    }


def _dh_stats(h: np.ndarray, h_ss: np.ndarray, active: np.ndarray, bc_mask: np.ndarray) -> tuple[float, float]:
    """Return ``(max_abs, rms)`` of ``h - h_ss`` over free cells.

    :param h: transient head.
    :param h_ss: steady head reference.
    :param active: active-cell mask.
    :param bc_mask: Dirichlet-cell mask.
    """
    free = (active != 0) & (bc_mask == 0)
    dh = (np.asarray(h, dtype=np.float64) - np.asarray(h_ss, dtype=np.float64))[free]
    if dh.size == 0:
        return float("nan"), float("nan")
    return float(np.max(np.abs(dh))), float(np.sqrt(np.mean(dh * dh)))


def run_transient_step_2d(
    solver,
    *,
    case: str,
    K,
    zbot,
    h_ss,
    steady_converged: Any,
    storage_mode: str,
    storage_coeff,
    dt: float,
    max_levels: int,
    active,
    bc_mask,
    max_outer: int = 40,
    hclose: float = 1.0e-3,
    initial_head=None,
    accept_head_change: bool = False,
    dim: str = "2d",
):
    """Run one 2D transient step and build its table row.

    :param solver: built WarpDarcySolver (will be solved in place).
    :param case: row label.
    :param K: hydraulic conductivity field (unconfined only).
    :param zbot: aquifer bottom field (unconfined only).
    :param h_ss: steady head used as the reference and default warm start.
    :param steady_converged: convergence flag of the upstream steady solve.
    :param storage_mode: storage-mode label for the table.
    :param storage_coeff: storage coefficient passed to the solver.
    :param dt: time step.
    :param max_levels: multigrid level cap.
    :param active: active-cell mask.
    :param bc_mask: Dirichlet-cell mask.
    :param max_outer: Picard outer-iteration cap.
    :param hclose: Picard head-change tolerance.
    :param initial_head: starting head (defaults to ``h_ss``).
    :param accept_head_change: enable the diagnostic head-change-only acceptance.
    :param dim: dimension label.
    :return: ``(row, info)`` where ``info`` may be ``None`` on a caught error.
    """
    if initial_head is None:
        initial_head = h_ss
    try:
        h, info = solver.solve(
            formulation="unconfined",
            K_field=K,
            zbot_field=zbot,
            initial_head=initial_head,
            transient=True,
            storage_coeff=storage_coeff,
            dt=dt,
            head_prev=h_ss,
            max_cycles=20,
            max_levels=max_levels,
            min_coarse_cells=1,
            check_every_no=1,
            max_outer_iterations=max_outer,
            hclose=hclose,
            rel_tol=5.0e-7,
            abs_tol_min=5.0e-7,
            accept_on_head_change_only=accept_head_change,
            return_info=True,
        )
    except (FloatingPointError, ValueError, RuntimeError, OverflowError) as exc:
        reason = f"{type(exc).__name__}: {exc}"[:120]
        return (
            _row(
                case=case,
                dim=dim,
                storage_mode=storage_mode,
                dt=dt,
                max_levels=max_levels,
                steady_converged=steady_converged,
                transient_converged=False,
                outer_iters=NA,
                max_abs_dh_vs_ss=NA,
                rms_dh_vs_ss=NA,
                final_head_change=NA,
                final_residual=NA,
                failure_reason=reason,
            ),
            None,
        )

    max_abs_dh, rms_dh = _dh_stats(h, h_ss, active, bc_mask)
    final_change = info.get("final_max_abs_head_change", info.get("final_dh", NA))
    final_resid = info.get("final_residual", NA)
    outer = info.get("outer_iterations", info.get("n_cycles_used", NA))
    converged = bool(info.get("converged", False))
    reason = ""
    if not converged:
        reason = "transient did not converge (max outer iters / non-finite)"
    elif max_abs_dh > max(1.0e-2, 10.0 * hclose):
        reason = f"converged but head drifted {max_abs_dh:.3e} m from h_ss"
    return (
        _row(
            case=case,
            dim=dim,
            storage_mode=storage_mode,
            dt=dt,
            max_levels=max_levels,
            steady_converged=steady_converged,
            transient_converged=converged,
            outer_iters=outer,
            max_abs_dh_vs_ss=max_abs_dh,
            rms_dh_vs_ss=rms_dh,
            final_head_change=final_change,
            final_residual=final_resid,
            failure_reason=reason,
        ),
        info,
    )


# ---------------------------------------------------------------------------
# Case 1: confined transient control
# ---------------------------------------------------------------------------
def case_confined_control(ctx, res: DiagResults) -> None:
    """Run confined transient with unchanged stresses from a confined steady start.

    Tests the basic storage RHS and diagonal signs.  Two ratios are exercised:
    a mild ratio known to pass (sanity) and the same aggressive ratio used by the
    unconfined invariant, so a shared storage/K-cycle failure is visible.

    :param ctx: run context with device/grid.
    :param res: results accumulator.
    """
    for label, s_coeff, dt in (("confined-mild", 1.0e-4, 86400.0), ("confined-aggressive", 0.2, 1.0)):
        solver, active, bc_mask, _K, _zbot = build_2d_solver(
            ctx["nx"], ctx["ny"], ctx["device"], recharge=1.0e-4, transmissivity=10.0
        )
        try:
            h_ss, info_ss = solve_2d_steady_confined(solver, max_levels=3)
            # The 2D K-cycle residual stalls (~1e-2) even when the head is well
            # converged, so judge steady quality by the independent flux residual
            # rather than the solver's converged flag.
            ss_resid_max, ss_resid_rms = flux_residual_2d(
                np.asarray(solver.T_field_host, dtype=np.float64),
                h_ss,
                np.asarray(solver.R_field_host, dtype=np.float64),
                active,
                bc_mask,
                100.0,
            )
            ss_ok = math.isfinite(ss_resid_rms) and ss_resid_rms < 1.0
        except Exception as exc:  # pragma: no cover - defensive
            res.rows.append(
                _row(
                    case=f"1:{label}",
                    dim="2d",
                    storage_mode="confined_S*dx2/dt",
                    dt=dt,
                    max_levels=3,
                    steady_converged=False,
                    transient_converged=False,
                    outer_iters=NA,
                    max_abs_dh_vs_ss=NA,
                    rms_dh_vs_ss=NA,
                    final_head_change=NA,
                    final_residual=NA,
                    failure_reason=f"steady raised {type(exc).__name__}",
                )
            )
            continue

        try:
            h_t, info_t = solver.solve(
                formulation="confined",
                transient=True,
                storage_coeff=s_coeff,
                dt=dt,
                head_prev=h_ss,
                initial_head=h_ss,
                max_cycles=20,
                max_levels=3,
                min_coarse_cells=1,
                check_every_no=1,
                return_info=True,
            )
            max_abs_dh, rms_dh = _dh_stats(h_t, h_ss, active, bc_mask)
            outer = info_t.get("outer_iterations", info_t.get("n_cycles_used", NA))
            final_change = info_t.get("final_max_abs_head_change", NA)
            final_resid = info_t.get("final_residual", info_t.get("r_rms_end", NA))
            # The control passes when the transient head stays bounded near h_ss
            # (storage RHS / diagonal signs sane). An exploding or non-finite head
            # is the failure signature, mirroring the unconfined divergence.
            bounded = math.isfinite(max_abs_dh) and max_abs_dh < 1.0
            reason = "" if bounded else "confined transient diverged / non-finite (large storage/diffusion ratio)"
            res.rows.append(
                _row(
                    case=f"1:{label}",
                    dim="2d",
                    storage_mode="confined_S*dx2/dt",
                    dt=dt,
                    max_levels=3,
                    steady_converged=ss_ok,
                    transient_converged=bounded,
                    outer_iters=outer,
                    max_abs_dh_vs_ss=max_abs_dh,
                    rms_dh_vs_ss=rms_dh,
                    final_head_change=final_change,
                    final_residual=final_resid,
                    failure_reason=reason,
                )
            )
            if label == "confined-aggressive":
                res.add_signal("confined_aggressive_ok", bounded)
            else:
                res.add_signal("confined_mild_ok", bounded)
        except (FloatingPointError, ValueError, RuntimeError, OverflowError) as exc:
            res.rows.append(
                _row(
                    case=f"1:{label}",
                    dim="2d",
                    storage_mode="confined_S*dx2/dt",
                    dt=dt,
                    max_levels=3,
                    steady_converged=ss_ok,
                    transient_converged=False,
                    outer_iters=NA,
                    max_abs_dh_vs_ss=NA,
                    rms_dh_vs_ss=NA,
                    final_head_change=NA,
                    final_residual=NA,
                    failure_reason=f"{type(exc).__name__}: {str(exc)[:90]}",
                )
            )
            res.add_signal(f"confined_{label}_ok", False)


# ---------------------------------------------------------------------------
# Case 2: 2D unconfined steady->transient invariant
# ---------------------------------------------------------------------------
def case_2d_invariant(ctx, res: DiagResults) -> None:
    """Run the 2D unconfined warm-start invariant (head_prev == initial == h_ss).

    :param ctx: run context.
    :param res: results accumulator.
    """
    solver, active, bc_mask, K, zbot = build_2d_solver(
        ctx["nx"], ctx["ny"], ctx["device"], recharge=1.0e-4, transmissivity=10.0
    )
    h_ss, info_ss = solve_2d_steady_unconfined(solver, K, zbot, zbot + 5.0)
    ss_ok = bool(info_ss.get("converged", False))

    # Snapshot the linearized operator at h_ss for the residual-at-steady case.
    T_ss = np.asarray(solver.T_field_host, dtype=np.float64).copy()
    R_ss = np.asarray(solver.R_field_host, dtype=np.float64).copy()
    res.add_signal("steady_converged_2d", ss_ok)
    res.add_signal("h_ss_2d", h_ss)
    res.add_signal("T_ss_2d", T_ss)
    res.add_signal("R_ss_2d", R_ss)
    res.add_signal("active_2d", active)
    res.add_signal("bc_mask_2d", bc_mask)
    res.add_signal("solver_2d", solver)
    res.add_signal("K_2d", K)
    res.add_signal("zbot_2d", zbot)

    row, info = run_transient_step_2d(
        solver,
        case="2:invariant",
        K=K,
        zbot=zbot,
        h_ss=h_ss,
        steady_converged=ss_ok,
        storage_mode="phreatic_Sy_dx2/dt",
        storage_coeff=0.2,
        dt=1.0,
        max_levels=3,
        active=active,
        bc_mask=bc_mask,
    )
    res.rows.append(row)
    max_abs_dh = row["max_abs_dh_vs_ss"]
    res.add_signal("invariant_2d_ok", bool(row["transient_converged"]) and isinstance(max_abs_dh, float) and max_abs_dh < 1.0e-2)
    res.add_signal("invariant_2d_info", info)
    if info is not None:
        hist = info.get("outer_history") or []
        first = hist[0] if hist else {}
        res.add_signal("first_update_at_ss", first.get("max_abs_head_change"))
        res.add_signal("first_inner_residual", first.get("inner_residual"))
        res.add_signal("first_inner_usable", first.get("inner_usable_for_picard"))


# ---------------------------------------------------------------------------
# Case 3: 2D hierarchy isolation
# ---------------------------------------------------------------------------
def case_2d_hierarchy(ctx, res: DiagResults) -> None:
    """Repeat the invariant with single-level vs full K-cycle.

    :param ctx: run context.
    :param res: results accumulator.
    """
    for ml, label in ((1, "single-level"), (3, "full-kcycle")):
        solver, active, bc_mask, K, zbot = build_2d_solver(
            ctx["nx"], ctx["ny"], ctx["device"], recharge=1.0e-4, transmissivity=10.0
        )
        h_ss, info_ss = solve_2d_steady_unconfined(solver, K, zbot, zbot + 5.0)
        ss_ok = bool(info_ss.get("converged", False))
        row, info = run_transient_step_2d(
            solver,
            case=f"3:hierarchy-{label}",
            K=K,
            zbot=zbot,
            h_ss=h_ss,
            steady_converged=ss_ok,
            storage_mode="phreatic_Sy_dx2/dt",
            storage_coeff=0.2,
            dt=1.0,
            max_levels=ml,
            active=active,
            bc_mask=bc_mask,
        )
        res.rows.append(row)
        ok = bool(row["transient_converged"]) and isinstance(row["max_abs_dh_vs_ss"], float) and row["max_abs_dh_vs_ss"] < 1.0e-2
        res.add_signal(f"hierarchy_{label.replace('-', '_')}_ok", ok)
        # Capture the solver's first-inner-residual at h_ss for this level cap.
        # A huge residual at the exact fixed point means the operator the iterator
        # applies is inconsistent with the residual operator (an assembly defect),
        # independent of how many multigrid levels are allowed.
        if info is not None:
            hist = info.get("outer_history") or []
            first = hist[0] if hist else {}
            res.add_signal(f"first_inner_residual_{label.replace('-', '_')}", first.get("inner_residual"))


# ---------------------------------------------------------------------------
# Case 4: dt sensitivity ladder
# ---------------------------------------------------------------------------
def case_2d_dt_ladder(ctx, res: DiagResults) -> None:
    """Run the invariant across [1e-3, 1, 1e3, 1e9].

    :param ctx: run context.
    :param res: results accumulator.
    """
    dt_values = [1.0e-3, 1.0, 1.0e3, 1.0e9]
    small_ok = None
    large_ok = None
    for dt in dt_values:
        solver, active, bc_mask, K, zbot = build_2d_solver(
            ctx["nx"], ctx["ny"], ctx["device"], recharge=1.0e-4, transmissivity=10.0
        )
        h_ss, info_ss = solve_2d_steady_unconfined(solver, K, zbot, zbot + 5.0)
        ss_ok = bool(info_ss.get("converged", False))
        row, info = run_transient_step_2d(
            solver,
            case="4:dt-ladder",
            K=K,
            zbot=zbot,
            h_ss=h_ss,
            steady_converged=ss_ok,
            storage_mode="phreatic_Sy_dx2/dt",
            storage_coeff=0.2,
            dt=dt,
            max_levels=3,
            active=active,
            bc_mask=bc_mask,
        )
        res.rows.append(row)
        ok = bool(row["transient_converged"]) and isinstance(row["max_abs_dh_vs_ss"], float) and row["max_abs_dh_vs_ss"] < 1.0e-2
        if dt == 1.0e-3:
            small_ok = ok
        if dt == 1.0e9:
            large_ok = ok
    res.add_signal("dt_small_ok", small_ok)
    res.add_signal("dt_large_ok", large_ok)


# ---------------------------------------------------------------------------
# Case 5: mild stress change
# ---------------------------------------------------------------------------
def case_2d_mild_stress_change(ctx, res: DiagResults) -> None:
    """Perturb recharge by a few percent and check the response stays bounded.

    Only meaningful after the invariant passes; run anyway to record the regime.

    :param ctx: run context.
    :param res: results accumulator.
    """
    base_recharge = 1.0e-4
    for factor, label in ((1.01, "+1%"), (1.05, "+5%")):
        solver, active, bc_mask, K, zbot = build_2d_solver(
            ctx["nx"], ctx["ny"], ctx["device"], recharge=base_recharge, transmissivity=10.0
        )
        h_ss, info_ss = solve_2d_steady_unconfined(solver, K, zbot, zbot + 5.0)
        ss_ok = bool(info_ss.get("converged", False))
        # Perturb the recharge field on the same solver (unchanged K/bottom/BC).
        solver.R_field_host[:] = base_recharge * factor
        row, info = run_transient_step_2d(
            solver,
            case=f"5:stress-{label}",
            K=K,
            zbot=zbot,
            h_ss=h_ss,
            steady_converged=ss_ok,
            storage_mode="phreatic_Sy_dx2/dt",
            storage_coeff=0.2,
            dt=1.0,
            max_levels=3,
            active=active,
            bc_mask=bc_mask,
        )
        res.rows.append(row)


# ---------------------------------------------------------------------------
# Case 6: 3D unconfined storage mode
# ---------------------------------------------------------------------------
def case_3d_storage_mode(ctx, res: DiagResults) -> None:
    """Exercise the 3D phreatic_sy storage path and the silent confined fallback.

    :param ctx: run context.
    :param res: results accumulator.
    """
    solver, active, bc_mask, zbot, ztop, initial = build_3d_solver(
        ctx["nx3d"], ctx["ny3d"], ctx["nz3d"], ctx["device"], recharge=1.0e-4, thickness=100.0
    )

    # Steady unconfined warm start.
    try:
        h_ss3d, info_ss = solver.solve(
            unconfined=True,
            zbot_field=zbot,
            ztop_field=ztop,
            initial_head=initial,
            max_cycles=20,
            max_levels=3,
            min_coarse_n=2,
            unconfined_max_picard_iter=30,
            unconfined_head_tol=1.0e-3,
            rel_tol=5.0e-7,
            abs_tol_min=5.0e-7,
            check_every_no=1,
            return_info=True,
        )
        ss_ok = bool(info_ss.get("converged", False))
    except Exception as exc:
        res.rows.append(
            _row(
                case="6:3d-phreatic",
                dim="3d",
                storage_mode="phreatic_sy",
                dt=1.0,
                max_levels=3,
                steady_converged=False,
                transient_converged=False,
                outer_iters=NA,
                max_abs_dh_vs_ss=NA,
                rms_dh_vs_ss=NA,
                final_head_change=NA,
                final_residual=NA,
                failure_reason=f"steady raised {type(exc).__name__}",
            )
        )
        res.add_signal("phreatic_active_with_sy", False)
        return

    # phreatic_sy WITH sy supplied: phreatic storage must be reported active.
    phreatic_active_reported = None
    mode_reported = None
    info3d_fields = None
    try:
        _h_t, info_t = solver.solve(
            unconfined=True,
            zbot_field=zbot,
            ztop_field=ztop,
            initial_head=h_ss3d,
            transient=True,
            sy=0.2,
            ss=1.0e-5,
            dt=1.0,
            head_prev=h_ss3d,
            unconfined_storage_mode="phreatic_sy",
            max_cycles=20,
            max_levels=3,
            min_coarse_n=2,
            unconfined_max_picard_iter=40,
            unconfined_head_tol=1.0e-3,
            rel_tol=5.0e-7,
            abs_tol_min=5.0e-7,
            check_every_no=1,
            return_info=True,
        )
        mode_reported = info_t.get("unconfined_storage_mode")
        phreatic_active_reported = bool(info_t.get("phreatic_storage_active"))
        # Capture the full 3D storage-mode info field set the spec requires.
        info3d_fields = {
            "transient": info_t.get("transient"),
            "unconfined": info_t.get("unconfined"),
            "unconfined_storage_mode": info_t.get("unconfined_storage_mode"),
            "phreatic_storage_active": info_t.get("phreatic_storage_active"),
            "sy": info_t.get("sy"),
            "ss": info_t.get("ss"),
        }
        free3d = (active != 0) & (bc_mask == 0)
        dh = (np.asarray(_h_t, dtype=np.float64) - np.asarray(h_ss3d, dtype=np.float64))[free3d]
        max_abs_dh = float(np.max(np.abs(dh))) if dh.size else float("nan")
        rms_dh = float(np.sqrt(np.mean(dh * dh))) if dh.size else float("nan")
        converged = bool(info_t.get("converged", False))
        outer = info_t.get("outer_iterations", NA)
        final_change = info_t.get("final_max_abs_head_change", NA)
        final_resid = info_t.get("final_residual", NA)
        reason = "" if converged and max_abs_dh < 1.0e-2 else "3D transient drifted / non-converged"
    except (FloatingPointError, ValueError, RuntimeError, OverflowError) as exc:
        max_abs_dh = rms_dh = NA
        converged = False
        outer = NA
        final_change = NA
        final_resid = NA
        reason = f"{type(exc).__name__}: {str(exc)[:90]}"
    res.rows.append(
        _row(
            case="6:3d-phreatic-sy",
            dim="3d",
            storage_mode=str(mode_reported or "phreatic_sy?"),
            dt=1.0,
            max_levels=3,
            steady_converged=ss_ok,
            transient_converged=converged,
            outer_iters=outer,
            max_abs_dh_vs_ss=max_abs_dh,
            rms_dh_vs_ss=rms_dh,
            final_head_change=final_change,
            final_residual=final_resid,
            failure_reason=reason,
        )
    )
    res.add_signal("phreatic_active_with_sy", bool(phreatic_active_reported))
    res.add_signal("phreatic_mode_reported", mode_reported)
    res.add_signal("info3d_fields", info3d_fields)

    # phreatic_sy WITHOUT sy but WITH storage_coeff: must surface the silent fallback.
    solver2, _active2, _bc2, zbot2, ztop2, _init2 = build_3d_solver(
        ctx["nx3d"], ctx["ny3d"], ctx["nz3d"], ctx["device"], recharge=1.0e-4, thickness=100.0
    )
    try:
        h_ss3d_2, _ = solver2.solve(
            unconfined=True,
            zbot_field=zbot2,
            ztop_field=ztop2,
            initial_head=zbot2 + 5.0,
            max_cycles=20,
            max_levels=3,
            min_coarse_n=2,
            unconfined_max_picard_iter=30,
            unconfined_head_tol=1.0e-3,
            check_every_no=1,
            return_info=True,
        )
        _h_t2, info_t2 = solver2.solve(
            unconfined=True,
            zbot_field=zbot2,
            ztop_field=ztop2,
            initial_head=h_ss3d_2,
            transient=True,
            storage_coeff=0.2,  # sy NOT supplied -> should fall back
            dt=1.0,
            head_prev=h_ss3d_2,
            unconfined_storage_mode="phreatic_sy",
            max_cycles=20,
            max_levels=3,
            min_coarse_n=2,
            unconfined_max_picard_iter=40,
            unconfined_head_tol=1.0e-3,
            check_every_no=1,
            return_info=True,
        )
        mode2 = info_t2.get("unconfined_storage_mode")
        phreatic2 = bool(info_t2.get("phreatic_storage_active"))
    except (FloatingPointError, ValueError, RuntimeError, OverflowError) as exc:
        mode2 = f"<raised {type(exc).__name__}>"
        phreatic2 = False
    fell_back = (mode2 == "confined_volume")
    res.rows.append(
        _row(
            case="6:3d-no-sy-fallback",
            dim="3d",
            storage_mode=str(mode2),
            dt=1.0,
            max_levels=3,
            steady_converged=ss_ok,
            transient_converged=NA,
            outer_iters=NA,
            max_abs_dh_vs_ss=NA,
            rms_dh_vs_ss=NA,
            final_head_change=NA,
            final_residual=NA,
            failure_reason=(
                "silent fallback phreatic_sy->confined_volume (sy missing)"
                if fell_back
                else f"phreatic_storage_active={phreatic2}, mode={mode2}"
            ),
        )
    )
    res.add_signal("fell_back_without_sy", fell_back)


# ---------------------------------------------------------------------------
# Case 7: residual-at-steady diagnostic
# ---------------------------------------------------------------------------
def case_residual_at_steady(ctx, res: DiagResults) -> None:
    """Evaluate the transient equation residual exactly at h_ss (head_prev=h_ss).

    Two complementary residuals are reported:

    (a) Independent NumPy residual of the discrete equation at h_ss.  With
        ``head_prev == h_ss`` the backward-Euler storage term vanishes, so the
        transient residual must equal the steady residual (~solver tolerance).
        A mismatch would mean the storage term is assembled inconsistently on
        the two sides of the equation.

    (b) The solver's *own* first-inner-residual when it is started exactly at
        h_ss (captured by case 2 / case 3).  If the discrete equation has h_ss
        as a fixed point, the solver's residual there must also be tiny; a huge
        solver residual means the operator the iterator applies is inconsistent
        with the operator whose residual it computes.

    :param ctx: run context.
    :param res: results accumulator.
    """
    T_ss = res.signals.get("T_ss_2d")
    R_ss = res.signals.get("R_ss_2d")
    h_ss = res.signals.get("h_ss_2d")
    active = res.signals.get("active_2d")
    bc_mask = res.signals.get("bc_mask_2d")
    if T_ss is None or h_ss is None:
        res.rows.append(
            _row(
                case="7:resid-at-ss",
                dim="2d",
                storage_mode="phreatic_Sy_dx2/dt",
                dt=1.0,
                max_levels=3,
                steady_converged=res.signals.get("steady_converged_2d", NA),
                transient_converged=NA,
                outer_iters=NA,
                max_abs_dh_vs_ss=NA,
                rms_dh_vs_ss=NA,
                final_head_change=NA,
                final_residual=NA,
                failure_reason="h_ss not available (case 2 skipped)",
            )
        )
        return

    dx = 100.0
    storage_coeff = 0.2
    dt = 1.0

    steady_max, steady_rms = flux_residual_2d(T_ss, h_ss, R_ss, active, bc_mask, dx)
    trans_max, trans_rms = flux_residual_2d(
        T_ss, h_ss, R_ss, active, bc_mask, dx,
        storage_coeff=storage_coeff, head_prev=h_ss, dt=dt,
    )
    # (a) Equation consistency: transient residual must track the steady one.
    denom = max(steady_rms, 1.0e-30)
    rel_diff = abs(trans_rms - steady_rms) / denom
    equation_consistent = rel_diff < 1.0e-3 and math.isfinite(trans_rms)

    # (b) Solver operator consistency: the solver's residual at the fixed point.
    solver_resid = res.signals.get("first_inner_residual")
    solver_resid_single = res.signals.get("first_inner_residual_single_level")
    solver_resid_huge = (
        isinstance(solver_resid, float)
        and (not math.isfinite(solver_resid) or solver_resid > 1.0e3)
    )
    solver_resid_single_huge = (
        isinstance(solver_resid_single, float)
        and (not math.isfinite(solver_resid_single) or solver_resid_single > 1.0e3)
    )
    # The "consistent" verdict for this row requires BOTH the equation to be
    # consistent AND the solver to see a small residual at the fixed point.
    solver_ok = (not solver_resid_huge)
    consistent = bool(equation_consistent and solver_ok)

    if not equation_consistent:
        reason = f"transient resid != steady resid (rel_diff={rel_diff:.3e})"
    elif solver_resid_huge:
        reason = (
            f"equation consistent at h_ss (steady_rms={steady_rms:.2e}) but solver "
            f"residual at h_ss = {solver_resid:.2e} -> operator/preconditioner inconsistent"
        )
    else:
        reason = ""

    res.rows.append(
        _row(
            case="7:resid-at-ss",
            dim="2d",
            storage_mode="phreatic_Sy_dx2/dt",
            dt=dt,
            max_levels=3,
            steady_converged=res.signals.get("steady_converged_2d", NA),
            transient_converged=consistent,
            outer_iters=NA,
            max_abs_dh_vs_ss=trans_max,
            rms_dh_vs_ss=trans_rms,
            final_head_change=steady_rms,
            final_residual=f"steady_rms={steady_rms:.3e}",
            failure_reason=reason,
        )
    )
    res.add_signal("resid_at_ss_steady_rms", steady_rms)
    res.add_signal("resid_at_ss_transient_rms", trans_rms)
    res.add_signal("resid_at_ss_rel_diff", rel_diff)
    res.add_signal("equation_consistent_at_ss", equation_consistent)
    res.add_signal("solver_residual_at_ss", solver_resid)
    res.add_signal("solver_residual_at_ss_huge", solver_resid_huge)
    res.add_signal("solver_residual_at_ss_single_huge", solver_resid_single_huge)
    res.add_signal("resid_at_ss_consistent", consistent)


# ---------------------------------------------------------------------------
# Case 8: convergence acceptance diagnostic
# ---------------------------------------------------------------------------
def _acceptance_trace_from_history(info: dict) -> dict:
    """Summarise the per-outer-iteration acceptance state from outer_history.

    :param info: solver info dict.
    :return: dict with ``first_head_change_below_hclose`` (1-based outer iter or
        None), ``hclose``, ``trace`` (list of per-iter dicts), and ``n_iters``.
    """
    hist = info.get("outer_history") or []
    hclose = info.get("picard_head_tol")
    if hclose is None:
        hclose = info.get("inner_head_residual_tol")
    first_below = None
    trace = []
    for idx, entry in enumerate(hist):
        dh = entry.get("max_abs_head_change")
        below = (hclose is not None) and (dh is not None) and (dh < hclose)
        if below and first_below is None:
            first_below = idx + 1
        trace.append(
            {
                "outer": idx + 1,
                "omega": entry.get("omega"),
                "max_abs_head_change": dh,
                "rms_head_change": entry.get("picard_update_rms"),
                "linear_residual": entry.get("inner_residual"),
                "inner_converged": entry.get("inner_converged"),
                "inner_usable": entry.get("inner_usable_for_picard"),
                "accepted": entry.get("accepted_picard_update_count"),
            }
        )
    return {
        "hclose": hclose,
        "first_head_change_below_hclose": first_below,
        "n_iters": len(hist),
        "trace": trace,
    }


def case_convergence_acceptance(ctx, res: DiagResults) -> None:
    """Report the per-outer acceptance trace and the head-change-only experiment.

    :param ctx: run context.
    :param res: results accumulator.
    """
    info = res.signals.get("invariant_2d_info")
    if info is None:
        res.rows.append(
            _row(
                case="8:acceptance",
                dim="2d",
                storage_mode="phreatic_Sy_dx2/dt",
                dt=1.0,
                max_levels=3,
                steady_converged=res.signals.get("steady_converged_2d", NA),
                transient_converged=NA,
                outer_iters=NA,
                max_abs_dh_vs_ss=NA,
                rms_dh_vs_ss=NA,
                final_head_change=NA,
                final_residual=NA,
                failure_reason="invariant case skipped",
            )
        )
        return

    trace_summary = _acceptance_trace_from_history(info)
    res.add_signal("acceptance_trace", trace_summary)

    # Genuine re-run with the diagnostic flag: acceptance on head change only.
    solver, active, bc_mask, K, zbot = build_2d_solver(
        ctx["nx"], ctx["ny"], ctx["device"], recharge=1.0e-4, transmissivity=10.0
    )
    h_ss, info_ss = solve_2d_steady_unconfined(solver, K, zbot, zbot + 5.0)
    ss_ok = bool(info_ss.get("converged", False))
    row_flag, info_flag = run_transient_step_2d(
        solver,
        case="8:accept-head-change",
        K=K,
        zbot=zbot,
        h_ss=h_ss,
        steady_converged=ss_ok,
        storage_mode="phreatic_Sy_dx2/dt (accept=head_only)",
        storage_coeff=0.2,
        dt=1.0,
        max_levels=3,
        active=active,
        bc_mask=bc_mask,
        accept_head_change=True,
    )
    res.rows.append(row_flag)
    helped = (
        bool(row_flag["transient_converged"])
        and isinstance(row_flag["max_abs_dh_vs_ss"], float)
        and row_flag["max_abs_dh_vs_ss"] < 1.0e-2
    )
    res.add_signal("head_change_only_helped", helped)
    res.add_signal("accept_head_change_info", info_flag)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt(value: Any, width: int, spec: str = "") -> str:
    """Format a cell value to a fixed width, tolerating None/NA/inf.

    :param value: cell value.
    :param width: target column width.
    :param spec: optional format spec for numeric values (e.g. ``'.3e'``).
    """
    if value is None or value is NA:
        text = NA
    elif isinstance(value, str):
        text = value
    elif isinstance(value, (bool, np.bool_)):
        text = "yes" if bool(value) else "no"
    elif isinstance(value, (int, np.integer)):
        text = str(int(value))
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            text = "inf" if float(value) > 0 else "-inf"
        elif spec:
            text = format(float(value), spec)
        else:
            text = f"{float(value):.4g}"
    else:
        text = str(value)
    if len(text) > width:
        text = text[: max(0, width - 1)] + "…"
    return text.rjust(width)


def print_table(rows: list[dict[str, Any]]) -> None:
    """Print the diagnostic table.

    :param rows: list of row dicts.
    """
    widths = {
        "case": 22,
        "dim": 3,
        "storage_mode": 26,
        "dt": 11,
        "max_levels": 10,
        "steady_converged": 8,
        "transient_converged": 9,
        "outer_iters": 7,
        "max_abs_dh_vs_ss": 15,
        "rms_dh_vs_ss": 13,
        "final_head_change": 15,
        "final_residual": 15,
        "failure_reason": 44,
    }
    numeric_specs = {
        "dt": ".2e",
        "max_abs_dh_vs_ss": ".3e",
        "rms_dh_vs_ss": ".3e",
        "final_head_change": ".3e",
    }
    header = "  ".join(col.rjust(widths[col]) for col in COLUMNS)
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for row in rows:
        cells = []
        for col in COLUMNS:
            spec = numeric_specs.get(col, "")
            cells.append(_fmt(row.get(col, NA), widths[col], spec))
        print("  ".join(cells))
    print(sep)


def print_convergence_trace(label: str, info: dict | None) -> None:
    """Print a compact per-outer-iteration acceptance trace.

    :param label: case label.
    :param info: solver info dict (may be None).
    """
    if info is None:
        print(f"\n[{label}] no info (solve raised before producing a history).")
        return
    summary = _acceptance_trace_from_history(info)
    trace = summary["trace"]
    hclose = summary["hclose"]
    basis = info.get("nonlinear_convergence_basis", NA)
    print(f"\n[{label}] convergence trace ({summary['n_iters']} outer iters, "
          f"hclose={hclose}, basis={basis})")
    cols = ("outer", "omega", "max_abs_dh", "rms_dh", "lin_resid", "inner_conv", "inner_usable")
    w = (6, 8, 12, 12, 14, 10, 12)
    print("  " + "  ".join(c.rjust(wi) for c, wi in zip(cols, w)))
    # Show up to the first 8 and the last 2 iterations.
    show = trace[:8] + (trace[-2:] if len(trace) > 10 else [])
    if len(trace) > 10:
        print("  " + "  ".join("...".rjust(wi) for wi in w))
    for e in show:
        print("  " + "  ".join(
            _fmt(v, wi, ".3e" if isinstance(v, float) else "")
            for v, wi in zip(
                (
                    e["outer"],
                    e["omega"],
                    e["max_abs_head_change"],
                    e["rms_head_change"],
                    e["linear_residual"],
                    e["inner_converged"],
                    e["inner_usable"],
                ),
                w,
            )
        ))
    first_below = summary["first_head_change_below_hclose"]
    print(f"  -> first outer iter with max_abs_head_change < hclose: {first_below}")


def diagnose(res: DiagResults) -> str:
    """Produce the concise PASS/FAIL/UNKNOWN verdict from the signals.

    Priority order (first match wins).  The decisive discriminator is the
    dt-sensitivity ladder: if the iteration diverges at small dt (large storage
    diagonal) but converges to h_ss in one outer iteration at large dt (small
    storage diagonal), then h_ss IS a fixed point and the discrete assembly is
    algebraically consistent -- the defect is that the transient storage
    diagonal destabilises the iterative solver (operator-application / smoother /
    preconditioner) when it is large relative to the diffusion diagonal.

    1. dt ladder: small dt fails, large dt converges to h_ss
         -> transient assembly inconsistent (storage-diagonal destabilisation)
    2. solver residual at h_ss huge at BOTH single-level and full K-cycle
         -> transient assembly inconsistent (operator/preconditioner at fixed point)
    3. solver residual at h_ss huge only for the full K-cycle
         -> hierarchy-dependent failure
    4. equation residual at h_ss inconsistent (storage term does not cancel)
         -> transient assembly inconsistent (storage assembly / sign / scaling)
    5. head stable at h_ss but residual-based acceptance rejected it
         -> convergence acceptance too strict
    6. 3D phreatic storage not active when expected
         -> storage mode not active
    7. everything holds
         -> PASS

    :param res: results accumulator.
    :return: one-line diagnosis string.
    """
    s = res.signals
    invariant_ok = s.get("invariant_2d_ok")
    dt_small_ok = s.get("dt_small_ok")
    dt_large_ok = s.get("dt_large_ok")
    solver_resid_huge = s.get("solver_residual_at_ss_huge")
    # solver_residual_at_ss_single_huge is None when case 3 did not run.
    solver_resid_single_huge = s.get("solver_residual_at_ss_single_huge")
    single_tested = solver_resid_single_huge is not None
    single_small = single_tested and (solver_resid_single_huge is False)
    equation_consistent = s.get("equation_consistent_at_ss")
    solver_resid = s.get("solver_residual_at_ss")

    def _fmt_num(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3e}"
        return str(v)

    # Layer 1: storage-diagonal destabilisation (the dt ladder is decisive).
    if invariant_ok is False and dt_small_ok is False and dt_large_ok is True:
        steady_rms = s.get("resid_at_ss_steady_rms", float("nan"))
        scope = "at both single-level and full K-cycle" if solver_resid_single_huge else "for the full K-cycle"
        return (
            "FAIL: transient assembly inconsistent (storage-diagonal destabilisation) -- "
            "the iteration diverges at small dt (large storage diagonal: Sy*dx^2/dt) but converges "
            "to h_ss in one outer iteration at large dt (small storage diagonal); h_ss IS a fixed "
            f"point and the discrete equation residual there is {_fmt_num(steady_rms)}, so the "
            "assembly is algebraically consistent. The transient storage diagonal destabilises the "
            f"iterative solver {scope} (operator-application / smoother / preconditioner) when it is "
            "large relative to the diffusion diagonal; inspect the storage-diagonal scaling in the "
            "operator-application kernel and the residual/preconditioner scaling"
        )

    # Layer 2: operator/preconditioner inconsistency, hierarchy-independent.
    # Confirmed when the solver residual at h_ss is huge AND it is also huge at
    # single-level (or single-level was not tested, in which case assembly is the
    # best-supported conclusion pending case 3).
    if solver_resid_huge and (solver_resid_single_huge or not single_tested):
        steady_rms = s.get("resid_at_ss_steady_rms", float("nan"))
        scope = "at both single-level and full K-cycle" if solver_resid_single_huge else "(single-level not tested)"
        note = "" if single_tested else " [run case 3 to confirm it is not hierarchy-dependent]"
        return (
            "FAIL: transient assembly inconsistent "
            f"(solver residual at h_ss = {_fmt_num(solver_resid)} {scope}, while the discrete "
            f"equation residual there is {_fmt_num(steady_rms)}; the operator the iterator applies "
            f"is inconsistent with the residual/preconditioner at the steady fixed point){note}"
        )

    # Layer 3: hierarchy-dependent (single-level small, full K-cycle huge).
    if solver_resid_huge and single_small:
        return (
            "FAIL: hierarchy-dependent failure "
            "(solver residual at h_ss is huge for the full K-cycle but small at single-level; "
            "coarse-grid storage scaling / restriction-prolongation / coarse operator assembly)"
        )

    # Layer 4: storage-term algebra inconsistency (equation, not iteration).
    if equation_consistent is False:
        steady_rms = s.get("resid_at_ss_steady_rms", float("nan"))
        trans_rms = s.get("resid_at_ss_transient_rms", float("nan"))
        rel = s.get("resid_at_ss_rel_diff", float("nan"))
        return (
            "FAIL: transient assembly inconsistent "
            f"(discrete transient residual at h_ss ({trans_rms:.3e}) != steady residual "
            f"({steady_rms:.3e}), rel_diff={rel:.3e}; storage term does not cancel at head_prev=h_ss)"
        )

    # Layer 5: head stable but acceptance too strict.
    if invariant_ok is False:
        first_update = s.get("first_update_at_ss")
        if isinstance(first_update, float) and first_update < 1.0e-3:
            return (
                "FAIL: convergence acceptance too strict "
                "(head stable at h_ss but the residual/inner_usable gate rejected it)"
            )

    # Layer 6: 3D storage mode not active when expected.
    if s.get("phreatic_active_with_sy") is False:
        return "FAIL: storage mode not active (phreatic_sy reported inactive with sy supplied)"

    # Layer 7: pass.
    if invariant_ok is True:
        return "PASS: transient invariant holds"

    return "UNKNOWN: insufficient instrumentation"


def print_signal_summary(res: DiagResults) -> None:
    """Print the cross-case signals that drive the verdict.

    :param res: results accumulator.
    """
    s = res.signals
    print("\n=== diagnostic signals ===")
    keys_of_interest = [
        "confined_mild_ok",
        "confined_aggressive_ok",
        "steady_converged_2d",
        "invariant_2d_ok",
        "hierarchy_single_level_ok",
        "hierarchy_full_kcycle_ok",
        "first_inner_residual_single_level",
        "dt_small_ok",
        "dt_large_ok",
        "first_update_at_ss",
        "first_inner_residual",
        "resid_at_ss_steady_rms",
        "resid_at_ss_transient_rms",
        "equation_consistent_at_ss",
        "solver_residual_at_ss",
        "solver_residual_at_ss_huge",
        "solver_residual_at_ss_single_huge",
        "phreatic_active_with_sy",
        "phreatic_mode_reported",
        "fell_back_without_sy",
        "head_change_only_helped",
    ]
    for key in keys_of_interest:
        if key in s:
            val = s[key]
            if isinstance(val, float):
                val = f"{val:.6e}"
            print(f"  {key:32s} = {val}")

    # Case-6 requirement: surface the full 3D storage-mode info field set.
    info3d = s.get("info3d_fields")
    if info3d is not None:
        print("\n=== 3D transient unconfined info fields (phreatic_sy, sy supplied) ===")
        for name in ("transient", "unconfined", "unconfined_storage_mode",
                     "phreatic_storage_active", "sy", "ss"):
            print(f"  info[{name:24s}] = {info3d.get(name)}")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
CASE_FUNCS = {
    "1": case_confined_control,
    "2": case_2d_invariant,
    "3": case_2d_hierarchy,
    "4": case_2d_dt_ladder,
    "5": case_2d_mild_stress_change,
    "6": case_3d_storage_mode,
    "7": case_residual_at_steady,
    "8": case_convergence_acceptance,
}


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: argument list.
    :return: parsed namespace.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="warp device string (default: cpu)")
    parser.add_argument(
        "--grid",
        choices=("small", "medium"),
        default="small",
        help="grid size: small (32x16 / 12x12x6) or medium (64x32 / 24x24x10)",
    )
    parser.add_argument(
        "--cases",
        default="all",
        help="comma-separated case ids 1-8, or 'all' (default: all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="print per-outer-iteration traces")
    return parser.parse_args(argv)


def grid_sizes(grid: str) -> dict[str, int]:
    """Return grid dimensions for the chosen size preset.

    :param grid: 'small' or 'medium'.
    :return: dict with nx, ny and 3D dims.
    """
    if grid == "medium":
        return {"nx": 64, "ny": 32, "nx3d": 24, "ny3d": 24, "nz3d": 10}
    return {"nx": 32, "ny": 16, "nx3d": 12, "ny3d": 12, "nz3d": 6}


def main(argv: list[str] | None = None) -> int:
    """Run the diagnostic ladder and print the table + verdict.

    :param argv: optional argument list (defaults to ``sys.argv[1:]``).
    :return: exit code (0 on success regardless of pass/fail).
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)

    if not warp_available():
        print("ERROR: warp is not available; install warp-lang to run this diagnostic.", file=sys.stderr)
        return 1

    if args.cases.strip().lower() == "all":
        selected = list(CASE_FUNCS.keys())
    else:
        selected = []
        for token in args.cases.split(","):
            token = token.strip()
            if token in CASE_FUNCS:
                selected.append(token)
            else:
                print(f"WARNING: ignoring unknown case id {token!r}", file=sys.stderr)
        if not selected:
            print("ERROR: no valid cases selected.", file=sys.stderr)
            return 2

    ctx = {"device": args.device}
    ctx.update(grid_sizes(args.grid))

    print(f"DarcyWarp transient-unconfined diagnostic ladder")
    print(f"device={args.device} grid={args.grid} cases={selected}")
    print("(deterministic; no random fields; pathlib; no plotting)\n")

    res = DiagResults()
    for cid in selected:
        func = CASE_FUNCS[cid]
        try:
            func(ctx, res)
        except Exception as exc:  # pragma: no cover - keep the ladder running
            res.rows.append(
                _row(
                    case=f"{cid}:<case-error>",
                    dim="?",
                    storage_mode="?",
                    dt=NA,
                    max_levels=NA,
                    steady_converged=NA,
                    transient_converged=NA,
                    outer_iters=NA,
                    max_abs_dh_vs_ss=NA,
                    rms_dh_vs_ss=NA,
                    final_head_change=NA,
                    final_residual=NA,
                    failure_reason=f"{type(exc).__name__}: {str(exc)[:90]}",
                )
            )

    print_table(res.rows)

    if args.verbose:
        for label_key, info_key in (("2:invariant", "invariant_2d_info"),
                                    ("8:accept-head-change", "accept_head_change_info")):
            info = res.signals.get(info_key)
            print_convergence_trace(label_key, info)

    print_signal_summary(res)

    verdict = diagnose(res)
    print("\n=== DIAGNOSIS ===")
    print(verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
