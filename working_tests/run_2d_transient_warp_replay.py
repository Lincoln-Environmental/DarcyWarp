#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""
2D transient Warp-vs-MF6 replay/comparison harness.

This module steps the 2D Warp unconfined transient solver through every stress
period of a case and compares the resulting per-period and final heads against
a MODFLOW 6 truth artifact produced by ``working_tests/run_2d_transient_vs_mf6.py``.

The MF6 truth artifact is a compressed ``.npz.lzma`` file storing per-period
heads plus every input needed for a deterministic Warp replay: initial heads,
active/boundary masks, top/bottom elevations, hydraulic conductivity, the
per-period recharge time series, and the storage parameters (Sy, Ss).

Design notes
------------
* Time units are carried through unchanged. MF6 writes the artifact with
  ``time_units="DAYS"`` (K in m/day, recharge in m/day, ``dt = dt_days``). The
  groundwater flow equation is invariant under a rescaling of the time unit, so
  those day-valued fields are passed to Warp verbatim - the heads come out in
  metres either way. No unit conversion is applied.
* Warp's unconfined transient term is ``storage_coeff * dx**2 / dt`` per active
  cell, i.e. phreatic (specific-yield) storage over the cell area. MF6 also
  carries the much smaller confined ``Ss * saturated_thickness`` contribution;
  that term (typically <1% of Sy for these cases) is not modelled by Warp and is
  documented as a known approximation. ``storage_coeff`` therefore defaults to
  ``Sy``.
* The MF6-free core (``run_warp_transient_replay``) takes plain NumPy spatial
  fields, so it can be unit-tested without Flopy or an MF6 binary. The artifact
  loader is a thin wrapper that feeds those fields plus the MF6 heads for
  comparison.
"""

from __future__ import annotations

import io
import json
import lzma
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DARCY_FLOAT", "float64")

from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver  # noqa: E402


DEFAULT_ARTIFACT_NAME = "mf6_transient_heads.npz.lzma"
DEFAULT_MIN_SAT = 0.1


def _warp_device(preferred: str = "auto") -> str:
    import warp as wp

    if preferred != "auto":
        return preferred
    try:
        return "cuda:0" if wp.is_cuda_available() else "cpu"
    except AttributeError:
        return "cuda:0"


def _load_compressed_npz(path: str | Path) -> dict:
    """Decompress a ``.npz.lzma`` fixture and return its arrays as a dict."""
    path = Path(path)
    buf = io.BytesIO(lzma.decompress(path.read_bytes()))
    with np.load(buf, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_transient_artifact(path: str | Path) -> dict:
    """
    Load a 2D transient MF6 truth artifact and return its arrays.

    Returns the raw artifact arrays (per-period MF6 heads, inputs, storage
    parameters, and provenance). See module docstring for the stored fields.
    """
    arrays = _load_compressed_npz(path)
    required = (
        "heads_per_period", "heads_final", "initial_head", "active", "bc_mask",
        "bc_values", "top", "bottom", "k_field", "recharge_rates", "sy", "ss",
        "nx", "ny", "dx", "dt_days",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"transient artifact {path} missing keys: {missing}")
    return arrays


def build_synthetic_spatial_fields(
    nx: int = 16,
    ny: int = 12,
    dx: float = 100.0,
    hydraulic_conductivity: float = 100.0,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
) -> dict:
    """
    Build a small, MF6-free 2D unconfined spatial field set for tests/replays.

    This mirrors the spatial construction used by the MF6 truth generator but
    uses simple rectangular top/bottom elevations so it does not depend on the
    model_builder DEM helpers. The returned dict is the same shape consumed by
    :func:`run_warp_transient_replay`.
    """
    from DARCY_WARP_PACKAGE.model_builder import (
        _build_dirichlet_boundary_mask,
        _build_domain,
    )

    active = _build_domain(nx=int(nx), ny=int(ny)).astype(np.int32)
    bc_bool = _build_dirichlet_boundary_mask(active)
    bc_mask = bc_bool.astype(np.int32)

    top = np.full((int(ny), int(nx)), 110.0, dtype=np.float64)
    bottom = np.full((int(ny), int(nx)), 10.0, dtype=np.float64)
    bc_values = np.full((int(ny), int(nx)), 100.0, dtype=np.float64)

    k_field = np.full((int(ny), int(nx)), float(hydraulic_conductivity), dtype=np.float64)
    k_field[active == 0] = 0.0

    initial_head = np.minimum(bottom + float(initial_saturated_thickness), top)
    initial_head[bc_mask != 0] = bc_values[bc_mask != 0]
    initial_head[active == 0] = 0.0

    return {
        "nx": int(nx),
        "ny": int(ny),
        "dx": float(dx),
        "active": active,
        "bc_mask": bc_mask,
        "bc_values": bc_values,
        "top": top,
        "bottom": bottom,
        "k": k_field,
        "initial_head": initial_head.astype(np.float64, copy=False),
        "workspace": Path(workspace) if workspace is not None else None,
    }


def spatial_fields_from_artifact(artifact: dict) -> dict:
    """Extract the plain-NumPy spatial fields consumed by the replay."""
    return {
        "nx": int(artifact["nx"]),
        "ny": int(artifact["ny"]),
        "dx": float(artifact["dx"]),
        "active": np.asarray(artifact["active"], dtype=np.int32),
        "bc_mask": np.asarray(artifact["bc_mask"], dtype=np.int32),
        "bc_values": np.asarray(artifact["bc_values"], dtype=np.float64),
        "top": np.asarray(artifact["top"], dtype=np.float64),
        "bottom": np.asarray(artifact["bottom"], dtype=np.float64),
        "k": np.asarray(artifact["k_field"], dtype=np.float64),
        "initial_head": np.asarray(artifact["initial_head"], dtype=np.float64),
        "workspace": None,
    }


def _initial_transmissivity(
    k: np.ndarray,
    initial_head: np.ndarray,
    bottom: np.ndarray,
    active: np.ndarray,
    min_sat: float = DEFAULT_MIN_SAT,
) -> np.ndarray:
    t = k * np.maximum(initial_head - bottom, float(min_sat))
    t[active == 0] = 0.0
    return t.astype(np.float64, copy=False)


def default_solve_controls() -> dict:
    """Default Warp solve controls for the transient replay (kcycle)."""
    return {
        "max_cycles": 40,
        "max_levels": 5,
        "min_coarse_cells": 1,
        "check_every_no": 1,
        "max_outer_iterations": 40,
        "hclose": 1.0e-4,
        "rel_tol": 5.0e-7,
        "abs_tol_min": 5.0e-7,
        "dh_rms_tol": 1.0e-4,
        "smoother": "chebyshev",
        "omega": 0.7,
        "chebyshev_enabled": True,
        "cheby_lambda_min": 0.1,
        "cheby_lambda_max": 2.0,
        "chebyshev_reset_factor": 1.2,
        "inner_forcing_eta": 0.10,
        "inner_head_residual_tol_min": 1.0e-4,
        "inner_head_residual_tol_max": 1.0e-2,
        "transmissivity_relaxation_enabled": False,
    }


def run_warp_transient_replay(
    spatial: dict,
    recharge_rates: np.ndarray,
    sy: float,
    dt: float,
    n_periods: int | None = None,
    device: str = "auto",
    diag_preconditioner_backend: str = "auto",
    min_sat: float = DEFAULT_MIN_SAT,
    solve_controls: dict | None = None,
) -> dict:
    """
    Step the 2D Warp unconfined transient solver through every stress period.

    This is the MF6-free core: it takes plain NumPy spatial fields plus the
    per-period recharge rates, specific yield, and time step, and returns the
    per-period Warp heads, the final heads, the last period's convergence info,
    and timing. No Flopy/MF6 dependency.

    Parameters
    ----------
    spatial:
        Dict from :func:`build_synthetic_spatial_fields` /
        :func:`spatial_fields_from_artifact`.
    recharge_rates:
        ``(n_periods,)`` recharge rate per period (uniform in space), in the same
        time unit as ``dt``.
    sy:
        Specific yield used as the phreatic storage coefficient.
    dt:
        Time-step length (same unit as ``recharge_rates`` / K).
    n_periods:
        Number of periods to step. Defaults to ``len(recharge_rates)``.
    """
    controls = dict(default_solve_controls())
    if solve_controls:
        controls.update(solve_controls)

    nx = int(spatial["nx"])
    ny = int(spatial["ny"])
    dx = float(spatial["dx"])
    active = np.asarray(spatial["active"], dtype=np.int32)
    bc_mask = np.asarray(spatial["bc_mask"], dtype=np.int32)
    bc_values = np.asarray(spatial["bc_values"], dtype=np.float64)
    top = np.asarray(spatial["top"], dtype=np.float64)
    bottom = np.asarray(spatial["bottom"], dtype=np.float64)
    k = np.asarray(spatial["k"], dtype=np.float64)
    initial_head = np.asarray(spatial["initial_head"], dtype=np.float64)

    rates = np.asarray(recharge_rates, dtype=np.float64).reshape(-1)
    if n_periods is None:
        n_periods = int(rates.shape[0])
    n_periods = int(n_periods)
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1.")
    if n_periods > rates.shape[0]:
        raise ValueError(
            f"n_periods={n_periods} exceeds available recharge rates ({rates.shape[0]})."
        )

    storage_coeff = float(sy)
    device = _warp_device(device)

    initial_transmissivity = _initial_transmissivity(k, initial_head, bottom, active, min_sat)
    # Uniform-in-space recharge field, rebuilt per period.
    recharge_field = np.zeros((ny, nx), dtype=np.float64)

    heads_per_period = np.zeros((n_periods, ny, nx), dtype=np.float64)
    period_infos: list[dict] = []
    period_times = np.zeros(n_periods, dtype=np.float64)

    t0 = time.perf_counter()
    head_prev = initial_head.copy()
    last_info: dict = {}

    with WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=dx,
        device=device,
        solver_type="kcycle",
        diag_preconditioner_backend=diag_preconditioner_backend,
    ) as solver:
        solver.build_from_fields(
            T_field=initial_transmissivity,
            R_field=recharge_field,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
        )

        for per in range(n_periods):
            rate = float(rates[per])
            recharge_field[...] = rate
            recharge_field[active == 0] = 0.0
            solver.R_field_host[...] = recharge_field

            t_per = time.perf_counter()
            head, info = solver.solve(
                formulation="unconfined",
                K_field=k,
                zbot_field=bottom,
                ztop_field=top,
                initial_head=head_prev,
                transient=True,
                storage_coeff=storage_coeff,
                dt=float(dt),
                head_prev=head_prev,
                return_info=True,
                **controls,
            )
            period_times[per] = time.perf_counter() - t_per

            head = np.asarray(head, dtype=np.float64)
            heads_per_period[per] = head
            head_prev = head
            last_info = dict(info) if isinstance(info, dict) else {}
            period_infos.append(last_info)

    total_time = time.perf_counter() - t0

    return {
        "heads_per_period": heads_per_period,
        "heads_final": heads_per_period[-1],
        "period_infos": period_infos,
        "last_info": last_info,
        "period_times": period_times,
        "total_time": total_time,
        "n_periods": n_periods,
        "device": device,
        "storage_coeff": storage_coeff,
        "dt": float(dt),
        "solve_controls": controls,
    }


def _head_metrics(warp_heads: np.ndarray, mf6_heads: np.ndarray, active: np.ndarray) -> dict:
    warp_heads = np.asarray(warp_heads, dtype=np.float64)
    mf6_heads = np.asarray(mf6_heads, dtype=np.float64)
    mask = (np.asarray(active) != 0) & np.isfinite(warp_heads) & np.isfinite(mf6_heads)
    diff = warp_heads - mf6_heads
    diff_masked = diff[mask]
    abs_diff = np.abs(diff_masked)
    return {
        "rmse": float(np.sqrt(np.mean(diff_masked * diff_masked))) if diff_masked.size else None,
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else None,
        "mean_bias_warp_minus_mf6": float(np.mean(diff_masked)) if diff_masked.size else None,
        "percent_within_0_01m": float(np.mean(abs_diff <= 0.01) * 100.0) if abs_diff.size else None,
        "percent_within_0_1m": float(np.mean(abs_diff <= 0.1) * 100.0) if abs_diff.size else None,
        "n_active": int(mask.sum()),
    }


def compare_transient(
    warp_result: dict,
    mf6_heads_per_period: np.ndarray,
    mf6_heads_final: np.ndarray,
    active: np.ndarray,
) -> dict:
    """
    Compare Warp per-period/final heads against MF6 truth.

    Returns per-period metrics, final metrics, and the worst (max-abs-diff)
    period index.
    """
    warp_hpp = np.asarray(warp_result["heads_per_period"], dtype=np.float64)
    mf6_hpp = np.asarray(mf6_heads_per_period, dtype=np.float64)
    if warp_hpp.shape != mf6_hpp.shape:
        raise ValueError(
            f"per-period head shape mismatch: warp {warp_hpp.shape}, mf6 {mf6_hpp.shape}"
        )

    per_period = [
        _head_metrics(warp_hpp[i], mf6_hpp[i], active) for i in range(warp_hpp.shape[0])
    ]
    final = _head_metrics(warp_result["heads_final"], mf6_heads_final, active)

    max_abs_values = [m["max_abs_diff"] for m in per_period if m["max_abs_diff"] is not None]
    worst_period = int(np.argmax(max_abs_values)) if max_abs_values else None

    return {
        "per_period": per_period,
        "final": final,
        "worst_period": worst_period,
    }


def save_summary(path: str | Path, summary: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(summary, f, indent=4, default=str)
    return path


def run_replay_from_artifact(
    artifact_path: str | Path,
    workspace: str | Path | None = None,
    device: str = "auto",
    diag_preconditioner_backend: str = "auto",
    solve_controls: dict | None = None,
) -> dict:
    """
    Load the MF6 truth artifact, replay Warp through every period, compare, save.

    Returns the full summary dict and writes ``transient_replay_summary.json``
    plus ``warp_transient_heads.npz`` under the workspace.
    """
    artifact_path = Path(artifact_path)
    if workspace is None:
        workspace = artifact_path.parent
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    artifact = load_transient_artifact(artifact_path)
    spatial = spatial_fields_from_artifact(artifact)
    spatial["workspace"] = workspace

    sy = float(artifact["sy"])
    ss = float(artifact["ss"])
    dt = float(artifact["dt_days"])
    recharge_rates = np.asarray(artifact["recharge_rates"], dtype=np.float64)
    n_periods = int(recharge_rates.shape[0])

    print(f"Transient replay: {spatial['nx']}x{spatial['ny']}, {n_periods} periods, dt={dt}")
    print(f"  Sy={sy}, Ss={ss} (Warp uses Sy as phreatic storage; Ss*sat term not modelled)")
    print(f"  artifact: {artifact_path}")

    warp_result = run_warp_transient_replay(
        spatial=spatial,
        recharge_rates=recharge_rates,
        sy=sy,
        dt=dt,
        n_periods=n_periods,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
        solve_controls=solve_controls,
    )

    comparison = compare_transient(
        warp_result,
        np.asarray(artifact["heads_per_period"], dtype=np.float64),
        np.asarray(artifact["heads_final"], dtype=np.float64),
        spatial["active"],
    )

    warp_npz = workspace.joinpath("warp_transient_heads.npz")
    np.savez_compressed(
        warp_npz,
        heads_per_period=warp_result["heads_per_period"],
        heads_final=warp_result["heads_final"],
        total_time=np.asarray(warp_result["total_time"], dtype=np.float64),
        period_times=warp_result["period_times"],
        last_info=np.asarray(json.dumps(warp_result["last_info"], default=str)),
        storage_coeff=np.asarray(warp_result["storage_coeff"], dtype=np.float64),
        dt=np.asarray(warp_result["dt"], dtype=np.float64),
        device=np.asarray(warp_result["device"]),
    )

    provenance = artifact.get("provenance")
    if provenance is not None:
        provenance = str(np.asarray(provenance).reshape(()))

    summary = {
        "artifact_path": str(artifact_path),
        "grid": {"nx": spatial["nx"], "ny": spatial["ny"], "dx": spatial["dx"]},
        "n_periods": n_periods,
        "dt": dt,
        "storage": {"sy": sy, "ss": ss, "warp_storage_coeff": warp_result["storage_coeff"]},
        "timing": {
            "warp_total_time": float(warp_result["total_time"]),
            "mf6_total_time": _scalar(artifact, "total_time"),
            "mf6_engine_time": _scalar(artifact, "engine_time"),
            "warp_period_time_mean": float(np.mean(warp_result["period_times"])),
            "warp_period_time_max": float(np.max(warp_result["period_times"])),
        },
        "convergence": _summarize_last_info(warp_result["last_info"]),
        "solve_settings": warp_result["solve_controls"],
        "device": warp_result["device"],
        "diag_preconditioner_backend": diag_preconditioner_backend,
        "comparison": comparison,
        "mf6_provenance": provenance,
    }

    summary_path = workspace.joinpath("transient_replay_summary.json")
    save_summary(summary_path, summary)

    print(f"Warp transient heads saved to {warp_npz}")
    print(f"Replay summary saved to {summary_path}")
    final_max = comparison["final"]["max_abs_diff"]
    final_rmse = comparison["final"]["rmse"]
    print(
        f"Final vs MF6: max_abs_diff={final_max:.6g} m, rmse={final_rmse:.6g} m "
        f"(worst period {comparison['worst_period']})"
    )
    return summary


def _scalar(artifact: dict, name: str) -> float | None:
    if name not in artifact:
        return None
    try:
        return float(np.asarray(artifact[name]).reshape(()))
    except (TypeError, ValueError):
        return None


def _summarize_last_info(info: dict) -> dict:
    if not isinstance(info, dict):
        return {}
    keys = (
        "converged", "outer_iterations", "formulation", "transient",
        "final_max_abs_head_change", "final_residual", "chebyshev_rejections",
        "chebyshev_resets", "accepted_picard_update_count",
        "strict_inner_nonconvergence_count", "unusable_inner_solve_count",
        "practical_inner_acceptance_count", "effectively_dry_cell_count",
    )
    out = {}
    for key in keys:
        if key in info:
            value = info[key]
            if isinstance(value, (int, np.integer)):
                out[key] = int(value)
            elif isinstance(value, (float, np.floating)):
                out[key] = float(value)
            else:
                out[key] = value
    return out


def default_artifact_path() -> Path:
    return data_store.joinpath(
        "working_tests", "mf6_transient_2d_unconfined", DEFAULT_ARTIFACT_NAME
    )


def main(
    artifact_path: str | Path | None = None,
    workspace: str | Path | None = None,
    device: str = "auto",
    diag_preconditioner_backend: str = "auto",
) -> dict:
    """
    Run the default transient replay against the MF6 truth artifact.

    If the artifact is missing, prints instructions to generate it rather than
    failing, so importing this module never requires MF6 to be installed.
    """
    artifact_path = Path(artifact_path) if artifact_path is not None else default_artifact_path()
    if not artifact_path.exists():
        print(f"MF6 transient artifact not found at {artifact_path}.")
        print("Generate it first with:  python working_tests/run_2d_transient_vs_mf6.py")
        return {"artifact_path": str(artifact_path), "ran": False}
    return run_replay_from_artifact(
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
    )


if __name__ == "__main__":
    # Configuration parameters
    artifact_path = None                 # defaults to the standard MF6 truth artifact
    workspace = None                     # defaults to the artifact's parent directory
    device = "auto"
    diag_preconditioner_backend = "device"

    main(
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
    )
