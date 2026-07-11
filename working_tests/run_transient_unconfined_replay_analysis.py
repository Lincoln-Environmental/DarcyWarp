#!/usr/bin/env python
"""
Focused 2D transient-unconfined replay mismatch analysis.

This script audits the production secant-Sy Warp-vs-MF6 replay. By default it
reads the existing MF6 truth artifact and existing Warp replay output, writes an
analysis JSON next to the replay output, and prints compact numeric tables. With
``--run-replays`` it runs only the production secant-Sy replay and the
freeze-after-outer secant-Sy variants.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from working_tests.transient_artifacts import (  # noqa: E402
    FORMULATION_UNCONFINED,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    WARM_START_UNCONFINED_STEADY_MF6,
    default_artifact_path,
    load_transient_artifact,
)
from working_tests.transient_replay_settings import (  # noqa: E402
    STORAGE_ACTIVE_SET_FREEZE_WHEN_STABLE,
    STORAGE_ACTIVE_SET_NONE,
    STORAGE_REFERENCE_CURRENT_PICARD,
    STORAGE_TOP_THRESHOLD_GE,
    default_solve_controls,
    production_secant_sy_settings,
    secant_sy_freeze_settings,
)
from working_tests.transient_replay_support import run_replay_from_artifact  # noqa: E402


NOT_AVAILABLE = "not_available"


def finite_scalar(value: Any) -> float | None:
    """:param value: scalar-like value. :return: finite float or ``None``."""
    try:
        value_f = float(np.asarray(value).reshape(()))
    except (TypeError, ValueError):
        return None
    return value_f if np.isfinite(value_f) else None


def scalar_string(value: Any) -> str:
    """:param value: scalar-like value. :return: string scalar representation."""
    try:
        return str(np.asarray(value).reshape(()))
    except Exception:
        return str(value)


def load_warp_npz(path: Path) -> dict:
    """:param path: ``warp_transient_heads.npz`` path. :return: arrays dict."""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def load_npz_dict(path: Path) -> dict:
    """:param path: generic ``.npz`` path. :return: arrays dict."""
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def mf6_storage_budget_artifact_path(artifact_path: Path) -> Path:
    """:param artifact_path: MF6 heads artifact. :return: sibling storage-budget artifact path."""
    return artifact_path.with_name("mf6_storage_budget_terms.npz")


def warp_storage_budget_artifact_path(workspace: Path) -> Path:
    """:param workspace: replay workspace. :return: Warp storage-budget artifact path."""
    return workspace.joinpath("warp_storage_budget_terms.npz")


def workspace_replay_summary_path(workspace: Path) -> Path:
    """:param workspace: replay workspace. :return: replay summary JSON path."""
    return workspace.joinpath("transient_replay_summary.json")


def load_workspace_replay_summary(workspace: Path) -> dict:
    """:param workspace: replay workspace. :return: replay summary dict or empty dict."""
    path = workspace_replay_summary_path(workspace=workspace)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def load_period_infos(warp: dict) -> list[dict]:
    """:param warp: Warp NPZ arrays. :return: per-period solver info dictionaries."""
    if "period_infos" not in warp:
        return []
    try:
        parsed = json.loads(scalar_string(warp["period_infos"]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def free_mask_from_artifact(artifact: dict) -> np.ndarray:
    """:param artifact: MF6 truth artifact. :return: active non-Dirichlet mask."""
    active = np.asarray(artifact["active"], dtype=np.int32)
    bc_mask = np.asarray(artifact["bc_mask"], dtype=np.int32)
    return (active != 0) & (bc_mask == 0)


def classify_crossing_array(
    *,
    head_old: np.ndarray,
    head_new: np.ndarray,
    top: np.ndarray,
    free_mask: np.ndarray,
    near_top_tol: float = 1.0e-3,
) -> np.ndarray:
    """
    Classify per-cell top-crossing behaviour over one stress period.

    :param head_old: Previous-period head.
    :param head_new: New-period head.
    :param top: Model top.
    :param free_mask: Active non-Dirichlet mask.
    :param near_top_tol: Near-top classification tolerance.
    :return: Object array of class labels.
    """
    h_old = np.asarray(head_old, dtype=np.float64)
    h_new = np.asarray(head_new, dtype=np.float64)
    top_arr = np.asarray(top, dtype=np.float64)
    free = np.asarray(free_mask, dtype=bool)
    out = np.full(h_old.shape, "other", dtype=object)

    below_old = h_old < top_arr
    below_new = h_new < top_arr
    above_old = ~below_old
    above_new = ~below_new
    near_top = (np.abs(h_old - top_arr) < float(near_top_tol)) | (np.abs(h_new - top_arr) < float(near_top_tol))

    out[free & below_old & below_new & ~near_top] = "below_to_below"
    out[free & above_old & above_new & ~near_top] = "above_to_above"
    out[free & below_old & above_new] = "below_to_above"
    out[free & above_old & below_new] = "above_to_below"
    out[free & near_top & (out == "other")] = "near_top_no_crossing"
    return out


def compare_storage_budgets(
    *,
    artifact: dict,
    artifact_path: Path,
    workspace: Path,
) -> dict:
    """
    Compare MF6 storage budget arrays against Warp storage-term diagnostics.

    :param artifact: MF6 heads artifact.
    :param artifact_path: MF6 artifact path.
    :param workspace: Replay workspace.
    :return: Storage-budget comparison summary.
    """
    mf6_path = mf6_storage_budget_artifact_path(artifact_path=artifact_path)
    warp_path = warp_storage_budget_artifact_path(workspace=workspace)
    if not mf6_path.exists() or not warp_path.exists():
        return {
            "available": False,
            "storage_budget_diagnostics_available": False,
            "mf6_storage_budget_artifact": str(mf6_path),
            "warp_storage_budget_artifact": str(warp_path),
        }

    mf6_budget = load_npz_dict(mf6_path)
    warp_budget = load_npz_dict(warp_path)
    free = free_mask_from_artifact(artifact=artifact)
    top = np.asarray(artifact["top"], dtype=np.float64)

    required_mf6_keys = ("storage_total_per_period",)
    required_warp_keys = (
        "storage_terms_per_period",
        "sy_storage_terms_per_period",
        "ss_storage_terms_per_period",
        "heads_old_per_period",
        "heads_new_per_period",
    )
    missing_mf6 = [key for key in required_mf6_keys if key not in mf6_budget]
    missing_warp = [key for key in required_warp_keys if key not in warp_budget]
    if missing_mf6 or missing_warp:
        return {
            "available": False,
            "storage_budget_diagnostics_available": False,
            "mf6_storage_budget_artifact": str(mf6_path),
            "warp_storage_budget_artifact": str(warp_path),
            "reason": {
                "missing_mf6_keys": missing_mf6,
                "missing_warp_keys": missing_warp,
            },
        }

    mf6_total = np.asarray(mf6_budget["storage_total_per_period"], dtype=np.float64)
    warp_total_raw = np.asarray(warp_budget["storage_terms_per_period"], dtype=np.float64)
    warp_sy_raw = np.asarray(warp_budget["sy_storage_terms_per_period"], dtype=np.float64)
    warp_ss_raw = np.asarray(warp_budget["ss_storage_terms_per_period"], dtype=np.float64)
    heads_old = np.asarray(warp_budget["heads_old_per_period"], dtype=np.float64)
    heads_new = np.asarray(warp_budget["heads_new_per_period"], dtype=np.float64)
    if (
        mf6_total.shape != warp_total_raw.shape
        or warp_total_raw.shape != warp_sy_raw.shape
        or warp_total_raw.shape != warp_ss_raw.shape
        or warp_total_raw.shape != heads_old.shape
        or warp_total_raw.shape != heads_new.shape
    ):
        return {
            "available": False,
            "storage_budget_diagnostics_available": False,
            "mf6_storage_budget_artifact": str(mf6_path),
            "warp_storage_budget_artifact": str(warp_path),
            "reason": {
                "mf6_total_shape": list(mf6_total.shape),
                "warp_total_shape": list(warp_total_raw.shape),
                "warp_sy_shape": list(warp_sy_raw.shape),
                "warp_ss_shape": list(warp_ss_raw.shape),
                "heads_old_shape": list(heads_old.shape),
                "heads_new_shape": list(heads_new.shape),
            },
        }

    diff_direct = mf6_total - warp_total_raw
    diff_negated = mf6_total + warp_total_raw
    rmse_direct = float(np.sqrt(np.mean(diff_direct[:, free] * diff_direct[:, free])))
    rmse_negated = float(np.sqrt(np.mean(diff_negated[:, free] * diff_negated[:, free])))
    sign = -1.0 if rmse_negated < rmse_direct else 1.0
    storage_sign_used = (
        "warp_storage_multiplied_by_-1_before_mf6_minus_warp_comparison"
        if sign < 0.0
        else "warp_storage_multiplied_by_+1_before_mf6_minus_warp_comparison"
    )
    warp_total = sign * warp_total_raw
    warp_sy = sign * warp_sy_raw
    warp_ss = sign * warp_ss_raw

    rows: list[dict] = []
    class_summary: dict[str, dict[str, float]] = {}
    worst_cells: list[dict] = []
    worst_period_index = 0
    worst_period_rmse = -math.inf

    for period_index in range(mf6_total.shape[0]):
        mf6_period = mf6_total[period_index]
        warp_period = warp_total[period_index]
        diff = mf6_period - warp_period
        diff_free = diff[free]
        abs_free = np.abs(diff_free)
        rows.append(
            {
                "period": int(period_index + 1),
                "storage_rmse": float(np.sqrt(np.mean(diff_free * diff_free))),
                "storage_max_abs": float(np.max(abs_free)),
                "storage_mean_bias": float(np.mean(diff_free)),
                "storage_p95_abs": float(np.percentile(abs_free, 95.0)),
                "storage_p99_abs": float(np.percentile(abs_free, 99.0)),
                "n_compared_cells": int(diff_free.size),
            }
        )
        if rows[-1]["storage_rmse"] > worst_period_rmse:
            worst_period_rmse = rows[-1]["storage_rmse"]
            worst_period_index = period_index

    crossing = classify_crossing_array(
        head_old=heads_old[worst_period_index],
        head_new=heads_new[worst_period_index],
        top=top,
        free_mask=free,
    )
    worst_diff = mf6_total[worst_period_index] - warp_total[worst_period_index]
    abs_rank = np.where(free, np.abs(worst_diff), -np.inf).reshape(-1)
    count = min(20, int(np.count_nonzero(free)))
    selected = np.argpartition(abs_rank, -count)[-count:]
    selected = selected[np.argsort(abs_rank[selected])[::-1]]
    for rank, flat_index in enumerate(selected, start=1):
        j, i = np.unravel_index(int(flat_index), worst_diff.shape)
        worst_cells.append(
            {
                "rank": int(rank),
                "period": int(worst_period_index + 1),
                "i": int(i),
                "j": int(j),
                "mf6_storage": float(mf6_total[worst_period_index, j, i]),
                "warp_storage": float(warp_total[worst_period_index, j, i]),
                "warp_sy_storage": float(warp_sy[worst_period_index, j, i]),
                "warp_ss_storage": float(warp_ss[worst_period_index, j, i]),
                "storage_diff": float(worst_diff[j, i]),
                "crossing_class": str(crossing[j, i]),
                "head_old": float(heads_old[worst_period_index, j, i]),
                "head_new": float(heads_new[worst_period_index, j, i]),
            }
        )

    for class_name in (
        "below_to_below",
        "above_to_above",
        "below_to_above",
        "above_to_below",
        "near_top_no_crossing",
        "other",
    ):
        mask = free & (crossing == class_name)
        if not np.any(mask):
            continue
        class_diff = worst_diff[mask]
        class_summary[class_name] = {
            "n_cells": int(np.count_nonzero(mask)),
            "storage_rmse": float(np.sqrt(np.mean(class_diff * class_diff))),
            "storage_max_abs": float(np.max(np.abs(class_diff))),
            "storage_mean_bias": float(np.mean(class_diff)),
            "p95_abs": float(np.percentile(np.abs(class_diff), 95.0)),
            "p99_abs": float(np.percentile(np.abs(class_diff), 99.0)),
        }

    return {
        "available": True,
        "storage_budget_diagnostics_available": True,
        "mf6_storage_budget_artifact": str(mf6_path),
        "warp_storage_budget_artifact": str(warp_path),
        "mf6_budget_record_names": [
            scalar_string(name) for name in np.asarray(mf6_budget.get("unique_record_names", []), dtype=object)
        ],
        "selected_mf6_storage_record": scalar_string(
            mf6_budget.get("selected_storage_record_name", "")
        ),
        "warp_storage_sign": float(sign),
        "storage_sign_used": storage_sign_used,
        "rows": rows,
        "worst_period": int(worst_period_index + 1),
        "worst_period_index_zero_based": int(worst_period_index),
        "worst_cells": worst_cells,
        "error_by_crossing_class": class_summary,
        "final_storage_rmse": float(rows[-1]["storage_rmse"]),
        "final_storage_max_abs": float(rows[-1]["storage_max_abs"]),
        "worst_period_storage_rmse": float(rows[worst_period_index]["storage_rmse"]),
        "worst_period_storage_max_abs": float(rows[worst_period_index]["storage_max_abs"]),
        "practical_target_passed": None,
    }


def _record_key_from_budget_name(record_name: str) -> str | None:
    """
    Map a raw MF6 budget record name to a coarse package key.

    :param record_name: Raw MF6 budget record name.
    :return: One of ``recharge``, ``chd``, ``ghb``, ``storage``, or ``None``.
    """
    name = str(record_name).strip().lower().replace("-", "_").replace(" ", "_")
    if "rch" in name:
        return "recharge"
    if "chd" in name:
        return "chd"
    if "ghb" in name:
        return "ghb"
    if "sto" in name or "storage" in name:
        return "storage"
    return None


def mf6_mass_balance_from_budget_artifact(artifact_path: Path) -> dict:
    """
    Derive period-wise MF6 gross IN/OUT totals from the saved budget artifact.

    :param artifact_path: MF6 heads artifact path.
    :return: MF6 mass-balance summary or an unavailable record.
    """
    budget_path = mf6_storage_budget_artifact_path(artifact_path=artifact_path)
    if not budget_path.exists():
        return {
            "available": False,
            "mf6_mass_balance_available": False,
            "reason": f"missing budget artifact: {budget_path}",
        }
    budget = load_npz_dict(budget_path)
    record_names = [scalar_string(name) for name in np.asarray(budget.get("unique_record_names", []), dtype=object)]
    period_count = int(np.asarray(budget.get("period_count", 0)).reshape(())) if "period_count" in budget else 0
    if period_count < 1:
        return {
            "available": False,
            "mf6_mass_balance_available": False,
            "reason": "budget artifact missing period_count",
            "record_names": record_names,
        }

    per_package: dict[str, np.ndarray] = {}
    for key in budget:
        if not key.startswith("record_") or not key.endswith("_per_period"):
            continue
        record_name = key[len("record_"):-len("_per_period")]
        package_key = _record_key_from_budget_name(record_name=record_name)
        if package_key is None:
            continue
        arr = np.asarray(budget[key], dtype=np.float64)
        if arr.ndim == 4 and arr.shape[1] == 1:
            arr = arr[:, 0, :, :]
        if arr.ndim != 3:
            continue
        per_package[package_key] = per_package.get(package_key, np.zeros_like(arr)) + arr

    if not per_package:
        return {
            "available": False,
            "mf6_mass_balance_available": False,
            "reason": "no package budget arrays found",
            "record_names": record_names,
        }

    rows: list[dict] = []
    for period_index in range(period_count):
        row = {"period": int(period_index + 1)}
        total_in = 0.0
        total_out = 0.0
        for package_key in ("recharge", "chd", "ghb", "storage"):
            arr = per_package.get(package_key)
            if arr is None:
                row[f"{package_key}_in"] = 0.0
                row[f"{package_key}_out"] = 0.0
                continue
            values = np.asarray(arr[period_index], dtype=np.float64)
            gross_in = float(np.sum(np.maximum(values, 0.0)))
            gross_out = float(np.sum(np.maximum(-values, 0.0)))
            row[f"{package_key}_in"] = gross_in
            row[f"{package_key}_out"] = gross_out
            total_in += gross_in
            total_out += gross_out
        in_minus_out = total_in - total_out
        denom = abs(total_in) + abs(total_out)
        row["total_in"] = float(total_in)
        row["total_out"] = float(total_out)
        row["in_minus_out"] = float(in_minus_out)
        row["percent_discrepancy"] = 0.0 if denom == 0.0 else float(100.0 * in_minus_out / denom)
        rows.append(row)

    return {
        "available": True,
        "mf6_mass_balance_available": True,
        "record_names": record_names,
        "rows": rows,
    }


def compare_mass_balance(
    *,
    artifact_path: Path,
    workspace: Path,
) -> dict:
    """
    Compare Warp and MF6 per-period gross mass-balance summaries.

    :param artifact_path: MF6 heads artifact path.
    :param workspace: replay workspace.
    :return: comparison summary.
    """
    replay_summary = load_workspace_replay_summary(workspace=workspace)
    warp_mass_balance = replay_summary.get("mass_balance") if isinstance(replay_summary, dict) else None
    if not isinstance(warp_mass_balance, dict) or not warp_mass_balance.get("warp_mass_balance_available", False):
        return {
            "available": False,
            "warp_mass_balance_available": False,
            "mf6_mass_balance_available": False,
            "reason": "Warp mass balance missing from replay summary",
        }
    mf6_mass_balance = mf6_mass_balance_from_budget_artifact(artifact_path=artifact_path)
    out = {
        "available": bool(mf6_mass_balance.get("available", False)),
        "warp_mass_balance_available": True,
        "warp_storage_budget_available": bool(warp_mass_balance.get("warp_storage_budget_available", False)),
        "mf6_mass_balance_available": bool(mf6_mass_balance.get("mf6_mass_balance_available", False)),
        "mf6_storage_budget_available": "storage_total_per_period" in load_npz_dict(mf6_storage_budget_artifact_path(artifact_path=artifact_path))
        if mf6_storage_budget_artifact_path(artifact_path=artifact_path).exists()
        else False,
        "storage_budget_comparison_available": False,
        "mf6_budget_record_names": mf6_mass_balance.get("record_names", []),
        "rows": [],
    }
    if not mf6_mass_balance.get("available", False):
        out["reason"] = mf6_mass_balance.get("reason", "MF6 mass balance not available")
        return out

    warp_rows = warp_mass_balance.get("per_period", [])
    mf6_rows = mf6_mass_balance.get("rows", [])
    if len(warp_rows) != len(mf6_rows):
        out["available"] = False
        out["reason"] = f"period-count mismatch: warp={len(warp_rows)} mf6={len(mf6_rows)}"
        return out

    comparison_rows: list[dict] = []
    for warp_row, mf6_row in zip(warp_rows, mf6_rows):
        comparison_rows.append(
            {
                "period": int(warp_row["period"]),
                "warp_total_in": float(warp_row["total_in"]),
                "warp_total_out": float(warp_row["total_out"]),
                "warp_discrepancy_pct": float(warp_row["percent_discrepancy"]),
                "mf6_total_in": float(mf6_row["total_in"]),
                "mf6_total_out": float(mf6_row["total_out"]),
                "mf6_discrepancy_pct": float(mf6_row["percent_discrepancy"]),
                "warp_minus_mf6_storage": float(
                    (float(warp_row["storage_in"]) - float(warp_row["storage_out"]))
                    - (float(mf6_row.get("storage_in", 0.0)) - float(mf6_row.get("storage_out", 0.0)))
                ),
                "warp_minus_mf6_total_budget": float(
                    float(warp_row["in_minus_out"]) - float(mf6_row["in_minus_out"])
                ),
            }
        )
    out["rows"] = comparison_rows
    return out


def period_error_table(artifact: dict, warp: dict) -> tuple[list[dict], dict]:
    """:param artifact: MF6 truth artifact. :param warp: Warp NPZ arrays. :return: table and pattern summary."""
    mf6_heads = np.asarray(artifact["heads_per_period"], dtype=np.float64)
    warp_heads = np.asarray(warp["heads_per_period"], dtype=np.float64)
    if mf6_heads.shape != warp_heads.shape:
        raise ValueError(f"head shape mismatch: MF6 {mf6_heads.shape}, Warp {warp_heads.shape}")

    free = free_mask_from_artifact(artifact)
    rows: list[dict] = []
    previous_warp = np.asarray(warp.get("warm_start_head", artifact["initial_head"]), dtype=np.float64)
    previous_mf6 = np.asarray(artifact["initial_head"], dtype=np.float64)

    for period_index in range(mf6_heads.shape[0]):
        wh = warp_heads[period_index]
        mh = mf6_heads[period_index]
        diff_full = wh - mh
        diff = diff_full[free]
        abs_diff = np.abs(diff)
        worst_flat = int(np.argmax(np.where(free, np.abs(diff_full), -np.inf)))
        worst_j, worst_i = np.unravel_index(worst_flat, wh.shape)
        warp_change = wh - previous_warp
        mf6_change = mh - previous_mf6
        change_diff = warp_change - mf6_change

        rows.append(
            {
                "period": int(period_index + 1),
                "max_abs_diff": float(np.max(abs_diff)),
                "rmse": float(np.sqrt(np.mean(diff * diff))),
                "mean_diff": float(np.mean(diff)),
                "median_diff": float(np.median(diff)),
                "p95_abs_diff": float(np.percentile(abs_diff, 95.0)),
                "p99_abs_diff": float(np.percentile(abs_diff, 99.0)),
                "min_diff": float(np.min(diff)),
                "max_diff": float(np.max(diff)),
                "max_diff_i": int(worst_i),
                "max_diff_j": int(worst_j),
                "warp_head_min": float(np.min(wh[free])),
                "warp_head_max": float(np.max(wh[free])),
                "mf6_head_min": float(np.min(mh[free])),
                "mf6_head_max": float(np.max(mh[free])),
                "max_abs_period_head_change_warp": float(np.max(np.abs(warp_change[free]))),
                "max_abs_period_head_change_mf6": float(np.max(np.abs(mf6_change[free]))),
                "rmse_period_head_change": float(np.sqrt(np.mean(change_diff[free] * change_diff[free]))),
                "percent_within_0_01m": float(np.mean(abs_diff <= 0.01) * 100.0),
                "percent_within_0_1m": float(np.mean(abs_diff <= 0.1) * 100.0),
            }
        )
        previous_warp = wh
        previous_mf6 = mh

    pattern = classify_error_pattern(rows=rows, free_count=int(np.count_nonzero(free)))
    return rows, pattern


def classify_error_pattern(rows: list[dict], free_count: int) -> dict:
    """:param rows: period error rows. :param free_count: active free-cell count. :return: pattern summary."""
    max_values = np.asarray([row["max_abs_diff"] for row in rows], dtype=np.float64)
    rmse_values = np.asarray([row["rmse"] for row in rows], dtype=np.float64)
    worst_index = int(np.argmax(max_values))
    monotonic = bool(np.all(np.diff(max_values) >= -1.0e-12))
    peaks_then_declines = bool(0 < worst_index < len(rows) - 1 and max_values[-1] < max_values[worst_index])
    if len(max_values) > 1:
        second = float(np.partition(max_values, -2)[-2])
    else:
        second = 0.0
    spikes = bool(max_values[worst_index] > 2.0 * max(second, 1.0e-30))
    p99_ratio = float(rows[worst_index]["p99_abs_diff"] / max(rows[worst_index]["max_abs_diff"], 1.0e-30))
    localised = bool(p99_ratio < 0.35)
    broad_smooth = bool(p99_ratio >= 0.65)
    return {
        "worst_period": int(worst_index + 1),
        "free_cell_count": int(free_count),
        "grows_monotonically": monotonic,
        "peaks_then_declines": peaks_then_declines,
        "spikes_at_one_period": spikes,
        "is_spatially_localised": localised,
        "is_broad_and_smooth": broad_smooth,
        "worst_period_p99_to_max_abs_ratio": p99_ratio,
        "max_abs_series": [float(value) for value in max_values],
        "rmse_series": [float(value) for value in rmse_values],
    }


def period_storage_diag(artifact: dict, warp: dict, period_index: int) -> np.ndarray:
    """:param artifact: MF6 artifact. :param warp: Warp arrays. :param period_index: zero-based period. :return: storage diagonal."""
    if "storage_coeffs_per_period" not in warp:
        raise KeyError("warp replay artifact is missing storage_coeffs_per_period")
    coeffs = np.asarray(warp["storage_coeffs_per_period"], dtype=np.float64)
    dx = float(np.asarray(artifact["dx"]).reshape(()))
    dt = float(np.asarray(artifact["dt_days"]).reshape(()))
    return coeffs[period_index] * dx * dx / dt


def nearest_mask_distance(i: int, j: int, mask: np.ndarray) -> int | str:
    """:param i: column. :param j: row. :param mask: boolean mask. :return: Manhattan distance or not_available."""
    coords = np.argwhere(mask)
    if coords.size == 0:
        return NOT_AVAILABLE
    distances = np.abs(coords[:, 0] - int(j)) + np.abs(coords[:, 1] - int(i))
    return int(np.min(distances))


def classify_cell(
    *,
    i: int,
    j: int,
    artifact: dict,
    warp_head: float,
    mf6_head: float,
) -> str:
    """:param i: column. :param j: row. :param artifact: MF6 artifact. :param warp_head: Warp head. :param mf6_head: MF6 head. :return: label."""
    active = np.asarray(artifact["active"], dtype=np.int32)
    bc_mask = np.asarray(artifact["bc_mask"], dtype=np.int32)
    bottom = np.asarray(artifact["bottom"], dtype=np.float64)
    top = np.asarray(artifact["top"], dtype=np.float64)
    if active[j, i] == 0:
        return "near inactive boundary"
    if bc_mask[j, i] != 0:
        return "near CHD"
    sat_warp = warp_head - float(bottom[j, i])
    sat_mf6 = mf6_head - float(bottom[j, i])
    full = float(top[j, i] - bottom[j, i])
    if abs(sat_warp - 0.1) < 0.5 or abs(sat_mf6 - 0.1) < 0.5:
        return "near min_sat"
    if abs(warp_head - float(top[j, i])) < 0.5 or abs(mf6_head - float(top[j, i])) < 0.5:
        return "near water table clipping"
    if nearest_mask_distance(i=i, j=j, mask=(bc_mask != 0)) != NOT_AVAILABLE:
        if int(nearest_mask_distance(i=i, j=j, mask=(bc_mask != 0))) <= 2:
            return "near CHD"
    if nearest_mask_distance(i=i, j=j, mask=(active == 0)) != NOT_AVAILABLE:
        if int(nearest_mask_distance(i=i, j=j, mask=(active == 0))) <= 2:
            return "near inactive boundary"
    if sat_warp > 0.1 and sat_mf6 > 0.1 and warp_head < float(top[j, i]) and mf6_head < float(top[j, i]):
        return "interior smooth-field error"
    if full <= 0.0:
        return "unknown"
    return "unknown"


def worst_cells_table(artifact: dict, warp: dict, period: int, limit: int = 20) -> list[dict]:
    """:param artifact: MF6 artifact. :param warp: Warp arrays. :param period: one-based period. :param limit: max rows. :return: rows."""
    mf6_heads = np.asarray(artifact["heads_per_period"], dtype=np.float64)
    warp_heads = np.asarray(warp["heads_per_period"], dtype=np.float64)
    free = free_mask_from_artifact(artifact)
    period_index = int(period) - 1
    wh = warp_heads[period_index]
    mh = mf6_heads[period_index]
    diff = wh - mh
    abs_for_rank = np.where(free, np.abs(diff), -np.inf)
    flat = abs_for_rank.reshape(-1)
    count = min(int(limit), int(np.count_nonzero(free)))
    selected = np.argpartition(flat, -count)[-count:]
    selected = selected[np.argsort(flat[selected])[::-1]]

    bottom = np.asarray(artifact["bottom"], dtype=np.float64)
    top = np.asarray(artifact["top"], dtype=np.float64)
    active = np.asarray(artifact["active"], dtype=np.int32)
    bc_mask = np.asarray(artifact["bc_mask"], dtype=np.int32)
    recharge_rates = np.asarray(artifact["recharge_rates"], dtype=np.float64)
    k_field = np.asarray(artifact["k_field"], dtype=np.float64)
    storage_diag = period_storage_diag(artifact=artifact, warp=warp, period_index=period_index)
    sy = float(np.asarray(artifact["sy"]).reshape(()))
    ss = float(np.asarray(artifact["ss"]).reshape(()))
    recharge = float(recharge_rates[period_index])
    rows: list[dict] = []
    chd_mask = bc_mask != 0
    inactive_mask = active == 0
    for rank, flat_index in enumerate(selected, start=1):
        j, i = np.unravel_index(int(flat_index), wh.shape)
        warp_head = float(wh[j, i])
        mf6_head = float(mh[j, i])
        rows.append(
            {
                "rank": int(rank),
                "i": int(i),
                "j": int(j),
                "warp_head": warp_head,
                "mf6_head": mf6_head,
                "diff": float(diff[j, i]),
                "abs_diff": float(abs(diff[j, i])),
                "bottom": float(bottom[j, i]),
                "top_or_dem_if_available": float(top[j, i]),
                "sat_thickness_warp": float(max(warp_head - bottom[j, i], 0.0)),
                "sat_thickness_mf6_if_available": float(max(mf6_head - bottom[j, i], 0.0)),
                "active": int(active[j, i]),
                "bc_mask": int(bc_mask[j, i]),
                "distance_or_flag_near_chd": nearest_mask_distance(i=int(i), j=int(j), mask=chd_mask),
                "distance_or_flag_near_ghb": NOT_AVAILABLE,
                "distance_or_flag_near_inactive": nearest_mask_distance(i=int(i), j=int(j), mask=inactive_mask),
                "recharge": recharge,
                "K_or_T": float(k_field[j, i]),
                "storage_diag": float(storage_diag[j, i]),
                "Sy": sy,
                "Ss": ss,
                "classification": classify_cell(
                    i=int(i),
                    j=int(j),
                    artifact=artifact,
                    warp_head=warp_head,
                    mf6_head=mf6_head,
                ),
            }
        )
    return rows


def picard_timing_table(artifact: dict, warp: dict, period: int) -> list[dict]:
    """:param artifact: MF6 artifact. :param warp: Warp arrays. :param period: one-based period. :return: Picard rows."""
    infos = load_period_infos(warp=warp)
    period_index = int(period) - 1
    if period_index < 0 or period_index >= len(infos):
        return []
    outer_history = infos[period_index].get("outer_history", [])
    if not isinstance(outer_history, list):
        return []

    storage_diag = period_storage_diag(
        artifact=artifact,
        warp=warp,
        period_index=period_index,
    )
    free = free_mask_from_artifact(artifact)
    storage_values = storage_diag[free]
    storage_min = float(np.min(storage_values)) if storage_values.size else None
    storage_max = float(np.max(storage_values)) if storage_values.size else None
    storage_mean = float(np.mean(storage_values)) if storage_values.size else None

    rows: list[dict] = []
    previous_storage_min = None
    previous_storage_max = None
    for entry in outer_history:
        if not isinstance(entry, dict):
            continue
        sat_min = entry.get("min_saturated_thickness", NOT_AVAILABLE)
        sat_max = entry.get("max_saturated_thickness", NOT_AVAILABLE)
        entry_storage_min = entry.get("storage_diag_min", storage_min)
        entry_storage_max = entry.get("storage_diag_max", storage_max)
        entry_storage_mean = entry.get("storage_diag_mean", storage_mean)
        rows.append(
            {
                "period": int(period),
                "outer_iter": entry.get("outer_iteration", NOT_AVAILABLE),
                "omega": entry.get("omega", NOT_AVAILABLE),
                "max_abs_head_change": entry.get("max_abs_head_change", entry.get("picard_update_max", NOT_AVAILABLE)),
                "rms_head_change": entry.get("picard_update_rms", NOT_AVAILABLE),
                "inner_converged": entry.get("inner_converged", NOT_AVAILABLE),
                "storage_diag_min": entry_storage_min,
                "storage_diag_max": entry_storage_max,
                "storage_diag_mean": entry_storage_mean,
                "sat_thickness_min": sat_min,
                "sat_thickness_max": sat_max,
                "sat_thickness_mean": NOT_AVAILABLE,
                "rhs_eff_min": NOT_AVAILABLE,
                "rhs_eff_max": NOT_AVAILABLE,
                "rhs_eff_mean": NOT_AVAILABLE,
                "max_abs_storage_diag_change": (
                    NOT_AVAILABLE
                    if previous_storage_min is None or previous_storage_max is None
                    else max(
                        abs(float(entry_storage_min) - previous_storage_min),
                        abs(float(entry_storage_max) - previous_storage_max),
                    )
                ),
                "rms_storage_diag_change": NOT_AVAILABLE,
                "inner_iterations": entry.get("inner_iterations", NOT_AVAILABLE),
                "inner_residual": entry.get("inner_residual", NOT_AVAILABLE),
            }
        )
        previous_storage_min = float(entry_storage_min)
        previous_storage_max = float(entry_storage_max)
    return rows


def mf6_semantics_summary(artifact: dict) -> dict:
    """:param artifact: MF6 artifact. :return: code-grounded MF6 semantics summary."""
    provenance = artifact.get("provenance")
    parsed = None
    if provenance is not None:
        try:
            parsed = json.loads(scalar_string(provenance))
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = None
    return {
        "source": "working_tests/run_2d_transient_vs_mf6.py",
        "tdis": "time_units=DAYS; perioddata=[(case.dt_days, 1, 1.0)] * case.n_weeks",
        "dis": "top=case.top; botm=case.bottom; idomain=case.active",
        "ic": "strt=transient_initial_head",
        "npf": "icelltype=[1] for unconfined; k=case.hydraulic_conductivity; save_saturation=True",
        "sto": "ss=case.ss; sy=case.sy; iconvert=1; transient={0: True}",
        "artifact_sy": finite_scalar(artifact["sy"]),
        "artifact_ss": finite_scalar(artifact["ss"]),
        "artifact_dt_days": finite_scalar(artifact["dt_days"]),
        "artifact_warm_start_mode": None if not isinstance(parsed, dict) else parsed.get("warm_start_mode"),
        "darcywarp_mf6_compatible_mode": (
            "mf6_convertible_secant_sy -> secant Sy crossing coefficient + Ss * clipped saturated_thickness_reference"
        ),
        "semantic_match_assessment": (
            "DarcyWarp still uses a single-layer 2D approximation, but the secant-Sy mode is the "
            "closest tested head match because it keeps a fractional Sy contribution when heads "
            "cross the model top within a timestep."
        ),
    }


def _replay_variant_configs(variant_set: str = "full") -> dict:
    """
    Return normal replay-analysis variants.

    The normal harness intentionally exposes only the validated production
    secant-Sy replay and secant-Sy freeze-after-outer variants.

    :param variant_set: ``full`` or ``speed``. Both normal sets expose the same
        production/freeze secant-Sy matrix.
    :return: Ordered mapping of replay variant name to config dict.
    """
    variant_mode = str(variant_set).strip().lower()
    if variant_mode not in {"full", "speed"}:
        raise ValueError("variant_set must be 'full' or 'speed'.")
    variants: dict[str, Any] = {
        "production_secant_sy": production_secant_sy_settings(),
    }
    for freeze_after_outer in (2, 3, 4, 5, 6, 8, 10):
        variants[f"secant_sy_freeze_after_outer_{freeze_after_outer}"] = secant_sy_freeze_settings(
            freeze_after_outer=freeze_after_outer,
        )
    return variants


WINNING_VARIANT_NAME = "production_secant_sy"

# Fields the direct MF6 replay and the winning variant must agree on (Task 2).
CONSISTENCY_FIELDS = (
    "unconfined_storage_mode",
    "storage_reference",
    "storage_top_threshold",
    "storage_active_set_strategy",
    "unconfined_startup_mode",
    "warm_start",
    "max_cycles",
    "max_outer_iterations",
    "hclose",
    "dh_rms_tol",
    "residual_floor_tol",
    "inner_head_residual_tol_min",
    "inner_head_residual_tol_max",
    "omega",
    "omega_min",
    "omega_max",
    "chebyshev_enabled",
    "chebyshev_order",
    "cheby_lambda_min",
    "cheby_lambda_max",
    "nu_pre",
    "nu_post",
    "nu_coarse",
)


def direct_replay_settings() -> dict:
    """:return: the effective direct MF6 replay settings (main/run_replay_from_artifact defaults merged with default_solve_controls)."""
    production = production_secant_sy_settings()
    controls = production["solve_controls"]
    return {
        "unconfined_storage_mode": production["unconfined_storage_mode"],
        "storage_reference": production["storage_reference"],
        "storage_top_threshold": production["storage_top_threshold"],
        "storage_active_set_strategy": production["storage_active_set_strategy"],
        "unconfined_startup_mode": controls["unconfined_startup_mode"],
        "warm_start": production["warm_start_mode"],
        "max_cycles": controls["max_cycles"],
        "max_outer_iterations": controls["max_outer_iterations"],
        "hclose": controls["hclose"],
        "dh_rms_tol": controls["dh_rms_tol"],
        "residual_floor_tol": controls["residual_floor_tol"],
        "inner_head_residual_tol_min": controls["inner_head_residual_tol_min"],
        "inner_head_residual_tol_max": controls["inner_head_residual_tol_max"],
        "omega": controls["omega"],
        "omega_min": controls["omega_min"],
        "omega_max": controls["omega_max"],
        "chebyshev_enabled": controls["chebyshev_enabled"],
        "chebyshev_order": controls["chebyshev_order"],
        "cheby_lambda_min": controls["cheby_lambda_min"],
        "cheby_lambda_max": controls["cheby_lambda_max"],
        "nu_pre": controls["nu_pre"],
        "nu_post": controls["nu_post"],
        "nu_coarse": controls["nu_coarse"],
    }


def winning_variant_settings_from_config(cfg: dict) -> dict:
    """:param cfg: a variant config dict from :func:`_replay_variant_configs`. :return: the effective settings that variant runs with, in the CONSISTENCY_FIELDS shape."""
    controls = cfg["solve_controls"]
    return {
        "unconfined_storage_mode": cfg["unconfined_storage_mode"],
        "storage_reference": cfg["storage_reference"],
        "storage_top_threshold": cfg["storage_top_threshold"],
        "storage_active_set_strategy": cfg.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE),
        "unconfined_startup_mode": controls["unconfined_startup_mode"],
        "warm_start": cfg.get("warm_start_mode", WARM_START_UNCONFINED_STEADY_MF6),
        "max_cycles": controls["max_cycles"],
        "max_outer_iterations": controls["max_outer_iterations"],
        "hclose": controls["hclose"],
        "dh_rms_tol": controls["dh_rms_tol"],
        "residual_floor_tol": controls["residual_floor_tol"],
        "inner_head_residual_tol_min": controls["inner_head_residual_tol_min"],
        "inner_head_residual_tol_max": controls["inner_head_residual_tol_max"],
        "omega": controls["omega"],
        "omega_min": controls["omega_min"],
        "omega_max": controls["omega_max"],
        "chebyshev_enabled": controls["chebyshev_enabled"],
        "chebyshev_order": controls["chebyshev_order"],
        "cheby_lambda_min": controls["cheby_lambda_min"],
        "cheby_lambda_max": controls["cheby_lambda_max"],
        "nu_pre": controls["nu_pre"],
        "nu_post": controls["nu_post"],
        "nu_coarse": controls["nu_coarse"],
    }


def check_direct_vs_winning_variant() -> dict:
    """Compare the direct MF6 replay settings against the winning variant.

    :return: report dict with ``consistent`` (bool), ``winning_variant`` name,
        ``mismatches`` (list of {field, direct, winning_variant}), and the two
        full setting dicts. Prints a loud warning when any material field differs.
    """
    configs = _replay_variant_configs()
    cfg = configs[WINNING_VARIANT_NAME]
    direct = direct_replay_settings()
    winning = winning_variant_settings_from_config(cfg)
    mismatches: list[dict] = []
    for field in CONSISTENCY_FIELDS:
        direct_value = direct.get(field)
        winning_value = winning.get(field)
        if direct_value != winning_value:
            mismatches.append(
                {
                    "field": field,
                    "direct": direct_value,
                    "winning_variant": winning_value,
                }
            )
    consistent = not mismatches
    if not consistent:
        print("\nWARNING: direct MF6 replay and the winning variant differ on material settings:")
        for m in mismatches:
            print(
                f"  {m['field']}: direct={m['direct']!r} winning_variant={m['winning_variant']!r}"
            )
    else:
        print(f"\nDirect MF6 replay matches winning variant '{WINNING_VARIANT_NAME}' on all material settings.")
    return {
        "consistent": bool(consistent),
        "winning_variant": WINNING_VARIANT_NAME,
        "direct_settings": direct,
        "winning_variant_settings": winning,
        "mismatches": mismatches,
    }


def _effective_solver_active_set_strategy(cfg: dict) -> str:
    """
    Return the lower-level solver active-set strategy for a replay config.

    :param cfg: Replay variant config.
    :return: Strategy passed into the solver.
    """
    if cfg.get("storage_freeze_after_outer") is not None:
        return STORAGE_ACTIVE_SET_FREEZE_WHEN_STABLE
    return cfg.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE)


def run_optional_replay_variants(
    *,
    artifact_path: Path,
    output_dir: Path,
    device: str,
    allow_warm_start_mismatch: bool,
    variant_set: str = "full",
) -> dict:
    """:param artifact_path: MF6 artifact. :param output_dir: output directory. :param device: Warp device. :param allow_warm_start_mismatch: permit current diagnostic mismatch. :return: variant summary."""
    variants = _replay_variant_configs(variant_set=variant_set)
    results: dict[str, Any] = {}
    for name, cfg in variants.items():
        workspace = output_dir.joinpath(f"replay_{name}")
        summary = run_replay_from_artifact(
            artifact_path=artifact_path,
            workspace=workspace,
            device=device,
            diag_preconditioner_backend="device" if device != "cpu" else "host",
            solve_controls=cfg["solve_controls"],
            warm_start_mode=cfg.get("warm_start_mode", WARM_START_UNCONFINED_STEADY_MF6),
            formulation=FORMULATION_UNCONFINED,
            unconfined_storage_mode=cfg["unconfined_storage_mode"],
            storage_reference=cfg["storage_reference"],
            storage_top_threshold=cfg["storage_top_threshold"],
            storage_active_set_strategy=_effective_solver_active_set_strategy(cfg),
            storage_hysteresis_eps=cfg.get("storage_hysteresis_eps", 0.0),
            storage_freeze_after_stable_iterations=cfg.get("storage_freeze_after_stable_iterations", 0),
            storage_freeze_after_outer=cfg.get("storage_freeze_after_outer"),
            storage_switch_fraction_tol=cfg.get("storage_switch_fraction_tol", 0.0),
            allow_warm_start_mismatch=allow_warm_start_mismatch,
        )
        comparison = summary.get("comparison", {}) if isinstance(summary, dict) else {}
        variant_rows, variant_pattern, variant_worst_cells, variant_picard, variant_storage = variant_diagnostics_from_workspace(
            artifact_path=artifact_path,
            workspace=workspace,
            worst_period_number=comparison.get(
                "worst_period_number_one_based",
                comparison.get("worst_period"),
            ),
        )
        results[name] = {
            "variant_name": name,
            "workspace": str(workspace),
            "settings": {
                "solve_controls": cfg["solve_controls"],
                "unconfined_storage_mode": cfg["unconfined_storage_mode"],
                "storage_reference": cfg["storage_reference"],
                "storage_top_threshold": cfg["storage_top_threshold"],
                "storage_active_set_strategy": cfg.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE),
                "storage_hysteresis_eps": cfg.get("storage_hysteresis_eps", 0.0),
                "storage_freeze_after_stable_iterations": cfg.get(
                    "storage_freeze_after_stable_iterations", 0
                ),
                "storage_freeze_after_outer": cfg.get("storage_freeze_after_outer"),
                "storage_switch_fraction_tol": cfg.get("storage_switch_fraction_tol", 0.0),
                "warm_start_mode": cfg.get("warm_start_mode", WARM_START_UNCONFINED_STEADY_MF6),
            },
            "final": summary.get("comparison", {}).get("final", {}),
            "worst_period": comparison.get(
                "worst_period_number_one_based",
                comparison.get("worst_period"),
            ),
            "worst_period_index_zero_based": comparison.get("worst_period_index_zero_based"),
            "period_error": variant_rows,
            "error_pattern": variant_pattern,
            "worst_cells": variant_worst_cells,
            "picard_diagnostics": variant_picard,
            "storage_budget": variant_storage,
            "mass_balance": summary.get("mass_balance", {}) if isinstance(summary, dict) else {},
            "mf6_mass_balance_comparison": compare_mass_balance(
                artifact_path=artifact_path,
                workspace=workspace,
            ),
            # Captured so full per-variant metrics can be reported without
            # reloading the workspace JSON.
            "convergence": summary.get("convergence", {}) if isinstance(summary, dict) else {},
            "period_convergence": summary.get("period_convergence", {}) if isinstance(summary, dict) else {},
            "timing": summary.get("timing", {}) if isinstance(summary, dict) else {},
            "mf6_replay_settings": summary.get("mf6_replay_settings", {}) if isinstance(summary, dict) else {},
            "runtime": finite_scalar((summary.get("timing", {}) if isinstance(summary, dict) else {}).get("warp_total_time")),
        }
    return results


def variant_diagnostics_from_workspace(
    *,
    artifact_path: Path,
    workspace: Path,
    worst_period_number: int | None,
) -> tuple[list[dict], dict, list[dict], list[dict], dict]:
    """:param artifact_path: MF6 artifact. :param workspace: variant workspace. :param worst_period_number: one-based replay summary worst period. :return: diagnostics."""
    artifact = load_transient_artifact(artifact_path)
    warp = load_warp_npz(workspace.joinpath("warp_transient_heads.npz"))
    period_rows, pattern = period_error_table(artifact=artifact, warp=warp)
    if worst_period_number is None:
        selected_period = int(pattern["worst_period"])
    else:
        selected_period = int(worst_period_number)
    worst_cells = worst_cells_table(
        artifact=artifact,
        warp=warp,
        period=selected_period,
        limit=20,
    )
    picard_rows = picard_timing_table(
        artifact=artifact,
        warp=warp,
        period=selected_period,
    )
    storage_budget = compare_storage_budgets(
        artifact=artifact,
        artifact_path=artifact_path,
        workspace=workspace,
    )
    return period_rows, pattern, worst_cells, picard_rows, storage_budget


def variant_worst_period_metrics(result: dict) -> dict:
    """:param result: variant result dictionary. :return: worst-period metrics."""
    rows = result.get("period_error", [])
    if not isinstance(rows, list) or not rows:
        return {}
    worst_row = max(rows, key=period_row_max_abs_key)
    return {
        "best_variant_worst_period": int(worst_row["period"]),
        "best_variant_worst_period_number": int(worst_row["period"]),
        "best_variant_worst_period_index_zero_based": int(worst_row["period"]) - 1,
        "best_variant_worst_period_max_abs_diff": float(worst_row["max_abs_diff"]),
        "best_variant_worst_period_rmse": float(worst_row["rmse"]),
    }


def period_row_max_abs_key(row: dict) -> float:
    """:param row: period metric row. :return: max absolute difference key."""
    return float(row.get("max_abs_diff", -math.inf))


def period_row_rmse_key(row: dict) -> float:
    """:param row: period metric row. :return: rmse sort key."""
    return float(row.get("rmse", -math.inf))


def variant_full_metrics(result: dict) -> dict:
    """Compute the full per-variant metric block (Tasks 4 and 5).

    :param result: a variant result dict from :func:`run_optional_replay_variants`.
    :return: dict with final and worst-period head-error metrics, the periods of
        max rmse / max abs, nonconverged period list, outer-iteration totals, and
        active-set switching fractions. Missing data degrades to ``None``.
    """
    final = result.get("final") or {}
    rows = result.get("period_error") or []
    period_conv = result.get("period_convergence") or {}
    periods = period_conv.get("periods") or []

    final_max_abs = finite_scalar(final.get("max_abs_diff"))
    final_rmse = finite_scalar(final.get("rmse"))
    final_bias = finite_scalar(final.get("mean_bias_warp_minus_mf6") or final.get("mean_diff"))
    final_percent_within_0_01m = finite_scalar(final.get("percent_within_0_01m"))
    final_percent_within_0_1m = finite_scalar(final.get("percent_within_0_1m"))

    if rows:
        worst_abs_row = max(rows, key=period_row_max_abs_key)
        worst_rmse_row = max(rows, key=period_row_rmse_key)
        worst_period = int(worst_abs_row["period"])
        worst_period_index_zero_based = int(worst_period - 1)
        worst_period_max_abs_diff = float(worst_abs_row["max_abs_diff"])
        worst_period_rmse = float(worst_abs_row["rmse"])
        worst_period_bias = float(worst_abs_row.get("mean_diff", math.nan))
        period_of_max_abs = int(worst_abs_row["period"])
        period_of_max_rmse = int(worst_rmse_row["period"])
    else:
        worst_period = None
        worst_period_index_zero_based = None
        worst_period_max_abs_diff = None
        worst_period_rmse = None
        worst_period_bias = None
        period_of_max_abs = None
        period_of_max_rmse = None

    nonconverged_periods = [
        int(p["period"]) for p in periods if isinstance(p, dict) and not p.get("converged", False)
    ]
    outer_iters = [
        int(p.get("outer_iterations", 0) or 0) for p in periods if isinstance(p, dict)
    ]
    total_outer_iterations = int(sum(outer_iters)) if outer_iters else None
    max_outer_iterations_per_period = int(max(outer_iters)) if outer_iters else None
    converged = bool(not nonconverged_periods) if periods else None
    period_1_outer_iterations = int(outer_iters[0]) if outer_iters else None

    last_changed = [
        float(p.get("last_top_switch_changed_fraction", 0.0) or 0.0)
        for p in periods
        if isinstance(p, dict)
    ]
    max_changed = [
        float(p.get("max_top_switch_changed_fraction", 0.0) or 0.0)
        for p in periods
        if isinstance(p, dict)
    ]
    last_top_switch_changed_fraction = last_changed[-1] if last_changed else None
    max_top_switch_changed_fraction = max(max_changed) if max_changed else None

    timing = result.get("timing") or {}
    runtime = finite_scalar(result.get("runtime"))
    mean_period_runtime = finite_scalar(timing.get("warp_period_time_mean"))
    max_period_runtime = finite_scalar(timing.get("warp_period_time_max"))
    period_1_runtime = finite_scalar(timing.get("warp_period_1_time"))

    convergence = result.get("convergence") or {}
    strict_picard_convergence_passed = convergence.get("strict_picard_convergence_passed")
    practical_picard_acceptance_passed = convergence.get("practical_picard_acceptance_passed")
    production_acceptance_passed = convergence.get("production_acceptance_passed")
    storage_budget = result.get("storage_budget") or {}
    storage_budget_diagnostics_available = bool(storage_budget.get("available", False))
    final_storage_rmse = finite_scalar(storage_budget.get("final_storage_rmse"))
    final_storage_max_abs = finite_scalar(storage_budget.get("final_storage_max_abs"))
    worst_period_storage_rmse = finite_scalar(storage_budget.get("worst_period_storage_rmse"))
    worst_period_storage_max_abs = finite_scalar(storage_budget.get("worst_period_storage_max_abs"))
    storage_error_by_crossing_class = storage_budget.get("error_by_crossing_class")
    storage_sign_used = storage_budget.get("storage_sign_used")
    mass_balance = result.get("mass_balance") or {}
    mass_balance_worst = mass_balance.get("worst_period", {}) if isinstance(mass_balance, dict) else {}
    max_abs_percent_discrepancy = finite_scalar(mass_balance.get("max_abs_percent_discrepancy"))
    max_abs_in_minus_out = finite_scalar(mass_balance.get("max_abs_in_minus_out"))
    all_period_practical_target_passed = bool(
        rows
        and all(float(r["rmse"]) < 0.01 and float(r["max_abs_diff"]) < 0.05 for r in rows)
    ) if rows else None
    final_period_practical_target_passed = bool(
        final_max_abs is not None
        and final_rmse is not None
        and final_max_abs < 0.05
        and final_rmse < 0.01
    )

    return {
        "variant_name": result.get("variant_name"),
        "final_max_abs_diff": final_max_abs,
        "final_rmse": final_rmse,
        "final_bias": final_bias,
        "final_percent_within_0_01m": final_percent_within_0_01m,
        "final_percent_within_0_1m": final_percent_within_0_1m,
        "final_period_practical_target_passed": final_period_practical_target_passed,
        "all_period_practical_target_passed": all_period_practical_target_passed,
        "worst_period": worst_period,
        "worst_period_number": worst_period,
        "worst_period_index_zero_based": worst_period_index_zero_based,
        "worst_period_max_abs_diff": worst_period_max_abs_diff,
        "worst_period_rmse": worst_period_rmse,
        "worst_period_bias": worst_period_bias,
        "period_of_max_rmse": period_of_max_rmse,
        "period_of_max_abs": period_of_max_abs,
        "converged": converged,
        "nonconverged_periods": nonconverged_periods,
        "total_outer_iterations": total_outer_iterations,
        "max_outer_iterations_per_period": max_outer_iterations_per_period,
        "period_1_outer_iterations": period_1_outer_iterations,
        "runtime": runtime,
        "period_1_runtime": period_1_runtime,
        "mean_period_runtime": mean_period_runtime,
        "max_period_runtime": max_period_runtime,
        "last_top_switch_changed_fraction": last_top_switch_changed_fraction,
        "max_top_switch_changed_fraction": max_top_switch_changed_fraction,
        "top_switch_frozen": convergence.get("top_switch_frozen"),
        "top_switch_frozen_outer_iteration": convergence.get("top_switch_frozen_outer_iteration"),
        "strict_picard_convergence_passed": strict_picard_convergence_passed,
        "practical_picard_acceptance_passed": practical_picard_acceptance_passed,
        "production_acceptance_passed": production_acceptance_passed,
        "storage_budget_diagnostics_available": storage_budget_diagnostics_available,
        "final_storage_rmse": final_storage_rmse,
        "final_storage_max_abs": final_storage_max_abs,
        "worst_period_storage_rmse": worst_period_storage_rmse,
        "worst_period_storage_max_abs": worst_period_storage_max_abs,
        "storage_error_by_crossing_class": storage_error_by_crossing_class,
        "storage_sign_used": storage_sign_used,
        "mass_balance_max_abs_percent_discrepancy": max_abs_percent_discrepancy,
        "mass_balance_max_abs_in_minus_out": max_abs_in_minus_out,
        "mass_balance_worst_period": mass_balance_worst.get("period"),
        "mass_balance_worst_period_percent_discrepancy": mass_balance_worst.get("percent_discrepancy"),
    }


def variant_candidate_key(item: tuple[float, float, str, dict]) -> tuple[float, float]:
    """:param item: variant candidate tuple. :return: sort key."""
    return (float(item[0]), float(item[1]))


def best_variant_summary(variant_results: dict | None) -> dict | None:
    """:param variant_results: optional variant results. :return: best variant summary."""
    if not variant_results:
        return None
    candidates: list[tuple[float, float, str, dict]] = []
    for name, result in variant_results.items():
        if not isinstance(result, dict) or result.get("skipped"):
            continue
        final = result.get("final", {})
        max_abs = finite_scalar(final.get("max_abs_diff"))
        rmse = finite_scalar(final.get("rmse"))
        if max_abs is None or rmse is None:
            continue
        candidates.append((rmse, max_abs, str(name), result))
    if not candidates:
        return None
    rmse, max_abs, name, result = min(candidates, key=variant_candidate_key)
    out = {
        "best_variant": name,
        "best_variant_final_max_abs_diff": float(max_abs),
        "best_variant_final_rmse": float(rmse),
    }
    out.update(variant_worst_period_metrics(result=result))
    out["best_variant_passes_practical_target"] = bool(rmse < 0.01 and max_abs < 0.05)
    return out


def diagnose(
    default_final: dict,
    pattern: dict,
    worst_cells: list[dict],
    variant_results: dict | None,
    *,
    winning_variant_name: str = WINNING_VARIANT_NAME,
    accepted_picard_dh_threshold: float = 1.0e-2,
) -> dict:
    """Produce independent head-convergence and storage-budget diagnosis flags.

    Final-period accuracy, all-period accuracy, nonlinear active-set
    convergence, and storage-budget availability are judged separately.

    :param default_final: direct-replay final error metrics.
    :param pattern: error pattern summary.
    :param worst_cells: worst-cell rows (kept for backward-compatible notes).
    :param variant_results: optional variant results from --run-replays.
    :param winning_variant_name: variant used for the detailed criteria.
    :param accepted_picard_dh_threshold: practical Picard head-change closure
        used as the relaxation on ``final dh_max <= hclose``.
    :return: diagnosis dict with the per-criterion flags and labels.
    """
    controls = default_solve_controls()
    hclose = float(controls.get("hclose", 1.0e-4))
    max_outer_cap = int(controls.get("max_outer_iterations", 100))

    variant_result = None
    if variant_results and winning_variant_name in variant_results:
        candidate = variant_results[winning_variant_name]
        if isinstance(candidate, dict) and not candidate.get("skipped"):
            variant_result = candidate

    storage_src = (variant_result or {}).get("storage_budget") or {}

    # ---- Final-period practical target (final rmse < 0.01 m, max_abs < 0.05 m).
    final_src = (variant_result or {}).get("final") or default_final
    final_max_abs = finite_scalar(final_src.get("max_abs_diff"))
    final_rmse = finite_scalar(final_src.get("rmse"))
    if final_max_abs is None:
        final_max_abs = finite_scalar(default_final.get("max_abs_diff"))
    if final_rmse is None:
        final_rmse = finite_scalar(default_final.get("rmse"))
    final_period_practical_target_passed = bool(
        final_max_abs is not None
        and final_rmse is not None
        and final_max_abs < 0.05
        and final_rmse < 0.01
    )

    # ---- All-period practical target (every period rmse < 0.01 m, max_abs < 0.05 m).
    period_rows = (variant_result or {}).get("period_error") or []
    if period_rows:
        all_period_practical_target_passed = bool(
            all(float(r["rmse"]) < 0.01 and float(r["max_abs_diff"]) < 0.05 for r in period_rows)
        )
    else:
        all_period_practical_target_passed = None

    # ---- Nonlinear active-set convergence (all periods converged, none hits the
    # outer cap, final-period Picard dh closed to hclose or the accepted threshold).
    period_conv = (variant_result or {}).get("period_convergence") or {}
    periods = period_conv.get("periods") or []
    if periods:
        all_converged = all(bool(p.get("converged", False)) for p in periods if isinstance(p, dict))
        none_hit_max = all(
            int(p.get("outer_iterations", 0) or 0) < int(p.get("picard_max_iter", max_outer_cap) or max_outer_cap)
            for p in periods
            if isinstance(p, dict)
        )
        final_period = periods[-1] if isinstance(periods[-1], dict) else {}
        final_dh = finite_scalar(
            final_period.get("final_max_abs_head_change")
            or final_period.get("picard_dh_max_end")
        )
        dh_ok = (final_dh is None) or (final_dh <= max(hclose, accepted_picard_dh_threshold))
        nonlinear_convergence_passed = bool(all_converged and none_hit_max and dh_ok)
    else:
        nonlinear_convergence_passed = None

    storage_budget_diagnostics_available = bool(storage_src.get("available", False))
    storage_budget_practical_target_passed = storage_src.get("practical_target_passed")
    if not storage_budget_diagnostics_available:
        storage_budget_practical_target_passed = None

    def _label(flag: bool | None, pass_text: str, fail_text: str) -> str:
        if flag is None:
            return "UNKNOWN: insufficient diagnostics"
        return pass_text if flag else fail_text

    final_label = _label(
        final_period_practical_target_passed,
        "PASS: final-period MF6 practical target met",
        "FAIL: final-period MF6 practical target not met",
    )
    all_label = _label(
        all_period_practical_target_passed,
        "PASS: all-period practical target met",
        "FAIL: all-period practical target not met",
    )
    nonlinear_label = _label(
        nonlinear_convergence_passed,
        "PASS: current-Picard active-set convergence resolved",
        "FAIL: current-Picard active-set convergence not resolved",
    )
    storage_budget_label = (
        "UNKNOWN: storage/water-budget target not evaluated"
        if not storage_budget_diagnostics_available or storage_budget_practical_target_passed is None
        else _label(
            bool(storage_budget_practical_target_passed),
            "PASS: storage/water-budget target met",
            "FAIL: storage/water-budget target not met",
        )
    )

    flags = [
        final_period_practical_target_passed,
        all_period_practical_target_passed,
        nonlinear_convergence_passed,
    ]
    if all(flag is True for flag in flags):
        overall_label = (
            f"PASS: {winning_variant_name} matches MF6 heads within final-period and all-period practical targets"
        )
    elif any(flag is None for flag in flags):
        overall_label = "UNKNOWN: insufficient head/convergence diagnostics (run with --run-replays)"
    else:
        overall_label = "FAIL: one or more of final-period / all-period / nonlinear-convergence not met"

    if all(flag is True for flag in flags) and not storage_budget_diagnostics_available:
        recommended_next_fix = (
            "Storage/water-budget diagnostics remain the next check. Head targets already pass with "
            "mf6_convertible_secant_sy using storage_reference=current_picard."
        )
    elif all(flag is True for flag in flags):
        recommended_next_fix = "No head-formulation fix required. Keep secant-Sy as the MF6 replay reference."
    else:
        recommended_next_fix = (
            "Head/convergence targets still fail in at least one criterion. Compare the direct replay settings "
            "against the secant-Sy winning variant before changing formulation."
        )

    best = best_variant_summary(variant_results=variant_results)
    out = {
        "label": overall_label,
        "labels": [final_label, all_label, nonlinear_label, storage_budget_label],
        "final_period_practical_target": "final rmse < 0.01 m and final max_abs_diff < 0.05 m",
        "final_period_practical_target_passed": final_period_practical_target_passed,
        "all_period_practical_target": "every period rmse < 0.01 m and every period max_abs_diff < 0.05 m",
        "all_period_practical_target_passed": all_period_practical_target_passed,
        "nonlinear_convergence": (
            "all periods converged, no period hits max_outer_iterations, "
            "final dh_max <= hclose or accepted project threshold"
        ),
        "nonlinear_convergence_passed": nonlinear_convergence_passed,
        "storage_budget_diagnostics_available": storage_budget_diagnostics_available,
        "storage_budget_practical_target_passed": storage_budget_practical_target_passed,
        "winning_variant": winning_variant_name,
        "final_max_abs_diff": final_max_abs,
        "final_rmse": final_rmse,
        "accepted_picard_dh_threshold": float(accepted_picard_dh_threshold),
        # Backward-compatible aliases.
        "strict_target": "max_abs_diff < 1e-6 m",
        "strict_target_passed": bool(final_max_abs is not None and final_max_abs < 1.0e-6),
        "practical_target": "rmse < 0.01 m and max_abs_diff < 0.05 m",
        "practical_target_passed": bool(final_period_practical_target_passed),
        "recommended_next_fix": recommended_next_fix,
    }
    if best is not None:
        out.update(best)
    return out


def print_period_table(rows: list[dict]) -> None:
    """:param rows: period metric rows."""
    print("\nPeriod error table")
    print("period  max_abs      rmse       mean       median     p95_abs    p99_abs   worst(i,j)")
    for row in rows:
        print(
            f"{row['period']:>6d}  {row['max_abs_diff']:>9.6f}  {row['rmse']:>9.6f}  "
            f"{row['mean_diff']:>9.6f}  {row['median_diff']:>9.6f}  "
            f"{row['p95_abs_diff']:>9.6f}  {row['p99_abs_diff']:>9.6f}  "
            f"({row['max_diff_i']},{row['max_diff_j']})"
        )


def print_worst_cells(rows: list[dict]) -> None:
    """:param rows: worst-cell rows."""
    print("\nWorst cells")
    print("rank   i    j      warp        mf6       diff    abs_diff  chd_dist inactive_dist  class")
    for row in rows:
        print(
            f"{row['rank']:>4d} {row['i']:>4d} {row['j']:>4d} "
            f"{row['warp_head']:>10.4f} {row['mf6_head']:>10.4f} {row['diff']:>10.4f} "
            f"{row['abs_diff']:>9.4f} {str(row['distance_or_flag_near_chd']):>9s} "
            f"{str(row['distance_or_flag_near_inactive']):>13s}  {row['classification']}"
        )


def print_picard_table(rows: list[dict]) -> None:
    """:param rows: Picard diagnostic rows."""
    if not rows:
        print("\nPicard diagnostics: not_available")
        return
    print("\nPicard diagnostics for worst period")
    print("period outer omega  dh_max      dh_rms      inner  inner_it  inner_res   sat_min    sat_max")
    for row in rows:
        print(
            f"{int(row['period']):>6d} {int(row['outer_iter']):>5d} "
            f"{float(row['omega']):>5.2f} {float(row['max_abs_head_change']):>10.4g} "
            f"{float(row['rms_head_change']):>10.4g} {str(row['inner_converged']):>6s} "
            f"{int(row['inner_iterations']):>8d} {float(row['inner_residual']):>10.4g} "
            f"{float(row['sat_thickness_min']):>9.4f} {float(row['sat_thickness_max']):>9.4f}"
        )


def print_storage_budget_table(storage_budget: dict) -> None:
    """:param storage_budget: output of :func:`compare_storage_budgets`."""
    if not storage_budget or not storage_budget.get("available", False):
        print("\nStorage budget diagnostics: not_available")
        return
    print("\nStorage budget table")
    print("period  rmse         max_abs      mean_bias    p95_abs      p99_abs    n_cells")
    for row in storage_budget.get("rows", []):
        n_compared = row.get("n_compared_cells", row.get("n_cells", 0))
        print(
            f"{int(row['period']):>6d}  {float(row['storage_rmse']):>11.6g}  "
            f"{float(row['storage_max_abs']):>11.6g}  {float(row['storage_mean_bias']):>11.6g}  "
            f"{float(row['storage_p95_abs']):>11.6g}  {float(row['storage_p99_abs']):>11.6g}  "
            f"{int(n_compared):>7d}"
        )
    print(f"  storage_sign_used={storage_budget.get('storage_sign_used', NOT_AVAILABLE)}")


def print_storage_worst_cells(rows: list[dict]) -> None:
    """:param rows: worst storage-budget cells."""
    if not rows:
        print("\nWorst storage-budget cells: not_available")
        return
    print("\nWorst storage-budget cells")
    print("rank  per    i    j    mf6_storage  warp_storage   sy_term      ss_term      diff        class")
    for row in rows:
        print(
            f"{int(row['rank']):>4d} {int(row['period']):>4d} {int(row['i']):>4d} {int(row['j']):>4d} "
            f"{float(row['mf6_storage']):>12.5g} {float(row['warp_storage']):>12.5g} "
            f"{float(row['warp_sy_storage']):>11.5g} {float(row['warp_ss_storage']):>11.5g} "
            f"{float(row['storage_diff']):>11.5g}  {row['crossing_class']}"
        )


def print_mass_balance_table(mass_balance: dict) -> None:
    """:param mass_balance: replay mass-balance summary."""
    if not mass_balance or not mass_balance.get("warp_mass_balance_available", False):
        print("\nWarp mass balance: not_available")
        return
    print("\nWarp mass balance table")
    print("period  recharge_in  recharge_out  chd_in      chd_out     storage_in  storage_out total_in    total_out   discrepancy_pct")
    for row in mass_balance.get("per_period", []):
        print(
            f"{int(row['period']):>6d}  {float(row['recharge_in']):>11.5g}  {float(row['recharge_out']):>12.5g}  "
            f"{float(row['chd_in']):>10.5g}  {float(row['chd_out']):>10.5g}  "
            f"{float(row['storage_in']):>10.5g}  {float(row['storage_out']):>11.5g}  "
            f"{float(row['total_in']):>10.5g}  {float(row['total_out']):>10.5g}  "
            f"{float(row['percent_discrepancy']):>15.6g}"
        )
    cumulative = mass_balance.get("cumulative", {})
    if cumulative:
        print("  cumulative_percent_discrepancy=" + str(cumulative.get("percent_discrepancy", NOT_AVAILABLE)))


def print_mf6_mass_balance_comparison(summary: dict) -> None:
    """:param summary: MF6-vs-Warp mass-balance comparison summary."""
    if not summary or not summary.get("available", False):
        reason = summary.get("reason", NOT_AVAILABLE) if isinstance(summary, dict) else NOT_AVAILABLE
        print(f"\nMF6 mass-balance comparison: not_available ({reason})")
        return
    print("\nMF6 mass-balance comparison")
    print("period  warp_in     warp_out    warp_pct   mf6_in      mf6_out     mf6_pct    warp-mf6_storage  warp-mf6_total")
    for row in summary.get("rows", []):
        print(
            f"{int(row['period']):>6d}  {float(row['warp_total_in']):>10.5g}  {float(row['warp_total_out']):>10.5g}  "
            f"{float(row['warp_discrepancy_pct']):>9.5g}  {float(row['mf6_total_in']):>10.5g}  {float(row['mf6_total_out']):>10.5g}  "
            f"{float(row['mf6_discrepancy_pct']):>9.5g}  {float(row['warp_minus_mf6_storage']):>16.5g}  "
            f"{float(row['warp_minus_mf6_total_budget']):>14.5g}"
        )


def print_summary_table(name: str, final_metrics: dict, worst_period: int | None) -> None:
    """:param name: replay name. :param final_metrics: final metrics. :param worst_period: worst period."""
    print(
        f"{name:<18s} max_abs={float(final_metrics.get('max_abs_diff', math.nan)):.6g} "
        f"rmse={float(final_metrics.get('rmse', math.nan)):.6g} "
        f"bias={float(final_metrics.get('mean_bias_warp_minus_mf6', math.nan)):.6g} "
        f"worst_period={(NOT_AVAILABLE if worst_period is None else int(worst_period))}"
    )


def _fmt_metric(value: Any) -> str:
    """:param value: metric value. :return: compact string (None -> n/a)."""
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6g}" if math.isfinite(value) else "inf"
    if isinstance(value, list):
        return ",".join(str(int(v)) for v in value) if value else "none"
    return str(value)


def print_variant_full_metrics(name: str, metrics: dict) -> None:
    """:param name: variant name. :param metrics: output of :func:`variant_full_metrics`."""
    print(f"\nFull metrics: {name}")
    for key in (
        "final_max_abs_diff", "final_rmse", "final_bias",
        "final_percent_within_0_01m", "final_percent_within_0_1m",
        "final_period_practical_target_passed", "all_period_practical_target_passed",
        "worst_period_number", "worst_period_index_zero_based", "worst_period_max_abs_diff", "worst_period_rmse",
        "final_storage_rmse", "final_storage_max_abs",
        "worst_period_storage_rmse", "worst_period_storage_max_abs",
        "period_of_max_rmse", "period_of_max_abs",
        "converged", "nonconverged_periods", "total_outer_iterations", "max_outer_iterations_per_period",
        "period_1_outer_iterations", "runtime", "period_1_runtime", "mean_period_runtime", "max_period_runtime",
        "last_top_switch_changed_fraction", "max_top_switch_changed_fraction",
        "top_switch_frozen", "top_switch_frozen_outer_iteration",
        "storage_budget_diagnostics_available", "storage_sign_used",
        "mass_balance_max_abs_percent_discrepancy", "mass_balance_max_abs_in_minus_out",
        "mass_balance_worst_period", "mass_balance_worst_period_percent_discrepancy",
    ):
        print(f"  {key:<34s} = {_fmt_metric(metrics.get(key))}")


FREEZE_TABLE_COLUMNS = (
    "variant", "storage_freeze_after_outer", "top_switch_frozen",
    "top_switch_frozen_outer_iteration", "converged", "nonconverged_periods",
    "total_outer_iterations", "max_outer_iterations_per_period", "runtime",
    "mean_period_runtime", "max_period_runtime",
    "final_max_abs_diff", "final_rmse", "final_bias",
    "final_storage_rmse", "final_storage_max_abs",
    "worst_period", "worst_period_max_abs_diff", "worst_period_rmse",
    "worst_period_storage_rmse", "worst_period_storage_max_abs",
    "last_top_switch_changed_fraction", "max_top_switch_changed_fraction",
)


def print_freeze_metrics_table(rows: list[dict]) -> None:
    """:param rows: per secant-Sy freeze-after-outer variant metric dicts."""
    if not rows:
        print("\nSecant-Sy freeze sweep: none run")
        return
    print("\nSecant-Sy freeze sweep")
    header = "  ".join(col for col in FREEZE_TABLE_COLUMNS)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = [_fmt_metric(row.get(col)) for col in FREEZE_TABLE_COLUMNS]
        print("  ".join(cell.rjust(len(col)) for cell, col in zip(cells, FREEZE_TABLE_COLUMNS)))
    print("-" * len(header))


SPEED_TABLE_COLUMNS = (
    "variant_name", "runtime", "period_1_runtime", "total_outer_iterations", "period_1_outer_iterations",
    "strict_picard_convergence_passed", "practical_picard_acceptance_passed", "production_acceptance_passed",
    "final_rmse", "final_max_abs_diff", "worst_period_rmse", "worst_period_max_abs_diff",
    "all_period_practical_target_passed", "mass_balance_max_abs_in_minus_out", "mass_balance_max_abs_percent_discrepancy",
)


def print_speed_sweep_table(rows: list[dict]) -> None:
    """:param rows: secant-Sy speed sweep metric rows."""
    if not rows:
        return
    print("\nSecant-Sy speed sweep")
    header = "  ".join(col for col in SPEED_TABLE_COLUMNS)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for row in rows:
        cells = [_fmt_metric(row.get(col)) for col in SPEED_TABLE_COLUMNS]
        print("  ".join(cell.rjust(len(col)) for cell, col in zip(cells, SPEED_TABLE_COLUMNS)))
    print("-" * len(header))


def parse_args(argv: list[str]) -> argparse.Namespace:
    """:param argv: command-line arguments. :return: parsed args."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--artifact", type=Path, default=default_artifact_path(FORMULATION_UNCONFINED))
    parser.add_argument(
        "--warp-npz",
        type=Path,
        default=data_store.joinpath("working_tests", "mf6_transient_2d_unconfined", "warp_transient_heads.npz"),
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=data_store.joinpath("working_tests", "mf6_transient_2d_unconfined", "transient_unconfined_replay_analysis.json"),
    )
    parser.add_argument("--worst-period", type=int, default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--run-replays", action="store_true")
    parser.add_argument("--variant-set", choices=("full", "speed"), default="full")
    parser.add_argument("--variant-workspace", type=Path, default=None)
    parser.add_argument("--variant-name", default="selected_variant")
    parser.add_argument("--ignore-existing-variant-results", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-warm-start-mismatch", action="store_true", default=True)
    parser.add_argument("--mass-balance", action="store_true", default=True)
    parser.add_argument("--save-budget-arrays", action="store_true", default=False)
    parser.add_argument("--compare-mf6-budget", choices=("auto", "true", "false"), default="auto")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """:param argv: optional command-line arguments. :return: process status."""
    args = parse_args(sys.argv[1:] if argv is None else argv)
    artifact = load_transient_artifact(args.artifact)
    warp_npz_path = (
        args.variant_workspace.joinpath("warp_transient_heads.npz")
        if args.variant_workspace is not None
        else args.warp_npz
    )
    warp = load_warp_npz(warp_npz_path)

    period_rows, pattern = period_error_table(artifact=artifact, warp=warp)
    worst_period = int(args.worst_period or pattern["worst_period"])
    worst_period_index_zero_based = int(worst_period - 1)
    worst_cells = worst_cells_table(
        artifact=artifact,
        warp=warp,
        period=worst_period,
        limit=int(args.top_n),
    )
    picard_rows = picard_timing_table(
        artifact=artifact,
        warp=warp,
        period=worst_period,
    )
    storage_budget = compare_storage_budgets(
        artifact=artifact,
        artifact_path=args.artifact,
        workspace=warp_npz_path.parent,
    )
    replay_summary = load_workspace_replay_summary(workspace=warp_npz_path.parent)
    mass_balance = replay_summary.get("mass_balance", {}) if isinstance(replay_summary, dict) else {}
    mf6_mass_balance_comparison = compare_mass_balance(
        artifact_path=args.artifact,
        workspace=warp_npz_path.parent,
    )
    default_final = period_rows[-1]
    final_metrics = {
        "max_abs_diff": default_final["max_abs_diff"],
        "rmse": default_final["rmse"],
        "mean_bias_warp_minus_mf6": default_final["mean_diff"],
        "percent_within_0_01m": default_final["percent_within_0_01m"],
        "percent_within_0_1m": default_final["percent_within_0_1m"],
    }

    variant_results = None
    if args.run_replays:
        variant_results = run_optional_replay_variants(
            artifact_path=args.artifact,
            output_dir=args.output_json.parent,
            device=str(args.device),
            allow_warm_start_mismatch=bool(args.allow_warm_start_mismatch),
            variant_set=str(args.variant_set),
        )
    elif args.variant_workspace is not None:
        variant_results = {
            str(args.variant_name): {
                "variant_name": str(args.variant_name),
                "workspace": str(args.variant_workspace),
                "settings": {
                    "unconfined_storage_mode": scalar_string(warp.get("unconfined_storage_mode", NOT_AVAILABLE)),
                    "storage_reference": scalar_string(warp.get("storage_reference", NOT_AVAILABLE)),
                    "storage_top_threshold": scalar_string(warp.get("storage_top_threshold", STORAGE_TOP_THRESHOLD_GE)),
                },
                "final": final_metrics,
                "worst_period": int(worst_period),
                "worst_period_index_zero_based": int(worst_period - 1),
                "period_error": period_rows,
                "error_pattern": pattern,
                "worst_cells": worst_cells,
                "picard_diagnostics": picard_rows,
                "storage_budget": storage_budget,
                "mass_balance": mass_balance,
                "mf6_mass_balance_comparison": mf6_mass_balance_comparison,
            }
        }
    elif args.output_json.exists() and not bool(args.ignore_existing_variant_results):
        try:
            existing_summary = json.loads(args.output_json.read_text())
        except (OSError, json.JSONDecodeError):
            existing_summary = {}
        existing_variants = existing_summary.get("variant_results")
        if isinstance(existing_variants, dict):
            normal_variant_names = set(_replay_variant_configs())
            variant_results = {
                name: result
                for name, result in existing_variants.items()
                if name in normal_variant_names
            }

    semantics = mf6_semantics_summary(artifact=artifact)
    consistency = check_direct_vs_winning_variant()
    diagnosis = diagnose(
        default_final=final_metrics,
        pattern=pattern,
        worst_cells=worst_cells,
        variant_results=variant_results,
    )

    # Full metrics for the production secant-Sy baseline and its freeze sweep.
    winning_full_metrics = None
    secant_freeze_metrics: list[dict] = []
    speed_sweep_metrics: list[dict] = []
    if variant_results:
        winning_result = variant_results.get(WINNING_VARIANT_NAME)
        if isinstance(winning_result, dict) and not winning_result.get("skipped"):
            winning_full_metrics = variant_full_metrics(winning_result)
        for name, result in variant_results.items():
            if not isinstance(result, dict) or result.get("skipped"):
                continue
            full_result = variant_full_metrics(result)
            if name.startswith("secant_sy_freeze_after_outer_"):
                settings = result.get("settings") or {}
                convergence = result.get("convergence") or {}
                full_result["variant"] = name
                full_result["storage_freeze_after_outer"] = settings.get("storage_freeze_after_outer")
                full_result["top_switch_frozen"] = convergence.get("top_switch_frozen")
                full_result["top_switch_frozen_outer_iteration"] = convergence.get(
                    "top_switch_frozen_outer_iteration"
                )
                full_result["converged"] = convergence.get("converged")
                full_result["runtime"] = result.get("runtime")
                secant_freeze_metrics.append(full_result)
            if str(name).startswith("mf6_secant_sy_speed_outer_"):
                speed_sweep_metrics.append(full_result)

    summary = {
        "artifact_path": str(args.artifact),
        "warp_npz_path": str(warp_npz_path),
        "selected_variant_name": str(args.variant_name) if args.variant_workspace is not None else None,
        "period_error": period_rows,
        "error_pattern": pattern,
        "worst_period": worst_period,
        "worst_period_number_one_based": worst_period,
        "worst_period_index_zero_based": worst_period_index_zero_based,
        "worst_cells": worst_cells,
        "picard_diagnostics": picard_rows,
        "storage_budget": storage_budget,
        "mass_balance": mass_balance,
        "mf6_mass_balance_comparison": mf6_mass_balance_comparison,
        "mf6_semantics": semantics,
        "direct_vs_winning_variant": consistency,
        "winning_variant_full_metrics": winning_full_metrics,
        "secant_sy_freeze_metrics": secant_freeze_metrics,
        "speed_sweep_metrics": speed_sweep_metrics,
        "default_replay": {
            "final": final_metrics,
            "runtime": finite_scalar(warp.get("total_time")),
            "mean_period_runtime": finite_scalar(np.mean(np.asarray(warp.get("period_times", []), dtype=np.float64))) if "period_times" in warp else None,
            "max_period_runtime": finite_scalar(np.max(np.asarray(warp.get("period_times", []), dtype=np.float64))) if "period_times" in warp else None,
            "period_1_runtime": finite_scalar(np.asarray(warp.get("period_times", []), dtype=np.float64)[0]) if "period_times" in warp and np.asarray(warp.get("period_times", []), dtype=np.float64).size > 0 else None,
            "warm_start_mode": scalar_string(warp.get("warm_start_mode", NOT_AVAILABLE)),
            "warm_start_used": scalar_string(warp.get("warm_start_used", NOT_AVAILABLE)),
            "unconfined_storage_mode": scalar_string(warp.get("unconfined_storage_mode", NOT_AVAILABLE)),
            "storage_reference": scalar_string(warp.get("storage_reference", NOT_AVAILABLE)),
            "storage_top_threshold": scalar_string(warp.get("storage_top_threshold", STORAGE_TOP_THRESHOLD_GE)),
            "storage_active_set_strategy": scalar_string(
                warp.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE)
            ),
        },
        "variant_results": variant_results,
        "diagnosis": diagnosis,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, default=str))

    print_period_table(period_rows)
    print_worst_cells(worst_cells)
    print_picard_table(picard_rows)
    print_storage_budget_table(storage_budget)
    print_storage_worst_cells(storage_budget.get("worst_cells", []) if isinstance(storage_budget, dict) else [])
    print_mass_balance_table(mass_balance)
    print_mf6_mass_balance_comparison(mf6_mass_balance_comparison)
    print("\nProduction secant-Sy replay summary")
    print_summary_table("default", final_metrics, worst_period)
    if variant_results:
        for name, result in variant_results.items():
            print_summary_table(name, result.get("final", {}), result.get("worst_period"))
        best = best_variant_summary(variant_results=variant_results)
        if best is not None:
            best_name = best["best_variant"]
            best_result = variant_results.get(best_name, {})
            print(f"\nWinning variant: {best_name}")
            if isinstance(best_result, dict):
                if isinstance(best_result.get("period_error"), list):
                    print_period_table(best_result["period_error"])
                if isinstance(best_result.get("worst_cells"), list):
                    print_worst_cells(best_result["worst_cells"])
                if isinstance(best_result.get("picard_diagnostics"), list):
                    print_picard_table(best_result["picard_diagnostics"])
                if isinstance(best_result.get("storage_budget"), dict):
                    print_storage_budget_table(best_result["storage_budget"])
                    print_storage_worst_cells(best_result["storage_budget"].get("worst_cells", []))
                if isinstance(best_result.get("mass_balance"), dict):
                    print_mass_balance_table(best_result["mass_balance"])
                if isinstance(best_result.get("mf6_mass_balance_comparison"), dict):
                    print_mf6_mass_balance_comparison(best_result["mf6_mass_balance_comparison"])
    print("\nMF6 semantics")
    print(json.dumps(semantics, indent=2, default=str))
    if winning_full_metrics is not None:
        print_variant_full_metrics(WINNING_VARIANT_NAME, winning_full_metrics)
    print_freeze_metrics_table(secant_freeze_metrics)
    print_speed_sweep_table(speed_sweep_metrics)
    print("\nDiagnosis")
    print(json.dumps(diagnosis, indent=2, default=str))
    print(f"\nAnalysis JSON: {args.output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
