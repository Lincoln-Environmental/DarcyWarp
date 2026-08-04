#!/usr/bin/env python
"""Resumable performance and accuracy matrix for production 2D transients.

The matrix deliberately stores only summaries.  MF6 workspaces, Warp head
arrays, and profiler output stay in the user-selected scratch workspace.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.sanity_case_config import (
    TRANSIENT_CAPACITY_LABELS,
    TRANSIENT_PRODUCTION_LABELS,
    TRANSIENT_SCALE_LABELS,
    TRANSIENT_SHAPE_LABELS,
    TRANSIENT_SMOKE_LABELS,
    SPATIAL_GRID_CASES,
)
from working_tests.run_2d_transient_warp_replay import (
    build_case_setup,
    ensure_case_artifact,
    run_production_replay,
)
from working_tests.transient_artifacts import load_transient_artifact
from working_tests.transient_replay_settings import production_secant_sy_settings


MATRIX_SCHEMA_VERSION = 3
HEAD_PARITY_TOLERANCES = {
    "face_fp64_vs_classic": 1.0e-6,
    "face_graph_vs_face_eager": 1.0e-9,
    "mixed_vs_face_graph": 1.0e-4,
}
PERFORMANCE_REGRESSION_FACTOR = 1.10

VARIANT_CONTROLS = {
    "classic_device_fp64": {
        "transient_face_operator_enabled": False,
        "transient_face_graphs_enabled": False,
        "transient_mixed_precision_enabled": False,
    },
    "face_eager_fp64": {
        "transient_face_operator_enabled": True,
        "transient_face_graphs_enabled": False,
        "transient_mixed_precision_enabled": False,
    },
    "face_graph_fp64": {
        "transient_face_operator_enabled": True,
        "transient_face_graphs_enabled": True,
        "transient_mixed_precision_enabled": False,
    },
    "face_graph_mixed": {
        "transient_face_operator_enabled": True,
        "transient_face_graphs_enabled": True,
        "transient_mixed_precision_enabled": True,
    },
}


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _warp_version() -> str:
    import warp as wp

    return str(getattr(wp, "__version__", getattr(wp.config, "version", "unknown")))


def atomic_write_json(path: Path, value: dict) -> None:
    """Atomically persist resumable matrix state."""
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging-{os.getpid()}")
    staging.write_text(json.dumps(value, indent=2, sort_keys=True, default=str))
    os.replace(staging, path)


def _selected_labels(tier: str) -> tuple[str, ...]:
    groups = {
        "smoke": TRANSIENT_SMOKE_LABELS,
        "shape": TRANSIENT_SHAPE_LABELS,
        "production": TRANSIENT_PRODUCTION_LABELS,
        "scale": TRANSIENT_SCALE_LABELS,
        "capacity": TRANSIENT_CAPACITY_LABELS,
    }
    if tier == "all":
        # "all" means every automatically runnable tier; manual-only capacity
        # cases still require --tier capacity or an explicit --cases selection.
        return tuple(dict.fromkeys(
            label
            for group in groups.values()
            for label in group
            if not SPATIAL_GRID_CASES[label]["manual_only"]
        ))
    return tuple(groups[tier])


def _period_rows(summary: dict) -> list[dict]:
    convergence = summary.get("period_convergence", {})
    if isinstance(convergence, dict):
        rows = convergence.get("periods", [])
    else:
        rows = convergence
    return [row for row in rows if isinstance(row, dict)]


def _mempool_snapshot(*, device: str) -> dict:
    """Return allocator counters when the selected device exposes them."""
    if not str(device).startswith("cuda"):
        return {}
    try:
        import warp as wp

        return {
            "used_current_bytes": int(wp.get_mempool_used_mem_current(device)),
            "used_high_water_bytes": int(wp.get_mempool_used_mem_high(device)),
        }
    except (AttributeError, RuntimeError, TypeError):
        return {}


def _matrix_row_identity(
    *,
    label: str,
    variant: str,
    repeats: int,
    physical_fingerprint: str,
    solver_control_fingerprint: str,
    precision_mode: str,
    commit: str,
    device: str,
    warp_version: str,
) -> dict:
    return {
        "schema_version": MATRIX_SCHEMA_VERSION,
        "case_label": label,
        "variant": variant,
        # repeats is part of the identity: rerunning with a different repeat
        # count must not silently reuse a row measured with fewer repeats.
        "repeats": int(repeats),
        "physical_case_fingerprint": physical_fingerprint,
        "solver_control_fingerprint": solver_control_fingerprint,
        "precision_mode": precision_mode,
        "git_commit": commit,
        "device": device,
        "warp_version": warp_version,
    }


def _row_key(identity: dict) -> str:
    """Make the resumable key include every immutable execution identity."""
    return _json_fingerprint(identity)


def _variant_row(
    *,
    summaries: list[dict],
    label: str,
    variant: str,
    repeats: int,
    physical_fingerprint: str,
    solver_controls: dict,
    commit: str,
    device: str,
    warp_version: str,
    wall_times: list[float],
    period_times_by_repeat: list[np.ndarray],
    mempool: dict,
    head_artifact_paths: list[Path],
) -> dict:
    # Timing/accuracy/mass-balance reporting uses the final timed repeat;
    # correctness and implementation-activity metrics are aggregated across
    # EVERY repeat so a passing row proves every repeat ran correctly.
    summary = summaries[-1]
    timing = summary.get("timing", {}) or {}
    period_matrices = [np.asarray(row, dtype=np.float64) for row in period_times_by_repeat if len(row)]
    period_times = np.vstack(period_matrices) if period_matrices else np.empty((0, 0), dtype=np.float64)
    periods = [row for repeat_summary in summaries for row in _period_rows(repeat_summary)]
    strict = [bool(row.get("strict_picard_convergence_passed", row.get("strict_converged", False))) for row in periods]
    practical = [bool(row.get("practical_picard_acceptance_passed", False)) for row in periods]
    retries = sum(int(row.get("adaptive_dt_retry_count", row.get("adaptive_dt_retries", row.get("retries", 0)))) for row in periods)
    substeps = sum(int(row.get("adaptive_dt_substep_count", row.get("substeps", 1))) for row in periods)
    inner_cycles = sum(int(row.get("total_inner_kcycles", row.get("inner_kcycles", 0))) for row in periods)
    outer_iterations = sum(int(row.get("outer_iterations", 0)) for row in periods)
    kcycle_graphs = sum(int(row.get("transient_face_kcycle_graph_count", 0)) for row in periods)
    refresh_graphs = sum(int(row.get("transient_face_refresh_graph_count", 0)) for row in periods)
    kcycle_fallbacks = sum(int(row.get("transient_face_kcycle_graph_fallback_count", 0)) for row in periods)
    refresh_fallbacks = sum(int(row.get("transient_face_refresh_graph_fallback_count", 0)) for row in periods)
    face_active = sum(1 for row in periods if bool(row.get("transient_face_operator", False)))
    graphs_active = sum(1 for row in periods if bool(row.get("transient_face_graphs", False)))
    mixed_active = sum(1 for row in periods if bool(row.get("transient_mixed_precision", False)))
    if period_times.size:
        period_one_runtime = float(np.median(period_times[:, 0]))
        later_runtime = float(np.median(np.mean(period_times[:, 1:], axis=1))) if period_times.shape[1] > 1 else None
    else:
        period_one_runtime = None
        later_runtime = None
    precision_mode = "mixed_correction" if VARIANT_CONTROLS[variant]["transient_mixed_precision_enabled"] else "fp64"
    control_fingerprint = _json_fingerprint(solver_controls)
    identity = _matrix_row_identity(
        label=label,
        variant=variant,
        repeats=repeats,
        physical_fingerprint=physical_fingerprint,
        solver_control_fingerprint=control_fingerprint,
        precision_mode=precision_mode,
        commit=commit,
        device=device,
        warp_version=warp_version,
    )
    production_acceptance = summary.get("production_acceptance", {}) or {}
    return {
        **identity,
        "row_key": _row_key(identity),
        "solver_controls": solver_controls,
        "timing": {
            "period_one_runtime_median_s": period_one_runtime,
            "mean_later_period_runtime_median_s": later_runtime,
            "total_runtime_median_s": float(np.median(wall_times)),
            "runtime_per_cell_period_s": float(
                np.median(wall_times)
                / max(period_times.shape[1] if period_times.ndim == 2 else 1, 1)
                / max(int(summary["grid"]["nx"]) * int(summary["grid"]["ny"]), 1)
            ),
            "warp_reported_total_s": float(timing.get("warp_total_time", np.nan)),
        },
        "total_outer_iterations": int(outer_iterations),
        "total_inner_kcycles": int(inner_cycles),
        "graph_count": int(kcycle_graphs + refresh_graphs),
        "graph_kcycle_count": int(kcycle_graphs),
        "graph_refresh_count": int(refresh_graphs),
        "graph_fallback_count": int(kcycle_fallbacks + refresh_fallbacks),
        "graph_kcycle_fallback_count": int(kcycle_fallbacks),
        "graph_refresh_fallback_count": int(refresh_fallbacks),
        # Implementation-activity proof (per-period-row flags, all repeats):
        # the matrix gates require these to cover every period row when the
        # variant requests the corresponding implementation.
        "face_operator_active_periods": int(face_active),
        "face_graphs_active_periods": int(graphs_active),
        "mixed_correction_active_periods": int(mixed_active),
        "adaptive_dt_retries": int(retries),
        "adaptive_dt_substeps": int(substeps),
        "practical_accepts": int(sum(practical)),
        "strict_converged_periods": int(sum(strict)),
        "strict_period_count": int(len(periods)),
        "head_accuracy": summary.get("head_accuracy", summary.get("comparison", {})),
        "mass_balance": summary.get("mass_balance", {}),
        "mempool": mempool,
        "head_parity": {},
        "production_acceptance": production_acceptance,
        "head_artifact_path": str(head_artifact_paths[-1]),
        "head_artifact_paths": [str(path) for path in head_artifact_paths],
        "completed": True,
    }


def _load_head_history(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        return np.asarray(payload["heads_per_period"], dtype=np.float64)


def _select_case_variant_rows(rows: dict, *, label: str, variants: tuple[str, ...]) -> dict:
    """Pick the best completed row per variant for a case.

    Rows whose identity differs only in repeat count can coexist in a resumed
    results file; the row measured with the most repeats wins so parity and
    gates never silently mix measurement depths.
    """
    selected: dict[str, dict] = {}
    for row in rows.values():
        if not (
            row.get("case_label") == label
            and row.get("variant") in variants
            and row.get("completed")
        ):
            continue
        previous = selected.get(row["variant"])
        if previous is None or int(row.get("repeats", 0)) >= int(previous.get("repeats", 0)):
            selected[row["variant"]] = row
    return selected


def _apply_case_parity(rows: dict, *, label: str, variants: tuple[str, ...]) -> None:
    selected = _select_case_variant_rows(rows, label=label, variants=variants)
    if not selected:
        return
    # Every timed repeat's heads are loaded; correctness must hold for all of
    # them, not just the final repeat.
    histories = {}
    for variant, row in selected.items():
        paths = row.get("head_artifact_paths") or [row.get("head_artifact_path", "")]
        loaded = [_load_head_history(Path(path)) for path in paths if path and Path(path).exists()]
        if loaded:
            histories[variant] = loaded
    reference_variant = "classic_device_fp64" if "classic_device_fp64" in histories else "face_graph_fp64"
    if reference_variant not in histories:
        for row in selected.values():
            row["head_parity"] = {
                "comparison": "unavailable_missing_head_history",
                "passed": False,
            }
        return
    reference = histories[reference_variant][-1]
    for variant, row in selected.items():
        if variant not in histories:
            row["head_parity"] = {
                "reference_variant": reference_variant,
                "comparison": "unavailable_missing_head_history",
                "passed": False,
            }
            continue
        if variant == "face_graph_fp64" and "face_eager_fp64" in histories:
            comparator = histories["face_eager_fp64"][-1]
            tolerance = HEAD_PARITY_TOLERANCES["face_graph_vs_face_eager"]
            label_name = "face_graph_vs_face_eager"
        elif variant == "face_eager_fp64" and reference_variant == "face_graph_fp64":
            comparator = reference
            tolerance = HEAD_PARITY_TOLERANCES["face_graph_vs_face_eager"]
            label_name = "face_graph_vs_face_eager"
        elif variant == "face_graph_mixed":
            comparator = histories.get("face_graph_fp64", [reference])[-1]
            tolerance = HEAD_PARITY_TOLERANCES["mixed_vs_face_graph"]
            label_name = "mixed_vs_face_graph"
        else:
            comparator = reference
            tolerance = HEAD_PARITY_TOLERANCES["face_fp64_vs_classic"]
            label_name = "reference_self" if variant == reference_variant else "face_fp64_vs_classic"
        differences = [np.abs(history - comparator) for history in histories[variant]]
        max_abs = float(max(float(np.max(diff)) for diff in differences))
        rmse = float(np.sqrt(np.mean(np.concatenate([diff.reshape(-1) for diff in differences]) ** 2)))
        row["head_parity"] = {
            "reference_variant": reference_variant,
            "comparison": label_name,
            "repeats_checked": len(differences),
            "max_abs_head_difference": max_abs,
            "rmse_head_difference": rmse,
            "tolerance_m": tolerance,
            "passed": bool(max_abs <= tolerance),
        }


def _apply_matrix_gates(results: dict) -> None:
    labels = {str(row.get("case_label")) for row in results.get("rows", {}).values()}
    variants = tuple(VARIANT_CONTROLS)
    by_case: dict[str, dict[str, dict]] = {}
    for label in labels:
        by_case[label] = _select_case_variant_rows(results["rows"], label=label, variants=variants)
    for label, case_rows in by_case.items():
        classic = case_rows.get("classic_device_fp64")
        classic_time = (classic or {}).get("timing", {}).get("total_runtime_median_s")
        for variant, row in case_rows.items():
            strict_gate = row.get("strict_converged_periods", 0) == row.get("strict_period_count", -1)
            no_practical_gate = row.get("practical_accepts", 0) == 0
            no_retry_gate = row.get("adaptive_dt_retries", 0) == 0
            acceptance = row.get("production_acceptance", {}) or {}
            performance_gate = True
            if variant != "classic_device_fp64" and classic_time and row.get("timing", {}).get("total_runtime_median_s"):
                performance_gate = row["timing"]["total_runtime_median_s"] <= float(classic_time) * PERFORMANCE_REGRESSION_FACTOR
            parity = row.get("head_parity", {}) or {}
            parity_gate = bool(parity.get("passed", True))
            # Implementation-activity gates: a passing row must prove the
            # requested implementation actually ran (face operator active,
            # CUDA graphs captured without eager fallback, mixed correction
            # active) on every period row of every repeat.
            controls = VARIANT_CONTROLS.get(variant, {})
            period_rows = int(row.get("strict_period_count", 0))
            face_required = bool(controls.get("transient_face_operator_enabled", False))
            graphs_required = bool(controls.get("transient_face_graphs_enabled", False))
            mixed_required = bool(controls.get("transient_mixed_precision_enabled", False))
            face_active_gate = True
            if face_required:
                face_active_gate = period_rows > 0 and row.get("face_operator_active_periods", 0) == period_rows
            graphs_active_gate = True
            if graphs_required:
                graphs_active_gate = (
                    period_rows > 0
                    and row.get("face_graphs_active_periods", 0) == period_rows
                    and row.get("graph_count", 0) > 0
                )
            graphs_no_fallback_gate = True
            if graphs_required:
                graphs_no_fallback_gate = row.get("graph_fallback_count", 0) == 0
            mixed_active_gate = True
            if mixed_required:
                mixed_active_gate = period_rows > 0 and row.get("mixed_correction_active_periods", 0) == period_rows
            row["acceptance_gates"] = {
                "strict_convergence_all_periods": bool(strict_gate),
                "no_practical_acceptance": bool(no_practical_gate),
                "no_adaptive_dt_retries": bool(no_retry_gate),
                "production_reporting_gate": bool(acceptance.get("production_acceptance_passed", False)),
                "head_parity": parity_gate,
                "performance_no_more_than_10pct_from_classic": bool(performance_gate),
                "face_operator_active": bool(face_active_gate),
                "graph_capture_active": bool(graphs_active_gate),
                "graph_no_eager_fallback": bool(graphs_no_fallback_gate),
                "mixed_correction_active": bool(mixed_active_gate),
            }
            row["matrix_passed"] = bool(all(row["acceptance_gates"].values()))


def run_variant(
    *,
    case_setup: dict,
    artifact_path: Path,
    variant: str,
    repeats: int,
    device: str,
    workspace: Path,
    commit: str,
    warp_version: str,
) -> dict:
    controls = dict(VARIANT_CONTROLS[variant])
    # Key the scratch workspace on the case-unique artifact directory name
    # (grid x periods x T-kind/seed x GHB mode); the artifact FILENAME is the
    # same for every case, so deriving it from the path stem collides across
    # grids and overwrites earlier cases' head artifacts.
    variant_workspace = workspace.joinpath(case_setup["artifact_path"].parent.name, variant)
    # Warp's mempool high-water mark is cumulative over the process lifetime
    # and cannot be reset; snapshot before AND after so each row can report
    # the high-water increase attributable to this variant instead of
    # inheriting earlier variants' peaks.
    mempool_before = _mempool_snapshot(device=device)
    # One untimed warm-up compiles kernels and captures the same numerical case.
    run_production_replay(
        artifact_path=artifact_path,
        workspace=variant_workspace.joinpath("warmup"),
        device=device,
        solve_control_overrides=controls,
    )
    timed_summaries: list[dict] = []
    period_times_by_repeat: list[np.ndarray] = []
    wall_times: list[float] = []
    head_artifact_paths: list[Path] = []
    for repeat in range(int(repeats)):
        started = time.perf_counter()
        summary = run_production_replay(
            artifact_path=artifact_path,
            workspace=variant_workspace.joinpath(f"repeat-{repeat + 1}"),
            device=device,
            solve_control_overrides=controls,
        )
        wall_times.append(float(time.perf_counter() - started))
        timed_summaries.append(summary)
        period_times_by_repeat.append(np.asarray(summary.get("timing", {}).get("warp_period_times", []), dtype=np.float64))
        head_artifact_paths.append(variant_workspace.joinpath(f"repeat-{repeat + 1}", "warp_transient_heads.npz"))
    solver_controls = dict(production_secant_sy_settings()["solve_controls"])
    solver_controls.update(controls)
    mempool = _mempool_snapshot(device=device)
    if mempool and mempool_before:
        mempool["baseline_used_current_bytes"] = mempool_before.get("used_current_bytes")
        mempool["baseline_used_high_water_bytes"] = mempool_before.get("used_high_water_bytes")
        mempool["variant_high_water_delta_bytes"] = (
            mempool["used_high_water_bytes"] - int(mempool_before.get("used_high_water_bytes", 0))
        )
        mempool["high_water_is_process_cumulative"] = True
    artifact = load_transient_artifact(artifact_path)
    row = _variant_row(
        summaries=timed_summaries,
        label=str(case_setup["label"]),
        variant=variant,
        repeats=int(repeats),
        physical_fingerprint=str(np.asarray(artifact["case_fingerprint"]).reshape(())),
        solver_controls=solver_controls,
        commit=commit,
        device=device,
        warp_version=warp_version,
        wall_times=wall_times,
        period_times_by_repeat=period_times_by_repeat,
        mempool=mempool,
        head_artifact_paths=head_artifact_paths,
    )
    row["repeat_wall_times_s"] = wall_times
    return row


def run_matrix(*, labels: tuple[str, ...], variants: tuple[str, ...], repeats: int,
               device: str, workspace: Path, output_path: Path,
               ghb_mode: str = "none") -> dict:
    commit = _git_commit()
    warp_version = _warp_version()
    if output_path.exists():
        results = json.loads(output_path.read_text())
        if int(results.get("schema_version", -1)) != MATRIX_SCHEMA_VERSION:
            raise ValueError("matrix output has an incompatible schema; use a new --output path")
        if (
            results.get("commit") != commit
            or results.get("device") != device
            or results.get("warp_version") != warp_version
            or results.get("ghb_mode", "none") != ghb_mode
        ):
            raise ValueError("matrix output identity differs (commit/device/Warp); refusing stale resume")
    else:
        results = {
            "schema_version": MATRIX_SCHEMA_VERSION,
            "commit": commit,
            "device": device,
            "warp_version": warp_version,
            "ghb_mode": ghb_mode,
            "rows": {},
        }
    results.setdefault("rows", {})
    for label in labels:
        geometry = SPATIAL_GRID_CASES[label]
        n_periods = 52 if label == "500x500" else 30 if label == "1000x1000" else 3 if label in TRANSIENT_SMOKE_LABELS else 1
        t_field_kind = "homogeneous" if label == "500x500" else "ugly_t"
        case_setup = build_case_setup(
            nx=int(geometry["nx"]), ny=int(geometry["ny"]), n_periods=n_periods,
            t_field_kind=t_field_kind, t_field_seed=42,
            ghb_conductance_mode=ghb_mode,
        )
        case_setup["label"] = label
        artifact_path = ensure_case_artifact(case_setup)
        artifact = load_transient_artifact(artifact_path)
        physical_fingerprint = str(np.asarray(artifact["case_fingerprint"]).reshape(()))
        for variant in variants:
            solver_controls = dict(production_secant_sy_settings()["solve_controls"])
            solver_controls.update(VARIANT_CONTROLS[variant])
            precision_mode = "mixed_correction" if VARIANT_CONTROLS[variant]["transient_mixed_precision_enabled"] else "fp64"
            identity = _matrix_row_identity(
                label=label,
                variant=variant,
                repeats=repeats,
                physical_fingerprint=physical_fingerprint,
                solver_control_fingerprint=_json_fingerprint(solver_controls),
                precision_mode=precision_mode,
                commit=commit,
                device=device,
                warp_version=warp_version,
            )
            key = _row_key(identity)
            if key in results["rows"]:
                existing = results["rows"][key]
                if not existing.get("completed") or any(existing.get(name) != value for name, value in identity.items()):
                    raise ValueError(f"matrix row {key} is incomplete or identity-mismatched")
                continue
            results["rows"][key] = run_variant(
                case_setup=case_setup, artifact_path=artifact_path, variant=variant,
                repeats=repeats, device=device, workspace=workspace,
                commit=commit, warp_version=warp_version,
            )
            atomic_write_json(output_path, results)
        _apply_case_parity(results["rows"], label=label, variants=variants)
        _apply_matrix_gates(results)
        atomic_write_json(output_path, results)
    _apply_matrix_gates(results)
    atomic_write_json(output_path, results)
    return results


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("smoke", "shape", "production", "scale", "capacity", "all"), default="smoke")
    parser.add_argument("--cases", default="", help="comma-separated catalog labels; overrides --tier")
    parser.add_argument("--variants", default=None, help="comma-separated variant names; default: classic_device_fp64,face_eager_fp64,face_graph_fp64")
    parser.add_argument("--include-mixed", action="store_true")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    # warp_matched is intentionally not offered: transient truth generation
    # rejects it (run_2d_transient_vs_mf6.py); only mf6_fixed_point is valid.
    parser.add_argument("--ghb-mode", choices=("none", "mf6_fixed_point"), default="none")
    parser.add_argument("--workspace", type=Path, default=Path("/tmp/darcywarp-transient-matrix"))
    parser.add_argument("--output", type=Path, default=Path("working_tests/transient_sanity_matrix.json"))
    args = parser.parse_args()
    labels = tuple(item.strip() for item in args.cases.split(",") if item.strip()) if args.cases else _selected_labels(args.tier)
    variants_explicit = args.variants is not None
    variants_arg = args.variants if variants_explicit else "classic_device_fp64,face_eager_fp64,face_graph_fp64"
    variants = tuple(item.strip() for item in variants_arg.split(",") if item.strip())
    if args.include_mixed and "face_graph_mixed" not in variants:
        variants += ("face_graph_mixed",)
    if args.ghb_mode != "none" and "classic_device_fp64" in variants:
        # The classic device path raises for GHB; only face-operator variants
        # can run GHB cases.
        if variants_explicit:
            raise ValueError(
                "variant classic_device_fp64 does not support GHB (the classic "
                "device path raises for GHB); remove it from --variants"
            )
        variants = tuple(variant for variant in variants if variant != "classic_device_fp64")
        print("GHB mode: dropping classic_device_fp64 from the default variants "
              "(the classic device path does not support GHB).")
    if not variants:
        raise ValueError("no runnable variants remain after GHB filtering")
    if args.repeats < 3:
        raise ValueError("--repeats must be at least 3 for performance comparisons.")
    unknown = [label for label in labels if label not in SPATIAL_GRID_CASES]
    if unknown:
        raise ValueError(f"unknown catalog labels: {unknown}")
    unknown_variants = [variant for variant in variants if variant not in VARIANT_CONTROLS]
    if unknown_variants:
        raise ValueError(f"unknown solver variants: {unknown_variants}")
    # manual_only grids are reachable only through an explicit tier
    # (shape/scale/capacity) or --cases; "all" excludes them in
    # _selected_labels, so no further guard is needed here.
    return run_matrix(
        labels=labels, variants=variants, repeats=args.repeats, device=args.device,
        workspace=args.workspace, output_path=args.output, ghb_mode=args.ghb_mode,
    )


if __name__ == "__main__":
    main()
