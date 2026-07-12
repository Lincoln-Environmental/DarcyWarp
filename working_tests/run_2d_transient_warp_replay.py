#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Thin production harness for the 2D transient unconfined MF6 replay."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from working_tests.transient_artifacts import FORMULATION_UNCONFINED  # noqa: E402
from working_tests.transient_replay_settings import (  # noqa: E402
    PRODUCTION_RUN_MODE,
    default_run_config,
    production_secant_sy_settings,
)
from working_tests.transient_replay_support import main as run_transient_replay  # noqa: E402


def build_grid_artifact_path(
    *,
    formulation: str,
    nx: int,
    ny: int,
    n_weeks: int,
) -> Path:
    """
    Build the grid-qualified MF6 truth artifact path used by the generator.
    """
    if int(n_weeks) <= 0:
        raise ValueError("n_weeks must be positive.")
    return data_store.joinpath(
        "working_tests",
        f"mf6_transient_2d_{formulation}_{int(nx)}x{int(ny)}_{int(n_weeks)}w",
        "mf6_transient_heads.npz.lzma",
    )


def run_production_replay(
    *,
    artifact_path: Path | None,
    workspace: Path | None,
    device: str,
    diag_preconditioner_backend: str,
    solve_control_overrides: dict | None,
    allow_warm_start_mismatch: bool,
    profile_performance: bool,
    save_heavy_diagnostics: bool,
    run_replay_matrix: bool,
) -> dict:
    """
    Run the production replay with the selected artifact.
    """
    run_mode = PRODUCTION_RUN_MODE
    compute_mass_balance = True
    solve_controls = dict(solve_control_overrides or {})

    production_settings = production_secant_sy_settings()
    production_solve_controls = dict(production_settings["solve_controls"])
    production_solve_controls.update(solve_controls)
    run_config = default_run_config(
        run_mode=run_mode,
        device=device,
        compute_mass_balance=compute_mass_balance,
        profile_performance=profile_performance,
        save_heavy_diagnostics=save_heavy_diagnostics,
        run_replay_matrix=run_replay_matrix,
    )

    return run_transient_replay(
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
        warm_start_mode=production_settings["warm_start_mode"],
        formulation=FORMULATION_UNCONFINED,
        solve_controls=production_solve_controls,
        unconfined_storage_mode=production_settings["unconfined_storage_mode"],
        storage_reference=production_settings["storage_reference"],
        storage_top_threshold=production_settings["storage_top_threshold"],
        storage_active_set_strategy=production_settings["storage_active_set_strategy"],
        storage_freeze_after_outer=production_settings["storage_freeze_after_outer"],
        allow_warm_start_mismatch=allow_warm_start_mismatch,
        run_config=run_config,
    )


def variant_workspace_name(
    *,
    candidate: dict,
) -> str:
    """
    Build a stable workspace name for one optimization candidate.
    """
    if "name" in candidate and str(candidate["name"]).strip():
        return str(candidate["name"]).strip()
    return (
        f"nu_pre_{int(candidate['nu_pre'])}_post_{int(candidate['nu_post'])}"
        f"_coarse_{int(candidate['nu_coarse'])}_max_levels_{int(candidate['max_levels'])}"
        f"_inner_{int(candidate['inner_early'])}_{int(candidate['inner_middle'])}_{int(candidate['inner_late'])}"
    )


def candidate_solve_controls(
    *,
    base_controls: dict,
    candidate: dict,
) -> dict:
    """
    Apply one optimization candidate to a base solve-control dictionary.
    """
    controls = dict(base_controls)
    controls["nu_pre"] = int(candidate["nu_pre"])
    controls["nu_post"] = int(candidate["nu_post"])
    controls["nu_coarse"] = int(candidate["nu_coarse"])
    controls["max_levels"] = int(candidate["max_levels"])
    controls["unconfined_inner_max_cycles_early"] = int(candidate["inner_early"])
    controls["unconfined_inner_max_cycles_middle"] = int(candidate["inner_middle"])
    controls["unconfined_inner_max_cycles_late"] = int(candidate["inner_late"])
    controls["unconfined_inner_middle_dh"] = float(candidate["inner_middle_dh"])
    controls["unconfined_inner_late_dh"] = float(candidate["inner_late_dh"])
    return controls


def make_optimization_candidate(
    *,
    nu_pre: int,
    nu_post: int,
    nu_coarse: int,
    max_levels: int,
    inner_early: int,
    inner_middle: int,
    inner_late: int,
    inner_middle_dh: float,
    inner_late_dh: float,
    name: str | None = None,
) -> dict:
    """
    Build one explicit optimization candidate.
    """
    return {
        "name": name,
        "nu_pre": int(nu_pre),
        "nu_post": int(nu_post),
        "nu_coarse": int(nu_coarse),
        "max_levels": int(max_levels),
        "inner_early": int(inner_early),
        "inner_middle": int(inner_middle),
        "inner_late": int(inner_late),
        "inner_middle_dh": float(inner_middle_dh),
        "inner_late_dh": float(inner_late_dh),
    }


def build_optimization_candidates(
    *,
    kcycle_settings: list[tuple[int, int, int, int]],
    inner_cycle_settings: list[tuple[int, int, int, float, float]],
) -> list[dict]:
    """
    Build an ordered cross-product of K-cycle and adaptive inner-cycle settings.
    """
    candidates: list[dict] = []
    seen_names: set[str] = set()
    for nu_pre, nu_post, nu_coarse, max_levels in kcycle_settings:
        for inner_early, inner_middle, inner_late, inner_middle_dh, inner_late_dh in inner_cycle_settings:
            name = (
                f"nu_{int(nu_pre)}_{int(nu_post)}_coarse_{int(nu_coarse)}"
                f"_levels_{int(max_levels)}"
                f"_inner_{int(inner_early)}_{int(inner_middle)}_{int(inner_late)}"
                f"_dh_{float(inner_middle_dh):g}_{float(inner_late_dh):g}"
            ).replace(".", "p").replace("-", "m")
            if name in seen_names:
                continue
            seen_names.add(name)
            candidates.append(
                make_optimization_candidate(
                    name=name,
                    nu_pre=nu_pre,
                    nu_post=nu_post,
                    nu_coarse=nu_coarse,
                    max_levels=max_levels,
                    inner_early=inner_early,
                    inner_middle=inner_middle,
                    inner_late=inner_late,
                    inner_middle_dh=inner_middle_dh,
                    inner_late_dh=inner_late_dh,
                )
            )
    return candidates


def replay_summary_is_accepted(
    *,
    summary: dict,
    accepted_mass_balance_classes: set[str],
) -> bool:
    """
    Return True when a replay satisfies the numerical acceptance gates.
    """
    production = summary.get("production_acceptance") or {}
    head_accuracy = summary.get("head_accuracy") or {}
    mass_balance = summary.get("mass_balance") or {}
    mass_balance_class = str(mass_balance.get("mass_balance_class", "")).strip().lower()
    return (
        bool(production.get("production_acceptance_passed", False))
        and bool(head_accuracy.get("passed", False))
        and bool(mass_balance.get("mass_balance_passed", False))
        and mass_balance_class in accepted_mass_balance_classes
    )


def replay_wall_time(summary: dict) -> float:
    """
    Extract the Warp wall time used for optimization ranking.
    """
    timing = summary.get("timing") or {}
    return float(timing.get("warp_total_time", float("inf")) or float("inf"))


def optimize_kcycle_settings(
    *,
    artifact_path: Path | None,
    workspace: Path | None,
    device: str,
    diag_preconditioner_backend: str,
    base_solve_controls: dict,
    candidates: list[dict],
    accepted_mass_balance_classes: set[str],
    stop_after_first_accepted: bool,
    allow_warm_start_mismatch: bool,
    profile_performance: bool,
    save_heavy_diagnostics: bool,
) -> dict:
    """
    Run K-cycle candidates and return the fastest numerically accepted result.
    """
    if artifact_path is None:
        raise ValueError("K-cycle optimization requires an explicit artifact path.")
    artifact_path = Path(artifact_path)
    optimization_workspace = (
        Path(workspace).joinpath("kcycle_optimization")
        if workspace is not None
        else artifact_path.parent.joinpath("kcycle_optimization")
    )
    optimization_workspace.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    for candidate in candidates:
        name = variant_workspace_name(
            candidate=candidate,
        )
        variant_workspace = optimization_workspace.joinpath(name)
        controls = candidate_solve_controls(
            base_controls=base_solve_controls,
            candidate=candidate,
        )
        print(f"\nK-cycle optimization candidate: {name}")
        summary = run_production_replay(
            artifact_path=artifact_path,
            workspace=variant_workspace,
            device=device,
            diag_preconditioner_backend=diag_preconditioner_backend,
            solve_control_overrides=controls,
            allow_warm_start_mismatch=allow_warm_start_mismatch,
            profile_performance=profile_performance,
            save_heavy_diagnostics=save_heavy_diagnostics,
            run_replay_matrix=False,
        )
        accepted = replay_summary_is_accepted(
            summary=summary,
            accepted_mass_balance_classes=accepted_mass_balance_classes,
        )
        result = {
            "variant_name": name,
            "workspace": str(variant_workspace),
            "nu_pre": int(candidate["nu_pre"]),
            "nu_post": int(candidate["nu_post"]),
            "nu_coarse": int(candidate["nu_coarse"]),
            "max_levels": int(candidate["max_levels"]),
            "inner_early": int(candidate["inner_early"]),
            "inner_middle": int(candidate["inner_middle"]),
            "inner_late": int(candidate["inner_late"]),
            "inner_middle_dh": float(candidate["inner_middle_dh"]),
            "inner_late_dh": float(candidate["inner_late_dh"]),
            "accepted": bool(accepted),
            "warp_total_time": replay_wall_time(summary),
            "summary": summary,
        }
        results.append(result)
        if bool(stop_after_first_accepted) and bool(accepted):
            print(f"\nStopping optimization after first accepted candidate: {name}")
            break

    accepted_results = [result for result in results if bool(result["accepted"])]
    if not accepted_results:
        raise RuntimeError("No K-cycle candidate satisfied the numerical acceptance gates.")

    best = accepted_results[0] if bool(stop_after_first_accepted) else min(
        accepted_results,
        key=lambda item: float(item["warp_total_time"]),
    )
    optimization_summary = {
        "stop_after_first_accepted": bool(stop_after_first_accepted),
        "selection_mode": "first_accepted" if bool(stop_after_first_accepted) else "fastest_accepted",
        "best_variant_name": best["variant_name"],
        "best_workspace": best["workspace"],
        "best_warp_total_time": best["warp_total_time"],
        "best_nu_pre": best["nu_pre"],
        "best_nu_post": best["nu_post"],
        "best_nu_coarse": best["nu_coarse"],
        "best_max_levels": best["max_levels"],
        "best_inner_early": best["inner_early"],
        "best_inner_middle": best["inner_middle"],
        "best_inner_late": best["inner_late"],
        "best_inner_middle_dh": best["inner_middle_dh"],
        "best_inner_late_dh": best["inner_late_dh"],
        "optimization_workspace": str(optimization_workspace),
        "candidates": [
            {
                "variant_name": result["variant_name"],
                "workspace": result["workspace"],
                "nu_pre": result["nu_pre"],
                "nu_post": result["nu_post"],
                "nu_coarse": result["nu_coarse"],
                "max_levels": result["max_levels"],
                "inner_early": result["inner_early"],
                "inner_middle": result["inner_middle"],
                "inner_late": result["inner_late"],
                "inner_middle_dh": result["inner_middle_dh"],
                "inner_late_dh": result["inner_late_dh"],
                "accepted": result["accepted"],
                "warp_total_time": result["warp_total_time"],
            }
            for result in results
        ],
    }
    optimization_summary_path = optimization_workspace.joinpath("kcycle_optimization_summary.json")
    optimization_summary_path.write_text(json.dumps(optimization_summary, indent=4), encoding="utf-8")
    if bool(stop_after_first_accepted):
        print("\nFirst accepted K-cycle candidate")
    else:
        print("\nFastest accepted K-cycle candidate")
    print(f"  variant: {best['variant_name']}")
    print(f"  warp_total_time: {best['warp_total_time']:.6g} s")
    print(
        "  settings: "
        f"nu=({best['nu_pre']}, {best['nu_post']}, {best['nu_coarse']}), "
        f"max_levels={best['max_levels']}, "
        f"inner=({best['inner_early']}, {best['inner_middle']}, {best['inner_late']}), "
        f"dh=({best['inner_middle_dh']}, {best['inner_late_dh']})"
    )
    print(f"  workspace: {best['workspace']}")
    print(f"  summary: {optimization_summary_path}")
    return {
        "best": best,
        "results": results,
        "optimization_workspace": str(optimization_workspace),
        "optimization_summary_path": str(optimization_summary_path),
    }


if __name__ == "__main__":
    formulation = FORMULATION_UNCONFINED
    use_grid_qualified_artifact = True
    mf6_nx = 1000
    mf6_ny = 1000
    mf6_n_periods = 10
    explicit_artifact_path = None
    workspace = None
    device = "auto"
    diag_preconditioner_backend = "device"

    max_outer_iterations = 100
    max_cycles = 200
    max_levels = 4
    min_coarse_cells = 500
    nu_pre = 3
    nu_post = 3
    nu_coarse = 1
    check_every_no = 1

    unconfined_inner_max_cycles_early = 2
    unconfined_inner_max_cycles_middle = 4
    unconfined_inner_max_cycles_late = 8
    unconfined_inner_middle_dh = 1.0
    unconfined_inner_late_dh = 1.0e-2

    optimize_kcycle = True
    stop_after_first_accepted = False
    kcycle_candidate_settings = [
        (1, 1, 1, 4),
        (2, 2, 1, 4),
        (3, 3, 1, 4),
        (1, 1, 1, 5),
        (2, 2, 1, 5),
        (3, 3, 1, 5),
        (1, 1, 1, 6),
        (2, 2, 1, 6),
        (3, 3, 1, 6),
        (4, 4, 1, 4),
        (4, 4, 1, 5),
        (4, 4, 1, 6),
    ]
    inner_cycle_candidate_settings = [
        (2, 4, 8, 1.0, 1.0e-2),
        (4, 8, 16, 1.0, 1.0e-2),
        (10, 25, 60, 1.0, 1.0e-2),
    ]
    kcycle_candidates = build_optimization_candidates(
        kcycle_settings=kcycle_candidate_settings,
        inner_cycle_settings=inner_cycle_candidate_settings,
    )
    accepted_mass_balance_classes = {"excellent", "good", "acceptable"}

    allow_warm_start_mismatch = False
    profile_performance = False
    save_heavy_diagnostics = False
    run_replay_matrix = False

    if explicit_artifact_path is not None:
        artifact_path = Path(explicit_artifact_path)
    elif use_grid_qualified_artifact:
        artifact_path = build_grid_artifact_path(
            formulation=formulation,
            nx=mf6_nx,
            ny=mf6_ny,
            n_weeks=mf6_n_periods,
        )
    else:
        artifact_path = None

    solve_control_overrides = {
        "max_outer_iterations": max_outer_iterations,
        "max_cycles": max_cycles,
        "max_levels": max_levels,
        "min_coarse_cells": min_coarse_cells,
        "nu_pre": nu_pre,
        "nu_post": nu_post,
        "nu_coarse": nu_coarse,
        "check_every_no": check_every_no,
        "unconfined_inner_max_cycles_early": unconfined_inner_max_cycles_early,
        "unconfined_inner_max_cycles_middle": unconfined_inner_max_cycles_middle,
        "unconfined_inner_max_cycles_late": unconfined_inner_max_cycles_late,
        "unconfined_inner_middle_dh": unconfined_inner_middle_dh,
        "unconfined_inner_late_dh": unconfined_inner_late_dh,
    }

    if optimize_kcycle:
        optimize_kcycle_settings(
            artifact_path=artifact_path,
            workspace=workspace,
            device=device,
            diag_preconditioner_backend=diag_preconditioner_backend,
            base_solve_controls=solve_control_overrides,
            candidates=kcycle_candidates,
            accepted_mass_balance_classes=accepted_mass_balance_classes,
            stop_after_first_accepted=stop_after_first_accepted,
            allow_warm_start_mismatch=allow_warm_start_mismatch,
            profile_performance=profile_performance,
            save_heavy_diagnostics=save_heavy_diagnostics,
        )
    else:
        run_production_replay(
            artifact_path=artifact_path,
            workspace=workspace,
            device=device,
            diag_preconditioner_backend=diag_preconditioner_backend,
            solve_control_overrides=solve_control_overrides,
            allow_warm_start_mismatch=allow_warm_start_mismatch,
            profile_performance=profile_performance,
            save_heavy_diagnostics=save_heavy_diagnostics,
            run_replay_matrix=run_replay_matrix,
        )
