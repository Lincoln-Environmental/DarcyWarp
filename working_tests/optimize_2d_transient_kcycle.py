#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""K-cycle / adaptive inner-controller tuning sweep for the 2D transient replay.

Experiment harness, not part of the production path: it runs solve-control
candidates through the production replay
(``run_2d_transient_warp_replay.run_production_replay``) and selects the
fastest numerically accepted variant. Production settings live in
``working_tests/transient_replay_settings.py`` — tune there, sweep here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_tests.run_2d_transient_warp_replay import (  # noqa: E402
    build_case_setup,
    ensure_case_artifact,
    run_production_replay,
)
from working_tests.transient_replay_settings import default_solve_controls  # noqa: E402

ACCEPTED_MASS_BALANCE_CLASSES = frozenset({"excellent", "good", "acceptable"})


def variant_workspace_name(
    *,
    candidate: dict,
) -> str:
    """
    Build a stable workspace name for one optimization candidate.
    """
    name = candidate.get("name")
    if name is not None and str(name).strip():
        return str(name).strip()
    return (
        f"nu_pre_{int(candidate['nu_pre'])}_post_{int(candidate['nu_post'])}"
        f"_coarse_{int(candidate['nu_coarse'])}_max_levels_{int(candidate['max_levels'])}"
        f"_inner_{int(candidate['inner_early'])}_{int(candidate['inner_middle'])}_{int(candidate['inner_late'])}"
        f"_adaptive_{adaptive_candidate_name_suffix(candidate=candidate)}"
    )


def adaptive_candidate_name_suffix(
    *,
    candidate: dict,
) -> str:
    """
    Build the workspace-name suffix for adaptive inner-controller controls.
    """
    controls = dict(candidate.get("adaptive_controls") or {})
    if not controls:
        return "default"
    enabled = int(bool(controls.get("adaptive_unconfined_inner_enabled", True)))
    if enabled == 0:
        return "legacy_dh"
    initial = int(controls.get("adaptive_inner_initial_block_cycles", 4))
    minimum = int(controls.get("adaptive_inner_min_block_cycles", 2))
    maximum = int(controls.get("adaptive_inner_max_block_cycles", 16))
    eta_initial = float(controls.get("adaptive_inner_eta_initial", 0.25))
    eta_min = float(controls.get("adaptive_inner_eta_min", 0.02))
    eta_max = float(controls.get("adaptive_inner_eta_max", 0.30))
    stall = float(controls.get("adaptive_inner_stall_contraction_ratio", 0.98))
    divergence = float(controls.get("adaptive_inner_divergence_contraction_ratio", 1.05))
    return (
        f"en_{enabled}_block_{initial}_{minimum}_{maximum}"
        f"_eta_{eta_initial:g}_{eta_min:g}_{eta_max:g}"
        f"_stall_{stall:g}_div_{divergence:g}"
    ).replace(".", "p").replace("-", "m")


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
    controls.update(dict(candidate.get("adaptive_controls") or {}))
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
    adaptive_controls: dict | None = None,
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
        "adaptive_controls": dict(adaptive_controls or {}),
    }


def build_optimization_candidates(
    *,
    kcycle_settings: list[tuple[int, int, int, int]],
    inner_cycle_settings: list[tuple[int, int, int, float, float]],
    adaptive_inner_settings: list[dict] | None = None,
) -> list[dict]:
    """
    Build candidate settings without multiplying adaptive sweeps by legacy schedules.
    """
    candidates: list[dict] = []
    seen_names: set[str] = set()
    adaptive_settings = list(adaptive_inner_settings or [{}])
    for nu_pre, nu_post, nu_coarse, max_levels in kcycle_settings:
        for adaptive_controls in adaptive_settings:
            adaptive_enabled = bool(adaptive_controls.get("adaptive_unconfined_inner_enabled", False))
            cycle_settings = inner_cycle_settings[:1] if adaptive_enabled else inner_cycle_settings
            for inner_early, inner_middle, inner_late, inner_middle_dh, inner_late_dh in cycle_settings:
                adaptive_suffix = adaptive_candidate_name_suffix(
                    candidate={"adaptive_controls": adaptive_controls},
                )
                name = (
                    f"nu_{int(nu_pre)}_{int(nu_post)}_coarse_{int(nu_coarse)}"
                    f"_levels_{int(max_levels)}"
                    f"_inner_{int(inner_early)}_{int(inner_middle)}_{int(inner_late)}"
                    f"_dh_{float(inner_middle_dh):g}_{float(inner_late_dh):g}"
                    f"_adaptive_{adaptive_suffix}"
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
                        adaptive_controls=adaptive_controls,
                    )
                )
    return candidates


def replay_summary_is_accepted(
    *,
    summary: dict,
    accepted_mass_balance_classes: frozenset[str] = ACCEPTED_MASS_BALANCE_CLASSES,
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
    candidates: list[dict],
    stop_after_first_accepted: bool = False,
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
            base_controls=default_solve_controls(),
            candidate=candidate,
        )
        print(f"\nK-cycle optimization candidate: {name}")
        summary = run_production_replay(
            artifact_path=artifact_path,
            workspace=variant_workspace,
            device=device,
            solve_control_overrides=controls,
        )
        accepted = replay_summary_is_accepted(summary=summary)
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
            "adaptive_controls": dict(candidate.get("adaptive_controls") or {}),
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
        "best_adaptive_controls": dict(best.get("adaptive_controls") or {}),
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
                "adaptive_controls": dict(result.get("adaptive_controls") or {}),
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
    print(f"  adaptive: {dict(best.get('adaptive_controls') or {})}")
    print(f"  workspace: {best['workspace']}")
    print(f"  summary: {optimization_summary_path}")
    return {
        "best": best,
        "results": results,
        "optimization_workspace": str(optimization_workspace),
        "optimization_summary_path": str(optimization_summary_path),
    }


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t-field-kind",
        choices=("ugly_t", "homogeneous"),
        default="ugly_t",
        help="transmissivity field kind (must match the artifact)",
    )
    parser.add_argument("--t-field-seed", type=int, default=42, help="ugly_t field random seed")
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=1000)
    parser.add_argument("--n-periods", type=int, default=30)
    parser.add_argument("--artifact", default=None, help="explicit artifact path (bypasses the case setup)")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--stop-after-first-accepted",
        action="store_true",
        help="stop the sweep at the first numerically accepted candidate "
             "instead of ranking all candidates by wall time",
    )
    args = parser.parse_args()

    case_setup = build_case_setup(
        nx=args.nx,
        ny=args.ny,
        n_periods=args.n_periods,
        t_field_kind=args.t_field_kind,
        t_field_seed=args.t_field_seed,
    )
    artifact_path = Path(args.artifact) if args.artifact is not None else ensure_case_artifact(case_setup)
    workspace = Path(args.workspace) if args.workspace else None

    # K-cycle hierarchy/smoothing candidates: (nu_pre, nu_post, nu_coarse, max_levels).
    # (1, 1, 1, 4) is the current light production configuration.
    kcycle_candidate_settings = [
        (1, 1, 1, 4),
    ]

    # Legacy dh-schedule inner-cycle caps: (early, middle, late, middle_dh, late_dh).
    # These are relevant to legacy fallback runs only; they do not control the
    # normal residual-driven path. build_optimization_candidates uses just the
    # first entry for adaptive candidates — the full list only multiplies the
    # sweep when a legacy {"adaptive_unconfined_inner_enabled": False} entry is
    # present in adaptive_inner_candidate_settings.
    inner_cycle_candidate_settings = [
        (5, 20, 40, 1.0, 1.0e-2),
    ]

    adaptive_inner_candidate_settings = [
        # Conservative adaptive controller
        {
            "adaptive_unconfined_inner_enabled": True,
            "adaptive_inner_initial_block_cycles": 5,
            "adaptive_inner_min_block_cycles": 5,
            "adaptive_inner_max_block_cycles": 20,
            "adaptive_inner_min_total_cycles": 5,
            "adaptive_inner_eta_initial": 0.05,
            "adaptive_inner_eta_min": 0.005,
            "adaptive_inner_eta_max": 0.10,
            "adaptive_inner_eta_gamma": 0.25,
            "adaptive_inner_eta_power": 1.5,
            "adaptive_inner_good_contraction_ratio": 0.40,
            "adaptive_inner_weak_contraction_ratio": 0.90,
            "adaptive_inner_stall_contraction_ratio": 0.9995,
            "adaptive_inner_divergence_contraction_ratio": 1.10,
            "adaptive_inner_stall_patience": 8,
            "adaptive_inner_minimum_usable_reduction_ratio": 0.10,
            "adaptive_inner_save_block_history": True,
        },

        # Stricter residual reduction
        {
            "adaptive_unconfined_inner_enabled": True,
            "adaptive_inner_initial_block_cycles": 5,
            "adaptive_inner_min_block_cycles": 5,
            "adaptive_inner_max_block_cycles": 20,
            "adaptive_inner_min_total_cycles": 10,
            "adaptive_inner_eta_initial": 0.025,
            "adaptive_inner_eta_min": 0.0025,
            "adaptive_inner_eta_max": 0.05,
            "adaptive_inner_eta_gamma": 0.20,
            "adaptive_inner_eta_power": 1.5,
            "adaptive_inner_good_contraction_ratio": 0.40,
            "adaptive_inner_weak_contraction_ratio": 0.90,
            "adaptive_inner_stall_contraction_ratio": 0.9995,
            "adaptive_inner_divergence_contraction_ratio": 1.10,
            "adaptive_inner_stall_patience": 10,
            "adaptive_inner_minimum_usable_reduction_ratio": 0.05,
            "adaptive_inner_save_block_history": True,
        },

        # Adaptive blocks, but require target rather than incomplete usability
        {
            "adaptive_unconfined_inner_enabled": True,
            "adaptive_inner_initial_block_cycles": 5,
            "adaptive_inner_min_block_cycles": 5,
            "adaptive_inner_max_block_cycles": 20,
            "adaptive_inner_min_total_cycles": 10,
            "adaptive_inner_eta_initial": 0.05,
            "adaptive_inner_eta_min": 0.005,
            "adaptive_inner_eta_max": 0.10,
            "adaptive_inner_eta_gamma": 0.25,
            "adaptive_inner_eta_power": 1.5,
            "adaptive_inner_stall_contraction_ratio": 0.9995,
            "adaptive_inner_divergence_contraction_ratio": 1.10,
            "adaptive_inner_stall_patience": 10,
            "adaptive_inner_minimum_usable_reduction_ratio": 0.01,
            "adaptive_inner_save_block_history": True,
        },
    ]

    candidates = build_optimization_candidates(
        kcycle_settings=kcycle_candidate_settings,
        inner_cycle_settings=inner_cycle_candidate_settings,
        adaptive_inner_settings=adaptive_inner_candidate_settings,
    )
    return optimize_kcycle_settings(
        artifact_path=artifact_path,
        workspace=workspace,
        device=args.device,
        candidates=candidates,
        stop_after_first_accepted=args.stop_after_first_accepted,
    )


if __name__ == "__main__":
    main()
