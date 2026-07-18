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
    t_field_kind: str = "homogeneous",
    t_field_seed: int = 42,
) -> Path:
    """
    Build the grid-qualified MF6 truth artifact path used by the generator.

    Heterogeneous-T cases get a ``_ugly_t_s<seed>`` suffix so they never
    collide with the legacy homogeneous-artifact directories at the same grid.
    """
    if int(n_weeks) <= 0:
        raise ValueError("n_weeks must be positive.")
    t_field_kind = str(t_field_kind).strip().lower()
    suffix = f"_ugly_t_s{int(t_field_seed)}" if t_field_kind == "ugly_t" else ""
    return data_store.joinpath(
        "working_tests",
        f"mf6_transient_2d_{formulation}_{int(nx)}x{int(ny)}_{int(n_weeks)}w{suffix}",
        "mf6_transient_heads.npz.lzma",
    )


def build_case_setup(
    *,
    nx: int = 1000,
    ny: int = 1000,
    n_periods: int = 30,
    dx: float = 100.0,
    t_field_kind: str = "ugly_t",
    t_field_seed: int = 42,
) -> dict:
    """
    Single source of truth for the transient replay case.

    Defaults adopt the hard heterogeneous-T example from the confined
    steady-state benchmarks (``model_builder.make_ugly_T_field``, K = T / 100 m
    following the ``export_mf6_truth_npz`` convention). The generator script
    ``run_2d_transient_vs_mf6.py`` pulls this setup when run standalone, and
    ``ensure_case_artifact`` generates the MF6 artifact when it is missing.
    """
    formulation = FORMULATION_UNCONFINED
    setup = {
        "formulation": formulation,
        "nx": int(nx),
        "ny": int(ny),
        "dx": float(dx),
        "n_periods": int(n_periods),
        "dt_days": 7.0,
        "sy": 0.10,
        "ss": 1.0e-5,
        "annual_recharge_m": 0.3,
        "recharge_schedule_weeks": 52,
        "initial_saturated_thickness": 100.0,
        "t_field_kind": str(t_field_kind).strip().lower(),
        "t_field_seed": int(t_field_seed),
        "warm_start_mode": "unconfined_steady_mf6",
    }
    setup["artifact_path"] = build_grid_artifact_path(
        formulation=formulation,
        nx=nx,
        ny=ny,
        n_weeks=n_periods,
        t_field_kind=setup["t_field_kind"],
        t_field_seed=setup["t_field_seed"],
    )
    return setup


def ensure_case_artifact(case_setup: dict) -> Path:
    """
    Return the case artifact path, generating the MF6 truth when missing.

    The generator is imported lazily so the replay stays Flopy-free until a
    truth artifact actually has to be built.
    """
    artifact_path = Path(case_setup["artifact_path"])
    if artifact_path.exists():
        return artifact_path
    from working_tests import run_2d_transient_vs_mf6 as truth_generator

    print(f"MF6 truth artifact missing; generating (this can take a while):")
    print(f"  {artifact_path}")
    truth_generator.main(
        nx=case_setup["nx"],
        ny=case_setup["ny"],
        dx=case_setup["dx"],
        sy=case_setup["sy"],
        ss=case_setup["ss"],
        n_weeks=case_setup["n_periods"],
        annual_recharge_m=case_setup["annual_recharge_m"],
        recharge_schedule_weeks=case_setup["recharge_schedule_weeks"],
        initial_saturated_thickness=case_setup["initial_saturated_thickness"],
        t_field_kind=case_setup["t_field_kind"],
        t_field_seed=case_setup["t_field_seed"],
        out_path=artifact_path,
        reuse_existing_warm_start=True,
        warm_start_mode=case_setup["warm_start_mode"],
        formulation=case_setup["formulation"],
    )
    if not artifact_path.exists():
        raise RuntimeError(f"MF6 truth generation did not produce {artifact_path}")
    return artifact_path


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


if __name__ == "__main__":
    formulation = FORMULATION_UNCONFINED

    # ------------------------------------------------------------------
    # Replay artifact
    # ------------------------------------------------------------------
    # The case setup (grid, hard-T field, periods, warm start) is owned here;
    # the generator pulls it when run standalone, and the MF6 truth artifact
    # is generated on first use when missing.
    use_grid_qualified_artifact = True
    case_setup = build_case_setup()

    explicit_artifact_path = None
    workspace = None

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device = "auto"
    diag_preconditioner_backend = "device"

    allow_warm_start_mismatch = False
    profile_performance = False
    save_heavy_diagnostics = False
    run_replay_matrix = False

    # ------------------------------------------------------------------
    # Global safety ceilings
    # ------------------------------------------------------------------
    # These are hard failure ceilings, not normal iteration targets.
    max_outer_iterations = 100
    max_cycles = 200

    # ------------------------------------------------------------------
    # Multigrid hierarchy
    # ------------------------------------------------------------------
    max_levels = 4
    min_coarse_cells = 500

    nu_pre = 1
    nu_post = 1
    nu_coarse = 1

    # The adaptive fast path should normally check between cycle blocks,
    # rather than synchronizing after every individual K-cycle.
    check_every_no = 1

    # ------------------------------------------------------------------
    # Legacy head-change-driven inner schedule
    # ------------------------------------------------------------------
    # These controls remain available when:
    #
    #   adaptive_unconfined_inner_enabled = False
    #
    # or if the adaptive controller explicitly falls back because a valid
    # head-equivalent residual cannot be calculated.
    unconfined_inner_max_cycles_early = 10
    unconfined_inner_max_cycles_middle = 20
    unconfined_inner_max_cycles_late = 40

    unconfined_inner_middle_dh = 1.0
    unconfined_inner_late_dh = 1.0e-2

    # ------------------------------------------------------------------
    # Final nonlinear convergence and acceptance
    # ------------------------------------------------------------------
    hclose = 1.0e-4

    strict_head_residual_tol = 1.0e-6
    practical_head_residual_tol = 1.0e-5

    # Deprecated compatibility alias if still accepted by the solver.
    practical_residual_tol = practical_head_residual_tol

    practical_dh_rms_tol = 3.0e-3
    practical_storage_diag_change_rms_tol = 30.0
    # Practical acceptance is a fallback, not the normal path: strict Picard
    # converges in 10-12 outer iterations on the 1000x1000 production case, so
    # the practical gate must not fire earlier than that (the old value 8
    # short-circuited every period just before strict success).
    min_practical_outer_iterations = 20

    # ------------------------------------------------------------------
    # Inexact inner linear-solve bounds
    # ------------------------------------------------------------------
    # Early Picard linearisations do not need to be solved as accurately
    # as the final nonlinear state.
    inner_head_residual_tol_min = 2.5e-6
    inner_head_residual_tol_max = 2.0e-4

    # Prevent the inner linear tolerance from becoming disproportionately
    # tight relative to the current nonlinear Picard update.
    inner_picard_scale_max_fraction = 0.10

    # ------------------------------------------------------------------
    # Residual-driven adaptive inner controller
    # ------------------------------------------------------------------
    adaptive_unconfined_inner_enabled = True

    # Use genuinely small cycle blocks so the controller can observe
    # contraction and alter the next allocation without oversolving.
    adaptive_inner_initial_block_cycles = 5
    adaptive_inner_min_block_cycles = 5
    adaptive_inner_max_block_cycles = 20
    adaptive_inner_min_total_cycles = 5

    # Inexact-solve forcing term.
    #
    # These values deliberately permit relatively loose early Picard
    # linear solves. The target tightens as nonlinear convergence improves.
    adaptive_inner_eta_initial = 0.05
    adaptive_inner_eta_min = 0.005
    adaptive_inner_eta_max = 0.10
    adaptive_inner_eta_gamma = 0.25
    adaptive_inner_eta_power = 1.5

    # Block residual-contraction classification.
    adaptive_inner_good_contraction_ratio = 0.40
    adaptive_inner_weak_contraction_ratio = 0.90

    # Permit slow but useful contraction. A block is not considered stalled
    # merely because it reduces the residual by only a few percent.
    adaptive_inner_stall_contraction_ratio = 0.9995

    # Roll back only when the residual clearly worsens.
    adaptive_inner_divergence_contraction_ratio = 1.10

    # Require repeated stalled blocks before terminating the inner solve.
    adaptive_inner_stall_patience = 8

    # This is interpreted as:
    #
    #   final_residual / initial_residual <= threshold
    #
    # Validation uses target achievement, not incomplete usability, for
    # acceptance; retain the accepted candidate setting for diagnostics.
    adaptive_inner_minimum_usable_reduction_ratio = 0.10

    adaptive_inner_residual_floor = 1.0e-12

    # Enable temporarily while diagnosing the adaptive controller.
    # Disable after the controller has been validated if the history is large.
    adaptive_inner_save_block_history = True

    # ------------------------------------------------------------------
    # Optional K-cycle optimization sweep
    # ------------------------------------------------------------------
    optimize_kcycle = False
    stop_after_first_accepted = False

    kcycle_candidate_settings = [
        # Current light production configuration.
        (1, 1, 1, 4),]
    #
    #     # Stronger smoothing with the same hierarchy.
    #     (2, 2, 1, 4),
    #     (3, 3, 1, 4),
    #
    #     # Deeper hierarchy candidates.
    #     (1, 1, 1, 5),
    #     (2, 2, 1, 5),
    #     (3, 3, 1, 5),
    # ]

    # These values are relevant to legacy fallback runs. They do not control
    # the normal residual-driven path unless adaptation is disabled or fails.
    inner_cycle_candidate_settings = [
        (5, 20, 40, 1.0, 1.0e-2),
        (5, 25, 60, 1.0, 1.0e-2),
        (5, 30, 80, 1.0, 1.0e-2),
        (5, 30, 80, 1.0, 1.0e-2),
    ]

    adaptive_inner_candidate_settings = [
        # # Always retain a validated legacy control in an optimization run.
        # {
        #     "adaptive_unconfined_inner_enabled": False,
        # },

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

    kcycle_candidates = build_optimization_candidates(
        kcycle_settings=kcycle_candidate_settings,
        inner_cycle_settings=inner_cycle_candidate_settings,
        adaptive_inner_settings=adaptive_inner_candidate_settings,
    )

    accepted_mass_balance_classes = {
        "excellent",
        "good",
        "acceptable",
    }

    # ------------------------------------------------------------------
    # Resolve artifact
    # ------------------------------------------------------------------
    if explicit_artifact_path is not None:
        artifact_path = Path(explicit_artifact_path)

    elif use_grid_qualified_artifact:
        artifact_path = ensure_case_artifact(case_setup)

    else:
        artifact_path = None

    # ------------------------------------------------------------------
    # Solver control overrides
    # ------------------------------------------------------------------
    solve_control_overrides = {
        # Hard ceilings
        "max_outer_iterations": max_outer_iterations,
        "max_cycles": max_cycles,

        # Hierarchy and smoothing
        "max_levels": max_levels,
        "min_coarse_cells": min_coarse_cells,
        "nu_pre": nu_pre,
        "nu_post": nu_post,
        "nu_coarse": nu_coarse,
        "check_every_no": check_every_no,

        # Final nonlinear acceptance
        "hclose": hclose,
        "strict_head_residual_tol": strict_head_residual_tol,
        "practical_head_residual_tol": practical_head_residual_tol,
        "practical_residual_tol": practical_residual_tol,
        "practical_dh_rms_tol": practical_dh_rms_tol,
        "practical_storage_diag_change_rms_tol": (
            practical_storage_diag_change_rms_tol
        ),
        "min_practical_outer_iterations": min_practical_outer_iterations,

        # Inexact linear solve
        "inner_head_residual_tol_min": inner_head_residual_tol_min,
        "inner_head_residual_tol_max": inner_head_residual_tol_max,
        "inner_picard_scale_max_fraction": (
            inner_picard_scale_max_fraction
        ),

        # Legacy head-change fallback
        "unconfined_inner_max_cycles_early": (
            unconfined_inner_max_cycles_early
        ),
        "unconfined_inner_max_cycles_middle": (
            unconfined_inner_max_cycles_middle
        ),
        "unconfined_inner_max_cycles_late": (
            unconfined_inner_max_cycles_late
        ),
        "unconfined_inner_middle_dh": unconfined_inner_middle_dh,
        "unconfined_inner_late_dh": unconfined_inner_late_dh,

        # Residual-driven adaptive controller
        "adaptive_unconfined_inner_enabled": (
            adaptive_unconfined_inner_enabled
        ),
        "adaptive_inner_initial_block_cycles": (
            adaptive_inner_initial_block_cycles
        ),
        "adaptive_inner_min_block_cycles": (
            adaptive_inner_min_block_cycles
        ),
        "adaptive_inner_max_block_cycles": (
            adaptive_inner_max_block_cycles
        ),
        "adaptive_inner_min_total_cycles": (
            adaptive_inner_min_total_cycles
        ),
        "adaptive_inner_eta_initial": adaptive_inner_eta_initial,
        "adaptive_inner_eta_min": adaptive_inner_eta_min,
        "adaptive_inner_eta_max": adaptive_inner_eta_max,
        "adaptive_inner_eta_gamma": adaptive_inner_eta_gamma,
        "adaptive_inner_eta_power": adaptive_inner_eta_power,
        "adaptive_inner_good_contraction_ratio": (
            adaptive_inner_good_contraction_ratio
        ),
        "adaptive_inner_weak_contraction_ratio": (
            adaptive_inner_weak_contraction_ratio
        ),
        "adaptive_inner_stall_contraction_ratio": (
            adaptive_inner_stall_contraction_ratio
        ),
        "adaptive_inner_divergence_contraction_ratio": (
            adaptive_inner_divergence_contraction_ratio
        ),
        "adaptive_inner_stall_patience": (
            adaptive_inner_stall_patience
        ),
        "adaptive_inner_minimum_usable_reduction_ratio": (
            adaptive_inner_minimum_usable_reduction_ratio
        ),
        "adaptive_inner_residual_floor": (
            adaptive_inner_residual_floor
        ),
        "adaptive_inner_save_block_history": (
            adaptive_inner_save_block_history
        ),
    }

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------
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
