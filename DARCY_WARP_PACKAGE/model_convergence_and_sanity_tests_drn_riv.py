# SPDX-License-Identifier: AGPL-3.0-only
"""DRN/RIV variant of ``model_convergence_and_sanity_tests.py``.

Sweeps the shared spatial grid registry (``sanity_case_config``) with the
centre-row boundary as MODFLOW **DRN** or **RIV** cells instead of the GHB,
comparing Warp against a live-or-cached MF6 truth and applying acceptance
gates per case, with results merged into per-package JSON logs.

Differences from the GHB harness, all forced by the physics:

* DRN/RIV are gated head-dependent boundaries assembled only by the
  ``unconfined_semismooth_newton_kcycle`` backend (the confined K-cycle/PCG
  and Picard paths refuse them), so every case here is an *unconfined* Newton
  solve, warm-started from a plain Picard pre-solve by the runner.
* Conductances are fixed and identical on the MF6 and Warp sides, so MF6 runs
  exactly once per case (no GHB conductance fixed point).
* ``compute_mass_balance_budget`` (``physics/budgets_2d.py``) carries the
  gated DRN/RIV terms and is evaluated per case on the converged Warp heads
  with the converged ``T(h)`` — the Warp-side budget must close to <= 1%
  exactly like the GHB harness.

All heavy lifting (case construction, MF6 truth with trust gates,
fingerprinted artifact caching, Warp solve, head/flux comparison) lives in
``working_tests/run_2d_unconfined_warp_vs_mf6.py`` — this script is the
sweep + gates + JSON-log driver.  MF6 and Warp runtimes are recorded per case
and summarised in a head-to-head table at the end of the sweep.

Run from the repo root:

    python DARCY_WARP_PACKAGE/model_convergence_and_sanity_tests_drn_riv.py
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Importing the runner pins DARCY_FLOAT=float64 (required for the parity
# agreement tolerances); do not import warped_darcy before this line.
from working_tests.run_2d_unconfined_warp_vs_mf6 import (
    build_simple_unconfined_case,
    load_results,
    run_case,
)

from DARCY_WARP_PACKAGE.physics.budgets_2d import compute_mass_balance_budget
from DARCY_WARP_PACKAGE.project_base import data_store
from DARCY_WARP_PACKAGE.sanity_case_config import (
    SPATIAL_GRID_CASES,
    DEFAULT_GRID_LABELS,
    DEFAULT_DX,
    DEFAULT_R_TRUTH,
    DEFAULT_T_SEED,
)

# Flow min_sat floor of the nonlinear operator (backend default).
_MIN_SAT = 0.1


def _warp_flow_budget(case, warp_heads: np.ndarray, case_key: str):
    """Warp-side discrete flow budget at the converged head.

    Rebuilds the converged transmissivity ``T(h) = K*clip(h-bot, min_sat,
    top-bot)`` exactly as the nonlinear operator evaluates it, then assembles
    the recharge/CHD/DRN/RIV budget on the same sign conventions as MF6.
    """
    h = np.asarray(warp_heads, dtype=np.float64)
    K = np.asarray(case.hydraulic_conductivity, dtype=np.float64)
    sat = np.clip(h - case.bottom, _MIN_SAT, np.maximum(case.top - case.bottom, _MIN_SAT))
    T_conv = K * sat
    budget = compute_mass_balance_budget(
        T_field=T_conv,
        R_field=np.asarray(case.recharge, dtype=np.float64),
        head=h,
        active=case.active,
        bc_mask=case.bc_mask,
        bc_values=case.bc_values,
        dx=float(case.dx),
        drn_mask=case.drn_mask,
        drn_elev=case.drn_elev,
        drn_cond=case.drn_cond,
        riv_mask=case.riv_mask,
        riv_stage=case.riv_stage,
        riv_rbot=case.riv_rbot,
        riv_cond=case.riv_cond,
        case=case_key,
    )
    return budget



if __name__ == "__main__":
    # --- sweep switches -------------------------------------------------
    # Boundary packages to sweep (each gets its own JSON results file).
    boundary_kinds = ("drn", "riv")
    # Grids: default to the non-manual_only shared catalog (100x100 ..
    # 1000x1000).  Set to None to drive the full SPATIAL_GRID_CASES registry
    # including the manual_only near-capacity grids (9M cells will not fit a
    # 16 GB GPU on the Newton path).
    grid_labels = DEFAULT_GRID_LABELS  # or: tuple(SPATIAL_GRID_CASES)
    # T field: "ugly_t" reproduces the hard heterogeneous field of the GHB
    # harness (K = make_ugly_T_field/100).  Newton on hard-T large grids is
    # the known slow/fragile regime; "uniform" (K=hydraulic_conductivity) is
    # the fallback if a case fails to converge.
    t_field_kind = "ugly_t"
    t_field_seed = int(DEFAULT_T_SEED)
    hydraulic_conductivity = 10.0
    recharge = float(DEFAULT_R_TRUTH)
    dx = float(DEFAULT_DX)
    device = "cuda:0"
    # Boundary parameters (elevations are m above the cell bottom; the
    # bottom ramps with the DEM, so these give active drains / coupled river
    # cells on every grid).  Sized per cell; independent of aquifer K.
    drn_elevation = 20.0
    drn_conductance = 200.0
    riv_stage_elevation = 290.0
    riv_rbot_elevation = 5.0
    riv_conductance = 200.0
    # Double solve gives cold (solve1) vs warm (solve2) Warp timings, matching
    # the cold/warm CUDA-runtime reporting of the GHB harness.
    do_double_solve = True
    check_every_no = 5
    # Force MF6 to re-run (fresh engine timing) instead of reusing the
    # fingerprint-matched cached truth.  Cached MF6 times are the recorded
    # generation times, so the runtime comparison is valid either way.
    refresh_mf6_truth = False

    # Acceptance gates: head agreement matches the GHB harness tolerances
    # (observed parity on the tiny-grid tests is ~5e-7 m, so these carry wide
    # margin); flux gates compare the host-side boundary fluxes against the
    # MF6 CBC budget records.
    gate_rmse_max = 1.0e-3
    gate_max_abs_max = 1.0e-2
    gate_flux_total_rel_max = 1.0e-3
    gate_mf6_budget_pct_max = 1.0

    workspace_root = data_store.joinpath("working_tests", "mf6_vs_warp_2d_unconfined_drn_riv")

    results_by_kind: dict[str, dict] = {kind: {} for kind in boundary_kinds}

    for boundary_kind in boundary_kinds:
        for label in grid_labels:
            cfg = SPATIAL_GRID_CASES[label]
            nx, ny = int(cfg["nx"]), int(cfg["ny"])
            case_key = f"{boundary_kind}_{label}"
            print("\n" + "=" * 60)
            print(f"Running DRN/RIV sanity case {case_key} (nx={nx}, ny={ny})")
            print("=" * 60)

            t0 = time.perf_counter()
            case_workspace = workspace_root / boundary_kind / label
            if refresh_mf6_truth:
                for artifact in ("mf6_heads.npz", "warp_heads.npz"):
                    artifact_path = case_workspace / artifact
                    if artifact_path.exists():
                        artifact_path.unlink()
            try:
                summary = run_case(
                    nx=nx,
                    ny=ny,
                    dx=dx,
                    hydraulic_conductivity=hydraulic_conductivity,
                    recharge=recharge,
                    workspace=case_workspace,
                    device=device,
                    t_field_kind=t_field_kind,
                    t_field_seed=t_field_seed,
                    boundary_kind=boundary_kind,
                    drn_elevation=drn_elevation,
                    drn_conductance=drn_conductance,
                    riv_stage_elevation=riv_stage_elevation,
                    riv_rbot_elevation=riv_rbot_elevation,
                    riv_conductance=riv_conductance,
                    do_double_solve=do_double_solve,
                    check_every_no=check_every_no,
                )
            except Exception as exc:
                # A failed case (non-converged Newton, MF6 trust gate, OOM)
                # is recorded and the sweep continues; it fails the gates below.
                print(f"CASE FAILED with {type(exc).__name__}: {exc}")
                results_by_kind[boundary_kind][case_key] = {
                    "nx": nx,
                    "ny": ny,
                    "boundary_kind": boundary_kind,
                    "sanity_passed": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
                continue
            case_wall_seconds = time.perf_counter() - t0

            # Warp-side flow budget on the converged heads.  The case build is
            # deterministic (same seed/params -> identical fields), so this
            # re-build is cheap and matches what run_case solved.
            case = build_simple_unconfined_case(
                nx=nx,
                ny=ny,
                dx=dx,
                hydraulic_conductivity=hydraulic_conductivity,
                recharge=recharge,
                workspace=case_workspace,
                t_field_kind=t_field_kind,
                t_field_seed=t_field_seed,
                boundary_kind=boundary_kind,
                drn_elevation=drn_elevation,
                drn_conductance=drn_conductance,
                riv_stage_elevation=riv_stage_elevation,
                riv_rbot_elevation=riv_rbot_elevation,
                riv_conductance=riv_conductance,
            )
            _, warp_heads = load_results(
                case_workspace / "mf6_heads.npz", case_workspace / "warp_heads.npz"
            )
            budget = _warp_flow_budget(case, warp_heads, case_key)
            budget_row = budget.iloc[0]
            print("Warp flow budget (MF6 sign convention):")
            for col in (
                "rcha_in", "chd_in", "chd_out", "drn_in", "drn_out",
                "riv_in", "riv_out", "total_in", "total_out",
                "percent_discrepancy",
            ):
                print(f"  {col}: {float(budget_row[col]):.6g}")
            budget_percent_discrepancy = float(budget_row["percent_discrepancy"])

            comparison = summary.get("comparison") or {}
            flux = comparison.get("boundary_flux_comparison") or {}
            mf6_budget_pct = summary.get("mf6_budget_discrepancy_max")

            mf6_engine_s = summary.get("mf6_engine_time")
            warp_cold_s = summary.get("warp_solve1_time")
            warp_warm_s = summary.get("warp_solve2_time")
            speedup = (
                float(mf6_engine_s) / float(warp_warm_s)
                if mf6_engine_s and warp_warm_s else None
            )
            cold_str = f"{float(warp_cold_s):.3f}s" if warp_cold_s else "n/a"
            print(
                f"Runtime: MF6 engine {float(mf6_engine_s):.3f}s vs Warp cold "
                f"{cold_str} / warm {float(warp_warm_s):.3f}s"
                + (f"  (MF6/Warp speedup {speedup:.2f}x)" if speedup else "")
            )

            acceptance_gates = {
                "warp_converged": bool(summary.get("solve2_converged", False)),
                "head_agreement_vs_mf6_rmse_le_1e-3": (
                    float(comparison.get("rmse", np.inf)) <= gate_rmse_max
                ),
                "head_agreement_vs_mf6_max_abs_le_1e-2": (
                    float(comparison.get("max_abs_diff", np.inf)) <= gate_max_abs_max
                ),
                "boundary_flux_total_rel_diff_le_1e-3": (
                    bool(flux) and float(flux.get("relative_total_diff", np.inf)) <= gate_flux_total_rel_max
                ),
                "gate_pattern_matches_mf6": bool(flux.get("gate_pattern_matches_mf6", False)),
                "mass_balance_percent_discrepancy_le_1pct": (
                    abs(budget_percent_discrepancy) <= gate_mf6_budget_pct_max
                ),
                "mf6_budget_discrepancy_le_1pct": (
                    mf6_budget_pct is not None and abs(float(mf6_budget_pct)) <= gate_mf6_budget_pct_max
                ),
            }
            sanity_passed = bool(all(acceptance_gates.values()))

            print("Acceptance gates:")
            for gate_name, gate_passed in acceptance_gates.items():
                print(f"  {gate_name}: {'PASS' if gate_passed else 'FAIL'}")

            results_by_kind[boundary_kind][case_key] = {
                "nx": nx,
                "ny": ny,
                "n_cells": int(summary.get("n_cells", nx * ny)),
                "boundary_kind": boundary_kind,
                "t_field_kind": str(summary.get("t_field_kind")),
                "t_field_seed": int(summary.get("t_field_seed", t_field_seed)),
                "solver_backend": "unconfined_semismooth_newton_kcycle",
                "n_boundary_cells": int(flux.get("n_boundary_cells", 0)),
                "n_gated_off_warp": flux.get("n_gated_off_warp"),
                "n_gated_off_mf6": flux.get("n_gated_off_mf6"),
                "head_comparison": {
                    "rmse": comparison.get("rmse"),
                    "max_abs_diff": comparison.get("max_abs_diff"),
                    "mean_bias_warp_minus_mf6": comparison.get("mean_bias_warp_minus_mf6"),
                },
                "boundary_flux_comparison": flux,
                "flow_budget": {
                    key: float(budget_row[key])
                    for key in (
                        "rcha_in", "rcha_out", "chd_in", "chd_out",
                        "drn_in", "drn_out", "riv_in", "riv_out",
                        "total_in", "total_out", "in_minus_out",
                        "percent_discrepancy",
                    )
                },
                "acceptance_gates": acceptance_gates,
                "sanity_passed": sanity_passed,
                "timings": {
                    "mf6_engine_seconds": summary.get("mf6_engine_time"),
                    "mf6_total_seconds": summary.get("mf6_total_time"),
                    "warp_seconds_cold_solve": summary.get("warp_solve1_time"),
                    "warp_seconds_warm_solve": summary.get("warp_solve2_time"),
                    # Includes the Picard warm-start pre-solve for gated cases.
                    "warp_seconds_total": summary.get("warp_total_time"),
                    "case_wall_seconds": float(case_wall_seconds),
                },
                "mf6_engine_over_warp_warm_speedup": speedup,
                "mf6_budget_discrepancy_max": mf6_budget_pct,
                "case_fingerprint": summary.get("case_fingerprint"),
                "workspace": str(summary.get("workspace")),
            }

    # --- JSON logs (one per boundary package; merged like the GHB harness) --
    failed_cases = []
    for boundary_kind, results in results_by_kind.items():
        results_path = data_store.joinpath(
            f"comparison_results_{boundary_kind}_{t_field_kind}.json"
        )
        try:
            if results_path.exists():
                with results_path.open("r") as f:
                    existing = json.load(f)
            else:
                existing = {}
        except Exception:
            existing = {}
        existing.update(results)
        with results_path.open("w") as f:
            json.dump(existing, f, indent=4, default=str)
        print(f"\nSaved {boundary_kind.upper()} comparison results to {results_path}")
        failed_cases.extend(
            key for key, row in results.items() if not row.get("sanity_passed", False)
        )

    # --- head-to-head runtime table ---------------------------------------
    print("\nMF6 vs Warp runtime comparison (seconds):")
    header = f"{'case':<24} {'MF6 engine':>10} {'Warp cold':>10} {'Warp warm':>10} {'MF6/Warp':>9}"
    print(header)
    print("-" * len(header))
    for boundary_kind, results in results_by_kind.items():
        for case_key, row in results.items():
            timings = row.get("timings") or {}
            mf6_s = timings.get("mf6_engine_seconds")
            warm_s = timings.get("warp_seconds_warm_solve")
            cold_s = timings.get("warp_seconds_cold_solve")
            speedup = row.get("mf6_engine_over_warp_warm_speedup")
            print(
                f"{case_key:<24} "
                f"{(f'{mf6_s:>10.3f}' if mf6_s is not None else ' ' * 9 + '-')} "
                f"{(f'{cold_s:>10.3f}' if cold_s else ' ' * 9 + '-')} "
                f"{(f'{warm_s:>10.3f}' if warm_s is not None else ' ' * 9 + '-')} "
                f"{(f'{speedup:>8.2f}x' if speedup else ' ' * 8 + '-')}"
            )

    if failed_cases:
        print(f"\nSANITY GATES FAILED for cases: {failed_cases}")
        raise SystemExit(1)
    print("\nAll DRN/RIV sanity gates passed.")
