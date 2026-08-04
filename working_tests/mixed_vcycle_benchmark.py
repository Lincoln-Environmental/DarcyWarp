#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 2 campaign benchmark: fixed FP32 V-cycle correction (Hypothesis 1).

Modes (one process each; DARCY_FLOAT pinned at import):

    # full mixed solves with V-cycle correction (FP32 hierarchy + FP64 outer)
    python working_tests/mixed_vcycle_benchmark.py --mode mixed-vcycle --vcycles 1
    python working_tests/mixed_vcycle_benchmark.py --mode mixed-vcycle --vcycles 2
    # reference: existing mixed solve with K-cycle correction
    python working_tests/mixed_vcycle_benchmark.py --mode mixed-kcycle
    # raw cycle-cost microbenchmark (V vs K, uncaptured), either precision
    python working_tests/mixed_vcycle_benchmark.py --mode vcost --precision float32
    python working_tests/mixed_vcycle_benchmark.py --mode vcost --precision float64

Benchmark integrity: every timed solve starts from the original DEM; one
solver invocation per timed run; accuracy vs cached FP64 reference heads and
cached MF6 truth (unchanged 2e-4 m gate); MF6 is never run here.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["mixed-vcycle", "mixed-kcycle", "mixed-fast", "vcost"], required=True)
parser.add_argument("--precision", choices=["float32", "float64"], default="float32",
                    help="vcost only: hierarchy precision")
parser.add_argument("--cases", type=str, default="100x100,1000x1001,2000x1000")
parser.add_argument("--reps", type=int, default=5)
parser.add_argument("--vcycles", type=int, default=1, help="V-cycles per outer (mixed-vcycle)")
parser.add_argument("--kcycles", type=int, default=5, help="K-cycles per outer (mixed-kcycle)")
parser.add_argument("--nu-pre", type=int, default=2)
parser.add_argument("--nu-post", type=int, default=2)
parser.add_argument("--nu-coarse", type=int, default=30)
parser.add_argument("--max-outer", type=int, default=60)
parser.add_argument("--vcost-cycles", type=int, default=30)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--no-ghb", dest="ghb", action="store_false", default=True)
parser.add_argument("--isotropic", action="store_true", default=False)
args = parser.parse_args()

if args.mode == "vcost":
    os.environ["DARCY_FLOAT"] = args.precision
else:
    os.environ["DARCY_FLOAT"] = "float32"

from DARCY_WARP_PACKAGE.model_builder import (  # noqa: E402
    _build_domain,
    _build_dem,
    build_truth_inputs,
    compare_head_fields,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver as wds  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy import compute_mass_balance_budget  # noqa: E402
import warp as wp  # noqa: E402

from working_tests.launch_profiler import LaunchProfiler  # noqa: E402
from working_tests.mf6_truth_cache import load_cached_mf6_truth  # noqa: E402
from DARCY_WARP_PACKAGE.sanity_case_config import SPATIAL_GRID_CASES  # noqa: E402

wp.init()

GRID_CASES = {
    label: {"nx": int(SPATIAL_GRID_CASES[label]["nx"]), "ny": int(SPATIAL_GRID_CASES[label]["ny"])}
    for label in ("100x100", "400x400", "1000x1001", "2000x1000")
}

DX = 100.0
R_TRUTH = 1.0e-4
THICKNESS = 300.0
T_SEED = 123
ISOTROPIC_T = 3000.0
ACC_GATE_M = 2.0e-4
REF_DIR = _REPO_ROOT / "working_tests" / "mixed_precision_ref"


def _mb(nbytes):
    return float(nbytes) / (1024.0 * 1024.0)


def build_case(label):
    cfg = GRID_CASES[label]
    nx, ny = int(cfg["nx"]), int(cfg["ny"])
    domain = _build_domain(nx=nx, ny=ny)
    dem = _build_dem(domain)
    if args.isotropic:
        T_field = np.full_like(domain, ISOTROPIC_T, dtype=np.float64)
    else:
        T_field = make_ugly_T_field(nx=nx, ny=ny, domain=domain, seed=int(T_SEED))
    R_field = np.full_like(domain, R_TRUTH, dtype=np.float64)
    (_, _, active64, bc_mask64, bc_values64, gh_mask64, gh_head64, _) = build_truth_inputs(
        nx=nx, ny=ny, dx=DX, T_truth=T_field, R_truth=R_field, use_ghb=bool(args.ghb), width=DX,
    )
    solver = wds(nx=nx, ny=ny, dx=DX, device=args.device, use_ghb=bool(args.ghb),
                 solver_type="pcg", aq_thickness=THICKNESS)
    solver.build_from_truth_inputs(T_truth=T_field, R_truth=R_field, width=DX)
    solver.build_hierarchy(max_levels=6, min_coarse_n=4, min_coarse_cells=500)
    return solver, dem, T_field, R_field, active64, bc_mask64, bc_values64, gh_mask64, gh_head64


def accuracy_block(label, head, active_mask):
    cfg_tag = f"ghb_{bool(args.ghb)}_t_isotropic_{bool(args.isotropic)}"
    ref_path = REF_DIR / f"warp_ref_{label}_{cfg_tag}.npz"
    if ref_path.exists():
        ref = np.asarray(np.load(ref_path)["heads"], dtype=np.float64)
        vs_fp64 = compare_head_fields(head_ref=ref, head_warp=head, active_mask=active_mask)
    else:
        vs_fp64 = {"max_abs_diff": None, "note": "run profile --mode fp64 first"}
    mf6_path = data_store.joinpath("mf6_truth_npz").joinpath(
        f"mf6_truth_{label}_{cfg_tag}.npz"
    )
    cached = load_cached_mf6_truth(mf6_path)
    if cached is not None:
        mf_head, t_mf = cached
        vs_mf6 = compare_head_fields(head_ref=mf_head, head_warp=head, active_mask=active_mask)
    else:
        t_mf, vs_mf6 = None, {"max_abs_diff": None, "note": "MF6 artifact missing"}
    return vs_fp64, vs_mf6, t_mf


def mass_balance_block(T_field, R_field, head, active64, bc_mask64, bc_values64,
                       gh_mask64, gh_head64):
    bud = compute_mass_balance_budget(
        T_field=T_field, R_field=R_field, head=head, active=active64,
        bc_mask=bc_mask64, bc_values=bc_values64, dx=DX,
        gh_mask=gh_mask64 if args.ghb else None,
        gh_head=gh_head64 if args.ghb else None,
        gh_width=np.where(gh_mask64 != 0, DX, 0.0) if args.ghb else None,
        gh_alpha=1.0,
        aq_thickness=THICKNESS,
    )
    try:
        rec = bud.to_dict("records")[0]
    except AttributeError:
        rec = dict(bud)
    return {
        "percent_discrepancy": float(rec["percent_discrepancy"]),
        "imbalance_fraction": float(rec["imbalance_fraction"]),
    }


def run_mixed(label):
    from DARCY_WARP_PACKAGE.solvers.mixed_precision import (
        MixedPrecisionDefectCorrectionSession,
    )
    from DARCY_WARP_PACKAGE.solvers.mixed_vcycle import (
        MixedPrecisionVcycleSession,
    )

    (solver, dem, T_field, R_field, active64, bc_mask64, bc_values64,
     gh_mask64, gh_head64) = build_case(label)
    active_mask = active64 != 0

    cls = MixedPrecisionDefectCorrectionSession if args.mode == "mixed-kcycle" else MixedPrecisionVcycleSession
    if args.mode == "mixed-fast":
        from DARCY_WARP_PACKAGE.solvers.mixed_fast import (
            MixedFastConfig,
            get_mixed_fast_session,
            solve_mixed_fast,
        )

        config = MixedFastConfig(
            inner_kcycles=int(args.kcycles),
            max_outer=int(args.max_outer),
            nu_pre=int(args.nu_pre),
            nu_post=int(args.nu_post),
            nu_coarse=int(args.nu_coarse),
        )
        session = get_mixed_fast_session(
            solver, bc_values_f64=bc_values64, gh_head_f64=gh_head64,
            R_f64=R_field, config=config,
        )

        def solve_once():
            return solve_mixed_fast(
                solver, dem, bc_values_f64=bc_values64,
                gh_head_f64=gh_head64, R_f64=R_field, config=config,
            )
    else:
        session = cls(solver, bc_values_f64=bc_values64, gh_head_f64=gh_head64,
                      R_f64=R_field, max_levels=6)

    wp.synchronize_device(args.device)
    mem_before = wp.get_mempool_used_mem_high(args.device)

    if args.mode != "mixed-fast":
        inner_controls = dict(
            inner_kcycles=int(args.vcycles) if args.mode == "mixed-vcycle" else int(args.kcycles),
            max_outer=int(args.max_outer),
            rel_tol=5.0e-7, abs_tol_min=5.0e-7,
        )
        if args.mode == "mixed-vcycle":
            inner_controls.update(
                nu_pre=int(args.nu_pre), nu_post=int(args.nu_post),
                nu_coarse=int(args.nu_coarse), omega=0.7,
            )

        def solve_once():
            return session.solve(dem, **inner_controls)

    head_cold, _ = solve_once()
    wp.synchronize_device(args.device)

    warm_times = []
    head, info = head_cold, None
    for _ in range(int(args.reps)):
        t0 = time.perf_counter()
        head, info = solve_once()
        wp.synchronize_device(args.device)
        warm_times.append(time.perf_counter() - t0)

    if getattr(solver, "_kcycle_graph", None) is not None:
        solver._kcycle_graph = None
    if getattr(session, "_correction_graph", None) is not None:
        session._correction_graph = None
    with LaunchProfiler() as prof:
        t0 = time.perf_counter()
        head_p, info_p = solve_once()
        wp.synchronize_device(args.device)
        profile_wall = time.perf_counter() - t0
    profile = prof.report()
    profile["solve_wall_seconds_instrumented"] = profile_wall

    head = np.asarray(head, dtype=np.float64)
    mem_after = wp.get_mempool_used_mem_high(args.device)
    solver.close()
    del solver, session
    gc.collect()

    vs_fp64, vs_mf6, t_mf = accuracy_block(label, head, active_mask)
    mb = mass_balance_block(T_field, R_field, head, active64, bc_mask64,
                            bc_values64, gh_mask64, gh_head64)

    hist = info["history"]
    contractions = [
        hist[k + 1]["r_rms64"] / hist[k]["r_rms64"]
        for k in range(len(hist) - 1)
        if hist[k]["r_rms64"] > 0
    ]
    return {
        "mode": args.mode,
        "vcycles_per_outer": int(args.vcycles) if args.mode == "mixed-vcycle" else None,
        "kcycles_per_outer": int(args.kcycles) if args.mode in ("mixed-kcycle", "mixed-fast") else None,
        "nu": [int(args.nu_pre), int(args.nu_post), int(args.nu_coarse)],
        "case": label,
        "warm_median_seconds": float(np.median(warm_times)),
        "warm_min_seconds": float(np.min(warm_times)),
        "warm_max_seconds": float(np.max(warm_times)),
        "n_warm_reps": len(warm_times),
        "converged": bool(info["converged"]),
        "outer_iterations": int(info["outer_iterations"]),
        "total_cycles": int(info["total_kcycles"]),
        "r_rms0_64": float(info["r_rms0_64"]),
        "r_rms_end_64": float(info["r_rms_end_64"]),
        "contraction_per_outer_median": float(np.median(contractions)) if contractions else None,
        "contraction_per_outer_last": float(contractions[-1]) if contractions else None,
        "r_rms64_history": [float(h["r_rms64"]) for h in hist],
        "profile": profile,
        "vs_fp64_warp": vs_fp64,
        "vs_mf6": vs_mf6,
        "accuracy_gate_max_abs_2e-4_pass": (
            None if vs_mf6.get("max_abs_diff") is None
            else bool(vs_mf6["max_abs_diff"] <= ACC_GATE_M)
        ),
        "mass_balance": mb,
        "memory": {
            "mempool_high_water_delta_mib": _mb(mem_after - mem_before),
            "mempool_high_water_mib": _mb(mem_after),
        },
        "mf6_seconds": t_mf,
    }


def run_vcost(label):
    from DARCY_WARP_PACKAGE.solvers.multigrid_kcycle import solve_kcycle_device_buffers
    from DARCY_WARP_PACKAGE.solvers.mixed_vcycle import solve_vcycle_device_buffers

    (solver, dem, T_field, R_field, active64, bc_mask64, bc_values64,
     gh_mask64, gh_head64) = build_case(label)
    nx, ny = solver.nx, solver.ny
    shape = (ny, nx)
    WP = wp.float32 if os.environ["DARCY_FLOAT"] == "float32" else wp.float64
    x = wp.zeros(shape, dtype=WP, device=args.device)
    r = wp.zeros(shape, dtype=WP, device=args.device)
    r.fill_(WP(1.0))  # arbitrary nonzero correction RHS
    zero_bc = wp.zeros(shape, dtype=WP, device=args.device)

    lvl0 = solver.mg_levels[0]
    front = (lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.storage_diag_wp,
             lvl0.active_wp, lvl0.bc_mask_wp, lvl0.bc_values_wp)

    controls = dict(nu_pre=2, nu_post=2, nu_coarse=2, omega=0.7,
                    smoother="chebyshev", cheby_lambda_min=0.1,
                    cheby_lambda_max=2.0, max_cycles=1,
                    check_every_no=1, dh_rms_tol=None, dh_max_tol=None)

    out = {"case": label, "precision": os.environ["DARCY_FLOAT"]}
    try:
        for kind in ("vcycle", "kcycle"):
            # warm-up (also compiles)
            for _ in range(3):
                if kind == "vcycle":
                    solve_vcycle_device_buffers(
                        model=solver, x_wp=x, rhs_wp=r, T_wp=solver.T_wp,
                        active_wp=solver.active_wp, bc_mask_wp=solver.bc_mask_wp,
                        bc_values_wp=zero_bc, levels=solver.mg_levels,
                        solve_controls=dict(controls, max_cycles=1),
                    )
                else:
                    solve_kcycle_device_buffers(
                        model=solver, x_wp=x, rhs_wp=r, T_wp=solver.T_wp,
                        storage_diag_wp=None, active_wp=solver.active_wp,
                        bc_mask_wp=solver.bc_mask_wp, bc_values_wp=zero_bc,
                        levels=solver.mg_levels,
                        solve_controls=dict(controls, max_cycles=1),
                        return_scalar_info=False, fixed_work_no_scalar_reads=True,
                    )
            wp.synchronize_device(args.device)

            times = []
            for _ in range(int(args.reps)):
                e0 = wp.Event(enable_timing=True)
                e1 = wp.Event(enable_timing=True)
                t0 = time.perf_counter()
                wp.record_event(e0)
                if kind == "vcycle":
                    solve_vcycle_device_buffers(
                        model=solver, x_wp=x, rhs_wp=r, T_wp=solver.T_wp,
                        active_wp=solver.active_wp, bc_mask_wp=solver.bc_mask_wp,
                        bc_values_wp=zero_bc, levels=solver.mg_levels,
                        solve_controls=dict(controls, max_cycles=int(args.vcost_cycles)),
                    )
                else:
                    solve_kcycle_device_buffers(
                        model=solver, x_wp=x, rhs_wp=r, T_wp=solver.T_wp,
                        storage_diag_wp=None, active_wp=solver.active_wp,
                        bc_mask_wp=solver.bc_mask_wp, bc_values_wp=zero_bc,
                        levels=solver.mg_levels,
                        solve_controls=dict(controls, max_cycles=int(args.vcost_cycles)),
                        return_scalar_info=False, fixed_work_no_scalar_reads=True,
                    )
                wp.record_event(e1)
                wp.synchronize_device(args.device)
                wall = time.perf_counter() - t0
                gpu_ms = wp.get_event_elapsed_time(e0, e1, synchronize=False)
                n = int(args.vcost_cycles)
                times.append({"gpu_ms_per_cycle": gpu_ms / n, "wall_s_per_cycle": wall / n})
            out[f"{kind}_gpu_ms_per_cycle_median"] = float(np.median([t["gpu_ms_per_cycle"] for t in times]))
            out[f"{kind}_wall_ms_per_cycle_median"] = float(1e3 * np.median([t["wall_s_per_cycle"] for t in times]))
    finally:
        (lvl0.x_wp, lvl0.b_wp, lvl0.T_wp, lvl0.storage_diag_wp,
         lvl0.active_wp, lvl0.bc_mask_wp, lvl0.bc_values_wp) = front
        solver.close()
        del solver
        gc.collect()
    return out


def main():
    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    tag = args.mode
    if args.mode == "mixed-vcycle":
        tag = f"mixed-vcycle-v{args.vcycles}-nu{args.nu_pre}{args.nu_post}"
    elif args.mode == "mixed-kcycle":
        tag = f"mixed-kcycle-k{args.kcycles}"
    elif args.mode == "mixed-fast":
        tag = f"mixed-fast-k{args.kcycles}-nuc{args.nu_coarse}"
    elif args.mode == "vcost":
        tag = f"vcost-{args.precision}"
    if args.mode != "vcost" and (not args.ghb or args.isotropic):
        tag += f"_ghb_{bool(args.ghb)}_iso_{bool(args.isotropic)}"
    out_path = _REPO_ROOT / "working_tests" / f"mixed_vcycle_results_{tag}.json"

    results = {}
    for label in selected:
        print(f"\n=== [{tag}] {label} ===", flush=True)
        if args.mode == "vcost":
            rec = run_vcost(label)
        else:
            rec = run_mixed(label)
        results[label] = rec
        printable = {k: v for k, v in rec.items() if k not in ("profile",)}
        print(json.dumps(printable, indent=2, default=str), flush=True)

    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
