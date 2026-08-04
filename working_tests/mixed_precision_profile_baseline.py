#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Phase 1 profiling harness for the mixed-precision campaign.

One mode per process (DARCY_FLOAT is pinned at import time):

    python working_tests/mixed_precision_profile_baseline.py --mode fp64  --cases 2000x1000
    python working_tests/mixed_precision_profile_baseline.py --mode fp32  --cases 2000x1000
    python working_tests/mixed_precision_profile_baseline.py --mode mixed --cases 2000x1000

Per case it records, in JSON:
  * build / hierarchy / cold-solve / warm-solve wall times (median/min/max);
  * a launch-instrumented profile of one warm solve: per-kernel GPU time
    (CUDA events), launch counts, host synchronizations and scalar readbacks;
  * mempool high-water memory;
  * accuracy vs cached FP64 Warp reference heads and cached MF6 truth.

Benchmark integrity: every timed solve starts from the same original DEM host
array; one solver invocation per timed run; MF6 runs only on cache miss.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["fp64", "fp32", "mixed"], required=True)
parser.add_argument("--cases", type=str, default="100x100,1000x1001,2000x1000")
parser.add_argument("--reps", type=int, default=5, help="timed warm repetitions")
parser.add_argument("--fp32-profile-cycles", type=int, default=20,
                    help="cycle cap for the instrumented fp32 run (it never converges)")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--out", type=str, default="")
args = parser.parse_args()

os.environ["DARCY_FLOAT"] = "float64" if args.mode == "fp64" else "float32"

from DARCY_WARP_PACKAGE.model_builder import (  # noqa: E402
    _build_domain,
    _build_dem,
    build_truth_inputs,
    compare_head_fields,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver as wds  # noqa: E402
from DARCY_WARP_PACKAGE.sanity_case_config import SPATIAL_GRID_CASES  # noqa: E402
import warp as wp  # noqa: E402

wp.init()

from working_tests.mf6_truth_cache import load_cached_mf6_truth  # noqa: E402
from working_tests.launch_profiler import LaunchProfiler  # noqa: E402

if args.mode == "mixed":
    from DARCY_WARP_PACKAGE.solvers.mixed_precision import (  # noqa: E402
        MixedPrecisionDefectCorrectionSession,
    )

GRID_CASES = {
    label: {"nx": int(SPATIAL_GRID_CASES[label]["nx"]), "ny": int(SPATIAL_GRID_CASES[label]["ny"])}
    for label in ("100x100", "100x250", "400x400", "100x1000", "250x1000", "1000x1001", "2000x1000")
}

DX = 100.0
R_TRUTH = 1.0e-4
THICKNESS = 300.0
T_SEED = 123
ISOTROPIC_T = 3000.0
ACC_GATE_M = 2.0e-4

REF_DIR = _REPO_ROOT / "working_tests" / "mixed_precision_ref"


# ---------------------------------------------------------------------------
# Launch / synchronization instrumentation
# ---------------------------------------------------------------------------


def kcycle_controls(max_cycles=200):
    return dict(
        max_cycles=max_cycles,
        nu_pre=2, nu_post=2, nu_coarse=2, omega=0.7,
        rel_tol=5.0e-7, abs_tol_min=5.0e-7,
        return_info=True, max_levels=6, check_every_no=5,
    )


def _mb(nbytes):
    return float(nbytes) / (1024.0 * 1024.0)


def main():
    device = args.device
    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    out_path = Path(args.out) if args.out else (
        _REPO_ROOT / "working_tests" / f"mixed_precision_profile_{args.mode}.json"
    )

    results = {}
    for label in selected:
        cfg = GRID_CASES[label]
        nx, ny = int(cfg["nx"]), int(cfg["ny"])
        print(f"\n=== [{args.mode}] {label} ===", flush=True)

        domain = _build_domain(nx=nx, ny=ny)
        dem = _build_dem(domain)
        active_mask = domain == 1
        T_field = make_ugly_T_field(nx=nx, ny=ny, domain=domain, seed=int(T_SEED))
        R_field = np.full_like(domain, R_TRUTH, dtype=np.float64)
        (_, _, _, _, bc_values64, _, gh_head64, _) = build_truth_inputs(
            nx=nx, ny=ny, dx=DX, T_truth=T_field, R_truth=R_field,
            use_ghb=True, width=DX,
        )

        wp.synchronize_device(device)
        mem_before = wp.get_mempool_used_mem_high(device)

        t0 = time.perf_counter()
        solver = wds(nx=nx, ny=ny, dx=DX, device=device, use_ghb=True,
                     solver_type="pcg", aq_thickness=THICKNESS)
        solver.build_from_truth_inputs(T_truth=T_field, R_truth=R_field, width=DX)
        t1 = time.perf_counter()
        build_seconds = t1 - t0

        t0 = time.perf_counter()
        solver.build_hierarchy(max_levels=6, min_coarse_n=4, min_coarse_cells=500)
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        hierarchy_seconds = t1 - t0

        session = None
        session_seconds = 0.0
        if args.mode == "mixed":
            t0 = time.perf_counter()
            session = MixedPrecisionDefectCorrectionSession(
                solver, bc_values_f64=bc_values64, gh_head_f64=gh_head64,
                R_f64=R_field, max_levels=6,
            )
            t1 = time.perf_counter()
            session_seconds = t1 - t0

            def solve_once():
                return session.solve(dem, inner_kcycles=5, max_outer=40,
                                     rel_tol=5.0e-7, abs_tol_min=5.0e-7)
        else:
            cycles = args.fp32_profile_cycles if args.mode == "fp32" else 200

            def solve_once():
                return solver.solve_multigrid_kcycle(initial_head=dem, **kcycle_controls(cycles))

        # cold solve (first invocation; hierarchy already built above)
        t0 = time.perf_counter()
        head_cold, info_cold = solve_once()
        wp.synchronize_device(device)
        t1 = time.perf_counter()
        cold_seconds = t1 - t0

        # timed warm reps
        warm_times = []
        head, info = head_cold, info_cold
        for _ in range(int(args.reps)):
            t0 = time.perf_counter()
            head, info = solve_once()
            wp.synchronize_device(device)
            t1 = time.perf_counter()
            warm_times.append(t1 - t0)

        # instrumented single warm solve (eager: cached CUDA graph invalidated
        # and capture disabled inside LaunchProfiler)
        if getattr(solver, "_kcycle_graph", None) is not None:
            solver._kcycle_graph = None
        with LaunchProfiler() as prof:
            t0 = time.perf_counter()
            head_p, info_p = solve_once()
            wp.synchronize_device(device)
            t1 = time.perf_counter()
        profile = prof.report()
        profile["solve_wall_seconds_instrumented"] = t1 - t0

        head = np.asarray(head, dtype=np.float64)
        mem_after = wp.get_mempool_used_mem_high(device)
        solver.close()
        del solver, session
        gc.collect()
        wp.synchronize_device(device)

        # accuracy: vs cached FP64 reference and cached MF6 truth
        ref_path = REF_DIR / f"warp_ref_{label}_ghb_True_t_isotropic_False.npz"
        if args.mode == "fp64":
            REF_DIR.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ref_path, heads=head)
            vs_fp64 = {"max_abs_diff": 0.0, "rms_diff": 0.0}
        elif ref_path.exists():
            ref = np.asarray(np.load(ref_path)["heads"], dtype=np.float64)
            vs_fp64 = compare_head_fields(head_ref=ref, head_warp=head, active_mask=active_mask)
        else:
            vs_fp64 = {"max_abs_diff": None, "note": "run --mode fp64 first"}

        mf6_path = data_store.joinpath("mf6_truth_npz").joinpath(
            f"mf6_truth_{label}_ghb_True_t_isotropic_False.npz"
        )
        cached = load_cached_mf6_truth(mf6_path)
        if cached is not None:
            mf_head, t_mf = cached
            vs_mf6 = compare_head_fields(head_ref=mf_head, head_warp=head, active_mask=active_mask)
        else:
            t_mf = None
            vs_mf6 = {"max_abs_diff": None, "note": "MF6 artifact missing; not running MF6 here"}

        info_out = {k: v for k, v in info.items() if k != "history"}
        results[label] = {
            "mode": args.mode,
            "nx": nx, "ny": ny,
            "starting_head": "original DEM for every timed solve",
            "build_seconds": build_seconds,
            "hierarchy_seconds": hierarchy_seconds,
            "session_init_seconds": session_seconds,
            "cold_solve_seconds": cold_seconds,
            "warm_solves_seconds": warm_times,
            "warm_median_seconds": float(np.median(warm_times)),
            "warm_min_seconds": float(np.min(warm_times)),
            "warm_max_seconds": float(np.max(warm_times)),
            "n_warm_reps": len(warm_times),
            "mf6_seconds": t_mf,
            "solver_info": info_out,
            "profile": profile,
            "vs_fp64_warp": vs_fp64,
            "vs_mf6": vs_mf6,
            "accuracy_gate_max_abs_2e-4_pass": (
                None if vs_mf6.get("max_abs_diff") is None
                else bool(vs_mf6["max_abs_diff"] <= ACC_GATE_M)
            ),
            "memory": {
                "mempool_high_water_delta_mib": _mb(mem_after - mem_before),
                "mempool_high_water_mib": _mb(mem_after),
            },
        }
        print(json.dumps({
            "case": label,
            "warm_median_s": round(results[label]["warm_median_seconds"], 4),
            "converged": info_out.get("converged"),
            "launches": profile["launches_total"],
            "gpu_busy_ms": round(profile["gpu_busy_ms"], 2),
            "host_syncs": profile["host_sync_calls"],
            "numpy_readbacks": profile["numpy_readbacks"],
            "vs_mf6_max": vs_mf6.get("max_abs_diff"),
        }, indent=2), flush=True)

    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            existing = {}
    existing.update(results)
    out_path.write_text(json.dumps(existing, indent=2, default=str))
    print(f"\nSaved profile to {out_path}")


if __name__ == "__main__":
    main()
