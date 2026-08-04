#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""EXPERIMENTAL mixed-precision benchmark harness (steady confined 2D).

Runs one mode per process (DARCY_FLOAT is fixed at import time):

    python working_tests/mixed_precision_benchmark.py --mode fp64
    python working_tests/mixed_precision_benchmark.py --mode fp32
    python working_tests/mixed_precision_benchmark.py --mode mixed

Benchmark integrity:
  * every timed solve (CUDA cold-runtime and CUDA-runtime-warm) starts from the
    same original DEM host array; "warm" only reuses compiled kernels / CUDA
    runtime state / allocations;
  * one solver invocation per timed run; all K-cycles and all defect-correction
    iterations are inside that single timed call;
  * MF6 heads are loaded from the existing artifact cache; MF6 runs only on a
    cache miss;
  * the Warp-vs-MF6 accuracy gate is unchanged (2e-4 m max head difference);
  * fp64 mode additionally caches FP64 Warp reference heads per case so fp32 /
    mixed modes can be compared against full-FP64 Warp.
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

# ---------------------------------------------------------------------------
# CLI first: DARCY_FLOAT must be fixed before warped_darcy is imported.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT))

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=["fp64", "fp32", "mixed"], required=True)
parser.add_argument(
    "--cases",
    type=str,
    default="100x100,100x1000,400x400,1000x1001,2000x1000",
    help="comma-separated subset of grid cases",
)
parser.add_argument("--ghb", dest="ghb", action="store_true", default=True)
parser.add_argument("--no-ghb", dest="ghb", action="store_false")
parser.add_argument("--isotropic", action="store_true", default=False)
parser.add_argument("--inner-kcycles", type=int, default=5)
parser.add_argument("--max-outer", type=int, default=40)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

os.environ["DARCY_FLOAT"] = "float64" if args.mode == "fp64" else "float32"

from DARCY_WARP_PACKAGE.model_builder import (  # noqa: E402
    _build_domain,
    _build_dem,
    build_truth_inputs,
    compare_head_fields,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model  # noqa: E402
from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver as wds  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy import compute_mass_balance_budget  # noqa: E402
from DARCY_WARP_PACKAGE.sanity_case_config import SPATIAL_GRID_CASES  # noqa: E402
import warp as wp  # noqa: E402

wp.init()

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
ACC_GATE_M = 2.0e-4  # unchanged Warp-vs-MF6 accuracy requirement

REF_DIR = _REPO_ROOT / "working_tests" / "mixed_precision_ref"


# ---------------------------------------------------------------------------
# MF6 artifact cache helpers (mirrors model_convergence_and_sanity_tests.py;
# duplicated because that module forces DARCY_FLOAT=float64 at import time).
# ---------------------------------------------------------------------------


def build_mf6_truth_path(truth_dir: Path, label: str, ghb: bool, isotropic: bool) -> Path:
    filename = f"mf6_truth_{label}_ghb_{bool(ghb)}_t_isotropic_{bool(isotropic)}.npz"
    return truth_dir.joinpath(filename)


def load_cached_mf6_truth(truth_path, label, nx, ny, dx, ghb, isotropic,
                          t_isotropic_value, thickness, width, recharge, seed):
    if not truth_path.exists():
        return None
    try:
        with np.load(truth_path, allow_pickle=False) as truth:
            expected_scalars = {
                "nx": int(nx), "ny": int(ny), "ghb": int(bool(ghb)),
                "t_isotropic": int(bool(isotropic)), "seed": int(seed),
            }
            for key, expected in expected_scalars.items():
                if key not in truth.files or int(truth[key]) != expected:
                    return None
            expected_floats = {
                "dx": float(dx), "t_isotropic_value": float(t_isotropic_value),
                "thickness": float(thickness), "width": float(width),
                "r_truth": float(recharge),
            }
            for key, expected in expected_floats.items():
                if key not in truth.files or not np.isclose(
                    float(truth[key]), expected, rtol=1.0e-6, atol=1.0e-12
                ):
                    return None
            if "label" not in truth.files or str(truth["label"]) != str(label):
                return None
            if "heads" not in truth.files:
                return None
            heads = np.asarray(truth["heads"], dtype=np.float64)
            if heads.shape != (int(ny), int(nx)) or not np.all(np.isfinite(heads)):
                return None
            mf6_seconds = None
            if "mf6_seconds" in truth.files:
                candidate = float(truth["mf6_seconds"])
                if np.isfinite(candidate) and candidate >= 0.0:
                    mf6_seconds = candidate
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return heads, mf6_seconds


def save_mf6_truth(truth_path, heads, label, nx, ny, dx, ghb, isotropic,
                   t_isotropic_value, thickness, width, recharge, seed,
                   mf6_seconds, output_dtype):
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = truth_path.with_name(f".{truth_path.name}.staging.npz")
    fd = np.dtype(output_dtype)
    np.savez_compressed(
        tmp,
        heads=np.asarray(heads, dtype=fd),
        nx=np.int32(nx), ny=np.int32(ny), dx=np.asarray(dx, dtype=fd),
        ghb=np.int32(1 if ghb else 0),
        t_isotropic=np.int32(1 if isotropic else 0),
        t_isotropic_value=np.asarray(t_isotropic_value, dtype=fd),
        thickness=np.asarray(thickness, dtype=fd),
        width=np.asarray(width, dtype=fd),
        r_truth=np.asarray(recharge, dtype=fd),
        seed=np.int32(seed), label=np.array(label),
        mf6_seconds=np.asarray(mf6_seconds, dtype=np.float64),
    )
    tmp.replace(truth_path)


def load_or_run_mf6_truth(truth_path, workspace, label, nx, ny, dx, ghb, isotropic,
                          t_isotropic_value, thickness, width, recharge_rate,
                          recharge_field, seed, hk_field, output_dtype, mf6_runner):
    cached = load_cached_mf6_truth(
        truth_path, label, nx, ny, dx, ghb, isotropic,
        t_isotropic_value, thickness, width, recharge_rate, seed,
    )
    if cached is not None:
        heads, mf6_seconds = cached
        print(f"Loading cached MF6 truth: {truth_path}")
        return heads, mf6_seconds, "cache"
    print(f"MF6 truth cache miss; running model for {label}: {truth_path}")
    heads, mf6_seconds = mf6_runner(
        nx=nx, ny=ny, grid_size=dx, nper=1, workspace=workspace,
        hk=hk_field, recharge=recharge_field, run=True, use_ghb=ghb,
    )
    save_mf6_truth(
        truth_path, heads, label, nx, ny, dx, ghb, isotropic,
        t_isotropic_value, thickness, width, recharge_rate, seed,
        float(mf6_seconds), output_dtype,
    )
    return np.asarray(heads, dtype=np.float64), float(mf6_seconds), "generated"


# ---------------------------------------------------------------------------
# Solve wrappers (one solver invocation per call; both start from the DEM)
# ---------------------------------------------------------------------------


def kcycle_solve(solver, dem):
    return solver.solve_multigrid_kcycle(
        max_cycles=200,
        nu_pre=2,
        nu_post=2,
        nu_coarse=2,
        omega=0.7,
        rel_tol=5.0e-7,
        abs_tol_min=5.0e-7,
        initial_head=dem,
        return_info=True,
        max_levels=6,
        check_every_no=5,
    )


def _mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 * 1024.0)


def main() -> None:
    ghb = bool(args.ghb)
    isotropic = bool(args.isotropic)
    selected = [c.strip() for c in args.cases.split(",") if c.strip()]
    width = DX
    truth_dir = data_store.joinpath("mf6_truth_npz")
    REF_DIR.mkdir(parents=True, exist_ok=True)

    results = {}

    for label in selected:
        cfg = GRID_CASES[label]
        nx, ny = int(cfg["nx"]), int(cfg["ny"])
        print("\n" + "=" * 70)
        print(f"[{args.mode}] case {label} (ghb={ghb}, isotropic={isotropic})")
        print("=" * 70)

        domain = _build_domain(nx=nx, ny=ny)
        dem = _build_dem(domain)
        active_mask = domain == 1

        if isotropic:
            T_field = np.full_like(domain, ISOTROPIC_T, dtype=np.float64)
        else:
            T_field = make_ugly_T_field(nx=nx, ny=ny, domain=domain, seed=int(T_SEED))
        R_field = np.full_like(domain, R_TRUTH, dtype=np.float64)

        # FP64 boundary/GHB master data (independent of solver build precision)
        (_, _, active64, bc_mask64, bc_values64, gh_mask64, gh_head64, _) = (
            build_truth_inputs(nx=nx, ny=ny, dx=DX, T_truth=T_field,
                               R_truth=R_field, use_ghb=ghb, width=width)
        )

        wp.synchronize_device(args.device)
        mem_before = wp.get_mempool_used_mem_high(args.device)

        with wds(
            nx=nx, ny=ny, dx=DX, device=args.device,
            use_ghb=ghb, solver_type="pcg", aq_thickness=THICKNESS,
        ) as solver:
            solver.build_from_truth_inputs(T_truth=T_field, R_truth=R_field, width=width)

            if args.mode == "mixed":
                session = MixedPrecisionDefectCorrectionSession(
                    solver,
                    bc_values_f64=bc_values64,
                    gh_head_f64=gh_head64 if ghb else None,
                    R_f64=R_field,
                    max_levels=6,
                )

                def solve_once():
                    return session.solve(
                        dem,
                        inner_kcycles=int(args.inner_kcycles),
                        max_outer=int(args.max_outer),
                        rel_tol=5.0e-7,
                        abs_tol_min=5.0e-7,
                    )
            else:
                def solve_once():
                    return kcycle_solve(solver, dem)

            t0 = time.perf_counter()
            head_cold, info_cold = solve_once()
            wp.synchronize_device(args.device)
            t1 = time.perf_counter()
            cold_seconds = t1 - t0

            t2 = time.perf_counter()
            head_warm, info = solve_once()
            wp.synchronize_device(args.device)
            t3 = time.perf_counter()
            warm_seconds = t3 - t2

            head = np.asarray(head_warm, dtype=np.float64)
            mem_after = wp.get_mempool_used_mem_high(args.device)

        wp.synchronize_device(args.device)
        gc.collect()
        wp.synchronize_device(args.device)

        # Consistency of cold/warm (both from DEM): should be (near-)identical
        cw = compare_head_fields(head_ref=np.asarray(head_cold, dtype=np.float64),
                                 head_warp=head, active_mask=active_mask)

        # FP64 Warp reference (cached per case/config)
        ref_path = REF_DIR / f"warp_ref_{label}_ghb_{ghb}_t_isotropic_{isotropic}.npz"
        if args.mode == "fp64":
            np.savez_compressed(ref_path, heads=head)
            vs_fp64 = {"max_abs_diff": 0.0, "rms_diff": 0.0}
        else:
            if ref_path.exists():
                ref = np.asarray(np.load(ref_path)["heads"], dtype=np.float64)
                vs_fp64 = compare_head_fields(head_ref=ref, head_warp=head,
                                              active_mask=active_mask)
            else:
                vs_fp64 = {"max_abs_diff": None, "rms_diff": None,
                           "note": "fp64 reference missing; run --mode fp64 first"}

        # MF6 truth (cache-first)
        ws = data_store.joinpath(
            f"Paper_mf6_truth_{label}_ghb_{ghb}_t_isotropic_{isotropic}"
        )
        mf6_path = build_mf6_truth_path(truth_dir, label, ghb, isotropic)
        mf_head, t_mf, mf6_source = load_or_run_mf6_truth(
            truth_path=mf6_path, workspace=ws, label=label, nx=nx, ny=ny, dx=DX,
            ghb=ghb, isotropic=isotropic, t_isotropic_value=ISOTROPIC_T,
            thickness=THICKNESS, width=width, recharge_rate=R_TRUTH,
            recharge_field=R_field, seed=int(T_SEED),
            hk_field=T_field / THICKNESS, output_dtype=np.dtype(np.float32),
            mf6_runner=make_mf_model,
        )
        vs_mf6 = compare_head_fields(head_ref=mf_head, head_warp=head,
                                     active_mask=active_mask)

        # Mass balance (FP64 host fields from the harness, solved head)
        bud = compute_mass_balance_budget(
            T_field=T_field, R_field=R_field, head=head,
            active=active64, bc_mask=bc_mask64, bc_values=bc_values64,
            dx=DX,
            gh_mask=gh_mask64 if ghb else None,
            gh_head=gh_head64 if ghb else None,
            gh_width=np.where(gh_mask64 != 0, width, 0.0) if ghb else None,
            gh_alpha=1.0, aq_thickness=THICKNESS,
        )
        try:
            budget_records = bud.to_dict("records")
        except AttributeError:
            budget_records = {k: (float(v) if np.isscalar(v) else str(v))
                              for k, v in bud.items()}

        info_out = {k: v for k, v in info.items() if k != "history"}
        if "history" in info:
            info_out["history"] = info["history"]

        results[label] = {
            "mode": args.mode,
            "nx": nx, "ny": ny,
            "ghb": ghb, "isotropic": isotropic,
            "starting_head": "original DEM for both timed solves (cold + warm)",
            "cold_start_vs_warm": cw,
            "timings": {
                "warp_seconds_cuda_cold_runtime": float(cold_seconds),
                "warp_seconds_cuda_warm_runtime": float(warm_seconds),
                "mf6_seconds": None if t_mf is None else float(t_mf),
            },
            "solver_info": info_out,
            "vs_fp64_warp": vs_fp64,
            "vs_mf6": vs_mf6,
            "accuracy_gate_max_abs_2e-4_pass": (
                None if vs_mf6.get("max_abs_diff") is None
                else bool(vs_mf6["max_abs_diff"] <= ACC_GATE_M)
            ),
            "mass_balance": budget_records,
            "memory": {
                "mempool_high_water_delta_mib": _mb(mem_after - mem_before),
                "mempool_high_water_mib": _mb(mem_after),
            },
            "mf6_result_source": str(mf6_source),
        }
        print(json.dumps({
            "case": label,
            "cold_s": round(cold_seconds, 3),
            "warm_s": round(warm_seconds, 3),
            "info": {k: v for k, v in info_out.items() if k != "history"},
            "vs_fp64_max": vs_fp64.get("max_abs_diff"),
            "vs_mf6_max": vs_mf6.get("max_abs_diff"),
            "vs_mf6_rms": vs_mf6.get("rms_diff"),
            "gate_pass": results[label]["accuracy_gate_max_abs_2e-4_pass"],
        }, indent=2, default=str))

    out_path = _REPO_ROOT / "working_tests" / (
        f"mixed_precision_results_{args.mode}_ghb_{ghb}_t_isotropic_{isotropic}.json"
    )
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
