#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""A/B: steady unconfined Picard with classic vs fast inner K-cycle.

Methodology: each run_case uses ``do_double_solve=True`` — solve 1 is the
warmup (Warp kernel JIT compilation + CUDA-graph capture happen there) and
is excluded from the reported time; ``warp_benchmark_time`` is the SECOND
solve from the same initial condition.  MF6 runs once per grid (the second
impl reuses the fingerprint-matched artifact).

Gates (per grid): both impls converged with the strict status, RMSE and
max_abs vs MF6 below the runner tolerance, fast-vs-classic head agreement,
and identical MF6 truth across the two impl runs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

import working_tests.run_2d_unconfined_warp_vs_mf6 as R  # noqa: E402

STRICT_STATUS = "Nonlinear head-change and inner residual tolerances met."
MF6_TOL = 5.0e-4
FAST_CLASSIC_TOL = 5.0e-6  # hclose=1e-4 acceptance-basin width (documented)

TIMING_METHODOLOGY = (
    "do_double_solve=True: solve1 is the warmup (JIT + CUDA-graph capture) "
    "and is excluded; solve2_time/warp_benchmark_time is the timed solve "
    "from the same initial condition."
)


def run_ab(
    grids: list[tuple[int, int]] | tuple[tuple[int, int], ...] = ((500, 500), (1000, 1000)),
    workspace: str | Path = "/tmp/unc_inner_ab",
    device: str = "auto",
) -> tuple[dict, bool]:
    base = Path(workspace)
    base.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    failures: list[str] = []
    for nx, ny in grids:
        heads_by_impl: dict[str, np.ndarray] = {}
        mf6_heads_by_impl: dict[str, np.ndarray] = {}
        mf6_fingerprints: dict[str, str] = {}
        for impl in ("classic", "fast"):
            ws = base / f"grid_{nx}x{ny}_{impl}"
            row = R.run_case(
                nx=nx,
                ny=ny,
                dx=100.0,
                hydraulic_conductivity=100.0,
                recharge=1.0e-4,
                workspace=ws,
                device=device,
                diag_preconditioner_backend="device",
                check_every_no=5,
                do_run_mf6=True,
                do_run_warp=True,
                do_double_solve=True,  # solve1 = warmup, solve2 = timed
                inner_implementation=impl,
            )
            key = f"{nx}x{ny}_{impl}"
            comp = row.get("comparison") or {}
            report = row.get("convergence_report") or {}
            results[key] = {
                "solve2_time": row["warp_solve2_time"],
                "converged": row["solve2_converged"],
                "outer_iterations": row["solve2_outer_iterations"],
                "rmse_vs_mf6": comp.get("rmse"),
                "max_abs_vs_mf6": comp.get("max_abs_diff"),
                "mf6_engine_time": row["mf6_engine_time"],
                "status": report.get("status"),
            }
            print(json.dumps({key: results[key]}, indent=2))

            if not bool(row.get("solve2_converged")):
                failures.append(f"{key}: Warp solve did not converge")
            if str(report.get("status")) != STRICT_STATUS:
                failures.append(f"{key}: non-strict status {report.get('status')!r}")
            if comp.get("rmse") is None or float(comp["rmse"]) >= MF6_TOL:
                failures.append(f"{key}: RMSE vs MF6 {comp.get('rmse')} >= {MF6_TOL}")
            if comp.get("max_abs_diff") is None or float(comp["max_abs_diff"]) >= MF6_TOL:
                failures.append(f"{key}: max_abs vs MF6 {comp.get('max_abs_diff')} >= {MF6_TOL}")

            with np.load(ws / "warp_heads.npz", allow_pickle=False) as npz:
                heads_by_impl[impl] = np.asarray(npz["heads"], dtype=np.float64)
            with np.load(ws / "mf6_heads.npz", allow_pickle=False) as npz:
                mf6_heads_by_impl[impl] = np.asarray(npz["heads"], dtype=np.float64)
                mf6_fingerprints[impl] = str(np.asarray(npz["case_fingerprint"]).reshape(()))

        # The two impl runs must share the identical MF6 truth.
        if mf6_fingerprints.get("classic") != mf6_fingerprints.get("fast"):
            failures.append(f"{nx}x{ny}: MF6 artifact fingerprint differs between impls")
        elif not np.allclose(mf6_heads_by_impl["classic"], mf6_heads_by_impl["fast"],
                             rtol=0.0, atol=1.0e-10):
            failures.append(f"{nx}x{ny}: MF6 heads differ between impls despite equal fingerprints")

        fvc = float(np.max(np.abs(heads_by_impl["fast"] - heads_by_impl["classic"])))
        results[f"{nx}x{ny}_fast_vs_classic_max_abs"] = fvc
        if fvc >= FAST_CLASSIC_TOL:
            failures.append(f"{nx}x{ny}: fast-vs-classic diff {fvc:.3e} >= {FAST_CLASSIC_TOL}")

    results["timing_methodology"] = TIMING_METHODOLOGY
    results["failures"] = failures
    out = base / "ab_results.json"
    out.write_text(json.dumps(results, indent=2))

    print("\n==== A/B SUMMARY ====")
    for key, value in results.items():
        print(key, json.dumps(value) if not isinstance(value, str) else value)
    print(f"\nResults written to {out}")
    print("OVERALL:", "PASS" if not failures else f"FAIL ({len(failures)} failures)")
    return results, not failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/tmp/unc_inner_ab")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--grids",
        default="500x500,1000x1000",
        help="comma-separated NxM grid list (default: 500x500,1000x1000)",
    )
    args = parser.parse_args()
    grids = []
    for token in args.grids.split(","):
        nx_s, ny_s = token.strip().lower().split("x")
        grids.append((int(nx_s), int(ny_s)))
    _, ok = run_ab(grids=grids, workspace=args.workspace, device=args.device)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
