#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Phase C validation: FP32 inner correction (mixed precision) vs FP64 on
the transient unconfined device fast path.

Runs the same small replay case through the production driver with
``transient_mixed_precision_enabled`` False (FP64 inner) vs True (FP32
inner correction).  Gates: strict Picard acceptance on all periods for
BOTH modes (practical acceptance firing counts as failure here), outer
counts within +2 per period, per-period accepted-head max-abs diff below
``--tol`` (default 1e-4 m), and the mixed run reports
``transient_mixed_precision=True``.

Usage:
    python working_tests/validate_transient_mixed.py [--nx 100 --ny 100 --n-periods 3]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from working_tests.run_2d_transient_warp_replay import (  # noqa: E402
    build_case_setup,
    ensure_case_artifact,
)
from working_tests.transient_artifacts import (  # noqa: E402
    WARM_START_UNCONFINED_STEADY_MF6,
    load_transient_artifact,
    select_artifact_warm_start,
    spatial_fields_from_artifact,
)
from working_tests.transient_replay_support import run_warp_transient_replay  # noqa: E402


def _run_case(*, spatial, recharge_rates, sy, ss, dt, n_periods, warm_start_head, device, mixed):
    return run_warp_transient_replay(
        spatial=spatial,
        recharge_rates=recharge_rates,
        sy=sy,
        ss=ss,
        dt=dt,
        n_periods=n_periods,
        device=device,
        solve_controls={"transient_mixed_precision_enabled": bool(mixed)},
        warm_start_mode=WARM_START_UNCONFINED_STEADY_MF6,
        warm_start_head=warm_start_head,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--n-periods", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--t-field-kind", default="ugly_t", choices=["ugly_t", "homogeneous"])
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--max-outer-drift", type=int, default=2)
    args = parser.parse_args()

    case_setup = build_case_setup(nx=args.nx, ny=args.ny, n_periods=args.n_periods,
                                  t_field_kind=args.t_field_kind)
    artifact_path = ensure_case_artifact(case_setup)
    artifact = load_transient_artifact(artifact_path)
    spatial = spatial_fields_from_artifact(artifact)
    warm_start_head, warm_start_used = select_artifact_warm_start(
        artifact=artifact, spatial=spatial,
        warm_start_mode=WARM_START_UNCONFINED_STEADY_MF6,
    )
    recharge_rates = np.asarray(artifact["recharge_rates"], dtype=np.float64)
    sy = float(artifact["sy"])
    ss = float(artifact["ss"])
    dt = float(artifact["dt_days"])

    print(f"Mixed-precision validation: {args.nx}x{args.ny}, {args.n_periods} periods, "
          f"warm_start={warm_start_used}")

    results = {}
    for mixed in (False, True):
        label = "mixed" if mixed else "fp64"
        print(f"--- running {label} inner solve ---")
        results[label] = _run_case(
            spatial=spatial, recharge_rates=recharge_rates, sy=sy, ss=ss, dt=dt,
            n_periods=args.n_periods, warm_start_head=warm_start_head,
            device=args.device, mixed=mixed,
        )

    heads_f64 = np.asarray(results["fp64"]["heads_per_period"], dtype=np.float64)
    heads_mx = np.asarray(results["mixed"]["heads_per_period"], dtype=np.float64)
    infos_f64 = results["fp64"]["period_infos"]
    infos_mx = results["mixed"]["period_infos"]

    print("\nperiod | outer(fp64) | outer(mixed) | strict(f) | strict(m) | practical(m) | max|dh| (m)")
    print("-------+-------------+--------------+-----------+-----------+--------------+-------------")
    all_strict = True
    no_practical = True
    outer_ok = True
    max_diff_all = 0.0
    for p in range(args.n_periods):
        info_f = infos_f64[p]
        info_m = infos_mx[p]
        strict_f = bool(info_f.get("strict_picard_convergence_passed", False))
        strict_m = bool(info_m.get("strict_picard_convergence_passed", False))
        practical_m = bool(info_m.get("practical_picard_acceptance_passed", False))
        retries = int(info_m.get("adaptive_dt_retry_count", 0))
        all_strict = all_strict and strict_f and strict_m
        no_practical = no_practical and (not practical_m) and retries == 0
        outer_f = int(info_f.get("outer_iterations", -1))
        outer_m = int(info_m.get("outer_iterations", -1))
        outer_ok = outer_ok and (outer_m <= outer_f + args.max_outer_drift)
        diff = float(np.max(np.abs(heads_mx[p] - heads_f64[p])))
        max_diff_all = max(max_diff_all, diff)
        print(
            f"{p + 1:6d} | {outer_f:11d} | {outer_m:12d} | "
            f"{str(strict_f):9s} | {str(strict_m):9s} | {str(practical_m):12s} | {diff:.3e}"
        )

    mixed_flag = bool(infos_mx[-1].get("transient_mixed_precision", False))
    time_f = float(results["fp64"]["total_time"])
    time_m = float(results["mixed"]["total_time"])
    print(f"\nmixed flag reported: {mixed_flag}")
    print(f"total solve time: fp64 {time_f:.2f} s, mixed {time_m:.2f} s")
    print(f"worst per-period head max-abs diff: {max_diff_all:.3e} m (tol {args.tol:.1e})")

    ok = all_strict and no_practical and outer_ok and mixed_flag and max_diff_all <= float(args.tol)
    print("MIXED VALIDATION:", "PASS" if ok else "FAIL")
    if not all_strict:
        print("  strict Picard acceptance failed on at least one period/mode")
    if not no_practical:
        print("  practical acceptance or adaptive-dt retry fired in mixed mode")
    if not outer_ok:
        print(f"  outer counts drifted by more than +{args.max_outer_drift}")
    if not mixed_flag:
        print("  mixed run did not report transient_mixed_precision=True")
    if max_diff_all > float(args.tol):
        print(f"  head diff {max_diff_all:.3e} exceeds tolerance {args.tol:.1e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
