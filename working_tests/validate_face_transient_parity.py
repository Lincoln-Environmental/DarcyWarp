#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Phase A parity check: face-array operator vs classic kernels on the
production 2D transient unconfined device fast path.

Runs the same small replay case twice through the production driver
(``unconfined_picard_kcycle``) with ``transient_face_operator_enabled``
True vs False and reports per-period accepted-head max-abs differences and
outer-iteration counts.  Gate: strict Picard acceptance on all periods for
BOTH modes and per-period head max-abs diff below ``--tol`` (default
1e-6 m).

Usage:
    python working_tests/validate_face_transient_parity.py [--nx 100 --ny 100 --n-periods 3]
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


def _run_case(*, spatial, recharge_rates, sy, ss, dt, n_periods, warm_start_head, device, face_enabled):
    return run_warp_transient_replay(
        spatial=spatial,
        recharge_rates=recharge_rates,
        sy=sy,
        ss=ss,
        dt=dt,
        n_periods=n_periods,
        device=device,
        solve_controls={"transient_face_operator_enabled": bool(face_enabled)},
        warm_start_mode=WARM_START_UNCONFINED_STEADY_MF6,
        warm_start_head=warm_start_head,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--n-periods", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tol", type=float, default=1.0e-6)
    args = parser.parse_args()

    case_setup = build_case_setup(nx=args.nx, ny=args.ny, n_periods=args.n_periods)
    artifact_path = ensure_case_artifact(case_setup)
    artifact = load_transient_artifact(artifact_path)
    spatial = spatial_fields_from_artifact(artifact)
    warm_start_head, warm_start_used = select_artifact_warm_start(
        artifact=artifact,
        spatial=spatial,
        warm_start_mode=WARM_START_UNCONFINED_STEADY_MF6,
    )
    recharge_rates = np.asarray(artifact["recharge_rates"], dtype=np.float64)
    sy = float(artifact["sy"])
    ss = float(artifact["ss"])
    dt = float(artifact["dt_days"])

    print(f"Face-operator parity: {args.nx}x{args.ny}, {args.n_periods} periods, "
          f"warm_start={warm_start_used}")

    results = {}
    for face_enabled in (False, True):
        label = "face" if face_enabled else "classic"
        print(f"--- running {label} path ---")
        results[label] = _run_case(
            spatial=spatial,
            recharge_rates=recharge_rates,
            sy=sy,
            ss=ss,
            dt=dt,
            n_periods=args.n_periods,
            warm_start_head=warm_start_head,
            device=args.device,
            face_enabled=face_enabled,
        )

    heads_classic = np.asarray(results["classic"]["heads_per_period"], dtype=np.float64)
    heads_face = np.asarray(results["face"]["heads_per_period"], dtype=np.float64)
    infos_classic = results["classic"]["period_infos"]
    infos_face = results["face"]["period_infos"]

    print("\nperiod | outer(classic) | outer(face) | strict(c) | strict(f) | max|dh| (m)")
    print("-------+----------------+-------------+-----------+-----------+-------------")
    all_strict = True
    max_diff_all = 0.0
    for p in range(args.n_periods):
        info_c = infos_classic[p]
        info_f = infos_face[p]
        strict_c = bool(info_c.get("strict_picard_convergence_passed", False))
        strict_f = bool(info_f.get("strict_picard_convergence_passed", False))
        all_strict = all_strict and strict_c and strict_f
        diff = float(np.max(np.abs(heads_face[p] - heads_classic[p])))
        max_diff_all = max(max_diff_all, diff)
        print(
            f"{p + 1:6d} | {int(info_c.get('outer_iterations', -1)):14d} | "
            f"{int(info_f.get('outer_iterations', -1)):11d} | "
            f"{str(strict_c):9s} | {str(strict_f):9s} | {diff:.3e}"
        )

    time_c = float(results["classic"]["total_time"])
    time_f = float(results["face"]["total_time"])
    print(f"\ntotal solve time: classic {time_c:.2f} s, face {time_f:.2f} s")
    print(f"worst per-period head max-abs diff: {max_diff_all:.3e} m (tol {args.tol:.1e})")

    ok = all_strict and max_diff_all <= float(args.tol)
    print("PARITY:", "PASS" if ok else "FAIL")
    if not all_strict:
        print("  strict Picard acceptance failed on at least one period/mode")
    if max_diff_all > float(args.tol):
        print(f"  head diff {max_diff_all:.3e} exceeds tolerance {args.tol:.1e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
