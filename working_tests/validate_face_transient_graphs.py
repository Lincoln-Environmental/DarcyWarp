#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Phase B equivalence check: CUDA-graph replay vs eager launches on the
face-operator 2D transient unconfined device fast path.

Runs the same small replay case twice through the production driver
(``unconfined_picard_kcycle``) with ``transient_face_graphs_enabled``
True vs False (face operator enabled in both) and reports per-period
accepted-head max-abs differences and outer-iteration counts.  Gate:
strict Picard acceptance on all periods for BOTH modes, identical outer
counts, per-period head max-abs diff below ``--tol`` (default 1e-9 m —
graph replay runs the identical kernels in the identical order, so heads
should match to round-off/bit level), and at least one captured K-cycle
and refresh graph reported in graph mode.

Usage:
    python working_tests/validate_face_transient_graphs.py [--nx 100 --ny 100 --n-periods 3]
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


def _run_case(*, spatial, recharge_rates, sy, ss, dt, n_periods, warm_start_head, device, graphs_enabled):
    return run_warp_transient_replay(
        spatial=spatial,
        recharge_rates=recharge_rates,
        sy=sy,
        ss=ss,
        dt=dt,
        n_periods=n_periods,
        device=device,
        solve_controls={
            "transient_face_operator_enabled": True,
            "transient_face_graphs_enabled": bool(graphs_enabled),
        },
        warm_start_mode=WARM_START_UNCONFINED_STEADY_MF6,
        warm_start_head=warm_start_head,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=100)
    parser.add_argument("--ny", type=int, default=100)
    parser.add_argument("--n-periods", type=int, default=3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tol", type=float, default=1.0e-9)
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

    print(f"Face-graph equivalence: {args.nx}x{args.ny}, {args.n_periods} periods, "
          f"warm_start={warm_start_used}")

    results = {}
    for graphs_enabled in (False, True):
        label = "graph" if graphs_enabled else "eager"
        print(f"--- running face {label} path ---")
        results[label] = _run_case(
            spatial=spatial,
            recharge_rates=recharge_rates,
            sy=sy,
            ss=ss,
            dt=dt,
            n_periods=args.n_periods,
            warm_start_head=warm_start_head,
            device=args.device,
            graphs_enabled=graphs_enabled,
        )

    heads_eager = np.asarray(results["eager"]["heads_per_period"], dtype=np.float64)
    heads_graph = np.asarray(results["graph"]["heads_per_period"], dtype=np.float64)
    infos_eager = results["eager"]["period_infos"]
    infos_graph = results["graph"]["period_infos"]

    print("\nperiod | outer(eager) | outer(graph) | strict(e) | strict(g) | max|dh| (m)")
    print("-------+--------------+--------------+-----------+-----------+-------------")
    all_strict = True
    same_outer = True
    max_diff_all = 0.0
    graphs_reported = False
    for p in range(args.n_periods):
        info_e = infos_eager[p]
        info_g = infos_graph[p]
        strict_e = bool(info_e.get("strict_picard_convergence_passed", False))
        strict_g = bool(info_g.get("strict_picard_convergence_passed", False))
        all_strict = all_strict and strict_e and strict_g
        outer_e = int(info_e.get("outer_iterations", -1))
        outer_g = int(info_g.get("outer_iterations", -1))
        same_outer = same_outer and (outer_e == outer_g)
        kc_graphs = int(info_g.get("transient_face_kcycle_graph_count", 0))
        rf_graphs = int(info_g.get("transient_face_refresh_graph_count", 0))
        graphs_reported = graphs_reported or (kc_graphs >= 1 and rf_graphs >= 1)
        diff = float(np.max(np.abs(heads_graph[p] - heads_eager[p])))
        max_diff_all = max(max_diff_all, diff)
        print(
            f"{p + 1:6d} | {outer_e:12d} | {outer_g:12d} | "
            f"{str(strict_e):9s} | {str(strict_g):9s} | {diff:.3e}"
        )

    info_g0 = infos_graph[-1]
    print(f"\ngraph mode flags: transient_face_graphs="
          f"{info_g0.get('transient_face_graphs')}, "
          f"kcycle_graphs={info_g0.get('transient_face_kcycle_graph_count')}, "
          f"refresh_graphs={info_g0.get('transient_face_refresh_graph_count')}")
    time_e = float(results["eager"]["total_time"])
    time_g = float(results["graph"]["total_time"])
    print(f"total solve time: eager {time_e:.2f} s, graph {time_g:.2f} s")
    print(f"worst per-period head max-abs diff: {max_diff_all:.3e} m (tol {args.tol:.1e})")

    ok = all_strict and same_outer and graphs_reported and max_diff_all <= float(args.tol)
    print("EQUIVALENCE:", "PASS" if ok else "FAIL")
    if not all_strict:
        print("  strict Picard acceptance failed on at least one period/mode")
    if not same_outer:
        print("  outer-iteration counts differ between graph and eager modes")
    if not graphs_reported:
        print("  graph mode reported no captured graphs (expected >= 1 kcycle and >= 1 refresh)")
    if max_diff_all > float(args.tol):
        print(f"  head diff {max_diff_all:.3e} exceeds tolerance {args.tol:.1e}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
