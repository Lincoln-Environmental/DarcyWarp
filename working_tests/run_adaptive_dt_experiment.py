"""Experiment driver for adaptive-dt / step-1 convergence work.

Runs the production replay on a chosen artifact with chosen solve-control
overrides and prints a compact per-period table so experiments can be
compared at a glance.

Usage:
  conda run -n darcywarp python working_tests/run_adaptive_dt_experiment.py \
      --artifact DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined_1000x1000_10w/mf6_transient_heads.npz.lzma \
      --workspace /tmp/adt_exp/baseline_10w \
      --controls '{"adaptive_dt_enabled": false}'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from working_tests.transient_replay_settings import (  # noqa: E402
    PRODUCTION_RUN_MODE,
    default_run_config,
    default_solve_controls,
)
from working_tests.transient_replay_support import run_replay_from_artifact  # noqa: E402


def _load_period_infos(workspace: Path) -> list[dict]:
    """Recover per-period solver info dicts saved inside warp_transient_heads.npz."""
    npz_path = workspace.joinpath("warp_transient_heads.npz")
    if not npz_path.exists():
        return []
    import numpy as np

    with np.load(npz_path, allow_pickle=False) as data:
        raw = data.get("period_infos")
        if raw is None:
            return []
        infos = json.loads(str(raw.reshape(())))
    return [row for row in infos if isinstance(row, dict)]


def _fmt(v, spec="{:.3g}"):
    if v is None:
        return "-"
    try:
        return spec.format(float(v))
    except (TypeError, ValueError):
        return str(v)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--n-periods", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--controls",
        default="{}",
        help="JSON dict of solve-control overrides, or @path/to/controls.json",
    )
    args = parser.parse_args()

    controls_arg = args.controls
    if controls_arg.startswith("@"):
        overrides = json.loads(Path(controls_arg[1:]).read_text())
    else:
        overrides = json.loads(controls_arg)

    solve_controls = default_solve_controls()
    solve_controls.update(overrides)

    run_config = default_run_config(run_mode=PRODUCTION_RUN_MODE, device=args.device)

    t0 = time.perf_counter()
    summary = run_replay_from_artifact(
        artifact_path=args.artifact,
        workspace=args.workspace,
        device=args.device,
        diag_preconditioner_backend="device",
        solve_controls=solve_controls,
        n_periods=args.n_periods,
        run_config=run_config,
    )
    wall = time.perf_counter() - t0

    print("\n=== per-period ===")
    header = (
        f"{'per':>3} {'time_s':>8} {'outer':>6} {'sub':>4} {'retry':>6} {'pfb':>4} "
        f"{'strict':>7} {'pract':>6} {'dh_max':>9} {'hres_rms':>9} {'mb_class':>16}"
    )
    print(header)
    print("-" * len(header))
    period_rows = _load_period_infos(Path(args.workspace))
    mb_rows = {
        int(r.get("period", -1)): r for r in (summary.get("mass_balance", {}).get("per_period") or [])
    }
    for idx, row in enumerate(period_rows, start=1):
        mb = mb_rows.get(idx, {})
        print(
            f"{idx:>3} {_fmt(row.get('period_total_seconds'), '{:.1f}'):>8} "
            f"{_fmt(row.get('adaptive_dt_total_outer_iterations', row.get('outer_iterations')), '{:.0f}'):>6} "
            f"{_fmt(row.get('adaptive_dt_substep_count'), '{:.0f}'):>4} "
            f"{_fmt(row.get('adaptive_dt_retry_count'), '{:.0f}'):>6} "
            f"{_fmt(row.get('adaptive_dt_practical_fallback_count'), '{:.0f}'):>4} "
            f"{str(row.get('strict_picard_convergence_passed')):>7} "
            f"{str(row.get('practical_picard_acceptance_passed')):>6} "
            f"{_fmt(row.get('final_max_abs_head_change')):>9} "
            f"{_fmt(row.get('final_head_residual_rms')):>9} "
            f"{str(mb.get('mass_balance_class', '-')):>16}"
        )

    acc = summary.get("head_accuracy") or {}
    print("\n=== accuracy vs MF6 (sanity) ===")
    print(json.dumps(acc, indent=2, default=str)[:2000])

    prod = summary.get("production_acceptance") or {}
    print("\n=== production acceptance ===")
    print(json.dumps(prod, indent=2, default=str)[:2500])

    mb = summary.get("mass_balance") or {}
    print(f"\nmass_balance_class={mb.get('mass_balance_class')} passed={mb.get('mass_balance_passed')}")
    print(f"total_wall_s={wall:.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
