#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run the river-loss anisotropy sweep as an experiment entry point.

The reusable case builder lives in
``DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section``.  This script
owns only CLI parsing and study-output reporting.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# The default runner implementation is the FP64 face-array K-cycle. Honour an
# explicit precision choice, but make a bare CLI invocation select its required
# precision before DarcyWarp configuration is imported.
os.environ.setdefault("DARCY_FLOAT", "float64")

from DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section import (
    CaseResult,
    CrossSectionConfig,
    run_sweep,
)
from DARCY_WARP_PACKAGE.config import WP_FLOAT
import warp as wp


def _float_list(values: list[str]) -> list[float]:
    parsed = [float(value) for value in values]
    if not parsed:
        raise argparse.ArgumentTypeError("At least one ratio is required.")
    return parsed


def write_results(
    results: list[CaseResult],
    config: CrossSectionConfig,
    output_directory: Path,
) -> None:
    """Write sweep CSV and configuration metadata."""

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "river_loss_anisotropy_sweep.csv"
    fieldnames = (
        list(asdict(results[0]).keys())
        if results
        else list(CaseResult.__dataclass_fields__.keys())
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    metadata = {
        "config": asdict(config),
        "flow_units": "m3/day per metre out-of-plane width",
        "positive_channel_inflow": "flow from channel fixed-head cells into aquifer",
        "positive_outlet_outflow": "flow from aquifer into far-field fixed-head cells",
    }
    with (output_directory / "river_loss_cross_section_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)

    print(f"Wrote {csv_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="river_loss_cross_section.py",
        description="Run a DarcyWarp anisotropic river-loss x-z cross-section sweep.",
    )
    parser.add_argument(
        "--ratios",
        nargs="+",
        default=["1", "3", "10", "30", "100", "300", "1000"],
        help="Kh/Kv ratios to test.",
    )
    parser.add_argument(
        "--anisotropy-target",
        choices=("braidplain", "regional", "both"),
        default="both",
    )
    parser.add_argument(
        "--outlet-modes",
        nargs="+",
        choices=("full_depth", "lower_only"),
        default=("full_depth", "lower_only"),
    )
    parser.add_argument("--braidplain-kh", type=float, default=100.0)
    parser.add_argument("--regional-kh", type=float, default=10.0)
    parser.add_argument("--dx", type=float, default=10.0)
    parser.add_argument("--dz", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--implementation",
        choices=("classic", "fast"),
        default="fast",
        help="Use the FP64 face-array K-cycle (default) or the classic solver.",
    )
    parser.add_argument("--max-cycles", type=int, default=200)
    parser.add_argument("--max-levels", type=int, default=6)
    parser.add_argument("--min-coarse-n", type=int, default=1)
    parser.add_argument("--rel-tol", type=float, default=5.0e-5)
    parser.add_argument("--abs-tol-min", type=float, default=5.0e-5)
    parser.add_argument("--dh-rms-tol", type=float, default=1.0e-4)
    parser.add_argument("--check-every-no", type=int, default=1)
    parser.add_argument("--practical-mass-imbalance-tol", type=float, default=1.0e-2)
    parser.add_argument("--nu-pre", type=int, default=6)
    parser.add_argument("--nu-post", type=int, default=6)
    parser.add_argument("--nu-coarse", type=int, default=2)
    parser.add_argument("--omega", type=float, default=0.8)
    parser.add_argument("--line-omega", type=float, default=0.8)
    parser.add_argument("--line-sweeps-pre", type=int, default=1)
    parser.add_argument("--line-sweeps-post", type=int, default=1)
    parser.add_argument("--line-sweeps-coarse", type=int, default=1)
    parser.add_argument("--vertical-line-max-nz", type=int, default=128)
    parser.add_argument(
        "--robust-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry a non-converged solve with stronger K-cycle smoothing.",
    )
    parser.add_argument("--robust-nu-pre", type=int, default=13)
    parser.add_argument("--robust-nu-post", type=int, default=13)
    parser.add_argument("--robust-nu-coarse", type=int, default=3)
    parser.add_argument("--robust-omega", type=float, default=0.7)
    parser.add_argument(
        "--smoother",
        choices=("chebyshev", "jacobi", "vertical_line", "chebyshev_vertical_line"),
        default="chebyshev",
    )
    parser.add_argument("--output-directory", type=Path, default=Path("cross_section_results"))
    parser.add_argument("--save-heads", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.implementation == "fast" and WP_FLOAT is not wp.float64:
        raise SystemExit(
            "The default fast 3D river solver requires FP64. Re-run with "
            "DARCY_FLOAT=float64, or select --implementation classic."
        )
    ratios = _float_list(list(args.ratios))
    cfg = CrossSectionConfig(
        braidplain_kh=float(args.braidplain_kh),
        regional_kh=float(args.regional_kh),
        dx=float(args.dx),
        dz=float(args.dz),
        device=str(args.device),
        implementation=str(args.implementation),
        max_cycles=int(args.max_cycles),
        max_levels=int(args.max_levels),
        min_coarse_n=int(args.min_coarse_n),
        rel_tol=float(args.rel_tol),
        abs_tol_min=float(args.abs_tol_min),
        dh_rms_tol=float(args.dh_rms_tol),
        check_every_no=int(args.check_every_no),
        practical_mass_imbalance_tol=float(args.practical_mass_imbalance_tol),
        nu_pre=int(args.nu_pre),
        nu_post=int(args.nu_post),
        nu_coarse=int(args.nu_coarse),
        omega=float(args.omega),
        line_omega=float(args.line_omega),
        line_sweeps_pre=int(args.line_sweeps_pre),
        line_sweeps_post=int(args.line_sweeps_post),
        line_sweeps_coarse=int(args.line_sweeps_coarse),
        vertical_line_max_nz=int(args.vertical_line_max_nz),
        robust_retry_enabled=bool(args.robust_retry),
        robust_nu_pre=int(args.robust_nu_pre),
        robust_nu_post=int(args.robust_nu_post),
        robust_nu_coarse=int(args.robust_nu_coarse),
        robust_omega=float(args.robust_omega),
        smoother=str(args.smoother),
    )
    results = run_sweep(
        base_config=cfg,
        ratios=ratios,
        outlet_modes=args.outlet_modes,
        anisotropy_target=args.anisotropy_target,
        output_directory=args.output_directory,
        save_heads=bool(args.save_heads),
    )
    write_results(results, cfg, args.output_directory)


if __name__ == "__main__":
    main()
