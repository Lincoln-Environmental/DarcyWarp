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
        "flow_units": "m3/day over the configured out_of_plane_width",
        "flow_normalization": "not normalized; divide reported flows by out_of_plane_width for m2/day",
        "section_scope": "half-channel, one-bank cross-section",
        "positive_channel_inflow": "flow from channel fixed-head cells into aquifer",
        "positive_outlet_outflow": "flow from aquifer into far-field fixed-head cells",
        "sweep": {
            "ratios": list(dict.fromkeys(result.varied_ratio for result in results)),
            "outlet_modes": list(dict.fromkeys(result.outlet_mode for result in results)),
            "anisotropy_targets": list(dict.fromkeys(result.anisotropy_target for result in results)),
        },
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
    parser.add_argument(
        "--solver",
        choices=("kcycle", "chebyshev"),
        default="kcycle",
        help="3D backend: fast implementation supports kcycle only; classic supports both.",
    )
    parser.add_argument("--max-cycles", type=int, default=200)
    parser.add_argument("--max-levels", type=int, default=6)
    parser.add_argument("--min-coarse-n", type=int, default=1)
    parser.add_argument("--rel-tol", type=float, default=5.0e-5)
    parser.add_argument("--abs-tol-min", type=float, default=5.0e-5)
    parser.add_argument("--dh-rms-tol", type=float, default=1.0e-4)
    parser.add_argument("--check-every-no", type=int, default=1)
    parser.add_argument(
        "--practical-mass-imbalance-tol",
        type=float,
        default=2.0e-6,
        help="Maximum accepted relative model budget imbalance (default: 2e-6).",
    )
    parser.add_argument("--nu-pre", type=int, default=6)
    parser.add_argument("--nu-post", type=int, default=6)
    parser.add_argument("--nu-coarse", type=int, default=2)
    parser.add_argument("--omega", type=float, default=0.8)
    parser.add_argument("--line-omega", type=float, default=None, help="Classic-only vertical-line damping.")
    parser.add_argument("--line-sweeps-pre", type=int, default=None, help="Classic-only pre-smoothing line sweeps.")
    parser.add_argument("--line-sweeps-post", type=int, default=None, help="Classic-only post-smoothing line sweeps.")
    parser.add_argument("--line-sweeps-coarse", type=int, default=None, help="Classic-only coarse line sweeps.")
    parser.add_argument("--vertical-line-max-nz", type=int, default=None, help="Classic-only maximum vertical line length.")
    parser.add_argument(
        "--robust-retry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Retry a failed convergence or budget gate with tighter solver controls.",
    )
    parser.add_argument("--robust-max-cycles", type=int, default=800)
    parser.add_argument("--robust-nu-pre", type=int, default=13)
    parser.add_argument("--robust-nu-post", type=int, default=13)
    parser.add_argument("--robust-nu-coarse", type=int, default=3)
    parser.add_argument("--robust-omega", type=float, default=0.7)
    parser.add_argument(
        "--robust-rel-tol",
        type=float,
        default=1.0e-10,
        help="Retry relative residual target; deliberately tighter than the economical first solve.",
    )
    parser.add_argument(
        "--robust-abs-tol-min",
        type=float,
        default=1.0e-10,
        help="Retry absolute residual floor; forces useful work from a warm-started retry.",
    )
    parser.add_argument("--robust-dh-rms-tol", type=float, default=1.0e-8)
    parser.add_argument(
        "--smoother",
        choices=("chebyshev", "jacobi", "vertical_line", "chebyshev_vertical_line"),
        default="chebyshev",
        help="Point smoothers (jacobi/chebyshev) apply to fast; line smoothers require classic.",
    )
    parser.add_argument("--output-directory", type=Path, default=Path("cross_section_results"))
    parser.add_argument("--save-heads", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.implementation == "fast":
        if args.solver != "kcycle":
            parser.error("--implementation fast supports --solver kcycle only.")
        if args.smoother not in {"jacobi", "chebyshev"}:
            parser.error("--implementation fast supports --smoother jacobi or chebyshev only.")
        if (
            args.line_omega is not None
            or args.line_sweeps_pre is not None
            or args.line_sweeps_post is not None
            or args.line_sweeps_coarse is not None
            or args.vertical_line_max_nz is not None
        ):
            parser.error("line-relaxation options apply only to --implementation classic.")
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
        solver=str(args.solver),
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
        line_omega=0.8 if args.line_omega is None else float(args.line_omega),
        line_sweeps_pre=1 if args.line_sweeps_pre is None else int(args.line_sweeps_pre),
        line_sweeps_post=1 if args.line_sweeps_post is None else int(args.line_sweeps_post),
        line_sweeps_coarse=1 if args.line_sweeps_coarse is None else int(args.line_sweeps_coarse),
        vertical_line_max_nz=128 if args.vertical_line_max_nz is None else int(args.vertical_line_max_nz),
        robust_retry_enabled=bool(args.robust_retry),
        robust_max_cycles=int(args.robust_max_cycles),
        robust_nu_pre=int(args.robust_nu_pre),
        robust_nu_post=int(args.robust_nu_post),
        robust_nu_coarse=int(args.robust_nu_coarse),
        robust_omega=float(args.robust_omega),
        robust_rel_tol=float(args.robust_rel_tol),
        robust_abs_tol_min=float(args.robust_abs_tol_min),
        robust_dh_rms_tol=float(args.robust_dh_rms_tol),
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
