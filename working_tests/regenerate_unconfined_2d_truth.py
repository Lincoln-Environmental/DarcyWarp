#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""
Build compressed 2D-unconfined truth fixtures from an already-completed
MF6-vs-Warp grid benchmark.

For each grid it bundles the MF6 reference heads + the deterministic Warp case
inputs + the exact solver settings into one ``.npz.lzma`` fixture
(lossless float64, stdlib lzma preset 9).  The resulting fixtures let
``tests/test_unconfined_2d_truth.py`` re-run Warp and check agreement against
MF6 without needing the MF6 binary or re-running the (slow) MF6 solves.

Usage:
    python working_tests/regenerate_unconfined_2d_truth.py

Inputs (already on disk from run_2d_unconfined_warp_vs_mf6.py):
    <benchmark_dir>/grid_{nx:04d}x{ny:04d}/mf6_heads.npz
    <benchmark_dir>/grid_{nx:04d}x{ny:04d}/warp_heads.npz
    <benchmark_dir>/grid_{nx:04d}x{ny:04d}/unconfined_benchmark_summary.json

Output:
    tests/fixtures/unconfined_2d/truth_{nx}x{ny}.npz.lzma
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["DARCY_FLOAT"] = "float64"

from DARCY_WARP_PACKAGE.unconfined_truth_io import (  # noqa: E402
    save_truth_artifact,
)

# run_2d_unconfined_warp_vs_mf6 lives in this directory (working_tests/).
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_2d_unconfined_warp_vs_mf6 import build_simple_unconfined_case  # noqa: E402

from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402


# ---- configuration (must match the benchmark run that produced the inputs) ----
BENCHMARK_DIR = data_store.joinpath(
    "working_tests", "mf6_vs_warp_2d_unconfined_grid_benchmark"
)
GRID_SIZES = [50, 100, 250, 500, 1000, 2000, 3000]
FIXTURE_DIR = REPO_ROOT.joinpath("tests", "fixtures", "unconfined_2d")

# Case parameters used by run_2d_unconfined_warp_vs_mf6.__main__:
DX = 100.0
HYDRAULIC_CONDUCTIVITY = 100.0
RECHARGE = 1.0e-4
INITIAL_SATURATED_THICKNESS = 100.0

# solve() kwargs that are arrays (excluded from the scalar settings dict).
_ARRAY_SOLVE_KEYS = {"K_field", "zbot_field", "ztop_field", "initial_head"}


def _load_json_summary(grid_dir: Path) -> dict:
    summary_path = grid_dir.joinpath("unconfined_benchmark_summary.json")
    if not summary_path.exists():
        return {}
    with summary_path.open("r") as f:
        return json.load(f)


def _scalar_solve_settings(warp_npz_path: Path) -> dict:
    """Read solve2_settings from the stored Warp run, drop array-valued keys."""
    with np.load(warp_npz_path, allow_pickle=False) as data:
        if "solve2_settings" not in data.files:
            raise KeyError(f"{warp_npz_path} has no 'solve2_settings' array")
        raw = str(np.asarray(data["solve2_settings"]).reshape(()))
    settings = json.loads(raw)
    return {k: v for k, v in settings.items() if k not in _ARRAY_SOLVE_KEYS}


def build_one_fixture(nx: int, ny: int, out_dir: Path = FIXTURE_DIR) -> Path:
    grid_dir = BENCHMARK_DIR.joinpath(f"grid_{nx:04d}x{ny:04d}")
    mf6_path = grid_dir.joinpath("mf6_heads.npz")
    warp_path = grid_dir.joinpath("warp_heads.npz")
    if not mf6_path.exists():
        raise FileNotFoundError(f"MF6 heads not found: {mf6_path}")
    if not warp_path.exists():
        raise FileNotFoundError(f"Warp heads not found: {warp_path}")

    with np.load(mf6_path, allow_pickle=False) as mf6:
        heads = np.asarray(mf6["heads"], dtype=np.float64)
        dx = float(mf6["dx"])

    summary = _load_json_summary(grid_dir)
    diag_preconditioner_backend = summary.get("diag_preconditioner_backend", "device")
    solve_settings = _scalar_solve_settings(warp_path)

    # Rebuild the exact deterministic case inputs used for the Warp solve.
    case = build_simple_unconfined_case(
        nx=nx,
        ny=ny,
        dx=dx,
        hydraulic_conductivity=HYDRAULIC_CONDUCTIVITY,
        recharge=RECHARGE,
        initial_saturated_thickness=INITIAL_SATURATED_THICKNESS,
    )

    # Derived Warp inputs (mirrors run_warp_unconfined exactly).
    initial_transmissivity = case.hydraulic_conductivity * np.maximum(
        case.initial_head - case.bottom, 0.1
    )
    initial_transmissivity[case.active == 0] = 0.0
    rhs_recharge = np.asarray(case.recharge, dtype=np.float64)

    constructor_settings = {
        "nx": int(nx),
        "ny": int(ny),
        "dx": float(dx),
        "solver_type": "kcycle",
        "diag_preconditioner_backend": str(diag_preconditioner_backend),
    }
    provenance = {
        "kind": "2d_unconfined_mf6_truth",
        "grid": f"{nx}x{ny}",
        "hydraulic_conductivity": HYDRAULIC_CONDUCTIVITY,
        "recharge": RECHARGE,
        "initial_saturated_thickness": INITIAL_SATURATED_THICKNESS,
        "darcy_float": os.environ.get("DARCY_FLOAT", "float64"),
        "lzma_preset": 9,
        "source_mf6_heads": str(mf6_path),
        "source_warp_heads": str(warp_path),
        "mf6_engine_time": summary.get("mf6_engine_time"),
        "original_warp_vs_mf6_max_abs_diff": (summary.get("comparison") or {}).get(
            "max_abs_diff"
        ),
        "generator": "working_tests/regenerate_unconfined_2d_truth.py",
    }

    out_path = out_dir.joinpath(f"truth_{nx}x{ny}.npz.lzma")
    save_truth_artifact(
        out_path,
        heads=heads,
        active=case.active,
        bc_mask=case.bc_mask,
        bc_values=case.bc_values,
        top=case.top,
        bottom=case.bottom,
        k_field=case.hydraulic_conductivity,
        recharge=case.recharge,
        initial_head=case.initial_head,
        initial_transmissivity=initial_transmissivity,
        rhs_recharge=rhs_recharge,
        solve_settings=solve_settings,
        constructor_settings=constructor_settings,
        provenance=provenance,
    )
    return out_path


def main(grid_sizes: list[int] = GRID_SIZES) -> list[Path]:
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    print(f"Writing truth fixtures to {FIXTURE_DIR}")
    print("-" * 72)
    for n in grid_sizes:
        out_path = build_one_fixture(n, n)
        size_mb = out_path.stat().st_size / 1e6
        print(f"  grid {n:>4}x{n:<4} -> {out_path.name}  ({size_mb:6.2f} MB)")
        written.append(out_path)
    print("-" * 72)
    total = sum(p.stat().st_size for p in written) / 1e6
    biggest = max(p.stat().st_size for p in written) / 1e6
    print(f"Total: {total:.2f} MB across {len(written)} fixtures; largest = {biggest:.2f} MB")
    return written


if __name__ == "__main__":
    main()
