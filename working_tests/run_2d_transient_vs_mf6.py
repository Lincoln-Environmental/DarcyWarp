#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""
Build a MODFLOW 6 truth artifact for the 2D transient unconfined path.

The runner creates a single-layer convertible MF6 model with weekly stress
periods, seasonal recharge, specific yield, and specific storage. Its default
output is ``DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined/
mf6_transient_heads.npz.lzma``. That compressed artifact stores every input
needed for a future Warp-vs-MF6 transient replay: per-period heads, final heads,
initial heads, active and boundary masks, bottom/top elevations, hydraulic
conductivity, recharge time series, and storage parameters.

The current Warp transient tests in ``tests/test_2d_transient.py`` exercise the
2D confined and unconfined Warp code paths directly. This runner is the separate
external MF6 truth-generation step; a completed run is evidenced by the
compressed ``mf6_transient_heads.npz.lzma`` file plus the MF6 ``.hds`` file in
the sibling ``mf6`` workspace.

Usage:
    python working_tests/run_2d_transient_vs_mf6.py
"""

from __future__ import annotations

import io
import json
import lzma
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # working_tests on path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import flopy  # noqa: E402

from DARCY_WARP_PACKAGE.model_builder import (  # noqa: E402
    _build_dem,
    _build_dirichlet_boundary_mask,
    _build_domain,
    _create_chd_single_period,
    _model_bottom,
)
from DARCY_WARP_PACKAGE.project_base import data_store, require_mf6  # noqa: E402


# ---- defaults --------------------------------------------------------------
N_WEEKS = 52
DT_DAYS = 7.0
ANNUAL_RECHARGE_M = 0.3
DEFAULT_NX = DEFAULT_NY = 250
DEFAULT_DX = 100.0
DEFAULT_K = 100.0
DEFAULT_SY = 0.20           # specific yield (unconfined storage)
DEFAULT_SS = 1.0e-5         # specific storage (1/m), confined/saturated portion
DEFAULT_INIT_SAT_THICKNESS = 100.0
MF6_MODEL_NAME = "tr2d_truth"


def build_seasonal_recharge(
    n_weeks: int = N_WEEKS,
    annual_depth_m: float = ANNUAL_RECHARGE_M,
    peak_week: float = N_WEEKS / 2.0,
    floor: float = 0.05,
    dt_days: float = DT_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a winter-dominated recharge time series for transient MF6 runs.

    A single-peak cosine shape floored so summer recharge is small but
    positive, then scaled so the weekly DEPTHS sum to exactly
    ``annual_depth_m``. Returns ``(depths_m, rates_m_per_day)`` each of length
    ``n_weeks``; ``rate_k = depth_k / dt_days``.

    Parameters
    ----------
    annual_depth_m : total recharge depth applied over the year (m).
    peak_week : week index of maximum recharge (default mid-year).
    floor : minimum recharge as a fraction of the peak (summer floor).
    dt_days : days per stress period (used to convert depth -> rate).
    """
    weeks = np.arange(n_weeks, dtype=np.float64)
    shape = floor + (1.0 - floor) * 0.5 * (1.0 + np.cos(2.0 * np.pi * (weeks - peak_week) / n_weeks))
    depths = shape * (float(annual_depth_m) / float(shape.sum()))
    rates = depths / float(dt_days)
    return depths, rates


@dataclass(frozen=True)
class TransientCase:
    nx: int
    ny: int
    dx: float
    active: np.ndarray
    bc_mask: np.ndarray
    bc_values: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    hydraulic_conductivity: np.ndarray
    initial_head: np.ndarray
    sy: float
    ss: float
    n_weeks: int
    dt_days: float
    recharge_depths: np.ndarray   # (n_weeks,) m
    recharge_rates: np.ndarray    # (n_weeks,) m/day


def build_transient_unconfined_case(
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    dx: float = DEFAULT_DX,
    hydraulic_conductivity: float = DEFAULT_K,
    initial_saturated_thickness: float = DEFAULT_INIT_SAT_THICKNESS,
    sy: float = DEFAULT_SY,
    ss: float = DEFAULT_SS,
    n_weeks: int = N_WEEKS,
    annual_recharge_m: float = ANNUAL_RECHARGE_M,
    dt_days: float = DT_DAYS,
    workspace: str | Path | None = None,
) -> TransientCase:
    """
    Build the spatial and temporal inputs for the transient unconfined model.

    The spatial grid, boundary conditions, elevations, and hydraulic
    conductivity match the steady unconfined benchmark case, but this function
    builds them directly so MF6 truth generation does not require importing
    the Warp solver. The returned ``TransientCase`` is intentionally plain
    NumPy data so it can be used both by this MF6 generator and by future Warp
    replay/comparison code without importing Flopy.
    """
    if workspace is not None:
        Path(workspace).mkdir(parents=True, exist_ok=True)

    active = _build_domain(nx=int(nx), ny=int(ny)).astype(np.int32)
    top = np.asarray(_build_dem(active), dtype=np.float64)
    bottom = np.asarray(_model_bottom(top), dtype=np.float64)
    bc_bool = _build_dirichlet_boundary_mask(active)

    bc_mask = bc_bool.astype(np.int32)
    bc_values = np.zeros((int(ny), int(nx)), dtype=np.float64)
    bc_values[bc_bool] = top[bc_bool]

    k_field = np.full((int(ny), int(nx)), float(hydraulic_conductivity), dtype=np.float64)
    k_field[active == 0] = 0.0

    initial_head = bottom + max(float(initial_saturated_thickness), 0.1)
    initial_head = np.minimum(initial_head, top)
    initial_head[bc_bool] = bc_values[bc_bool]
    initial_head[active == 0] = 0.0

    depths, rates = build_seasonal_recharge(
        n_weeks=n_weeks,
        annual_depth_m=annual_recharge_m,
        dt_days=dt_days,
    )
    return TransientCase(
        nx=int(nx),
        ny=int(ny),
        dx=float(dx),
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        top=top,
        bottom=bottom,
        hydraulic_conductivity=k_field,
        initial_head=initial_head.astype(np.float64, copy=False),
        sy=float(sy),
        ss=float(ss),
        n_weeks=int(n_weeks),
        dt_days=float(dt_days),
        recharge_depths=depths,
        recharge_rates=rates,
    )


def _save_compressed_npz(out_path: Path, arrays: dict[str, np.ndarray], preset: int = 9) -> None:
    """
    Save an ``np.savez`` payload through stdlib LZMA compression.

    The repository already uses ``.npz.lzma`` fixtures for large truth arrays.
    This helper keeps the transient MF6 artifact in the same lossless format
    without adding another compression dependency.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    out_path.write_bytes(lzma.compress(buf.getvalue(), preset=preset))


def run_mf6_transient(
    case: TransientCase,
    out_path: str | Path | None = None,
    mf6_workspace: str | Path | None = None,
) -> Path:
    """
    Run MODFLOW 6 for one transient unconfined case and write the truth artifact.

    Parameters
    ----------
    case:
        Complete spatial, boundary, storage, and recharge inputs for the model.
    out_path:
        Optional compressed ``.npz.lzma`` output path. When omitted, the
        artifact is written under ``data/working_tests/mf6_transient_2d_unconfined``.
    mf6_workspace:
        Optional directory for MF6 input/output files. When omitted, a sibling
        ``mf6`` directory next to ``out_path`` is used.

    Returns
    -------
    Path
        Path to the compressed truth artifact containing per-period MF6 heads
        and all replay inputs.
    """
    if out_path is None:
        out_path = data_store.joinpath(
            "working_tests", "mf6_transient_2d_unconfined", "mf6_transient_heads.npz.lzma"
        )
    out_path = Path(out_path)
    mf6_ws = Path(mf6_workspace) if mf6_workspace is not None else out_path.parent.joinpath("mf6")
    mf6_ws.mkdir(parents=True, exist_ok=True)

    name = MF6_MODEL_NAME
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=str(require_mf6()),
        version="mf6",
        sim_ws=str(mf6_ws),
    )
    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=case.n_weeks,
        perioddata=[(case.dt_days, 1, 1.0)] * case.n_weeks,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, model_nam_file=f"{name}.nam", save_flows=True
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        outer_maximum=150,
        outer_dvclose=1.0e-6,
        inner_maximum=500,
        inner_dvclose=1.0e-8,
        rcloserecord=[1.0e-6, "RELATIVE_RCLOSE"],
        scaling_method="DIAGONAL",
    )
    sim.register_ims_package(ims, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=1,
        nrow=case.ny,
        ncol=case.nx,
        delr=case.dx,
        delc=case.dx,
        top=case.top,
        botm=case.bottom,
        idomain=case.active,
    )
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=case.initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=[1],                       # convertible -> unconfined
        k=case.hydraulic_conductivity,
        k33=case.hydraulic_conductivity,
        k33overk=False,
        save_specific_discharge=True,
        save_saturation=True,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        ss=case.ss,
        sy=case.sy,
        iconvert=1,
        transient={0: True},                 # all periods transient (inherited)
    )

    fixed_head_cells = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
    fixed_head_cells[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    chd_spd = _create_chd_single_period(boundary_heads=fixed_head_cells, active=case.active)
    flopy.mf6.ModflowGwfchd(gwf, pname="chd", stress_period_data=chd_spd, save_flows=True)

    # Per-period recharge: uniform-in-space, seasonal-in-time. MF6 reuses the
    # previous period's CHD; recharge is specified explicitly for every period.
    recharge_spd = {
        per: np.full((case.ny, case.nx), float(case.recharge_rates[per]), dtype=np.float64)
        for per in range(case.n_weeks)
    }
    flopy.mf6.ModflowGwfrcha(gwf, pname="recharge", recharge=recharge_spd)

    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "ALL")],
        head_filerecord=[f"{name}.hds"],
        budget_filerecord=[f"{name}.cbb"],
        printrecord=[],
    )

    t_total_start = time.perf_counter()
    sim.write_simulation(silent=True)
    t_engine_start = time.perf_counter()
    ok, _ = sim.run_simulation(silent=True, report=False)
    engine_time = time.perf_counter() - t_engine_start
    total_time = time.perf_counter() - t_total_start
    if not ok:
        raise RuntimeError("MF6 transient run failed.")

    hds_path = mf6_ws.joinpath(f"{name}.hds")
    heads_all = flopy.utils.HeadFile(str(hds_path)).get_alldata()  # (ntimes, nlay, ny, nx)
    heads_per_period = np.asarray(heads_all[:, 0, :, :], dtype=np.float64)  # (n_weeks, ny, nx)

    arrays = {
        "heads_per_period": heads_per_period,
        "heads_final": heads_per_period[-1],
        "initial_head": np.asarray(case.initial_head, dtype=np.float64),
        "active": np.asarray(case.active, dtype=np.int32),
        "bc_mask": np.asarray(case.bc_mask, dtype=np.int32),
        "bc_values": np.asarray(case.bc_values, dtype=np.float64),
        "top": np.asarray(case.top, dtype=np.float64),
        "bottom": np.asarray(case.bottom, dtype=np.float64),
        "k_field": np.asarray(case.hydraulic_conductivity, dtype=np.float64),
        "recharge_depths": np.asarray(case.recharge_depths, dtype=np.float64),
        "recharge_rates": np.asarray(case.recharge_rates, dtype=np.float64),
        "sy": np.asarray(case.sy, dtype=np.float64),
        "ss": np.asarray(case.ss, dtype=np.float64),
        "nx": np.asarray(case.nx, dtype=np.int32),
        "ny": np.asarray(case.ny, dtype=np.int32),
        "dx": np.asarray(case.dx, dtype=np.float64),
        "n_weeks": np.asarray(case.n_weeks, dtype=np.int32),
        "dt_days": np.asarray(case.dt_days, dtype=np.float64),
        "engine_time": np.asarray(engine_time, dtype=np.float64),
        "total_time": np.asarray(total_time, dtype=np.float64),
        "provenance": np.asarray(json.dumps({
            "kind": "2d_unconfined_transient_mf6_truth",
            "n_weeks": case.n_weeks,
            "dt_days": case.dt_days,
            "annual_recharge_m": float(case.recharge_depths.sum()),
            "sy": case.sy,
            "ss": case.ss,
            "hydraulic_conductivity": case.hydraulic_conductivity.flat[0],
            "initial_saturated_thickness": DEFAULT_INIT_SAT_THICKNESS,
            "generator": "working_tests/run_2d_transient_vs_mf6.py",
        }, default=str)),
    }
    _save_compressed_npz(out_path, arrays)

    annual = float(case.recharge_depths.sum())
    print(f"MF6 transient heads saved to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(
        f"  n_weeks={case.n_weeks} dt={case.dt_days}d  recharge annual sum={annual:.4f} m "
        f"(target {ANNUAL_RECHARGE_M}); Sy={case.sy} Ss={case.ss}"
    )
    print(f"  MF6 total time: {total_time:.2f}s (engine {engine_time:.2f}s)")
    finite = np.isfinite(heads_per_period)
    print(
        f"  heads per-period shape={heads_per_period.shape} "
        f"range=[{np.nanmin(heads_per_period):.3f}, {np.nanmax(heads_per_period):.3f}] m "
        f"(all finite={bool(finite.all())})"
    )
    return out_path


def main(
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    dx: float = DEFAULT_DX,
    hydraulic_conductivity: float = DEFAULT_K,
    sy: float = DEFAULT_SY,
    ss: float = DEFAULT_SS,
    n_weeks: int = N_WEEKS,
    annual_recharge_m: float = ANNUAL_RECHARGE_M,
    out_path: str | Path | None = None,
) -> Path:
    """
    Build the default transient unconfined MF6 case and write its truth file.

    Parameters mirror the user-adjustable case settings in the module defaults.
    The returned path is the compressed truth artifact created by
    ``run_mf6_transient``.
    """
    case = build_transient_unconfined_case(
        nx=nx,
        ny=ny,
        dx=dx,
        hydraulic_conductivity=hydraulic_conductivity,
        sy=sy,
        ss=ss,
        n_weeks=n_weeks,
        annual_recharge_m=annual_recharge_m,
    )
    print(f"Transient 2D unconfined MF6 truth: {nx}x{ny}, {n_weeks} weekly steps, K={hydraulic_conductivity}")
    return run_mf6_transient(case, out_path=out_path)


if __name__ == "__main__":
    # Configuration parameters
    nx = DEFAULT_NX
    ny = DEFAULT_NY
    dx = DEFAULT_DX
    hydraulic_conductivity = DEFAULT_K
    sy = DEFAULT_SY
    ss = DEFAULT_SS
    n_weeks = N_WEEKS
    annual_recharge_m = ANNUAL_RECHARGE_M
    out_path = None

    main(
        nx=nx,
        ny=ny,
        dx=dx,
        hydraulic_conductivity=hydraulic_conductivity,
        sy=sy,
        ss=ss,
        n_weeks=n_weeks,
        annual_recharge_m=annual_recharge_m,
        out_path=out_path,
    )
