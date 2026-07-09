from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["DARCY_FLOAT"] = "float64"
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
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver  # noqa: E402


DEFAULT_DH_TOL = 1.0e-4
DEFAULT_RESIDUAL_FLOOR_TOL = 1.0e-4
DEFAULT_MF6_AGREEMENT_TOL = 5.0e-4
DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN = 1.0e-4
BENCHMARK_GRID_SIZES = [(250, 250), (500, 500), (1000, 1000), (2000, 2000), (3000, 3000)]


@dataclass(frozen=True)
class Unconfined2DCase:
    nx: int
    ny: int
    dx: float
    workspace: Path
    active: np.ndarray
    bc_mask: np.ndarray
    bc_values: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    hydraulic_conductivity: np.ndarray
    recharge: np.ndarray
    initial_head: np.ndarray


def _warp_device(preferred: str = "cuda:0") -> str:
    import warp as wp

    if preferred != "auto":
        return preferred
    try:
        return "cuda:0" if wp.is_cuda_available() else "cpu"
    except AttributeError:
        return "cuda:0"


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _load_npz_json(npz_path: Path, name: str) -> dict | list:
    if not npz_path.exists():
        return {}
    with np.load(npz_path, allow_pickle=False) as data:
        if name not in data:
            return {}
        raw = str(np.asarray(data[name]).reshape(()))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_info": raw}


def _load_npz_scalar(npz_path: Path, name: str, default: float | None = None) -> float | None:
    if not npz_path.exists():
        return default
    with np.load(npz_path, allow_pickle=False) as data:
        if name not in data:
            return default
        return float(np.asarray(data[name]).reshape(()))


def _format_optional_float(value: object, spec: str, missing: str = "n/a") -> str:
    value = _finite_float(value)
    if value is None:
        return missing
    return format(value, spec)


def build_simple_unconfined_case(
    nx: int = 250,
    ny: int = 250,
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
) -> Unconfined2DCase:
    """
    Build a shared 2D unconfined benchmark case for MF6 and Warp.
    """
    if workspace is None:
        workspace = data_store.joinpath("working_tests", "mf6_vs_warp_2d_unconfined")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    active = _build_domain(nx=int(nx), ny=int(ny)).astype(np.int32)
    top = np.asarray(_build_dem(active), dtype=np.float64)
    bottom = np.asarray(_model_bottom(top), dtype=np.float64)
    bc_bool = _build_dirichlet_boundary_mask(active)

    bc_mask = bc_bool.astype(np.int32)
    bc_values = np.zeros((int(ny), int(nx)), dtype=np.float64)
    bc_values[bc_bool] = top[bc_bool]

    k_field = np.full((int(ny), int(nx)), float(hydraulic_conductivity), dtype=np.float64)
    k_field[active == 0] = 0.0

    recharge_field = np.full((int(ny), int(nx)), float(recharge), dtype=np.float64)
    recharge_field[active == 0] = 0.0

    initial_head = bottom + max(float(initial_saturated_thickness), 0.1)
    initial_head = np.minimum(initial_head, top)
    initial_head[bc_bool] = bc_values[bc_bool]
    initial_head[active == 0] = 0.0

    return Unconfined2DCase(
        nx=int(nx),
        ny=int(ny),
        dx=float(dx),
        workspace=workspace,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        top=top,
        bottom=bottom,
        hydraulic_conductivity=k_field,
        recharge=recharge_field,
        initial_head=initial_head.astype(np.float64, copy=False),
    )


def run_mf6_unconfined(case: Unconfined2DCase, out_path: str | Path | None = None) -> Path:
    """
    Run the MF6 single-layer unconfined truth model and save heads to NPZ.
    """
    out_path = Path(out_path) if out_path is not None else case.workspace.joinpath("mf6_heads.npz")
    mf6_ws = case.workspace.joinpath("mf6")
    mf6_ws.mkdir(parents=True, exist_ok=True)

    name = "unconf2d_truth"
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
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    )
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=f"{name}.nam",
        save_flows=True,
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="MODERATE",
        linear_acceleration="CG",
        outer_maximum=100,
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
    flopy.mf6.ModflowGwfic(
        gwf,
        pname="ic",
        strt=case.initial_head,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=[1],
        k=case.hydraulic_conductivity,
        k33=case.hydraulic_conductivity,
        k33overk=False,
        save_specific_discharge=True,
        save_saturation=True,
    )

    fixed_head_cells = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
    fixed_head_cells[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    chd_spd = _create_chd_single_period(boundary_heads=fixed_head_cells, active=case.active)
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="chd",
        stress_period_data=chd_spd,
        save_flows=True,
    )
    flopy.mf6.ModflowGwfrcha(
        gwf,
        pname="recharge",
        recharge=case.recharge,
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "LAST")],
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
        raise RuntimeError("MF6 unconfined run failed.")

    hds_path = mf6_ws.joinpath(f"{name}.hds")
    heads_raw = flopy.utils.HeadFile(str(hds_path)).get_data()
    heads = np.asarray(heads_raw[0], dtype=np.float64) if heads_raw.ndim == 3 else np.asarray(heads_raw)

    np.savez_compressed(
        out_path,
        heads=heads,
        engine_time=np.asarray(engine_time, dtype=np.float64),
        total_time=np.asarray(total_time, dtype=np.float64),
        nx=np.asarray(case.nx, dtype=np.int32),
        ny=np.asarray(case.ny, dtype=np.int32),
        dx=np.asarray(case.dx, dtype=np.float64),
    )
    print(f"MF6 unconfined heads saved to {out_path}")
    print(f"MF6 metrics - Total time: {total_time:.4f}s, Engine time: {engine_time:.4f}s\n")
    return out_path


def _save_outer_history_first25(info: dict, out_path: Path) -> None:
    """
    Save the first 25 Picard outer iterations as a CSV for debugging early-phase behaviour.

    :param info: solver info dictionary containing ``outer_history``.
    :param out_path: path to write the CSV file.
    """
    history = info.get("outer_history", []) if isinstance(info, dict) else []
    if not isinstance(history, list) or not history:
        return

    rows = history[:25]
    columns = [
        "outer_iteration",
        "inner_max_cycles_used",
        "inner_converged",
        "inner_head_change_converged",
        "inner_usable_for_picard",
        "h_rms_end",
        "inner_head_residual_tol_used",
        "picard_update_max",
        "picard_update_rms",
        "picard_scale",
        "omega",
        "chebyshev_ready",
        "chebyshev_used",
        "chebyshev_reset",
        "chebyshev_rejected",
        "trial_measure",
        "previous_measure",
        "clipped_update",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            writer.writerow({key: row.get(key) for key in columns})


def _solve_summary(info: object, elapsed: float, settings: dict) -> dict:
    summary = {
        "time": float(elapsed),
        "settings": dict(settings),
        "converged": False,
        "outer_iterations": None,
        "final_max_abs_head_change": None,
        "final_residual": None,
        "inner_solve_failures": None,
        "strict_inner_nonconvergence_count": None,
        "unusable_inner_solve_count": None,
        "practical_inner_acceptance_count": None,
        "accepted_picard_update_count": None,
        "outer_chebyshev_ready_count": None,
        "outer_chebyshev_used_count": None,
        "outer_chebyshev_reset_count": None,
        "chebyshev_rejections": None,
        "chebyshev_resets": None,
    }
    if isinstance(info, dict):
        summary["converged"] = bool(info.get("converged", False))
        for key in (
            "outer_iterations",
            "final_max_abs_head_change",
            "final_residual",
            "inner_solve_failures",
            "strict_inner_nonconvergence_count",
            "unusable_inner_solve_count",
            "practical_inner_acceptance_count",
            "accepted_picard_update_count",
            "outer_chebyshev_ready_count",
            "outer_chebyshev_used_count",
            "outer_chebyshev_reset_count",
            "chebyshev_rejections",
            "chebyshev_resets",
            "effectively_dry_cell_count",
        ):
            if info.get(key) is not None:
                value = info[key]
                summary[key] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return summary


def run_warp_unconfined(
    case: Unconfined2DCase,
    out_path: str | Path | None = None,
    device: str = "auto",
    chebyshev_enabled: bool = True,
    inner_smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_double_solve: bool = True,
) -> Path:
    """
    Run the same unconfined problem in the main 2D Warp solver and save heads to NPZ.
    """
    out_path = Path(out_path) if out_path is not None else case.workspace.joinpath("warp_heads.npz")
    device = _warp_device(device)

    initial_transmissivity = case.hydraulic_conductivity * np.maximum(case.initial_head - case.bottom, 0.1)
    initial_transmissivity[case.active == 0] = 0.0
    rhs_recharge = np.asarray(case.recharge, dtype=np.float64)

    t0 = time.perf_counter()
    with WarpDarcySolver(
        nx=case.nx,
        ny=case.ny,
        dx=case.dx,
        device=device,
        solver_type="kcycle",
        diag_preconditioner_backend=diag_preconditioner_backend,
    ) as warp_solver:
        warp_solver.build_from_fields(
            T_field=initial_transmissivity,
            R_field=rhs_recharge,
            active=case.active,
            bc_mask=case.bc_mask,
            bc_values=case.bc_values,
        )
        solve1_kwargs = {
            "formulation": "unconfined",
            "K_field": case.hydraulic_conductivity,
            "zbot_field": case.bottom,
            "ztop_field": case.top,
            "initial_head": case.initial_head.copy(),
            "max_cycles": 80,
            "max_levels": 5,
            "min_coarse_cells": 500,
            "rel_tol": 5.0e-7,
            "abs_tol_min": 5.0e-7,
            "dh_rms_tol": DEFAULT_DH_TOL,
            "residual_floor_tol": DEFAULT_RESIDUAL_FLOOR_TOL,
            "inner_forcing_eta": float(inner_forcing_eta),
            "inner_head_residual_tol_min": float(inner_head_residual_tol_min),
            "inner_head_residual_tol_max": float(inner_head_residual_tol_max),
            "chebyshev_reset_factor": float(chebyshev_reset_factor),
            "transmissivity_relaxation_enabled": bool(transmissivity_relaxation_enabled),
            "unconfined_startup_mode": str(unconfined_startup_mode),
            "smoother": str(inner_smoother),
            "cheby_lambda_min": float(cheby_lambda_min),
            "cheby_lambda_max": float(cheby_lambda_max),
            "max_outer_iterations": 60,
            "omega": 0.7,
            "omega_min": 0.1,
            "omega_max": 0.9,
            "hclose": DEFAULT_DH_TOL,
            "min_saturated_thickness": 0.1,
            "initial_saturated_thickness": 100.0,
            "max_head_change_per_outer_iteration": 10.0,
            "chebyshev_enabled": bool(chebyshev_enabled),
            "chebyshev_order": 3,
            "chebyshev_rejection_factor": 1.2,
        }
        if check_every_no is not None:
            solve1_kwargs["check_every_no"] = int(check_every_no)
        solve2_kwargs = dict(solve1_kwargs)

        if do_double_solve:
            t_solve1 = time.perf_counter()
            heads1, info1 = warp_solver.solve(**solve1_kwargs)
            solve1_time = time.perf_counter() - t_solve1
        else:
            heads1, info1 = None, None
            solve1_time = 0.0

        # Benchmark timing uses solve 2 from the same initial condition, matching
        # the 3D runner convention.
        t_solve2 = time.perf_counter()
        heads, info = warp_solver.solve(**solve2_kwargs)
        solve2_time = time.perf_counter() - t_solve2

    total_time = time.perf_counter() - t0
    solve1_summary = _solve_summary(info1, solve1_time, solve1_kwargs) if do_double_solve else {}
    solve2_summary = _solve_summary(info, solve2_time, solve2_kwargs)

    first25_path = Path(out_path).parent.joinpath("outer_history_first25.csv")
    _save_outer_history_first25(info, first25_path)

    np.savez_compressed(
        out_path,
        heads=np.asarray(heads, dtype=np.float64),
        heads_solve1=np.asarray(heads1, dtype=np.float64),
        total_time=np.asarray(total_time, dtype=np.float64),
        solve1_time=np.asarray(solve1_time, dtype=np.float64),
        solve2_time=np.asarray(solve2_time, dtype=np.float64),
        info=np.asarray(json.dumps(info, default=str)),
        info_solve1=np.asarray(json.dumps(info1, default=str) if info1 else "{}"),
        info_solve2=np.asarray(json.dumps(info, default=str)),
        summary_solve1=np.asarray(json.dumps(solve1_summary, default=str)),
        summary_solve2=np.asarray(json.dumps(solve2_summary, default=str)),
        solve1_settings=np.asarray(json.dumps(solve1_kwargs, default=str)),
        solve2_settings=np.asarray(json.dumps(solve2_kwargs, default=str)),
        nx=np.asarray(case.nx, dtype=np.int32),
        ny=np.asarray(case.ny, dtype=np.int32),
        dx=np.asarray(case.dx, dtype=np.float64),
        device=np.asarray(device),
    )
    print(f"Warp unconfined heads saved to {out_path}")
    print(
        f"Warp metrics - Total time: {total_time:.4f}s, "
        f"solve1: {solve1_time:.4f}s, solve2: {solve2_time:.4f}s"
    )
    if isinstance(info, dict):
        print("Warp nonlinear solve 2 summary:")
        for key in (
            "converged",
            "outer_iterations",
            "smoother",
            "final_max_abs_head_change",
            "final_residual",
            "final_h_rms_inner_residual",
            "chebyshev_rejections",
            "chebyshev_resets",
            "accepted_picard_update_count",
            "strict_inner_nonconvergence_count",
            "unusable_inner_solve_count",
            "practical_inner_acceptance_count",
            "outer_chebyshev_ready_count",
            "outer_chebyshev_used_count",
            "outer_chebyshev_reset_count",
            "inner_forcing_eta",
            "inner_head_residual_tol_min",
            "inner_head_residual_tol_max",
            "inner_solve_failures",
            "effectively_dry_cell_count",
        ):
            if key == "final_h_rms_inner_residual":
                print(f"  {key}: {info.get('inner_h_rms_end')}")
            else:
                print(f"  {key}: {info.get(key)}")
    print()
    return out_path


def load_results(mf6_path: str | Path, warp_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(mf6_path, allow_pickle=False) as mf6_npz:
        mf6_heads = np.asarray(mf6_npz["heads"], dtype=np.float64)
    with np.load(warp_path, allow_pickle=False) as warp_npz:
        warp_heads = np.asarray(warp_npz["heads"], dtype=np.float64)
    return mf6_heads, warp_heads


def compare_results(
    mf6_path: str | Path,
    warp_path: str | Path,
    active: np.ndarray | None = None,
) -> dict[str, float]:
    mf6_heads, warp_heads = load_results(mf6_path, warp_path)
    if mf6_heads.shape != warp_heads.shape:
        raise ValueError(f"Shape mismatch: MF6 {mf6_heads.shape}, Warp {warp_heads.shape}")

    if active is None:
        mask = np.isfinite(mf6_heads) & np.isfinite(warp_heads)
    else:
        mask = (np.asarray(active) != 0) & np.isfinite(mf6_heads) & np.isfinite(warp_heads)
        if mask.shape != mf6_heads.shape:
            raise ValueError(f"active shape {mask.shape} does not match heads {mf6_heads.shape}")

    diff = warp_heads - mf6_heads
    diff_masked = diff[mask]
    abs_diff = np.abs(diff_masked)
    metrics = {
        "rmse": float(np.sqrt(np.mean(diff_masked * diff_masked))),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_bias_warp_minus_mf6": float(np.mean(diff_masked)),
        "percent_within_0_01m": float(np.mean(abs_diff <= 0.01) * 100.0),
        "percent_within_0_1m": float(np.mean(abs_diff <= 0.1) * 100.0),
        "percent_within_1_0m": float(np.mean(abs_diff <= 1.0) * 100.0),
    }
    print("\nWarp vs MF6 unconfined head comparison, active cells")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6g}")
    return metrics


def _convergence_report(
    info: dict,
    comparison: dict[str, float] | None = None,
    mf6_agreement_tol: float = DEFAULT_MF6_AGREEMENT_TOL,
) -> dict:
    final_dh = _finite_float(info.get("final_max_abs_head_change"))
    hclose = None
    history = info.get("outer_history", [])
    if isinstance(history, list) and history:
        hclose = _finite_float(info.get("picard_head_tol"))
    max_abs_diff = _finite_float((comparison or {}).get("max_abs_diff"))

    head_change_converged = None if final_dh is None or hclose is None else bool(final_dh <= hclose)
    inner_residual_converged = bool(info.get("inner_residual_converged", False))
    inner_head_change_converged = bool(info.get("inner_head_change_converged", False))
    inner_practically_converged = bool(info.get("inner_practically_converged", False))
    agrees_with_mf6 = None if max_abs_diff is None else bool(max_abs_diff < float(mf6_agreement_tol))

    if bool(info.get("converged", False)):
        if inner_residual_converged:
            status = "Nonlinear head-change and inner residual tolerances met."
        elif inner_practically_converged:
            status = "Nonlinear head-change tolerance met via practical inner convergence."
        else:
            status = "Nonlinear head-change tolerance met."
    elif head_change_converged and agrees_with_mf6:
        status = "Reported nonlinear convergence is false, but head change and MF6 agreement are acceptable."
    else:
        status = "Convergence criteria not met."

    return {
        "head_change_converged": head_change_converged,
        "inner_residual_converged": inner_residual_converged,
        "inner_head_change_converged": inner_head_change_converged,
        "inner_practically_converged": inner_practically_converged,
        "agrees_with_mf6": agrees_with_mf6,
        "status": status,
        "final_max_abs_head_change": final_dh,
        "hclose": hclose,
        "residual_floor_tol": _finite_float(info.get("residual_floor_tol")),
        "max_abs_diff": max_abs_diff,
        "mf6_agreement_tol": float(mf6_agreement_tol),
    }


def run_case(
    nx: int = 250,
    ny: int = 250,
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
    device: str = "auto",
    chebyshev_enabled: bool = True,
    inner_smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
    do_double_solve: bool = True,
) -> dict:
    case = build_simple_unconfined_case(
        nx=nx,
        ny=ny,
        dx=dx,
        hydraulic_conductivity=hydraulic_conductivity,
        recharge=recharge,
        initial_saturated_thickness=initial_saturated_thickness,
        workspace=workspace,
    )

    print(f"Running 2D unconfined case: nx={case.nx}, ny={case.ny}, dx={case.dx}")
    print(f"Workspace: {case.workspace}\n")

    mf6_path = case.workspace.joinpath("mf6_heads.npz")
    warp_path = case.workspace.joinpath("warp_heads.npz")

    if do_run_mf6:
        run_mf6_unconfined(case, out_path=mf6_path)
    if do_run_warp:
        run_warp_unconfined(
            case,
            out_path=warp_path,
            device=device,
            chebyshev_enabled=chebyshev_enabled,
            inner_smoother=inner_smoother,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
            do_double_solve=do_double_solve,
        )

    metrics = {}
    if mf6_path.exists() and warp_path.exists():
        metrics = compare_results(mf6_path, warp_path, active=case.active)
    else:
        print("Skipping comparison because both MF6 and Warp heads were not generated or found.")

    warp_info = _load_npz_json(warp_path, "info_solve2")
    solve1_info = _load_npz_json(warp_path, "info_solve1")
    solve2_report = _convergence_report(warp_info, comparison=metrics)
    summary = {
        "nx": int(nx),
        "ny": int(ny),
        "n_cells": int(nx * ny),
        "dx": float(dx),
        "workspace": str(case.workspace),
        "chebyshev_enabled": bool(chebyshev_enabled),
        "inner_smoother": str(inner_smoother),
        "cheby_lambda_min": float(cheby_lambda_min),
        "cheby_lambda_max": float(cheby_lambda_max),
        "diag_preconditioner_backend": str(diag_preconditioner_backend),
        "check_every_no": None if check_every_no is None else int(check_every_no),
        "unconfined_startup_mode": str(unconfined_startup_mode),
        "mf6_engine_time": _load_npz_scalar(mf6_path, "engine_time"),
        "mf6_total_time": _load_npz_scalar(mf6_path, "total_time"),
        "warp_total_time": _load_npz_scalar(warp_path, "total_time"),
        "warp_solve1_time": _load_npz_scalar(warp_path, "solve1_time"),
        "warp_solve2_time": _load_npz_scalar(warp_path, "solve2_time"),
        "warp_benchmark_time": _load_npz_scalar(warp_path, "solve2_time"),
        "solve1_converged": bool(solve1_info.get("converged", False)) if solve1_info else None,
        "solve2_converged": bool(warp_info.get("converged", False)) if warp_info else None,
        "solve2_inner_smoother": warp_info.get("smoother"),
        "solve2_outer_iterations": warp_info.get("outer_iterations"),
        "solve2_final_max_abs_head_change": _finite_float(warp_info.get("final_max_abs_head_change")),
        "solve2_final_residual": _finite_float(warp_info.get("final_residual")),
        "solve2_final_h_rms_inner_residual": _finite_float(warp_info.get("inner_h_rms_end")),
        "solve2_chebyshev_rejections": warp_info.get("chebyshev_rejections"),
        "solve2_chebyshev_resets": warp_info.get("chebyshev_resets"),
        "solve2_strict_inner_nonconvergence_count": warp_info.get("strict_inner_nonconvergence_count"),
        "solve2_unusable_inner_solve_count": warp_info.get("unusable_inner_solve_count"),
        "solve2_practical_inner_acceptance_count": warp_info.get("practical_inner_acceptance_count"),
        "solve2_accepted_picard_update_count": warp_info.get("accepted_picard_update_count"),
        "solve2_outer_chebyshev_ready_count": warp_info.get("outer_chebyshev_ready_count"),
        "solve2_outer_chebyshev_used_count": warp_info.get("outer_chebyshev_used_count"),
        "solve2_outer_chebyshev_reset_count": warp_info.get("outer_chebyshev_reset_count"),
        "solve2_inner_forcing_eta": warp_info.get("inner_forcing_eta"),
        "solve2_inner_head_residual_tol_min": warp_info.get("inner_head_residual_tol_min"),
        "solve2_inner_head_residual_tol_max": warp_info.get("inner_head_residual_tol_max"),
        "solve2_inner_solve_failures": warp_info.get("inner_solve_failures"),
        "solve2_effectively_dry_cell_count": warp_info.get("effectively_dry_cell_count"),
        "convergence_report": solve2_report,
        "comparison": metrics,
    }

    summary_path = case.workspace.joinpath("unconfined_benchmark_summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=4)
    print(f"Benchmark summary saved to {summary_path}")
    print(f"Solve 2 convergence report: {solve2_report['status']}")
    return summary


def run_grid_benchmark(
    grid_sizes: list[int | tuple[int, int]] | tuple[int | tuple[int, int], ...] = tuple(BENCHMARK_GRID_SIZES),
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
    device: str = "auto",
    chebyshev_enabled: bool = True,
    inner_smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
    do_double_solve: bool = True,
) -> list[dict]:
    """
    Run the 2D unconfined MF6-vs-Warp benchmark over a range of grid sizes.
    """
    if workspace is None:
        workspace = data_store.joinpath("working_tests", "mf6_vs_warp_2d_unconfined_grid_benchmark")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    normalized_sizes: list[tuple[int, int]] = []
    for item in grid_sizes:
        if isinstance(item, tuple):
            nx_i, ny_i = item
        else:
            nx_i = int(item)
            ny_i = int(item)
        normalized_sizes.append((int(nx_i), int(ny_i)))

    print("\n" + "=" * 72)
    print("2D unconfined Warp vs MF6 grid-size benchmark")
    print(f"grid sizes: {normalized_sizes}")
    print(f"dx: {dx}")
    print(f"inner smoother: {inner_smoother}")
    print(f"workspace: {workspace}")
    print("=" * 72)

    current_keys = [(int(nx), int(ny)) for nx, ny in normalized_sizes]
    summary_path = workspace.joinpath("grid_benchmark_summary.json")
    previous_results: dict[tuple[int, int], dict] = {}
    results_dict: dict[tuple[int, int], dict] = {}
    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                existing = json.load(f)
            for row in existing:
                previous_results[(int(row["nx"]), int(row["ny"]))] = row
        except Exception:
            pass

    for nx, ny in normalized_sizes:
        print("\n" + "-" * 72)
        print(f"Benchmark grid: nx={nx}, ny={ny}")
        print("-" * 72)

        case_workspace = workspace.joinpath(f"grid_{nx:04d}x{ny:04d}")
        row = run_case(
            nx=nx,
            ny=ny,
            dx=dx,
            hydraulic_conductivity=hydraulic_conductivity,
            recharge=recharge,
            initial_saturated_thickness=initial_saturated_thickness,
            workspace=case_workspace,
            device=device,
            chebyshev_enabled=chebyshev_enabled,
            inner_smoother=inner_smoother,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
            do_run_mf6=do_run_mf6,
            do_run_warp=do_run_warp,
            do_double_solve=do_double_solve,
        )

        key = (int(nx), int(ny))
        old_row = previous_results.get(key)
        if old_row is not None:
            if not do_run_mf6:
                row["mf6_engine_time"] = old_row.get("mf6_engine_time")
                row["mf6_total_time"] = old_row.get("mf6_total_time")
            if not do_run_warp:
                for name in (
                    "warp_total_time",
                    "warp_solve1_time",
                    "warp_solve2_time",
                    "warp_benchmark_time",
                    "solve1_converged",
                    "solve2_converged",
                    "solve2_inner_smoother",
                    "solve2_outer_iterations",
                    "solve2_final_max_abs_head_change",
                    "solve2_final_residual",
                    "solve2_chebyshev_rejections",
                    "solve2_chebyshev_resets",
                    "solve2_inner_solve_failures",
                    "solve2_effectively_dry_cell_count",
                    "convergence_report",
                ):
                    row[name] = old_row.get(name)
            if not (do_run_mf6 and do_run_warp) and not row.get("comparison"):
                row["comparison"] = old_row.get("comparison", {})

        results_dict[key] = row
        results = [results_dict[k] for k in current_keys if k in results_dict]
        with summary_path.open("w") as f:
            json.dump(results, f, indent=4)
        print(f"Updated grid benchmark summary: {summary_path}")

    print("\nGrid benchmark complete.")
    return [results_dict[k] for k in current_keys if k in results_dict]


def run_diag_preconditioner_backend_matrix(
    grid_sizes: list[int | tuple[int, int]] | tuple[int | tuple[int, int], ...] = tuple(BENCHMARK_GRID_SIZES),
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
    device: str = "auto",
    chebyshev_enabled: bool = True,
    inner_smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
) -> list[dict]:
    """
    Run the tuned backend/check-frequency benchmark matrix for unconfined 2D cases.
    """
    if workspace is None:
        workspace = data_store.joinpath("working_tests", "mf6_vs_warp_2d_unconfined_backend_matrix")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    normalized_sizes: list[tuple[int, int]] = []
    for item in grid_sizes:
        if isinstance(item, tuple):
            nx_i, ny_i = item
        else:
            nx_i = int(item)
            ny_i = int(item)
        normalized_sizes.append((int(nx_i), int(ny_i)))

    scenarios = (
        {"case_id": "A", "diag_preconditioner_backend": "host", "check_every_no": 1},
        {"case_id": "B", "diag_preconditioner_backend": "host", "check_every_no": 5},
        {"case_id": "C", "diag_preconditioner_backend": "device", "check_every_no": 5},
        {"case_id": "D", "diag_preconditioner_backend": "device", "check_every_no": 10},
    )

    results: list[dict] = []
    for scenario in scenarios:
        case_id = str(scenario["case_id"])
        backend = str(scenario["diag_preconditioner_backend"])
        check_every_no = int(scenario["check_every_no"])
        for nx, ny in normalized_sizes:
            case_workspace = workspace.joinpath(
                f"{case_id}_backend_{backend}_check_{check_every_no}_grid_{nx:04d}x{ny:04d}"
            )
            row = run_case(
                nx=nx,
                ny=ny,
                dx=dx,
                hydraulic_conductivity=hydraulic_conductivity,
                recharge=recharge,
                initial_saturated_thickness=initial_saturated_thickness,
                workspace=case_workspace,
                device=device,
                chebyshev_enabled=chebyshev_enabled,
                inner_smoother=inner_smoother,
                cheby_lambda_min=cheby_lambda_min,
                cheby_lambda_max=cheby_lambda_max,
                inner_forcing_eta=inner_forcing_eta,
                inner_head_residual_tol_min=inner_head_residual_tol_min,
                inner_head_residual_tol_max=inner_head_residual_tol_max,
                chebyshev_reset_factor=chebyshev_reset_factor,
                transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
                unconfined_startup_mode=unconfined_startup_mode,
                diag_preconditioner_backend=backend,
                check_every_no=check_every_no,
                do_run_mf6=do_run_mf6,
                do_run_warp=do_run_warp,
            )
            row["case_id"] = case_id
            results.append(row)

    summary_json_path = workspace.joinpath("backend_matrix_summary.json")
    with summary_json_path.open("w") as f:
        json.dump(results, f, indent=4)

    if results:
        summary_csv_path = workspace.joinpath("backend_matrix_summary.csv")
        columns = list(results[0].keys())
        with summary_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

    print(f"Backend matrix summary saved to {summary_json_path}")
    return results


def run_chebyshev_lambda_sweep(
    nx: int = 500,
    ny: int = 500,
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    cheby_lambda_min_values: list[float] | tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.5),
    cheby_lambda_max_values: list[float] | tuple[float, ...] = (1.7, 1.8, 1.95, 2.0, 2.1, 2.2, 2.5),
    workspace: str | Path | None = None,
    device: str = "auto",
    do_run_mf6: bool = True,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_double_solve: bool = True,
) -> list[dict]:
    """
    Run a single 2D unconfined case across a range of Chebyshev lambda bounds.

    MF6 is run once and reused for comparison. A CSV summary of the sweep is
    written to the workspace root.

    :param nx: number of columns.
    :param ny: number of rows.
    :param dx: cell size.
    :param hydraulic_conductivity: uniform K value.
    :param recharge: recharge rate.
    :param initial_saturated_thickness: initial saturated thickness.
    :param cheby_lambda_min_values: iterable of lower Chebyshev bounds to test.
    :param cheby_lambda_max_values: iterable of upper Chebyshev bounds to test.
    :param workspace: root directory for the sweep outputs.
    :param device: Warp device.
    :param do_run_mf6: whether to run the MF6 truth model.
    :param inner_forcing_eta: dynamic inner tolerance fraction.
    :param inner_head_residual_tol_max: dynamic inner tolerance ceiling.
    :param chebyshev_reset_factor: residual-increase reset threshold.
    :param transmissivity_relaxation_enabled: optional T-relaxation flag.
    :param unconfined_startup_mode: "initial_head" or "confined_pre_solve".
    :return: list of per-combination result dictionaries.
    """
    if workspace is None:
        workspace = data_store.joinpath(
            "working_tests",
            "mf6_vs_warp_2d_unconfined_lambda_sweep",
        )
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    base_case = build_simple_unconfined_case(
        nx=int(nx),
        ny=int(ny),
        dx=float(dx),
        hydraulic_conductivity=float(hydraulic_conductivity),
        recharge=float(recharge),
        initial_saturated_thickness=float(initial_saturated_thickness),
        workspace=workspace,
    )

    mf6_path = workspace.joinpath("mf6_heads.npz")
    if do_run_mf6:
        run_mf6_unconfined(base_case, out_path=mf6_path)

    results: list[dict] = []
    for lambda_min in cheby_lambda_min_values:
        for lambda_max in cheby_lambda_max_values:
            if float(lambda_min) >= float(lambda_max):
                print(f"Skipping invalid lambda pair: min={lambda_min}, max={lambda_max}")
                continue

            combo_workspace = workspace.joinpath(f"lambda_min_{float(lambda_min):.4f}_max_{float(lambda_max):.4f}")
            combo_workspace.mkdir(parents=True, exist_ok=True)
            combo_case = build_simple_unconfined_case(
                nx=int(nx),
                ny=int(ny),
                dx=float(dx),
                hydraulic_conductivity=float(hydraulic_conductivity),
                recharge=float(recharge),
                initial_saturated_thickness=float(initial_saturated_thickness),
                workspace=combo_workspace,
            )

            warp_path = combo_workspace.joinpath("warp_heads.npz")
            run_warp_unconfined(
                combo_case,
                out_path=warp_path,
                device=device,
                chebyshev_enabled=True,
                inner_smoother="chebyshev",
                cheby_lambda_min=float(lambda_min),
                cheby_lambda_max=float(lambda_max),
                inner_forcing_eta=float(inner_forcing_eta),
                inner_head_residual_tol_min=float(inner_head_residual_tol_min),
                inner_head_residual_tol_max=float(inner_head_residual_tol_max),
                chebyshev_reset_factor=float(chebyshev_reset_factor),
                transmissivity_relaxation_enabled=bool(transmissivity_relaxation_enabled),
                unconfined_startup_mode=str(unconfined_startup_mode),
                diag_preconditioner_backend=str(diag_preconditioner_backend),
                check_every_no=check_every_no,
                do_double_solve=do_double_solve,
            )

            metrics = {}
            if mf6_path.exists() and warp_path.exists():
                metrics = compare_results(mf6_path, warp_path, active=combo_case.active)

            warp_info = _load_npz_json(warp_path, "info_solve2")
            row = {
                "nx": int(nx),
                "ny": int(ny),
                "dx": float(dx),
                "cheby_lambda_min": float(lambda_min),
                "cheby_lambda_max": float(lambda_max),
                "diag_preconditioner_backend": str(diag_preconditioner_backend),
                "check_every_no": None if check_every_no is None else int(check_every_no),
                "unconfined_startup_mode": str(unconfined_startup_mode),
                "converged": bool(warp_info.get("converged", False)) if warp_info else None,
                "outer_iterations": warp_info.get("outer_iterations"),
                "final_max_abs_head_change": _finite_float(warp_info.get("final_max_abs_head_change")),
                "final_residual": _finite_float(warp_info.get("final_residual")),
                "inner_h_rms_end": _finite_float(warp_info.get("inner_h_rms_end")),
                "unusable_inner_solve_count": warp_info.get("unusable_inner_solve_count"),
                "practical_inner_acceptance_count": warp_info.get("practical_inner_acceptance_count"),
                "accepted_picard_update_count": warp_info.get("accepted_picard_update_count"),
                "outer_chebyshev_ready_count": warp_info.get("outer_chebyshev_ready_count"),
                "outer_chebyshev_used_count": warp_info.get("outer_chebyshev_used_count"),
                "outer_chebyshev_reset_count": warp_info.get("outer_chebyshev_reset_count"),
                "solve2_time": _load_npz_scalar(warp_path, "solve2_time"),
                "rmse": metrics.get("rmse"),
                "max_abs_diff": metrics.get("max_abs_diff"),
                "workspace": str(combo_workspace),
            }
            results.append(row)
            print(
                f"lambda_min={lambda_min:.4f} lambda_max={lambda_max:.4f} -> "
                f"converged={row['converged']} outer_iter={row['outer_iterations']} "
                f"time={_format_optional_float(row['solve2_time'], '.4f')}s "
                f"rmse={_format_optional_float(row['rmse'], '.6g')} "
                f"max_abs_diff={_format_optional_float(row['max_abs_diff'], '.6g')}"
            )

    summary_path = workspace.joinpath("lambda_sweep_summary.csv")
    if results:
        columns = list(results[0].keys())
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nLambda sweep summary saved to {summary_path}")

    return results


def main(
    grid_sizes=BENCHMARK_GRID_SIZES,
    dx=100.0,
    hydraulic_conductivity=100.0,
    recharge=1.0e-4,
    initial_saturated_thickness=100.0,
    workspace=None,
    device="auto",
    chebyshev_enabled=True,
    inner_smoother="chebyshev",
    cheby_lambda_min=0.1,
    cheby_lambda_max=2.0,
    inner_forcing_eta=0.10,
    inner_head_residual_tol_min=DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max=1.0e-2,
    chebyshev_reset_factor=1.2,
    transmissivity_relaxation_enabled=False,
    unconfined_startup_mode="confined_pre_solve",
    diag_preconditioner_backend="device",
    check_every_no=5,
    do_run_mf6=False,
    do_run_warp=True,
    run_lambda_sweep=False,
    run_backend_matrix=False,
    do_double_solve=False,
):
    if run_backend_matrix:
        results = run_diag_preconditioner_backend_matrix(
            grid_sizes=grid_sizes,
            dx=dx,
            hydraulic_conductivity=hydraulic_conductivity,
            recharge=recharge,
            initial_saturated_thickness=initial_saturated_thickness,
            workspace=workspace,
            device=device,
            chebyshev_enabled=chebyshev_enabled,
            inner_smoother=inner_smoother,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            do_run_mf6=do_run_mf6,
            do_run_warp=do_run_warp,
        )
    elif run_lambda_sweep:
        results = run_chebyshev_lambda_sweep(
            nx=500,
            ny=500,
            dx=dx,
            hydraulic_conductivity=hydraulic_conductivity,
            recharge=recharge,
            initial_saturated_thickness=initial_saturated_thickness,
            cheby_lambda_min_values=(0.05, 0.1, 0.15, 0.2, 0.25, 0.5),
            cheby_lambda_max_values=(1.7, 1.8, 1.95, 2.0, 2.1, 2.2, 2.5),
            workspace=None,
            device=device,
            do_run_mf6=do_run_mf6,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
        )
    else:
        results = run_grid_benchmark(
            grid_sizes=grid_sizes,
            dx=dx,
            hydraulic_conductivity=hydraulic_conductivity,
            recharge=recharge,
            initial_saturated_thickness=initial_saturated_thickness,
            workspace=workspace,
            device=device,
            chebyshev_enabled=chebyshev_enabled,
            inner_smoother=inner_smoother,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
            do_run_mf6=do_run_mf6,
            do_run_warp=do_run_warp,
            do_double_solve=do_double_solve,
        )
    print(json.dumps(results, indent=4))

if __name__ == '__main__':
    # Configuration parameters
    grid_sizes = BENCHMARK_GRID_SIZES
    dx = 100.0
    hydraulic_conductivity = 100.0
    recharge = 1.0e-4
    initial_saturated_thickness = 100.0
    workspace = None
    device = "auto"
    chebyshev_enabled = True
    inner_smoother = "chebyshev"
    cheby_lambda_min = 0.1
    cheby_lambda_max = 2.0
    inner_forcing_eta = 0.10
    inner_head_residual_tol_min = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN
    inner_head_residual_tol_max = 1.0e-2
    chebyshev_reset_factor = 1.2
    transmissivity_relaxation_enabled = False
    unconfined_startup_mode = "confined_pre_solve"  # or "initial_head"
    diag_preconditioner_backend = "device"
    check_every_no = 5
    do_run_mf6 = False
    do_run_warp = True
    run_lambda_sweep = False
    run_backend_matrix = False
    do_double_solve = False

    main(
        grid_sizes=grid_sizes,
        dx=dx,
        hydraulic_conductivity=hydraulic_conductivity,
        recharge=recharge,
        initial_saturated_thickness=initial_saturated_thickness,
        workspace=workspace,
        device=device,
        chebyshev_enabled=chebyshev_enabled,
        inner_smoother=inner_smoother,
        cheby_lambda_min=cheby_lambda_min,
        cheby_lambda_max=cheby_lambda_max,
        inner_forcing_eta=inner_forcing_eta,
        inner_head_residual_tol_min=inner_head_residual_tol_min,
        inner_head_residual_tol_max=inner_head_residual_tol_max,
        chebyshev_reset_factor=chebyshev_reset_factor,
        transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
        unconfined_startup_mode=unconfined_startup_mode,
        diag_preconditioner_backend=diag_preconditioner_backend,
        check_every_no=check_every_no,
        do_run_mf6=do_run_mf6,
        do_run_warp=do_run_warp,
        run_lambda_sweep=run_lambda_sweep,
        run_backend_matrix=run_backend_matrix,
        do_double_solve=do_double_solve,
    )
