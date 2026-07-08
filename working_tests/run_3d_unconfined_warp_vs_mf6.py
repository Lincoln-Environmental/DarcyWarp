from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["DARCY_FLOAT"] = "float64"

from DARCY_WARP_PACKAGE.model_builder import (  # noqa: E402
    _build_dem,
    _build_dirichlet_boundary_mask,
    _build_domain,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model_multilayer, fill_nan_with_nearest  # noqa: E402
from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D  # noqa: E402


BENCHMARK_LAYERS = [1,2,3,4,5]
BENCHMARK_SMOOTHERS = ["chebyshev", "chebyshev_vertical_line"]
DEFAULT_DH_TOL = 1.0e-4
DEFAULT_RESIDUAL_FLOOR_TOL = 1.0e-4
DEFAULT_MF6_AGREEMENT_TOL = 5.0e-5


@dataclass(frozen=True)
class Unconfined3DCase:
    nx: int
    ny: int
    nlay: int
    dx: float
    dz: float
    workspace: Path
    hk_2d: np.ndarray
    top_3d: np.ndarray
    bottom_3d: np.ndarray
    recharge_2d: np.ndarray
    active_3d: np.ndarray
    bc_mask_3d: np.ndarray
    bc_values_3d: np.ndarray
    rhs_3d: np.ndarray
    initial_head_3d: np.ndarray


def _warp_device(preferred: str = "cuda:0") -> str:
    import warp as wp

    if preferred != "auto":
        return preferred
    try:
        return "cuda:0" if wp.is_cuda_available() else "cpu"
    except AttributeError:
        return "cuda:0"


def build_simple_unconfined_multilayer_case(
    nx: int = 1000,
    ny: int = 200,
    nlay: int = 2,
    dx: float = 100.0,
    layer_thickness: float = 150.0,
    transmissivity: float = 3000.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    heterogeneous_t: bool = False,
    seed: int = 123,
    workspace: str | Path | None = None,
) -> Unconfined3DCase:
    """
    Build one simple multi-layer confined benchmark case shared by MF6 and Warp.
    """
    nlay = int(nlay)
    dz = float(layer_thickness)
    total_thickness = dz * nlay

    if workspace is None:
        workspace = data_store.joinpath("working_tests", "mf6_vs_warp_3d")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    domain = _build_domain(nx=int(nx), ny=int(ny))
    dem = _build_dem(domain)
    dirichlet_mask = _build_dirichlet_boundary_mask(domain)

    if heterogeneous_t:
        t_field = make_ugly_T_field(nx=int(nx), ny=int(ny), domain=domain, seed=int(seed))
    else:
        t_field = np.full((int(ny), int(nx)), float(transmissivity), dtype=np.float64)

    hk_2d = np.asarray(t_field, dtype=np.float64) / float(total_thickness)
    recharge_2d = np.full((int(ny), int(nx)), float(recharge), dtype=np.float64)
    recharge_2d[domain == 0] = 0.0

    active_3d = np.repeat(domain[np.newaxis, :, :], nlay, axis=0).astype(np.int32)
    bc_mask_3d = np.repeat(dirichlet_mask[np.newaxis, :, :], nlay, axis=0).astype(np.int32)

    bc_values_3d = np.zeros((nlay, int(ny), int(nx)), dtype=np.float64)
    for layer in range(nlay):
        bc_values_3d[layer, dirichlet_mask] = dem[dirichlet_mask]

    rhs_3d = np.zeros((nlay, int(ny), int(nx)), dtype=np.float64)
    rhs_3d[0, :, :] = recharge_2d * float(dx) * float(dx)

    top_2d = fill_nan_with_nearest(dem).astype(np.float64)
    top_3d = np.zeros((nlay, int(ny), int(nx)), dtype=np.float64)
    bottom_3d = np.zeros((nlay, int(ny), int(nx)), dtype=np.float64)
    for i in range(nlay):
        top_3d[i, :, :] = top_2d - i * dz
        bottom_3d[i, :, :] = top_2d - (i + 1) * dz

    initial_head_3d = bottom_3d + max(float(initial_saturated_thickness), 0.1)
    initial_head_3d = np.minimum(initial_head_3d, top_3d)

    for layer in range(nlay):
        initial_head_3d[layer, dirichlet_mask] = dem[dirichlet_mask]
        bc_values_3d[layer, dirichlet_mask] = dem[dirichlet_mask]

    initial_head_3d[active_3d == 0] = 0.0

    return Unconfined3DCase(
        nx=int(nx),
        ny=int(ny),
        nlay=nlay,
        dx=float(dx),
        dz=dz,
        workspace=workspace,
        hk_2d=hk_2d,
        top_3d=top_3d,
        bottom_3d=bottom_3d,
        recharge_2d=recharge_2d,
        active_3d=active_3d,
        bc_mask_3d=bc_mask_3d,
        bc_values_3d=bc_values_3d,
        rhs_3d=rhs_3d,
        initial_head_3d=initial_head_3d,
    )


def run_mf6(case: Unconfined3DCase, out_path: str | Path | None = None) -> Path:
    """
    Run the multi-layer MF6 unconfined truth model and save heads to NPZ.
    """
    out_path = Path(out_path) if out_path is not None else case.workspace.joinpath("mf6_heads.npz")
    mf6_ws = case.workspace.joinpath("mf6")

    t0 = time.perf_counter()
    heads, engine_time = make_mf_model_multilayer(
        nx=case.nx,
        ny=case.ny,
        nlay=case.nlay,
        grid_size=case.dx,
        workspace=mf6_ws,
        hk=case.hk_2d,
        vertical_k=case.hk_2d,
        recharge=case.recharge_2d,
        layer_thickness=case.dz,
        run=True,
        use_ghb=False,
        icelltype=1,
    )
    total_time = time.perf_counter() - t0

    np.savez_compressed(
        out_path,
        heads=np.asarray(heads, dtype=np.float64),
        engine_time=np.asarray(engine_time, dtype=np.float64),
        total_time=np.asarray(total_time, dtype=np.float64),
        nx=np.asarray(case.nx, dtype=np.int32),
        ny=np.asarray(case.ny, dtype=np.int32),
        nlay=np.asarray(case.nlay, dtype=np.int32),
        dx=np.asarray(case.dx, dtype=np.float64),
        dz=np.asarray(case.dz, dtype=np.float64),
    )
    print(f"MF6 heads saved to {out_path}")
    print(f"MF6 metrics - Total time: {total_time:.4f}s, Engine time: {engine_time:.4f}s\n")
    return out_path


def run_warp_unconfined(
    case: Unconfined3DCase,
    out_path: str | Path | None = None,
    device: str = "auto",
    solver: str = "kcycle",
    smoother: str = "chebyshev",
    adaptive_kcycle: bool = True,
    line_omega: float = 0.8,
    line_sweeps_pre: int = 1,
    line_sweeps_post: int = 1,
    line_sweeps_coarse: int = 1,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_double_solve: bool = True,
    chebyshev_enabled: bool = True,
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = 1.0e-4,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
) -> Path:
    """
    Run the same multi-layer problem in Warp and save heads to NPZ.
    """
    def _is_converged(info: object) -> bool:
        return isinstance(info, dict) and bool(info.get("converged", False))

    def _solve_summary(info: object, elapsed: float, mode: str, settings: dict) -> dict:
        summary: dict = {
            "mode": str(mode),
            "time": float(elapsed),
            "settings": dict(settings),
            "converged": False,
            # Nonlinear (Picard) convergence group
            "picard": {
                "picard_converged": None,
                "picard_n_iter_used": None,
                "picard_dh_rms_end": None,
                "picard_dh_max_end": None,
            },
            # Linear K-cycle convergence group (from the last inner solve)
            "kcycle": {
                "n_cycles_used": None,
                "r_rms_end": None,
                "tol_abs": None,
                "dh_rms_lastcheck": None,
                "dh_max_lastcheck": None,
                "h_rms_end": None,
            },
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
            "effectively_dry_cell_count": None,
        }
        if isinstance(info, dict):
            summary["converged"] = bool(info.get("converged", False))

            pic = summary["picard"]
            for key in ("picard_converged", "picard_n_iter_used"):
                if info.get(key) is not None:
                    value = info[key]
                    pic[key] = int(value) if isinstance(value, (int, np.integer)) else bool(value)
            for key in ("picard_dh_rms_end", "picard_dh_max_end"):
                if info.get(key) is not None:
                    pic[key] = float(info[key])

            kc = summary["kcycle"]
            if info.get("n_cycles_used") is not None:
                kc["n_cycles_used"] = int(info["n_cycles_used"])
            for key in ("r_rms_end", "tol_abs", "dh_rms_lastcheck", "dh_max_lastcheck", "h_rms_end"):
                if info.get(key) is not None:
                    kc[key] = float(info[key])

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

    out_path = Path(out_path) if out_path is not None else case.workspace.joinpath("warp_heads.npz")
    device = _warp_device(device)
    solver = str(solver).lower()
    if solver not in {"kcycle", "chebyshev"}:
        raise ValueError("solver must be 'kcycle' or 'chebyshev'.")
    smoother = str(smoother).lower()
    if smoother not in {"chebyshev", "jacobi", "vertical_line", "chebyshev_vertical_line"}:
        raise ValueError("smoother must be 'chebyshev', 'jacobi', 'vertical_line', or 'chebyshev_vertical_line'.")
    if solver == "chebyshev":
        print("Using standalone Chebyshev debug path. Use solver='kcycle' for the MF6 comparison benchmark.")

    hk_3d = np.repeat(case.hk_2d[np.newaxis, :, :], case.nlay, axis=0)
    initial_head = np.asarray(case.initial_head_3d, dtype=np.float64)

    t0 = time.perf_counter()
    with WarpDarcySolver3D(
        nx=case.nx,
        ny=case.ny,
        nz=case.nlay,
        dx=case.dx,
        dy=case.dx,
        dz=case.dz,
        device=device,
        solver=solver,
        diag_preconditioner_backend=diag_preconditioner_backend,
    ) as warp_solver:
        warp_solver.build_from_K_fields(
            kx_field=hk_3d,
            ky_field=hk_3d,
            kz_field=hk_3d,
            active=case.active_3d,
            bc_mask=case.bc_mask_3d,
            bc_values=case.bc_values_3d,
            rhs=case.rhs_3d,
            initial_head=case.initial_head_3d,
        )
        if solver == "kcycle":
            simple_kwargs = {
                "unconfined": True,
                "zbot_field": case.bottom_3d,
                "unconfined_min_sat": 0.1,
                "unconfined_max_picard_iter": 60,
                "unconfined_relax": 0.7,
                "unconfined_head_tol": DEFAULT_DH_TOL,
                "unconfined_startup_mode": str(unconfined_startup_mode),
                "initial_head": initial_head.copy(),
                "max_cycles": 80,
                "rel_tol": 5.0e-7,
                "abs_tol_min": 5.0e-7,
                "check_every_no": check_every_no if check_every_no is not None else 1,
                "max_levels": 6,
                "smoother": smoother,
                "nu_pre": 6,
                "nu_post": 6,
                "nu_coarse": 2,
                "omega": 0.7,
                "dh_rms_tol": DEFAULT_DH_TOL,
                "line_omega": float(line_omega),
                "line_sweeps_pre": int(line_sweeps_pre),
                "line_sweeps_post": int(line_sweeps_post),
                "line_sweeps_coarse": int(line_sweeps_coarse),
                "cheby_lambda_min": float(cheby_lambda_min),
                "cheby_lambda_max": float(cheby_lambda_max),
                "chebyshev_enabled": bool(chebyshev_enabled),
                "chebyshev_reset_factor": float(chebyshev_reset_factor),
                "inner_forcing_eta": float(inner_forcing_eta),
                "inner_head_residual_tol_min": float(inner_head_residual_tol_min),
                "inner_head_residual_tol_max": float(inner_head_residual_tol_max),
                "transmissivity_relaxation_enabled": bool(transmissivity_relaxation_enabled),
            }
            robust_kwargs = dict(simple_kwargs)
            robust_kwargs.update(
                {
                    "unconfined_max_picard_iter": 100,
                    "max_cycles": 400,
                }
            )
        else:
            simple_kwargs = {
                "unconfined": True,
                "zbot_field": case.bottom_3d,
                "unconfined_min_sat": 0.1,
                "unconfined_max_picard_iter": 60,
                "unconfined_relax": 0.7,
                "unconfined_head_tol": DEFAULT_DH_TOL,
                "unconfined_startup_mode": str(unconfined_startup_mode),
                "initial_head": initial_head.copy(),
                "max_iter": 400,
                "rel_tol": 5.0e-7,
                "check_every_no": check_every_no if check_every_no is not None else 1,
                "cheby_lambda_min": float(cheby_lambda_min),
                "cheby_lambda_max": float(cheby_lambda_max),
                "chebyshev_enabled": bool(chebyshev_enabled),
                "chebyshev_reset_factor": float(chebyshev_reset_factor),
                "inner_forcing_eta": float(inner_forcing_eta),
                "inner_head_residual_tol_min": float(inner_head_residual_tol_min),
                "inner_head_residual_tol_max": float(inner_head_residual_tol_max),
                "transmissivity_relaxation_enabled": bool(transmissivity_relaxation_enabled),
            }
            robust_kwargs = dict(simple_kwargs)

        solve1_mode = "simple"
        solve1_kwargs = dict(simple_kwargs)
        solve1_call_kwargs = dict(solve1_kwargs)
        solve1_call_kwargs["initial_head"] = initial_head.copy()
        
        if do_double_solve:
            t_solve1 = time.perf_counter()
            heads1, info1 = warp_solver.solve(**solve1_call_kwargs)
            solve1_time = time.perf_counter() - t_solve1
        else:
            heads1, info1 = None, None
            solve1_time = 0.0

        if solver == "kcycle" and adaptive_kcycle and info1 is not None and not _is_converged(info1):
            solve2_mode = "robust"
            solve2_kwargs = dict(robust_kwargs)
            print("Warp solve1 did not converge with simple settings; solve2 will use robust settings.")
        else:
            solve2_mode = "simple"
            solve2_kwargs = dict(simple_kwargs)

        solve2_call_kwargs = dict(solve2_kwargs)
        solve2_call_kwargs["initial_head"] = initial_head.copy()
        t_solve2 = time.perf_counter()
        heads, info = warp_solver.solve(**solve2_call_kwargs)
        solve2_time = time.perf_counter() - t_solve2

    if heads is None or info is None:
        raise RuntimeError("Warp solve2 did not return heads and convergence information.")

    total_time = time.perf_counter() - t0
    _ = heads1

    solve1_summary = _solve_summary(info1, solve1_time, solve1_mode, solve1_kwargs) if do_double_solve else {}
    solve2_summary = _solve_summary(info, solve2_time, solve2_mode, solve2_kwargs)
    adaptive_settings = {
        "adaptive_kcycle": bool(adaptive_kcycle),
        "rule": "solve1_simple_then_solve2_robust_only_if_solve1_failed",
    }
    speed_controls = {
        "check_every_no": None if check_every_no is None else int(check_every_no),
        "do_double_solve": bool(do_double_solve),
        "cheby_lambda_min": float(cheby_lambda_min),
        "cheby_lambda_max": float(cheby_lambda_max),
        "chebyshev_enabled": bool(chebyshev_enabled),
        "chebyshev_reset_factor": float(chebyshev_reset_factor),
        "inner_forcing_eta": float(inner_forcing_eta),
        "inner_head_residual_tol_min": float(inner_head_residual_tol_min),
        "inner_head_residual_tol_max": float(inner_head_residual_tol_max),
        "transmissivity_relaxation_enabled": bool(transmissivity_relaxation_enabled),
        "unconfined_startup_mode": str(unconfined_startup_mode),
        "diag_preconditioner_backend": str(diag_preconditioner_backend),
        "smoother": str(smoother),
        "chebyshev_vertical_line": str(smoother) == "chebyshev_vertical_line",
        "line_omega": float(line_omega),
        "line_sweeps_pre": int(line_sweeps_pre),
        "line_sweeps_post": int(line_sweeps_post),
        "line_sweeps_coarse": int(line_sweeps_coarse),
    }

    np.savez_compressed(
        out_path,
        heads=np.asarray(heads, dtype=np.float64),
        total_time=np.asarray(total_time, dtype=np.float64),
        solve1_time=np.asarray(solve1_time, dtype=np.float64),
        solve2_time=np.asarray(solve2_time, dtype=np.float64),
        info=np.asarray(json.dumps(info, default=str)),
        info_solve1=np.asarray(json.dumps(info1, default=str) if info1 else "{}"),
        info_solve2=np.asarray(json.dumps(info, default=str)),
        summary_solve1=np.asarray(json.dumps(solve1_summary, default=str)),
        summary_solve2=np.asarray(json.dumps(solve2_summary, default=str)),
        adaptive_settings=np.asarray(json.dumps(adaptive_settings, default=str)),
        speed_controls=np.asarray(json.dumps(speed_controls, default=str)),
        solve1_mode=np.asarray(solve1_mode),
        solve2_mode=np.asarray(solve2_mode),
        solve1_settings=np.asarray(json.dumps(solve1_kwargs, default=str)),
        solve2_settings=np.asarray(json.dumps(solve2_kwargs, default=str)),
        nx=np.asarray(case.nx, dtype=np.int32),
        ny=np.asarray(case.ny, dtype=np.int32),
        nlay=np.asarray(case.nlay, dtype=np.int32),
        dx=np.asarray(case.dx, dtype=np.float64),
        dz=np.asarray(case.dz, dtype=np.float64),
        device=np.asarray(device),
        solver=np.asarray(solver),
        smoother=np.asarray(smoother),
    )
    print(f"Warp heads saved to {out_path}")
    print(
        f"Warp metrics - Total time: {total_time:.4f}s, "
        f"solve1({solve1_mode}): {solve1_time:.4f}s, "
        f"solve2({solve2_mode}): {solve2_time:.4f}s"
    )
    if isinstance(info, dict):
        print("Warp Convergence Info (solve 2):")
        for k, v in info.items():
            print(f"  {k}: {v}")
    else:
        print(f"Warp Convergence Info: {info}")
    print()
    return out_path

def load_results(
    mf6_path: str | Path,
    warp_path: str | Path,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load saved MF6 and Warp head arrays.
    """
    with np.load(mf6_path, allow_pickle=False) as mf6_npz:
        mf6_heads = np.asarray(mf6_npz["heads"], dtype=np.float64)
    with np.load(warp_path, allow_pickle=False) as warp_npz:
        warp_heads = np.asarray(warp_npz["heads"], dtype=np.float64)
    return mf6_heads, warp_heads


def compare_results(
    mf6_path: str | Path,
    warp_path: str | Path,
    active_3d: np.ndarray | None = None,
) -> dict[str, float]:
    """
    Load saved results and compare Warp heads against MF6 heads over all layers.
    """
    mf6_heads, warp_heads = load_results(mf6_path, warp_path)
    if mf6_heads.shape != warp_heads.shape:
        raise ValueError(f"Shape mismatch: MF6 {mf6_heads.shape}, Warp {warp_heads.shape}")

    if active_3d is None:
        mask = np.isfinite(mf6_heads) & np.isfinite(warp_heads)
    else:
        mask = (np.asarray(active_3d) != 0) & np.isfinite(mf6_heads) & np.isfinite(warp_heads)
        if mask.shape != mf6_heads.shape:
            raise ValueError(f"active_3d shape {mask.shape} does not match heads {mf6_heads.shape}")

    diff = np.asarray(warp_heads - mf6_heads, dtype=np.float64)
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

    print("\nWarp vs MF6 head comparison, all active cells and layers")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6g}")
    return metrics


def _load_npz_scalar(npz_path: Path, name: str, default: float | None = None) -> float | None:
    if not npz_path.exists():
        return default
    with np.load(npz_path, allow_pickle=False) as data:
        if name not in data:
            return default
        return float(np.asarray(data[name]).reshape(()))


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


def _load_npz_string(npz_path: Path, name: str, default: str | None = None) -> str | None:
    if not npz_path.exists():
        return default
    with np.load(npz_path, allow_pickle=False) as data:
        if name not in data:
            return default
        return str(np.asarray(data[name]).reshape(()))


def _load_warp_info(npz_path: Path) -> dict:
    return _load_npz_json(npz_path, "info")


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


def _convergence_report(
    info: dict,
    comparison: dict[str, float] | None = None,
    dh_tol: float = DEFAULT_DH_TOL,
    residual_floor_tol: float = DEFAULT_RESIDUAL_FLOOR_TOL,
    mf6_agreement_tol: float = DEFAULT_MF6_AGREEMENT_TOL,
) -> dict:
    r_rms_end = _finite_float(info.get("r_rms_end"))
    tol_abs = _finite_float(info.get("tol_abs"))
    dh_rms_lastcheck = _finite_float(info.get("dh_rms_lastcheck"))
    max_abs_diff = _finite_float((comparison or {}).get("max_abs_diff"))

    residual_converged = None
    if r_rms_end is not None and tol_abs is not None:
        residual_converged = bool(r_rms_end <= tol_abs)

    head_change_converged = None
    if dh_rms_lastcheck is not None:
        head_change_converged = bool(dh_rms_lastcheck <= float(dh_tol))

    practically_converged = None
    if head_change_converged is not None and r_rms_end is not None:
        practically_converged = bool(head_change_converged and r_rms_end <= float(residual_floor_tol))

    agrees_with_mf6 = None
    if max_abs_diff is not None:
        agrees_with_mf6 = bool(max_abs_diff < float(mf6_agreement_tol))

    if residual_converged:
        status = "Residual tolerance met."
    elif practically_converged and agrees_with_mf6:
        status = "Residual tolerance not met, but the solution is stationary and agrees with MF6 to < 5e-5 m."
    elif practically_converged:
        status = "Residual tolerance not met, but practical convergence criteria are met."
    else:
        status = "Convergence criteria not met."

    return {
        "residual_converged": residual_converged,
        "head_change_converged": head_change_converged,
        "practically_converged": practically_converged,
        "agrees_with_mf6": agrees_with_mf6,
        "status": status,
        "r_rms_end": r_rms_end,
        "tol_abs": tol_abs,
        "dh_rms_lastcheck": dh_rms_lastcheck,
        "dh_tol": float(dh_tol),
        "residual_floor_tol": float(residual_floor_tol),
        "mf6_agreement_tol": float(mf6_agreement_tol),
        "max_abs_diff": max_abs_diff,
    }


def run_case(
    nx: int = 1000,
    ny: int = 200,
    nlay: int = 2,
    dx: float = 100.0,
    layer_thickness: float = 150.0,
    transmissivity: float = 3000.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    heterogeneous_t: bool = False,
    seed: int = 123,
    workspace: str | Path | None = None,
    device: str = "auto",
    solver: str = "kcycle",
    smoother: str = "chebyshev",
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
    adaptive_kcycle: bool = True,
    line_omega: float = 0.8,
    line_sweeps_pre: int = 1,
    line_sweeps_post: int = 1,
    line_sweeps_coarse: int = 1,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_double_solve: bool = True,
    chebyshev_enabled: bool = True,
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    chebyshev_reset_factor: float = 1.2,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = 1.0e-4,
    inner_head_residual_tol_max: float = 1.0e-2,
    transmissivity_relaxation_enabled: bool = False,
) -> dict[str, float]:
    case = build_simple_unconfined_multilayer_case(
        nx=nx,
        ny=ny,
        nlay=nlay,
        dx=dx,
        layer_thickness=layer_thickness,
        transmissivity=transmissivity,
        recharge=recharge,
        initial_saturated_thickness=initial_saturated_thickness,
        heterogeneous_t=heterogeneous_t,
        seed=seed,
        workspace=workspace,
    )

    print(f"Running {nlay}-layer case: nx={case.nx}, ny={case.ny}, dx={case.dx}, dz={case.dz}")
    print(f"Workspace: {case.workspace}\n")

    mf6_path = case.workspace.joinpath("mf6_heads.npz")
    warp_path = case.workspace.joinpath("warp_heads.npz")

    if do_run_mf6:
        run_mf6(case, out_path=mf6_path)
    if do_run_warp:
        run_warp_unconfined(
            case,
            out_path=warp_path,
            device=device,
            solver=solver,
            smoother=smoother,
            adaptive_kcycle=adaptive_kcycle,
            line_omega=line_omega,
            line_sweeps_pre=line_sweeps_pre,
            line_sweeps_post=line_sweeps_post,
            line_sweeps_coarse=line_sweeps_coarse,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
            do_double_solve=do_double_solve,
            chebyshev_enabled=chebyshev_enabled,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
        )

    if mf6_path.exists() and warp_path.exists():
        metrics = compare_results(mf6_path, warp_path, active_3d=case.active_3d)
        return metrics
    else:
        print("Skipping comparison because both MF6 and Warp heads were not generated or found.")
        return {}


def run_layer_benchmark(
    nx: int = 1000,
    ny: int = 1000,
    layers: list[int] | tuple[int, ...] = tuple(BENCHMARK_LAYERS),
    dx: float = 100.0,
    layer_thickness: float = 50.0,
    transmissivity: float = 3000.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    heterogeneous_t: bool = False,
    seed: int = 123,
    workspace: str | Path | None = None,
    device: str = "auto",
    solver: str = "kcycle",
    smoother: str | list[str] | tuple[str, ...] = "chebyshev",
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
    adaptive_kcycle: bool = True,
    line_omega: float = 0.8,
    line_sweeps_pre: int = 1,
    line_sweeps_post: int = 1,
    line_sweeps_coarse: int = 1,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_double_solve: bool = True,
    chebyshev_enabled: bool = True,
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    chebyshev_reset_factor: float = 1.2,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = 1.0e-4,
    inner_head_residual_tol_max: float = 1.0e-2,
    transmissivity_relaxation_enabled: bool = False,
) -> list[dict]:
    if workspace is None:
        workspace = data_store.joinpath("working_tests", "mf6_vs_warp_3d_unconfined_layer_benchmark")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    layer_values = [int(v) for v in layers]
    if isinstance(smoother, str):
        smoother_values = [str(smoother)]
    else:
        smoother_values = [str(v) for v in smoother]
    print("\n" + "=" * 72)
    print("3D Warp vs MF6 layer benchmark")
    print(f"grid: nx={nx}, ny={ny}, dx={dx}")
    print(f"layers: {layer_values}")
    print(f"smoothers: {smoother_values}")
    print(f"workspace: {workspace}")
    print("=" * 72)

    results_dict: dict[tuple[str, int], dict] = {}
    summary_path = workspace.joinpath("layer_benchmark_summary.json")
    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                existing_results = json.load(f)
            for r in existing_results:
                key_smoother = str(r.get("smoother", r.get("warp_solver", {}).get("smoother", "")))
                results_dict[(key_smoother, int(r["nlay"]))] = r
        except Exception:
            pass

    for smoother_name in smoother_values:
        for nlay in layer_values:
            print("\n" + "-" * 72)
            print(f"Benchmark layer count: {nlay}, smoother: {smoother_name}")
            print("-" * 72)

            case_workspace = workspace.joinpath(f"layers_{nlay:02d}_{smoother_name}")
            metrics = run_case(
                nx=nx,
                ny=ny,
                nlay=nlay,
                dx=dx,
                layer_thickness=layer_thickness,
                transmissivity=transmissivity,
                recharge=recharge,
                initial_saturated_thickness=initial_saturated_thickness,
                heterogeneous_t=heterogeneous_t,
                seed=seed,
                workspace=case_workspace,
                device=device,
                solver=solver,
                smoother=smoother_name,
                do_run_mf6=do_run_mf6,
                do_run_warp=do_run_warp,
                adaptive_kcycle=adaptive_kcycle,
                line_omega=line_omega,
                line_sweeps_pre=line_sweeps_pre,
                line_sweeps_post=line_sweeps_post,
                line_sweeps_coarse=line_sweeps_coarse,
                unconfined_startup_mode=unconfined_startup_mode,
                diag_preconditioner_backend=diag_preconditioner_backend,
                check_every_no=check_every_no,
                do_double_solve=do_double_solve,
                chebyshev_enabled=chebyshev_enabled,
                cheby_lambda_min=cheby_lambda_min,
                cheby_lambda_max=cheby_lambda_max,
                chebyshev_reset_factor=chebyshev_reset_factor,
                inner_forcing_eta=inner_forcing_eta,
                inner_head_residual_tol_min=inner_head_residual_tol_min,
                inner_head_residual_tol_max=inner_head_residual_tol_max,
                transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            )

            mf6_path = case_workspace.joinpath("mf6_heads.npz")
            warp_path = case_workspace.joinpath("warp_heads.npz")
            warp_info = _load_warp_info(warp_path)
            warp_info_solve1 = _load_npz_json(warp_path, "info_solve1")
            warp_info_solve2 = _load_npz_json(warp_path, "info_solve2")
            warp_summary_solve1 = _load_npz_json(warp_path, "summary_solve1")
            warp_summary_solve2 = _load_npz_json(warp_path, "summary_solve2")
            warp_adaptive_settings = _load_npz_json(warp_path, "adaptive_settings")
            solve1_mode = _load_npz_string(warp_path, "solve1_mode")
            solve2_mode = _load_npz_string(warp_path, "solve2_mode")
            solve1_settings = _load_npz_json(warp_path, "solve1_settings")
            solve2_settings = _load_npz_json(warp_path, "solve2_settings")
            warp_speed_controls = _load_npz_json(warp_path, "speed_controls")
            solve1_report = _convergence_report(
                warp_info_solve1,
                comparison=metrics,
                dh_tol=float(solve1_settings.get("dh_rms_tol", DEFAULT_DH_TOL)),
                residual_floor_tol=DEFAULT_RESIDUAL_FLOOR_TOL,
                mf6_agreement_tol=DEFAULT_MF6_AGREEMENT_TOL,
            )
            solve2_report = _convergence_report(
                warp_info_solve2,
                comparison=metrics,
                dh_tol=float(solve2_settings.get("dh_rms_tol", DEFAULT_DH_TOL)),
                residual_floor_tol=DEFAULT_RESIDUAL_FLOOR_TOL,
                mf6_agreement_tol=DEFAULT_MF6_AGREEMENT_TOL,
            )
            solve2_r_rms_end = _finite_float(warp_info_solve2.get("r_rms_end"))
            solve2_tol_abs = _finite_float(warp_info_solve2.get("tol_abs"))
            solve2_dh_rms_lastcheck = _finite_float(warp_info_solve2.get("dh_rms_lastcheck"))
            solve2_dh_max_lastcheck = _finite_float(warp_info_solve2.get("dh_max_lastcheck"))

            row = {
                "nlay": int(nlay),
                "smoother": str(smoother_name),
                "line_omega": warp_info.get("line_omega"),
                "line_sweeps_pre": warp_info.get("line_sweeps_pre"),
                "line_sweeps_post": warp_info.get("line_sweeps_post"),
                "line_sweeps_coarse": warp_info.get("line_sweeps_coarse"),
                "check_every_no": warp_info.get("check_every_no"),
                "nx": int(nx),
                "ny": int(ny),
                "n_cells": int(nx * ny * nlay),
                "workspace": str(case_workspace),
                "solve2_converged": bool(warp_info_solve2.get("converged", False)) if warp_info_solve2 else None,
                "picard_converged": bool(warp_info_solve2.get("picard_converged", False)) if warp_info_solve2 else None,
                "picard_n_iter_used": warp_info_solve2.get("picard_n_iter_used") if warp_info_solve2 else None,
                "picard_dh_rms_end": warp_info_solve2.get("picard_dh_rms_end") if warp_info_solve2 else None,
                "picard_dh_max_end": warp_info_solve2.get("picard_dh_max_end") if warp_info_solve2 else None,
                "n_cycles_used": warp_info_solve2.get("n_cycles_used") if warp_info_solve2 else None,
                "r_rms_end": solve2_r_rms_end,
                "tol_abs": solve2_tol_abs,
                "dh_rms_lastcheck": solve2_dh_rms_lastcheck,
                "dh_max_lastcheck": solve2_dh_max_lastcheck,
                "solve2_outer_iterations": warp_info_solve2.get("outer_iterations"),
                "solve2_final_max_abs_head_change": warp_info_solve2.get("final_max_abs_head_change"),
                "solve2_final_residual": warp_info_solve2.get("final_residual"),
                "solve2_final_h_rms_inner_residual": warp_info_solve2.get("inner_h_rms_end"),
                "solve2_chebyshev_rejections": warp_info_solve2.get("chebyshev_rejections"),
                "solve2_chebyshev_resets": warp_info_solve2.get("chebyshev_resets"),
                "solve2_strict_inner_nonconvergence_count": warp_info_solve2.get("strict_inner_nonconvergence_count"),
                "solve2_unusable_inner_solve_count": warp_info_solve2.get("unusable_inner_solve_count"),
                "solve2_practical_inner_acceptance_count": warp_info_solve2.get("practical_inner_acceptance_count"),
                "solve2_accepted_picard_update_count": warp_info_solve2.get("accepted_picard_update_count"),
                "solve2_outer_chebyshev_ready_count": warp_info_solve2.get("outer_chebyshev_ready_count"),
                "solve2_outer_chebyshev_used_count": warp_info_solve2.get("outer_chebyshev_used_count"),
                "solve2_outer_chebyshev_reset_count": warp_info_solve2.get("outer_chebyshev_reset_count"),
                "solve2_inner_forcing_eta": warp_info_solve2.get("inner_forcing_eta"),
                "solve2_inner_head_residual_tol_min": warp_info_solve2.get("inner_head_residual_tol_min"),
                "solve2_inner_head_residual_tol_max": warp_info_solve2.get("inner_head_residual_tol_max"),
                "solve2_inner_solve_failures": warp_info_solve2.get("inner_solve_failures"),
                "solve2_effectively_dry_cell_count": warp_info_solve2.get("effectively_dry_cell_count"),
                "convergence_report": solve2_report,
                "timing": {
                    "mf6_engine_time": _load_npz_scalar(mf6_path, "engine_time"),
                    "mf6_total_time": _load_npz_scalar(mf6_path, "total_time"),
                    "warp_total_time": _load_npz_scalar(warp_path, "total_time"),
                    "warp_solve1_time": _load_npz_scalar(warp_path, "solve1_time"),
                    "warp_solve2_time": _load_npz_scalar(warp_path, "solve2_time"),
                    "warp_benchmark_time": _load_npz_scalar(warp_path, "solve2_time"),
                },
                "convergence": {
                    "solve1": {
                        "mode": solve1_mode,
                        "converged": bool(warp_info_solve1.get("converged", False)) if warp_info_solve1 else None,
                        "n_cycles_used": (
                            int(warp_info_solve1["n_cycles_used"])
                            if "n_cycles_used" in warp_info_solve1
                            else None
                        ),
                        "r_rms_end": (
                            float(warp_info_solve1["r_rms_end"])
                            if "r_rms_end" in warp_info_solve1
                            else None
                        ),
                        "settings": solve1_settings,
                        "summary": warp_summary_solve1,
                        "report": solve1_report,
                    },
                    "solve2": {
                        "mode": solve2_mode,
                        "converged": bool(warp_info_solve2.get("converged", False)) if warp_info_solve2 else None,
                        "n_cycles_used": (
                            int(warp_info_solve2["n_cycles_used"])
                            if "n_cycles_used" in warp_info_solve2
                            else None
                        ),
                        "r_rms_end": (
                            float(warp_info_solve2["r_rms_end"])
                            if "r_rms_end" in warp_info_solve2
                            else None
                        ),
                        "settings": solve2_settings,
                        "summary": warp_summary_solve2,
                        "report": solve2_report,
                    },
                    "benchmark_solve": "solve2",
                    "adaptive": {
                        "settings": warp_adaptive_settings,
                    },
                },
                "warp_solver": {
                    "solver_type": warp_info.get("solver_type"),
                    "smoother": warp_info.get("smoother"),
                    "line_omega": warp_info.get("line_omega"),
                    "line_sweeps_pre": warp_info.get("line_sweeps_pre"),
                    "line_sweeps_post": warp_info.get("line_sweeps_post"),
                    "line_sweeps_coarse": warp_info.get("line_sweeps_coarse"),
                    "check_every_no": warp_info.get("check_every_no"),
                    "n_levels": int(warp_info["n_levels"]) if "n_levels" in warp_info else None,
                    "level_shapes": warp_info.get("level_shapes"),
                },
                "comparison": metrics,
            }

            # Merge with existing row to preserve data from tools that weren't run this time
            result_key = (str(smoother_name), int(nlay))
            if result_key in results_dict:
                old_row = results_dict[result_key]
                if not do_run_mf6:
                    row["timing"]["mf6_engine_time"] = old_row.get("timing", {}).get("mf6_engine_time")
                    row["timing"]["mf6_total_time"] = old_row.get("timing", {}).get("mf6_total_time")
                if not do_run_warp:
                    row["timing"]["warp_total_time"] = old_row.get("timing", {}).get("warp_total_time")
                    row["timing"]["warp_solve1_time"] = old_row.get("timing", {}).get("warp_solve1_time")
                    row["timing"]["warp_solve2_time"] = old_row.get("timing", {}).get("warp_solve2_time")
                    row["timing"]["warp_benchmark_time"] = old_row.get("timing", {}).get("warp_benchmark_time")
                    row["convergence"] = old_row.get("convergence", {})
                    row["warp_solver"] = old_row.get("warp_solver", {})
                if not (do_run_mf6 and do_run_warp) and not metrics:
                    row["comparison"] = old_row.get("comparison", {})

            results_dict[result_key] = row

            results = [results_dict[k] for k in sorted(results_dict.keys())]
            with summary_path.open("w") as f:
                json.dump(results, f, indent=4)
            print(f"Updated benchmark summary: {summary_path}")
            print(f"Solve 2 convergence report: {solve2_report['status']}")

    print("\nLayer benchmark complete.")
    return [results_dict[k] for k in sorted(results_dict.keys())]


if __name__ == "__main__":
    # Configuration parameters
    nx = 250
    ny = 250
    layers = BENCHMARK_LAYERS
    dx = 100.0
    layer_thickness = 50.0
    transmissivity = 3000.0
    recharge = 1.0e-4
    initial_saturated_thickness = 100.0
    heterogeneous_t = False
    seed = 123
    workspace = None
    device = "auto"
    solver = "kcycle"
    smoother = 'chebyshev_vertical_line'
    do_run_mf6 = True
    do_run_warp = True
    adaptive_kcycle = True
    line_omega = 0.8
    line_sweeps_pre = 1
    line_sweeps_post = 1
    line_sweeps_coarse = 1
    unconfined_startup_mode = "confined_pre_solve"
    diag_preconditioner_backend = "device"
    check_every_no = 5
    do_double_solve = False
    chebyshev_enabled = True
    cheby_lambda_min = 0.1
    cheby_lambda_max = 2.0
    chebyshev_reset_factor = 1.2
    inner_forcing_eta = 0.10
    inner_head_residual_tol_min = 1.0e-4
    inner_head_residual_tol_max = 1.0e-2
    transmissivity_relaxation_enabled = False

    run_layer_benchmark(
        nx=nx,
        ny=ny,
        layers=layers,
        dx=dx,
        layer_thickness=layer_thickness,
        transmissivity=transmissivity,
        recharge=recharge,
        initial_saturated_thickness=initial_saturated_thickness,
        heterogeneous_t=heterogeneous_t,
        seed=seed,
        workspace=workspace,
        device=device,
        solver=solver,
        smoother=smoother,
        do_run_mf6=do_run_mf6,
        do_run_warp=do_run_warp,
        adaptive_kcycle=adaptive_kcycle,
        line_omega=line_omega,
        line_sweeps_pre=line_sweeps_pre,
        line_sweeps_post=line_sweeps_post,
        line_sweeps_coarse=line_sweeps_coarse,
        unconfined_startup_mode=unconfined_startup_mode,
        diag_preconditioner_backend=diag_preconditioner_backend,
        check_every_no=check_every_no,
        do_double_solve=do_double_solve,
        chebyshev_enabled=chebyshev_enabled,
        cheby_lambda_min=cheby_lambda_min,
        cheby_lambda_max=cheby_lambda_max,
        chebyshev_reset_factor=chebyshev_reset_factor,
        inner_forcing_eta=inner_forcing_eta,
        inner_head_residual_tol_min=inner_head_residual_tol_min,
        inner_head_residual_tol_max=inner_head_residual_tol_max,
        transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
    )
