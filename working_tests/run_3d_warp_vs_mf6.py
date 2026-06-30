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
from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model_multilayer  # noqa: E402
from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D  # noqa: E402


@dataclass(frozen=True)
class Case3D:
    nx: int
    ny: int
    nlay: int
    dx: float
    dz: float
    workspace: Path
    hk_2d: np.ndarray
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


def build_simple_multilayer_case(
    nx: int = 1000,
    ny: int = 200,
    nlay: int = 2,
    dx: float = 100.0,
    layer_thickness: float = 150.0,
    transmissivity: float = 3000.0,
    recharge: float = 1.0e-4,
    heterogeneous_t: bool = False,
    seed: int = 123,
    workspace: str | Path | None = None,
) -> Case3D:
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

    initial_head_3d = np.repeat(dem[np.newaxis, :, :], nlay, axis=0).astype(np.float64)

    return Case3D(
        nx=int(nx),
        ny=int(ny),
        nlay=nlay,
        dx=float(dx),
        dz=dz,
        workspace=workspace,
        hk_2d=hk_2d,
        recharge_2d=recharge_2d,
        active_3d=active_3d,
        bc_mask_3d=bc_mask_3d,
        bc_values_3d=bc_values_3d,
        rhs_3d=rhs_3d,
        initial_head_3d=initial_head_3d,
    )


def run_mf6(case: Case3D, out_path: str | Path | None = None) -> Path:
    """
    Run the multi-layer MF6 truth model and save heads to NPZ.
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


def run_warp(
    case: Case3D,
    out_path: str | Path | None = None,
    device: str = "auto",
    solver: str = "kcycle",
    smoother: str = "chebyshev",
) -> Path:
    """
    Run the same multi-layer problem in Warp and save heads to NPZ.
    """
    out_path = Path(out_path) if out_path is not None else case.workspace.joinpath("warp_heads.npz")
    device = _warp_device(device)
    solver = str(solver).lower()
    if solver not in {"kcycle", "chebyshev"}:
        raise ValueError("solver must be 'kcycle' or 'chebyshev'.")
    smoother = str(smoother).lower()
    if smoother not in {"chebyshev", "jacobi"}:
        raise ValueError("smoother must be 'chebyshev' or 'jacobi'.")
    if solver == "chebyshev":
        print("Using standalone Chebyshev debug path. Use solver='kcycle' for the MF6 comparison benchmark.")

    hk_3d = np.repeat(case.hk_2d[np.newaxis, :, :], case.nlay, axis=0)

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
            solve_kwargs = {
                "max_cycles": 200,
                "rel_tol": 5.0e-7,
                "abs_tol_min": 5.0e-7,
                "check_every_no": 1,
                "max_levels": 6,
                "smoother": smoother,
                "nu_pre": 10,
                "nu_post": 10,
                "nu_coarse": 3,
                "omega": 0.75,
            }
        else:
            solve_kwargs = {
                "max_iter": 400,
                "rel_tol": 5.0e-7,
                "abs_tol_min": 5.0e-7,
            }
        heads, info = warp_solver.solve(**solve_kwargs)
    total_time = time.perf_counter() - t0

    np.savez_compressed(
        out_path,
        heads=np.asarray(heads, dtype=np.float64),
        total_time=np.asarray(total_time, dtype=np.float64),
        info=np.asarray(json.dumps(info, default=str)),
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
    print(f"Warp metrics - Total time: {total_time:.4f}s")
    if isinstance(info, dict):
        print("Warp Convergence Info:")
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


def run_case(
    nx: int = 1000,
    ny: int = 200,
    nlay: int = 2,
    dx: float = 100.0,
    layer_thickness: float = 150.0,
    transmissivity: float = 3000.0,
    recharge: float = 1.0e-4,
    heterogeneous_t: bool = False,
    seed: int = 123,
    workspace: str | Path | None = None,
    device: str = "auto",
    solver: str = "kcycle",
    smoother: str = "chebyshev",
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
) -> dict[str, float]:
    case = build_simple_multilayer_case(
        nx=nx,
        ny=ny,
        nlay=nlay,
        dx=dx,
        layer_thickness=layer_thickness,
        transmissivity=transmissivity,
        recharge=recharge,
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
        run_warp(case, out_path=warp_path, device=device, solver=solver, smoother=smoother)

    if mf6_path.exists() and warp_path.exists():
        metrics = compare_results(mf6_path, warp_path, active_3d=case.active_3d)
        metrics_path = case.workspace.joinpath("comparison_metrics.json")
        with metrics_path.open("w") as f:
            json.dump(metrics, f, indent=4)
        print(f"Comparison metrics saved to {metrics_path}")
        return metrics
    else:
        print("Skipping comparison because both MF6 and Warp heads were not generated or found.")
        return {}


if __name__ == "__main__":
    # Configuration parameters
    nx = 500
    ny = 500
    nlay = 20
    dx = 100.0
    layer_thickness = 50.0
    transmissivity = 3000.0
    recharge = 1.0e-4
    heterogeneous_t = False
    seed = 123
    workspace = None
    device = "auto"
    solver = "kcycle"
    smoother = "chebyshev"
    do_run_mf6 = True
    do_run_warp = True

    run_case(
        nx=nx,
        ny=ny,
        nlay=nlay,
        dx=dx,
        layer_thickness=layer_thickness,
        transmissivity=transmissivity,
        recharge=recharge,
        heterogeneous_t=heterogeneous_t,
        seed=seed,
        workspace=workspace,
        device=device,
        solver=solver,
        smoother=smoother,
        do_run_mf6=do_run_mf6,
        do_run_warp=do_run_warp,
    )
