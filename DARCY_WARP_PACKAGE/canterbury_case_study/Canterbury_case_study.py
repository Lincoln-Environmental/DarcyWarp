from __future__ import annotations

import argparse
import time
from pathlib import Path
import json

import numpy as np
import torch
from scipy.optimize import dual_annealing, minimize, OptimizeResult

from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import (
    CanterburyCaseInputs,
    export_active_cells_geotiff,
    load_case_inputs,
)
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


def _prepare_obs_arrays(case: CanterburyCaseInputs):
    df = case.obs_df
    if not {"i", "j", "gwl", "std_gwl"}.issubset(df.columns):
        raise ValueError("obs_df must include i, j, gwl, std_gwl columns")

    i_idx = df["i"].astype(int).to_numpy()
    j_idx = df["j"].astype(int).to_numpy()

    in_bounds = (
            (i_idx >= 0)
            & (i_idx < case.ny)
            & (j_idx >= 0)
            & (j_idx < case.nx)
    )

    i_idx = i_idx[in_bounds]
    j_idx = j_idx[in_bounds]
    gwl = df.loc[in_bounds, "gwl"].to_numpy(dtype=float)
    std = df.loc[in_bounds, "std_gwl"].to_numpy(dtype=float)

    active_mask = case.active[i_idx, j_idx] == 1

    i_idx = i_idx[active_mask]
    j_idx = j_idx[active_mask]
    gwl = gwl[active_mask]
    std = std[active_mask]

    weights = 1.0 / np.clip(std, 0.2, None)
    return i_idx, j_idx, gwl, weights


def _weighted_sse(residual: np.ndarray, weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    if np.all(w <= 0.0):
        return float(np.sum(residual * residual))
    # Use sum((w_i * r_i)^2) = sum(w_i^2 * r_i^2)
    return float(np.sum((w * residual) ** 2))


def _compute_pilot_active_mask(
        active: np.ndarray,
        nx_p: int,
        ny_p: int,
        touch_radius: int = 1,
        pilot_neighbor_buffer: int = 1,
) -> np.ndarray:
    """
    Mark pilot points as active only if their mapped fine-grid cell touches active domain.

    A pilot maps to the nearest fine-grid index. It is kept active when at least one
    active cell exists in a square neighborhood of radius ``touch_radius`` around that
    mapped index.
    """
    if nx_p < 2 or ny_p < 2:
        raise ValueError("nx_p and ny_p must be >= 2")
    if int(touch_radius) < 0:
        raise ValueError("touch_radius must be >= 0")
    if int(pilot_neighbor_buffer) < 0:
        raise ValueError("pilot_neighbor_buffer must be >= 0")

    active_arr = np.asarray(active, dtype=np.int32)
    ny, nx = active_arr.shape

    pilot_x_coarse, pilot_y_coarse = np.meshgrid(
        np.arange(nx_p, dtype=np.float64),
        np.arange(ny_p, dtype=np.float64),
    )

    max_fine_x = float(nx - 1)
    max_fine_y = float(ny - 1)
    pilot_xs = max_fine_x / float(nx_p - 1) * pilot_x_coarse
    pilot_ys = max_fine_y / float(ny_p - 1) * pilot_y_coarse

    pilot_j = np.rint(pilot_xs).astype(np.int32)
    pilot_i = np.rint(pilot_ys).astype(np.int32)

    r = int(touch_radius)
    pilot_active = np.zeros((ny_p, nx_p), dtype=bool)
    for py in range(ny_p):
        for px in range(nx_p):
            i0 = max(0, int(pilot_i[py, px]) - r)
            i1 = min(ny, int(pilot_i[py, px]) + r + 1)
            j0 = max(0, int(pilot_j[py, px]) - r)
            j1 = min(nx, int(pilot_j[py, px]) + r + 1)
            pilot_active[py, px] = bool(np.any(active_arr[i0:i1, j0:j1] != 0))

    # Buffer active pilots in pilot-grid space so first nearest neighbors stay active.
    for _ in range(int(pilot_neighbor_buffer)):
        if np.all(pilot_active):
            break
        p = np.pad(pilot_active.astype(np.int8, copy=False), ((1, 1), (1, 1)), mode="constant")
        neighbors_on = (
                (p[0:-2, 0:-2] != 0) | (p[0:-2, 1:-1] != 0) | (p[0:-2, 2:] != 0)
                | (p[1:-1, 0:-2] != 0) | (p[1:-1, 1:-1] != 0) | (p[1:-1, 2:] != 0)
                | (p[2:, 0:-2] != 0) | (p[2:, 1:-1] != 0) | (p[2:, 2:] != 0)
        )
        pilot_active = pilot_active | neighbors_on

    return pilot_active


def _print_pilot_activity(
        pilot_nx: int,
        pilot_ny: int,
        pilot_active_count: int,
        pilot_total_count: int,
        deactivate_outside_pilots: bool,
        pilot_touch_radius: int,
        pilot_neighbor_buffer: int,
) -> None:
    pilot_inactive_count = int(pilot_total_count) - int(pilot_active_count)
    print(
        "Pilot points "
        f"({int(pilot_nx)}x{int(pilot_ny)}): "
        f"total={int(pilot_total_count)}, "
        f"active={int(pilot_active_count)}, "
        f"inactive={pilot_inactive_count}, "
        f"deactivate_outside_pilots={bool(deactivate_outside_pilots)}, "
        f"touch_radius={int(pilot_touch_radius)}, "
        f"pilot_neighbor_buffer={int(pilot_neighbor_buffer)}"
    )


def _run_pso(
        objective,
        bounds: list[tuple[float, float]],
        x0: np.ndarray,
        maxiter: int,
        swarm_size: int,
        seed: int,
        inertia: float = 0.72,
        cognitive: float = 1.49,
        social: float = 1.49,
) -> dict:
    """Basic global-best PSO for box-constrained objectives."""
    if int(swarm_size) < 2:
        raise ValueError("swarm_size must be >= 2 for PSO.")
    if int(maxiter) < 1:
        raise ValueError("maxiter must be >= 1 for PSO.")

    lower = np.asarray([float(lo) for lo, _ in bounds], dtype=np.float64)
    upper = np.asarray([float(hi) for _, hi in bounds], dtype=np.float64)
    if np.any(upper <= lower):
        raise ValueError("All PSO bounds must satisfy upper > lower.")
    n_dim = lower.size

    x0_arr = np.asarray(x0, dtype=np.float64).reshape(-1)
    if x0_arr.size != n_dim:
        raise ValueError(f"x0 size {x0_arr.size} does not match bounds dimension {n_dim}.")

    rng = np.random.default_rng(int(seed))
    positions = rng.uniform(lower, upper, size=(int(swarm_size), n_dim))
    positions[0, :] = np.clip(x0_arr, lower, upper)

    span = upper - lower
    velocities = rng.uniform(-1.0, 1.0, size=(int(swarm_size), n_dim)) * (0.1 * span)

    pbest_pos = positions.copy()
    pbest_val = np.empty(int(swarm_size), dtype=np.float64)
    nfev = 0

    for i in range(int(swarm_size)):
        f_i = float(objective(positions[i, :]))
        if not np.isfinite(f_i):
            f_i = np.inf
        pbest_val[i] = f_i
        nfev += 1

    g_idx = int(np.argmin(pbest_val))
    gbest_pos = pbest_pos[g_idx, :].copy()
    gbest_val = float(pbest_val[g_idx])

    for it in range(int(maxiter)):
        r1 = rng.random((int(swarm_size), n_dim))
        r2 = rng.random((int(swarm_size), n_dim))
        velocities = (
                float(inertia) * velocities
                + float(cognitive) * r1 * (pbest_pos - positions)
                + float(social) * r2 * (gbest_pos[None, :] - positions)
        )

        positions = positions + velocities
        positions = np.clip(positions, lower, upper)

        for i in range(int(swarm_size)):
            f_i = float(objective(positions[i, :]))
            if not np.isfinite(f_i):
                f_i = np.inf
            nfev += 1

            if f_i < pbest_val[i]:
                pbest_val[i] = f_i
                pbest_pos[i, :] = positions[i, :]
                if f_i < gbest_val:
                    gbest_val = float(f_i)
                    gbest_pos = positions[i, :].copy()

        print(f"pso iter {it + 1}/{int(maxiter)} best_obj={gbest_val:.4e}")

    return {
        "x": gbest_pos.astype(np.float64, copy=False),
        "fun": float(gbest_val),
        "nit": int(maxiter),
        "nfev": int(nfev),
        "success": True,
        "status": 0,
        "message": "PSO completed.",
    }


class _DualAnnealingEarlyStop(RuntimeError):
    """Raised to stop dual annealing on user-defined early-stop criteria."""


def _is_meaningful_improvement(
        prev_best: float,
        new_best: float,
        rel_improve: float,
        abs_improve: float,
) -> bool:
    if not np.isfinite(prev_best):
        return True
    delta = float(prev_best) - float(new_best)
    if delta <= 0.0:
        return False
    threshold = max(float(abs_improve), abs(float(prev_best)) * float(rel_improve))
    return bool(delta >= threshold)


def precompute_rbf_cache(
        nx: int,
        ny: int,
        nx_p: int,
        ny_p: int,
        epsilon: float = 10.0,
) -> dict:
    """
    Precompute matrices for fast RBF interpolation from pilot points to grid.
    """
    pilot_x_coarse, pilot_y_coarse = np.meshgrid(
        np.arange(nx_p, dtype=np.float32),
        np.arange(ny_p, dtype=np.float32),
    )

    max_fine_x = float(nx - 1)
    max_fine_y = float(ny - 1)

    pilot_xs = max_fine_x / float(nx_p - 1) * pilot_x_coarse
    pilot_ys = max_fine_y / float(ny_p - 1) * pilot_y_coarse

    pilot_points = np.vstack((pilot_xs.ravel(), pilot_ys.ravel())).T
    n_p = pilot_points.shape[0]

    fine_x, fine_y = np.meshgrid(
        np.arange(nx, dtype=np.float32),
        np.arange(ny, dtype=np.float32),
    )
    fine_coords = np.vstack((fine_x.ravel(), fine_y.ravel())).T

    dx_p = pilot_points[:, 0][:, None] - pilot_points[:, 0][None, :]
    dy_p = pilot_points[:, 1][:, None] - pilot_points[:, 1][None, :]
    d2_p = dx_p * dx_p + dy_p * dy_p

    eps2 = float(epsilon * epsilon)
    A = np.sqrt(d2_p / eps2 + 1.0).astype(np.float32)

    dx_fp = fine_coords[:, 0][:, None] - pilot_points[:, 0][None, :]
    dy_fp = fine_coords[:, 1][:, None] - pilot_points[:, 1][None, :]
    d2_fp = dx_fp * dx_fp + dy_fp * dy_fp

    B = np.sqrt(d2_fp / eps2 + 1.0).astype(np.float32)

    try:
        A_inv = np.linalg.inv(A).astype(np.float32)
    except np.linalg.LinAlgError:
        A_inv = np.linalg.pinv(A).astype(np.float32)

    return {
        "A_inv": A_inv,
        "B": B,
        "nx": int(nx),
        "ny": int(ny),
        "nx_p": int(nx_p),
        "ny_p": int(ny_p),
        "epsilon": float(epsilon),
    }


def get_T_field_from_pilots_cached(
        T_pilot_raw: np.ndarray,
        T_min: float,
        T_max: float,
        rbf_cache: dict,
        perturbation_strength: float = 0.0,
) -> np.ndarray:
    """
    RBF interpolation of pilot-point transmissivity to full grid.
    """
    A_inv = rbf_cache["A_inv"]
    B = rbf_cache["B"]
    nx = rbf_cache["nx"]
    ny = rbf_cache["ny"]
    ny_p = rbf_cache["ny_p"]
    nx_p = rbf_cache["nx_p"]

    T_pilot_raw = np.asarray(T_pilot_raw, dtype=np.float32)
    if T_pilot_raw.shape != (ny_p, nx_p):
        raise ValueError(
            f"T_pilot_raw shape {T_pilot_raw.shape} does not match "
            f"(ny_p, nx_p)=({ny_p}, {nx_p})"
        )

    if perturbation_strength > 0.0:
        perturb = np.random.randn(*T_pilot_raw.shape).astype(np.float32) * perturbation_strength
        T_pilot = T_pilot_raw + perturb
    else:
        T_pilot = T_pilot_raw.copy()

    T_pilot = np.clip(T_pilot, T_min, T_max)
    T_pilot[T_pilot <= 0.0] = T_min

    values = np.log(T_pilot + 1.0e-8).astype(np.float32, copy=False).ravel()
    coeffs = A_inv @ values
    T_field_flat_log = B @ coeffs

    T_field_flat = np.exp(T_field_flat_log).astype(np.float32, copy=False)
    T_field = T_field_flat.reshape(ny, nx)

    if np.isnan(T_field).any():
        nan_mask = np.isnan(T_field)
        if np.all(nan_mask):
            T_field[:] = T_min
        else:
            mean_val = float(np.nanmean(T_field))
            T_field[nan_mask] = mean_val

    return np.clip(T_field, T_min, T_max)


def _resample_pilot_grid(logT_grid: np.ndarray, ny_new: int, nx_new: int) -> np.ndarray:
    ny_old, nx_old = logT_grid.shape
    if ny_old == ny_new and nx_old == nx_new:
        return logT_grid.copy()

    x_old = np.linspace(0.0, 1.0, nx_old)
    x_new = np.linspace(0.0, 1.0, nx_new)
    y_old = np.linspace(0.0, 1.0, ny_old)
    y_new = np.linspace(0.0, 1.0, ny_new)

    tmp = np.empty((ny_old, nx_new), dtype=float)
    for j in range(ny_old):
        tmp[j, :] = np.interp(x_new, x_old, logT_grid[j, :])

    out = np.empty((ny_new, nx_new), dtype=float)
    for i in range(nx_new):
        out[:, i] = np.interp(y_new, y_old, tmp[:, i])

    return out


def dual_annealing_calibration(
        case: CanterburyCaseInputs,
        device: str = "cuda:0",
        use_ghb: bool = True,
        gh_alpha: float = 1.0,
        aq_thickness: float = 300.0,
        T_min: float = 100.0,
        T_max: float = 20000.0,
        R_scale_min: float = 0.5,
        R_scale_max: float = 0.9,
        r_scale_fixed: float | None = 0.6,
        pilot_nx: int = 9,
        pilot_ny: int = 9,
        rbf_epsilon: float = 10.0,
        perturbation_strength: float = 0.0,
        reg_weight: float = 1e1,
        sigma_dlogT: float = 0.2,
        pilot_logt_init: float | np.ndarray | None = None,
        pilot_init_jitter: float = 0.0,
        deactivate_outside_pilots: bool = True,
        pilot_touch_radius: int = 1,
        pilot_neighbor_buffer: int = 1,
        max_cycles: int = 200,
        nu_pre: int = 2,
        nu_post: int = 2,
        nu_coarse: int = 2,
        omega: float = 0.7,
        rel_tol: float = 5.0e-7,
        abs_tol_min: float = 5.0e-7,
        max_levels: int = 6,
        check_every_no: int = 1,
        maxiter: int = 20,
        maxfun: int | None = None,
        seed: int = 42,
        local_maxiter: int = 10,
        no_local_search: bool = False,
        stagnation_patience_nfev: int | None = None,
        stagnation_rel_improve: float = 5.0e-3,
        stagnation_abs_improve: float = 0.0,
        stagnation_min_nfev: int = 0,
        optimizer_algorithm: str = "dual_annealing",
        pso_swarm_size: int = 16,
        pso_inertia: float = 0.72,
        pso_cognitive: float = 1.49,
        pso_social: float = 1.49,
) -> dict:
    """
    Short calibration using dual annealing for pilot-point T and optional recharge scale.

    :param case: Prepared Canterbury case inputs.
    :param device: Warp device string, e.g. "cuda:0".
    :param use_ghb: Toggle general-head boundary terms.
    :param gh_alpha: GHB conductance scaling factor.
    :param aq_thickness: Aquifer thickness used to convert T to K where needed.
    :param T_min: Minimum transmissivity bound for pilot points.
    :param T_max: Maximum transmissivity bound for pilot points.
    :param R_scale_min: Minimum recharge scale (only if r_scale_fixed is None).
    :param R_scale_max: Maximum recharge scale (only if r_scale_fixed is None).
    :param r_scale_fixed: Fixed recharge scale; set to None to optimize.
    :param pilot_nx: Pilot grid columns.
    :param pilot_ny: Pilot grid rows.
    :param rbf_epsilon: RBF shape parameter for pilot interpolation.
    :param perturbation_strength: Random perturbation applied to pilot values (0 disables).
    :param reg_weight: Weight applied to the smoothness penalty.
    :param sigma_dlogT: 1-sigma expected adjacent pilot log10(T) difference.
    :param pilot_logt_init: Optional initial log10(T) pilot values.
    :param pilot_init_jitter: Random jitter added to initial pilots.
    :param deactivate_outside_pilots: If True, freeze pilots not touching active domain.
    :param pilot_touch_radius: Neighborhood radius (in fine cells) for pilot/domain touch.
    :param pilot_neighbor_buffer: Number of pilot-grid neighbor rings to keep active.
    :param max_cycles: Max multigrid cycles per forward solve.
    :param nu_pre: Pre-smoothing iterations.
    :param nu_post: Post-smoothing iterations.
    :param nu_coarse: Coarse-grid iterations.
    :param omega: Jacobi damping parameter.
    :param rel_tol: Relative residual tolerance.
    :param abs_tol_min: Minimum absolute residual tolerance.
    :param max_levels: Max multigrid levels.
    :param check_every_no: Residual check interval.
    :param maxiter: Dual annealing iterations or PSO iterations.
    :param maxfun: Optional hard cap on objective calls for dual annealing.
    :param seed: RNG seed.
    :param local_maxiter: Powell local search max iterations.
    :param no_local_search: Disable dual-annealing local search when True.
    :param stagnation_patience_nfev: Stop if no meaningful improvement for this many objective calls.
    :param stagnation_rel_improve: Relative improvement threshold used by stagnation check.
    :param stagnation_abs_improve: Absolute improvement threshold used by stagnation check.
    :param stagnation_min_nfev: Minimum objective calls before stagnation check is active.
    :param optimizer_algorithm: "dual_annealing" or "pso".
    :param pso_swarm_size: Particle count if optimizer_algorithm="pso".
    :param pso_inertia: Inertia coefficient for PSO.
    :param pso_cognitive: Cognitive coefficient for PSO.
    :param pso_social: Social coefficient for PSO.
    :return: Dict with optimizer outputs and best parameters.
    """
    obs_i, obs_j, obs_gwl, obs_w = _prepare_obs_arrays(case)

    logT_min = float(np.log10(T_min))
    logT_max = float(np.log10(T_max))

    if pilot_nx < 2 or pilot_ny < 2:
        raise ValueError("pilot_nx and pilot_ny must be >= 2")
    if maxfun is not None and int(maxfun) < 1:
        raise ValueError("maxfun must be >= 1 when provided.")
    if stagnation_patience_nfev is not None and int(stagnation_patience_nfev) < 1:
        raise ValueError("stagnation_patience_nfev must be >= 1 when provided.")
    if float(stagnation_rel_improve) < 0.0:
        raise ValueError("stagnation_rel_improve must be >= 0.")
    if float(stagnation_abs_improve) < 0.0:
        raise ValueError("stagnation_abs_improve must be >= 0.")
    if int(stagnation_min_nfev) < 0:
        raise ValueError("stagnation_min_nfev must be >= 0.")

    rbf_cache = precompute_rbf_cache(
        nx=case.nx,
        ny=case.ny,
        nx_p=pilot_nx,
        ny_p=pilot_ny,
        epsilon=float(rbf_epsilon),
    )

    if deactivate_outside_pilots:
        pilot_active_mask = _compute_pilot_active_mask(
            active=case.active,
            nx_p=pilot_nx,
            ny_p=pilot_ny,
            touch_radius=int(pilot_touch_radius),
            pilot_neighbor_buffer=int(pilot_neighbor_buffer),
        )
    else:
        pilot_active_mask = np.ones((pilot_ny, pilot_nx), dtype=bool)
    pilot_active_flat = pilot_active_mask.reshape(-1)
    n_pilots_total = int(pilot_nx * pilot_ny)
    n_pilots_active = int(np.count_nonzero(pilot_active_flat))
    if n_pilots_active == 0:
        raise ValueError("No active pilot points after applying pilot activity mask.")
    _print_pilot_activity(
        pilot_nx=pilot_nx,
        pilot_ny=pilot_ny,
        pilot_active_count=n_pilots_active,
        pilot_total_count=n_pilots_total,
        deactivate_outside_pilots=deactivate_outside_pilots,
        pilot_touch_radius=pilot_touch_radius,
        pilot_neighbor_buffer=pilot_neighbor_buffer,
    )

    optimize_r_scale = r_scale_fixed is None
    bounds = [(logT_min, logT_max) for _ in range(n_pilots_active)]
    if optimize_r_scale:
        bounds.append((float(R_scale_min), float(R_scale_max)))

    best = {
        "objective": np.inf,
        "wsse": np.inf,
        "params": None,
        "T_pilots": None,
        "logT": None,
        "x": None,
        "pilot_active_mask": pilot_active_mask.copy(),
    }

    logT_mid = 0.5 * (logT_min + logT_max)
    if pilot_logt_init is None:
        pilot_logt_init = logT_mid
    T_init_val = float(10.0 ** logT_mid)
    T_init = np.full((case.ny, case.nx), T_init_val, dtype=np.float32)
    T_init = np.where(case.active == 1, T_init, 0.0).astype(np.float32, copy=False)
    if optimize_r_scale:
        r_scale_init = float(0.5 * (R_scale_min + R_scale_max))
    else:
        r_scale_init = float(r_scale_fixed)
    R_init = case.recharge_base * float(r_scale_init)
    R_init = np.where(case.active == 1, R_init, 0.0).astype(np.float32, copy=False)

    init_rng = np.random.default_rng(int(seed))
    if np.ndim(pilot_logt_init) == 0:
        logT0_full = np.full(n_pilots_total, float(pilot_logt_init), dtype=np.float64)
    else:
        logT_init_arr = np.asarray(pilot_logt_init, dtype=np.float64)
        if logT_init_arr.shape == (pilot_ny, pilot_nx):
            logT0_full = logT_init_arr.reshape(-1).copy()
        elif logT_init_arr.shape == (n_pilots_total,):
            logT0_full = logT_init_arr.copy()
        else:
            raise ValueError(
                f"pilot_logt_init shape {logT_init_arr.shape} does not match "
                f"(pilot_ny, pilot_nx)=({pilot_ny}, {pilot_nx}) or ({n_pilots_total},)"
            )

    if float(pilot_init_jitter) > 0.0:
        logT0_full = logT0_full + init_rng.normal(0.0, float(pilot_init_jitter), size=n_pilots_total)

    logT0_full = np.clip(logT0_full, logT_min, logT_max)
    if optimize_r_scale:
        x0 = np.concatenate([logT0_full[pilot_active_flat], [float(r_scale_init)]]).astype(np.float64, copy=False)
    else:
        x0 = logT0_full[pilot_active_flat].astype(np.float64, copy=False)

    with WarpDarcySolver(
            nx=case.nx,
            ny=case.ny,
            dx=case.dx,
            device=device,
            use_ghb=use_ghb,
            solver_type="pcg",
            aq_thickness=float(aq_thickness),
        ) as solver:
        solver.build_from_fields(
            T_field=T_init,
            R_field=R_init,
            active=case.active,
            bc_mask=case.bc_mask,
            bc_values=case.bc_values,
            gh_mask=case.gh_mask,
            gh_head=case.gh_head,
            gh_width=case.gh_width,
            gh_alpha=float(gh_alpha),
        )
        solver.build_hierarchy(max_levels=6, min_coarse_n=4)
        nfev_counter = {"n": 0}
        stagnation_state = {
            "last_meaningful_improve_nfev": 0,
            "stop_reason": None,
        }

        def objective(x: np.ndarray) -> float:
            nfev_counter["n"] += 1
            if optimize_r_scale:
                logT_active = np.asarray(x[:-1], dtype=np.float32)
                r_scale = float(x[-1])
            else:
                logT_active = np.asarray(x, dtype=np.float32)
                r_scale = float(r_scale_fixed)

            if logT_active.size != n_pilots_active:
                raise ValueError(
                    f"logT parameter size {logT_active.size} does not match active pilot count {n_pilots_active}"
                )

            logT_flat_full = np.asarray(logT0_full, dtype=np.float32).copy()
            logT_flat_full[pilot_active_flat] = logT_active
            logT_grid = logT_flat_full.reshape(pilot_ny, pilot_nx)
            T_pilots = np.power(10.0, logT_grid).reshape(pilot_ny, pilot_nx)
            T_field = get_T_field_from_pilots_cached(
                T_pilot_raw=T_pilots,
                T_min=T_min,
                T_max=T_max,
                rbf_cache=rbf_cache,
                perturbation_strength=float(perturbation_strength),
            )
            T_field = np.where(case.active == 1, T_field, 0.0).astype(np.float32, copy=False)

            R_field = case.recharge_base * float(r_scale)
            R_field = np.where(case.active == 1, R_field, 0.0).astype(np.float32, copy=False)

            t0 = time.perf_counter()
            solver.update_T_in_place(T_field)
            solver.update_R_in_place(R_field)
            head, info = solver.solve_multigrid_kcycle(
                max_cycles=int(max_cycles),
                nu_pre=int(nu_pre),
                nu_post=int(nu_post),
                nu_coarse=int(nu_coarse),
                omega=float(omega),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                initial_head=case.model_top,
                return_info=True,
                max_levels=int(max_levels),
                check_every_no=int(check_every_no),
            )
            t1 = time.perf_counter()

            pred = head[obs_i, obs_j]
            residual = pred - obs_gwl
            phi_obs = _weighted_sse(residual, obs_w)

            dx = logT_grid[:, 1:] - logT_grid[:, :-1]
            dy = logT_grid[1:, :] - logT_grid[:-1, :]
            w_reg = 1.0 / float(sigma_dlogT)
            valid_dx = pilot_active_mask[:, 1:] & pilot_active_mask[:, :-1]
            valid_dy = pilot_active_mask[1:, :] & pilot_active_mask[:-1, :]
            phi_reg = float(
                np.sum((w_reg * dx[valid_dx]) ** 2) + np.sum((w_reg * dy[valid_dy]) ** 2)
            )
            objective = float(phi_obs + float(reg_weight) * phi_reg)

            prev_best = float(best["objective"])
            if objective < best["objective"]:
                best["objective"] = objective
                best["wsse"] = phi_obs
                best["params"] = {"r_scale": float(r_scale)}
                best["T_pilots"] = T_pilots.copy()
                best["logT"] = logT_flat_full.copy()
                best["x"] = np.asarray(x, dtype=np.float64).copy()
                if _is_meaningful_improvement(
                    prev_best=prev_best,
                    new_best=objective,
                    rel_improve=float(stagnation_rel_improve),
                    abs_improve=float(stagnation_abs_improve),
                ):
                    stagnation_state["last_meaningful_improve_nfev"] = int(nfev_counter["n"])

            if int(stagnation_patience_nfev or 0) > 0:
                if int(nfev_counter["n"]) >= int(stagnation_min_nfev):
                    stale_nfev = int(nfev_counter["n"]) - int(stagnation_state["last_meaningful_improve_nfev"])
                    if stale_nfev >= int(stagnation_patience_nfev):
                        stagnation_state["stop_reason"] = (
                            "stagnation_stop: "
                            f"no meaningful improvement for {stale_nfev} objective calls "
                            f"(patience={int(stagnation_patience_nfev)}, "
                            f"rel={float(stagnation_rel_improve):.3g}, "
                            f"abs={float(stagnation_abs_improve):.3g})"
                        )
                        raise _DualAnnealingEarlyStop(stagnation_state["stop_reason"])

            active_t_vals = T_pilots[pilot_active_mask]
            if active_t_vals.size > 0:
                t_min_rep = float(np.min(active_t_vals))
                t_max_rep = float(np.max(active_t_vals))
            else:
                t_min_rep = float(np.min(T_pilots))
                t_max_rep = float(np.max(T_pilots))

            print(
                f"eval R_scale={r_scale:.3f} phi_obs={phi_obs:.4e} phi_reg={phi_reg:.4e} obj={objective:.4e} "
                f"T_pilots_active=[{t_min_rep:.1f},{t_max_rep:.1f}] "
                f"graph_reuse={info.get('cuda_graph_reused', False)} "
                f"time={t1 - t0:.2f}s"
            )
            return objective

        optimizer_key = str(optimizer_algorithm).lower().replace("-", "_")
        if optimizer_key == "dual_annealing":
            minimizer_kwargs = {
                "method": "Powell",
                "bounds": bounds,
                "options": {"maxiter": int(local_maxiter)},
            }
            da_kwargs = {
                "func": objective,
                "bounds": bounds,
                "maxiter": int(maxiter),
                "seed": int(seed),
                "no_local_search": bool(no_local_search),
                "minimizer_kwargs": minimizer_kwargs,
                "x0": x0,
            }
            if maxfun is not None:
                da_kwargs["maxfun"] = int(maxfun)
            opt_start = time.perf_counter()
            try:
                result = dual_annealing(**da_kwargs)
            except _DualAnnealingEarlyStop as exc:
                msg = str(exc)
                best_x = best["x"] if best["x"] is not None else np.asarray(x0, dtype=np.float64).copy()
                best_fun = float(best["objective"])
                if not np.isfinite(best_fun):
                    best_fun = float("inf")
                result = OptimizeResult(
                    x=np.asarray(best_x, dtype=np.float64),
                    fun=float(best_fun),
                    nfev=int(nfev_counter["n"]),
                    nit=0,
                    success=True,
                    status=2,
                    message=msg,
                )
                print(f"Dual annealing early stop: {msg}")
            opt_end = time.perf_counter()
        elif optimizer_key == "pso":
            opt_start = time.perf_counter()
            result = _run_pso(
                objective=objective,
                bounds=bounds,
                x0=np.asarray(x0, dtype=np.float64),
                maxiter=int(maxiter),
                swarm_size=int(pso_swarm_size),
                seed=int(seed),
                inertia=float(pso_inertia),
                cognitive=float(pso_cognitive),
                social=float(pso_social),
            )
            opt_end = time.perf_counter()
        else:
            raise ValueError("optimizer_algorithm must be 'dual_annealing' or 'pso'.")

    return {
        "result": result,
        "best_wsse": best["wsse"],
        "best_objective": best["objective"],
        "best_params": best["params"],
        "best_T_pilots": best["T_pilots"],
        "best_logT": best["logT"],
        "pilot_shape": (pilot_ny, pilot_nx),
        "T_min": float(T_min),
        "T_max": float(T_max),
        "rbf_epsilon": float(rbf_epsilon),
        "pilot_active_mask": pilot_active_mask.copy(),
        "pilot_active_count": int(n_pilots_active),
        "pilot_total_count": int(n_pilots_total),
        "deactivate_outside_pilots": bool(deactivate_outside_pilots),
        "pilot_touch_radius": int(pilot_touch_radius),
        "pilot_neighbor_buffer": int(pilot_neighbor_buffer),
        "optimizer": optimizer_key,
        "optimization_wall_seconds": float(opt_end - opt_start),
        "stagnation_stop_reason": stagnation_state["stop_reason"],
    }


def lbfgsb_calibration(
        case: CanterburyCaseInputs,
        device: str = "cuda:0",
        use_ghb: bool = True,
        gh_alpha: float = 1.0,
        aq_thickness: float = 300.0,
        T_min: float = 100.0,
        T_max: float = 20000.0,
        R_scale_min: float = 0.5,
        R_scale_max: float = 0.9,
        r_scale_fixed: float | None = 0.6,
        pilot_nx: int = 5,
        pilot_ny: int = 5,
        rbf_epsilon: float = 10.0,
        perturbation_strength: float = 0.0,
        reg_weight: float = 1e1,
        sigma_dlogT: float = 0.2,
        pilot_logt_init: float | np.ndarray | None = None,
        pilot_init_jitter: float = 0.1,
        deactivate_outside_pilots: bool = True,
        pilot_touch_radius: int = 1,
        pilot_neighbor_buffer: int = 1,
        max_cycles: int = 200,
        nu_pre: int = 2,
        nu_post: int = 2,
        nu_coarse: int = 2,
        omega: float = 0.7,
        rel_tol: float = 5.0e-7,
        abs_tol_min: float = 5.0e-7,
        max_levels: int = 6,
        check_every_no: int = 1,
        maxiter: int = 25,
        fd_step_logt: float = 1.0e-1,
        fd_step_rscale: float = 2.0e-2,
        seed: int = 42,
) -> dict:
    """
    Simple local calibration using L-BFGS-B for pilot-point T and optional recharge scale.

    :param case: Prepared Canterbury case inputs.
    :param device: Warp device string, e.g. "cuda:0".
    :param use_ghb: Toggle general-head boundary terms.
    :param gh_alpha: GHB conductance scaling factor.
    :param aq_thickness: Aquifer thickness used to convert T to K where needed.
    :param T_min: Minimum transmissivity bound for pilot points.
    :param T_max: Maximum transmissivity bound for pilot points.
    :param R_scale_min: Minimum recharge scale (only if r_scale_fixed is None).
    :param R_scale_max: Maximum recharge scale (only if r_scale_fixed is None).
    :param r_scale_fixed: Fixed recharge scale; set to None to optimize.
    :param pilot_nx: Pilot grid columns.
    :param pilot_ny: Pilot grid rows.
    :param rbf_epsilon: RBF shape parameter for pilot interpolation.
    :param perturbation_strength: Random perturbation applied to pilot values (0 disables).
    :param reg_weight: Weight applied to the smoothness penalty.
    :param sigma_dlogT: 1-sigma expected adjacent pilot log10(T) difference.
    :param pilot_logt_init: Optional initial log10(T) pilot values.
    :param pilot_init_jitter: Random jitter added to initial pilots.
    :param deactivate_outside_pilots: If True, freeze pilots not touching active domain.
    :param pilot_touch_radius: Neighborhood radius (in fine cells) for pilot/domain touch.
    :param pilot_neighbor_buffer: Number of pilot-grid neighbor rings to keep active.
    :param max_cycles: Max multigrid cycles per forward solve.
    :param nu_pre: Pre-smoothing iterations.
    :param nu_post: Post-smoothing iterations.
    :param nu_coarse: Coarse-grid iterations.
    :param omega: Jacobi damping parameter.
    :param rel_tol: Relative residual tolerance.
    :param abs_tol_min: Minimum absolute residual tolerance.
    :param max_levels: Max multigrid levels.
    :param check_every_no: Residual check interval.
    :param maxiter: L-BFGS-B max iterations.
    :param fd_step_logt: Finite-difference step for log10(T) pilots.
    :param fd_step_rscale: Finite-difference step for recharge scale.
    :param seed: RNG seed.
    :return: Dict with optimizer outputs and best parameters.
    """
    obs_i, obs_j, obs_gwl, obs_w = _prepare_obs_arrays(case)

    logT_min = float(np.log10(T_min))
    logT_max = float(np.log10(T_max))

    if pilot_nx < 2 or pilot_ny < 2:
        raise ValueError("pilot_nx and pilot_ny must be >= 2")

    rbf_cache = precompute_rbf_cache(
        nx=case.nx,
        ny=case.ny,
        nx_p=pilot_nx,
        ny_p=pilot_ny,
        epsilon=float(rbf_epsilon),
    )

    if deactivate_outside_pilots:
        pilot_active_mask = _compute_pilot_active_mask(
            active=case.active,
            nx_p=pilot_nx,
            ny_p=pilot_ny,
            touch_radius=int(pilot_touch_radius),
            pilot_neighbor_buffer=int(pilot_neighbor_buffer),
        )
    else:
        pilot_active_mask = np.ones((pilot_ny, pilot_nx), dtype=bool)
    pilot_active_flat = pilot_active_mask.reshape(-1)
    n_pilots_total = int(pilot_nx * pilot_ny)
    n_pilots_active = int(np.count_nonzero(pilot_active_flat))
    if n_pilots_active == 0:
        raise ValueError("No active pilot points after applying pilot activity mask.")
    _print_pilot_activity(
        pilot_nx=pilot_nx,
        pilot_ny=pilot_ny,
        pilot_active_count=n_pilots_active,
        pilot_total_count=n_pilots_total,
        deactivate_outside_pilots=deactivate_outside_pilots,
        pilot_touch_radius=pilot_touch_radius,
        pilot_neighbor_buffer=pilot_neighbor_buffer,
    )

    optimize_r_scale = r_scale_fixed is None
    bounds = [(logT_min, logT_max) for _ in range(n_pilots_active)]
    if optimize_r_scale:
        bounds.append((float(R_scale_min), float(R_scale_max)))

    best = {
        "objective": np.inf,
        "wsse": np.inf,
        "params": None,
        "T_pilots": None,
        "logT": None,
        "pilot_active_mask": pilot_active_mask.copy(),
    }

    logT_mid = 0.5 * (logT_min + logT_max)
    if pilot_logt_init is None:
        pilot_logt_init = logT_mid
    T_init_val = float(10.0 ** logT_mid)
    T_init = np.full((case.ny, case.nx), T_init_val, dtype=np.float32)
    T_init = np.where(case.active == 1, T_init, 0.0).astype(np.float32, copy=False)
    if optimize_r_scale:
        r_scale_init = float(0.5 * (R_scale_min + R_scale_max))
    else:
        r_scale_init = float(r_scale_fixed)
    R_init = case.recharge_base * float(r_scale_init)
    R_init = np.where(case.active == 1, R_init, 0.0).astype(np.float32, copy=False)

    init_rng = np.random.default_rng(int(seed))
    if np.ndim(pilot_logt_init) == 0:
        logT0_full = np.full(n_pilots_total, float(pilot_logt_init), dtype=np.float64)
    else:
        logT_init_arr = np.asarray(pilot_logt_init, dtype=np.float64)
        if logT_init_arr.shape == (pilot_ny, pilot_nx):
            logT0_full = logT_init_arr.reshape(-1).copy()
        elif logT_init_arr.shape == (n_pilots_total,):
            logT0_full = logT_init_arr.copy()
        else:
            raise ValueError(
                f"pilot_logt_init shape {logT_init_arr.shape} does not match "
                f"(pilot_ny, pilot_nx)=({pilot_ny}, {pilot_nx}) or ({n_pilots_total},)"
            )

    if float(pilot_init_jitter) > 0.0:
        logT0_full = logT0_full + init_rng.normal(0.0, float(pilot_init_jitter), size=n_pilots_total)

    logT0_full = np.clip(logT0_full, logT_min, logT_max)
    if optimize_r_scale:
        x0 = np.concatenate([logT0_full[pilot_active_flat], [float(r_scale_init)]]).astype(np.float64, copy=False)
    else:
        x0 = logT0_full[pilot_active_flat].astype(np.float64, copy=False)

    with WarpDarcySolver(
            nx=case.nx,
            ny=case.ny,
            dx=case.dx,
            device=device,
            use_ghb=use_ghb,
            solver_type="pcg",
            aq_thickness=float(aq_thickness),
    ) as solver:
        solver.build_from_fields(
            T_field=T_init,
            R_field=R_init,
            active=case.active,
            bc_mask=case.bc_mask,
            bc_values=case.bc_values,
            gh_mask=case.gh_mask,
            gh_head=case.gh_head,
            gh_width=case.gh_width,
            gh_alpha=float(gh_alpha),
        )
        solver.build_hierarchy(max_levels=6, min_coarse_n=4)

        last_eval = {"x": None, "f": None}

        def objective(x: np.ndarray) -> float:
            if optimize_r_scale:
                logT_active = np.asarray(x[:-1], dtype=np.float64)
                r_scale = float(x[-1])
            else:
                logT_active = np.asarray(x, dtype=np.float64)
                r_scale = float(r_scale_fixed)

            if logT_active.size != n_pilots_active:
                raise ValueError(
                    f"logT parameter size {logT_active.size} does not match active pilot count {n_pilots_active}"
                )

            logT_flat_full = np.asarray(logT0_full, dtype=np.float64).copy()
            logT_flat_full[pilot_active_flat] = logT_active
            logT_grid = logT_flat_full.reshape(pilot_ny, pilot_nx)
            T_pilots = np.power(10.0, logT_grid).reshape(pilot_ny, pilot_nx)
            T_field = get_T_field_from_pilots_cached(
                T_pilot_raw=T_pilots,
                T_min=T_min,
                T_max=T_max,
                rbf_cache=rbf_cache,
                perturbation_strength=float(perturbation_strength),
            )
            T_field = np.where(case.active == 1, T_field, 0.0).astype(np.float32, copy=False)

            R_field = case.recharge_base * float(r_scale)
            R_field = np.where(case.active == 1, R_field, 0.0).astype(np.float32, copy=False)

            t0 = time.perf_counter()
            solver.update_T_in_place(T_field)
            solver.update_R_in_place(R_field)
            head, info = solver.solve_multigrid_kcycle(
                max_cycles=int(max_cycles),
                nu_pre=int(nu_pre),
                nu_post=int(nu_post),
                nu_coarse=int(nu_coarse),
                omega=float(omega),
                rel_tol=float(rel_tol),
                abs_tol_min=float(abs_tol_min),
                initial_head=case.model_top,
                return_info=True,
                max_levels=int(max_levels),
                check_every_no=int(check_every_no),
            )
            t1 = time.perf_counter()

            pred = head[obs_i, obs_j]
            residual = pred - obs_gwl
            phi_obs = _weighted_sse(residual, obs_w)

            dx = logT_grid[:, 1:] - logT_grid[:, :-1]
            dy = logT_grid[1:, :] - logT_grid[:-1, :]
            w_reg = 1.0 / float(sigma_dlogT)
            valid_dx = pilot_active_mask[:, 1:] & pilot_active_mask[:, :-1]
            valid_dy = pilot_active_mask[1:, :] & pilot_active_mask[:-1, :]
            phi_reg = float(
                np.sum((w_reg * dx[valid_dx]) ** 2) + np.sum((w_reg * dy[valid_dy]) ** 2)
            )
            objective = float(phi_obs + float(reg_weight) * phi_reg)

            if objective < best["objective"]:
                best["objective"] = objective
                best["wsse"] = phi_obs
                best["params"] = {"r_scale": float(r_scale)}
                best["T_pilots"] = T_pilots.copy()
                best["logT"] = logT_flat_full.copy()

            active_t_vals = T_pilots[pilot_active_mask]
            if active_t_vals.size > 0:
                t_min_rep = float(np.min(active_t_vals))
                t_max_rep = float(np.max(active_t_vals))
            else:
                t_min_rep = float(np.min(T_pilots))
                t_max_rep = float(np.max(T_pilots))

            print(
                f"eval R_scale={r_scale:.3f} phi_obs={phi_obs:.4e} phi_reg={phi_reg:.4e} obj={objective:.4e} "
                f"T_pilots_active=[{t_min_rep:.1f},{t_max_rep:.1f}] "
                f"graph_reuse={info.get('cuda_graph_reused', False)} "
                f"time={t1 - t0:.2f}s"
            )
            last_eval["x"] = np.array(x, copy=True)
            last_eval["f"] = float(objective)
            return objective

        def fd_grad(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=np.float64)
            if last_eval["x"] is not None and np.array_equal(x, last_eval["x"]):
                f0 = float(last_eval["f"])
            else:
                f0 = float(objective(x))

            grad = np.zeros_like(x, dtype=np.float64)
            for i in range(x.size):
                if optimize_r_scale and i == x.size - 1:
                    step = float(fd_step_rscale)
                else:
                    step = float(fd_step_logt)
                lo, hi = bounds[i]
                if x[i] + step > hi:
                    if x[i] - step < lo:
                        step = 0.5 * (hi - lo)
                    else:
                        step = -step
                elif x[i] - step < lo:
                    step = abs(step)

                x_step = x.copy()
                x_step[i] = float(np.clip(x_step[i] + step, lo, hi))
                f1 = float(objective(x_step))
                denom = step if step != 0.0 else 1.0
                grad[i] = (f1 - f0) / denom

            return grad

        opt_start = time.perf_counter()
        result = minimize(
            objective,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            jac=fd_grad,
            options={"maxiter": int(maxiter), "maxls": 50, "ftol": 1.0e-9, "gtol": 1.0e-6},
        )
        opt_end = time.perf_counter()

    return {
        "result": result,
        "best_wsse": best["wsse"],
        "best_objective": best["objective"],
        "best_params": best["params"],
        "best_T_pilots": best["T_pilots"],
        "best_logT": best["logT"],
        "pilot_shape": (pilot_ny, pilot_nx),
        "T_min": float(T_min),
        "T_max": float(T_max),
        "rbf_epsilon": float(rbf_epsilon),
        "pilot_active_mask": pilot_active_mask.copy(),
        "pilot_active_count": int(n_pilots_active),
        "pilot_total_count": int(n_pilots_total),
        "deactivate_outside_pilots": bool(deactivate_outside_pilots),
        "pilot_touch_radius": int(pilot_touch_radius),
        "pilot_neighbor_buffer": int(pilot_neighbor_buffer),
        "optimizer": "lbfgsb",
        "optimization_wall_seconds": float(opt_end - opt_start),
    }


def staged_dual_annealing_calibration(
        case: CanterburyCaseInputs,
        device: str = "cuda:0",
        use_ghb: bool = True,
        gh_alpha: float = 2.0,
        aq_thickness: float = 300.0,
        T_min: float = 100.0,
        T_max: float = 10000.0,
        R_scale_min: float = 0.45,
        R_scale_max: float = 0.8,
        r_scale_fixed: float | None = 0.5,
        stage1_pilot_nx: int = 15,
        stage1_pilot_ny: int = 10,
        stage2_pilot_nx: int = 15,
        stage2_pilot_ny: int = 10,
        rbf_epsilon: float = 30.0,
        perturbation_strength: float = 0.0,
        reg_weight: float = 1,
        sigma_dlogT: float = 0.2,
        stage1_maxiter: int = 200,
        stage2_maxiter: int = 200,
        stage1_maxfun: int | None = 25000,
        stage2_maxfun: int | None = 25000,
        stage1_max_cycles: int = 200,
        stage2_max_cycles: int = 200,
        stage1_rel_tol: float = 5.0e-6,
        stage1_abs_tol_min: float = 5.0e-6,
        stage2_rel_tol: float = 5.0e-6,
        stage2_abs_tol_min: float = 5.0e-6,
        stage1_check_every_no: int = 5,
        stage2_check_every_no: int = 10,
        pilot_init_jitter: float = 0.2,
        deactivate_outside_pilots: bool = True,
        pilot_touch_radius: int = 1,
        pilot_neighbor_buffer: int = 1,
        stage1_no_local_search: bool = True,
        stage2_no_local_search: bool = False,
        stage1_stagnation_patience_nfev: int | None = 5000,
        stage2_stagnation_patience_nfev: int | None = 5000,
        stage1_stagnation_min_nfev: int = 4000,
        stage2_stagnation_min_nfev: int = 5000,
        stagnation_rel_improve: float = 5.0e-3,
        stagnation_abs_improve: float = 0.0,
        stage1_optimizer: str = "pso",
        stage1_pso_swarm_size: int = 24,
        stage1_pso_inertia: float = 0.72,
        stage1_pso_cognitive: float = 1.49,
        stage1_pso_social: float = 1.49,
        seed: int = 42,
) -> dict:
    """
    Two-stage dual-annealing calibration using coarse then fine pilot grids.

    :param case: Prepared Canterbury case inputs.
    :param device: Warp device string, e.g. "cuda:0".
    :param use_ghb: Toggle general-head boundary terms.
    :param gh_alpha: GHB conductance scaling factor.
    :param aq_thickness: Aquifer thickness used to convert T to K where needed.
    :param T_min: Minimum transmissivity bound for pilot points.
    :param T_max: Maximum transmissivity bound for pilot points.
    :param R_scale_min: Minimum recharge scale (only if r_scale_fixed is None).
    :param R_scale_max: Maximum recharge scale (only if r_scale_fixed is None).
    :param r_scale_fixed: Fixed recharge scale; set to None to optimize.
    :param stage1_pilot_nx: Stage 1 pilot grid columns.
    :param stage1_pilot_ny: Stage 1 pilot grid rows.
    :param stage2_pilot_nx: Stage 2 pilot grid columns.
    :param stage2_pilot_ny: Stage 2 pilot grid rows.
    :param rbf_epsilon: RBF shape parameter for pilot interpolation.
    :param perturbation_strength: Random perturbation applied to pilot values (0 disables).
    :param reg_weight: Weight applied to the smoothness penalty.
    :param sigma_dlogT: 1-sigma expected adjacent pilot log10(T) difference.
    :param stage1_maxiter: Dual annealing iterations for stage 1.
    :param stage2_maxiter: Dual annealing iterations for stage 2.
    :param stage1_maxfun: Objective-call cap for stage 1 dual annealing.
    :param stage2_maxfun: Objective-call cap for stage 2 dual annealing.
    :param stage1_max_cycles: Max multigrid cycles per solve (stage 1).
    :param stage2_max_cycles: Max multigrid cycles per solve (stage 2).
    :param stage1_rel_tol: Relative residual tolerance (stage 1).
    :param stage1_abs_tol_min: Minimum absolute residual tolerance (stage 1).
    :param stage2_rel_tol: Relative residual tolerance (stage 2).
    :param stage2_abs_tol_min: Minimum absolute residual tolerance (stage 2).
    :param stage1_check_every_no: Residual check interval (stage 1).
    :param stage2_check_every_no: Residual check interval (stage 2).
    :param pilot_init_jitter: Random jitter added to initial pilots.
    :param deactivate_outside_pilots: If True, freeze pilots not touching active domain.
    :param pilot_touch_radius: Neighborhood radius (in fine cells) for pilot/domain touch.
    :param pilot_neighbor_buffer: Number of pilot-grid neighbor rings to keep active.
    :param stage1_no_local_search: Disable local search in stage 1 when True.
    :param stage2_no_local_search: Disable local search in stage 2 when True.
    :param stage1_stagnation_patience_nfev: Stage-1 stagnation patience in objective calls.
    :param stage2_stagnation_patience_nfev: Stage-2 stagnation patience in objective calls.
    :param stage1_stagnation_min_nfev: Stage-1 minimum objective calls before stagnation check.
    :param stage2_stagnation_min_nfev: Stage-2 minimum objective calls before stagnation check.
    :param stagnation_rel_improve: Relative improvement threshold for stagnation check.
    :param stagnation_abs_improve: Absolute improvement threshold for stagnation check.
    :param stage1_optimizer: "dual_annealing" or "pso" for stage 1.
    :param stage1_pso_swarm_size: Particle count if stage1_optimizer="pso".
    :param stage1_pso_inertia: Inertia coefficient for stage-1 PSO.
    :param stage1_pso_cognitive: Cognitive coefficient for stage-1 PSO.
    :param stage1_pso_social: Social coefficient for stage-1 PSO.
    :param seed: RNG seed.
    :return: Dict with stage1/stage2 outputs.
    """
    print(f"Stage 1: coarse pilot grid optimization ({stage1_optimizer})")
    stage1 = dual_annealing_calibration(
        case=case,
        device=device,
        use_ghb=use_ghb,
        gh_alpha=gh_alpha,
        aq_thickness=aq_thickness,
        T_min=T_min,
        T_max=T_max,
        R_scale_min=R_scale_min,
        R_scale_max=R_scale_max,
        r_scale_fixed=r_scale_fixed,
        pilot_nx=stage1_pilot_nx,
        pilot_ny=stage1_pilot_ny,
        rbf_epsilon=rbf_epsilon,
        perturbation_strength=perturbation_strength,
        reg_weight=reg_weight,
        sigma_dlogT=sigma_dlogT,
        deactivate_outside_pilots=deactivate_outside_pilots,
        pilot_touch_radius=pilot_touch_radius,
        pilot_neighbor_buffer=pilot_neighbor_buffer,
        max_cycles=stage1_max_cycles,
        rel_tol=stage1_rel_tol,
        abs_tol_min=stage1_abs_tol_min,
        check_every_no=stage1_check_every_no,
        maxiter=stage1_maxiter,
        maxfun=stage1_maxfun,
        seed=seed,
        no_local_search=stage1_no_local_search,
        stagnation_patience_nfev=stage1_stagnation_patience_nfev,
        stagnation_min_nfev=stage1_stagnation_min_nfev,
        stagnation_rel_improve=stagnation_rel_improve,
        stagnation_abs_improve=stagnation_abs_improve,
        optimizer_algorithm=stage1_optimizer,
        pso_swarm_size=stage1_pso_swarm_size,
        pso_inertia=stage1_pso_inertia,
        pso_cognitive=stage1_pso_cognitive,
        pso_social=stage1_pso_social,
    )

    logT_seed = None
    if stage1.get("best_logT") is not None:
        logT1 = np.asarray(stage1["best_logT"], dtype=float).reshape(stage1_pilot_ny, stage1_pilot_nx)
        logT_seed = _resample_pilot_grid(logT1, stage2_pilot_ny, stage2_pilot_nx)

    print("Stage 2: fine pilot grid optimization")
    stage2 = dual_annealing_calibration(
        case=case,
        device=device,
        use_ghb=use_ghb,
        gh_alpha=gh_alpha,
        aq_thickness=aq_thickness,
        T_min=T_min,
        T_max=T_max,
        R_scale_min=R_scale_min,
        R_scale_max=R_scale_max,
        r_scale_fixed=r_scale_fixed,
        pilot_nx=stage2_pilot_nx,
        pilot_ny=stage2_pilot_ny,
        rbf_epsilon=rbf_epsilon,
        perturbation_strength=perturbation_strength,
        reg_weight=reg_weight,
        sigma_dlogT=sigma_dlogT,
        pilot_logt_init=logT_seed,
        pilot_init_jitter=pilot_init_jitter,
        deactivate_outside_pilots=deactivate_outside_pilots,
        pilot_touch_radius=pilot_touch_radius,
        pilot_neighbor_buffer=pilot_neighbor_buffer,
        max_cycles=stage2_max_cycles,
        rel_tol=stage2_rel_tol,
        abs_tol_min=stage2_abs_tol_min,
        check_every_no=stage2_check_every_no,
        maxiter=stage2_maxiter,
        maxfun=stage2_maxfun,
        seed=seed,
        no_local_search=stage2_no_local_search,
        stagnation_patience_nfev=stage2_stagnation_patience_nfev,
        stagnation_min_nfev=stage2_stagnation_min_nfev,
        stagnation_rel_improve=stagnation_rel_improve,
        stagnation_abs_improve=stagnation_abs_improve,
        optimizer_algorithm="dual_annealing",
    )

    total_nfev = None
    if stage1.get("result") is not None and stage2.get("result") is not None:
        s1_nfev = stage1["result"].get("nfev")
        s2_nfev = stage2["result"].get("nfev")
        if s1_nfev is not None and s2_nfev is not None:
            total_nfev = int(s1_nfev) + int(s2_nfev)

    total_wall = None
    if stage1.get("optimization_wall_seconds") is not None and stage2.get("optimization_wall_seconds") is not None:
        total_wall = float(stage1["optimization_wall_seconds"]) + float(stage2["optimization_wall_seconds"])

    return {
        "stage1": stage1,
        "stage2": stage2,
        "optimization_wall_seconds_total": total_wall,
        "model_calls_total": total_nfev,
    }


def _extract_best_solution(outputs: dict) -> dict:
    if "stage2" in outputs:
        return outputs["stage2"]
    if "best_logT" in outputs or "best_T_pilots" in outputs:
        return outputs
    raise ValueError("Could not locate best parameters in outputs.")


def _serialize_best_params(params: dict | None) -> dict:
    if not params:
        return {}
    serialized = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            serialized[key] = value.tolist()
        elif isinstance(value, (np.floating, np.integer)):
            serialized[key] = value.item()
        else:
            serialized[key] = value
    return serialized


def _summarize_opt_result(result: dict | None) -> dict:
    if result is None:
        return {}

    summary: dict[str, object] = {}
    if "success" in result:
        summary["success"] = bool(result["success"])
    if "status" in result:
        summary["status"] = int(result["status"])
    if "message" in result:
        msg = result["message"]
        if isinstance(msg, (list, tuple)):
            summary["message"] = [str(m) for m in msg]
        else:
            summary["message"] = str(msg)
    if "fun" in result:
        summary["fun"] = float(result["fun"])
    if "nit" in result:
        summary["nit"] = int(result["nit"])
    if "nfev" in result:
        summary["nfev"] = int(result["nfev"])
    if "njev" in result:
        summary["njev"] = int(result["njev"])
    if "nhev" in result:
        summary["nhev"] = int(result["nhev"])
    return summary


def _summarize_best(best: dict) -> dict:
    summary = {
        "best_params": _serialize_best_params(best.get("best_params")),
    }
    result = best.get("result")
    if result is not None:
        summary.update(_summarize_opt_result(result))
    if best.get("best_wsse") is not None:
        summary["best_wsse"] = float(best["best_wsse"])
    if best.get("best_objective") is not None:
        summary["best_objective"] = float(best["best_objective"])
    if best.get("pilot_shape") is not None:
        pilot_shape = best["pilot_shape"]
        summary["pilot_shape"] = [int(pilot_shape[0]), int(pilot_shape[1])]
    if best.get("pilot_active_mask") is not None:
        pilot_active_mask = np.asarray(best["pilot_active_mask"], dtype=bool)
        summary["pilot_active_mask"] = pilot_active_mask.astype(int).tolist()
        summary["pilot_active_count"] = int(np.count_nonzero(pilot_active_mask))
        summary["pilot_total_count"] = int(pilot_active_mask.size)
    if "deactivate_outside_pilots" in best:
        summary["deactivate_outside_pilots"] = bool(best["deactivate_outside_pilots"])
    if "pilot_touch_radius" in best:
        summary["pilot_touch_radius"] = int(best["pilot_touch_radius"])
    if "pilot_neighbor_buffer" in best:
        summary["pilot_neighbor_buffer"] = int(best["pilot_neighbor_buffer"])
    if best.get("best_T_pilots") is not None:
        summary["best_T_pilots"] = np.asarray(best["best_T_pilots"], dtype=float).tolist()
    if best.get("best_logT") is not None:
        summary["best_logT"] = np.asarray(best["best_logT"], dtype=float).tolist()
    if "optimizer" in best:
        summary["optimizer"] = best["optimizer"]
    if "rbf_epsilon" in best:
        summary["rbf_epsilon"] = float(best["rbf_epsilon"])
    if "T_min" in best:
        summary["T_min"] = float(best["T_min"])
    if "T_max" in best:
        summary["T_max"] = float(best["T_max"])
    if "optimization_wall_seconds" in best and best.get("optimization_wall_seconds") is not None:
        summary["optimization_wall_seconds"] = float(best["optimization_wall_seconds"])
    if "stagnation_stop_reason" in best and best.get("stagnation_stop_reason") is not None:
        summary["stagnation_stop_reason"] = str(best["stagnation_stop_reason"])
    return summary


def _build_results_summary(outputs: dict) -> dict:
    if "stage2" in outputs:
        stage1 = _summarize_best(outputs["stage1"])
        stage2 = _summarize_best(outputs["stage2"])
        totals: dict[str, object] = {}

        nfev_total = 0
        has_nfev = False
        for s in (stage1, stage2):
            if "nfev" in s:
                nfev_total += int(s["nfev"])
                has_nfev = True
        if has_nfev:
            totals["model_calls_total"] = int(nfev_total)

        wall_total = 0.0
        has_wall = False
        for s in (stage1, stage2):
            if "optimization_wall_seconds" in s:
                wall_total += float(s["optimization_wall_seconds"])
                has_wall = True
        if has_wall:
            totals["optimization_wall_seconds_total"] = float(wall_total)

        if outputs.get("model_calls_total") is not None:
            totals["model_calls_total"] = int(outputs["model_calls_total"])
        if outputs.get("optimization_wall_seconds_total") is not None:
            totals["optimization_wall_seconds_total"] = float(outputs["optimization_wall_seconds_total"])

        payload = {
            "stage1": stage1,
            "stage2": stage2,
        }
        if totals:
            payload["calibration_totals"] = totals
        return payload

    single = _summarize_best(outputs)
    totals = {}
    if "nfev" in single:
        totals["model_calls_total"] = int(single["nfev"])
    if "optimization_wall_seconds" in single:
        totals["optimization_wall_seconds_total"] = float(single["optimization_wall_seconds"])
    if totals:
        single["calibration_totals"] = totals
    return single


def run_best_solution_and_save_outputs(
        case: CanterburyCaseInputs,
        outputs: dict,
        out_dir: Path,
        device: str = "cuda:0",
        use_ghb: bool = True,
        gh_alpha: float = 1.0,
        aq_thickness: float = 300.0,
        rbf_epsilon: float = 10.0,
        T_min: float | None = None,
        T_max: float | None = None,
        max_cycles: int = 200,
        nu_pre: int = 2,
        nu_post: int = 2,
        nu_coarse: int = 2,
        omega: float = 0.7,
        rel_tol: float = 5.0e-7,
        abs_tol_min: float = 5.0e-7,
        max_levels: int = 6,
        check_every_no: int = 1,
) -> dict:
    """
    Run a final solve using the best parameters and save head + obs vs sim plot.
    """
    best = _extract_best_solution(outputs)
    pilot_shape = best.get("pilot_shape")
    if pilot_shape is None:
        raise ValueError("Missing pilot_shape in outputs; cannot rebuild T field.")
    pilot_ny, pilot_nx = int(pilot_shape[0]), int(pilot_shape[1])

    best_T_pilots = best.get("best_T_pilots")
    if best_T_pilots is None:
        best_logT = best.get("best_logT")
        if best_logT is None:
            raise ValueError("Missing best_logT and best_T_pilots in outputs.")
        logT_grid = np.asarray(best_logT, dtype=float).reshape(pilot_ny, pilot_nx)
        best_T_pilots = np.power(10.0, logT_grid)
    else:
        best_T_pilots = np.asarray(best_T_pilots, dtype=float).reshape(pilot_ny, pilot_nx)

    r_scale = 0.6
    best_params = best.get("best_params") or {}
    if "r_scale" in best_params:
        r_scale = float(best_params["r_scale"])

    if "rbf_epsilon" in best:
        rbf_epsilon = float(best["rbf_epsilon"])

    if T_min is None and "T_min" in best:
        T_min = float(best["T_min"])
    if T_max is None and "T_max" in best:
        T_max = float(best["T_max"])

    rbf_cache = precompute_rbf_cache(
        nx=case.nx,
        ny=case.ny,
        nx_p=pilot_nx,
        ny_p=pilot_ny,
        epsilon=float(rbf_epsilon),
    )

    if T_min is None:
        T_min = float(np.min(best_T_pilots))
    if T_max is None:
        T_max = float(np.max(best_T_pilots))

    T_field = get_T_field_from_pilots_cached(
        T_pilot_raw=best_T_pilots,
        T_min=float(T_min),
        T_max=float(T_max),
        rbf_cache=rbf_cache,
        perturbation_strength=0.0,
    )
    T_field = np.where(case.active == 1, T_field, 0.0).astype(np.float32, copy=False)

    R_field = case.recharge_base * float(r_scale)
    R_field = np.where(case.active == 1, R_field, 0.0).astype(np.float32, copy=False)

    with WarpDarcySolver(
            nx=case.nx,
            ny=case.ny,
            dx=case.dx,
            device=device,
            use_ghb=use_ghb,
            solver_type="pcg",
            aq_thickness=float(aq_thickness),
    ) as solver:
        solver.build_from_fields(
            T_field=T_field,
            R_field=R_field,
            active=case.active,
            bc_mask=case.bc_mask,
            bc_values=case.bc_values,
            gh_mask=case.gh_mask,
            gh_head=case.gh_head,
            gh_width=case.gh_width,
            gh_alpha=float(gh_alpha),
        )
        solver.build_hierarchy(max_levels=int(max_levels), min_coarse_n=4)
        head, info = solver.solve_multigrid_kcycle(
            max_cycles=int(max_cycles),
            nu_pre=int(nu_pre),
            nu_post=int(nu_post),
            nu_coarse=int(nu_coarse),
            omega=float(omega),
            rel_tol=float(rel_tol),
            abs_tol_min=float(abs_tol_min),
            initial_head=case.model_top,
            return_info=True,
            max_levels=int(max_levels),
            check_every_no=int(check_every_no),
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)
    head_path = out_dir / "best_head.npz"
    np.savez(head_path, head=head)

    obs_i, obs_j, obs_gwl, _obs_w = _prepare_obs_arrays(case)
    sim_gwl = head[obs_i, obs_j]

    import matplotlib.pyplot as plt

    plt.figure()
    plt.scatter(obs_gwl, sim_gwl, s=8, alpha=0.6, edgecolors="none")
    min_v = float(min(np.min(obs_gwl), np.min(sim_gwl)))
    max_v = float(max(np.max(obs_gwl), np.max(sim_gwl)))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--", color="black", linewidth=1.0)
    plt.xlabel("Observed head (m)")
    plt.ylabel("Simulated head (m)")
    plt.title("Canterbury case study: observed vs simulated heads")
    plt.tight_layout()
    scatter_path = out_dir / "obs_vs_sim_scatter.png"
    plt.savefig(scatter_path, dpi=200)
    plt.close()

    return {
        "head_path": str(head_path),
        "scatter_path": str(scatter_path),
        "r_scale": float(r_scale),
        "solver_info": info,
    }


def run_case_study(
    grid_size: int = 100,
    inputs_dir: Path | None = None,
    device: str = "cuda:0",
    staged: bool = True,
    optimizer: str = "dual_annealing",
    r_scale_fixed: float | None = 0.5,
    deactivate_outside_pilots: bool = True,
    pilot_touch_radius: int = 1,
    pilot_neighbor_buffer: int = 1,
    stage1_optimizer: str = "dual_annealing",
    stage1_pso_swarm_size: int = 16,
    stage1_pso_inertia: float = 0.72,
    stage1_pso_cognitive: float = 1.49,
    stage1_pso_social: float = 1.49,
    # Staged dual-annealing stop controls
    stage1_maxfun: int | None = 25000,
    stage2_maxfun: int | None = 60000,
    stage1_no_local_search: bool = True,
    stage2_no_local_search: bool = False,
    stage1_stagnation_patience_nfev: int | None = 5000,
    stage2_stagnation_patience_nfev: int | None = 8000,
    stage1_stagnation_min_nfev: int = 5000,
    stage2_stagnation_min_nfev: int = 8000,
    # Global stagnation thresholds used by staged and non-staged dual annealing
    stagnation_rel_improve: float = 5.0e-3,
    stagnation_abs_improve: float = 0.0,
    # Non-staged dual-annealing stop controls
    maxfun: int | None = None,
    no_local_search: bool = False,
    stagnation_patience_nfev: int | None = None,
    stagnation_min_nfev: int = 0,
):
    case = load_case_inputs(grid_size=grid_size, inputs_dir=inputs_dir)
    out_path = Path(__file__).parent.joinpath("results")
    case.obs_df.to_csv(out_path.joinpath("obs.csv"), index=False)
    extent = (1490750.0, 1581450.0, 5138850.0, 5201550.0)
    export_active_cells_geotiff(
        active=case.active,
        out_path=out_path.joinpath(f"canterbury_active_{grid_size}m.tif"),
        extent=extent,
        crs="EPSG:2193",
        rows_increase_south=True,
    )
    export_active_cells_geotiff(
        active=case.gh_mask,
        out_path=out_path.joinpath(f"canterbury_ghb_{grid_size}m.tif"),
        extent=extent,
        crs="EPSG:2193",
        rows_increase_south=True,
    )
    optimizer_key = optimizer.lower().replace("-", "_")
    if optimizer_key in {"lbfgsb", "l_bfgs_b"}:
        return lbfgsb_calibration(
            case=case,
            device=device,
            r_scale_fixed=r_scale_fixed,
            deactivate_outside_pilots=deactivate_outside_pilots,
            pilot_touch_radius=pilot_touch_radius,
            pilot_neighbor_buffer=pilot_neighbor_buffer,
        )
    if optimizer_key == "dual_annealing":
        if staged:
            return staged_dual_annealing_calibration(
                case=case,
                device=device,
                r_scale_fixed=r_scale_fixed,
                deactivate_outside_pilots=deactivate_outside_pilots,
                pilot_touch_radius=pilot_touch_radius,
                pilot_neighbor_buffer=pilot_neighbor_buffer,
                stage1_optimizer=stage1_optimizer,
                stage1_pso_swarm_size=stage1_pso_swarm_size,
                stage1_pso_inertia=stage1_pso_inertia,
                stage1_pso_cognitive=stage1_pso_cognitive,
                stage1_pso_social=stage1_pso_social,
                stage1_maxfun=stage1_maxfun,
                stage2_maxfun=stage2_maxfun,
                stage1_no_local_search=stage1_no_local_search,
                stage2_no_local_search=stage2_no_local_search,
                stage1_stagnation_patience_nfev=stage1_stagnation_patience_nfev,
                stage2_stagnation_patience_nfev=stage2_stagnation_patience_nfev,
                stage1_stagnation_min_nfev=stage1_stagnation_min_nfev,
                stage2_stagnation_min_nfev=stage2_stagnation_min_nfev,
                stagnation_rel_improve=stagnation_rel_improve,
                stagnation_abs_improve=stagnation_abs_improve,
            )
        return dual_annealing_calibration(
            case=case,
            device=device,
            r_scale_fixed=r_scale_fixed,
            deactivate_outside_pilots=deactivate_outside_pilots,
            pilot_touch_radius=pilot_touch_radius,
            pilot_neighbor_buffer=pilot_neighbor_buffer,
            maxfun=maxfun,
            no_local_search=no_local_search,
            stagnation_patience_nfev=stagnation_patience_nfev,
            stagnation_min_nfev=stagnation_min_nfev,
            stagnation_rel_improve=stagnation_rel_improve,
            stagnation_abs_improve=stagnation_abs_improve,
            optimizer_algorithm="dual_annealing",
        )
    raise ValueError("optimizer must be 'dual_annealing' or 'lbfgsb'")


def _int_or_none(value: str) -> int | None:
    txt = str(value).strip().lower()
    if txt in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected integer or 'none', got: {value}") from exc


def _float_or_none(value: str) -> float | None:
    txt = str(value).strip().lower()
    if txt in {"none", "null"}:
        return None
    try:
        return float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Expected float or 'none', got: {value}") from exc


def _build_cli_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Canterbury case study calibration with configurable stopping criteria.",
    )
    parser.add_argument("--grid-size", type=int, default=100, help="Input grid size key (default: 100).")
    parser.add_argument("--inputs-dir", type=Path, default=None, help="Optional input directory override.")
    parser.add_argument("--device", default="cuda:0", help="Warp device string, e.g. cuda:0 or cpu.")
    parser.add_argument(
        "--optimizer",
        choices=["dual_annealing", "lbfgsb"],
        default="dual_annealing",
        help="Top-level optimizer selection.",
    )
    parser.add_argument(
        "--staged",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use staged dual-annealing run (default: enabled).",
    )
    parser.add_argument(
        "--r-scale-fixed",
        type=_float_or_none,
        default=0.5,
        help="Fixed recharge scale, or 'none' to optimize it.",
    )
    parser.add_argument(
        "--deactivate-outside-pilots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze pilots not touching active domain (default: enabled).",
    )
    parser.add_argument("--pilot-touch-radius", type=int, default=1, help="Pilot/domain touch radius.")
    parser.add_argument("--pilot-neighbor-buffer", type=int, default=1, help="Pilot-grid buffer rings.")

    # Stage-1 optimizer controls.
    parser.add_argument(
        "--stage1-optimizer",
        choices=["dual_annealing", "pso"],
        default="dual_annealing",
        help="Stage 1 optimizer in staged mode.",
    )
    parser.add_argument("--stage1-pso-swarm-size", type=int, default=16, help="PSO swarm size for stage 1.")
    parser.add_argument("--stage1-pso-inertia", type=float, default=0.72, help="PSO inertia for stage 1.")
    parser.add_argument("--stage1-pso-cognitive", type=float, default=1.49, help="PSO cognitive term for stage 1.")
    parser.add_argument("--stage1-pso-social", type=float, default=1.49, help="PSO social term for stage 1.")

    # Staged dual-annealing stop controls.
    parser.add_argument(
        "--stage1-maxfun",
        type=_int_or_none,
        default=25000,
        help="Objective-call cap for stage 1 dual annealing, or 'none'.",
    )
    parser.add_argument(
        "--stage2-maxfun",
        type=_int_or_none,
        default=60000,
        help="Objective-call cap for stage 2 dual annealing, or 'none'.",
    )
    parser.add_argument(
        "--stage1-stagnation-patience-nfev",
        type=_int_or_none,
        default=5000,
        help="Stage-1 stagnation patience in objective calls, or 'none'.",
    )
    parser.add_argument(
        "--stage2-stagnation-patience-nfev",
        type=_int_or_none,
        default=8000,
        help="Stage-2 stagnation patience in objective calls, or 'none'.",
    )
    parser.add_argument(
        "--stage1-stagnation-min-nfev",
        type=int,
        default=5000,
        help="Minimum stage-1 calls before stagnation checks.",
    )
    parser.add_argument(
        "--stage2-stagnation-min-nfev",
        type=int,
        default=8000,
        help="Minimum stage-2 calls before stagnation checks.",
    )
    parser.add_argument(
        "--stage1-no-local-search",
        dest="stage1_no_local_search",
        action="store_true",
        default=True,
        help="Disable stage-1 local search (default).",
    )
    parser.add_argument(
        "--stage1-local-search",
        dest="stage1_no_local_search",
        action="store_false",
        help="Enable stage-1 local search.",
    )
    parser.add_argument(
        "--stage2-no-local-search",
        dest="stage2_no_local_search",
        action="store_true",
        default=False,
        help="Disable stage-2 local search.",
    )
    parser.add_argument(
        "--stage2-local-search",
        dest="stage2_no_local_search",
        action="store_false",
        help="Enable stage-2 local search (default).",
    )

    # Shared stagnation thresholds.
    parser.add_argument(
        "--stagnation-rel-improve",
        type=float,
        default=5.0e-3,
        help="Relative objective improvement threshold for stagnation logic.",
    )
    parser.add_argument(
        "--stagnation-abs-improve",
        type=float,
        default=0.0,
        help="Absolute objective improvement threshold for stagnation logic.",
    )

    # Non-staged dual-annealing stop controls.
    parser.add_argument(
        "--maxfun",
        type=_int_or_none,
        default=None,
        help="Objective-call cap for non-staged dual annealing, or 'none'.",
    )
    parser.add_argument(
        "--stagnation-patience-nfev",
        type=_int_or_none,
        default=None,
        help="Stagnation patience for non-staged dual annealing, or 'none'.",
    )
    parser.add_argument(
        "--stagnation-min-nfev",
        type=int,
        default=0,
        help="Minimum calls before non-staged stagnation checks.",
    )
    parser.add_argument(
        "--no-local-search",
        dest="no_local_search",
        action="store_true",
        default=False,
        help="Disable local search in non-staged dual annealing.",
    )
    parser.add_argument(
        "--local-search",
        dest="no_local_search",
        action="store_false",
        help="Enable local search in non-staged dual annealing (default).",
    )
    return parser


if __name__ == "__main__":
    cli = _build_cli_parser()
    args = cli.parse_args()

    out_dir = Path(__file__).parent.joinpath("results")
    grid_size = int(args.grid_size)
    device = str(args.device)
    inputs_dir = args.inputs_dir
    run_start = time.perf_counter()
    outputs = run_case_study(
        grid_size=grid_size,
        inputs_dir=inputs_dir,
        device=device,
        staged=bool(args.staged),
        optimizer=str(args.optimizer),
        r_scale_fixed=args.r_scale_fixed,
        deactivate_outside_pilots=bool(args.deactivate_outside_pilots),
        pilot_touch_radius=int(args.pilot_touch_radius),
        pilot_neighbor_buffer=int(args.pilot_neighbor_buffer),
        stage1_optimizer=str(args.stage1_optimizer),
        stage1_pso_swarm_size=int(args.stage1_pso_swarm_size),
        stage1_pso_inertia=float(args.stage1_pso_inertia),
        stage1_pso_cognitive=float(args.stage1_pso_cognitive),
        stage1_pso_social=float(args.stage1_pso_social),
        stage1_maxfun=args.stage1_maxfun,
        stage2_maxfun=args.stage2_maxfun,
        stage1_no_local_search=bool(args.stage1_no_local_search),
        stage2_no_local_search=bool(args.stage2_no_local_search),
        stage1_stagnation_patience_nfev=args.stage1_stagnation_patience_nfev,
        stage2_stagnation_patience_nfev=args.stage2_stagnation_patience_nfev,
        stage1_stagnation_min_nfev=int(args.stage1_stagnation_min_nfev),
        stage2_stagnation_min_nfev=int(args.stage2_stagnation_min_nfev),
        stagnation_rel_improve=float(args.stagnation_rel_improve),
        stagnation_abs_improve=float(args.stagnation_abs_improve),
        maxfun=args.maxfun,
        no_local_search=bool(args.no_local_search),
        stagnation_patience_nfev=args.stagnation_patience_nfev,
        stagnation_min_nfev=int(args.stagnation_min_nfev),
    )
    calibration_end = time.perf_counter()
    # domain extent (np.float64(1490750.0), np.float64(1581450.0), np.float64(5138850.0), np.float64(5201550.0)) nztm increasing south

    best = _extract_best_solution(outputs)
    result = best["result"]
    if "stage2" in outputs:
        print("\nDual annealing finished (stage 2)")
    else:
        optimizer_label = best.get("optimizer", outputs.get("optimizer", "dual_annealing"))
        if optimizer_label == "lbfgsb":
            print("\nL-BFGS-B finished")
        else:
            print("\nDual annealing finished")
    print("best x:", result.x)
    print("best fun:", result.fun)
    print("tracked best params:", best.get("best_params"))

    # Save the full outputs (recoverable)
    out_dir.mkdir(exist_ok=True)
    torch.save(outputs, out_dir.joinpath("results.pt"))
    result_dict = _build_results_summary(outputs)

    final_best_run_wall_seconds = None
    if "stage2" in outputs:
        # Final run with best parameters + save head and obs vs sim plot
        case_inputs = load_case_inputs(grid_size=grid_size, inputs_dir=inputs_dir)
        final_best_start = time.perf_counter()
        best_outputs = run_best_solution_and_save_outputs(
            case=case_inputs,
            outputs=outputs,
            out_dir=out_dir,
            device=device,
        )
        final_best_end = time.perf_counter()
        final_best_run_wall_seconds = float(final_best_end - final_best_start)
        print("Saved best head to:", best_outputs["head_path"])
        print("Saved obs vs sim scatter to:", best_outputs["scatter_path"])

    run_end = time.perf_counter()
    run_metadata = {
        "calibration_wall_seconds": float(calibration_end - run_start),
        "end_to_end_wall_seconds": float(run_end - run_start),
    }
    if final_best_run_wall_seconds is not None:
        run_metadata["final_best_run_wall_seconds"] = float(final_best_run_wall_seconds)
    result_dict["run_metadata"] = run_metadata

    # Save as JSON for readability
    with open(out_dir.joinpath("results_summary.json"), "w") as f:
        json.dump(result_dict, f, indent=2)
