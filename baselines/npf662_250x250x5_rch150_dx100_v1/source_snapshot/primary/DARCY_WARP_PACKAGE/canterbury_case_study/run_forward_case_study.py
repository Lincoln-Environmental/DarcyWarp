from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import load_case_inputs
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


def _prepare_obs_arrays(case) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = case.obs_df
    required = {"i", "j", "gwl"}
    if not required.issubset(df.columns):
        raise ValueError(f"obs_df must include columns {sorted(required)}.")

    std_col = "std_gwl" if "std_gwl" in df.columns else "std" if "std" in df.columns else None
    if std_col is None:
        raise ValueError("obs_df must include either 'std_gwl' or 'std'.")

    i_idx = df["i"].astype(int).to_numpy()
    j_idx = df["j"].astype(int).to_numpy()
    gwl = df["gwl"].to_numpy(dtype=float)
    std = df[std_col].to_numpy(dtype=float)

    in_bounds = (
        (i_idx >= 0)
        & (i_idx < case.ny)
        & (j_idx >= 0)
        & (j_idx < case.nx)
    )
    i_idx = i_idx[in_bounds]
    j_idx = j_idx[in_bounds]
    gwl = gwl[in_bounds]
    std = std[in_bounds]

    active_obs = case.active[i_idx, j_idx] == 1
    i_idx = i_idx[active_obs]
    j_idx = j_idx[active_obs]
    gwl = gwl[active_obs]
    std = std[active_obs]

    sigma = np.clip(std, 0.2, None)
    return i_idx, j_idx, gwl, sigma


def _compute_metrics(pred: np.ndarray, obs: np.ndarray, sigma: np.ndarray) -> dict:
    residual = np.asarray(pred, dtype=float) - np.asarray(obs, dtype=float)
    weighted = residual / np.asarray(sigma, dtype=float)
    return {
        "n_obs": int(residual.size),
        "rmse": float(np.sqrt(np.mean(residual * residual))),
        "weighted_rmse": float(np.sqrt(np.mean(weighted * weighted))),
        "weighted_sse": float(np.sum(weighted * weighted)),
        "mean_residual": float(np.mean(residual)),
    }


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        arr = np.asarray(obj, dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        summary = {
            "mode": "grid",
            "shape": [int(v) for v in arr.shape],
        }
        if finite.size > 0:
            summary.update(
                {
                    "min": float(np.min(finite)),
                    "max": float(np.max(finite)),
                    "mean": float(np.mean(finite)),
                    "median": float(np.median(finite)),
                }
            )
        return summary
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _summarize_scalar_or_grid(value):
    arr = np.asarray(value)
    if arr.ndim == 0:
        return float(arr)
    return _json_default(arr)


def run_forward_case_study(
    grid_size: int = 100,
    inputs_dir: Path | None = None,
    device: str = "cuda:0",
    use_ghb: bool = True,
    gh_alpha: float | np.ndarray = 2.0,
    aq_thickness: float | np.ndarray = 300.0,
    t_min: float = 100.0,
    t_max: float = 10000.0,
    r_scale: float = 0.5,
    max_cycles: int = 200,
    nu_pre: int = 2,
    nu_post: int = 2,
    nu_coarse: int = 2,
    omega: float = 0.7,
    rel_tol: float = 5.0e-6,
    abs_tol_min: float = 5.0e-6,
    max_levels: int = 6,
    check_every_no: int = 10,
) -> dict:
    requested_device = str(device).strip().lower()
    if not requested_device.startswith("cuda"):
        raise ValueError(
            "This forward check is configured for GPU only. Use a CUDA device such as 'cuda:0'."
        )

    case = load_case_inputs(grid_size=grid_size, inputs_dir=inputs_dir)

    # Base transmissivity matches calibration initialization: log-midpoint of bounds.
    t_base = float(10.0 ** (0.5 * (np.log10(float(t_min)) + np.log10(float(t_max)))))
    t_field = np.full((case.ny, case.nx), t_base, dtype=np.float32)
    t_field = np.where(case.active == 1, t_field, 0.0).astype(np.float32, copy=False)

    r_field = case.recharge_base * float(r_scale)
    r_field = np.where(case.active == 1, r_field, 0.0).astype(np.float32, copy=False)

    start = time.perf_counter()
    with WarpDarcySolver(
        nx=case.nx,
        ny=case.ny,
        dx=case.dx,
        device=str(device),
        use_ghb=use_ghb,
        solver_type="pcg",
        aq_thickness=aq_thickness,
    ) as solver:
        solver.build_from_fields(
            T_field=t_field,
            R_field=r_field,
            active=case.active,
            bc_mask=case.bc_mask,
            bc_values=case.bc_values,
            gh_mask=case.gh_mask,
            gh_head=case.gh_head,
            gh_width=case.gh_width,
            gh_alpha=gh_alpha,
        )
        solver.build_hierarchy(max_levels=int(max_levels), min_coarse_n=4)
        head, solver_info = solver.solve_multigrid_kcycle(
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
    solve_seconds = float(time.perf_counter() - start)

    obs_i, obs_j, obs, sigma = _prepare_obs_arrays(case)
    sim = np.asarray(head[obs_i, obs_j], dtype=float)
    metrics = _compute_metrics(pred=sim, obs=obs, sigma=sigma)

    solver_info = dict(solver_info or {})
    convergence = {
        "converged": bool(solver_info.get("converged", False)),
        "n_cycles_used": int(solver_info["n_cycles_used"]) if solver_info.get("n_cycles_used") is not None else None,
        "max_cycles": int(solver_info["max_cycles"]) if solver_info.get("max_cycles") is not None else int(max_cycles),
        "r_rms_end": float(solver_info["r_rms_end"]) if solver_info.get("r_rms_end") is not None else None,
        "r_max_end": float(solver_info["r_max_end"]) if solver_info.get("r_max_end") is not None else None,
        "tol_abs": float(solver_info["tol_abs"]) if solver_info.get("tol_abs") is not None else None,
        "tol_rel": float(solver_info["tol_rel"]) if solver_info.get("tol_rel") is not None else None,
    }

    return {
        "grid_shape": [int(case.ny), int(case.nx)],
        "n_active": int(np.count_nonzero(case.active)),
        "base_parameters": {
            "t_base": float(t_base),
            "r_scale": float(r_scale),
            "gh_alpha": _summarize_scalar_or_grid(gh_alpha),
            "aq_thickness": _summarize_scalar_or_grid(aq_thickness),
        },
        "solver": {
            "device": str(device),
            "use_ghb": bool(use_ghb),
            "max_cycles": int(max_cycles),
            "nu_pre": int(nu_pre),
            "nu_post": int(nu_post),
            "nu_coarse": int(nu_coarse),
            "omega": float(omega),
            "rel_tol": float(rel_tol),
            "abs_tol_min": float(abs_tol_min),
            "max_levels": int(max_levels),
            "check_every_no": int(check_every_no),
        },
        "metrics": metrics,
        "convergence": convergence,
        "solver_info": solver_info,
        "timing": {
            "solve_seconds": solve_seconds,
        },
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a one-shot Canterbury case-study forward solve and report convergence."
    )
    parser.add_argument("--grid-size", type=int, default=100)
    parser.add_argument("--inputs-dir", type=Path, default=None)
    parser.add_argument("--device", default="cuda:0", help="CUDA device, e.g. cuda:0 (GPU only).")
    parser.add_argument(
        "--use-ghb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable general-head boundaries.",
    )
    parser.add_argument("--gh-alpha", type=float, default=2.0)
    parser.add_argument("--aq-thickness", type=float, default=300.0)
    parser.add_argument("--t-min", type=float, default=100.0)
    parser.add_argument("--t-max", type=float, default=10000.0)
    parser.add_argument("--r-scale", type=float, default=0.5)
    parser.add_argument("--max-cycles", type=int, default=200)
    parser.add_argument("--nu-pre", type=int, default=2)
    parser.add_argument("--nu-post", type=int, default=2)
    parser.add_argument("--nu-coarse", type=int, default=2)
    parser.add_argument("--omega", type=float, default=0.7)
    parser.add_argument("--rel-tol", type=float, default=5.0e-6)
    parser.add_argument("--abs-tol-min", type=float, default=5.0e-6)
    parser.add_argument("--max-levels", type=int, default=6)
    parser.add_argument("--check-every-no", type=int, default=10)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path(__file__).parent / "results" / "forward_base_summary.json",
        help="Optional output JSON path.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    result = run_forward_case_study(
        grid_size=int(args.grid_size),
        inputs_dir=args.inputs_dir,
        device=str(args.device),
        use_ghb=bool(args.use_ghb),
        gh_alpha=args.gh_alpha,
        aq_thickness=args.aq_thickness,
        t_min=float(args.t_min),
        t_max=float(args.t_max),
        r_scale=float(args.r_scale),
        max_cycles=int(args.max_cycles),
        nu_pre=int(args.nu_pre),
        nu_post=int(args.nu_post),
        nu_coarse=int(args.nu_coarse),
        omega=float(args.omega),
        rel_tol=float(args.rel_tol),
        abs_tol_min=float(args.abs_tol_min),
        max_levels=int(args.max_levels),
        check_every_no=int(args.check_every_no),
    )

    out_json = None if str(args.out_json).strip().lower() in {"none", "null"} else Path(args.out_json)
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with out_json.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=_json_default)
        print(f"Wrote summary JSON: {out_json}")

    conv = result["convergence"]
    metrics = result["metrics"]
    print(
        "Case-study forward convergence:",
        f"converged={conv['converged']}",
        f"cycles={conv['n_cycles_used']}/{conv['max_cycles']}",
        f"r_rms_end={conv['r_rms_end']}",
        f"tol_abs={conv['tol_abs']}",
        f"rmse={metrics['rmse']:.4f}",
        f"weighted_rmse={metrics['weighted_rmse']:.4f}",
        f"solve_s={result['timing']['solve_seconds']:.3f}",
    )


if __name__ == "__main__":
    main()
