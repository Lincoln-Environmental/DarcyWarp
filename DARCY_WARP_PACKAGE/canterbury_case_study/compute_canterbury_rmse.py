from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import load_case_inputs


def _prepare_obs_arrays(case):
    df = case.obs_df
    if not {"i", "j", "gwl"}.issubset(df.columns):
        raise ValueError("obs_df must include i, j, gwl columns")

    if "std_gwl" in df.columns:
        std_col = "std_gwl"
    elif "std" in df.columns:
        std_col = "std"
    else:
        raise ValueError("obs_df must include std_gwl or std column")

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
    std = df.loc[in_bounds, std_col].to_numpy(dtype=float)

    active_mask = case.active[i_idx, j_idx] == 1

    i_idx = i_idx[active_mask]
    j_idx = j_idx[active_mask]
    gwl = gwl[active_mask]
    std = std[active_mask]

    weights = 1.0 / np.clip(std, 0.2, None)
    return i_idx, j_idx, gwl, weights


def _load_head(path: Path) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing head file: {path}")
    with np.load(path) as data:
        if "head" not in data.files:
            raise KeyError(f"'head' array not found in {path}")
        head = np.asarray(data["head"], dtype=float)
    if head.ndim != 2:
        raise ValueError(f"Expected 2D head array, got shape {head.shape}")
    return head


def _compute_metrics(head: np.ndarray, case) -> dict:
    i_idx, j_idx, obs_gwl, weights = _prepare_obs_arrays(case)
    sim = head[i_idx, j_idx]
    residual = sim - obs_gwl

    rmse = float(np.sqrt(np.mean(residual * residual)))
    weighted_sse = float(np.sum((weights * residual) ** 2))
    weighted_rmse = float(np.sqrt(weighted_sse / residual.size))
    mean_residual = float(np.mean(residual))

    return {
        "n_obs": int(residual.size),
        "rmse": rmse,
        "weighted_rmse": weighted_rmse,
        "weighted_sse": weighted_sse,
        "mean_residual": mean_residual,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Canterbury case-study RMSE from best_head.npz."
    )
    parser.add_argument("--grid-size", type=int, default=100, help="Grid size in meters.")
    parser.add_argument("--inputs-dir", type=Path, default=None, help="Override inputs directory.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).parent / "results",
        help="Directory containing best_head.npz.",
    )
    parser.add_argument(
        "--head-path",
        type=Path,
        default=None,
        help="Path to head npz (defaults to results-dir/best_head.npz).",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="Optional output JSON path for metrics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    case = load_case_inputs(grid_size=args.grid_size, inputs_dir=args.inputs_dir)

    head_path = args.head_path or (args.results_dir / "best_head.npz")
    head = _load_head(head_path)
    if head.shape != (case.ny, case.nx):
        raise ValueError(
            f"Head shape {head.shape} does not match grid ({case.ny}, {case.nx})."
        )

    metrics = _compute_metrics(head, case)
    print(json.dumps(metrics, indent=2))

    if args.out_json is not None:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
