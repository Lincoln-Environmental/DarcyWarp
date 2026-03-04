from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd


@dataclass
class CanterburyCaseInputs:
    nx: int
    ny: int
    dx: float
    grid: np.ndarray
    active: np.ndarray
    model_top: np.ndarray
    min_elev: np.ndarray
    gh_mask: np.ndarray
    gh_head: np.ndarray
    gh_width: np.ndarray
    bc_mask: np.ndarray
    bc_values: np.ndarray
    recharge_base: np.ndarray
    obs_df: pd.DataFrame
    fixed_nth: np.ndarray
    fixed_sth: np.ndarray
    fixed_wst: np.ndarray
    fixed_sea: np.ndarray


def _load_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing inputs file: {path}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load_case_inputs(
    grid_size: int = 100,
    inputs_dir: Path | None = None,
    recharge_floor: float = 1.0e-6,
) -> CanterburyCaseInputs:
    """
    Load Canterbury case inputs from the prepared npz/csv files.
    """
    if inputs_dir is None:
        inputs_dir = Path(__file__).parent.joinpath("inputs")

    grid_npz = _load_npz(inputs_dir / f"grid_{grid_size}.npz")
    grid = np.asarray(grid_npz["grid"], dtype=np.int32)
    model_top = np.asarray(grid_npz["model_top"], dtype=np.float32)
    min_elev = np.asarray(grid_npz["min_elev"], dtype=np.float32)
    gh_head = np.asarray(grid_npz["gh_head_np"], dtype=np.float32)
    gh_width = np.asarray(grid_npz["gh_width_np"], dtype=np.float32)

    if grid.ndim != 3 or grid.shape[0] != 1:
        raise ValueError(f"Expected grid shape (1, ny, nx), got {grid.shape}")

    active = grid[0] == 1
    ny, nx = active.shape

    # GHB arrays only on active cells
    gh_mask = (gh_width > 0.0) & active
    gh_head = np.where(gh_mask, gh_head, 0.0).astype(np.float32, copy=False)
    gh_width = np.where(gh_mask, gh_width, 0.0).astype(np.float32, copy=False)

    bc_npz = _load_npz(inputs_dir / f"fixed_bcs_{grid_size}.npz")
    fixed_nth = np.asarray(bc_npz["fixed_nth"], dtype=np.float32)
    fixed_sth = np.asarray(bc_npz["fixed_sth"], dtype=np.float32)
    fixed_wst = np.asarray(bc_npz["fixed_wst"], dtype=np.float32)
    fixed_sea = np.asarray(bc_npz["fixed_sea"], dtype=np.float32)

    if fixed_nth.shape != (ny, nx):
        raise ValueError("fixed_nth shape mismatch with grid")
    if fixed_sth.shape != (ny, nx):
        raise ValueError("fixed_sth shape mismatch with grid")
    if fixed_sea.shape != (ny, nx):
        raise ValueError("fixed_sea shape mismatch with grid")

    # Combine boundary value grids (west is no-flow in the original setup)
    bc_values = fixed_nth + fixed_sth + fixed_sea
    bc_values = np.nan_to_num(bc_values, nan=0.0).astype(np.float32, copy=False)

    bc_mask = (
        (fixed_nth != 0.0)
        | (fixed_sth != 0.0)
        | (fixed_sea != 0.0)
    )
    bc_mask = (bc_mask & active).astype(np.int32)
    bc_values = np.where(bc_mask != 0, bc_values, 0.0).astype(np.float32, copy=False)

    rch_npz = _load_npz(inputs_dir / f"recharge_{grid_size}.npz")
    recharge_grid = np.asarray(rch_npz["grid"], dtype=np.float32)
    if recharge_grid.shape != (ny, nx):
        raise ValueError("recharge grid shape mismatch with model grid")

    recharge_base = recharge_grid / 365.25
    recharge_base = np.where(np.isfinite(recharge_base), recharge_base, recharge_floor)
    recharge_base = np.where(recharge_base > 0.0, recharge_base, recharge_floor)
    recharge_base = recharge_base.astype(np.float32, copy=False)

    obs_path = inputs_dir / f"ss_obs_processed_{grid_size}.csv"
    if not obs_path.exists():
        raise FileNotFoundError(f"Missing obs CSV: {obs_path}")
    obs_df = pd.read_csv(obs_path)
    # drop sites where std equals the sentinel value used in the source dataset
    if "std_gwl" in obs_df.columns:
        obs_df = obs_df[obs_df["std_gwl"] != 2.520473050615595]
    elif "std" in obs_df.columns:
        obs_df = obs_df[obs_df["std"] != 2.520473050615595]

    return CanterburyCaseInputs(
        nx=int(nx),
        ny=int(ny),
        dx=float(grid_size),
        grid=grid,
        active=active.astype(np.int32),
        model_top=model_top,
        min_elev=min_elev,
        gh_mask=gh_mask.astype(np.int32),
        gh_head=gh_head,
        gh_width=gh_width,
        bc_mask=bc_mask,
        bc_values=bc_values,
        recharge_base=recharge_base,
        obs_df=obs_df,
        fixed_nth=fixed_nth,
        fixed_sth=fixed_sth,
        fixed_wst=fixed_wst,
        fixed_sea=fixed_sea,
    )


def export_active_cells_geotiff(
    active: np.ndarray,
    out_path: Path,
    extent: tuple[float, float, float, float],
    crs: str = "EPSG:2193",
    rows_increase_south: bool = True,
) -> None:
    """
    Export the active cell mask as a GeoTIFF.

    :param active: Active mask array (ny, nx) with 1 for active and 0 for inactive.
    :param out_path: Output GeoTIFF path.
    :param extent: (xmin, xmax, ymin, ymax) domain extent in projected coordinates.
    :param crs: CRS string, default NZTM (EPSG:2193).
    :param rows_increase_south: If True, row index increases southward (common for rasters).
    """
    try:
        import rasterio
        from rasterio.transform import from_bounds
    except Exception as exc:
        raise RuntimeError("rasterio is required to write GeoTIFF outputs.") from exc

    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    if xmin >= xmax or ymin >= ymax:
        raise ValueError(f"Invalid extent: {extent}")

    if active.ndim != 2:
        raise ValueError(f"Expected 2D active mask, got shape {active.shape}")

    ny, nx = active.shape
    data = np.asarray(active, dtype=np.uint8)
    if not rows_increase_south:
        data = np.flipud(data)

    transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="uint8",
        crs=crs,
        transform=transform,
        nodata=0,
        compress="lzw",
    ) as dst:
        dst.write(data, 1)


if __name__ == "__main__":
    case = load_case_inputs()
    print(
        "Loaded Canterbury inputs:",
        f"grid=({case.ny},{case.nx}), dx={case.dx}",
        f"obs={len(case.obs_df)}",
    )
