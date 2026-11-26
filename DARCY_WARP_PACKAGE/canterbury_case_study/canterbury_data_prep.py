from __future__ import annotations

from dataclasses import dataclass, field
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
    boundary_audit: dict = field(default_factory=dict)


def fitted_transmissivity_pilots(fit: dict, perturbation_strength: float) -> np.ndarray:
    """Recover the exact pilot grid used for the winning hydraulic solve."""
    pilots = fit.get("best_realised_T_pilots")
    if pilots is None:
        if perturbation_strength > 0.0:
            raise ValueError("Stochastic fit has no saved realised pilot grid and cannot be replayed exactly.")
        pilots = fit.get("best_T_pilots")
        if pilots is None:
            if fit.get("best_logT") is None:
                raise ValueError("Missing fitted transmissivity pilots and best_logT.")
            pilots = np.power(10.0, np.asarray(fit["best_logT"], dtype=float))
    return np.asarray(pilots, dtype=float).reshape(fit["pilot_shape"])


def _load_npz(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing inputs file: {path}")
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def resolve_fixed_heads(*, active, stages, masks, policy):
    """Resolve prescribed heads by source and report missing or overlapping stages.

    The prepared archive uses zero background outside each source raster.
    Explicit source masks take precedence over that encoding and can
    identify prescribed stages that are exactly zero.
    """
    if policy not in {"sea_precedence", "legacy_archive"}:
        raise ValueError("Unknown Canterbury boundary policy")
    source_precedence = ("sea", "south", "north")
    assignment_order = tuple(reversed(source_precedence))
    if set(stages) != set(source_precedence):
        raise ValueError("Canterbury stages must contain sea, south and north sources")
    assigned = {}
    missing = []
    for name in assignment_order:
        stage = stages[name]
        selected = active & masks.get(name, stage != 0.0)
        for row, col in np.argwhere(selected & ~np.isfinite(stage)):
            missing.append(dict(row=int(row), column=int(col), source=name))
        assigned[name] = selected & np.isfinite(stage)
    overlaps = []
    combined_count = sum(assigned.values())
    for row, col in np.argwhere(combined_count > 1):
        values = {name: float(stages[name][row, col]) for name in assignment_order
                  if assigned[name][row, col]}
        selected_source = next(name for name in source_precedence
                               if assigned[name][row, col])
        overlaps.append(dict(row=int(row), column=int(col), stages=values,
            selected_source=selected_source if policy == "sea_precedence" else "legacy_sum"))
    if policy == "legacy_archive":
        fixed = active & np.logical_or.reduce([stage != 0.0 for stage in stages.values()])
        values = np.where(fixed, np.nan_to_num(sum(stages.values()), nan=0.0), 0.0)
    else:
        fixed = combined_count > 0
        values = np.zeros(active.shape, dtype=np.float32)
        for name in assignment_order:
            stage = stages[name]
            values[assigned[name]] = stage[assigned[name]]
    return fixed.astype(np.int32), values, dict(
        policy=policy, missing_stages=missing, overlaps=overlaps,
        source_precedence=list(source_precedence), west_boundary="no flow",
    )


def load_case_inputs(
    grid_size: int = 100,
    inputs_dir: Path | None = None,
    recharge_floor: float = 1.0e-6,
    boundary_policy: str = "sea_precedence",
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

    stages = dict(north=fixed_nth, south=fixed_sth, sea=fixed_sea)
    source_keys = dict(north="fixed_nth_mask", south="fixed_sth_mask", sea="fixed_sea_mask")
    masks = {name: np.asarray(bc_npz[key], dtype=bool) for name, key in source_keys.items() if key in bc_npz}
    bc_mask, bc_values, boundary_audit = resolve_fixed_heads(
        active=active, stages=stages, masks=masks, policy=boundary_policy,
    )

    rch_npz = _load_npz(inputs_dir / f"recharge_{grid_size}.npz")
    if "x_coords" in rch_npz and "y_coords" in rch_npz:
        for item in [*boundary_audit["missing_stages"], *boundary_audit["overlaps"]]:
            item.update(nztm_easting=float(rch_npz["x_coords"][item["column"]]),
                nztm_northing=float(rch_npz["y_coords"][item["row"]]))
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
        boundary_audit=boundary_audit,
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
