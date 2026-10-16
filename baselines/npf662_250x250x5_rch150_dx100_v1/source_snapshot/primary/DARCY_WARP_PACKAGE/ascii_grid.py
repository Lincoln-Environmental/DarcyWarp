from __future__ import annotations

from pathlib import Path
import numpy as np


def write_ascii_grid(
    path: str | Path,
    grid: np.ndarray,
    *,
    x_min: float,
    y_min: float,
    cellsize: float,
    nodata_value: float = -9999.0,
) -> Path:
    """Write an ESRI ASCII raster (".asc").

    Contract:
    - `grid` must be a 2D array shaped (ny, nx)
    - The *model extent* is defined as:
        x in [x_min, x_min + nx*cellsize]
        y in [y_min, y_min + ny*cellsize]
    - The ASC header uses `xllcorner`, `yllcorner`.

    Notes on orientation:
    - ESRI ASCII expects the first row written to be the *top* (north) row.
    - In most NumPy grids, row 0 is the top row already (north-most). We therefore
      write rows in increasing i.
    """

    p = Path(path)
    arr = np.asarray(grid)
    if arr.ndim != 2:
        raise ValueError(f"grid must be 2D (ny,nx); got shape {arr.shape}")

    ny, nx = arr.shape

    # Replace NaNs by nodata for safety
    if np.issubdtype(arr.dtype, np.floating):
        arr_out = np.where(np.isfinite(arr), arr, nodata_value)
    else:
        arr_out = arr

    header = (
        f"ncols         {nx}\n"
        f"nrows         {ny}\n"
        f"xllcorner     {float(x_min)}\n"
        f"yllcorner     {float(y_min)}\n"
        f"cellsize      {float(cellsize)}\n"
        f"NODATA_value  {float(nodata_value)}\n"
    )

    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        f.write(header)
        # Space-separated values per row
        # Use fmt that works for both ints and floats.
        np.savetxt(f, arr_out, fmt="%g", delimiter=" ")

    return p


def export_case_active_mask_to_asc(
    case,
    out_path: str | Path,
    *,
    origin: tuple[float, float] = (0.0, 0.0),
    nodata_value: float = -9999.0,
    active_value: int = 1,
    inactive_value: int = 0,
) -> Path:
    """Export `case.active` mask to ESRI ASCII with extent = model domain bounds.

    This is intended to be "quick" and work with your existing case objects.

    Required attrs on `case`:
    - case.active : array-like (ny, nx) where True/1 means active
    - case.dx     : cell size in model units

    Parameters
    - origin: (x_min, y_min) lower-left corner of the model domain.
      If you have real-world coordinates, pass them here.

    Output grid values
    - inactive -> `inactive_value`
    - active   -> `active_value`
    """

    if not hasattr(case, "active"):
        raise AttributeError("case must have an 'active' attribute")
    if not hasattr(case, "dx"):
        raise AttributeError("case must have a 'dx' attribute (cell size)")

    active = np.asarray(case.active).astype(bool)
    mask_grid = np.where(active, int(active_value), int(inactive_value)).astype(np.int32)

    x_min, y_min = origin
    return write_ascii_grid(
        out_path,
        mask_grid,
        x_min=float(x_min),
        y_min=float(y_min),
        cellsize=float(case.dx),
        nodata_value=float(nodata_value),
    )
