# SPDX-License-Identifier: AGPL-3.0-or-later
"""Common grid and basemap geometry for Canterbury publication figures."""

from __future__ import annotations

import numpy as np


def _set_map_aspect(ax, lat_edges: np.ndarray) -> None:
    """Set map aspect using the mean latitude of the displayed grid.

    :param ax: Matplotlib axes to modify.
    :param lat_edges: Latitude edges in degrees; invalid latitude uses equal aspect.
    """
    mean_lat = float(np.nanmean(lat_edges))
    if not np.isfinite(mean_lat):
        ax.set_aspect("equal", adjustable="box")
        return
    cos_lat = float(np.cos(np.deg2rad(mean_lat)))
    if cos_lat <= 0.0 or not np.isfinite(cos_lat):
        ax.set_aspect("equal", adjustable="box")
        return
    ax.set_aspect(1.0 / cos_lat, adjustable="box")


def _build_edges(extent: tuple[float, float, float, float], nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    """Build cell-edge coordinates with north at row zero.

    :param extent: (xmin, xmax, ymin, ymax) in the plotted coordinate system.
    :param nx: Number of grid columns.
    :param ny: Number of grid rows.
    :returns: X edges west to east and Y edges north to south, lengths nx+1 and ny+1.
    """
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    x_edges = np.linspace(xmin, xmax, nx + 1)
    # Row 0 is north, so y edges run from north (max) to south (min).
    y_edges = np.linspace(ymax, ymin, ny + 1)
    return x_edges, y_edges


def _add_basemap(ax, crs: str, provider: str, zoom: int | None) -> None:
    """Add an optional contextily basemap underneath the plotted data.

    :param ax: Matplotlib axes containing the map.
    :param crs: Coordinate reference system of the axes.
    :param provider: none, opentopo or the default OpenStreetMap provider.
    :param zoom: Tile zoom level, or None for automatic zoom.
    :raises RuntimeError: If contextily cannot be imported.
    """
    if provider == "none":
        return
    try:
        import contextily as cx
    except Exception as exc:
        raise RuntimeError("contextily is required for basemap tiles. Install with `pip install contextily`.") from exc

    if provider == "opentopo":
        source = cx.providers.OpenTopoMap
    else:
        source = cx.providers.OpenStreetMap.Mapnik

    zoom_value = "auto" if zoom is None else zoom
    cx.add_basemap(
        ax,
        crs=crs,
        source=source,
        zoom=zoom_value,
        attribution_size=6,
        zorder=0,
    )
