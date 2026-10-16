from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import load_case_inputs

DEFAULT_EXTENT = (1490750.0, 1581450.0, 5138850.0, 5201550.0)
COLORBAR_LABEL_SIZE = 8
COLORBAR_TICK_SIZE = 7


def _set_map_aspect(ax, lat_edges: np.ndarray) -> None:
    mean_lat = float(np.nanmean(lat_edges))
    if not np.isfinite(mean_lat):
        ax.set_aspect("equal", adjustable="box")
        return
    cos_lat = float(np.cos(np.deg2rad(mean_lat)))
    if cos_lat <= 0.0 or not np.isfinite(cos_lat):
        ax.set_aspect("equal", adjustable="box")
        return
    ax.set_aspect(1.0 / cos_lat, adjustable="box")


def _default_out_dir() -> Path:
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "paper" / "tables_figures" / "canterbury_case_study"


def _load_head(head_path: Path) -> np.ndarray:
    head_path = Path(head_path)
    if not head_path.exists():
        raise FileNotFoundError(f"Missing head file: {head_path}")
    with np.load(head_path) as data:
        if "head" not in data.files:
            raise KeyError(f"'head' array not found in {head_path}")
        head = np.asarray(data["head"], dtype=float)
    if head.ndim != 2:
        raise ValueError(f"Expected 2D head array, got shape {head.shape}")
    return head


def _build_edges(extent: tuple[float, float, float, float], nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    x_edges = np.linspace(xmin, xmax, nx + 1)
    # Row 0 is north, so y edges run from north (max) to south (min).
    y_edges = np.linspace(ymax, ymin, ny + 1)
    return x_edges, y_edges


def _add_basemap(ax, crs: str, provider: str, zoom: int | None) -> None:
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


def _plot_overlays(
    ax,
    case,
    transformer,
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    head: np.ndarray,
    lon_edges: np.ndarray | None = None,
    lat_edges: np.ndarray | None = None,
) -> None:
    obs_df = case.obs_df
    obs_plotted = False
    if {"nztm_x", "nztm_y", "i", "j", "gwl"}.issubset(obs_df.columns):
        i_idx = obs_df["i"].astype(int).to_numpy()
        j_idx = obs_df["j"].astype(int).to_numpy()
        in_bounds = (
            (i_idx >= 0)
            & (i_idx < case.ny)
            & (j_idx >= 0)
            & (j_idx < case.nx)
        )
        if np.any(in_bounds):
            i_idx = i_idx[in_bounds]
            j_idx = j_idx[in_bounds]
            obs_gwl = obs_df.loc[in_bounds, "gwl"].to_numpy(dtype=float)
            obs_x = obs_df.loc[in_bounds, "nztm_x"].to_numpy(dtype=float)
            obs_y = obs_df.loc[in_bounds, "nztm_y"].to_numpy(dtype=float)

            active_mask = case.active[i_idx, j_idx] == 1
            if np.any(active_mask):
                i_idx = i_idx[active_mask]
                j_idx = j_idx[active_mask]
                obs_gwl = obs_gwl[active_mask]
                obs_x = obs_x[active_mask]
                obs_y = obs_y[active_mask]

                sim = head[i_idx, j_idx]
                residual = sim - obs_gwl
                finite_mask = np.isfinite(residual)
                if np.any(finite_mask):
                    residual = residual[finite_mask]
                    obs_x = obs_x[finite_mask]
                    obs_y = obs_y[finite_mask]

                    obs_lon, obs_lat = transformer.transform(obs_x, obs_y)
                    bounds = np.array(
                        [-15.0, -10.0, -5.0, 5.0, 10.0, 15.0],
                        dtype=float,
                    )
                    colors = [
                        "#2166ac",
                        "#92c5de",
                        "#f7f7f7",
                        "#f4a582",
                        "#b2182b",
                    ]
                    cmap = ListedColormap(colors)
                    norm = BoundaryNorm(bounds, cmap.N, clip=True)
                    scatter = ax.scatter(
                        obs_lon,
                        obs_lat,
                        s=14,
                        c=residual,
                        cmap=cmap,
                        norm=norm,
                        alpha=0.85,
                        edgecolors="black",
                        linewidth=0.25,
                        label="Obs misfit",
                        zorder=5,
                    )
                    cbar = ax.figure.colorbar(
                        scatter,
                        ax=ax,
                        orientation="horizontal",
                        shrink=0.5,
                        pad=0.08,
                    )
                    cbar.set_ticks([-12.5, -7.5, 0, 7.5, 12.5])
                    cbar.set_ticklabels(
                        [
                            "<-10",
                            "-10 to -5",
                            "-5 to 5",
                            "5 to 10",
                            ">10",
                        ]
                    )
                    cbar.set_label("Model - obs (m)", fontsize=COLORBAR_LABEL_SIZE)
                    # place cbar on bottom
                    cbar.ax.yaxis.set_ticks_position("left")
                    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)
                    obs_plotted = True

    if not obs_plotted and {"nztm_x", "nztm_y"}.issubset(obs_df.columns):
        obs_lon, obs_lat = transformer.transform(
            obs_df["nztm_x"].to_numpy(),
            obs_df["nztm_y"].to_numpy(),
        )
        ax.scatter(
            obs_lon,
            obs_lat,
            s=8,
            c="black",
            alpha=0.6,
            label="Obs heads",
            zorder=5,
        )

    bc_layers = [
        ("Fixed head (north)", case.fixed_nth != 0.0, "#1f77b4"),
        ("Fixed head (south)", case.fixed_sth != 0.0, "#17becf"),
        ("Fixed head (sea)", case.fixed_sea != 0.0, "#2ca02c"),
    ]

    for label, mask, color in bc_layers:
        idx = np.where(mask)
        if idx[0].size == 0:
            continue
        xs = x_centers[idx[1]]
        ys = y_centers[idx[0]]
        lon, lat = transformer.transform(xs, ys)
        ax.scatter(lon, lat, s=10, c=color, alpha=0.8, label=label, zorder=4)

    if lon_edges is not None and lat_edges is not None and np.any(case.gh_mask):
        ghb_plot = np.ma.masked_where(case.gh_mask == 0, case.gh_mask)
        ghb_cmap = ListedColormap(["#d62728"])
        ax.pcolormesh(
            lon_edges,
            lat_edges,
            ghb_plot,
            shading="auto",
            cmap=ghb_cmap,
            alpha=0.6,
            edgecolors="none",
            linewidth=0.1,
            antialiased=False,
            zorder=3,
        )
        ax.scatter(
            [],
            [],
            s=36,
            facecolors="#d62728",
            edgecolors="none",
            marker="s",
            alpha=0.6,
            label="GHB cells",
            zorder=1,
        )


def _plot_head_map(
    head: np.ndarray,
    case,
    extent: tuple[float, float, float, float],
    out_path: Path,
    provider: str,
    zoom: int | None,
    dpi: int,
) -> None:
    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError("pyproj is required for NZTM -> WGS84 transforms.") from exc

    ny, nx = head.shape
    x_edges, y_edges = _build_edges(extent, nx, ny)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    xg, yg = np.meshgrid(x_edges, y_edges)
    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    lon_edges, lat_edges = transformer.transform(xg, yg)

    head_plot = np.where(case.active != 0, head, np.nan)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.nanmin(lon_edges)), float(np.nanmax(lon_edges)))
    ax.set_ylim(float(np.nanmin(lat_edges)), float(np.nanmax(lat_edges)))
    _set_map_aspect(ax, lat_edges)

    _add_basemap(ax, crs="EPSG:4326", provider=provider, zoom=zoom)

    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        head_plot,
        shading="auto",
        cmap="viridis",
        alpha=0.75,
        zorder=2,
    )
    cbar = fig.colorbar(mesh, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Head (m)", fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    _plot_overlays(
        ax,
        case,
        transformer,
        x_centers,
        y_centers,
        head,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    # ax.set_title("Canterbury case study: calibrated heads")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _plot_head_contours(
    head: np.ndarray,
    case,
    extent: tuple[float, float, float, float],
    out_path: Path,
    provider: str,
    zoom: int | None,
    dpi: int,
) -> None:
    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError("pyproj is required for NZTM -> WGS84 transforms.") from exc

    ny, nx = head.shape
    x_edges, y_edges = _build_edges(extent, nx, ny)
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    xc, yc = np.meshgrid(x_centers, y_centers)
    lon_c, lat_c = transformer.transform(xc, yc)
    xg, yg = np.meshgrid(x_edges, y_edges)
    lon_edges, lat_edges = transformer.transform(xg, yg)

    head_plot = np.where(case.active != 0, head, np.nan)
    h_min = float(np.nanmin(head_plot))
    h_max = float(np.nanmax(head_plot))
    levels = np.linspace(h_min, h_max, 12)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.nanmin(lon_edges)), float(np.nanmax(lon_edges)))
    ax.set_ylim(float(np.nanmin(lat_edges)), float(np.nanmax(lat_edges)))
    _set_map_aspect(ax, lat_edges)

    _add_basemap(ax, crs="EPSG:4326", provider=provider, zoom=zoom)

    ax.contour(
        lon_c,
        lat_c,
        head_plot,
        levels=levels,
        colors="black",
        linewidths=0.6,
        zorder=2,
    )

    _plot_overlays(
        ax,
        case,
        transformer,
        x_centers,
        y_centers,
        head,
        lon_edges=lon_edges,
        lat_edges=lat_edges,
    )
    ax.legend(loc="upper right", fontsize=8, frameon=True)
    # ax.set_title("Canterbury case study: head contours")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Canterbury case-study heads with an OpenStreetMap underlay."
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
        "--out-dir",
        type=Path,
        default=_default_out_dir(),
        help="Output directory for figures.",
    )
    parser.add_argument(
        "--extent",
        nargs=4,
        type=float,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX"),
        default=DEFAULT_EXTENT,
        help="NZTM extent for the grid.",
    )
    parser.add_argument(
        "--basemap",
        choices=("osm", "opentopo", "none"),
        default="osm",
        help="Basemap provider.",
    )
    parser.add_argument("--zoom", type=int, default=None, help="Basemap zoom override.")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI.")
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

    out_dir = Path(args.out_dir)

    # make output directory if it doesn't exist'
    out_dir.mkdir(exist_ok=True)

    _plot_head_map(
        head=head,
        case=case,
        extent=tuple(args.extent),
        out_path=out_dir / "canterbury_heads_map.png",
        provider=args.basemap,
        zoom=args.zoom,
        dpi=args.dpi,
    )
    _plot_head_contours(
        head=head,
        case=case,
        extent=tuple(args.extent),
        out_path=out_dir / "canterbury_heads_contours.png",
        provider=args.basemap,
        zoom=args.zoom,
        dpi=args.dpi,
    )


if __name__ == "__main__":

    main()
