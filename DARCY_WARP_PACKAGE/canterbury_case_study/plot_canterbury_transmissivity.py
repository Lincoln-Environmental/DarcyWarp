from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
from matplotlib.colors import LogNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import load_case_inputs


DEFAULT_EXTENT = (1490750.0, 1581450.0, 5138850.0, 5201550.0)
COLORBAR_LABEL_SIZE = 5
COLORBAR_TICK_SIZE = 7


def precompute_rbf_cache(
    nx: int,
    ny: int,
    nx_p: int,
    ny_p: int,
    epsilon: float = 10.0,
) -> dict:
    pilot_x_coarse, pilot_y_coarse = np.meshgrid(
        np.arange(nx_p, dtype=np.float32),
        np.arange(ny_p, dtype=np.float32),
    )

    max_fine_x = float(nx - 1)
    max_fine_y = float(ny - 1)

    pilot_xs = max_fine_x / float(nx_p - 1) * pilot_x_coarse
    pilot_ys = max_fine_y / float(ny_p - 1) * pilot_y_coarse

    pilot_points = np.vstack((pilot_xs.ravel(), pilot_ys.ravel())).T

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


def _build_edges(
    extent: tuple[float, float, float, float],
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = [float(v) for v in extent]
    x_edges = np.linspace(xmin, xmax, nx + 1)
    # Row 0 is north, so y edges run from north (max) to south (min).
    y_edges = np.linspace(ymax, ymin, ny + 1)
    return x_edges, y_edges


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


def _nice_scale_km(target_km: float) -> float:
    if target_km <= 0.0:
        return 0.0
    magnitude = 10.0 ** np.floor(np.log10(target_km))
    candidates = np.array([1.0, 2.0, 5.0, 10.0]) * magnitude
    below = candidates[candidates <= target_km]
    if below.size == 0:
        return float(candidates[0])
    return float(below.max())


def _add_scale_bar(ax, length_km: float, pad: float = 0.06) -> None:
    if length_km <= 0.0:
        return
    try:
        from pyproj import Geod
    except Exception as exc:
        raise RuntimeError("pyproj is required to draw the scale bar.") from exc

    lon_min, lon_max = ax.get_xlim()
    lat_min, lat_max = ax.get_ylim()
    lon_range = lon_max - lon_min
    lat_range = lat_max - lat_min

    lon0 = lon_min + pad * lon_range
    lat0 = lat_min + pad * lat_range

    geod = Geod(ellps="WGS84")
    lon1, lat1, _ = geod.fwd(lon0, lat0, 90.0, length_km * 1000.0)
    _, lat_tick, _ = geod.fwd(lon0, lat0, 0.0, length_km * 1000.0 * 0.08)
    tick = abs(lat_tick - lat0)

    ax.plot(
        [lon0, lon1],
        [lat0, lat1],
        color="black",
        linewidth=2.0,
        solid_capstyle="butt",
        zorder=6,
    )
    ax.plot([lon0, lon0], [lat0 - tick, lat0 + tick], color="black", linewidth=2.0, zorder=6)
    ax.plot([lon1, lon1], [lat1 - tick, lat1 + tick], color="black", linewidth=2.0, zorder=6)
    ax.text(
        (lon0 + lon1) * 0.5,
        lat0 + tick * 1.6,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color="black",
        zorder=6,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot pilot transmissivity and reconstructed transmissivity field "
            "from Canterbury case-study results_summary.json."
        )
    )
    parser.add_argument(
        "--results-summary",
        type=Path,
        default=Path(__file__).parent / "results" / "results_summary.json",
        help="Path to results_summary.json.",
    )
    parser.add_argument(
        "--stage",
        choices=["auto", "stage1", "stage2", "root"],
        default="auto",
        help="Which stage to plot when the results contain staged outputs.",
    )
    parser.add_argument("--grid-size", type=int, default=100, help="Grid size in meters.")
    parser.add_argument("--inputs-dir", type=Path, default=None, help="Override inputs directory.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for plots and npz (defaults to results-summary directory).",
    )
    parser.add_argument(
        "--log-scale",
        action="store_true",
        help="Use a logarithmic color scale for transmissivity plots.",
    )
    parser.add_argument("--cmap", type=str, default="viridis", help="Matplotlib colormap.")
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
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Alpha for the transmissivity overlay (0-1).",
    )
    parser.add_argument(
        "--scale-bar-km",
        type=float,
        default=None,
        help="Scale bar length in km (default is 1/10 map width).",
    )
    parser.add_argument(
        "--no-scale-bar",
        action="store_true",
        help="Disable the scale bar.",
    )
    parser.add_argument(
        "--no-pilot-plot",
        action="store_true",
        help="Skip the pilot-point transmissivity plot.",
    )
    parser.add_argument(
        "--no-field-plot",
        action="store_true",
        help="Skip the reconstructed transmissivity field plot.",
    )
    return parser.parse_args()


def _load_results_summary(path: Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing results summary: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("results_summary.json must contain a JSON object at the top level.")
    return data


def _select_summary(data: dict, stage: str) -> tuple[dict, str]:
    if stage == "auto":
        if "stage2" in data:
            return data["stage2"], "stage2"
        if "stage1" in data:
            return data["stage1"], "stage1"
        return data, "root"
    if stage == "root":
        return data, "root"
    if stage not in data:
        raise KeyError(f"Stage '{stage}' not found in results summary.")
    return data[stage], stage


def _extract_pilots(summary: dict) -> tuple[np.ndarray, float, float, float, tuple[int, int]]:
    pilot_shape = summary.get("pilot_shape")
    if pilot_shape is None or len(pilot_shape) != 2:
        raise ValueError("Missing or invalid pilot_shape in results summary.")
    pilot_ny, pilot_nx = int(pilot_shape[0]), int(pilot_shape[1])

    if "best_T_pilots" in summary:
        T_pilots = np.asarray(summary["best_T_pilots"], dtype=float)
    elif "best_logT" in summary:
        logT = np.asarray(summary["best_logT"], dtype=float)
        T_pilots = np.power(10.0, logT)
    else:
        raise ValueError("results summary must include best_T_pilots or best_logT.")

    if T_pilots.shape != (pilot_ny, pilot_nx):
        T_pilots = T_pilots.reshape(pilot_ny, pilot_nx)

    T_min = float(summary.get("T_min", np.min(T_pilots)))
    T_max = float(summary.get("T_max", np.max(T_pilots)))
    rbf_epsilon = float(summary.get("rbf_epsilon", 10.0))
    return T_pilots, T_min, T_max, rbf_epsilon, (pilot_ny, pilot_nx)


def _plot_transmissivity_map(
    data: np.ndarray,
    out_path: Path,
    title: str,
    cmap: str,
    log_scale: bool,
    lon_edges: np.ndarray,
    lat_edges: np.ndarray,
    provider: str,
    zoom: int | None,
    alpha: float,
    dpi: int,
    scale_bar_km: float | None,
    mask: np.ndarray | None = None,
) -> None:
    out_path = Path(out_path)
    data = np.asarray(data, dtype=float)
    if mask is not None:
        data = np.where(mask, data, np.nan)

    cmap_obj = plt.get_cmap(cmap)
    try:
        cmap_obj = cmap_obj.copy()
    except AttributeError:
        pass
    cmap_obj.set_bad(color=(0.0, 0.0, 0.0, 0.0))

    norm = None
    if log_scale:
        finite = np.isfinite(data) & (data > 0.0)
        if not np.any(finite):
            raise ValueError("No positive transmissivity values available for log scaling.")
        vmin = float(np.nanmin(data[finite]))
        vmax = float(np.nanmax(data[finite]))
        norm = LogNorm(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_xlim(float(np.nanmin(lon_edges)), float(np.nanmax(lon_edges)))
    ax.set_ylim(float(np.nanmin(lat_edges)), float(np.nanmax(lat_edges)))
    _set_map_aspect(ax, lat_edges)

    _add_basemap(ax, crs="EPSG:4326", provider=provider, zoom=zoom)

    im = ax.pcolormesh(
        lon_edges,
        lat_edges,
        data,
        shading="auto",
        cmap=cmap_obj,
        norm=norm,
        alpha=alpha,
        zorder=2,
    )

    label = "Transmissivity"
    if log_scale:
        label = f"{label} (log scale)"
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_SIZE)

    if scale_bar_km is not None:
        _add_scale_bar(ax, scale_bar_km)

    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    results_summary = _load_results_summary(args.results_summary)
    summary, stage_label = _select_summary(results_summary, args.stage)
    T_pilots, T_min, T_max, rbf_epsilon, pilot_shape = _extract_pilots(summary)

    case = load_case_inputs(grid_size=args.grid_size, inputs_dir=args.inputs_dir)
    pilot_ny, pilot_nx = pilot_shape
    rbf_cache = precompute_rbf_cache(
        nx=case.nx,
        ny=case.ny,
        nx_p=pilot_nx,
        ny_p=pilot_ny,
        epsilon=rbf_epsilon,
    )
    T_field = get_T_field_from_pilots_cached(
        T_pilot_raw=T_pilots,
        T_min=T_min,
        T_max=T_max,
        rbf_cache=rbf_cache,
        perturbation_strength=0.0,
    )
    T_field = np.where(case.active == 1, T_field, 0.0).astype(np.float32, copy=False)

    extent = tuple(args.extent)
    try:
        from pyproj import Transformer
    except Exception as exc:
        raise RuntimeError("pyproj is required for NZTM -> WGS84 transforms.") from exc

    transformer = Transformer.from_crs("EPSG:2193", "EPSG:4326", always_xy=True)
    x_edges, y_edges = _build_edges(extent, case.nx, case.ny)
    xg, yg = np.meshgrid(x_edges, y_edges)
    lon_edges, lat_edges = transformer.transform(xg, yg)

    pilot_x_edges, pilot_y_edges = _build_edges(extent, pilot_nx, pilot_ny)
    xg_p, yg_p = np.meshgrid(pilot_x_edges, pilot_y_edges)
    lon_edges_p, lat_edges_p = transformer.transform(xg_p, yg_p)

    if args.scale_bar_km is None:
        map_width_km = (extent[1] - extent[0]) / 1000.0
        scale_bar_km = _nice_scale_km(map_width_km * 0.1)
    else:
        scale_bar_km = float(args.scale_bar_km)
    if args.no_scale_bar:
        scale_bar_km = None

    out_dir = args.out_dir or Path(args.results_summary).parent
    out_dir.mkdir(exist_ok=True)

    field_npz = out_dir / f"transmissivity_field_{stage_label}.npz"
    np.savez(
        field_npz,
        T_field=T_field,
        T_pilots=T_pilots,
        active=case.active,
        rbf_epsilon=rbf_epsilon,
        T_min=T_min,
        T_max=T_max,
    )

    if not args.no_pilot_plot:
        pilot_path = out_dir / f"transmissivity_pilots_{stage_label}.png"
        _plot_transmissivity_map(
            T_pilots,
            pilot_path,
            f"Pilot transmissivity ({stage_label})",
            args.cmap,
            args.log_scale,
            lon_edges_p,
            lat_edges_p,
            args.basemap,
            args.zoom,
            args.alpha,
            args.dpi,
            scale_bar_km,
        )

    if not args.no_field_plot:
        field_path = out_dir / f"transmissivity_field_{stage_label}.png"
        _plot_transmissivity_map(
            T_field,
            field_path,
            f"Interpolated transmissivity ({stage_label})",
            args.cmap,
            args.log_scale,
            lon_edges,
            lat_edges,
            args.basemap,
            args.zoom,
            args.alpha,
            args.dpi,
            scale_bar_km,
            mask=case.active == 1,
        )

    print(f"Wrote transmissivity field to: {field_npz}")
    if not args.no_pilot_plot:
        print(f"Wrote pilot transmissivity plot to: {pilot_path}")
    if not args.no_field_plot:
        print(f"Wrote interpolated transmissivity plot to: {field_path}")


if __name__ == "__main__":
    main()
