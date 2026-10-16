from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "paper" / "tables_figures"
COMPLETION_CURVE_CANDIDATES = (
    OUT_DIR / "t_change" / "idealised_completion_curves.png",
    OUT_DIR / "recharge_change" / "idealised_completion_curves.png",
)


PALETTE = {
    "bg": "#f7f4ef",
    "panel_darcywarp": "#e4f2f6",
    "panel_mf6": "#f0f0f0",
    "panel_accent": "#e6efe9",
    "darcywarp": "#1b7f8b",
    "mf6": "#6c7076",
    "accent": "#d98c3b",
    "text": "#1a1a1a",
    "grid": "#9aa4a6",
}


def draw_grid(ax, x0: float, y0: float, w: float, h: float, rows: int, cols: int, color: str, alpha: float) -> None:
    for i in range(rows + 1):
        y = y0 + h * i / rows
        ax.plot([x0, x0 + w], [y, y], color=color, alpha=alpha, lw=0.6, zorder=3)
    for j in range(cols + 1):
        x = x0 + w * j / cols
        ax.plot([x, x], [y0, y0 + h], color=color, alpha=alpha, lw=0.6, zorder=3)


def resolve_completion_curve() -> Path | None:
    for path in COMPLETION_CURVE_CANDIDATES:
        if path.exists():
            return path
    return None


def draw_image(ax, img_path: Path, x0: float, y0: float, w: float, h: float) -> None:
    img = mpimg.imread(img_path)
    ax.imshow(
        img,
        extent=(x0, x0 + w, y0, y0 + h),
        origin="upper",
        aspect="auto",
        zorder=2,
    )
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=PALETTE["grid"], lw=0.8, zorder=4))


def draw_heatmap(
    ax,
    x0: float,
    y0: float,
    w: float,
    h: float,
    cmap: str,
    rows: int = 5,
    cols: int = 5,
    grid_alpha: float = 0.3,
) -> None:
    xs = np.linspace(0.0, 1.0, 12)
    ys = np.linspace(0.0, 1.0, 12)
    data = np.outer(ys, xs)
    ax.imshow(
        data,
        extent=(x0, x0 + w, y0, y0 + h),
        origin="lower",
        cmap=cmap,
        alpha=0.9,
        zorder=2,
        interpolation="bilinear",
    )
    ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor=PALETTE["grid"], lw=0.8, zorder=4))
    draw_grid(ax, x0, y0, w, h, rows=5, cols=5, color=PALETTE["grid"], alpha=0.3)


def draw_chip(ax, x0: float, y0: float, w: float, h: float, color: str, label: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color,
            edgecolor="none",
            zorder=5,
        )
    )
    pin_len = h * 0.15
    for i in range(6):
        px = x0 + w * (i + 0.5) / 6.0
        ax.plot([px, px], [y0 - pin_len, y0], color=color, lw=2, zorder=5)
        ax.plot([px, px], [y0 + h, y0 + h + pin_len], color=color, lw=2, zorder=5)
    ax.text(
        x0 + w / 2.0,
        y0 + h / 2.0,
        label,
        color="white",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        zorder=6,
    )


def draw_bottom_box(ax, x0: float, y0: float, w: float, h: float, title: str) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor="white",
            edgecolor=PALETTE["grid"],
            lw=0.8,
            zorder=1,
        )
    )
    ax.text(
        x0 + w / 2.0,
        y0 + 0.02,
        title,
        color=PALETTE["text"],
        ha="center",
        va="bottom",
        fontsize=9,
    )


def draw_head_agreement_icon(ax, x0: float, y0: float, w: float, h: float) -> None:
    xs = np.linspace(x0 + 0.08 * w, x0 + 0.92 * w, 30)
    ys1 = y0 + h * 0.6 + 0.08 * h * np.sin(np.linspace(0, 2.5, 30))
    ys2 = ys1 + 0.02 * h
    ax.plot(xs, ys1, color=PALETTE["darcywarp"], lw=2)
    ax.plot(xs, ys2, color=PALETTE["mf6"], lw=2, alpha=0.9)


def draw_residual_icon(ax, x0: float, y0: float, w: float, h: float) -> None:
    bar_w = 0.18 * w
    x_left = x0 + 0.28 * w
    x_right = x0 + 0.54 * w
    y_base = y0 + 0.22 * h
    height = 0.45 * h
    ax.add_patch(Rectangle((x_left, y_base), bar_w, height, color=PALETTE["darcywarp"], zorder=3))
    ax.add_patch(Rectangle((x_right, y_base), bar_w, height, color=PALETTE["mf6"], zorder=3))
    ax.plot([x_left - 0.08 * w, x_right + 0.26 * w], [y_base + height, y_base + height], color=PALETTE["grid"], lw=1)


def draw_mass_balance_icon(ax, x0: float, y0: float, w: float, h: float) -> None:
    box_w = 0.32 * w
    box_h = 0.38 * h
    box_x = x0 + 0.34 * w
    box_y = y0 + 0.34 * h
    ax.add_patch(Rectangle((box_x, box_y), box_w, box_h, fill=False, edgecolor=PALETTE["grid"], lw=1.2))
    ax.add_patch(
        FancyArrowPatch(
            (x0 + 0.15 * w, box_y + box_h / 2.0),
            (box_x, box_y + box_h / 2.0),
            arrowstyle="->",
            mutation_scale=12,
            lw=2,
            color=PALETTE["accent"],
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (box_x + box_w, box_y + box_h / 2.0),
            (x0 + 0.85 * w, box_y + box_h / 2.0),
            arrowstyle="->",
            mutation_scale=12,
            lw=2,
            color=PALETTE["accent"],
        )
    )
    ax.add_patch(Circle((box_x + box_w / 2.0, box_y + box_h / 2.0), 0.06 * h, color=PALETTE["darcywarp"]))


def draw_badge(
    ax,
    x0: float,
    y0: float,
    w: float,
    h: float,
    text: str,
    color: str,
    fontsize: float = 8.5,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            w,
            h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=color,
            edgecolor="none",
            zorder=7,
        )
    )
    ax.text(
        x0 + w / 2.0,
        y0 + h / 2.0,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="white",
        fontweight="bold",
        zorder=8,
    )


def draw_multigrid_icon(ax, x0: float, y0: float, w: float, h: float, color: str) -> None:
    for i in range(3):
        inset = i * (0.08 * w)
        ax.add_patch(
            Rectangle(
                (x0 + inset, y0 + inset),
                w - 2 * inset,
                h - 2 * inset,
                fill=False,
                edgecolor=color,
                lw=1.2,
                zorder=4,
            )
        )


def draw_scatter_inset(ax, x0: float, y0: float, w: float, h: float) -> None:
    ax.add_patch(Rectangle((x0, y0), w, h, facecolor="white", edgecolor=PALETTE["grid"], lw=0.8, zorder=6))
    t = np.linspace(0.1, 0.9, 10)
    ax.plot(x0 + w * t, y0 + h * t, color=PALETTE["grid"], lw=1.0, zorder=7)
    ax.plot(
        x0 + w * t,
        y0 + h * t + 0.02 * h * np.sin(np.linspace(0.0, 3.0, t.size)),
        linestyle="None",
        marker="o",
        markersize=3,
        color=PALETTE["darcywarp"],
        zorder=8,
    )


def draw_layout_a(ax) -> None:
    ax.text(
        0.5,
        0.95,
        "DarcyWarp GPU solver",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
        color=PALETTE["text"],
    )
    ax.text(
        0.5,
        0.92,
        "2D steady-state transmissivity | matched MODFLOW-style stresses",
        ha="center",
        va="center",
        fontsize=8.5,
        color=PALETTE["text"],
    )

    row_y0 = 0.2
    row_h = 0.6
    panel_w = 0.305
    gap = 0.04
    left_x0 = 0.005
    mid_x0 = left_x0 + panel_w + gap
    right_x0 = mid_x0 + panel_w + gap
    pad_x = panel_w * 0.06
    label_y = row_y0 + row_h * 0.9
    sub_label_y = row_y0 + row_h * 0.84
    map_y = row_y0 + row_h * 0.18

    def panel_box(x0: float) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x0, row_y0),
                panel_w,
                row_h,
                boxstyle="round,pad=0.01,rounding_size=0.02",
                facecolor="white",
                edgecolor="#c6cacc",
                lw=0.8,
            )
        )

    panel_box(left_x0)
    panel_box(mid_x0)
    panel_box(right_x0)

    # Inputs
    ax.text(
        left_x0 + pad_x,
        label_y,
        "Inputs",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["text"],
    )

    grid_x = left_x0 + pad_x
    grid_w = panel_w - 2.0 * pad_x
    map_h = row_h * 0.6
    draw_heatmap(ax, grid_x, map_y, grid_w, map_h, cmap="Greens", rows=10, cols=10, grid_alpha=0.45)

    # Recharge arrow
    ax.add_patch(
        FancyArrowPatch(
            (grid_x + grid_w * 0.5, map_y + map_h + 0.02),
            (grid_x + grid_w * 0.5, map_y + map_h * 0.86),
            arrowstyle="->",
            mutation_scale=12,
            lw=2,
            color=PALETTE["accent"],
        )
    )
    ax.text(
        grid_x + grid_w * 0.5,
        map_y + map_h + 0.03,
        "Recharge",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["text"],
    )

    # Boundary conditions: CHD on top, bottom, and right; GHB as internal line halfway down
    bc_lw = 3.0
    bc_z = 9

    # CHD top
    ax.plot(
        [grid_x, grid_x + grid_w],
        [map_y + map_h, map_y + map_h],
        color=PALETTE["mf6"],
        lw=bc_lw,
        zorder=bc_z,
    )
    # CHD bottom
    ax.plot(
        [grid_x, grid_x + grid_w],
        [map_y, map_y],
        color=PALETTE["mf6"],
        lw=bc_lw,
        zorder=bc_z,
    )
    # CHD right
    ax.plot(
        [grid_x + grid_w, grid_x + grid_w],
        [map_y, map_y + map_h],
        color=PALETTE["mf6"],
        lw=bc_lw,
        zorder=bc_z,
    )
    ax.text(
        grid_x + grid_w - 0.004,
        map_y + map_h + 0.012,
        "CHD",
        ha="right",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["text"],
        zorder=bc_z + 1,
    )

    # GHB internal line halfway down (left to right)
    mid_y = map_y + 0.5 * map_h
    t = np.linspace(0.0, 1.0, 160)
    wiggle = 0.006 * map_h * np.sin(6.0 * np.pi * t)
    ax.plot(
        grid_x + grid_w * t,
        mid_y + wiggle,
        color=PALETTE["darcywarp"],
        lw=2.4,
        zorder=bc_z,
    )
    ax.text(
        grid_x + grid_w * 0.5,
        mid_y + 0.012,
        "GHB",
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=PALETTE["text"],
        zorder=bc_z + 1,
    )

    ax.text(
        grid_x + grid_w * 0.5,
        map_y - 0.025,
        "100x100 grid",
        ha="center",
        va="top",
        fontsize=8,
        color=PALETTE["text"],
    )

    # Head fit vs MF6
    ax.text(
        mid_x0 + pad_x,
        label_y,
        "Head fit vs MF6",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["darcywarp"],
    )
    ax.text(
        mid_x0 + pad_x,
        sub_label_y,
        "DarcyWarp vs MODFLOW 6",
        ha="left",
        va="center",
        fontsize=8.5,
        color=PALETTE["text"],
    )

    scatter_x = mid_x0 + pad_x
    scatter_w = panel_w - 2.0 * pad_x
    draw_scatter_inset(ax, scatter_x, map_y, scatter_w, map_h)

    # Completion curves
    ax.text(
        right_x0 + pad_x,
        label_y,
        "Completion curves",
        ha="left",
        va="center",
        fontsize=11,
        fontweight="bold",
        color=PALETTE["text"],
    )
    ax.text(
        right_x0 + pad_x,
        sub_label_y,
        "Idealised ensemble throughput",
        ha="left",
        va="center",
        fontsize=8.5,
        color=PALETTE["text"],
    )

    heat_x = right_x0 + pad_x
    heat_w = panel_w - 2.0 * pad_x
    completion_curve = resolve_completion_curve()
    if completion_curve is not None:
        draw_image(ax, completion_curve, heat_x, map_y, heat_w, map_h)
    else:
        draw_heatmap(ax, heat_x, map_y, heat_w, map_h, cmap="cividis", rows=10, cols=10, grid_alpha=0.3)

    # Flow arrows
    flow_y = map_y + map_h * 0.5
    ax.add_patch(
        FancyArrowPatch(
            (left_x0 + panel_w + 0.006, flow_y),
            (mid_x0 - 0.006, flow_y),
            arrowstyle="->",
            mutation_scale=16,
            lw=2,
            color=PALETTE["accent"],
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            (mid_x0 + panel_w + 0.006, flow_y),
            (right_x0 - 0.006, flow_y),
            arrowstyle="->",
            mutation_scale=16,
            lw=2,
            color=PALETTE["accent"],
        )
    )

def draw_layout_b(ax) -> None:
    # Top strip
    top_y0 = 0.84
    top_h = 0.12
    ax.add_patch(
        FancyBboxPatch(
            (0.04, top_y0),
            0.92,
            top_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor=PALETTE["panel_accent"],
            edgecolor="none",
            zorder=1,
        )
    )
    draw_grid(ax, 0.06, top_y0 + 0.02, 0.08, top_h - 0.04, rows=3, cols=3, color=PALETTE["grid"], alpha=0.5)
    ax.text(
        0.52,
        top_y0 + top_h * 0.65,
        "Matched grids + stresses",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color=PALETTE["text"],
    )
    ax.text(
        0.52,
        top_y0 + top_h * 0.3,
        "2D steady-state transmissivity",
        ha="center",
        va="center",
        fontsize=9,
        color=PALETTE["text"],
    )

    # Middle panels
    mid_y0 = 0.27
    mid_h = 0.52
    left_x0, left_w = 0.05, 0.42
    right_x0, right_w = 0.53, 0.42

    ax.add_patch(
        FancyBboxPatch(
            (left_x0, mid_y0),
            left_w,
            mid_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=PALETTE["panel_darcywarp"],
            edgecolor="none",
        )
    )
    ax.add_patch(
        FancyBboxPatch(
            (right_x0, mid_y0),
            right_w,
            mid_h,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            facecolor=PALETTE["panel_mf6"],
            edgecolor="none",
        )
    )

    ax.text(
        left_x0 + 0.02,
        mid_y0 + mid_h * 0.9,
        "DarcyWarp (GPU)",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["darcywarp"],
    )
    ax.text(
        left_x0 + 0.02,
        mid_y0 + mid_h * 0.82,
        "K-cycle multigrid",
        ha="left",
        va="center",
        fontsize=9,
        color=PALETTE["text"],
    )

    ax.text(
        right_x0 + 0.02,
        mid_y0 + mid_h * 0.9,
        "MODFLOW 6 (CPU)",
        ha="left",
        va="center",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["mf6"],
    )
    ax.text(
        right_x0 + 0.02,
        mid_y0 + mid_h * 0.82,
        "Sparse FD reference",
        ha="left",
        va="center",
        fontsize=9,
        color=PALETTE["text"],
    )

    heat_w = left_w * 0.78
    heat_h = mid_h * 0.32
    heat_x = left_x0 + left_w * 0.11
    heat_y = mid_y0 + mid_h * 0.42
    draw_heatmap(ax, heat_x, heat_y, heat_w, heat_h, cmap="cividis")
    draw_chip(ax, left_x0 + left_w * 0.2, mid_y0 + mid_h * 0.12, left_w * 0.22, mid_h * 0.14, PALETTE["darcywarp"], "GPU")
    ax.text(
        left_x0 + left_w * 0.47,
        mid_y0 + mid_h * 0.16,
        "Python + Warp",
        ha="left",
        va="center",
        fontsize=8.5,
        color=PALETTE["text"],
    )

    heat_x_r = right_x0 + right_w * 0.11
    heat_y_r = heat_y
    completion_curve = resolve_completion_curve()
    if completion_curve is not None:
        draw_image(ax, completion_curve, heat_x_r, heat_y_r, heat_w, heat_h)
        ax.text(
            heat_x_r + heat_w * 0.02,
            heat_y_r + heat_h * 0.98,
            "Idealised completion curves",
            ha="left",
            va="top",
            fontsize=7,
            color=PALETTE["text"],
        )
    else:
        draw_heatmap(ax, heat_x_r, heat_y_r, heat_w, heat_h, cmap="cividis")
    draw_chip(ax, right_x0 + right_w * 0.2, mid_y0 + mid_h * 0.12, right_w * 0.22, mid_h * 0.14, PALETTE["mf6"], "CPU")
    ax.text(
        right_x0 + right_w * 0.47,
        mid_y0 + mid_h * 0.16,
        "MF6 IMS solve",
        ha="left",
        va="center",
        fontsize=8.5,
        color=PALETTE["text"],
    )

    # Comparison arrow
    ax.add_patch(
        FancyArrowPatch(
            (left_x0 + left_w + 0.01, mid_y0 + mid_h * 0.52),
            (right_x0 - 0.01, mid_y0 + mid_h * 0.52),
            arrowstyle="->",
            mutation_scale=18,
            lw=2,
            color=PALETTE["accent"],
        )
    )
    ax.text(
        0.5,
        mid_y0 + mid_h * 0.58,
        "Matched heads",
        ha="center",
        va="bottom",
        fontsize=9,
        color=PALETTE["text"],
    )

    # Bottom validation boxes
    bot_y0 = 0.06
    bot_h = 0.16
    gap = 0.03
    box_w = (0.9 - 2 * gap) / 3.0
    boxes = [0.05, 0.05 + box_w + gap, 0.05 + 2 * (box_w + gap)]

    draw_bottom_box(ax, boxes[0], bot_y0, box_w, bot_h, "Head agreement")
    draw_head_agreement_icon(ax, boxes[0], bot_y0, box_w, bot_h)

    draw_bottom_box(ax, boxes[1], bot_y0, box_w, bot_h, "Residual consistency")
    draw_residual_icon(ax, boxes[1], bot_y0, box_w, bot_h)

    draw_bottom_box(ax, boxes[2], bot_y0, box_w, bot_h, "Mass balance closure")
    draw_mass_balance_icon(ax, boxes[2], bot_y0, box_w, bot_h)


def make_graphical_abstract(
    out_dir: Path,
    basename: str,
    width: float,
    height: float,
    dpi: int,
    formats: list[str],
    layout: str,
) -> None:
    fig, ax = plt.subplots(figsize=(width, height), dpi=dpi)
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_aspect("auto")
    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0)

    layout = layout.lower().strip()
    if layout == "a":
        draw_layout_a(ax)
    else:
        draw_layout_b(ax)

    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in formats:
        out_path = out_dir / f"{basename}.{ext}"
        fig.savefig(out_path, dpi=dpi, facecolor=PALETTE["bg"])

    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a graphical abstract for the DarcyWarp paper (layout A or B)."
    )
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR, help="Output directory.")
    parser.add_argument("--basename", default="graphical_abstract", help="Base filename for outputs.")
    parser.add_argument("--width", type=float, default=13.0, help="Figure width in inches.")
    parser.add_argument("--height", type=float, default=5.0, help="Figure height in inches.")
    parser.add_argument("--dpi", type=int, default=200, help="Output DPI.")
    parser.add_argument(
        "--layout",
        default="a",
        choices=("a", "b"),
        help="Layout style: a (pipeline) or b (comparison panels).",
    )
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated list of output formats (e.g., png,pdf,svg).",
    )
    args = parser.parse_args()

    fmt_list = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    make_graphical_abstract(
        out_dir=args.out_dir,
        basename=args.basename,
        width=args.width,
        height=args.height,
        dpi=args.dpi,
        formats=fmt_list,
        layout=args.layout,
    )
    for ext in fmt_list:
        print(f"Wrote: {Path(args.out_dir) / f'{args.basename}.{ext}'}")
