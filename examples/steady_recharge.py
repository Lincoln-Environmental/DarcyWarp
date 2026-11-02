"""Solve a confined aquifer with uniform recharge and check its analytical head.

Run from the repository root with ``python -m examples.steady_recharge``.
All inputs use metres and seconds. No external model data or MODFLOW executable
is needed. Change the explicit parameters in the final ``main`` call to explore
the example; the default device is the CPU.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


def save_figure(
    output_dir: Path,
    x_m: np.ndarray,
    y_m: np.ndarray,
    head_m: np.ndarray,
    analytical_head_m: np.ndarray,
    dx_m: float,
) -> None:
    """Save a head map and an independently calculated analytical profile."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matplotlib.rcParams["svg.hashsalt"] = "darcywarp-steady-recharge"
    navy = "#123a4a"
    teal = "#008c99"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    fig.set_facecolor("white")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(colors=navy)
        axis.xaxis.label.set_color(navy)
        axis.yaxis.label.set_color(navy)

    extent = (
        x_m[0] - 0.5 * dx_m,
        x_m[-1] + 0.5 * dx_m,
        y_m[0] - 0.5 * dx_m,
        y_m[-1] + 0.5 * dx_m,
    )
    image = axes[0].imshow(
        head_m,
        origin="lower",
        extent=extent,
        cmap="viridis",
        interpolation="nearest",
        aspect="equal",
    )
    axes[0].set_title("Hydraulic head", color=navy, fontweight="bold")
    axes[0].set_xlabel("Distance east from western fixed-head cells (m)")
    axes[0].set_ylabel("Distance north (m)")
    colourbar = fig.colorbar(image, ax=axes[0], orientation="horizontal", pad=0.18)
    colourbar.set_label("Head (m)", color=navy)

    centre_row = head_m.shape[0] // 2
    axes[1].plot(
        x_m,
        analytical_head_m[centre_row],
        color=navy,
        linewidth=2.0,
        label="Analytical solution",
    )
    axes[1].plot(
        x_m[::2],
        head_m[centre_row, ::2],
        linestyle="none",
        marker="o",
        markersize=4.5,
        markerfacecolor="white",
        markeredgecolor=teal,
        markeredgewidth=1.4,
        label="DarcyWarp",
    )
    axes[1].set_title("Centreline comparison", color=navy, fontweight="bold")
    axes[1].set_xlabel("Distance between fixed-head cell centres (m)")
    axes[1].set_ylabel("Head (m)")
    axes[1].grid(alpha=0.18)
    axes[1].legend(frameon=False, loc="lower center")
    fig.suptitle(
        "DarcyWarp | Steady recharge in a confined aquifer",
        fontsize=16,
        color=navy,
        fontweight="bold",
    )
    fig.savefig(output_dir / "steady_recharge.png", dpi=180, facecolor="white")
    fig.savefig(output_dir / "steady_recharge.svg", facecolor="white", metadata={"Date": None})
    plt.close(fig)


def main(
    *,
    nx: int = 41,
    ny: int = 17,
    dx_m: float = 25.0,
    transmissivity_m2_s: float = 1.0e-3,
    recharge_m_s: float = 1.0e-8,
    boundary_head_m: float = 10.0,
    device: str = "cpu",
    output_dir: str | Path = "outputs/steady_recharge",
    make_plot: bool = True,
) -> dict[str, float | int | bool | str]:
    """Run a uniform confined aquifer and save its inputs and checked solution.

    Fixed heads apply at the centres of the west/east boundary cells. All
    north/south outer faces are impermeable. Recharge enters the interior
    (non-fixed-head) cells, whose balance equations are solved.
    """
    if nx < 3 or ny < 2:
        raise ValueError("Use at least 3 columns and 2 rows.")
    if dx_m <= 0.0 or transmissivity_m2_s <= 0.0 or recharge_m_s <= 0.0:
        raise ValueError("Cell size, transmissivity and recharge must be positive.")

    # Precision is selected when the solver modules are imported. This example
    # deliberately uses FP64 so the analytical check has a fixed tolerance.
    os.environ.setdefault("DARCY_FLOAT", "float64")
    if os.environ["DARCY_FLOAT"] != "float64":
        raise ValueError("Run this example in a fresh process with DARCY_FLOAT=float64.")

    from DARCY_WARP_PACKAGE import create_solver

    shape = (ny, nx)
    transmissivity = np.full(shape, transmissivity_m2_s, dtype=np.float64)
    recharge = np.full(shape, recharge_m_s, dtype=np.float64)
    active = np.ones(shape, dtype=np.int32)
    boundary_mask = np.zeros(shape, dtype=np.int32)
    boundary_mask[:, 0] = 1
    boundary_mask[:, -1] = 1
    boundary_heads = np.full(shape, boundary_head_m, dtype=np.float64)
    recharge[boundary_mask != 0] = 0.0

    model = create_solver(
        dim=2,
        solver="confined_pcg",
        nx=nx,
        ny=ny,
        dx=dx_m,
        device=device,
    )
    try:
        model.build_from_fields(
            T_field=transmissivity,
            R_field=recharge,
            active=active,
            bc_mask=boundary_mask,
            bc_values=boundary_heads,
        )
        head, info = model.solve(
            initial_head=boundary_heads,
            max_iter=1000,
            rel_tol=1.0e-10,
            abs_tol_min=1.0e-12,
        )
        head = head.copy()
    finally:
        model.close()
    if not info["converged"] or not np.all(np.isfinite(head)):
        raise RuntimeError(f"The example did not converge to finite heads: {info}")

    x_m = np.arange(nx, dtype=np.float64) * dx_m
    y_m = np.arange(ny, dtype=np.float64) * dx_m
    length_m = x_m[-1]
    analytical_profile = boundary_head_m + (
        recharge_m_s * x_m * (length_m - x_m) / (2.0 * transmissivity_m2_s)
    )
    analytical_head = np.broadcast_to(analytical_profile, shape).copy()
    max_error_m = float(np.max(np.abs(head - analytical_head)))
    np.testing.assert_allclose(head, analytical_head, rtol=0.0, atol=1.0e-5)

    recharge_in_m3_s = float(np.sum(recharge) * dx_m**2)
    boundary_out_m3_s = float(
        transmissivity_m2_s
        * np.sum(head[:, 1] - head[:, 0] + head[:, -2] - head[:, -1])
    )
    relative_imbalance = (recharge_in_m3_s - boundary_out_m3_s) / recharge_in_m3_s
    if abs(relative_imbalance) > 1.0e-5:
        raise RuntimeError(f"Relative water-balance error is {relative_imbalance:.3e}.")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination / "steady_recharge.npz",
        head_m=head,
        analytical_head_m=analytical_head,
        x_m=x_m,
        y_m=y_m,
        dx_m=dx_m,
        transmissivity_m2_s=transmissivity,
        recharge_m_s=recharge,
        active=active,
        boundary_mask=boundary_mask,
        boundary_head_m=boundary_heads,
        max_error_m=max_error_m,
        recharge_in_m3_s=recharge_in_m3_s,
        boundary_out_m3_s=boundary_out_m3_s,
        relative_imbalance=relative_imbalance,
        solver_backend=info["solver_backend"],
        device=device,
    )
    if make_plot:
        save_figure(destination, x_m, y_m, head, analytical_head, dx_m)

    summary = {
        "converged": bool(info["converged"]),
        "iterations": int(info["n_iter_used"]),
        "max_head_m": float(np.max(head)),
        "max_error_m": max_error_m,
        "relative_imbalance": relative_imbalance,
        "device": device,
        "output_dir": str(destination),
    }
    print(f"Converged on {device}: {summary['iterations']} iterations.")
    print(f"Maximum head: {summary['max_head_m']:.6f} m.")
    print(f"Maximum analytical head error: {max_error_m:.3e} m.")
    print(f"Relative water-balance error: {relative_imbalance:.3e}.")
    print(f"Results: {destination.resolve()}")
    return summary


if __name__ == "__main__":
    main(device="cpu", output_dir="outputs/steady_recharge", make_plot=True)
