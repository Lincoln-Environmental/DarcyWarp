# SPDX-License-Identifier: AGPL-3.0-or-later
"""Verify vertical flow through a heterogeneous, confined 3D aquifer.

Run from the repository root: python -m examples.layered_confined_3d
Units are metres and days. This synthetic Cartesian model requires no GIS,
MODFLOW, external data or GPU. The exact solution is a series-resistance
calculation between prescribed-head CELL CENTRES, not the outer faces.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def main(*, device: str = "cpu", output_dir: str | Path = "outputs/layered_confined_3d") -> dict:
    os.environ.setdefault("DARCY_FLOAT", "float64")
    if os.environ["DARCY_FLOAT"] != "float64":
        raise ValueError("Run in a fresh process with DARCY_FLOAT=float64.")
    from DARCY_WARP_PACKAGE import create_solver
    from DARCY_WARP_PACKAGE.config import NP_FLOAT
    from DARCY_WARP_PACKAGE.physics.budget_3d import compute_3d_flow_budget

    if np.dtype(NP_FLOAT) != np.dtype(np.float64):
        raise ValueError("DarcyWarp was already imported in another precision; start a fresh process.")
    nx, ny, nz = 5, 4, 9
    dx, dy, dz = 25.0, 20.0, 2.0
    top_head, bottom_head = 10.0, 2.0
    vertical_k = np.array([8., 8., 2., 2., 0.2, 0.2, 2., 8., 8.])
    shape = (nz, ny, nx)
    conductivity = np.broadcast_to(vertical_k[:, None, None], shape).copy()
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[0] = fixed[-1] = 1
    values = np.zeros(shape)
    values[0], values[-1] = top_head, bottom_head
    source = np.zeros(shape)

    # Derive the exact profile independently from half-cell resistances.
    area = dx * dy
    resistance = (0.5 * dz / vertical_k[:-1] + 0.5 * dz / vertical_k[1:]) / area
    exact_column_flux = (top_head - bottom_head) / float(resistance.sum())
    exact_profile = top_head - exact_column_flux * np.concatenate(([0.], np.cumsum(resistance)))
    exact_head = np.broadcast_to(exact_profile[:, None, None], shape).copy()
    with create_solver(dim=3, solver="kcycle", nx=nx, ny=ny, nz=nz,
                       dx=dx, dy=dy, dz=dz, device=device, implementation="fast") as model:
        model.build_from_K_fields(kx_field=conductivity, ky_field=conductivity,
            kz_field=conductivity, active=active, bc_mask=fixed, bc_values=values,
            rhs=source, initial_head=np.full(shape, bottom_head))
        head, info = model.solve(max_cycles=2000, max_levels=1, check_every_no=1,
            rel_tol=1e-12, abs_tol_min=1e-10, dh_rms_tol=1e-10, final_budget_relative_tol=1e-10)
        budget = compute_3d_flow_budget(head=head, source=source, faces=model._session.faces,
            active=active, bc_mask=fixed)
    if not info["converged"]:
        raise RuntimeError(f"The synthetic example did not converge: {info}")
    np.testing.assert_allclose(head, exact_head, rtol=0.0, atol=1e-7)
    # Independently reconstruct all vertical face fluxes from resistance.
    fluxes = (head[:-1] - head[1:]) / resistance[:, None, None]
    np.testing.assert_allclose(fluxes, exact_column_flux, rtol=1e-8, atol=1e-8)
    if budget["relative_cell_imbalance"] > 1e-8:
        raise RuntimeError(f"Unacceptable water balance: {budget}")
    summary = dict(synthetic=True, formulation="legacy_cartesian_confined",
        units="metres, days", device=device, shape=list(shape), converged=bool(info["converged"]),
        cycles=int(info["n_cycles_used"]), max_head_error_m=float(np.max(np.abs(head-exact_head))),
        max_face_flux_error_m3_day=float(np.max(np.abs(fluxes-exact_column_flux))),
        exact_total_discharge_m3_day=float(exact_column_flux * nx * ny),
        relative_cell_imbalance=float(budget["relative_cell_imbalance"]))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "layered_confined_3d.npz", head_m=head,
        analytical_head_m=exact_head, vertical_flux_m3_day=fluxes, conductivity_m_day=conductivity,
        active=active, fixed=fixed, boundary_head_m=values, dx_m=dx, dy_m=dy, dz_m=dz)
    (destination / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(device="cpu", output_dir="outputs/layered_confined_3d")
