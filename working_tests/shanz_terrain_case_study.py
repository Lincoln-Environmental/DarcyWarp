"""Representative SHANZ-style shallow-aquifer/basement terrain case study.

A standalone synthetic Purau/Orton Bradley-style groundwater model: a steep
DEM with large relief over a shallow geological domain — a permeable
shallow aquifer a few metres below ground with realistic lateral thickness
variation over an independently varying weathered zone and low-permeability
basalt basement. The water table starts near the land surface, recharge
varies spatially, and the coast drains the domain through a fixed
sea-level head in the aquifer plus gated discharge-only springs on the
topsoil coastal subface.

The case compares the previous parallel terrain-following layers (domain-
mean thicknesses) against the new independently varying geological
interfaces, each at the 0.01 m (default) and 0.001 m saturation floors,
reporting groundwater heads, water-table depth, lateral flow, aquifer
leakage to the underlying units, coastal discharge, surface
groundwater-discharge potential, and convergence and performance. This is
a standalone verification task; no SHANZ coupling is modified.

Usage:
    python working_tests/shanz_terrain_case_study.py [--nx 96] [--ny 96] \
        [--out docs/shanz_case_study_results.json]
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def build_case(*, nx, ny, dx):
    """Steep coastal hillslope with independently varying geology [m]."""
    y, x = np.mgrid[:ny, :nx]
    xn, yn = x/(nx-1), y/(ny-1)
    # Large-relief DEM: ridges draining to a coastal strip at x near 0.
    surface = 6.+130.*(xn**.8)*(1+.25*np.sin(3*np.pi*yn)*np.sin(2*np.pi*xn))
    interfaces = np.stack((
        surface,
        surface-(.3+.2*np.sin(2*np.pi*xn)*np.cos(4*np.pi*yn)+.15*xn),            # topsoil 0.1..0.5 m
        surface-(5.+1.5*np.sin(2*np.pi*xn+1.3)+1.2*np.cos(3*np.pi*yn)+1.5*xn),   # aquifer base
        surface-(14.+2.*np.sin(1.7*np.pi*xn+.4)+1.5*np.cos(2.3*np.pi*yn+.9)),    # weathered base
        surface-(40.+22.*xn+8.*np.sin(np.pi*xn)*np.cos(np.pi*yn)),               # basalt basement
    ))
    assert np.all(interfaces[:-1]-interfaces[1:] > 0), "geological interfaces must not intersect"
    sea_level = 2.                                     # total hydraulic head [m]
    layers = len(interfaces)-1
    active = np.ones((ny, nx), dtype=bool)
    coastal = xn < 1./(nx-1)
    fixed = np.zeros((layers, ny, nx), dtype=np.int32)
    fixed[1:] = np.broadcast_to(coastal, (ny, nx)).astype(np.int32)   # sea in aquifer and below
    values = np.full((layers, ny, nx), float(sea_level))
    shape = (layers, ny, nx)
    initial = np.broadcast_to(surface-1., shape).copy()
    recharge = .0012*(1+.6*np.sin(2*np.pi*xn)*np.sin(3*np.pi*yn))     # [m/day]
    source = np.zeros(shape)
    source[0] = recharge*dx*dx                          # recharge uses map area
    k = np.ones(shape)
    k[0] = 2.                                           # topsoil
    k[1] = 8.                                           # permeable shallow aquifer
    k[2] = .5                                           # weathered material
    k[3] = .1                                           # basalt basement [m/day]
    sy = np.broadcast_to(np.array([.15, .25, .08, .05])[:layers, None, None], shape).copy()
    ss = np.full(shape, 1e-5)
    spring_j = (ny//4, ny//2, 3*ny//4)
    def spring_elevation(j):
        # Mid-layer at the physical subface point. The face sits half a
        # cell further downslope: the one-sided face slope extrapolates
        # the interface to 2*z0 - z1 there, exactly as the operator does.
        top_point = 1.5*float(interfaces[0][j, 0])-.5*float(interfaces[0][j, 1])
        bottom_point = 1.5*float(interfaces[1][j, 0])-.5*float(interfaces[1][j, 1])
        return .5*(top_point+bottom_point)
    contacts = tuple(
        TerrainContact3D(cell=(0, j, 0), direction="west",
                         area=.25*dx*float(interfaces[0][j, 0]-interfaces[1][j, 0]),
                         distance=dx/2, elevation=spring_elevation(j),
                         head=float(sea_level), kind="discharge_only")
        for j in spring_j)
    return dict(surface=surface, interfaces=interfaces, active=active, fixed=fixed,
                values=values, initial=initial, source=source, k=k, sy=sy, ss=ss,
                contacts=contacts, dx=dx, sea_level=sea_level, layers=layers)


def parallel_interfaces(case):
    """The previous approximation: parallel layers of domain-mean thickness."""
    thicknesses = (case["interfaces"][:-1]-case["interfaces"][1:]).mean(axis=(1, 2))
    zeta = np.concatenate(([0.], np.cumsum(thicknesses)))
    return case["surface"][None]-zeta[:, None, None]


def run_variant(case, *, interfaces, floor, device, max_cycles, warm_start=None):
    ny, nx = case["surface"].shape
    solver = WarpDarcySolver3D(nx=nx, ny=ny, nz=case["layers"], dx=case["dx"], dy=case["dx"],
        dz=5., device=device, implementation="fast")
    initial = case["initial"] if warm_start is None else np.asarray(warm_start)
    solver.build_from_terrain(z_interfaces=interfaces, active_columns=case["active"],
        kx_field=case["k"], ky_field=.6*case["k"], kz_field=.15*case["k"],
        bc_mask=case["fixed"], bc_values=case["values"], rhs=case["source"],
        initial_head=initial, contacts=case["contacts"])
    start = perf_counter()
    with solver:
        head, info = solver.solve(unconfined=True, unconfined_min_sat=floor,
            unconfined_head_tol=1e-6, final_head_residual_tol=1e-6, rel_tol=1e-12,
            final_budget_relative_tol=1e-6, unconfined_max_picard_iter=800,
            max_cycles=max_cycles, krylov_restart=24, max_levels=6, min_coarse_n=16)
    elapsed = perf_counter()-start
    from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
    from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainMetrics3D
    grid = TerrainGrid3D(z_interfaces=interfaces, dx=case["dx"], dy=case["dx"],
        active_columns=case["active"])
    operator = TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=grid),
        kx=case["k"], ky=.6*case["k"], kz=.15*case["k"], head=head, min_sat=floor,
        contacts=case["contacts"])
    qx, qy, qz, boundary = operator.fluxes(head=head)
    surface = case["surface"]
    water_table = head.max(axis=0)
    free = case["fixed"] == 0
    diagnostics = info.get("dry_cell_approximation", {})
    return head, dict(
        converged=bool(info["converged"]),
        outer_iterations=int(info["n_outer_iterations"]),
        krylov_iterations=int(info["krylov_iterations"]),
        seconds=elapsed,
        operator_refresh_seconds=info.get("operator_refresh_seconds"),
        head_min=float(head.min()), head_max=float(head.max()),
        water_table_depth_mean=float(np.mean(surface-water_table)),
        water_table_depth_max=float(np.max(surface-water_table)),
        lateral_flow_abs_total=float(np.abs(qx).sum()+np.abs(qy).sum()),
        aquifer_leakage_downward=float(qz[1].sum()),
        weathered_to_basement_leakage=float(qz[2].sum()),
        coastal_discharge_total=float(boundary.sum()),
        coastal_subface_outflows=[float(v) for v in boundary],
        surface_discharge_potential_cells=int(np.count_nonzero((head > surface+1e-6).any(axis=0))),
        cell_residual_rms=info["budget"]["cell_residual_rms"],
        relative_cell_imbalance=info["budget"]["relative_cell_imbalance"],
        horizontal_floor_excess_flow=diagnostics.get("horizontal_floor_excess_flow"),
        floor_excess_to_recharge=diagnostics.get("horizontal_floor_to_recharge"),
        shape=[int(v) for v in head.shape],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=96)
    parser.add_argument("--dx", type=float, default=20.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cycles", type=int, default=8000)
    parser.add_argument("--out", default="docs/shanz_case_study_results.json")
    args = parser.parse_args()
    case = build_case(nx=args.nx, ny=args.ny, dx=args.dx)
    results = dict(nx=args.nx, ny=args.ny, dx=args.dx, device=args.device,
        layers=case["layers"],
        relief_m=float(case["surface"].max()-case["surface"].min()),
        shallow_domain_depth_m=float((case["surface"]-case["interfaces"][-1]).mean()),
        variants={})
    heads = {}
    warm = None
    # The variable-geometry case converges cleanest, so it seeds the others.
    for label, interfaces, floor in (
        ("variable_floor_0.010", case["interfaces"], .01),
        ("variable_floor_0.001", case["interfaces"], .001),
        ("parallel_floor_0.010", parallel_interfaces(case), .01),
        ("parallel_floor_0.001", parallel_interfaces(case), .001),
    ):
        print(f"running {label} ...", flush=True)
        head, summary = run_variant(case, interfaces=np.asarray(interfaces), floor=floor,
            device=args.device, max_cycles=args.max_cycles, warm_start=warm)
        heads[label] = head
        if summary["converged"] or warm is None:
            warm = head
        results["variants"][label] = summary
        print(" ", {key: summary[key] for key in ("converged", "outer_iterations",
              "krylov_iterations", "seconds")}, flush=True)
    variable = heads["variable_floor_0.010"]
    for other in ("parallel_floor_0.010", "variable_floor_0.001"):
        difference = heads[other]-variable
        results["variants"][other]["head_rmse_vs_variable_0.010"] = float(
            np.sqrt(np.mean(difference**2)))
        results["variants"][other]["head_max_abs_diff_vs_variable_0.010"] = float(
            np.abs(difference).max())
        results["variants"][other]["water_table_depth_rmse_vs_variable_0.010"] = float(
            np.sqrt(np.mean(((case["surface"]-heads[other].max(axis=0))
                             -(case["surface"]-variable.max(axis=0)))**2)))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
