"""3D terrain convergence-baseline ladder with stagnation diagnostics.

Runs the representative SHANZ terrain case at increasing resolutions with
the CURRENT host-driven Picard/GMRES solver and records, per outer
iteration and per case: Krylov work, inner-target behaviour, accepted
head changes, physical residual, independent cellwise imbalance, dry-floor
excess transmission, damping and rollback counts. Establishes the
reproducible convergence baseline that the GPU-resident Picard port must
match or beat, and isolates where the nonlinear contraction stalls.

Usage:
    python working_tests/terrain_convergence_baseline.py --sizes 16 32 64 \
        --out docs/terrain_convergence_baseline.json
"""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from shanz_terrain_case_study import build_case, parallel_interfaces
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def run_size(case, *, interfaces, floor, device, max_cycles, max_outer, warm_start=None,
             unconfined_relax=0.7):
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
            final_budget_relative_tol=1e-8, unconfined_max_picard_iter=max_outer,
            unconfined_relax=unconfined_relax,
            max_cycles=max_cycles, krylov_restart=24, max_levels=6, min_coarse_n=16)
    elapsed = perf_counter()-start
    history = []
    for entry in info["outer_history"]:
        history.append(dict(outer=entry["outer_iteration"],
            krylov=entry["krylov_iterations"],
            proposed=float(entry["proposed_head_change_max"]),
            accepted=float(entry["accepted_head_change_max"]),
            head_residual=float(entry["head_residual_rms"]),
            cell_residual=float(entry["physical_residual_rms"]),
            relaxation=float(entry["relaxation"])))
    budget = info["budget"]
    dry = info.get("dry_cell_approximation", {})
    from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
    from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D
    grid = TerrainGrid3D(z_interfaces=interfaces, dx=case["dx"], dy=case["dx"],
        active_columns=case["active"])
    operator = TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=grid),
        kx=case["k"], ky=.6*case["k"], kz=.15*case["k"], head=head, min_sat=floor,
        contacts=case["contacts"])
    fluxes = operator.fluxes(head=head)
    return head, dict(
        converged=bool(info["converged"]),
        outer_iterations=int(info["n_outer_iterations"]),
        krylov_iterations=int(info["krylov_iterations"]),
        krylov_per_outer=float(info["krylov_iterations"])/max(info["n_outer_iterations"],1),
        seconds=elapsed,
        device_seconds=elapsed-info.get("operator_refresh_seconds", 0.),
        host_refresh_seconds=info.get("operator_refresh_seconds"),
        final_head_residual=float(budget["head_residual_rms"]),
        final_cell_residual=float(budget["cell_residual_rms"]),
        relative_cell_imbalance=float(budget["relative_cell_imbalance"]),
        domain_imbalance=float(budget["imbalance"]),
        throughput=float(budget["throughput"]),
        zero_capacity_seeds=int(info.get("n_zero_capacity_storage_seeds", 0)),
        discharge_seeds=int(info.get("n_discharge_component_seeds", 0)),
        operator_refreshes=int(info.get("operator_refreshes", 0)),
        floor_excess_horizontal=dry.get("horizontal_floor_excess_flow"),
        floor_excess_to_throughput=dry.get("horizontal_floor_to_budget_throughput"),
        dry_cells=dry.get("n_dry_cells"),
        history=history,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--dx", type=float, default=20.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--max-outer", type=int, default=400)
    parser.add_argument("--floor", type=float, default=.01)
    parser.add_argument("--variant", choices=("variable", "parallel"), default="variable")
    parser.add_argument("--out", default="docs/terrain_convergence_baseline.json")
    args = parser.parse_args()
    results = dict(device=args.device, floor=args.floor, variant=args.variant, sizes={})
    warm = None
    for n in args.sizes:
        case = build_case(nx=n, ny=n, dx=args.dx)
        interfaces = case["interfaces"] if args.variant == "variable" else parallel_interfaces(case)
        print(f"running {args.variant} {n}x{n} ...", flush=True)
        head, summary = run_size(case, interfaces=np.asarray(interfaces), floor=args.floor,
            device=args.device, max_cycles=args.max_cycles, max_outer=args.max_outer,
            warm_start=warm)
        if summary["converged"]:
            warm = head
        results["sizes"][n] = summary
        tail = summary["history"][-3:]
        print(" ", json.dumps({k: v for k, v in summary.items() if k != "history"}, default=float)[:260], flush=True)
        for entry in tail:
            print("   tail:", json.dumps(entry, default=float), flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
