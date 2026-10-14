"""3D terrain convergence-baseline ladder with stagnation diagnostics.

Runs the representative SHANZ terrain case at increasing resolutions with
the host or device Picard/GMRES solver and records, per outer
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
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from shanz_terrain_case_study import build_case, parallel_interfaces
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def run_size(*, case, interfaces, floor, device, max_cycles, max_outer,
             solve_controls, ky_ratio, kz_ratio, warm_start=None, device_picard=False):
    ny, nx = case["surface"].shape
    solver = WarpDarcySolver3D(nx=nx, ny=ny, nz=case["layers"], dx=case["dx"], dy=case["dx"],
        dz=float(np.mean(interfaces[:-1]-interfaces[1:])),
        device=device, implementation="fast")
    initial = case["initial"] if warm_start is None else np.asarray(warm_start)
    if initial.shape != case["initial"].shape:
        raise ValueError("warm_start must match the case grid")
    solver.build_from_terrain(z_interfaces=interfaces, active_columns=case["active"],
        kx_field=case["k"], ky_field=ky_ratio*case["k"], kz_field=kz_ratio*case["k"],
        bc_mask=case["fixed"], bc_values=case["values"], rhs=case["source"],
        initial_head=initial, contacts=case["contacts"])
    start = perf_counter()
    with solver:
        head, info = solver.solve(unconfined=True, unconfined_min_sat=floor,
            unconfined_max_picard_iter=max_outer, device_picard=device_picard,
            max_cycles=max_cycles, **solve_controls)
    elapsed = perf_counter()-start
    history = []
    for index, row in enumerate(info["outer_history"]):
        entry = row.copy()
        if (entry.get("acceptance_mode") == "merit_decrease"
                and not entry.get("inner_retried", False)
                and index+1 < len(info["outer_history"])):
            entry.pop("backtracking_trials", None)
        history.append(entry)
    budget = info["budget"]
    dry = info.get("dry_cell_approximation", {})
    return head, dict(
        converged=bool(info["converged"]),
        stop_reason=info.get("device_stop_reason"),
        active_cells=int(np.count_nonzero(case["active"]))*case["layers"],
        free_cells=int(np.count_nonzero(np.broadcast_to(case["active"], case["fixed"].shape)
            & (case["fixed"] == 0))),
        basement_k=float(np.min(case["k"][-1])),
        outer_iterations=int(info["n_outer_iterations"]),
        krylov_iterations=int(info["krylov_iterations"]),
        krylov_per_outer=float(info["krylov_iterations"])/max(info["n_outer_iterations"],1),
        seconds=elapsed,
        final_head_residual=float(budget["head_residual_rms"]),
        device_host_head_residual_difference=(history[-1]["head_residual_rms"]
            -budget["head_residual_rms"]) if device_picard and history else None,
        device_host_flow_residual_difference=(history[-1]["physical_residual_rms"]
            -budget["cell_residual_rms"]) if device_picard and history else None,
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


def write_results(*, results, output_path):
    """Keep reviewable summaries and compressed, complete outer histories."""
    archive_path = output_path.with_suffix(".history.json.gz")
    histories = {}
    summary = {key: value for key, value in results.items() if key != "sizes"}
    summary["sizes"] = {}
    summary["history_archive"] = archive_path.name
    for size, row in results["sizes"].items():
        histories[size] = row["history"]
        summary["sizes"][size] = {key: value for key, value in row.items() if key != "history"}
    archive_path.write_bytes(gzip.compress(json.dumps(histories, default=float).encode(), mtime=0))
    output_path.write_text(json.dumps(summary, indent=2, default=float))


def run_ladder(*, args, solve_controls, ky_ratio, kz_ratio):
    results = dict(device=args.device, device_picard=args.device_picard,
        floor=args.floor, variant=args.variant, max_cycles=args.max_cycles,
        max_outer=args.max_outer, solve_controls=solve_controls, ky_ratio=ky_ratio, kz_ratio=kz_ratio,
        purpose="Scientific convergence verification; no GPU performance conclusions", sizes={})
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    for n in args.sizes:
        case = build_case(nx=n, ny=n, dx=args.dx)
        if args.basement_k is not None:
            case["k"][-1] = args.basement_k
        interfaces = case["interfaces"] if args.variant == "variable" else parallel_interfaces(case=case)
        print(f"running {args.variant} {n}x{n} ...", flush=True)
        unused_head, summary = run_size(case=case, interfaces=np.asarray(interfaces),
            floor=args.floor, device=args.device, max_cycles=args.max_cycles,
            max_outer=args.max_outer, device_picard=args.device_picard,
            solve_controls=solve_controls, ky_ratio=ky_ratio, kz_ratio=kz_ratio)
        results["sizes"][n] = summary
        print(json.dumps({k: v for k, v in summary.items() if k != "history"}, default=float), flush=True)
        for entry in summary["history"][-3:]:
            print("tail:", json.dumps(entry, default=float), flush=True)
        write_results(results=results, output_path=out)
    print(f"wrote {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--dx", type=float, default=20.)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-cycles", type=int, default=2000)
    parser.add_argument("--max-outer", type=int, default=400)
    parser.add_argument("--floor", type=float, default=.01)
    parser.add_argument("--basement-k", type=float, default=None)
    parser.add_argument("--variant", choices=("variable", "parallel"), default="variable")
    parser.add_argument("--device-picard", action="store_true")
    parser.add_argument("--out", default="docs/terrain_convergence_baseline.json")
    args = parser.parse_args()
    solve_controls = dict(unconfined_head_tol=1e-6, final_head_residual_tol=1e-6,
        rel_tol=1e-12, abs_tol_min=1e-12, final_budget_relative_tol=1e-8,
        unconfined_relax=.7, krylov_restart=24, max_levels=6, min_coarse_n=16)
    ky_ratio = .6
    kz_ratio = .15
    run_ladder(args=args, solve_controls=solve_controls, ky_ratio=ky_ratio, kz_ratio=kz_ratio)
