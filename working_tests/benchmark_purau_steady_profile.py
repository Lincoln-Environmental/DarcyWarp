"""Bounded real-Purau test of the named native steady profile (no storm runs).

Reads saved prepared geometry/forcing arrays, never treats a saved trial head
as an accepted checkpoint. Numerical and physical validity are separate gates.
"""
import argparse
from dataclasses import asdict
import hashlib
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

import numpy as np


def benchmark_profile(*, run_directory: Path, output_directory: Path,
                      shanz_repository: Path, levels: list[int], repetitions: int,
                      device: str) -> None:
    from DARCY_WARP_PACKAGE.solvers.steady_config import production_unconfined_steady_controls
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    from shanz.initialization.darcy_hot_start import independent_steady_budget, require_fast_kcycle
    from benchmark_purau_darcy_kcycle import load_frozen_inputs, gpu_snapshot
    from run_purau_darcy_hot_start import write_report
    import warp as wp
    if output_directory.exists():
        raise FileExistsError(output_directory)
    if repetitions < 1:
        raise ValueError('repetitions must be positive')
    output_directory.mkdir(parents=True)
    started = time.perf_counter()
    arrays, config, dx, coast, recharge = load_frozen_inputs(run_directory=run_directory)
    input_wall = time.perf_counter()-started
    active = arrays['active']
    shape = active.shape
    zeros = np.zeros(shape)
    fields = dict(active=active, dem=arrays['dem_m'], bottom_m=arrays['darcy_bottom_m'],
        network=arrays['drainage_mask'], conductance=arrays['drain_conductance_m2_day'],
        coastal=coast, grid=SimpleNamespace(dx=dx, dy=dx))
    records = []
    first_head = None
    for cap in levels:
        controls = production_unconfined_steady_controls(max_levels=cap)
        model = WarpDarcySolver(nx=shape[1], ny=shape[0], dx=dx, device=device,
            use_drn=True, diag_preconditioner_backend='device')
        try:
            started = time.perf_counter()
            model.build_from_fields(T_field=arrays['reference_transmissivity_m2_day'],
                R_field=recharge, active=active.astype(int), bc_mask=zeros.astype(int), bc_values=zeros,
                drn_mask=arrays['drainage_mask'].astype(int), drn_elev=arrays['dem_m'],
                drn_cond=arrays['drain_conductance_m2_day'], coastal_boundary=coast)
            wp.synchronize_device(device)
            build_wall = time.perf_counter()-started
            for repeat in range(repetitions):
                wp.synchronize_device(device)
                started = time.perf_counter()
                head, info = model.solve(initial_head=arrays['dem_m'].copy(),
                    reference_transmissivity_field=arrays['reference_transmissivity_m2_day'],
                    zbot_field=arrays['darcy_bottom_m'], ztop_field=arrays['dem_m'],
                    newton_profile_timing=True, **controls)
                wp.synchronize_device(device)
                wall = time.perf_counter()-started
                require_fast_kcycle(info=info)
                assert info['solver_backend'] == controls['solver']
                budget = independent_steady_budget(fields=fields, head=head, config=config)
                verified = bool(info['converged'] and budget['budget_passed'] and budget['physically_valid'])
                if first_head is None and info['converged']:
                    first_head = head.copy()
                filename = output_directory/f'levels_{cap}_repeat_{repeat+1}_diagnostic_head.npz'
                started = time.perf_counter()
                np.savez_compressed(filename, head_m=head, active=active)
                export_wall = time.perf_counter()-started
                record = dict(max_levels=cap, repeat=repeat+1, solve_wall_s=wall,
                    build_wall_s=build_wall, export_wall_s=export_wall,
                    solver_controls=controls, native=info, groundwater_budget=budget,
                    scientifically_accepted=verified, diagnostic_head_path=str(filename.resolve()),
                    head_difference_from_first_m=None if first_head is None else float(np.max(np.abs(head-first_head)[active])),
                    gpu_snapshot=gpu_snapshot())
                records.append(record)
                write_report(path=output_directory/f'levels_{cap}_repeat_{repeat+1}.json', value=record)
                print(f"cap={cap} repeat={repeat+1}: {wall:.3f}s Newton={info['newton_iterations']} "
                    f"FGMRES={info['fgmres_iterations']} native={info['converged']} "
                    f"budget={budget['budget_passed']} physical={budget['physically_valid']}", flush=True)
        finally:
            model.close()
    write_report(path=output_directory/'report.json', value=dict(
        profile='production_unconfined_steady_controls', configuration=asdict(config),
        shanz_repository=str(shanz_repository), input_loading_wall_s=input_wall,
        source_archive=str((run_directory/'steady_state.npz').resolve()),
        source_archive_sha256=hashlib.sha256((run_directory/'steady_state.npz').read_bytes()).hexdigest(),
        initial_iterate='DEM on every repeat; no hot-head speedup comparison', runs=records))


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    shanz_repository = Path('/home/patrickdurney/PycharmProjects/SHANZ')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-directory', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--shanz-repository', type=Path, default=shanz_repository)
    parser.add_argument('--max-levels', type=int, nargs='+', choices=(5, 6, 7), default=[5, 6])
    parser.add_argument('--repetitions', type=int, default=2)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    sys.path.insert(0, str(args.shanz_repository))
    sys.path.insert(0, str(args.shanz_repository/'scripts'))
    sys.path.insert(0, str(repository))
    os.environ['DARCY_FLOAT'] = 'float64'
    benchmark_profile(run_directory=args.run_directory, output_directory=args.output_directory,
        shanz_repository=args.shanz_repository, levels=args.max_levels, repetitions=args.repetitions,
        device=args.device)
