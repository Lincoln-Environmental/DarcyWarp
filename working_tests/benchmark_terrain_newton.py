"""Sequential reproducible terrain Newton validation; edit controls below."""
import hashlib
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import warp as wp

from shanz_terrain_case_study import build_case
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def compute_processes():
    raw=subprocess.check_output(args=['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True)
    return [line.strip() for line in raw.splitlines() if line.strip() and not line.strip().startswith(str(os.getpid())+',')]


def source_hashes():
    paths=[Path('DARCY_WARP_PACKAGE/solvers')/name for name in
        ('terrain_newton_3d.py','terrain_newton_kernels.py','terrain_device_3d.py','terrain_fast_3d.py')]
    return {str(path):hashlib.sha256(string=path.read_bytes()).hexdigest() for path in paths}


def run_case(*, size, basement_k, method, max_outer, output, device, controls, floor, ky_ratio, kz_ratio, verification_initial_path=None):
    other_processes_before = compute_processes()
    case = build_case(nx=size, ny=size, dx=controls['dx'])
    case['k'][-1] = basement_k
    if verification_initial_path is not None:
        case['initial'] = np.load(file=verification_initial_path)['head']
    solver = WarpDarcySolver3D(nx=size, ny=size, nz=case['layers'], dx=case['dx'], dy=case['dx'],
        dz=float(np.mean(a=case['interfaces'][:-1]-case['interfaces'][1:])), device=device, implementation='fast')
    wp.synchronize_device(device=device)
    setup_start = perf_counter()
    solver.build_from_terrain(z_interfaces=case['interfaces'], active_columns=case['active'],
        kx_field=case['k'], ky_field=ky_ratio*case['k'], kz_field=kz_ratio*case['k'],
        bc_mask=case['fixed'], bc_values=case['values'], rhs=case['source'],
        initial_head=case['initial'], contacts=case['contacts'])
    wp.synchronize_device(device=device)
    setup_seconds = perf_counter()-setup_start
    solve_options = {key:value for key,value in controls.items() if key != 'dx'}
    with solver:
        wp.synchronize_device(device=device)
        start = perf_counter()
        head, info = solver.solve(unconfined=True, device_picard=True, terrain_nonlinear_method=method,
            unconfined_min_sat=floor, unconfined_max_picard_iter=max_outer, **solve_options)
        wp.synchronize_device(device=device)
        seconds = perf_counter()-start
    other_processes_after = compute_processes()
    history = info.pop('outer_history')
    result = dict(initialization="original_cold" if verification_initial_path is None else "warm_verification_only",
        verification_initial_path=str(verification_initial_path) if verification_initial_path is not None else None, size=size, basement_k=basement_k, method=method, max_outer=max_outer,
        solve_wall_seconds=seconds, build_seconds=setup_seconds,
        dx=case["dx"], other_compute_processes_before=other_processes_before,
        other_compute_processes_after=other_processes_after, source_sha256=source_hashes(),
        outside_outer_loop_seconds=seconds-info["seconds"], controls=solve_options, floor=floor,
        info=info, python=sys.version, interpreter=sys.executable, warp=wp.__version__,
        revision=subprocess.check_output(args=['git','rev-parse','HEAD'],text=True).strip(),
        precision=os.environ.get('DARCY_FLOAT'), device=wp.get_device(ident=device).name)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(data=json.dumps(result,indent=2,default=float))
    output.with_suffix('.history.json.gz').write_bytes(data=gzip.compress(data=json.dumps(history).encode(),mtime=0))
    np.savez_compressed(file=output.with_suffix('.npz'), head=head)
    print(json.dumps(dict(output=str(output),converged=info['converged'],reason=info['device_stop_reason'],
        outers=info['n_outer_iterations'],krylov=info['krylov_iterations'],seconds=seconds,budget=info['budget'])),flush=True)
    return result


if __name__ == "__main__":
    wp.init()
    output_root = Path('audit_outputs/validated')
    device = 'cuda:0'
    floor = .01
    ky_ratio = .6
    kz_ratio = .15
    controls = dict(dx=20., unconfined_head_tol=1e-6, final_head_residual_tol=1e-6,
        rel_tol=1e-12, abs_tol_min=1e-12, final_budget_relative_tol=1e-8,
        unconfined_relax=.7, krylov_restart=24, max_levels=6, min_coarse_n=16, max_cycles=2000)
    max_outer = 2000
    picard_comparison_cap = 4000
    picard_control_cap = 8000
    repeats = 3
    warmup_outers = 2
    size_controls = [16, 32, 96]
    conductivity_controls = [.1, .01, .001]
    primary_size = 64
    primary_basement_k = .0001
    methods = ['picard', 'newton']
    for method in methods:
        run_case(size=primary_size, basement_k=primary_basement_k, method=method, max_outer=warmup_outers,
            output=output_root/f'warmup_{method}.json', device=device, controls=controls,
            floor=floor, ky_ratio=ky_ratio, kz_ratio=kz_ratio)
    for repeat in range(repeats):
        for method in methods:
            run_case(size=primary_size, basement_k=primary_basement_k, method=method,
                max_outer=picard_comparison_cap if method == 'picard' else max_outer,
                output=output_root/f'low_k_{method}_{repeat}.json',device=device,controls=controls,
                floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
    for basement_k in conductivity_controls:
        for method in methods:
            run_case(size=primary_size,basement_k=basement_k,method=method,
                max_outer=picard_control_cap if method == 'picard' else max_outer,
                output=output_root/f'control_{method}_{primary_size}_{basement_k:g}.json',device=device,controls=controls,
                floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
    for size in size_controls:
        for method in methods:
            run_case(size=size,basement_k=primary_basement_k,method=method,
                max_outer=picard_control_cap if method == 'picard' else max_outer,
                output=output_root/f'control_{method}_{size}_{primary_basement_k:g}.json',device=device,controls=controls,
                floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
