"""Time fast FP64 steady solves from the existing MF6 fixture inputs."""
from pathlib import Path
from time import perf_counter
import argparse
import hashlib
import json
import statistics
import subprocess

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.unconfined_truth_io import load_truth_artifact, build_solve_kwargs
from DARCY_WARP_PACKAGE.model_builder import _build_domain, _build_dem, make_ugly_T_field
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


def prepare(*, artifact_path, formulation, confined_controls):
    if formulation == 'unconfined':
        artifact = load_truth_artifact(path=artifact_path)
        constructor = dict(artifact['constructor_settings'])
        fields = dict(T_field=artifact['initial_transmissivity'], R_field=artifact['rhs_recharge'],
            active=artifact['active'], bc_mask=artifact['bc_mask'], bc_values=artifact['bc_values'])
        controls = build_solve_kwargs(artifact=artifact)
        controls['inner_implementation'] = 'fast'
        controls['solver'] = 'unconfined_picard_kcycle'
        return constructor, fields, controls, artifact['heads'], artifact['active'], artifact['provenance'].get('mf6_engine_time')
    with np.load(file=artifact_path) as loaded:
        artifact = {name:loaded[name] for name in loaded.files}
    nx, ny = int(artifact['nx']), int(artifact['ny'])
    domain = _build_domain(nx=nx, ny=ny)
    conductivity = np.full(shape=(ny,nx), fill_value=float(artifact['t_isotropic_value'])) if bool(artifact['t_isotropic']) else make_ugly_T_field(nx=nx, ny=ny, domain=domain, seed=int(artifact['seed']))
    fields = dict(T_truth=conductivity, R_truth=np.full(shape=(ny,nx), fill_value=float(artifact['r_truth'])), width=float(artifact['width']))
    constructor = dict(nx=nx, ny=ny, dx=float(artifact['dx']), use_ghb=bool(artifact['ghb']),
        aq_thickness=float(artifact['thickness']), solver_type='kcycle')
    controls = dict(confined_controls, implementation='fast', initial_head=_build_dem(domain=domain),
        solver='confined_kcycle', return_info=True)
    return constructor, fields, controls, artifact['heads'], domain == 1, float(artifact['mf6_seconds'])


def benchmark(*, artifact_path, formulation, confined_controls, output, repeats, device, head_gate):
    constructor, fields, controls, truth, active, mf6_seconds = prepare(
        artifact_path=artifact_path, formulation=formulation, confined_controls=confined_controls)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    root = Path(__file__).resolve().parents[1]
    report = dict(artifact=str(artifact_path), artifact_sha256=hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        commit=subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
        source_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((root/'DARCY_WARP_PACKAGE/solvers').glob('*.py'))},
        formulation=formulation, constructor=constructor,
        controls={name:value for name,value in controls.items() if not isinstance(value,np.ndarray)},
        mf6_seconds=mf6_seconds, head_gate=head_gate, rows=rows)
    for repeat in range(repeats+1):
        with WarpDarcySolver(device=device, **constructor) as model:
            started = perf_counter()
            if formulation == 'unconfined':
                model.build_from_fields(**fields)
            else:
                model.build_from_truth_inputs(**fields)
            wp.synchronize_device(device=device)
            build_seconds = perf_counter()-started
            started = perf_counter()
            head, info = model.solve(**controls)
            wp.synchronize_device(device=device)
            elapsed = perf_counter()-started
        mask = np.asarray(active) != 0
        finite = bool(np.isfinite(head[mask]).all())
        error = float(np.max(np.abs(head[mask]-truth[mask])))
        row = dict(repeat=repeat, warmup=repeat==0, seconds=elapsed, build_seconds=build_seconds,
            converged=bool(info.get('converged')), maximum_head_error=error,
            passed=bool(info.get('converged') and finite and error <= head_gate),
            iterations=info.get('outer_iterations',info.get('n_cycles_used')))
        rows.append(row)
        np.savez(file=output/'latest_heads.npz', heads=head)
        (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row),flush=True)
    report['median_seconds'] = statistics.median(row['seconds'] for row in rows if not row['warmup'])
    report['passed'] = all(row['passed'] for row in rows)
    (output/'results.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(dict(median_seconds=report['median_seconds'],passed=report['passed'])),flush=True)
    return report['passed']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact', type=Path, required=True)
    parser.add_argument('--formulation', choices=('confined','unconfined'),required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--device',default='cuda:0')
    args = parser.parse_args()
    confined_controls = dict(max_cycles=200, max_levels=6, nu_pre=2, nu_post=2, nu_coarse=10,
        omega=0.7, rel_tol=5e-7, abs_tol_min=5e-7, check_every_no=5)
    head_gate = 2e-4 if args.formulation == 'confined' else 5e-4
    if args.repeats < 1:
        raise ValueError('At least one timed repeat is required')
    wp.init()
    passed = benchmark(artifact_path=args.artifact, formulation=args.formulation,
        confined_controls=confined_controls, output=args.output, repeats=args.repeats,
        device=args.device, head_gate=head_gate)
    raise SystemExit(0 if passed else 1)
