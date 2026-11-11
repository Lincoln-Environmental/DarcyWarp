"""Compare numerical solve time when physical K changes between calibrations."""
from contextlib import ExitStack
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from time import perf_counter
import argparse
import json
import numpy as np
import warp as wp
from DARCY_WARP_PACKAGE import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls
from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import generate_case
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
from benchmark_npf662_vs_mf6 import suite_settings


def make_model(*, case, device):
    nz, ny, nx = case.active.shape
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
        device=device, implementation='fast', formulation='mf6_6_6_2_npf')
    model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx,
        ky_field=case.ky, kz_field=case.kz, active=case.active, bc_mask=case.fixed,
        bc_values=case.values, rhs=case.source, initial_head=case.initial)
    return model


def solve_step(*, model, case, previous, native, options):
    model.update_rhs(rhs=case.source)
    model.update_initial_head(initial_head=previous)
    start = perf_counter()
    head, info = model.solve(unconfined=True, npf_cell_type=case.cell_types,
        head_prev=previous, npf_device_native=native, **options)
    wp.synchronize_device(device=model.device)
    public = perf_counter()-start
    assert info['converged'], info['stop_reason']
    return head, dict(public_seconds=public, numerical_seconds=info['seconds'],
        setup_seconds=info['setup_seconds'], krylov=info['krylov_iterations'],
        readbacks=info['krylov_scalar_readbacks'], outer=info['outer_iterations'])


def run(*, size, device, k_factors, source_factors, options, output, head_tolerance):
    definition = suite_settings(sizes=[size], variants=['convertible'])[0]
    geology = dict(definition['geology'], layer_k=np.asarray(definition['geology']['layer_k']))
    base = generate_case(settings=geology, convertible=definition['convertible'])
    output.mkdir(parents=True, exist_ok=False)
    results = dict(size=size, k_factors=k_factors, periods=len(source_factors),
        case_sha256=base.digest(), convertible_layers=definition['convertible'],
        controls={name:asdict(value) if is_dataclass(value) else value for name, value in options.items()},
        timing='Numerical seconds = synchronized Newton/Krylov interval, excludes setup and final audit. Public solve and K update reported separately.',
        comparison='Host versus device execution with identical solver controls and K. Independent physical audit at every endpoint. No MF6 speedup claimed for changed-K cases.', cases=[])
    with ExitStack() as stack:
        models = {native:stack.enter_context(make_model(case=base, device=device)) for native in (False, True)}
        for native, model in models.items():
            solve_step(model=model, case=base, previous=base.initial, native=native, options=options)
        for factor in k_factors:
            case = replace(base, kx=base.kx*factor, ky=base.ky*factor, kz=base.kz*factor)
            row = dict(k_factor=factor, runs={}, worst_head_difference_m=0.)
            previous = {native:case.initial.copy() for native in models}
            for native, model in models.items():
                start = perf_counter()
                model.update_K_fields(kx_field=case.kx, ky_field=case.ky, kz_field=case.kz)
                wp.synchronize_device(device=device)
                row['runs'][str(native)] = dict(k_update_seconds=perf_counter()-start, steps=[])
            for period, source_factor in enumerate(source_factors):
                step = replace(case, source=case.source*source_factor)
                heads = {}
                for native, model in models.items():
                    heads[native], info = solve_step(model=model, case=step, previous=previous[native],
                        native=native, options=options)
                    row['runs'][str(native)]['steps'].append(info)
                checks, unused = NPF662TransientReference(case=step, previous=previous[True],
                    dt=options['dt'], ss=options['ss'], sy=options['sy']).evaluate(
                        head=heads[True], relative_tolerance=options['npf_final_relative_tolerance'])
                assert checks['independent_accepted']
                row['runs']['True']['steps'][-1]['independent_physical'] = checks
                error = float(np.max(np.abs(heads[True]-heads[False])))
                assert error <= head_tolerance, error
                row['worst_head_difference_m'] = max(row['worst_head_difference_m'], error)
                previous = heads
                if (period+1)%5 == 0:
                    print(f'K factor {factor}: step {period+1} accepted', flush=True)
            for item in row['runs'].values():
                for name in ('public_seconds', 'numerical_seconds', 'setup_seconds', 'readbacks', 'krylov'):
                    item[name] = sum(step[name] for step in item['steps'])
            results['cases'].append(row)
            (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')
            print(f"K={factor}: numerical old={row['runs']['False']['numerical_seconds']:.3f}s device={row['runs']['True']['numerical_seconds']:.3f}s", flush=True)
    results['passed'] = True
    (output/'results.json').write_text(json.dumps(results, indent=2)+'\n')


if __name__ == "__main__":
    size = 500
    device = 'cuda:0'
    k_factors = [.5, 1., 2.]
    source_factors = np.repeat([1., 1.5, .25, 1.], 5)
    options = dict(transient=True, dt=7., ss=1e-5, sy=.15, npf_preconditioner='kcycle',
        npf_kcycle_controls=NPF662KCycleControls(levels=5, pre_sweeps=2, post_sweeps=2, coarse_sweeps=20),
        npf_final_relative_tolerance=1e-7, max_outer=80, max_cycles=4000, krylov_restart=48)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('benchmark_results/gpu_native/calibration_runtime_500'))
    parser.add_argument('--controls', type=Path)
    arguments = parser.parse_args()
    output = arguments.output
    if arguments.controls:
        payload = json.loads(arguments.controls.read_text())
        options.update(payload['solve'], npf_kcycle_controls=NPF662KCycleControls(**payload['multigrid']),
            npf_newton_controls=NPF662NewtonControls(**payload['newton']))
        options.pop('npf_device_native', None)
    head_tolerance = 1e-6
    run(size=size, device=device, k_factors=k_factors, source_factors=source_factors,
        options=options, output=output, head_tolerance=head_tolerance)
