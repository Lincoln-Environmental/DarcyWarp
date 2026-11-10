"""Twenty weekly NPF/STO steps on the exact saved 20-case steady parity matrix."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import numpy as np
import warp as wp
from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case, save_npf662_case
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
from DARCY_WARP_PACKAGE.validation.npf662_transient_mf6 import run_transient_mf6
from DARCY_WARP_PACKAGE.validation.json_output import atomic_json
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls


def main():
    """Replay saved physical inputs with independent native and Warp histories."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steady-evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mf6', type=Path, required=True)
    parser.add_argument('--mf6-sha256', default='21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176')
    parser.add_argument('--case', nargs='+')
    parser.add_argument('--ss', type=float, default=1e-5)
    parser.add_argument('--sy', type=float, default=.15)
    args = parser.parse_args()
    factors = np.repeat([1., 1.5, .25, 1.], 5)
    dt = 7.
    definitions = []
    for folder in ['validated_16_32', 'validated_50']:
        previous = json.loads(args.steady_evidence.joinpath(folder, 'results.json').read_text())
        for case in previous['cases']:
            identity = next(row['case_sha256'] for row in previous['runs']
                if row['case_id'] == case['case_id'] and 'case_sha256' in row)
            definitions.append(dict(name=case['case_id'], input=args.steady_evidence/folder/case['case_id']/'inputs.npz',
                sha256=identity, source_package='rch'))
    for phase in ['confined', 'unconfined']:
        gate = json.loads(args.steady_evidence.joinpath('layered_'+phase, 'gate.json').read_text())
        for row in gate['cases']:
            definitions.append(dict(name=phase+'_'+row['case_id'], sha256=row['input_hash'],
                input=args.steady_evidence/('layered_'+phase)/row['case_id']/'inputs.npz', source_package='wel'))
    if args.case:
        unknown = set(args.case)-{item['name'] for item in definitions}
        if unknown:
            parser.error(f'Unknown cases: {sorted(unknown)}')
        definitions = [item for item in definitions if item['name'] in args.case]
    args.output.mkdir(parents=True, exist_ok=False)
    source_root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve()]+list(source_root.joinpath('DARCY_WARP_PACKAGE').rglob('*.py'))
    hashes = {}
    for path in paths:
        relative = path.relative_to(source_root)
        target = args.output/'source_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        hashes[str(relative)] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256=hashes, dt_days=dt, periods=20, end_time_days=140.,
        ss_inverse_m=args.ss, sy=args.sy, source_factors=factors.tolist(), device='cpu',
        warp=wp.__version__, numpy=np.__version__, mf6_sha256=args.mf6_sha256,
        gates=dict(head_rmse_m=1e-3, head_max_m=5e-3, relative_face_flow=1e-6,
            storage_relative_throughput=1e-6, relative_cell_imbalance=1e-8), cases=[])
    atomic_json(path=args.output/'frozen_settings.json', value=result)
    for definition in definitions:
        name = definition['name']
        directory = args.output/name
        directory.mkdir()
        case = load_npf662_case(path=definition['input'])
        if case.digest() != definition['sha256']:
            raise ValueError(f'Saved steady case identity changed: {name}')
        save_npf662_case(case=case, path=directory/'inputs.npz')
        row = dict(case=name, input_hash=case.digest(), shape=list(case.active.shape),
            source_package=definition['source_package'], periods=[], passed=False)
        result['cases'].append(row)
        try:
            native, native_info = run_transient_mf6(case=case, directory=directory/'mf6', executable=args.mf6,
                executable_sha256=args.mf6_sha256, source_factors=factors, dt=dt, ss=args.ss, sy=args.sy,
                source_package=definition['source_package'])
            row['native'] = native_info
            previous = case.initial.copy()
            heads = []
            fluxes = []
            storage_rates = []
            nz, ny, nx = case.active.shape
            with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy, device='cpu',
                    formulation='mf6_6_6_2_npf', implementation='fast') as model:
                model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx, ky_field=case.ky,
                    kz_field=case.kz, active=case.active, bc_mask=case.fixed, bc_values=case.values,
                    rhs=case.source, initial_head=case.initial)
                for period, factor in enumerate(factors):
                    step = replace(case, source=case.source*factor)
                    old = previous.copy()
                    model.update_rhs(rhs=step.source)
                    model.update_initial_head(initial_head=previous)
                    head, info = model.solve(unconfined=bool(np.any(case.cell_types)), npf_cell_type=case.cell_types,
                        transient=True, dt=dt, head_prev=previous, ss=args.ss, sy=args.sy,
                        npf_preconditioner='kcycle', npf_kcycle_controls=NPF662KCycleControls(use_cuda_graph=False),
                        max_outer=100, max_cycles=4000, krylov_restart=48)
                    if not np.array_equal(previous, old):
                        raise AssertionError('Previous timestep state mutated during nonlinear solve')
                    checks, fields = NPF662TransientReference(case=step, previous=previous, dt=dt,
                        ss=args.ss, sy=args.sy).evaluate(head=head)
                    free = (case.active != 0)&(case.fixed == 0)
                    delta = head[free]-native['heads'][period][free]
                    rmse = float(np.sqrt(np.mean(delta**2)))
                    maximum = float(np.max(np.abs(delta)))
                    qdelta = fields['flux']-native['fluxes'][period]
                    qrelative = float(np.linalg.norm(qdelta)/max(np.linalg.norm(native['fluxes'][period]), np.finfo(float).tiny))
                    storage_relative = float(np.sum(np.abs(fields['storage_rate']-native['storage_rates'][period]))/
                        max(checks['gross_throughput'], np.finfo(float).tiny))
                    accepted = bool(info['converged'] and checks['independent_accepted']
                        and native_info['periods'][period]['accepted'] and rmse <= 1e-3 and maximum <= 5e-3
                        and qrelative <= 1e-6 and storage_relative <= 1e-6)
                    period_row = dict(period=period+1, time_days=dt*(period+1), accepted=accepted,
                        head_rmse_m=rmse, head_max_m=maximum, flux_relative_l2=qrelative,
                        storage_relative_throughput=storage_relative, physical=checks,
                        outer_iterations=info['outer_iterations'], krylov_iterations=info['krylov_iterations'],
                        workspace_reused=info['workspace_reused'], hierarchy_reused=info['multigrid']['hierarchy_reused'])
                    row['periods'].append(period_row)
                    atomic_json(path=directory/f'warp_period_{period+1:02d}.json', value=finite_diagnostics(value=info))
                    heads.append(head.copy())
                    fluxes.append(fields['flux'].copy())
                    storage_rates.append(fields['storage_rate'].copy())
                    previous = head.copy()
                    atomic_json(path=args.output/'results.json', value=finite_diagnostics(value=result))
                    if not accepted:
                        print(name, 'period', period+1, 'FAIL', period_row, flush=True)
            temporary = directory/'.warp_history.npz.tmp'
            with temporary.open('wb') as stream:
                np.savez_compressed(stream, heads=np.asarray(heads), fluxes=np.asarray(fluxes),
                    storage_rates=np.asarray(storage_rates), times=native['times'])
            temporary.replace(directory/'warp_history.npz')
            row.update(passed=bool(native_info['passed'] and len(row['periods']) == 20
                and all(item['accepted'] for item in row['periods'])),
                worst_head_m=max(item['head_max_m'] for item in row['periods']),
                worst_head_rmse_m=max(item['head_rmse_m'] for item in row['periods']),
                worst_flux_relative_l2=max(item['flux_relative_l2'] for item in row['periods']),
                worst_storage_relative_throughput=max(item['storage_relative_throughput'] for item in row['periods']))
        except Exception as error:
            row['error'] = f'{type(error).__name__}: {error}'
        atomic_json(path=args.output/'results.json', value=finite_diagnostics(value=result))
        print(name, 'passed=', row['passed'], 'periods=', len(row['periods']), 'error=', row.get('error'),
            'head=', row.get('worst_head_m'), flush=True)
    result['passed'] = bool(len(result['cases']) == len(definitions) and all(row['passed'] for row in result['cases']))
    atomic_json(path=args.output/'results.json', value=finite_diagnostics(value=result))
    print('Completed:', len(result['cases']), 'cases, passed=', result['passed'], flush=True)
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
