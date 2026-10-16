"""Fresh-process cold checks, including imports, CUDA context, JIT and audit."""
from pathlib import Path
from time import perf_counter
import json
import os
import subprocess
import sys


def run_worker(*, baseline, output, case_id):
    import numpy as np
    import warp as wp
    wp.config.kernel_cache_dir = str((output/'kernel_cache').resolve())
    from benchmark_npf662_baseline import MemorySampler, write_json
    from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case
    from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    from DARCY_WARP_PACKAGE.solvers import npf662_newton

    settings = json.loads(s=(baseline/'frozen_settings.json').read_text())['settings']
    wp.init(); start = perf_counter()
    compiled = []
    for module in ('DARCY_WARP_PACKAGE.solvers.npf662_kernels', 'warp.optim.linear'):
        stamp = perf_counter(); wp.load_module(module=module, device='cuda:0')
        wp.synchronize_device(device='cuda:0'); compiled.append(dict(module=module, seconds=perf_counter()-stamp))
    memory = MemorySampler(**settings['memory_options']); memory.start()
    case = load_npf662_case(path=baseline/case_id/'inputs.npz')
    nz, ny, nx = case.active.shape
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy, device='cuda:0',
        implementation='fast', solver='kcycle', formulation='mf6_6_6_2_npf')
    model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx, ky_field=case.ky,
        kz_field=case.kz, active=case.active, bc_mask=case.fixed, bc_values=case.values,
        rhs=case.source, initial_head=case.initial)
    built = perf_counter()
    try:
        head, info = model.solve(unconfined=bool(np.any(case.cell_types)), npf_cell_type=case.cell_types,
            **settings['newton_options'])
        wp.synchronize_device(device='cuda:0'); solved = perf_counter()
        checks, unused = NPF662MatrixFreeReference(case=case).evaluate(head=head)
        accepted_at = perf_counter()
        peak = memory.stop()
        info.get('effective_options', {}).pop('cell_type', None)
        write_json(path=output/'worker.json', value=dict(case_hash=case.digest(),
            accepted=bool(info['converged'] and checks['independent_accepted']),
            explicit_compilation=compiled, construction_and_explicit_compilation_seconds=built-start,
            public_solve_wall_seconds=solved-built, independent_audit_seconds=accepted_at-solved,
            post_import_context_to_acceptance_seconds=accepted_at-start, memory=peak, info=info,
            independent_checks=checks, runtime=dict(python=sys.version, warp=wp.__version__, numpy=np.__version__)))
    finally:
        model.close()


def run_parent(*, baseline, output, cases):
    output.mkdir(parents=True, exist_ok=False)
    results = {}
    for case_id in cases:
        directory = output/case_id; directory.mkdir()
        environment = os.environ.copy()
        environment['NPF_COLD_CHILD'] = case_id
        start = perf_counter()
        with (directory/'stdout.log').open(mode='w') as stream:
            run = subprocess.run(args=[sys.executable, str(Path(__file__).resolve())], env=environment,
                stdout=stream, stderr=subprocess.STDOUT, check=False)
        seconds = perf_counter()-start
        if run.returncode != 0:
            raise RuntimeError(f'Cold process failed; retain {directory}')
        worker = json.loads(s=(directory/'worker.json').read_text())
        worker['fresh_process_wall_seconds'] = seconds
        worker['timing_boundary'] = 'Before interpreter launch through process exit; includes imports, context, compilation, model/workspace/graphs, solve, independent acceptance and small JSON output.'
        results[case_id] = worker
        print(f"Cold process {case_id}: accepted={worker['accepted']}, end-to-end={seconds:.6g} s", flush=True)
    (output/'results.json').write_text(data=json.dumps(results, indent=2, allow_nan=False))


if __name__ == "__main__":
    baseline = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/run_001')
    output = Path('benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1/cold_process_001')
    cases = ['confined', 'unconfined']
    child_case = os.environ.get('NPF_COLD_CHILD')
    if child_case is None:
        run_parent(baseline=baseline, output=output, cases=cases)
    else:
        run_worker(baseline=baseline, output=output/child_case, case_id=child_case)
