"""Re-measure archived original forcing with K-cycle only and unchanged final gates."""
from dataclasses import asdict, replace
import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp


def qualify_original(*, grids, configurations, saved, initial_head, top, repeats, output, device, max_outer, relative_gate):
    """Separate preconditioner execution from Newton control changes.

    :param grids: Original regional grid shapes.
    :param configurations: Explicit Newton controls; all runs use FGMRES/K-cycle.
    :param saved: Existing qualified K-cycle controls.
    :param initial_head: Original physical initial datum.
    :param top: Original grid top elevation.
    :param repeats: Warm measurements from the identical original guess.
    :param output: Evidence destination.
    :param device: Actual CUDA device.
    :param max_outer: Original public nonlinear iteration budget.
    :param relative_gate: Original physical relative acceptance gate.
    :return: Repeated timing and actual iteration evidence.
    """
    from regional_package_fields import heterogeneous_fields
    from qualify_package_kcycle import build_model, summary, timed_solve
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
    output.mkdir(parents=True, exist_ok=True)
    report = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(), device=device,
        comparison='Archived original forcing; no diagonal reruns. Original physical final gates retained.', rows=[])
    for label, shape in grids.items():
        fields = heterogeneous_fields(shape=shape)
        fields.update(initial_head=np.full(shape, initial_head), top_elevation=top)
        digest = hashlib.sha256()
        for name in ('kx_field', 'ky_field', 'kz_field', 'R_field', 'active', 'bc_mask', 'bc_values'):
            digest.update(np.ascontiguousarray(fields[name]).tobytes())
        digest.update(repr((fields['wel_entries'], fields['drn_entries'], fields['riv_entries'])).encode())
        for name, nonlinear in configurations:
            started = perf_counter()
            with build_model(fields=fields, device=device, formulation='mf6_6_6_2_npf', raw_control=False) as model:
                construction = perf_counter() - started
                options = dict(npf_preconditioner='kcycle', npf_krylov_method='fgmres',
                    npf_newton_controls=nonlinear, npf_kcycle_controls=NPF662KCycleControls(**saved['multigrid']),
                    max_outer=max_outer, max_cycles=saved['solve']['max_cycles'],
                    npf_final_relative_tolerance=relative_gate, top_elevation=top)
                measurements = []
                for repeat in range(repeats + 1):
                    model.update_initial_head(initial_head=fields['initial_head'])
                    head, info, row = timed_solve(model=model, options=options, preconditioner='kcycle', direct=False)
                    measurements.append(row)
                result = dict(grid=label, shape=list(shape), configuration=name, nonlinear=asdict(nonlinear),
                    multigrid=saved['multigrid'], physical_relative_gate=relative_gate,
                    forcing_sha256=digest.hexdigest(), construction_seconds=construction,
                    cold=measurements[0], warm=measurements[1:], warm_summary=summary(rows=measurements[1:]))
                report['rows'].append(result)
                (output / 'result.json').write_text(json.dumps(report, indent=2))
                print(label, name, result['warm_summary'], 'Newton', measurements[-1]['outer_iterations'],
                    'Krylov', measurements[-1]['krylov_iterations'], flush=True)
    return report


if __name__ == "__main__":
    os.environ.setdefault('DARCY_FLOAT', 'float64')
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls
    root = Path(__file__).resolve().parents[1]
    saved = json.loads((root / 'working_tests/npf_ghb_five_level_controls.json').read_text())
    original = NPF662NewtonControls()
    qualified = NPF662NewtonControls(**saved['newton'])
    full_step_cap = 1.e6
    configurations = [('original_newton_controls', original), ('qualified_newton_controls', qualified),
        ('qualified_confined_full_step', replace(qualified, max_head_change_m=full_step_cap))]
    grids = {'100k': (10, 112, 90), '1M': (5, 450, 450)}
    initial_head = 40.
    top = 40.
    max_outer = 2000
    relative_gate = 1.e-8
    repeats = 3
    output = root / 'benchmark_results/package_audit_repair/original_forcing_kcycle'
    device = 'cuda:0'
    wp.init()
    qualify_original(grids=grids, configurations=configurations, saved=saved,
        initial_head=initial_head, top=top, repeats=repeats, output=output, device=device,
        max_outer=max_outer, relative_gate=relative_gate)
