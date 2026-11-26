"""Explicit production mixed-precision WEL forcing, in its required FP32 process."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def independent_heads(*, shape, transmissivity, source, fixed_head):
    """Assemble an independent confined face matrix and solve its free rows."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
    ids = np.arange(np.prod(shape)).reshape(shape)
    a = np.concatenate((ids[:, :-1].ravel(), ids[:-1, :].ravel()))
    b = np.concatenate((ids[:, 1:].ravel(), ids[1:, :].ravel()))
    off = coo_matrix((np.full(a.size * 2, -transmissivity),
        (np.concatenate((a, b)), np.concatenate((b, a)))), shape=(ids.size, ids.size)).tocsr()
    matrix = off + diags(-np.asarray(off.sum(axis=1)).ravel())
    free = np.ones(shape, dtype=bool)
    free[:, 0] = False
    keep = free.ravel()
    head = np.full(ids.size, fixed_head)
    head[keep] = spsolve(matrix[keep][:, keep], source.ravel()[keep] - matrix[keep][:, ~keep] @ head[~keep])
    return head.reshape(shape)


def run_mixed_case(*, device, shape, dx, transmissivity, fixed_head, well_rates, recharge_rates, tolerance, mixed_controls):
    """Compare repeated production mixed solves to independently assembled WEL roots."""
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import MixedFastConfig, solve_mixed_fast
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, 0] = 1
    values = np.full(shape, fixed_head)
    cell = (shape[0] // 2, shape[1] // 2)
    errors = []
    with WarpDarcySolver(nx=shape[1], ny=shape[0], dx=dx, device=device) as model:
        model.build_from_fields(T_field=np.full(shape, transmissivity), R_field=0.,
            active=active, bc_mask=fixed, bc_values=values)
        for well, recharge in zip(well_rates, recharge_rates):
            model.update_wel_in_place(wel_entries=[(cell, well)])
            model.update_uniform_recharge_in_place(recharge_rate=recharge)
            pure_recharge = np.full(shape, recharge)
            head, info = solve_mixed_fast(model=model, initial_head_f64=values,
                bc_values_f64=values, gh_head_f64=None, R_f64=pure_recharge,
                config=MixedFastConfig(**mixed_controls))
            assert info['converged'] and info['production_precision_path'], info
            assert info['cuda_graph_replays'] > 0, info
            source = pure_recharge * dx * dx
            source[cell] += well
            expected = independent_heads(shape=shape, transmissivity=transmissivity,
                source=source, fixed_head=fixed_head)
            error = float(np.max(np.abs(head - expected)))
            errors.append(error)
            assert error <= tolerance, (error, head[cell], expected[cell])
            np.testing.assert_array_equal(model.accepted_wel_rates(head=head, info=info), [well])
    return errors


def test_production_mixed_wel_updates_match_independent_reference():
    """Run real CUDA with the production mixed precision model's FP32 storage."""
    import warp as wp
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    environment = dict(os.environ, DARCY_FLOAT='float32')
    result = subprocess.run([sys.executable, str(Path(__file__).resolve())], env=environment,
        capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    device = 'cuda:0'
    shape = (16, 16)
    dx = 100.
    transmissivity = 50.
    fixed_head = 10.
    well_rates = [-400., -800., -400.]
    recharge_rates = [0., 1.e-4, 0.]
    tolerance = 1.e-6
    mixed_controls = dict(min_coarse_cells=100, max_outer=400,
        rel_tol=1.e-12, abs_tol_min=1.e-12, dh_rms_tol=1.e-10)
    errors = run_mixed_case(device=device, shape=shape, dx=dx, transmissivity=transmissivity,
        fixed_head=fixed_head, well_rates=well_rates, recharge_rates=recharge_rates, tolerance=tolerance, mixed_controls=mixed_controls)
    print(json.dumps(dict(head_errors_m=errors)))
