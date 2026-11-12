"""Sanity reporting must retain full histories and distinguish reference criteria."""
import numpy as np
import pytest

from run_3d_sanity import measure, native_elapsed_seconds, sanity_reference_valid


def test_native_elapsed_includes_minutes():
    assert native_elapsed_seconds(text='Elapsed run time:  5 Minutes, 46.408 Seconds') == 346.408
    assert native_elapsed_seconds(text='Elapsed run time:  0.233 Seconds') == .233


def test_reference_uses_sanity_gate_and_still_rejects_invalid_physics():
    physical = dict(input_valid=True, boundary_condition_valid=True, physical_metrics_valid=True)
    info = dict(passed=False, periods=[dict(source_allocation_valid=True,
        physical=physical, native_relative_cell_imbalance=1e-5)])
    assert sanity_reference_valid(info=info, mode='transient', gates=dict(budget_percent=1.))
    physical['physical_metrics_valid'] = False
    assert not sanity_reference_valid(info=info, mode='transient', gates=dict(budget_percent=1.))


@pytest.mark.parametrize('missing', [False, True])
def test_bad_reference_does_not_truncate_physically_accepted_gpu_driver_history(missing):
    import warp as wp
    from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase
    from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
    wp.init()
    shape = (1, 1, 2)
    one = np.ones(shape)
    case = NPF662LayeredCase(kx=one, ky=one, kz=one, active=one.astype(int),
        fixed=np.array([0, 1]).reshape(shape), values=2.*one,
        source=np.array([.4, 0.]).reshape(shape), initial=np.array([.5, 2.]).reshape(shape),
        dx=1., dy=1., z_interfaces=np.stack([3.*one[0], 0.*one[0]]), cell_type=0)
    dt, ss = 7., .2
    old = case.initial.copy()
    heads, fluxes = [], []
    for period in range(3):
        capacity = ss*3./dt
        head = old.copy()
        head[0, 0, 0] = (.4+6.+capacity*old[0, 0, 0])/(3.+capacity)
        checks, fields = NPF662TransientReference(case=case, previous=old, dt=dt, ss=ss, sy=0.).evaluate(head=head)
        assert checks['independent_accepted']
        heads.append(head.copy())
        fluxes.append(fields['flux'].copy())
        old = head.copy()
    reference = dict(heads=np.stack(arrays=heads), fluxes=np.stack(arrays=fluxes))
    row, actual = measure(case=case, family='npf', mode='transient', label='diagonal',
        reference=None if missing else reference, native_accepted=False, device='cpu', profiles={}, factors=np.ones(3),
        dt=dt, ss=ss, sy=0., npf_options=dict(max_outer=30, max_cycles=100, krylov_restart=4),
        kcycle_controls=NPF662KCycleControls(use_cuda_graph=False),
        gates=dict(rmse=1e-3, maximum=1e-2), progress_every=3)
    assert not row['accepted']
    assert row['gpu_physical_accepted']
    assert len(row['periods']) == 3
    assert all(period['solver_converged'] for period in row['periods'])
    if missing:
        assert all(period['head_max_m'] is None and period['ghb_law_parity'] is None for period in row['periods'])
    np.testing.assert_allclose(actual=actual, desired=reference['heads'], rtol=0., atol=1e-9)
