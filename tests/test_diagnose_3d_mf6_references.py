from types import SimpleNamespace

import numpy as np
import pytest

from working_tests.diagnose_3d_mf6_references import audit_saved_history, compare_history, diagnostic_options


@pytest.mark.parametrize('key,value', [('outer_dvclose', 1e-4), ('inner_dvclose', 1e-6),
                                     ('rcloserecord', [1e-4, 'RELATIVE_RCLOSE'])])
def test_diagnostic_cannot_change_baseline_convergence_tolerances(key, value):
    original = dict(outer_dvclose=1e-6, inner_dvclose=1e-8, rcloserecord=[1e-6, 'RELATIVE_RCLOSE'])
    with pytest.raises(ValueError, match='preserve baseline convergence tolerance'):
        diagnostic_options(original=original, overrides={key: value}, tolerance_keys=tuple(original))


def test_strategy_change_keeps_owned_unchanged_tolerances():
    original = dict(outer_dvclose=1e-6, inner_dvclose=1e-8, rcloserecord=[1e-6, 'RELATIVE_RCLOSE'])
    changed = diagnostic_options(original=original, overrides=dict(under_relaxation='NONE'),
                                 tolerance_keys=tuple(original))
    assert all(changed[key] == original[key] for key in original)
    assert changed['rcloserecord'] is not original['rcloserecord']


def ghb_histories():
    shape = (1, 1, 2)
    case = SimpleNamespace(active=np.ones(shape=shape, dtype=int),
        fixed=np.asarray(a=[0, 1]).reshape(shape),
        ghb_conductance=np.asarray(a=[3., 0.]).reshape(shape),
        ghb_head=np.asarray(a=[4., 0.]).reshape(shape))
    heads = np.asarray(a=[[1., 0.], [2., 0.]]).reshape((2, *shape))
    reference = dict(heads=heads.copy(), ghb_inflows=case.ghb_conductance * (case.ghb_head - heads),
                     times=np.asarray(a=[7., 14.]))
    return case, heads, reference


def test_native_ghb_sign_error_fails_even_when_heads_match():
    case, heads, reference = ghb_histories()
    good = compare_history(heads=heads, reference=reference, case=case, expected_periods=2,
                           roundoff_ulps=32., flux_floor=1e-14, expected_times=np.asarray(a=[7., 14.]))
    assert good['head_max_m'] == 0.
    assert good['ghb_law_valid']
    reference['ghb_inflows'] *= -1.
    bad = compare_history(heads=heads, reference=reference, case=case, expected_periods=2,
                          roundoff_ulps=32., flux_floor=1e-14)
    assert bad['head_max_m'] == 0.
    assert not bad['ghb_law_valid']


def test_ghb_comparison_allows_flux_difference_due_to_measured_head_error():
    case, heads, reference = ghb_histories()
    heads[:, 0, 0, 0] += 1e-6
    result = compare_history(heads=heads, reference=reference, case=case, expected_periods=2,
                             roundoff_ulps=32., flux_floor=1e-14)
    assert result['ghb_law_valid']
    assert result['head_max_m'] > 0.


def test_incomplete_histories_and_shifted_native_times_are_rejected():
    case, heads, reference = ghb_histories()
    with pytest.raises(ValueError, match='Complete matching time histories'):
        compare_history(heads=heads[:1], reference=reference, case=case, expected_periods=2,
                        roundoff_ulps=32., flux_floor=1e-14)
    with pytest.raises(ValueError, match='timestamps'):
        compare_history(heads=heads, reference=reference, case=case, expected_periods=2,
                        roundoff_ulps=32., flux_floor=1e-14, expected_times=np.asarray(a=[7., 15.]))


def test_saved_gpu_history_is_independently_reaudited():
    from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase
    from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
    shape = (1, 1, 2)
    one = np.ones(shape=shape)
    case = NPF662LayeredCase(kx=one, ky=one, kz=one, active=one.astype(int),
        fixed=np.asarray(a=[0, 1]).reshape(shape), values=2.*one,
        source=np.asarray(a=[.4, 0.]).reshape(shape), initial=2.*one,
        dx=1., dy=1., z_interfaces=np.stack(arrays=[3.*one[0], 0.*one[0]]), cell_type=0)
    head = np.asarray(a=[2.+.4/3., 2.]).reshape(shape)
    checks, _ = NPF662MatrixFreeReference(case=case).evaluate(head=head)
    assert checks['independent_accepted']
    original = [dict(physical=checks)]
    good = audit_saved_history(heads=head[None], case=case, factors=np.ones(shape=1),
        mode='steady', dt=7., ss=1e-5, sy=0., original_periods=original, roundoff_acceptance=False)
    assert good[0]['independent_accepted']
    head[0, 0, 0] += .01
    bad = audit_saved_history(heads=head[None], case=case, factors=np.ones(shape=1),
        mode='steady', dt=7., ss=1e-5, sy=0., original_periods=original, roundoff_acceptance=False)
    assert not bad[0]['independent_accepted']
