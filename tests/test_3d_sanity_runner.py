"""Sanity reporting must retain full histories and distinguish reference criteria."""
import numpy as np
import pytest

from run_3d_sanity import measure, native_elapsed_seconds, sanity_reference_valid


def test_native_elapsed_includes_minutes():
    assert native_elapsed_seconds(text='Elapsed run time:  5 Minutes, 46.408 Seconds') == 346.408
    assert native_elapsed_seconds(text='Elapsed run time:  0.233 Seconds') == .233


def test_reference_uses_sanity_gate_and_still_rejects_invalid_physics():
    physical = dict(input_valid=True, boundary_condition_valid=True, physical_metrics_valid=True, relative_cell_imbalance=1e-5)
    info = dict(passed=False, native_success=True, periods=[dict(source_allocation_valid=True,
        chd_budget_valid=True, ghb_law_valid=True, storage_difference_relative_throughput=1e-6,
        physical=physical, native_relative_cell_imbalance=1e-5)])
    assert sanity_reference_valid(info=info, mode='transient', gates=dict(budget_percent=1.))
    physical['physical_metrics_valid'] = False
    assert not sanity_reference_valid(info=info, mode='transient', gates=dict(budget_percent=1.))


@pytest.mark.parametrize('reference_state', ['missing', 'uncertified', 'reversed', 'valid', 'zero_flow'])
def test_history_qualification_enforces_native_face_flows(reference_state):
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
    if reference_state == 'zero_flow':
        case.source.fill(0.)
        case.initial[:] = case.values
    dt, ss = 7., .2
    old = case.initial.copy()
    heads, fluxes, storage = [], [], []
    for period in range(3):
        capacity = ss*3./dt
        head = old.copy()
        head[0, 0, 0] = (case.source[0, 0, 0]+6.+capacity*old[0, 0, 0])/(3.+capacity)
        checks, fields = NPF662TransientReference(case=case, previous=old, dt=dt, ss=ss, sy=0.).evaluate(head=head)
        assert checks['independent_accepted']
        heads.append(head.copy())
        fluxes.append(fields['flux'].copy())
        storage.append(fields['storage_rate'].copy())
        old = head.copy()
    reference = dict(heads=np.stack(arrays=heads), fluxes=np.stack(arrays=fluxes),
        storage_rates=np.stack(arrays=storage), saturations=np.ones((3, *shape)))
    if reference_state == 'reversed':
        reference['fluxes'] *= -1.
    native_accepted = reference_state in {'reversed', 'valid', 'zero_flow'}
    row, actual = measure(case=case, family='npf', mode='transient', label='diagonal',
        reference=None if reference_state == 'missing' else reference, native_accepted=native_accepted, device='cpu', profiles={}, factors=np.ones(3),
        dt=dt, ss=ss, sy=0., npf_options=dict(max_outer=30, max_cycles=100, krylov_restart=4),
        kcycle_controls=NPF662KCycleControls(use_cuda_graph=False),
        gates=dict(rmse=1e-3, maximum=1e-2, budget_percent=1., flux_relative_l2=1e-2), progress_every=3)
    assert row['accepted'] is (reference_state in {'valid', 'zero_flow'})
    assert row['gpu_physical_accepted']
    assert len(row['periods']) == 3
    assert all(period['solver_converged'] for period in row['periods'])
    if reference_state == 'reversed':
        assert all(period['flux_relative_l2'] == pytest.approx(2.) and not period['accepted'] for period in row['periods'])
    if reference_state == 'missing':
        assert all(period['head_max_m'] is None and period['ghb_law_parity'] is None for period in row['periods'])
    np.testing.assert_allclose(actual=actual, desired=reference['heads'], rtol=0., atol=1e-9)


def test_native_cache_retry_after_timeout_and_transient_build_pin(tmp_path, pinned_mf6):
    from dataclasses import replace
    from pathlib import Path
    from test_3d_ghb_mf6 import ghb_case
    from run_3d_sanity import prepare_reference
    from sanity_3d_controls import read_profiles

    initial = ghb_case(convertible=False)
    source = initial.source.copy()
    source[1:] = 0.
    case = replace(initial, source=source)
    profiles = read_profiles(root=Path(__file__).resolve().parents[1])['mf6']
    options = dict(case=case, family='npf', mode='steady', cache=tmp_path/'cache',
        cached_only=False, executable=pinned_mf6, native_options=profiles['steady_confined'],
        factors=np.ones(2), dt=.1, ss=.001, sy=.1, gates={'budget_percent': 1.})
    with pytest.raises(RuntimeError, match='wall_timeout'):
        prepare_reference(timeout_seconds=1e-8, **options)
    _, manifest = prepare_reference(timeout_seconds=20, **options)
    assert manifest['accepted'] and not manifest['reused']
    _, cached = prepare_reference(timeout_seconds=20, **options)
    assert cached['reused'] and cached['native_single_run_seconds'] == manifest['native_single_run_seconds']
    binary = tmp_path/'mf6'
    binary.write_bytes(pinned_mf6.read_bytes() + b'\n')
    binary.chmod(0o755)
    options.update(mode='transient', executable=binary, native_options=profiles['transient'])
    with pytest.raises(ValueError, match='pinned MF6 contract'):
        prepare_reference(timeout_seconds=20, **options)


def test_concurrent_native_cache_publishers_return_same_manifest(tmp_path, pinned_mf6, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace
    from pathlib import Path
    from threading import Barrier
    from DARCY_WARP_PACKAGE.validation import npf662_mf6
    from test_3d_ghb_mf6 import ghb_case
    from run_3d_sanity import prepare_reference
    from sanity_3d_controls import read_profiles

    initial = ghb_case(convertible=False)
    source = initial.source.copy()
    source[1:] = 0.
    options = dict(case=replace(initial, source=source), family='npf', mode='steady',
        cache=tmp_path, cached_only=False, executable=pinned_mf6,
        native_options=read_profiles(root=Path(__file__).resolve().parents[1])['mf6']['steady_confined'],
        factors=np.ones(1), dt=1., ss=.001, sy=.1, gates={'budget_percent': 1.}, timeout_seconds=20)
    both_writing = Barrier(2)
    original = npf662_mf6.run_reference_mf6

    def concurrent_solve(**kwargs):
        both_writing.wait(timeout=10)
        return original(**kwargs)

    monkeypatch.setattr(npf662_mf6, 'run_reference_mf6', concurrent_solve)
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [pool.submit(prepare_reference, **options) for _ in range(2)]
        first, second = [future.result() for future in outcomes]
    np.testing.assert_array_equal(first[0]['heads'], second[0]['heads'])
    assert {first[1]['reused'], second[1]['reused']} == {True, False}
    assert first[1]['native_single_run_seconds'] == second[1]['native_single_run_seconds']
    assert list(tmp_path.glob('*.attempt-*')) == []
