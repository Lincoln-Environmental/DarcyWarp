"""Exact WEL/recharge balances asserting the maintained confined execution paths."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('implementation', ['classic', 'fast'])
@pytest.mark.parametrize('transient', [False, True])
def test_two_dimensional_confined_backend_after_source_updates(device, implementation, transient):
    """One free cell balances WEL, recharge, one face and physical storage exactly."""
    shape = (1, 2)
    fixed = np.array([[1, 0]], dtype=np.int32)
    initial = np.full(shape, 10.)
    with WarpDarcySolver(nx=2, ny=1, dx=100., device=device, solver_type='kcycle') as model:
        model.build_from_fields(T_field=np.full(shape, 20.), R_field=0., active=np.ones(shape),
            bc_mask=fixed, bc_values=initial)
        for rate, recharge in [(-100., .001), (-200., .002), (-100., .001)]:
            model.update_wel_in_place(wel_entries=[((0, 1), rate)])
            model.update_uniform_recharge_in_place(recharge_rate=recharge)
            options = dict(implementation=implementation, initial_head=initial,
                rel_tol=1.e-12, abs_tol_min=1.e-12, max_cycles=100,
                min_coarse_cells=2, nu_coarse=10, check_every_no=1)
            if transient:
                options.update(transient=True, dt=2., head_prev=initial, storage_coeff=.1)
            head, info = model.solve(**options)
            assert info['converged'], info
            assert info['solver_backend'] == 'confined_kcycle'
            expected_implementation = 'fast_transient_face_f64' if transient and implementation == 'fast' else implementation
            assert info.get('implementation', 'classic') == expected_implementation
            if transient and implementation == 'fast' and device.startswith('cuda'):
                assert info['cuda_graph_replays'] == info['n_cycles_used'] > 0
                assert info['graph_count'] > 0
                assert info['graph_fallback_count'] == 0
            denominator = 20. + (500. if transient else 0.)
            expected = 10. + (rate + recharge * 10000.) / denominator
            assert head[0, 1] == pytest.approx(expected, abs=2.e-12)
            np.testing.assert_array_equal(model.accepted_wel_rates(head=head, info=info), [rate])


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('transient', [False, True])
def test_direct_fast_cartesian_source_updates(device, transient):
    """Volumetric sources stay on direct Cartesian K-cycle rather than Newton."""
    shape = (1, 1, 2)
    initial = np.full(shape, 10.)
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=100., dy=100., dz=10.,
            device=device, solver='kcycle', implementation='fast') as model:
        model.build_from_K_fields(kx_field=np.full(shape, 2.), ky_field=np.full(shape, 2.),
            kz_field=np.full(shape, 2.), active=np.ones(shape), bc_mask=np.array([[[1, 0]]]),
            bc_values=initial, initial_head=initial)
        for rate, recharge in [(-100., .001), (-200., .002), (-100., .001)]:
            model.update_wel(entries=[((0, 0, 1), rate)])
            model.update_R_in_place(R_field=recharge)
            options = dict(rel_tol=1.e-12, abs_tol_min=1.e-12, max_cycles=100)
            if transient:
                options.update(transient=True, dt=2., head_prev=initial, storage_coeff=.1)
            head, info = model.solve(**options)
            assert info['converged'], info
            assert info['session_backend'] == 'fast_3d'
            assert 'npf_preconditioner' not in info
            denominator = 20. + (5000. if transient else 0.)
            expected = 10. + (rate + recharge * 10000.) / denominator
            assert head[0, 0, 1] == pytest.approx(expected, abs=2.e-12)
