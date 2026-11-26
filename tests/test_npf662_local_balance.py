"""A high-flow CHD reach must not set the storage or GHB tolerance in another aquifer."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.config import NP_FLOAT
from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference

pytestmark = pytest.mark.skipif(NP_FLOAT != np.float64, reason='Pinned NPF requires FP64')


def aquifer_model(device, *, connected, steady=False):
    shape = (1, 4, 3)
    active = np.zeros(shape, dtype=int)
    active[0, 0, :] = 1
    active[0, 2:, 1] = 1
    active[0, 1, 1] = int(connected)
    fixed = np.zeros_like(active)
    fixed[0, 0, [0, 2]] = 1
    initial = np.full(shape, 10.0)
    initial[0, 0, :] = [11.0, 10.0, 9.0]
    conductivity = np.full(shape, 100.0)
    conductivity[0, 1, 1] = 1.0e-6
    source = np.zeros(shape)
    ghb = {}
    if steady:
        conductance = np.zeros(shape)
        conductance[0, 2:, 1] = 8.0e-7
        ghb = dict(ghb_conductance=conductance, ghb_head=20.0)
    else:
        source[0, 2:, 1] = 8.0e-6
    model = WarpDarcySolver3D(nx=3, ny=4, nz=1, dx=10.0, dy=10.0, dz=10.0,
        device=device, implementation='fast', formulation='mf6_6_6_2_npf')
    model.build_from_K_fields(kx_field=conductivity, ky_field=conductivity, kz_field=conductivity,
        active=active, bc_mask=fixed, bc_values=initial*fixed, rhs=source, initial_head=initial, **ghb)
    return model, initial


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('connected', [False, True])
def test_storage_in_low_flow_cells(device, connected):
    model, initial = aquifer_model(device, connected=connected)
    with model:
        case = case_from_model(model=model, top=10.0)
        reference = NPF662TransientReference(case=case, previous=initial, ss=1.0e-5, sy=0.0, dt=365.0)
        checks, initial_fields = reference.evaluate(head=initial)
        assert not checks['independent_accepted']
        assert not checks['local_balance_valid']
        heads, info = model.solve_transient_npf(source_factors=[1.0]*3, dt=365.0,
            head_prev=initial, ss=1.0e-5, sy=0.0)
    assert info['converged'] and info['accepted_steps'] == 3
    # Closed case follows Q*dt/(Ss*V); connected values are the native MF6
    # 6.6.2 solution of the four free-cell equations with the active aquitard.
    expected = [10.232356995, 10.425394414, 10.589075225] if connected else [10.292, 10.584, 10.876]
    for timestep, cell_head in enumerate(expected):
        np.testing.assert_allclose(heads[timestep, 0, 2:, 1], cell_head, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('preconditioner', ['diagonal', 'kcycle'])
def test_steady_weak_ghb_datum(device, preconditioner):
    model, initial = aquifer_model(device, connected=False, steady=True)
    with model:
        reference = NPF662Reference(case=case_from_model(model=model, top=10.0))
        checks, initial_fields = reference.evaluate(head=initial)
        assert not checks['independent_accepted']
        head, info = model.solve(npf_preconditioner=preconditioner)
    assert info['converged']
    np.testing.assert_allclose(head[0, 2:, 1], 20.0, rtol=0.0, atol=1.0e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('steady', [False, True])
def test_common_head_shift_with_opposing_large_wells(device, steady):
    # The initial 1 m gradient exactly carries the opposing 1000 m3/day wells.
    # A small net injection shifts the common head, even though both rows still
    # pass a residual test relative to the flow through this same pair of cells.
    initial = np.array([[[11.0, 10.0]]])
    conductivity = np.full_like(initial, 100.0)
    source = np.array([[[1000.0, -1000.0]]])
    if steady:
        source[0, 0, 1] += 8.0e-6
        boundary = dict(ghb_conductance=np.array([[[8.0e-7, 0.0]]]), ghb_head=11.0)
        solve = dict(npf_preconditioner='kcycle')
        left = 11.0 + source.sum() / 8.0e-7
        expected = np.array([[[left, left + source[0, 0, 1] / 1000.0]]])
    else:
        source += 8.0e-6
        boundary = {}
        solve = dict(transient=True, head_prev=initial, dt=365.0, ss=1.0e-5, sy=0.0, npf_device_native=True)
        expected = initial + 0.292
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=10.0, dy=10.0, dz=10.0,
            device=device, implementation='fast', formulation='mf6_6_6_2_npf') as model:
        model.build_from_K_fields(kx_field=conductivity, ky_field=conductivity, kz_field=conductivity,
            active=np.ones_like(initial), bc_mask=np.zeros_like(initial), bc_values=np.zeros_like(initial),
            rhs=source, initial_head=initial, **boundary)
        case = case_from_model(model=model, top=10.0)
        if steady:
            reference = NPF662Reference(case=case)
        else:
            reference = NPF662TransientReference(case=case, previous=initial, ss=1.0e-5, sy=0.0, dt=365.0)
        checks, initial_fields = reference.evaluate(head=initial)
        assert not checks['independent_accepted']
        head, info = model.solve(**solve)
        assert info['converged']
        np.testing.assert_allclose(head, expected, rtol=0.0, atol=1.0e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_cancelling_well_entries_cannot_relax_the_solved_equation(device):
    """Gross WEL reporting cannot accept the deliberately wrong initial head."""
    shape = (1, 1, 1)
    initial = np.ones(shape)
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1., dy=1., dz=1., device=device,
            solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf') as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=np.ones(shape), bc_mask=np.zeros(shape),
            bc_values=np.zeros(shape), initial_head=initial,
            ghb_conductance=np.ones(shape), ghb_head=np.zeros(shape),
            riv_entries=[((0, 0, 0), 1., 0., -1.)],
            wel_entries=[((0, 0, 0), 1.e9), ((0, 0, 0), -1.e9)])
        head, info = model.solve()
        assert info['converged']
        assert info['krylov_iterations'] > 0
        assert info['device_final_metrics']['gross_throughput'] >= 2.e9
        assert info['device_final_metrics']['equation_throughput'] < 1.e-10
        np.testing.assert_allclose(head, np.zeros(shape), atol=1.e-12, rtol=0.)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('steady', [False, True])
def test_permanently_inactive_drain_does_not_change_weak_component(device, steady):
    """An allocated DRN contributes restraint only when its derivative is active."""
    model, initial = aquifer_model(device=device, connected=False, steady=steady)
    with model:
        model.update_drn_entries(entries=[((0, 2, 1), 1., 1000.)])
        if steady:
            head, info = model.solve()
            expected = 20.
        else:
            heads, info = model.solve_transient_npf(source_factors=[1.], dt=365.,
                head_prev=initial, ss=1.e-5, sy=0.)
            head = heads[0]
            expected = 10.292
        assert info['converged']
        np.testing.assert_allclose(head[0, 2:, 1], expected, atol=1.e-6, rtol=0.)
