"""Gross physical throughput and prescribed-head exchange include all WEL entries."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.physics.budgets_2d import evaluate_picard_transient_balance


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('recharge_cancellation', [False, True])
def test_2d_physical_balance_keeps_gross_well_and_recharge(device, recharge_cancellation):
    """Zero net forcing still has its independently specified package throughput."""
    shape = (4, 4)
    cell = (1, 2)
    recharge = np.zeros(shape)
    rates = [(cell, 100.), (cell, -100.)]
    if recharge_cancellation:
        recharge[cell] = 1.
        rates = [(cell, -100.)]
    fixed = np.zeros(shape)
    fixed[:, 0] = 1
    head = np.full(shape, 5.)
    with WarpDarcySolver(nx=4, ny=4, dx=10., device=device) as model:
        model.build_from_fields(T_field=head, R_field=recharge, active=np.ones(shape),
            bc_mask=fixed, bc_values=head, wel_entries=rates)
        balance = evaluate_picard_transient_balance(model=model, head=head, previous=head,
            conductivity=np.ones(shape), bottom=np.zeros(shape), top=np.full(shape, 10.),
            min_sat=.1, dt=1., sy=.2, ss=0., storage_coeff=None, head_tolerance=1.e-10,
            local_head_tolerance=1.e-10)
        assert balance['accepted']
        assert balance['total_in'] == pytest.approx(100.)
        assert balance['total_out'] == pytest.approx(100.)
        assert balance['in_minus_out'] == 0.


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('rate', [-100., 100.])
@pytest.mark.parametrize('implementation', ['classic', 'fast'])
def test_cartesian_well_on_chd_has_compensating_physical_exchange(device, rate, implementation):
    """The source has no head effect but must enter the physical CHD ledger."""
    shape = (1, 1, 2)
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=1., dy=1., dz=1., device=device,
            solver='kcycle', implementation=implementation) as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
            active=np.ones(shape), bc_mask=np.array([[[1, 0]]]), bc_values=np.full(shape, 3.),
            initial_head=np.full(shape, 3.), wel_entries=[((0, 0, 0), rate)])
        head, info = model.solve()
        assert info['converged']
        assert info['budget']['chd_inflow'] == max(-rate, 0.)
        assert info['budget']['chd_outflow'] == max(rate, 0.)
        np.testing.assert_array_equal(head, np.full(shape, 3.))


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('implementation', ['classic', 'fast'])
def test_cartesian_opposing_entries_keep_physical_throughput(device, implementation):
    """The reporting scale retains entry magnitudes even when the equation source is zero."""
    shape = (1, 1, 2)
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=1., dy=1., dz=1., device=device,
            solver='kcycle', implementation=implementation) as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
            active=np.ones(shape), bc_mask=np.array([[[1, 0]]]), bc_values=np.full(shape, 3.),
            initial_head=np.full(shape, 3.), wel_entries=[((0, 0, 1), 100.), ((0, 0, 1), -100.)])
        unused_head, info = model.solve()
        assert info['converged']
        assert info['budget']['throughput'] == pytest.approx(200.)
        assert info['budget']['total_physical_throughput'] == pytest.approx(200.)


def test_gross_reporting_does_not_relax_cartesian_conservation():
    """Large cancelling inputs cannot turn an unresolved face flow into acceptance."""
    from DARCY_WARP_PACKAGE.physics.budget_3d import (
        compute_3d_flow_budget, include_gross_source_throughput, is_3d_budget_conserved,
    )
    shape = (1, 1, 2)
    positive = np.array([[[1., 0.]]])
    negative = np.array([[[0., 1.]]])
    zero = np.zeros(shape)
    fixed = np.array([[[1, 0]]])
    budget = compute_3d_flow_budget(head=np.array([[[3., 3.01]]]), source=zero,
        faces=(positive, negative, zero, zero, zero, zero), active=np.ones(shape), bc_mask=fixed)
    include_gross_source_throughput(budget=budget, source=zero,
        gross_source=np.array([[[0., 2.e12]]]), active=np.ones(shape), fixed=fixed)
    assert budget['relative_cell_imbalance'] < 1.e-12
    assert not is_3d_budget_conserved(budget=budget, absolute_tolerance=1.e-12, relative_tolerance=1.e-3)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('implementation', ['classic', 'fast'])
def test_cartesian_chd_preserves_raw_source_convention_with_explicit_well(device, implementation):
    """Raw RHS on CHD remains excluded; only the explicit WEL is compensated."""
    shape = (2, 3, 4)
    cell = (1, 2, 3)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[cell] = 1
    raw = np.zeros(shape)
    raw[cell] = 50.
    with WarpDarcySolver3D(nx=4, ny=3, nz=2, dx=1., dy=1., dz=1., device=device,
            solver='kcycle', implementation=implementation) as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
            active=np.ones(shape), bc_mask=fixed, bc_values=np.full(shape, 3.),
            initial_head=np.full(shape, 3.), rhs=raw)
        head, info = model.solve()
        assert info['converged']
        assert info['budget']['boundary_net'] == 0.
        model.update_wel(entries=[(cell, -100.)])
        head, info = model.solve()
        assert info['converged']
        assert info['budget']['chd_inflow'] == 100.
        assert info['budget']['source_all_total'] == -100.
        assert info['budget']['source_all_gross'] == 100.
        np.testing.assert_array_equal(head, np.full(shape, 3.))


def test_two_dimensional_isolated_components_keep_independent_gross_sources():
    """Physical storage restrains two disconnected cells with balanced entry pairs."""
    shape = (1, 2)
    head = np.full(shape, 5.)
    with WarpDarcySolver(nx=2, ny=1, dx=10., device='cpu') as model:
        model.build_from_fields(T_field=np.zeros(shape), R_field=0., active=np.ones(shape),
            bc_mask=np.zeros(shape), bc_values=head, wel_entries=[((0, 0), 100.), ((0, 0), -100.),
                ((0, 1), 200.), ((0, 1), -200.)])
        balance = evaluate_picard_transient_balance(model=model, head=head, previous=head,
            conductivity=np.zeros(shape), bottom=np.zeros(shape), top=np.full(shape, 10.),
            min_sat=.1, dt=1., sy=.2, ss=0., storage_coeff=None, head_tolerance=1.e-10,
            local_head_tolerance=1.e-10)
        assert balance['accepted'] and balance['component_count'] == 2
        assert balance['total_in'] == balance['total_out'] == 300.
        assert [item['throughflow'] for item in balance['component_balances']] == [100., 200.]


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('rate', [-100., 100.])
def test_two_dimensional_newton_chd_budget_absorbs_the_well(device, rate):
    """Accepted Newton output retains WEL and its physical prescribed-head compensation."""
    shape = (1, 2)
    initial = np.full(shape, 5.)
    with WarpDarcySolver(nx=2, ny=1, dx=10., device=device) as model:
        model.build_from_fields(T_field=initial, R_field=0., active=np.ones(shape),
            bc_mask=np.array([[1, 0]]), bc_values=initial, wel_entries=[((0, 0), rate)])
        head, info = model.solve(solver='unconfined_semismooth_newton_kcycle', formulation='unconfined',
            K_field=np.ones(shape), zbot_field=np.zeros(shape), ztop_field=np.full(shape, 10.),
            initial_head=initial)
        assert info['converged']
        budget = info['budget_summary']
        assert budget['chd_in'] == max(-rate, 0.)
        assert budget['chd_out'] == max(rate, 0.)
        assert budget['wel_in'] == max(rate, 0.)
        assert budget['wel_out'] == max(-rate, 0.)
        np.testing.assert_array_equal(head, initial)
