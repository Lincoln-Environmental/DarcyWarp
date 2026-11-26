"""Direct reproductions of the November sparse-package/source audit."""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.physics.wel import validate_well_entries, well_entries
from DARCY_WARP_PACKAGE.solvers import npf662_kernels
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from test_3d_wel import _anchored_model


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_raw_source_survives_repeated_well_replacement(device):
    """Raw +50 remains owned while WEL changes and is removed."""
    raw = np.zeros((1, 3, 3))
    raw[0, 1, 1] = 50.
    with _anchored_model(device=device, rhs=raw) as model:
        for rate, expected in [(-100., -50.), (-100., -50.), (-200., -150.), (None, 50.)]:
            entries = None if rate is None else [((0, 1, 1), rate)]
            model.update_wel(entries=entries)
            assert model._session.rhs[0, 1, 1] == expected
            assert model._volumetric_source[0, 1, 1] == 50.
            assert model.physical_source_budgets()['volumetric']['net'] == 50.
        np.testing.assert_array_equal(raw, model._volumetric_source)


@pytest.mark.parametrize('cell', [(True, 0), (0.5, 0), (np.bool_(True), 0), (0,), (0, 0, 0, 0)])
def test_well_helper_rejects_invalid_raw_coordinates(cell):
    """The convenience helper must not truncate or coerce coordinates."""
    with pytest.raises(ValueError):
        well_entries(*[cell], rates=[1.])


@pytest.mark.parametrize('rate', [[1.], [[1.]], float('inf'), 1.e100])
def test_well_rates_are_scalar_and_representable(rate):
    """Reject nested rates and destination-precision overflow."""
    with pytest.raises(ValueError):
        validate_well_entries(entries=[((0, 0), rate)], shape=(1, 1),
                              active=np.ones((1, 1)), dtype=np.float32)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('package', ['drn', 'riv'])
def test_native_activation_at_large_flat_cell_index(device, package):
    """An isolated boundary in layer 2 uses entry indices for its properties."""
    shape = (3, 3, 4)
    cell = (2, 2, 3)
    active = np.zeros(shape, dtype=np.int32)
    active[cell] = 1
    options = ({'drn_entries': [(cell, 1., 2.)]} if package == 'drn'
               else {'riv_entries': [(cell, 2., 3., 1.)]})
    source = np.zeros(shape)
    source[cell] = 1. if package == 'drn' else 0.
    with WarpDarcySolver3D(nx=4, ny=3, nz=3, dx=1., dy=1., dz=1., device=device,
                          solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf') as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=active, bc_mask=np.zeros(shape, dtype=np.int32),
            bc_values=np.zeros(shape), rhs=source, initial_head=np.zeros(shape), **options)
        head, info = model.solve(transient=True, dt=1., head_prev=np.zeros(shape), ss=0., sy=0.,
            npf_device_native=True, npf_preconditioner='kcycle', npf_krylov_method='fgmres')
        assert info['converged'], info
        assert info['npf_preconditioner'] == 'kcycle'
        assert info['preconditioner_cycles'] > 0
        assert head[cell] == pytest.approx(3., abs=1e-12)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_sparse_previous_restraint_diagonal(device):
    """Each independently switching CSR entry contributes its own derivative."""
    from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace
    shape = (2, 2, 3)
    cell = (1, 1, 2)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    with WarpDarcySolver3D(nx=3, ny=2, nz=2, dx=1., dy=1., dz=1., device=device,
                          solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf') as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32), bc_mask=fixed,
            bc_values=np.zeros(shape), initial_head=np.full(shape, 4.),
            drn_entries=[(cell, 2., 1.), (cell, 7., 5.)],
            riv_entries=[(cell, 3., 6., 2.), (cell, 11., 8., 5.)])
        case = case_from_model(model=model, top=0., cell_type=0)
        workspace = NPF662Workspace(model=model, case=case, restart=4, krylov_budget=20)
        try:
            workspace.refresh_previous_operator(activation_head=workspace.head)
            expected = np.zeros(shape)
            expected[cell] = 5.
            np.testing.assert_array_equal(workspace.previous_ghb_operator.numpy().reshape(shape), expected)
        finally:
            workspace.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_all_source_component_transitions(device):
    """All eight component combinations retain independent fields and budgets."""
    from itertools import product
    shape = (1, 3, 3)
    cell = (0, 1, 1)
    with _anchored_model(device=device, rhs=np.zeros(shape)) as model:
        for raw_on, recharge_on, well_on in product((False, True), repeat=3):
            for repeat in range(2):
                raw = np.zeros(shape)
                raw[cell] = 50. if raw_on else 0.
                model.update_rhs(rhs=raw)
                model.update_R_in_place(R_field=0.01 if recharge_on else None)
                model.update_wel(entries=[(cell, -100.)] if well_on else None)
                expected = raw.copy()
                if recharge_on:
                    expected[:, :, 1:] += 4.
                if well_on:
                    expected[cell] -= 100.
                np.testing.assert_array_equal(model._session.rhs, expected)
                budgets = model.physical_source_budgets()
                assert budgets['volumetric']['net'] == (50. if raw_on else 0.)
                assert budgets['wel']['net'] == (-100. if well_on else 0.)
                assert budgets['recharge']['net'] == (24. if recharge_on else 0.)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_two_dimensional_invalid_updates_are_atomic(device):
    """Invalid recharge or WEL cannot alter owned inputs or device forcing."""
    from test_2d_wel import _anchored_2d
    model = _anchored_2d(device=device)
    try:
        model.update_wel_in_place(wel_entries=[((2, 2), -100.)])
        entries = model.wel_entries
        source = model.R_field_host.copy()
        recharge = model._recharge_input_host.copy()
        device_source = model.R_wp.numpy().copy()
        for value in (np.nan, np.inf, 1.e309):
            with pytest.raises(ValueError):
                model.update_uniform_recharge_in_place(recharge_rate=value)
            with pytest.raises(ValueError):
                model.update_R_in_place(R_field=value)
            with pytest.raises(ValueError):
                model.update_wel_in_place(wel_entries=[((2, 2), value)])
            assert model.wel_entries is entries
            np.testing.assert_array_equal(model.R_field_host, source)
            np.testing.assert_array_equal(model._recharge_input_host, recharge)
            np.testing.assert_array_equal(model.R_wp.numpy(), device_source)
    finally:
        model.close()


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_historical_native_audit_owns_sparse_equation(device):
    """A later WEL update cannot rewrite an already solved transient equation."""
    shape = (1, 3, 3)
    with _anchored_model(device=device, well=-1.) as model:
        head, info = model.solve(transient=True, dt=2., head_prev=np.zeros(shape), ss=.1, sy=0.,
            npf_device_native=True, npf_preconditioner='kcycle')
        assert info['converged']
        state = model._session._npf662_workspace_holder['native_state']
        expected = state.case.source.copy()
        model.update_wel(entries=[((0, 1, 1), -1000.)])
        np.testing.assert_array_equal(state.independent().case.source, expected)
        checks, unused = state.evaluate(head=head)
        assert checks['independent_accepted']


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('mutation', ['head', 'cell_rate', 'entry_rate', 'foreign_model', 'dt', 'rejected'])
def test_accepted_exchange_rejects_unowned_metadata(device, mutation):
    """Caller metadata cannot replace a session's historical physical outputs."""
    from copy import deepcopy
    from test_3d_drn_riv_entries import _one_cell_model
    with _one_cell_model(device=device, drn_entries=[((0, 0, 0), 1., 2.)]) as model:
        head, info = model.solve(npf_preconditioner='kcycle')
        assert info['converged']
        supplied = deepcopy(info)
        altered_head = head.copy()
        if mutation == 'head':
            altered_head += 1.
        elif mutation == 'cell_rate':
            supplied['drn_discharge_rate_array'] += 1.
        elif mutation == 'entry_rate':
            supplied['drn_entry_rates'] += 1.
        elif mutation == 'dt':
            supplied['accepted_state']['dt'] = 999.
        elif mutation == 'rejected':
            supplied['converged'] = False
        if mutation == 'foreign_model':
            with _one_cell_model(device=device, drn_entries=[((0, 0, 0), 1., 2.)]) as foreign:
                with pytest.raises((RuntimeError, ValueError)):
                    foreign.accepted_drn_discharge(head=altered_head, info=supplied)
        else:
            with pytest.raises((RuntimeError, ValueError)):
                model.accepted_drn_discharge(head=altered_head, info=supplied)
        expected = model.accepted_drn_discharge(head=head, info=info)
        model.update_rhs(rhs=np.full((1, 1, 1), 2.))
        np.testing.assert_array_equal(model.accepted_drn_discharge(head=head, info=info), expected)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_opposing_river_entries_keep_gross_exchange(device):
    """Two nonzero opposing river entries retain gross package throughput."""
    from test_3d_drn_riv_entries import _one_cell_model
    with _one_cell_model(device=device,
            riv_entries=[((0, 0, 0), 2., 2., 1.), ((0, 0, 0), 2., 4., 1.)]) as model:
        model.update_rhs(rhs=np.zeros((1, 1, 1)))
        head, info = model.solve(npf_preconditioner='kcycle')
        assert info['converged']
        assert head.item() == pytest.approx(3., abs=1e-12)
        assert info['riv_budget']['inflow'] == pytest.approx(2.)
        assert info['riv_budget']['outflow'] == pytest.approx(2.)
        assert info['device_final_metrics']['gross_throughput'] == pytest.approx(4.)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('rate', [-100., 100.])
def test_well_on_prescribed_head_enters_chd_ledger(device, rate):
    """CHD must supply an imposed abstraction or remove an imposed injection."""
    shape = (1, 1, 1)
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1., dy=1., dz=1., device=device,
            solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf') as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32),
            bc_mask=np.ones(shape, dtype=np.int32), bc_values=np.full(shape, 3.),
            initial_head=np.full(shape, 3.), wel_entries=[((0, 0, 0), rate)])
        head, info = model.solve(npf_preconditioner='kcycle')
        assert info['converged']
        assert info['chd_budget']['chd_inflow'] == max(-rate, 0.)
        assert info['chd_budget']['chd_outflow'] == max(rate, 0.)
