"""Hand-assembled flow and storage equations for the independent reference tools."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.drn_riv_3d import validate_boundary_entries
from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
from DARCY_WARP_PACKAGE.validation.npf662_reference import (
    NPF662LayeredCase,
    NPF662Reference,
    load_npf662_case,
    save_npf662_case,
)
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference


def layered_column():
    """Two 6 m² cells, thicknesses 2 and 4 m, vertical K of 1 and 2 m/day."""
    shape = (2, 1, 1)
    ones = np.ones(shape)
    return NPF662LayeredCase(
        kx=ones.copy(), ky=ones.copy(), kz=np.array([1., 2.]).reshape(shape),
        active=ones.astype(int), fixed=np.array([1, 0]).reshape(shape),
        values=10 * ones, source=np.zeros(shape), initial=10 * ones,
        dx=2., dy=3., z_interfaces=np.array([9., 7., 3.]).reshape(3, 1, 1),
    )


def test_exact_dry_zero_row_checks_the_unmodified_equation():
    """A dry isolated zero row can coexist with wet, grounded cells."""
    shape = (1, 1, 3)
    ones = np.ones(shape)
    head = np.array([[[0.5, 0.5, 2.0]]])
    case = NPF662LayeredCase(kx=ones, ky=ones, kz=ones, active=ones.astype(int),
        fixed=np.array([[[1, 0, 0]]]), values=head, source=np.zeros(shape), initial=head,
        dx=1., dy=1., z_interfaces=np.array([[[1., 1., 3.]], [[0., 0., 2.]]]), cell_type=1)
    forced = replace(case, source=np.array([[[0., 0., 1.]]]))
    # The three acceptance paths must agree on the physical zero equation.
    # STO is zero at this endpoint, but a previous wet state must not be
    # exempt merely because the current cell is dry.
    for evaluator in (NPF662Reference(case=case), NPF662MatrixFreeReference(case=case),
                      NPF662TransientReference(case=case, previous=head, dt=1., ss=0.1, sy=0.2)):
        accepted, _ = evaluator.evaluate(head=head)
        assert accepted["independent_accepted"]
        assert accepted["exact_dry_zero_rows"] == 1
    for evaluator in (NPF662Reference(case=forced), NPF662MatrixFreeReference(case=forced),
                      NPF662TransientReference(case=forced, previous=head, dt=1., ss=0.1, sy=0.2)):
        rejected, fields = evaluator.evaluate(head=head)
        assert not rejected["independent_accepted"]
        assert rejected["exact_dry_zero_rows"] == 0
        assert fields["residual"][0, 0, 2] == 1.
    previous_wet = head.copy()
    previous_wet[0, 0, 2] = 2.5
    draining = NPF662TransientReference(case=case, previous=previous_wet,
                                        dt=1., ss=0.1, sy=0.2)
    rejected, _ = draining.evaluate(head=head)
    assert not rejected["independent_accepted"]
    assert rejected["exact_dry_zero_rows"] == 0


    # A saturated singleton with no active face, source, boundary, or storage
    # term is also an exact zero equation and needs no hydraulic datum.
    isolated_shape = (1, 1, 2)
    isolated = NPF662LayeredCase(
        kx=np.ones(isolated_shape), ky=np.ones(isolated_shape),
        kz=np.ones(isolated_shape), active=np.array([[[1, 0]]]),
        fixed=np.zeros(isolated_shape, dtype=int), values=np.zeros(isolated_shape),
        source=np.zeros(isolated_shape), initial=np.array([[[1.5, 0.]]]),
        dx=1., dy=1., z_interfaces=np.array([[[2., 2.]], [[0., 0.]]]),
        cell_type=1)
    isolated_head = isolated.initial.copy()
    for evaluator in (NPF662Reference(case=isolated),
                      NPF662MatrixFreeReference(case=isolated),
                      NPF662TransientReference(case=isolated, previous=isolated_head,
                          dt=1., ss=0., sy=0.)):
        checks, unused = evaluator.evaluate(head=isolated_head)
        assert checks['independent_accepted']
        assert checks['exact_dry_zero_rows'] == 1


def test_balanced_ungrounded_component_uses_initial_mean_as_head_datum():
    case = layered_column()
    case.fixed[:] = 0
    case.initial[:] = 10.
    head = np.full(case.active.shape, 20.)
    evaluators = (NPF662Reference(case=case), NPF662MatrixFreeReference(case=case))
    for evaluator in evaluators:
        checks, unused = evaluator.evaluate(head=head)
        assert checks['independent_accepted']
        assert checks['ungrounded_current_components'] == 0
        assert checks['balanced_nullspace_components'] == 1
    gauged, records = evaluators[0].apply_nullspace_datum(head=head)
    np.testing.assert_array_equal(gauged, case.initial)
    assert len(records) == 1
    assert records[0]['applied_shift_m'] == -10.

    balanced_case = layered_column()
    balanced_case.fixed[:] = 0
    balanced_case.source[:, 0, 0] = [3., -3.]
    balanced_head = np.array([[[10.5]], [[9.5]]])
    for evaluator in (NPF662Reference(case=balanced_case),
                      NPF662MatrixFreeReference(case=balanced_case)):
        checks, unused = evaluator.evaluate(head=balanced_head)
        assert checks['independent_accepted']
        assert checks['balanced_nullspace_components'] == 1

    transient = NPF662TransientReference(case=case, previous=case.initial,
        dt=1., ss=0., sy=0.)
    checks, unused = transient.evaluate(head=head)
    assert checks['independent_accepted']
    assert checks['balanced_nullspace_components'] == 1

    stored = NPF662TransientReference(case=case, previous=case.initial,
        dt=1., ss=0.1, sy=0.2)
    checks, unused = stored.evaluate(head=head)
    assert checks['balanced_nullspace_components'] == 0
    assert checks['ungrounded_current_components'] == 0
    assert checks['storage_accumulation_m3_day'] != 0.


def test_legacy_archives_replay_the_confined_gradient_and_reservoir_exchange():
    archived = Path(__file__).parent / 'fixtures' / 'npf662'
    implicit = load_npf662_case(path=archived / 'experiments_v1.npz')
    explicit = load_npf662_case(path=archived / 'final_experiments_v1.npz')
    # Older exports omitted ICELLTYPE. Both archives describe the same aquifer.
    for field in ('active', 'fixed', 'kx', 'ky', 'kz', 'dx', 'dy', 'dz', 'top',
                  'values', 'source', 'initial', 'cell_types'):
        np.testing.assert_array_equal(getattr(implicit, field), getattr(explicit, field))
    head, checks, fields = NPF662Reference(case=implicit).solve()
    assert checks['independent_accepted']
    expected = np.broadcast_to(np.linspace(8., 9., 7), (3, 5, 7))
    np.testing.assert_allclose(head, expected, rtol=0., atol=1e-10)
    # 1 m head difference over 12 m drives 1 m³/day across each 12 m² face.
    reservoir = fields['fixed_reservoir_required_inflow']
    np.testing.assert_allclose(reservoir[:, :, 0], -1., rtol=0., atol=1e-10)
    np.testing.assert_allclose(reservoir[:, :, -1], 1., rtol=0., atol=1e-10)


def test_matrix_free_audit_detects_equal_and_opposite_local_residuals():
    case = layered_column()
    case.fixed[:] = 0
    case.ghb_conductance = np.ones_like(case.source)
    case.ghb_head = case.values.copy()
    case.source[:, 0, 0] = [1., -1.]
    checks, fields = NPF662MatrixFreeReference(case=case).evaluate(head=case.initial)
    np.testing.assert_array_equal(fields['residual'].ravel(), [1., -1.])
    assert not checks['independent_accepted']


def test_matrix_free_river_has_constant_flow_below_bed():
    case = layered_column()
    cell = (1, 0, 0)
    case.riv_entries = [(cell, 2., 9., 8.)]
    head = np.full_like(case.values, 7.)
    _, fields = NPF662MatrixFreeReference(case=case).evaluate(head=head)
    assert fields['riv_outflow'][cell] == pytest.approx(-2.)
    # Only the vertical face (C = 3) contributes a slope below the bed.
    assert fields['principal_diagonal'][cell] == pytest.approx(3.)


def test_newton_jacobian_differentiates_donor_saturation_and_river_slope():
    shape = (1, 1, 2)
    ones = np.ones(shape)
    case = NPF662LayeredCase(
        kx=ones.copy(), ky=ones.copy(), kz=ones.copy(), active=ones.astype(int),
        fixed=np.array([1, 0]).reshape(shape), values=np.array([2., 1.]).reshape(shape),
        initial=np.array([2., 3.]).reshape(shape), source=0 * ones, dx=1., dy=1.,
        z_interfaces=np.array([[[4., 4.]], [[0., 0.]]]), cell_type=1,
        riv_entries=[((0, 0, 1), 2., 3., 1.)],
    )
    auditor = NPF662Reference(case=case)
    direction = np.array([0., 1.]).reshape(shape)
    step = 1e-5
    _, plus = auditor.evaluate(head=case.initial + step * direction)
    _, minus = auditor.evaluate(head=case.initial - step * direction)
    derivative = -(plus['residual'] - minus['residual']) / (2 * step)
    jacobian_column = auditor.jacobian(head=case.initial) @ direction.ravel()
    # Saturation increases with free-cell head, so freezing it gives the wrong slope.
    assert jacobian_column[1] > 5.5
    assert jacobian_column[1] == pytest.approx(derivative[0, 0, 1], rel=1e-8)


def test_confined_storage_uses_volume_and_ignores_specific_yield():
    case = layered_column()
    audit = NPF662TransientReference(case=case, previous=case.initial, dt=4., ss=.01, sy=.3)
    storage, tangent = audit.storage(head=case.initial + 2.)
    # Lower cell volume is 24 m³. Elastic accumulation is Ss V dh / dt.
    assert storage[1, 0, 0] == pytest.approx(.12)
    assert tangent[1, 0, 0] == pytest.approx(.06)


def test_convertible_gravity_storage_stops_at_layer_top():
    case = layered_column()
    case.cell_type = 1
    previous = np.array([10., 5.]).reshape(case.active.shape)
    auditor = NPF662TransientReference(case=case, previous=previous, dt=2., ss=0., sy=.25)
    storage, tangent = auditor.storage(head=case.initial)
    # Water table rises 5 -> 10 m, but reaches this layer's top at 7 m.
    assert storage[1, 0, 0] == pytest.approx(6. * .25 * 2. / 2., abs=2e-5)
    assert tangent[1, 0, 0] == 0.


def test_sparse_archive_restores_individual_drains_and_their_fluxes(tmp_path):
    case = layered_column()
    case.drn_entries = [((1, 0, 0), 2., 6.), ((1, 0, 0), 5., 11.)]
    path = tmp_path / 'drains.npz'
    save_npf662_case(case=case, path=path)
    replay = load_npf662_case(path=path)
    assert replay.digest() == case.digest()
    entries, terms, _, _ = NPF662MatrixFreeReference(case=replay).boundary_terms(head=case.initial)
    np.testing.assert_array_equal(entries, [8., 0.])
    assert terms[0][1, 0, 0] == 8.


def test_disabling_the_last_drain_can_be_serialized_and_replayed(tmp_path):
    case = layered_column()
    case.drn_entries = []
    case.validate()
    path = tmp_path / 'disabled.npz'
    save_npf662_case(case=case, path=path)
    replay = load_npf662_case(path=path)
    _, fields = NPF662Reference(case=replay).evaluate(head=case.initial)
    np.testing.assert_array_equal(fields['residual'], np.zeros_like(case.source))
    assert replay.digest() == case.digest()


def test_revalidating_owned_entries_cannot_move_a_fractional_cell():
    case = layered_column()
    entries = validate_boundary_entries(entries=[((1, 0, 0), 1., 6.)], package='DRN',
        shape=case.active.shape, active=case.active, fixed=case.fixed)
    # Truncating 1.5 to 1 would attach the package to a valid, different layer.
    case.drn_entries = replace(entries, cells=np.array([[1.5, 0., 0.]]))
    with pytest.raises(ValueError):
        case.validate()


def test_an_inactive_gap_has_no_vertical_bypass():
    case = layered_column()
    shape = (3, 1, 1)
    ones = np.ones(shape)
    case = replace(case, kx=ones, ky=ones, kz=ones, active=np.array([1, 0, 1]).reshape(shape),
        fixed=np.array([1, 0, 0]).reshape(shape), values=10 * ones, source=0 * ones,
        initial=10 * ones, z_interfaces=np.array([9., 7., 5., 3.]).reshape(4, 1, 1),
        ghb_conductance=np.array([0., 0., 2.]).reshape(shape), ghb_head=4 * ones)
    head, checks, fields = NPF662Reference(case=case).solve()
    assert checks['independent_accepted']
    assert fields['flux'].size == 0
    np.testing.assert_allclose(head.ravel()[[0, 2]], [10., 4.], atol=1e-12, rtol=0.)
