"""Independent one-cell checks for native NPF DRN/RIV equations."""
import numpy as np

from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case, NPF662Reference


def single_cell_case(*, head, source, drn=None, riv=None):
    """Build a closed one-cell NPF case with optional integrated boundaries."""
    shape = (1, 1, 1)
    options = {}
    if drn is not None:
        options.update(drn_conductance=drn[0], drn_elevation=drn[1])
    if riv is not None:
        options.update(riv_conductance=riv[0], riv_stage=riv[1], riv_bottom=riv[2])
    case = NPF662Case(
        kx=np.ones(shape), ky=np.ones(shape), kz=np.ones(shape),
        active=np.ones(shape, dtype=np.int32), fixed=np.zeros(shape, dtype=np.int32),
        values=np.zeros(shape), source=np.full(shape, source), initial=np.full(shape, head),
        dx=1.0, dy=1.0, dz=1.0, **options,
    )
    case.validate()
    return case


def test_npf_reference_separates_drains_and_gaining_rivers():
    """Check direct signed flows, cell balance, and gross package accounting."""
    head = 4.0
    drain_rate = 2.0 * (head - 1.0)
    river_rate = 3.0 * (head - 5.0)
    reference = NPF662Reference(case=single_cell_case(
        head=head, source=drain_rate + river_rate,
        drn=(2.0, 1.0), riv=(3.0, 5.0, 2.0)))

    checks, fields = reference.evaluate(head=np.full((1, 1, 1), head))

    assert checks["independent_accepted"]
    assert checks["drn_inflow"] == 0.0
    assert checks["drn_outflow"] == drain_rate
    assert checks["drn_net"] == -drain_rate
    assert checks["riv_inflow"] == -river_rate
    assert checks["riv_outflow"] == 0.0
    assert checks["riv_net"] == -river_rate
    np.testing.assert_array_equal(fields["drn_outflow"], [[[drain_rate]]])
    np.testing.assert_array_equal(fields["riv_outflow"], [[[river_rate]]])


def test_npf_reference_uses_zero_generalized_derivative_at_thresholds():
    """Check exact branch derivatives at both DRN and RIV switching heads."""
    shape = (1, 1, 1)
    drain = NPF662Reference(case=single_cell_case(head=1.0, source=0.0, drn=(2.0, 1.0)))
    river = NPF662Reference(case=single_cell_case(head=2.0, source=0.0, riv=(3.0, 4.0, 2.0)))

    assert drain.jacobian(head=np.ones(shape)).toarray()[0, 0] == 0.0
    assert river.jacobian(head=np.full(shape, 2.0)).toarray()[0, 0] == 0.0
    assert drain.evaluate(head=np.ones(shape))[0]["ungrounded_current_components"] == 1
    assert river.evaluate(head=np.full(shape, 2.0))[0]["ungrounded_current_components"] == 1


def test_npf_reference_residual_preserves_small_flux_at_large_datum():
    """Check a 0.125 head difference at a 1e12 datum without C*h subtraction."""
    base = 1.0e12
    head = base + 0.125
    rate = 3.1 * (head - base)
    reference = NPF662Reference(case=single_cell_case(
        head=head, source=rate, riv=(3.1, base, base - 1.0)))

    checks, fields = reference.evaluate(head=np.full((1, 1, 1), head))

    assert checks["independent_accepted"]
    assert fields["riv_outflow"].item() == rate
    assert fields["residual"].item() == 0.0


def test_npf_reference_allows_transient_threshold_crossing_from_storage():
    """Check storage can restrain a cell while a drain changes branch."""
    from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference

    case = single_cell_case(head=0.0, source=2.0, drn=(1.0, 1.0))
    reference = NPF662TransientReference(case=case, previous=np.zeros((1, 1, 1)),
        dt=1.0, ss=1.0, sy=0.0)
    checks, fields = reference.evaluate(head=np.ones((1, 1, 1)))

    assert fields["storage_rate"].item() == 1.0
    assert fields["drn_outflow"].item() == 0.0
    assert fields["residual"].item() == 1.0
    assert checks["ungrounded_current_components"] == 0
