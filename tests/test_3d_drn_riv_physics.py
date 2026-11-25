"""Analytical tests of 3D DRN/RIV physics, independent of a flow solver."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.drn_riv_3d import (
    drn_terms,
    riv_terms,
    validate_drn_fields,
    validate_riv_fields,
)


def grid(values):
    """Construct a one-row structured 3D grid."""
    return np.asarray(values, dtype=np.float64).reshape(1, 1, -1)


def domain(size):
    """Construct a free-cell domain."""
    return dict(active=np.ones((1, 1, size)), fixed=np.zeros((1, 1, size)))


def test_drn_activation_and_linearisation():
    head = grid([1, 2, 3, 4])
    c, e = validate_drn_fields(conductance=3., elevation=2., **domain(4))
    terms = drn_terms(head=head, conductance=c, elevation=e)
    np.testing.assert_array_equal(terms.discharge, grid([0, 0, 3, 6]))
    np.testing.assert_array_equal(terms.diagonal, grid([0, 0, 3, 3]))
    np.testing.assert_array_equal(terms.rhs, grid([0, 0, 6, 6]))
    np.testing.assert_array_equal(terms.diagonal * head - terms.rhs, terms.discharge)


def test_riv_disconnection_reversal_and_linearisation():
    head = grid([0, 1, 2, 3, 4])
    c, s, b = validate_riv_fields(conductance=2., stage=3., bottom=1., **domain(5))
    terms = riv_terms(head=head, conductance=c, stage=s, bottom=b)
    np.testing.assert_array_equal(terms.discharge, grid([-4, -4, -2, 0, 2]))
    np.testing.assert_array_equal(terms.diagonal, grid([0, 0, 2, 2, 2]))
    np.testing.assert_array_equal(terms.rhs, grid([4, 4, 6, 6, 6]))
    np.testing.assert_array_equal(terms.diagonal * head - terms.rhs, terms.discharge)


@pytest.mark.parametrize("package", ["DRN", "RIV"])
def test_mask_ownership_and_inactive_heads(package):
    cells = dict(active=grid([1, 0, 1]), fixed=grid([0, 0, 1]))
    mask = grid([1, 0, 0])
    head = grid([3, np.nan, 1e30])
    if package == "DRN":
        c, e = validate_drn_fields(conductance=2., elevation=1., mask=mask, **cells)
        terms = drn_terms(head=head, conductance=c, elevation=e)
    else:
        c, s, b = validate_riv_fields(conductance=2., stage=1., bottom=0.,
                                      mask=mask, **cells)
        terms = riv_terms(head=head, conductance=c, stage=s, bottom=b)
    np.testing.assert_array_equal(terms.discharge, grid([4, 0, 0]))
    np.testing.assert_array_equal(terms.diagonal, grid([2, 0, 0]))


@pytest.mark.parametrize("package", ["DRN", "RIV"])
@pytest.mark.parametrize("invalid", ["negative", "nonfinite", "shape", "mask",
                                      "inactive", "fixed", "outside", "sentinel", "overflow"])
def test_reject_invalid_fields(package, invalid):
    args = dict(conductance=grid([2, 2]), **domain(2))
    if package == "DRN":
        args["elevation"] = 2.
        elevation_name = "elevation"
        validate = validate_drn_fields
    else:
        args.update(stage=2., bottom=1.)
        elevation_name = "stage"
        validate = validate_riv_fields
    if invalid == "negative":
        args["conductance"] = -1.
    elif invalid == "nonfinite":
        args["conductance"] = np.inf
    elif invalid == "shape":
        args["conductance"] = np.ones((2, 1))
    elif invalid == "mask":
        args["mask"] = grid([1, 0.5])
    elif invalid == "inactive":
        args["active"] = grid([0, 1])
    elif invalid == "fixed":
        args["fixed"] = grid([1, 0])
    elif invalid == "outside":
        args["mask"] = grid([1, 0])
    elif invalid == "sentinel":
        args[elevation_name] = 1e20
    elif invalid == "overflow":
        args["conductance"] = 1e308
        args[elevation_name] = 10.
    with pytest.raises(ValueError):
        validate(**args)


def test_riv_requires_stage_at_or_above_bottom():
    with pytest.raises(ValueError, match="stage"):
        validate_riv_fields(conductance=1., stage=0., bottom=1., **domain(1))


def test_input_ownership_and_disabled_package():
    original = grid([1, 2])
    c, e = validate_drn_fields(conductance=original, elevation=original, **domain(2))
    original[:] = 0.
    np.testing.assert_array_equal(c, grid([1, 2]))
    np.testing.assert_array_equal(e, grid([1, 2]))
    c, e = validate_drn_fields(conductance=None, elevation=None, **domain(2))
    terms = drn_terms(head=grid([0, 0]), conductance=c, elevation=e)
    np.testing.assert_array_equal(terms.discharge, grid([0, 0]))
    c, s, b = validate_riv_fields(conductance=None, stage=None, bottom=None, **domain(2))
    terms = riv_terms(head=grid([0, 0]), conductance=c, stage=s, bottom=b)
    np.testing.assert_array_equal(terms.rhs, grid([0, 0]))


def test_overlapping_packages_keep_gross_exchange_separate():
    head = grid([3])
    c, e = validate_drn_fields(conductance=1., elevation=1., **domain(1))
    drain = drn_terms(head=head, conductance=c, elevation=e)
    c, s, b = validate_riv_fields(conductance=1., stage=5., bottom=1., **domain(1))
    river = riv_terms(head=head, conductance=c, stage=s, bottom=b)
    assert drain.discharge.item() == 2.
    assert river.discharge.item() == -2.
    assert (drain.discharge + river.discharge).item() == 0.
    assert (drain.diagonal + river.diagonal).item() == 2.


@pytest.mark.parametrize("previous,source,expected", [
    (0., 0., 1.),    # DRN remains inactive; RIV recharges below the bed.
    (1., 5.5, 2.5),  # Both boundary derivatives active.
    (4., -16., 1.),   # Cross both thresholds while draining storage.
])
def test_one_cell_backward_euler_balance(previous, source, expected):
    # Storage coefficient/dt = 4; DRN C=1,e=2; RIV C=2,s=3,b=1.
    # Analytical root of 4*(h-h_old)+q_drn(h)+q_riv(h)=source.
    c_d, e = validate_drn_fields(conductance=1., elevation=2., **domain(1))
    c_r, s, b = validate_riv_fields(conductance=2., stage=3., bottom=1., **domain(1))
    h = grid([expected])
    d = drn_terms(head=h, conductance=c_d, elevation=e)
    r = riv_terms(head=h, conductance=c_r, stage=s, bottom=b)
    residual = 4. * (h - previous) + d.discharge + r.discharge - source
    np.testing.assert_allclose(residual, 0., rtol=0., atol=1e-14)
    reconstructed = (source + 4. * previous + d.rhs + r.rhs) / (
        4. + d.diagonal + r.diagonal
    )
    np.testing.assert_allclose(reconstructed, h, rtol=0., atol=1e-14)


def test_direct_flux_preserves_small_difference_at_large_datum():
    base = 1e12
    head = grid([base + 0.125])
    c, s, b = validate_riv_fields(conductance=3.1, stage=base, bottom=base - 1., **domain(1))
    terms = riv_terms(head=head, conductance=c, stage=s, bottom=b)
    assert terms.discharge.item() == 3.1 * 0.125
