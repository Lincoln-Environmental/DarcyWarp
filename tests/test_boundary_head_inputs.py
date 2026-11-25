"""Physical GHB and CHD stages must survive model construction and reuse."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE import create_solver
from DARCY_WARP_PACKAGE.config import NP_FLOAT as NP_FLOAT_3D
from DARCY_WARP_PACKAGE.warped_darcy import NP_FLOAT as NP_FLOAT_2D


def ghb_fields(stage, width=1.0):
    shape = (4, 4)
    active = np.ones(shape, dtype=np.int32)
    active[0, 0] = 0
    fixed = np.zeros_like(active)
    fixed[0, 3] = 1
    mask = np.zeros_like(active)
    mask[0, [0, 3]] = 1  # These records overlap the inactive and prescribed cells.
    mask[1, 1:3] = 1
    boundary = np.full(shape, np.nan)
    boundary[0, 0] = 1e30
    boundary[0, 3] = -np.inf
    boundary[1, 1] = stage
    boundary[1, 2] = -1e30
    widths = np.full(shape, width)
    widths[1, 2] = 0  # This record has no exchange.
    return dict(T_field=np.ones(shape), R_field=np.zeros(shape), active=active, bc_mask=fixed,
                bc_values=np.full(shape, 3.), gh_mask=mask, gh_width=widths, gh_head=boundary)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf, 1e20, -1e30])
def test_2d_active_ghb_rejects_invalid_stages_without_replacing_prior_model(invalid):
    with create_solver(dim=2, nx=4, ny=4, dx=1., device="cpu", use_ghb=True, aq_thickness=1.) as model:
        valid = ghb_fields(0.)
        model.build_from_fields(**valid)
        controls = dict(implementation="classic", max_cycles=300, check_every_no=1)
        baseline, info = model.solve(**controls)
        assert info["converged"]
        assert 0. < baseline[1, 1] < 3.  # The zero-stage boundary drains the aquifer.
        replacement = ghb_fields(invalid)
        replacement["T_field"] *= 2
        replacement["R_field"] += 5
        with pytest.raises(ValueError, match="active GHB head"):
            model.build_from_fields(**replacement)
        repeated, info = model.solve(**controls)
        assert info["converged"]
        np.testing.assert_array_equal(repeated, baseline)
        np.testing.assert_array_equal(model.T_field_host, valid["T_field"])
        np.testing.assert_array_equal(model.R_field_host, valid["R_field"])
        assert np.count_nonzero(model.gh_mask_host) == 1
        assert np.all(np.isfinite(model.gh_head_host))


def test_2d_ghb_hierarchy_and_factor_update_preserve_precision_and_previous_state():
    stage = NP_FLOAT_2D(171.123456789123)
    width = NP_FLOAT_2D(1.123456789123)
    with create_solver(dim=2, nx=4, ny=4, dx=1., device="cpu", use_ghb=True, aq_thickness=1.) as model:
        model.build_from_fields(**ghb_fields(stage, width))
        model.build_hierarchy(max_levels=2, min_coarse_n=2, min_coarse_cells=1)
        fine, coarse = model.mg_levels
        for values in (model.gh_head_host, fine.gh_head_host, fine.gh_head_wp.numpy()):
            assert values.dtype == np.dtype(NP_FLOAT_2D) and values[1, 1] == stage
        for values in (model.gh_width_host, fine.gh_width_host, fine.ghb_factor_host):
            assert values.dtype == np.dtype(NP_FLOAT_2D) and values[1, 1] == width
        # The coarse correction equation uses zero stage; its factors keep model precision.
        assert coarse.ghb_factor_host.dtype == np.dtype(NP_FLOAT_2D)
        assert coarse.ghb_factor_host[0, 0] == width
        np.testing.assert_array_equal(coarse.gh_head_host, 0.)
        with pytest.raises(ValueError, match="aq_thickness"):
            model.update_ghb_factor_in_place(gh_alpha=4., aq_thickness=-1.)
        model.update_ghb_factor_in_place(gh_alpha=2.)
        assert model.ghb_factor_host[1, 1] == 2 * width
        assert model.ghb_factor_wp.numpy()[1, 1] == 2 * width
        assert fine.ghb_factor_host[1, 1] == 2 * width


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_cartesian_active_chd_marker_is_rejected_on_rebuild_and_update(implementation):
    if implementation == "fast" and np.dtype(NP_FLOAT_3D) != np.dtype(np.float64):
        pytest.skip("Fast 3D requires FP64")
    shape = (1, 1, 3)
    valid_heads = np.array([[[9., -1e30, 1e30]]])
    invalid_heads = np.array([[[1e20, 0., 0.]]])
    fields = dict(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
                  active=np.array([[[1, 1, 0]]]), bc_mask=np.array([[[1, 0, 0]]]), rhs=np.zeros(shape))
    with create_solver(dim=3, nx=3, ny=1, nz=1, dx=1., dz=1., device="cpu",
                       implementation=implementation) as model:
        model.build_from_K_fields(**fields, bc_values=valid_heads)
        head, info = model.solve()
        assert info["converged"]
        with pytest.raises(ValueError, match="active prescribed head"):
            model.build_from_K_fields(**fields, bc_values=invalid_heads)
        with pytest.raises(ValueError, match="active prescribed head"):
            model.update_boundary_values(bc_values=-invalid_heads)
        repeated, info = model.solve()
        assert info["converged"]
        np.testing.assert_array_equal(repeated, head)
        np.testing.assert_allclose(repeated[0, 0, :2], 9., rtol=0., atol=1e-5)
