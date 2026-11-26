"""Observational inputs, spatial orientation, and public hydraulic model builders."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE import Model3DInputs, create_solver
from DARCY_WARP_PACKAGE.ascii_grid import write_ascii_grid
from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import (
    load_case_inputs, resolve_fixed_heads,
)


def test_archived_missing_north_stages_preserve_zero_head_ghb_and_nztm_locations():
    case = load_case_inputs(grid_size=100, boundary_policy="sea_precedence")
    missing = case.boundary_audit["missing_stages"]
    expected = {(58, 834), (59, 841), (59, 842), (59, 843), (60, 852)}
    assert {(record["row"], record["column"]) for record in missing} == expected
    for record in missing:
        row, col = record["row"], record["column"]
        assert record["source"] == "north"
        assert (record["nztm_easting"], record["nztm_northing"]) == (
            1490750. + 100.*col, 5201550. - 100.*row)
        assert case.active[row, col] and not case.bc_mask[row, col]
    # The archived river remains a conducting zero-stage GHB at this site.
    assert case.gh_mask[60, 852] and case.gh_width[60, 852] > 0.
    assert case.gh_head[60, 852] == 0.
    assert case.bc_mask[625, 455]
    assert case.bc_values[625, 455] == pytest.approx(0.1)


def test_sea_and_south_precedence_remain_physical_when_input_order_reverses():
    active = np.array([[True, True, False]])
    north = np.array([[8., 9., 10.]])
    south = np.array([[3., 4., 5.]])
    sea = np.zeros_like(north)
    masks = {"sea": np.array([[True, False, True]])}
    # The archive assembly commonly inserts north then south then sea. The
    # source priority should also hold when a caller inserts them in reverse.
    inputs = {"sea": sea, "south": south, "north": north}
    fixed, heads, audit = resolve_fixed_heads(
        active=active, stages=inputs, masks=masks, policy="sea_precedence")
    np.testing.assert_array_equal(fixed, [[1, 1, 0]])
    np.testing.assert_array_equal(heads, [[0., 4., 0.]])
    selections = {record["column"]: record["selected_source"] for record in audit["overlaps"]}
    assert selections == {0: "sea", 1: "south"}


def test_exported_ascii_grid_places_north_row_above_south_and_preserves_zero(tmp_path):
    destination = tmp_path/"heads.asc"
    write_ascii_grid(destination, np.array([[8., np.nan, 0.], [2., 4., 6.]]),
        x_min=1550000., y_min=5140000., cellsize=100.)
    with destination.open(encoding="utf-8") as raster:
        header = {}
        for _ in range(6):
            name, value = next(raster).split()
            header[name.lower()] = float(value)
        rows = np.loadtxt(raster)
    assert header["ncols"] == 3 and header["nrows"] == 2 and header["cellsize"] == 100.
    # First data row must contain the head at the northwest cell centre.
    northwest = (header["xllcorner"] + 0.5*header["cellsize"],
        header["yllcorner"] + (header["nrows"] - 0.5)*header["cellsize"])
    assert northwest == (1550050., 5140150.)
    np.testing.assert_array_equal(rows, [[8., header["nodata_value"], 0.], [2., 4., 6.]])


def test_2d_build_filters_ineffective_ghb_and_rejects_sentinel_on_live_contact(solver_device):
    shape = (4, 4)
    active = np.zeros(shape, dtype=int)
    active[1, 1:3] = 1
    fixed = np.zeros_like(active)
    fixed[1, 1] = 1
    stages = np.full(shape, -1e30)
    stages[1, 2] = 0.
    fields = dict(T_field=np.ones(shape), R_field=np.zeros(shape), active=active,
        bc_mask=fixed, bc_values=10.*fixed, gh_mask=np.ones(shape, dtype=int),
        gh_head=stages, gh_width=np.ones(shape), gh_alpha=1.)
    with create_solver(dim=2, solver="pcg", nx=4, ny=4, dx=1., device=solver_device,
            use_ghb=True, aq_thickness=1.) as model:
        model.build_from_fields(**fields)
        head, info = model.solve()
        assert info["converged"]
        assert head[1, 2] == pytest.approx(5., abs=1e-7)
        invalid_stages = stages.copy()
        invalid_stages[1, 2] = -1e30
        fields["gh_head"] = invalid_stages
        with pytest.raises(ValueError):
            model.build_from_fields(**fields)
        final_head, final_info = model.solve()
        assert final_info["converged"]
        np.testing.assert_allclose(final_head[active != 0], [10., 5.], rtol=0., atol=1e-7)


@pytest.mark.parametrize("supply_faces", [False, True], ids=["cell-conductivity", "supplied-faces"])
def test_model3dinputs_preserves_material_geometry_and_independent_source_units(solver_device, supply_faces):
    shape = (1, 1, 2)
    k = np.full(shape, 2.)
    fixed = np.array([[[1, 0]]])
    # Unit area recharge integrates to 6 m³/day and offsets a 3 m³/day well.
    inputs = Model3DInputs(kx=k, ky=k, kz=k, active=np.ones(shape), bc_mask=fixed,
        bc_values=8.*fixed, dx=2., dy=3., dz=4., R_field=1., wel_entries=[((0, 0, 1), -3.)])
    k[:] = 100.
    fixed[:] = 0
    faces = None
    face_c = 2.*3.*4./2.
    if supply_faces:
        face_c = 5.
        x_plus = np.array([[[face_c, 0.]]])
        x_minus = np.array([[[0., face_c]]])
        faces = (x_plus, x_minus, np.zeros(shape), np.zeros(shape), np.zeros(shape), np.zeros(shape))
    with inputs.build_solver(device=solver_device, implementation="fast", face_conductance=faces) as model:
        head, info = model.solve(linear_acceleration="fgmres")
        assert info["converged"]
        np.testing.assert_allclose(head.ravel(), [8., 8.+3./face_c], rtol=0., atol=1e-7)
        assert info["source_budgets"]["wel"]["net"] == -3.
        assert info["source_budgets"]["recharge"]["net"] == 6.


def test_sparse_river_bed_marker_is_rejected_before_replacing_valid_reach(solver_device):
    shape = (1, 1, 1)
    ones = np.ones(shape)
    with create_solver(dim=3, nx=1, ny=1, nz=1, dx=1., dz=1., device=solver_device,
            implementation="fast", formulation="mf6_6_6_2_npf") as model:
        model.build_from_K_fields(kx_field=ones, ky_field=ones, kz_field=ones, active=ones,
            bc_mask=np.zeros(shape), bc_values=np.zeros(shape), ghb_conductance=1., ghb_head=2.,
            riv_entries=[((0, 0, 0), 1., 4., 1.)])
        # A dry/no-data marker is finite but cannot be the physical elevation of a conducting bed.
        with pytest.raises(ValueError):
            model.update_riv_entries([((0, 0, 0), 1., 4., -1e30)])
        head, info = model.solve()
        assert info["converged"] and head.item() == pytest.approx(3.)
        assert model.accepted_riv_discharge(head=head, info=info).item() == pytest.approx(-1.)


def test_transient_restart_updates_storage_and_reuses_accepted_head(solver_device):
    shape = (1, 1, 1)
    ones = np.ones(shape)
    inputs = Model3DInputs(kx=ones, ky=ones, kz=ones, active=ones, bc_mask=np.zeros(shape),
        bc_values=np.zeros(shape), rhs=3.*ones, dx=2., dy=3., dz=4.)
    with inputs.build_solver(device=solver_device, implementation="fast", formulation="mf6_6_6_2_npf") as model:
        first, first_info = model.solve_transient_npf(dt=[2.], head_prev=4.*ones, ss=0.25, sy=0.)
        assert first_info["converged"] and first[0].item() == pytest.approx(5.)
        model.update_rhs(-4.*ones)
        second, second_info = model.solve_transient_npf(dt=[3.], head_prev=first[-1], ss=0.5, sy=0.)
        assert second_info["converged"]
        # Both storage changes equal Q*dt, using each run's own 24 m³ cell, Ss and input head.
        assert (second[-1].item()-first[-1].item())*24.*0.5 == pytest.approx(-12.)
        assert second[-1].item() == pytest.approx(4.)
