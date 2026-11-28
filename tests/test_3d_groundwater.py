"""Independent hydraulic checks through the public Cartesian and NPF solvers."""

from copy import copy
import json
from pathlib import Path

import numpy as np
import pytest

from DARCY_WARP_PACKAGE import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
from DARCY_WARP_PACKAGE.validation.npf662_reference import (
    NPF662Case, load_npf662_case, save_npf662_case,
)


NPF = "mf6_6_6_2_npf"
CELL = (0, 0, 0)


def groundwater_model(device, shape=(1, 1, 1), *, spacing=(1., 1., 1.),
                      formulation=NPF, interfaces=None, **fields):
    """Construct a public model; coordinates use (z, y, x), spacing (x, y, z)."""
    nz, ny, nx = shape
    dx, dy, dz = spacing
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
        device=device, implementation="fast", solver="kcycle", formulation=formulation)
    inputs = dict(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
        bc_mask=np.zeros(shape, dtype=np.int32), active=np.ones(shape, dtype=np.int32),
        bc_values=np.zeros(shape), initial_head=np.zeros(shape))
    inputs.update(fields)
    if interfaces is None:
        model.build_from_K_fields(**inputs)
    else:
        model.build_from_layers(z_interfaces=interfaces, **inputs)
    return model


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["vertical", "north-south", "east-west"])
def test_directional_series_faces_and_flux(solver_device, axis):
    shape = [1, 1, 1]
    shape[axis] = 3
    shape = tuple(shape)
    fixed = np.array([1, 0, 1]).reshape(shape)
    boundary_head = np.array([12., 0., 3.]).reshape(shape)
    fields = dict(kx_field=np.full(shape, 20.), ky_field=np.full(shape, 30.), kz_field=np.full(shape, 40.))
    field_name = ("kz_field", "ky_field", "kx_field")[axis]
    fields[field_name] = np.array([1., 4., 16.]).reshape(shape)
    spacing = (2., 3., 5.)
    distance = spacing[2-axis]
    area = np.prod(spacing)/distance
    r_left = 0.5*distance*(1./1. + 1./4.)/area
    r_right = 0.5*distance*(1./4. + 1./16.)/area
    flow = 9./(r_left + r_right)
    with groundwater_model(solver_device, shape, spacing=spacing, formulation="legacy",
            bc_mask=fixed, bc_values=boundary_head, **fields) as model:
        h, info = model.solve(linear_acceleration="fgmres")
        assert info["converged"]
        np.testing.assert_allclose(h.ravel(), [12., 12.-flow*r_left, 3.], rtol=0., atol=1e-8)
        assert info["budget"]["chd_inflow"] == pytest.approx(flow, rel=1e-8)
        assert info["budget"]["chd_outflow"] == pytest.approx(flow, rel=1e-8)


def test_layered_vertical_resistance_uses_both_half_thicknesses(solver_device):
    shape = (3, 1, 1)
    interfaces = np.array([12., 10., 7., 1.]).reshape(4, 1, 1)
    kz = np.array([1., 3., 2.]).reshape(shape)
    with groundwater_model(solver_device, shape, spacing=(2., 4., 9.), interfaces=interfaces,
            kz_field=kz, bc_mask=np.array([1, 0, 1]).reshape(shape),
            bc_values=np.array([12., 0., 3.]).reshape(shape)) as model:
        head, info = model.solve()
        resistance_upper = (1./1. + 1.5/3.)/8.
        resistance_lower = (1.5/3. + 3./2.)/8.
        flow = 9./(resistance_upper + resistance_lower)
        assert info["converged"]
        assert head[1, 0, 0] == pytest.approx(12. - flow*resistance_upper, abs=1e-8)
        assert info["chd_budget"]["chd_outflow"] == pytest.approx(flow, rel=1e-8)


def test_surface_recharge_uses_highest_eligible_column_and_area_once(solver_device):
    shape = (2, 1, 5)
    active = np.array([[[1, 0, 0, 0, 1]], [[0, 0, 1, 0, 1]]])
    fixed = np.zeros(shape, dtype=int)
    fixed[0, 0, 4] = 1
    ghb = np.zeros(shape)
    ghb[0, 0, 0] = ghb[1, 0, 2] = 1.
    recharge = np.array([[0.25, 100., -0.5, 100., 100.]])
    with groundwater_model(solver_device, shape, spacing=(2., 3., 4.), active=active,
            bc_mask=fixed, bc_values=7.*fixed, ghb_conductance=ghb, ghb_head=0.,
            R_field=recharge) as model:
        head, info = model.solve(npf_preconditioner="kcycle")
        assert info["converged"]
        np.testing.assert_allclose(head[active != 0], [1.5, 7., -3., 7.], rtol=0., atol=1e-8)
        assert info["source_budgets"]["recharge"] == {"inflow": 1.5, "outflow": 3., "net": -1.5}


def test_well_replacement_preserves_raw_rates_on_nonunit_cells(solver_device):
    shape = (1, 1, 2)
    raw = np.array([[[0., 4.]]])
    # Cx = K*dy*dz/dx = 6. Pumping is volumetric and is never scaled by this face area.
    with groundwater_model(solver_device, shape, spacing=(2., 3., 4.),
            bc_mask=np.array([[[1, 0]]]), bc_values=np.full(shape, 10.), rhs=raw) as model:
        for rate in (-1., -7., 0.):
            entries = [((0, 0, 1), rate)] if rate else None
            model.update_wel(entries)
            head, info = model.solve(npf_preconditioner="kcycle")
            assert info["converged"]
            assert head[0, 0, 1] == pytest.approx(10.+(4.+rate)/6., abs=1e-8)
            assert info["source_budgets"]["volumetric"]["net"] == 4.
            assert info["source_budgets"]["wel"]["net"] == rate


def test_opposing_wells_do_not_relax_head_acceptance_or_disappear_from_budgets(solver_device):
    # The well rates cancel in this cell's equation. Its only active boundary
    # is a weak GHB at 5 m, so an initial 6 m head must be corrected.
    rates = [-1.e9, 1.e9]
    with groundwater_model(solver_device, spacing=(3., 2., 4.), ghb_conductance=2.e-7,
            ghb_head=5., drn_entries=[(CELL, 1., 10.)], initial_head=np.full((1, 1, 1), 6.),
            wel_entries=[(CELL, rates[0]), (CELL, rates[1])]) as model:
        head, info = model.solve(npf_preconditioner="kcycle")
        assert info["converged"] and info["krylov_iterations"] > 0
        assert head.item() == pytest.approx(5., abs=1.e-7)
        assert info["source_budgets"]["wel"] == {"inflow": 1.e9, "outflow": 1.e9, "net": 0.}
        np.testing.assert_array_equal(model.accepted_wel_rates(head=head, info=info), rates)


@pytest.mark.parametrize(("package", "external_head", "expected_head", "expected_outflow"), [
    ("drn", 0., 0., 0.), ("drn", 7., 3., 4.),
    ("riv", -4., 0., -4.), ("riv", 6., 4., 2.),
], ids=["drain-closed", "drain-open", "river-below-bed", "river-gaining"])
def test_package_piecewise_roots(solver_device, package, external_head, expected_head, expected_outflow):
    # GHB C=1, DRN C=2 and elevation=1, or RIV C=2, stage=3, bottom=1.
    fields = {"drn_entries": [(CELL, 2., 1.)]} if package == "drn" else {"riv_entries": [(CELL, 2., 3., 1.)]}
    with groundwater_model(solver_device, ghb_conductance=1., ghb_head=external_head, **fields) as model:
        head, info = model.solve()
        assert info["converged"]
        assert head.item() == pytest.approx(expected_head, abs=1e-8)
        rates = model.accepted_boundary_entry_rates(package=package, head=head, info=info)
        np.testing.assert_allclose(rates, [expected_outflow], rtol=0., atol=1e-8)
        assert info[f"{package}_budget"]["net"] == pytest.approx(-expected_outflow, abs=1e-8)


def test_opposing_river_reaches_retain_individual_exchange(solver_device):
    with groundwater_model(solver_device, riv_entries=[(CELL, 2., 2., 0.), (CELL, 2., 4., 0.)]) as model:
        head, info = model.solve()
        assert info["converged"]
        assert head.item() == pytest.approx(3., abs=1e-8)
        rates = model.accepted_boundary_entry_rates(package="riv", head=head, info=info)
        np.testing.assert_allclose(rates, [2., -2.], rtol=0., atol=1e-8)
        assert info["riv_budget"] == pytest.approx(dict(inflow=2., outflow=2., net=0.), abs=1e-8)


def test_dry_face_cannot_share_a_strong_boundary_restraint(solver_device):
    shape = (1, 1, 3)
    start = np.array([[[10., 9., 30.]]])
    k = np.full(shape, 100.)
    source = np.array([[[1000., -1000.+8e-6, -100.]]])
    boundary_c = np.array([[[8e-7, 0., 100.]]])
    boundary_h = np.array([[[10., 0., 31.]]])
    with groundwater_model(solver_device, shape, spacing=(10., 10., 10.), kx_field=k,
            ky_field=k, kz_field=k, rhs=source, initial_head=start, ghb_conductance=boundary_c,
            ghb_head=boundary_h) as model:
        head, info = model.solve(unconfined=True, npf_cell_type=np.array([[[0, 0, 1]]]),
            top_elevation=41., npf_preconditioner="kcycle")
        # Cell 2 is dry (30 < 31) and upstream of cell 1 at the initial and final
        # heads. Its connecting face is closed; only the weak GHB sets heads in cells 0/1.
        left = 10. + source[0, 0, :2].sum()/8e-7
        right = left + source[0, 0, 1]/1000.
        assert info["converged"]
        np.testing.assert_allclose(head.ravel(), [left, right, 30.], rtol=0., atol=1e-6)


def strip_problem(shape=(2, 12, 20)):
    fixed = np.zeros(shape, dtype=int)
    fixed[:, :, 0] = fixed[:, :, -1] = 1
    exact = np.broadcast_to(np.linspace(8., 3., shape[2]), shape)
    fields = dict(bc_mask=fixed, bc_values=exact, kz_field=np.full(shape, 17.),
        ky_field=np.full(shape, 0.3), initial_head=np.full(shape, 2.))
    return shape, fields, exact


def test_saturated_fp64_fgmres_executes_multilevel_kcycle(solver_device):
    shape, fields, exact = strip_problem()
    controls = NPF662KCycleControls(levels=3, cached_line_factors=True,
        coefficient_state="saturated", preconditioner_precision="float64", cycle_kind="k")
    # The middle free cell is exactly on a zero-rate drain branch above its solved head.
    fields["drn_entries"] = [((0, 4, 8), 2., 20.)]
    with groundwater_model(solver_device, shape, **fields) as model:
        head, info = model.solve(npf_kcycle_controls=controls)
        hierarchy = info["multigrid"]
        assert info["converged"] and info["krylov_method"] == "FGMRES"
        assert info["npf_preconditioner"] == "kcycle" and hierarchy["actual_levels"] >= 2
        assert hierarchy["precision"] == "float64" and hierarchy["coefficient_state"] == "saturated"
        assert hierarchy["applications"] > 0 and hierarchy["cycles"] > 0
        if solver_device.startswith("cuda"):
            assert hierarchy["graph_replays"] > 0
        np.testing.assert_allclose(head, exact, rtol=0., atol=1e-6)


def test_current_fp32_fgmres_executes_qualified_five_level_profile(solver_device, record_property):
    from working_tests.qualify_package_kcycle import options_for

    profile_path = Path(__file__).resolve().parents[1]/"working_tests"/"npf_ghb_five_level_controls.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    options = options_for(saved=profile, state="current", precision="float32", preconditioner="kcycle",
        graphs=True, line_smoother="jacobi", levels=5)
    shape, fields, exact = strip_problem(shape=(2, 16, 32))
    cell = (0, 6, 10)
    # At the strip solution, C=2 times a 0.5 m head difference drains 1 m³/day.
    # Co-located injection sustains the original linear gradient while the active
    # drain contributes to the Newton Jacobian and the current-coefficient hierarchy.
    fields["drn_entries"] = [(cell, 2., float(exact[cell])-0.5)]
    fields["wel_entries"] = [(cell, 1.)]
    with groundwater_model(solver_device, shape, **fields) as model:
        head, info = model.solve(**options)
        hierarchy = info["multigrid"]
        assert info["converged"] and info["krylov_method"] == "FGMRES"
        assert head.dtype == np.float64 and hierarchy["precision"] == "float32"
        assert info["npf_preconditioner"] == "kcycle" and hierarchy["controls"]["cycle_kind"] == "k"
        assert hierarchy["coefficient_state"] == "current" and hierarchy["controls"]["device_hierarchy_build"]
        assert hierarchy["actual_levels"] == 5
        assert hierarchy["dimensions"] == [[2, 16, 32], [2, 8, 16], [2, 4, 8], [2, 2, 4], [2, 2, 2]]
        for key, value in profile["newton"].items():
            assert info["controls"][key] == value
        for key, value in profile["multigrid"].items():
            assert hierarchy["controls"][key] == value
        assert info["krylov_restart"] == profile["solve"]["krylov_restart"]
        assert info["max_krylov_per_outer"] == profile["solve"]["max_cycles"]
        assert info["max_outer"] == profile["solve"]["max_outer"]
        assert info["final_acceptance_thresholds"]["relative_cell_imbalance"] == profile["solve"]["npf_final_relative_tolerance"]
        assert info["krylov_iterations"] > 0 and hierarchy["applications"] > 0 and hierarchy["cycles"] > 0
        # One refresh initializes device coefficients; another assembles the current operator.
        assert hierarchy["device_storage_refresh_count"] >= 2
        if solver_device.startswith("cuda"):
            assert info["tiled_reductions"]
            assert hierarchy["graph_captures"] > 0 and hierarchy["graph_replays"] > 0
            assert hierarchy["refresh_graph_captures"] > 0 and hierarchy["refresh_graph_replays"] > 0
        np.testing.assert_allclose(head, exact, rtol=0., atol=1e-6)
        rates = model.accepted_boundary_entry_rates(package="drn", head=head, info=info)
        np.testing.assert_allclose(rates, [1.], rtol=0., atol=2e-6)
        record_property("device", solver_device)
        record_property("krylov_iterations", info["krylov_iterations"])
        for key in ("actual_levels", "applications", "cycles", "device_storage_refresh_count", "graph_replays"):
            record_property(key, hierarchy[key])
        record_property("maximum_head_error_m", float(np.max(np.abs(head-exact))))


def test_direct_cartesian_fgmres_reaches_the_linear_strip_solution(solver_device):
    shape, fields, exact = strip_problem()
    controls = NPF662KCycleControls(levels=3, cached_line_factors=True)
    with groundwater_model(solver_device, shape, formulation="legacy", **fields) as model:
        head, info = model.solve(linear_acceleration="fgmres", linear_kcycle_controls=controls)
        assert info["converged"] and info["final_budget_acceptance"]
        assert info["linear_acceleration"] == "fgmres" and info["n_levels"] >= 2
        assert info["preconditioner_applications"] > 0 and info["krylov_iterations"] > 0
        np.testing.assert_allclose(head, exact, rtol=0., atol=1e-7)


def test_resident_history_conserves_scheduled_wells_recharge_and_storage(solver_device):
    # Cell volume=24 and Ss=0.125, so three cubic metres change head by one metre.
    with groundwater_model(solver_device, spacing=(2., 3., 4.), rhs=np.ones((1, 1, 1)),
            R_field=0.5, wel_entries=[(CELL, -4.)]) as model:
        heads, info = model.solve_transient_npf(head_prev=np.zeros((1, 1, 1)), dt=[1., 2., 0.5],
            ss=0.125, sy=0., source_factors=[2., 0., -1.], recharge_rates=[0.5, -0.5, 1.],
            wel_rates=[[(CELL, -2.)], [(CELL, 6.)], None])
        # Raw/recharge/WEL give +3,+3,+5 m³/time in the three stress periods.
        assert info["converged"] and info["accepted_steps"] == 3
        np.testing.assert_allclose(heads.ravel(), [1., 3., 3.+2.5/3.], rtol=0., atol=1e-8)
        assert info["output_history_readbacks"] == 1
        totals = info["accepted_exchange_volume_totals"]["sources"]
        assert totals["wel"] == {"inflow": 12., "outflow": 2., "net": 10.}
        assert totals["recharge"] == {"inflow": 6., "outflow": 6., "net": 0.}
        assert totals["volumetric"]["net"] == 1.5
        assert model.wel_budget()["net"] == -4.


def test_resident_drain_switches_on_and_off_after_the_well_changes(solver_device):
    with groundwater_model(solver_device, drn_entries=[(CELL, 2., 1.)]) as model:
        heads, info = model.solve_transient_npf(head_prev=np.zeros((1, 1, 1)), dt=[1., 1.],
            ss=1., sy=0., wel_rates=[[(CELL, 4.)], [(CELL, -2.)]])
        assert info["converged"]
        np.testing.assert_allclose(heads.ravel(), [2., 0.], atol=1e-8, rtol=0.)
        expected = [2., 0.]
        for index, period in enumerate(info["periods"]):
            rate = model.accepted_boundary_entry_rates(package="drn", head=heads[index], info=period)
            np.testing.assert_allclose(rate, [expected[index]], atol=1e-8, rtol=0.)
        assert info["accepted_exchange_volume_totals"]["drn"]["outflow"] == pytest.approx(2.)


@pytest.mark.parametrize("bad_package", ["river", "well"])
def test_invalid_update_leaves_the_original_equation_intact(solver_device, bad_package):
    with groundwater_model(solver_device, ghb_head=3., ghb_conductance=1.,
            wel_entries=[(CELL, -1.)], riv_entries=[(CELL, 1., 4., 0.)]) as model:
        if bad_package == "river":
            with pytest.raises(ValueError):
                model.update_riv_entries([(CELL, 1., 0., 4.)])
        else:
            with pytest.raises(ValueError):
                model.update_wel([((0, 0, 1), -99.)])
        head, info = model.solve()
        assert info["converged"] and head.item() == pytest.approx(3.)
        assert info["source_budgets"]["wel"]["net"] == -1.
        assert info["riv_budget"]["net"] == pytest.approx(1.)


def test_build_consumes_sparse_boundary_records_exactly_once(solver_device):
    shape = (1, 1, 2)
    entries = iter([((0, 0, 1), 1., 0.)])
    with groundwater_model(solver_device, shape, bc_mask=np.array([[[1, 0]]]),
            bc_values=np.full(shape, 10.), drn_entries=entries) as model:
        head, info = model.solve()
        assert info["converged"]
        assert head[0, 0, 1] == pytest.approx(5., abs=1e-8)
        rates = model.accepted_boundary_entry_rates(package="drn", head=head, info=info)
        np.testing.assert_allclose(rates, [5.], rtol=0., atol=1e-8)


def test_inspecting_a_copy_of_the_token_keeps_the_original_equation_alive(solver_device):
    with groundwater_model(solver_device, ghb_head=3., ghb_conductance=1.,
            drn_entries=[(CELL, 1., 1.)]) as model:
        head, info = model.solve()
        assert info["converged"] and head.item() == pytest.approx(2.)
        diagnostic = copy(info["accepted_state"])
        assert diagnostic["identifier"] == info["accepted_state"]["identifier"]
        del diagnostic
        rates = model.accepted_boundary_entry_rates(package="drn", head=head, info=info)
        np.testing.assert_allclose(rates, [1.], rtol=0., atol=1e-8)


def test_canonical_archive_restores_cell_and_threshold_of_each_sparse_reach(tmp_path):
    shape = (1, 2, 2)
    ones = np.ones(shape)
    zeros = np.zeros(shape)
    entries = [((0, 1, 1), 3., 8., 2.), ((0, 0, 1), 1., 4., 1.), ((0, 1, 1), 2., 5., 3.)]
    case = NPF662Case(kx=ones, ky=ones, kz=ones, active=ones.astype(int), fixed=zeros.astype(int),
        values=zeros, source=zeros, initial=zeros, dx=2., dy=3., dz=4., riv_entries=entries)
    path = tmp_path/"reach-inputs.npz"
    save_npf662_case(case=case, path=path)
    recovered = load_npf662_case(path=path)
    assert recovered.digest() == case.digest()
    assert sorted(recovered.riv_entries) == sorted(entries)


def test_trial_correction_accounts_for_newly_active_drain(solver_device):
    from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model, independent_reference
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace

    shape = (2, 1, 1)
    fixed = np.array([0, 1], dtype=np.int32).reshape(shape)
    head = np.array([1., 6.]).reshape(shape)
    with groundwater_model(device=solver_device, shape=shape, bc_mask=fixed,
                            bc_values=head, initial_head=head,
                            drn_entries=[(CELL, 9., 4.)], riv_entries=[(CELL, 2., 5., 3.)]) as model:
        case = case_from_model(model=model, top=1., cell_type=0)
        reference, _ = independent_reference(model=model, case=case)
        workspace = NPF662Workspace(model=model, case=case, restart=2, krylov_budget=0, reference=reference)
        try:
            current = workspace.evaluate(trial=False, linearize=True)
            workspace.trial.assign(np.array([10., 6.]))
            proposed = workspace.evaluate(trial=True)
            assert proposed['physical_rms'] > current['physical_rms']
            corrected, changed = workspace.rebalance_drain_trial(current=current, trial=proposed,
                physical_tolerance=1.e-12, relative_tolerance=1.e-8)
            assert changed
            # (6 - h) = 9 * (h - 4) + 2 * (h - 5) on the active branches.
            np.testing.assert_allclose(workspace.trial.numpy(), [52. / 12., 6.], rtol=0., atol=1.e-10)
            np.testing.assert_array_equal(workspace.head.numpy().reshape(shape), head)
            assert corrected['physical_rms'] < 1.e-9
        finally:
            workspace.close()


def test_wetting_trial_balances_both_cells_on_a_face(solver_device):
    from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model, independent_reference
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace

    shape = (1, 1, 2)
    interfaces = np.broadcast_to(np.array([1., 0.])[:, None, None], (2, 1, 2)).copy()
    head = np.array([[[.11, .10]]])
    source = np.array([[[.05, 0.]]])
    neighbour_head = .20
    with groundwater_model(device=solver_device, shape=shape, interfaces=interfaces, initial_head=head,
                            rhs=source, ghb_conductance=np.array([[[0., .1]]]),
                            ghb_head=np.array([[[0., neighbour_head]]])) as model:
        case = case_from_model(model=model, top=1., cell_type=1)
        reference, _ = independent_reference(model=model, case=case)
        workspace = NPF662Workspace(model=model, case=case, restart=2, krylov_budget=0, reference=reference)
        try:
            current = workspace.evaluate(trial=False, linearize=True)
            workspace.trial.assign(np.array([.8, neighbour_head]))
            proposed = workspace.evaluate(trial=True)
            corrected, changed = workspace.rebalance_wetting_trial(current=current, trial=proposed,
                physical_tolerance=1.e-12, relative_tolerance=1.e-8)
            assert changed and corrected['physical_rms'] < proposed['physical_rms']
            candidate = workspace.trial.numpy().reshape(shape)
            assert head[0, 0, 0] < candidate[0, 0, 0] < .8
            assert candidate[0, 0, 1] == neighbour_head
            # For fixed trial neighbour heads the norm minimiser splits the
            # .05 source equally between the donor residual and the receiver.
            _, fields = reference.evaluate(head=candidate)
            np.testing.assert_allclose(fields['residual'], np.array([[[.025, .025]]]), atol=2.e-9)
        finally:
            workspace.close()


def test_closed_component_with_inactive_drain_opens_only_at_its_threshold(solver_device):
    shape = (1, 1, 5)
    active = np.array([[[1, 0, 1, 0, 1]]], dtype=np.int32)
    wells = [((0, 0, 0), 5.), ((0, 0, 2), 5.), ((0, 0, 4), 5.)]
    drains = [((0, 0, 0), 2., 4.), ((0, 0, 0), 3., 10.),
              ((0, 0, 2), 2., 4.), ((0, 0, 2), 3., 10.),
              ((0, 0, 4), 2., 4.), ((0, 0, 4), 3., 10.)]
    with groundwater_model(solver_device, shape=shape, active=active,
            initial_head=np.full(shape, -3.), wel_entries=wells,
            drn_entries=drains) as model:
        head, info = model.solve()
        assert info["converged"] and len(info["active_set_seeds"]) == 3
        np.testing.assert_allclose(head[active != 0], 6.5, rtol=0., atol=1e-8)
        rates = model.accepted_boundary_entry_rates(package="drn", head=head, info=info)
        np.testing.assert_allclose(rates, [5., 0., 5., 0., 5., 0.], atol=1e-8, rtol=0.)



def test_fgmres_true_residual_stagnation_stops_before_budget():
    """Restart-2 GMRES on a cyclic permutation has no decreasing residual."""
    import warp as wp
    from DARCY_WARP_PACKAGE.solvers.npf662_fgmres import RestartedFGMRES3D

    class CyclicOperator:
        device = "cpu"
        shape = (1, 1, 3)
        n = 3

        def __init__(self):
            self.rhs = wp.array([1., 0., 0.], dtype=wp.float64, device=self.device)
            self.delta = wp.zeros(3, dtype=wp.float64, device=self.device)
            self.free = wp.ones(3, dtype=wp.int32, device=self.device)

        def matvec(self, x, y, z, alpha, beta):
            z.assign(alpha * np.roll(x.numpy(), 1) + beta * y.numpy())

    class IdentityPreconditioner:
        device_seconds = 0.

        def apply(self, *, vector, out, diagonal, free):
            wp.copy(dest=out, src=vector)

        def collect_timing(self):
            pass

    operator = CyclicOperator()
    operator.scale = wp.ones(3, dtype=wp.float64, device="cpu")
    solver = RestartedFGMRES3D(workspace=operator, restart=2)
    try:
        used, residual, target = solver.solve(
            preconditioner=IdentityPreconditioner(), max_iterations=3000,
            absolute_tolerance=1e-8, stagnation_restarts=4)
        assert solver.last_info["stopping"] == "stagnation"
        assert used == 8
        assert residual == pytest.approx(1.)
        assert solver.last_info["relative_reduction"] == pytest.approx(0.)
    finally:
        solver.close()


def test_stalled_current_preconditioner_retries_same_jacobian(solver_device, monkeypatch):
    """The nonlinear driver must reject a stalled direction and recover."""
    from DARCY_WARP_PACKAGE.solvers.npf662_fgmres import RestartedFGMRES3D
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls

    original_solve = RestartedFGMRES3D.solve
    calls = []

    def stalled_once(self, **kwargs):
        calls.append(int(self.owner.mode.numpy()[0]))
        if len(calls) == 1:
            initial = self._norm(x=self.owner.rhs)
            self.owner.delta.zero_()
            self.last_info = dict(stopping="stagnation", initial_residual=initial,
                relative_reduction=0., restarts=3, blocks=4,
                scalar_readbacks=0, synchronization_seconds=0.,
                multigrid_device_seconds=0., preconditioner_enqueue_seconds=0.)
            return 4, initial, kwargs["absolute_tolerance"]
        return original_solve(self, **kwargs)

    monkeypatch.setattr(RestartedFGMRES3D, "solve", stalled_once)
    shape, fields, exact = strip_problem(shape=(1, 8, 16))
    controls = NPF662KCycleControls(levels=4, cached_line_factors=True,
        coefficient_state="current", preconditioner_precision="float32")
    with groundwater_model(solver_device, shape, **fields) as model:
        head, info = model.solve(npf_kcycle_controls=controls,
            npf_newton_controls=NPF662NewtonControls(), krylov_restart=8)
        assert info["converged"]
        assert info["stalled_linear_solves"] >= 1
        assert info["saturated_preconditioner_retries"] >= 1
        assert calls[:2] == [1, 1]
        first = info["outer_history"][0]["attempts"][0]
        assert first["rejected"] == "inadequate_linear_progress"
        assert first["relative_reduction"] == 0.
        np.testing.assert_allclose(head, exact, rtol=0., atol=1e-6)
