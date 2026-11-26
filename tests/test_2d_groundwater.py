"""Two-dimensional groundwater equations, package exchanges and accepted histories."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.storage_2d import exact_unconfined_storage_terms
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


def build_model(device, transmissivity, fixed, head, dx=5.0, recharge=0.0,
                wells=None, ghb=None, drain=None, river=None, active=None):
    shape = head.shape
    model = WarpDarcySolver(nx=shape[1], ny=shape[0], dx=dx, device=device,
                            solver_type="kcycle", use_ghb=ghb is not None,
                            use_drn=drain is not None, use_riv=river is not None)
    boundary_fields = {}
    for package in (ghb, drain, river):
        if package is not None:
            boundary_fields.update(package)
    if active is None:
        active = np.ones(shape, dtype=np.int32)
    model.build_from_fields(T_field=transmissivity, R_field=recharge, active=active,
                            bc_mask=fixed, bc_values=head, wel_entries=wells, **boundary_fields)
    return model


def confined_options(implementation):
    return dict(implementation=implementation, rel_tol=1e-11, abs_tol_min=1e-11,
                max_cycles=400, nu_coarse=12, min_coarse_cells=4, check_every_no=1)


@pytest.mark.parametrize("method", ["pcg", "classic", "fast"])
def test_confined_strip_resistances_and_cumulative_downstream_forcing(solver_device, method):
    """Each serial face carries the sum of all sources downstream of that face."""
    ny, nx, dx = 4, 6, 3.0
    t = np.array([3.0, 7.0, 19.0, 5.0, 11.0, 17.0])
    recharge = np.array([0.0, 0.03, -0.01, 0.0, 0.02, -0.03])
    q = dx ** 2 * recharge
    q[-1] -= 0.8
    wells = [((row, nx - 1), -0.8) for row in range(ny)]
    expected = np.full(nx, 12.0)
    for column in range(1, nx):
        resistance = 0.5 / t[column - 1] + 0.5 / t[column]
        expected[column] = expected[column - 1] + resistance * q[column:].sum()
    initial = np.full((ny, nx), 12.0)
    fixed = np.zeros_like(initial, dtype=np.int32)
    fixed[:, 0] = 1
    with build_model(solver_device, np.tile(t, (ny, 1)), fixed, initial, dx,
                     np.tile(recharge, (ny, 1)), wells) as model:
        if method == "pcg":
            head, info = model.solve(solver="confined_pcg", rel_tol=1e-11, abs_tol_min=1e-11)
            assert info["solver_backend"] == "confined_pcg"
        else:
            head, info = model.solve(initial_head=initial, **confined_options(method))
            assert info["solver_backend"] == "confined_kcycle"
            assert info.get("implementation", "classic") == method
            assert info["n_cycles_used"] > 0
        assert info["converged"]
        np.testing.assert_allclose(head, np.tile(expected, (ny, 1)), rtol=0, atol=1e-7)


def test_general_head_boundary_tracks_aquifer_conductivity(solver_device):
    """A uniformly recharged aquifer has exactly one GHB discharge per cell."""
    shape = (5, 6)
    stage = np.full(shape, 14.0)
    fixed = np.zeros(shape, dtype=np.int32)
    ghb = dict(gh_mask=np.ones_like(fixed), gh_width=np.full(shape, 2.0),
               gh_head=stage, gh_alpha=0.5, aq_thickness=4.0)
    dx, recharge = 5.0, 0.04
    with build_model(solver_device, np.full(shape, 8.0), fixed, stage, dx,
                     recharge, ghb=ghb) as model:
        for transmissivity in (8.0, 16.0):
            model.update_T_in_place(np.full(shape, transmissivity))
            head, info = model.solve(solver="confined_pcg", rel_tol=1e-12, abs_tol_min=1e-12)
            conductance = 0.5 * 2.0 * dx * transmissivity / 4.0
            assert info["converged"]
            np.testing.assert_allclose(head, stage + recharge * dx ** 2 / conductance,
                                       rtol=0, atol=1e-9)


@pytest.mark.parametrize("same_cell", [False, True], ids=["distinct_chd_cells", "one_chd_cell"])
def test_chd_compensates_net_well_rate_per_prescribed_cell(solver_device, same_cell):
    shape = (4, 5)
    initial = np.full(shape, 6.0)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, 0] = 1
    receiving_cell = (1, 0) if same_cell else (2, 0)
    wells = [((1, 0), -3.5), (receiving_cell, 3.5)]
    with build_model(solver_device, initial, fixed, initial, wells=wells) as model:
        head, info = model.solve(solver="unconfined_semismooth_newton_kcycle",
                                 formulation="unconfined", K_field=np.ones(shape),
                                 zbot_field=np.zeros(shape), ztop_field=np.full(shape, 10.0),
                                 initial_head=initial)
        assert info["converged"]
        np.testing.assert_allclose(head, initial, atol=1e-10, rtol=0)
        expected = 0.0 if same_cell else 3.5
        assert info["budget_summary"]["chd_in"] == pytest.approx(expected)
        assert info["budget_summary"]["chd_out"] == pytest.approx(expected)
        assert info["source_budgets"]["wel"] == dict(inflow=3.5, outflow=3.5, net=0.0)
        assert info["source_budgets"]["prescribed_head_wel_exchange"] == dict(
            inflow=expected, outflow=expected, net=0.0)


@pytest.mark.parametrize(("package", "target", "outward"), [
    ("drn", 6.0, 0.6), ("drn", 4.0, 0.0),
    ("riv", 6.0, 0.2), ("riv", 4.0, -0.4),
], ids=["drain_flowing", "drain_closed", "river_draining", "river_below_bed"])
def test_backward_euler_crosses_boundary_threshold(solver_device, package, target, outward):
    """A known change in storage plus the endpoint boundary flow fixes recharge."""
    shape = (4, 5)
    initial = np.full(shape, 5.0)
    fixed = np.zeros(shape, dtype=np.int32)
    mask = np.ones_like(fixed)
    conductance = np.full(shape, 0.4)
    drain = None
    river = None
    if package == "drn":
        drain = dict(drn_mask=mask, drn_elev=np.full(shape, 4.5), drn_cond=conductance)
    else:
        river = dict(riv_mask=mask, riv_stage=np.full(shape, 5.5),
                     riv_rbot=np.full(shape, 4.5), riv_cond=conductance)
    dx, sy = 2.0, 0.2
    recharge = sy * (target - 5.0) + outward / dx ** 2
    with build_model(solver_device, initial, fixed, initial, dx, recharge,
                     drain=drain, river=river) as model:
        head, info = model.solve(solver="unconfined_semismooth_newton_kcycle",
                                 formulation="unconfined", K_field=np.ones(shape),
                                 zbot_field=np.zeros(shape), ztop_field=np.full(shape, 10.0),
                                 initial_head=initial, head_prev=initial, transient=True,
                                 dt=1.0, sy=sy, ss=0.0)
        assert info["converged"]
        np.testing.assert_allclose(head, target, atol=1e-7, rtol=0)
        budget = info["budget_summary"]
        np.testing.assert_allclose([budget[package + "_in"], budget[package + "_out"]],
                                   [max(-outward, 0) * head.size, max(outward, 0) * head.size],
                                   rtol=0, atol=1e-6)
        assert abs(budget["in_minus_out"]) < 1e-6


@pytest.mark.parametrize(("old", "new", "yield_volume", "elastic_volume"), [
    (-1.0, 2.0, 0.4, 0.08),
    (6.5, 3.5, -0.1, -0.475),
], ids=["rewet_through_layer_bottom", "drain_through_layer_top"])
def test_convertible_storage_piecewise_integral(old, new, yield_volume, elastic_volume):
    total, specific_yield, specific_storage = exact_unconfined_storage_terms(
        head_old=np.array([[old]]), head_new=np.array([[new]]), bottom=np.array([[0.0]]),
        top=np.array([[4.0]]), specific_yield=0.2, specific_storage=0.04, dt=2.0)
    assert 2.0 * specific_yield.item() == pytest.approx(yield_volume)
    assert 2.0 * specific_storage.item() == pytest.approx(elastic_volume)
    assert 2.0 * total.item() == pytest.approx(yield_volume + elastic_volume)


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_confined_diffusion_decay_of_discrete_sine_mode(solver_device, implementation):
    """Backward Euler damps the first mode by 1/(1+dt*eigenvalue/storage)."""
    ny, nx, dx = 4, 9, 5.0
    previous = 7.0 + np.tile(np.sin(np.pi * np.arange(nx) / (nx - 1)), (ny, 1))
    fixed = np.zeros((ny, nx), dtype=np.int32)
    fixed[:, [0, -1]] = 1
    transmissivity, dt, storage = 15.0, 2.5, 0.12
    with build_model(solver_device, np.full((ny, nx), transmissivity), fixed,
                     np.full((ny, nx), 7.0), dx) as model:
        head, info = model.solve(transient=True, dt=dt, head_prev=previous, initial_head=previous,
                                 storage_coeff=storage, **confined_options(implementation))
        eigenvalue = 4.0 * transmissivity * np.sin(np.pi / (2 * (nx - 1))) ** 2
        expected = 7.0 + (previous - 7.0) / (1.0 + dt * eigenvalue / (storage * dx ** 2))
        assert info["converged"]
        assert info.get("implementation", "classic") == (
            "fast_transient_face_f64" if implementation == "fast" else "classic")
        np.testing.assert_allclose(head, expected, rtol=0, atol=1e-8)


def closed_history(model, initial, backend, recharge, wells, dt, controls):
    shape = initial.shape
    return model.solve_transient_2d_unconfined(
        solver=backend, initial_head=initial, recharge_rates=recharge, wel_rates=wells,
        k_field=np.ones(shape), zbot_field=np.zeros(shape), ztop_field=np.full(shape, 10.0),
        sy=0.2, ss=0.0, dt=dt, active=np.ones(shape, dtype=np.int32),
        bc_mask=np.zeros(shape, dtype=np.int32), bc_values=initial, solve_controls=controls,
    )


@pytest.mark.parametrize(("backend", "device_picard"), [
    ("unconfined_picard_kcycle", False), ("unconfined_picard_kcycle", True),
    ("unconfined_semismooth_newton_kcycle", False), ("unconfined_fas", False),
], ids=["picard", "device_picard", "newton", "fas"])
def test_history_reverses_wells_and_tracks_each_period_volume(solver_device, backend, device_picard):
    shape = (5, 6)
    initial = np.full(shape, 4.0)
    fixed = np.zeros(shape, dtype=np.int32)
    recharge = [0.02, -0.01, 0.0]
    well_rates = [-0.25, 0.5, -0.5]
    wells = [[(cell, rate) for cell in np.ndindex(shape)] for rate in well_rates]
    controls = dict(unconfined_head_tol=1e-10, strict_head_residual_tol=1e-10,
                    unconfined_relax=1.0, omega=1.0, omega_max=1.0)
    if device_picard:
        controls.update(use_device_transient_fast_path=True, transient_face_operator_enabled=True)
    if backend == "unconfined_fas":
        controls.update(fas_fallback_enabled=False, fas_residual_rms_tolerance=1e-11,
                        fas_head_equivalent_rms_tolerance=1e-11, fas_max_cycles=100)
    with build_model(solver_device, initial, fixed, initial, dx=5.0) as model:
        heads, info = closed_history(model, initial, backend, recharge, wells, 2.0, controls)
        expected = 4.0
        for index, period in enumerate(info["period_infos"]):
            expected += 2.0 * (25.0 * recharge[index] + well_rates[index]) / (25.0 * 0.2)
            assert period["converged"]
            assert period["solver_backend"] == backend
            if device_picard:
                assert period["device_side_picard_fast_path_active"]
            if backend in ("unconfined_fas", "unconfined_semismooth_newton_kcycle"):
                assert period["experimental_fallback_timestep_count"] == 0
                assert abs(period["budget_summary"]["in_minus_out"]) < 1e-7
            np.testing.assert_allclose(heads[index], expected, rtol=0, atol=1e-8)
            np.testing.assert_array_equal(model.accepted_wel_rates(head=heads[index], info=period),
                                          np.full(initial.size, well_rates[index]))
            budgets = period["source_volume_budgets"]
            assert budgets["wel"]["net"] == pytest.approx(2.0 * initial.size * well_rates[index])
            assert budgets["recharge"]["net"] == pytest.approx(50.0 * initial.size * recharge[index])
        assert len(info["period_infos"]) == len(recharge)


def fp32_mixed_layout(device):
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import MixedFastConfig, solve_mixed_fast

    shape = (4, 6)
    t, dx = 20.0, 5.0
    initial = np.full(shape, 8.0)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, 0] = 1
    wells = [((row, 5), -2.0) for row in range(shape[0])]
    expected = initial - 2.0 / t * np.arange(shape[1])
    with build_model(device, np.full(shape, t), fixed, initial, dx, wells=wells) as model:
        for order in ("C", "F"):
            head, info = solve_mixed_fast(
                model=model, initial_head_f64=initial, bc_values_f64=initial, gh_head_f64=None,
                R_f64=np.zeros(shape, order=order), config=MixedFastConfig(
                    min_coarse_cells=4, max_outer=400, rel_tol=1e-12, abs_tol_min=1e-12, dh_rms_tol=1e-10))
            assert info["production_precision_path"] and info["converged"]
            assert info["outer_iterations"] > 0
            np.testing.assert_allclose(head, expected, rtol=0, atol=1e-8)


def fp32_well_schedule(device):
    shape = (5, 6)
    initial = np.full(shape, 5.0)
    entries = []
    for cell in np.ndindex(shape):
        entries.extend([(cell, -100_000_000.0), (cell, 100_000_001.0)])
    with build_model(device, initial, np.zeros(shape, dtype=np.int32), initial, dx=10.0,
                     wells=entries) as model:
        for schedule in (None, [entries]):
            heads, info = closed_history(model, initial, "unconfined_picard_kcycle",
                                         [0.0], schedule, 1.0,
                                         dict(unconfined_head_tol=1e-5, strict_head_residual_tol=1e-5,
                                              unconfined_relax=1.0, omega=1.0, omega_max=1.0))
            assert info["period_infos"][0]["converged"]
            assert info["period_infos"][0]["source_budgets"]["wel"]["net"] == initial.size
            np.testing.assert_allclose(heads[0], 5.05, rtol=0, atol=1e-6)


@pytest.mark.parametrize("scenario", ["mixed_layout", "scheduled_wells"])
def test_fp32_production_forcing(solver_device, scenario):
    script = Path(__file__).resolve()
    search_path = [str(script.parents[1])]
    if os.environ.get("PYTHONPATH"):
        search_path.append(os.environ["PYTHONPATH"])
    environment = {**os.environ, "DARCY_FLOAT": "float32", "PYTHONPATH": os.pathsep.join(search_path)}
    result = subprocess.run([sys.executable, str(script), scenario, str(solver_device)],
                            env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_invalid_well_update_preserves_accepted_drawdown(solver_device):
    shape = (4, 6)
    initial = np.full(shape, 8.0)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, 0] = 1
    wells = [((row, 5), -2.0) for row in range(shape[0])]
    with build_model(solver_device, np.full(shape, 20.0), fixed, initial, wells=wells) as model:
        with pytest.raises(ValueError):
            model.update_wel_in_place(wel_entries=[((1, 1), 3.0), ((4, 5), -100.0)])
        head, info = model.solve(solver="confined_pcg", rel_tol=1e-12, abs_tol_min=1e-12)
        assert info["converged"]
        np.testing.assert_allclose(head, initial - 0.1 * np.arange(shape[1]), rtol=0, atol=1e-8)
        np.testing.assert_array_equal(model.accepted_wel_rates(head=head, info=info), [-2.0] * 4)


def test_accepted_newton_history_rejects_altered_physical_duration(solver_device):
    shape = (4, 5)
    initial = np.full(shape, 6.0)
    fixed = np.zeros(shape, dtype=np.int32)
    entries = [(cell, -0.25) for cell in np.ndindex(shape)]
    with build_model(solver_device, initial, fixed, initial, wells=entries) as model:
        heads, info = closed_history(model, initial, "unconfined_semismooth_newton_kcycle",
                                     [0.0], None, 2.0, {})
        period = info["period_infos"][0]
        assert period["converged"] and period["period_dt"] == 2.0
        np.testing.assert_allclose(heads[0], 5.9, rtol=0, atol=1e-8)
        invalid = dict(period, period_dt=200.0)
        with pytest.raises(ValueError):
            model.accepted_wel_rates(head=heads[0], info=invalid)


def test_disconnected_strips_keep_independent_head_datums_and_well_rates(solver_device):
    shape = (5, 6)
    active = np.ones(shape, dtype=np.int32)
    active[2, :] = 0
    fixed = np.zeros_like(active)
    fixed[:, 0] = active[:, 0]
    datum = np.zeros(shape)
    datum[:2, :] = 7.0
    datum[3:, :] = 16.0
    wells = [((0, 5), 3.0), ((1, 5), 3.0), ((3, 5), -1.5), ((4, 5), -1.5)]
    with build_model(solver_device, np.full(shape, 10.0), fixed, datum, wells=wells,
                     active=active) as model:
        head, info = model.solve(solver="confined_pcg", rel_tol=1e-12, abs_tol_min=1e-12)
        assert info["converged"]
        for row, rate in ((0, 3.0), (1, 3.0), (3, -1.5), (4, -1.5)):
            np.testing.assert_allclose(head[row], datum[row] + rate / 10.0 * np.arange(shape[1]),
                                       rtol=0, atol=1e-8)
        assert info["source_budgets"]["wel"] == dict(inflow=6.0, outflow=3.0, net=3.0)


if __name__ == "__main__":
    if os.environ["DARCY_FLOAT"] != "float32":
        raise RuntimeError("Run floating-point forcing probes in an FP32 interpreter")
    workers = {"mixed_layout": fp32_mixed_layout, "scheduled_wells": fp32_well_schedule}
    workers[sys.argv[1]](sys.argv[2])
