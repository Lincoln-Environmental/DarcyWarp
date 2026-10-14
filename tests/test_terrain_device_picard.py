"""Regression checks for the device Picard controller and physical state."""
from types import SimpleNamespace

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.solvers.terrain_device_3d import (
    DevicePicardControls, DeviceTerrainPicard, DeviceTerrainState)
from DARCY_WARP_PACKAGE.solvers.terrain_fast_3d import (
    pack_device_subfaces, pack_device_contact_refresh)
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import (
    TerrainReferenceOperator3D, TerrainContact3D)
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d
from test_3d_terrain_gpu import build_solver
from DARCY_WARP_PACKAGE.solvers.terrain_gmres_3d import TerrainGMRES3D
from test_3d_scientific_audit import device


@pytest.fixture(params=["cpu", "cuda:0"])
def terrain_test_device(request, monkeypatch):
    wp.init()
    backend = request.param
    if backend.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")

    def selected_device():
        return backend

    monkeypatch.setattr("test_3d_terrain_gpu.device", selected_device)
    return backend


def reuse_controls():
    return {"unconfined": True, "device_picard": True, "max_cycles": 13,
        "krylov_restart": 4, "max_levels": 3, "min_coarse_n": 2,
        "unconfined_max_picard_iter": 100, "rel_tol": 1e-12, "abs_tol_min": 1e-12,
        "final_head_residual_tol": 1e-9, "final_budget_relative_tol": 1e-9}


def assert_installed_terrain_state(*, solver, head, info, expected_fixed):
    session = solver._session
    workspace = session.workspace
    np.testing.assert_array_equal(workspace.active.numpy().reshape(session.shape), session.grid.active)
    np.testing.assert_array_equal(workspace.fixed.numpy().reshape(session.shape), expected_fixed)
    np.testing.assert_array_equal(session.snapshot[6], expected_fixed)
    np.testing.assert_array_equal(session.snapshot[7], session.grid.active)
    np.testing.assert_array_equal(session.snapshot_packed_positive,
        workspace.coefficients.numpy()[:, (0, 3, 6)] > 0)
    np.testing.assert_array_equal(workspace.levels[0]["active"].numpy(), session.grid.active)
    np.testing.assert_array_equal(workspace.levels[0]["bc_mask"].numpy(), expected_fixed)
    np.testing.assert_array_equal(head[session.bc_mask != 0], session.bc_values[session.bc_mask != 0])
    np.testing.assert_array_equal(head[session.grid.active == 0], 0.)
    reference = TerrainReferenceOperator3D(metrics=session.metrics, kx=session.kx,
        ky=session.ky, kz=session.kz, head=head, contacts=session.contacts)
    budget = compute_terrain_budget_3d(operator=reference, head=head, source=session.rhs,
        fixed=expected_fixed, storage_change=np.zeros(session.shape),
        storage_diagonal=np.zeros(session.shape), return_fields=True)
    assert info["converged"], info
    assert budget["head_residual_rms"] <= 1e-9
    assert budget["cell_residual_rms"] <= 1e-12 or budget["relative_cell_imbalance"] <= 1e-9
    for field in ("cell_residual_rms", "head_residual_rms", "relative_cell_imbalance"):
        assert info["budget"][field] == pytest.approx(budget[field], rel=1e-10, abs=1e-14)


def assert_workspace_transition(*, old_workspace, new_workspace, reused, backend):
    if reused:
        assert new_workspace is old_workspace
    else:
        assert new_workspace is not old_workspace
        assert old_workspace.krylov._cycle_graph is None
        assert old_workspace.krylov._A is None and old_workspace.krylov._M is None
    if backend.startswith("cuda"):
        assert old_workspace.krylov.graph_captures == 1
        assert new_workspace.krylov.graph_captures == 1
        assert new_workspace.krylov.graph_replays > 0


def test_device_workspace_rebuilds_when_contact_connects_isolated_cell(terrain_test_device):
    grid = TerrainGrid3D(surface_elevation=np.zeros((1, 5)), zeta_faces=np.array([0., 1.]),
        active_columns=np.array([[1, 1, 1, 0, 1]]), dx=1., dy=1.)
    fixed = np.array([1, 0, 0, 0, 0], dtype=np.int32).reshape(grid.shape)
    initial = np.array([2., 2., 2., 0., 7.]).reshape(grid.shape)
    source = np.array([0., .01, .01, 0., 0.]).reshape(grid.shape)
    contact = TerrainContact3D(cell=(0, 0, 4), direction="east", area=1.,
        distance=.5, elevation=-.5, head=4.)
    options = reuse_controls()
    with build_solver(grid=grid, kx=1., ky=1., kz=1., fixed=fixed, values=initial,
        source=source, initial=initial) as solver:
        old_head, old_info = solver.solve(**options)
        expected_fixed = fixed.copy()
        expected_fixed[0, 0, 4] = 1
        assert old_info["n_isolated_cells_excluded"] == 1
        assert old_head[0, 0, 4] == 7.
        assert_installed_terrain_state(solver=solver, head=old_head, info=old_info,
            expected_fixed=expected_fixed)
        old_workspace = solver._session.workspace
        solver._session.update_contacts(contacts=(contact,))
        head, info = solver.solve(**options)
        assert info["n_isolated_cells_excluded"] == 0
        assert head[0, 0, 4] == pytest.approx(4., abs=1e-9)
        assert_installed_terrain_state(solver=solver, head=head, info=info, expected_fixed=fixed)
        assert not info["workspace_reused"]
        assert_workspace_transition(old_workspace=old_workspace,
            new_workspace=solver._session.workspace, reused=False, backend=terrain_test_device)
        with build_solver(grid=grid, kx=1., ky=1., kz=1., fixed=fixed, values=initial,
            source=source, initial=initial, contacts=(contact,)) as fresh:
            fresh_head, fresh_info = fresh.solve(**options)
            assert_installed_terrain_state(solver=fresh, head=fresh_head, info=fresh_info,
                expected_fixed=fixed)
        np.testing.assert_allclose(head, fresh_head, rtol=0., atol=2e-9)


@pytest.mark.parametrize("update", ["active", "reactivate", "geometry", "source", "missing_snapshot"])
def test_device_workspace_topology_update_and_numeric_reuse(terrain_test_device, update):
    columns = np.ones((3, 6), dtype=bool)
    if update == "reactivate":
        columns[1, 2] = False
    grid = TerrainGrid3D(surface_elevation=np.zeros((3, 6)), zeta_faces=np.array([0., 1., 3.]),
        active_columns=columns, dx=2., dy=3.)
    fixed = np.zeros(grid.shape, dtype=np.int32)
    fixed[:, :, 0] = fixed[:, :, -1] = 1
    values = np.full(grid.shape, 3.)
    values[:, :, 0], values[:, :, -1] = 2., 4.
    initial = np.full(grid.shape, 3.)
    source = np.zeros(grid.shape)
    options = reuse_controls()
    with build_solver(grid=grid, kx=1., ky=.6, kz=.1, fixed=fixed, values=values,
        source=source, initial=initial) as solver:
        _unused_head, old_info = solver.solve(**options)
        assert old_info["converged"]
        old_workspace = solver._session.workspace
        old_graph = getattr(old_workspace.krylov, "_cycle_graph", None)
        old_replays = getattr(old_workspace.krylov, "graph_replays", 0)
        reused = update in ("geometry", "source")
        if update in ("active", "reactivate"):
            columns = grid.active_columns.copy()
            columns[1, 2] = update == "reactivate"
            grid = TerrainGrid3D(z_interfaces=grid.z_interfaces, dx=grid.dx, dy=grid.dy,
                active_columns=columns)
            solver._session.update_geometry(grid=grid)
        elif update == "geometry":
            y, x = np.mgrid[:3, :6]
            grid = TerrainGrid3D(surface_elevation=.1*x+.05*y,
                zeta_faces=np.array([0., 1., 3.]), active_columns=columns,
                dx=grid.dx, dy=grid.dy)
            solver._session.update_geometry(grid=grid)
        else:
            source[fixed == 0] = .01
            solver.update_rhs(rhs=source)
            if update == "missing_snapshot":
                solver._session.snapshot = None
        head, info = solver.solve(**options)
        assert_installed_terrain_state(solver=solver, head=head, info=info, expected_fixed=fixed)
        assert info["workspace_reused"] == reused
        workspace = solver._session.workspace
        assert_workspace_transition(old_workspace=old_workspace, new_workspace=workspace,
            reused=reused, backend=terrain_test_device)
        if reused and terrain_test_device.startswith("cuda"):
            assert workspace.krylov._cycle_graph is old_graph
            assert workspace.krylov.graph_replays > old_replays
        with build_solver(grid=grid, kx=1., ky=.6, kz=.1, fixed=fixed, values=values,
            source=source, initial=initial) as fresh:
            fresh_head, fresh_info = fresh.solve(**options)
            assert_installed_terrain_state(solver=fresh, head=fresh_head, info=fresh_info,
                expected_fixed=fixed)
            assert len(workspace.levels) == len(fresh._session.workspace.levels)
            for installed, expected in zip(workspace.levels, fresh._session.workspace.levels):
                for field in ("active", "bc_mask"):
                    np.testing.assert_array_equal(installed[field].numpy(), expected[field].numpy())
        np.testing.assert_allclose(head, fresh_head, rtol=0., atol=2e-9)
        if update == "missing_snapshot":
            # The shared host check must also fail safely when its cache
            # metadata is absent, rather than indexing a missing snapshot.
            solver._session.snapshot = None
            host_options = dict(options, device_picard=False)
            host_head, host_info = solver.solve(**host_options)
            assert not host_info["workspace_reused"]
            assert_installed_terrain_state(solver=solver, head=host_head,
                info=host_info, expected_fixed=fixed)
            np.testing.assert_allclose(host_head, fresh_head, rtol=0., atol=2e-9)


class ScriptedKrylov:
    def __init__(self, *, state, results):
        self.state = state
        self.results = iter(results)
        self.budgets = []
        self.tolerances = []

    def __call__(self):
        self.budgets.append(self._maxiter)
        self.tolerances.append(self._atol)
        used, error_l2, delta = next(self.results)
        self.state.delta.assign(np.array([delta]))
        return used, error_l2, self._atol


def controller(*, results, budget, restart, sqrt_free):
    driver = DeviceTerrainPicard.__new__(DeviceTerrainPicard)
    driver.state = SimpleNamespace(delta=wp.zeros(1, dtype=wp.float64, device=device()),
        delta_snapshot=wp.zeros(1, dtype=wp.float64, device=device()))
    driver.controls = DevicePicardControls()
    driver.krylov_budget = budget
    driver.restart = restart
    driver.sqrt_free = sqrt_free
    krylov = ScriptedKrylov(state=driver.state, results=results)
    driver.workspace = SimpleNamespace(krylov=krylov)
    return driver, krylov


def test_inner_norms_and_nonmultiple_budget():
    driver, krylov = controller(results=[(8, 200., 1.), (3, 50., 2.)],
        budget=11, restart=4, sqrt_free=10.)
    result = driver.run_inner(initial_rms=100., target_rms=.01)
    assert krylov.budgets == [8, 3]
    assert krylov.tolerances == [.1, .1]
    assert result["cycles"] == 11
    assert result["restart_blocks"] == 3
    assert result["head_rms"] == 5.
    assert result["usable"] and not result["diverged"]


def test_reached_reports_latest_norm_without_zero_iteration_loop():
    driver, krylov = controller(results=[(3, .002, 1.)],
        budget=11, restart=4, sqrt_free=2.)
    result = driver.run_inner(initial_rms=1., target_rms=.01)
    assert result["reached"] and result["head_rms"] == .001
    assert result["cycles"] == 3
    assert len(krylov.budgets) == 1


def test_diverging_block_rolls_back_but_retains_useful_previous_correction():
    driver, unused_krylov = controller(results=[(8, 5., 1.), (3, 20., 2.)],
        budget=11, restart=4, sqrt_free=1.)
    result = driver.run_inner(initial_rms=10., target_rms=.01)
    assert result["diverged"] and result["usable"]
    assert result["head_rms"] == 5.
    np.testing.assert_array_equal(driver.state.delta.numpy(), [1.])


def test_retry_uses_remaining_budget_and_accepts_reduction_without_target():
    driver, krylov = controller(results=[(8, np.nan, 5.), (3, 9., 3.)],
        budget=11, restart=4, sqrt_free=1.)
    result = driver.run_inner(initial_rms=10., target_rms=.01)
    assert krylov.budgets == [8, 3]
    assert result["retried"] and result["usable"] and not result["reached"]
    assert result["cycles"] == 11 and result["head_rms"] == 9.
    np.testing.assert_array_equal(driver.state.delta.numpy(), [3.])


def test_zero_budget_and_zero_progress_terminate():
    driver, krylov = controller(results=[], budget=0, restart=4, sqrt_free=1.)
    result = driver.run_inner(initial_rms=1., target_rms=.01)
    assert not result["usable"] and result["cycles"] == 0
    assert not krylov.budgets
    driver, krylov = controller(results=[(0, 1., 0.), (0, 1., 0.)],
        budget=11, restart=4, sqrt_free=1.)
    result = driver.run_inner(initial_rms=1., target_rms=.01)
    assert not result["usable"] and result["cycles"] == 0
    assert len(krylov.budgets) == 2


def test_acceptance_requires_physical_balance_and_finite_residuals():
    driver = DeviceTerrainPicard.__new__(DeviceTerrainPicard)
    driver.head_tolerance = 1e-6
    driver.abs_tol_min = 1e-12
    driver.budget_relative_tol = 1e-8
    current = dict(head=1., flow=1., relative=1., finite=True)
    candidate = dict(head=.5, flow=2., relative=1., finite=True)
    assert not driver.candidate_improves(current=current, candidate=candidate)
    assert not driver.physically_converged(metrics=dict(head=1e-7, flow=1e-11,
        relative=1e-7, finite=True))
    assert driver.physically_converged(metrics=dict(head=1e-7, flow=1e-11,
        relative=1e-9, finite=True))
    assert not driver.physically_converged(metrics=dict(head=0., flow=0.,
        relative=0., finite=False))
    # Finite physical norms can still overflow when divided by a tiny
    # accepted-state scale; reject their merit without raising OverflowError.
    assert not driver.candidate_improves(
        current=dict(head=1e-6, flow=1e-12, finite=True),
        candidate=dict(head=1e150, flow=1e150, relative=1., finite=True))


@pytest.fixture()
def physical_driver():
    y, x = np.mgrid[:3, :4]
    top = 20.+.4*x+.2*y
    middle = top-(3.+.1*x)
    bottom = middle-(6.+.1*y)
    grid = TerrainGrid3D(z_interfaces=np.stack((top, middle, bottom)), dx=5., dy=5.)
    contacts = (
        TerrainContact3D(cell=(0, 0, 1), direction="top", area=2., distance=.5,
            elevation=float(grid.elevations[0, 0, 1]+.5), head=18., conductance=.3),
        TerrainContact3D(cell=(1, 1, 0), direction="west", area=1., distance=2.5,
            elevation=float(grid.elevations[1, 1, 0]), head=12., kind="discharge_only"))
    fixed = np.zeros(grid.shape, dtype=np.int32)
    fixed[:, :, -1] = 1
    initial = grid.bottoms+.6*grid.thickness
    source = np.full(grid.shape, .01)
    k = np.ones(grid.shape)
    k[-1] = .001
    with build_solver(grid=grid, kx=k, ky=.6*k, kz=.1*k, fixed=fixed,
        values=initial, source=source, initial=initial, contacts=contacts) as solver:
        solver.solve(max_cycles=0)
        session = solver._session
        reference = TerrainReferenceOperator3D(metrics=session.metrics, kx=k, ky=.6*k,
            kz=.1*k, head=initial, contacts=contacts)
        rows, unused_count = pack_device_subfaces(operator=reference, datum=20.)
        state = DeviceTerrainState(session=session, device=device(), workspace=session.workspace)
        state.upload_static(source=source, sy=np.full(grid.shape, .2),
            ss=np.full(grid.shape, 1e-5), free=(grid.active != 0)&(fixed == 0),
            subfaces=rows, contact_refresh=pack_device_contact_refresh(operator=reference, datum=20.))
        state.update_head(host_head=initial)
        state.update_previous(host_previous=initial)
        driver = DeviceTerrainPicard(session=session, state=state, workspace=session.workspace,
            device=device(), controls=DevicePicardControls(), datum=20., restart=4,
            krylov_budget=40, unconfined=True, transient=True, dt=1., abs_tol_min=1e-12,
            head_tolerance=1e-9, budget_relative_tol=1e-10, rel_tol=1e-12,
            min_sat=.01, outer_limit=10, unconfined_relax=.7)
        yield driver
        state.close()


@pytest.mark.parametrize("fraction", [0., .2, .7, 1.])
def test_candidate_contacts_and_physical_residual_match_independent_reference(physical_driver, fraction):
    driver = physical_driver
    s = driver.state
    session = driver.session
    original = s.head.numpy().copy()
    candidate = session.grid.bottoms+fraction*session.grid.thickness
    candidate[session.bc_mask != 0] = session.bc_values[session.bc_mask != 0]
    s.candidate.assign(candidate.ravel())
    driver.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
    original_rel = s.head_rel.numpy().copy()
    driver.evaluate(head_buffer=s.candidate, rel_buffer=s.candidate_rel)
    driver.physical_residual(head=s.candidate_rel)
    operator = TerrainReferenceOperator3D(metrics=session.metrics, kx=session.kx,
        ky=session.ky, kz=session.kz, head=candidate, min_sat=.01, contacts=session.contacts)
    expected_rows, unused_count = pack_device_subfaces(operator=operator, datum=driver.datum)
    actual = np.stack((s.sub_c.numpy(), s.sub_bz.numpy(), s.sub_bx.numpy(), s.sub_by.numpy()), axis=1)
    np.testing.assert_allclose(actual, expected_rows[:, 1:5], rtol=2e-13, atol=2e-14)
    flux = operator.fluxes(head=candidate)[3]
    assert s.sub_open.numpy()[1] == int(flux[1] > 0)
    storage, change = session.storage_terms(head=candidate, previous=original.reshape(session.shape),
        transient=True, unconfined=True, mode="secant_sy", sy=np.full(session.shape, .2),
        ss=np.full(session.shape, 1e-5), coeff=None, dt=1.)
    budget = compute_terrain_budget_3d(operator=operator, head=candidate, source=session.rhs,
        fixed=session.bc_mask, storage_change=change, storage_diagonal=storage, return_fields=True)
    np.testing.assert_allclose(s.residual.numpy().reshape(session.shape),
        budget["cell_residual_field"], rtol=2e-12, atol=2e-12)
    combined = driver.dual_check(head_buffer=s.head_rel, candidate_buffer=s.candidate_rel)
    metrics = driver.residual_metrics(combined=combined)
    assert driver.n_free == 18
    assert metrics["head"] == pytest.approx(budget["head_residual_rms"], rel=2e-12)
    assert metrics["flow"] == pytest.approx(budget["cell_residual_rms"], rel=2e-12)
    assert metrics["relative"] == pytest.approx(budget["relative_cell_imbalance"], rel=2e-12)
    np.testing.assert_array_equal(s.head.numpy(), original)
    np.testing.assert_array_equal(s.head_rel.numpy(), original_rel)


def reject_candidate(*, current, candidate):
    return False


def accept_candidate(*, current, candidate):
    return True


def test_backtrack_restores_full_state_and_reports_true_head_rms(physical_driver, monkeypatch):
    driver = physical_driver
    s = driver.state
    driver.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
    driver.physical_residual(head=s.head_rel)
    combined = driver.dual_check(head_buffer=s.head_rel, candidate_buffer=s.head_rel)
    current = driver.residual_metrics(combined=combined)
    arrays = [s.head, s.head_rel, s.coefficients, s.storage, s.scale, s.boundary,
        s.sub_c, s.sub_bx, s.sub_by, s.sub_bz, s.sub_open, s.rhs, s.residual]
    for level in driver.workspace.levels:
        arrays.extend((*level["faces"], level["storage"], level["diag"]))
    snapshots = [array.numpy().copy() for array in arrays]
    correction = np.linspace(.1, 1., s.n)
    s.delta.assign(correction)
    monkeypatch.setattr(driver, "candidate_improves", reject_candidate)
    outcome = driver.backtrack(current=current)
    assert not outcome["accepted"] and outcome["backtracks"] == 12
    for array, expected in zip(arrays, snapshots):
        np.testing.assert_allclose(array.numpy(), expected, rtol=2e-13, atol=2e-13)
    monkeypatch.setattr(driver, "candidate_improves", accept_candidate)
    outcome = driver.backtrack(current=current)
    free = s.free.numpy() != 0
    assert outcome["dh_rms"] == pytest.approx(np.sqrt(np.mean((.7*correction[free])**2)))
    assert outcome["dh_max"] == pytest.approx(np.max(.7*correction[free]))


def scalar_driver(*, starting_head, correction, residual_function):
    driver = DeviceTerrainPicard.__new__(DeviceTerrainPicard)
    driver.device = device()
    driver.n_free = 1
    driver.outer_limit_val = 10
    driver.unconfined_relax = 1.
    driver.controls = DevicePicardControls()
    driver.head_tolerance = 1e-6
    driver.abs_tol_min = 1e-12
    driver.budget_relative_tol = 1e-8
    head = wp.array(np.array([starting_head]), dtype=wp.float64, device=device())
    driver.state = SimpleNamespace(n=1, head=head, head_rel=wp.clone(head),
        starting=wp.zeros_like(head), candidate=wp.zeros_like(head),
        candidate_rel=wp.zeros_like(head), delta=wp.array(np.array([correction]),
        dtype=wp.float64, device=device()), fixed=wp.zeros(1, dtype=wp.int32, device=device()),
        bc_values=wp.zeros_like(head), active=wp.ones(1, dtype=wp.int32, device=device()))

    def evaluate(*, head_buffer, rel_buffer):
        wp.copy(src=head_buffer, dest=rel_buffer)

    def physical_residual(*, head):
        return None

    def dual_check(*, head_buffer, candidate_buffer):
        value = float(candidate_buffer.numpy()[0])
        original = float(head_buffer.numpy()[0])
        residual = residual_function(value=value)
        return np.array([(value-original)**2, abs(value-original), residual**2,
            residual**2, 1., 0., abs(residual), abs(correction)])

    driver.evaluate = evaluate
    driver.physical_residual = physical_residual
    driver.dual_check = dual_check
    return driver


def square_minus_one(*, value):
    return value*value-1.


def square_plus_one(*, value):
    return value*value+1.


def test_first_candidate_backtracks_against_current_residual():
    driver = scalar_driver(starting_head=.2, correction=3., residual_function=square_minus_one)
    update = driver.backtrack(current=dict(head=.96, flow=.96, relative=.96, finite=True))
    assert update["accepted"] and update["backtracks"] == 2
    assert driver.state.head.numpy()[0] == pytest.approx(.95)
    assert update["metrics"]["flow"] < .96


def test_histories_budgets_and_host_device_handoff(physical_driver, monkeypatch):
    session = physical_driver.session
    previous = physical_driver.state.head.numpy().reshape(session.shape)
    options = dict(device=device(), unconfined=True, transient=True, dt=1.,
        sy=.2, ss=1e-5, head_prev=previous, max_cycles=13, krylov_restart=4,
        max_levels=3, min_coarse_n=2, unconfined_max_picard_iter=100,
        final_head_residual_tol=1e-8, final_budget_relative_tol=1e-9)
    unused_head, info = session.solve(device_picard=True, **options)
    assert info["n_free_cells"] == 18
    history = info["outer_history"]
    assert [row["outer_iteration"] for row in history] == list(range(1, len(history)+1))
    assert info["n_outer_iterations"] == len(history)
    assert info["krylov_iterations"] == sum(row["krylov_iterations"] for row in history)
    assert all(row["krylov_iterations"] <= 13 for row in history)
    assert all(row["backtracking_evaluations"] == len(row["backtracking_trials"]) for row in history)
    assert all(row["backtracks"] == len(row["backtracking_trials"])-int(row["accepted"]) for row in history)
    assert info["gmres_restart_blocks"] == sum(row["restart_blocks"] for row in history)
    assert info["multigrid_cycles"] == info["krylov_iterations"]+info["gmres_restart_blocks"]
    if history:
        assert history[-1]["head_residual_rms"] == pytest.approx(info["budget"]["head_residual_rms"], abs=1e-12)
        assert history[-1]["physical_residual_rms"] == pytest.approx(info["budget"]["cell_residual_rms"], abs=1e-12)
    # Switching drivers on an existing session must preserve valid cache
    # snapshots and refresh the principal hierarchy/row scaling as needed.
    workspace = session.workspace
    assert 0 < workspace.krylov._maxiter < options["max_cycles"]
    host_budgets = []
    original_call = TerrainGMRES3D.__call__

    def record_host_budget(self, *args, **kwargs):
        host_budgets.append(self._maxiter)
        return original_call(self, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TerrainGMRES3D, "__call__", record_host_budget)
        host_head, host_info = session.solve(device_picard=False, **options)
    assert session.workspace is workspace and host_info["workspace_reused"]
    assert host_budgets and set(host_budgets) == {options["max_cycles"]}
    with build_solver(grid=session.grid, kx=session.kx, ky=session.ky, kz=session.kz,
        fixed=session.bc_mask, values=session.bc_values, source=session.rhs,
        initial=session.initial_head, contacts=session.contacts) as fresh:
        fresh_head, fresh_info = fresh._session.solve(device_picard=False, **options)
    assert host_info["converged"] and fresh_info["converged"]
    np.testing.assert_allclose(host_head, fresh_head, rtol=0., atol=2e-8)
    unused_head, device_info = session.solve(device_picard=True, **options)
    assert session.workspace is workspace and device_info["workspace_reused"]
    assert device_info["converged"]


def test_original_small_source_is_not_lost_to_transient_rhs_rounding():
    grid = TerrainGrid3D(surface_elevation=np.array([[1e8]]),
        zeta_faces=np.array([0., 1.]), dx=1., dy=1.)
    previous = np.full(grid.shape, 1e8)
    source = np.full(grid.shape, 1e-8)
    with build_solver(grid=grid, kx=1., ky=1., kz=1.,
        fixed=np.zeros(grid.shape, dtype=np.int32), values=np.zeros(grid.shape),
        source=source, initial=previous) as solver:
        head, info = solver.solve(unconfined=True, transient=True, dt=1., sy=.2, ss=1.,
            head_prev=previous, device_picard=True, max_cycles=13,
            unconfined_max_picard_iter=10, final_budget_relative_tol=1e-9,
            final_head_residual_tol=1e-12)
    assert not info["converged"]
    np.testing.assert_array_equal(head, previous)
    assert info["budget"]["cell_residual_rms"] == pytest.approx(1e-8)
    assert info["budget"]["relative_cell_imbalance"] == pytest.approx(1.)



def test_bounded_recovery_retains_picard_step_without_relaxing_final_gate():
    driver = scalar_driver(starting_head=0., correction=.1, residual_function=square_plus_one)
    current = dict(head=1., flow=1., relative=1., finite=True)
    rejected = driver.backtrack(current=current)
    assert not rejected["accepted"]
    assert driver.state.head.numpy()[0] == 0.
    recovered = driver.backtrack(current=current, allow_growth=True)
    assert recovered["accepted"] and recovered["acceptance_mode"] == "bounded_recovery"
    assert recovered["metrics"]["flow"] > current["flow"]
    old_merit, new_merit = driver.residual_merit(current=current, candidate=recovered["metrics"])
    assert new_merit <= driver.controls.nonlinear_growth_limit**2*old_merit
    assert not driver.physically_converged(metrics=recovered["metrics"])



def test_shallow_aquifer_32_converges_through_bounded_picard_recovery():
    """A monotone-only device line search formerly stopped at outer 20.
    The same equations admit a strict physical solution with recovery.
    """
    from working_tests.shanz_terrain_case_study import build_case

    case = build_case(nx=32, ny=32, dx=20.)
    grid = TerrainGrid3D(z_interfaces=case["interfaces"], dx=case["dx"], dy=case["dx"],
        active_columns=case["active"])
    with build_solver(grid=grid, kx=case["k"], ky=.6*case["k"], kz=.15*case["k"],
        fixed=case["fixed"], values=case["values"], source=case["source"],
        initial=case["initial"], contacts=case["contacts"]) as solver:
        unused_head, info = solver.solve(unconfined=True, device_picard=True,
            unconfined_min_sat=.01, unconfined_head_tol=1e-6,
            final_head_residual_tol=1e-6, rel_tol=1e-12, abs_tol_min=1e-12,
            final_budget_relative_tol=1e-8, unconfined_max_picard_iter=200,
            unconfined_relax=.7, max_cycles=2000, krylov_restart=24,
            max_levels=6, min_coarse_n=16)
    assert info["converged"], info["budget"]
    assert info["budget"]["head_residual_rms"] <= 1e-6
    assert info["budget"]["relative_cell_imbalance"] <= 1e-8
    assert any(row["acceptance_mode"] == "bounded_recovery" for row in info["outer_history"])



@pytest.mark.parametrize("mirrored", [False, True])
def test_isolated_exclusion_preserves_head_and_normalizes_only_free_equations(mirrored):
    grid = TerrainGrid3D(surface_elevation=np.zeros((1, 3)), zeta_faces=np.array([0., 1.]),
        dx=1., dy=1.)
    fixed = np.array([1, 0, 0], dtype=np.int32).reshape(grid.shape)
    initial = np.array([1., 2., 7.]).reshape(grid.shape)
    values = np.array([1., 0., 0.]).reshape(grid.shape)
    kx = np.array([1., 1., 0.]).reshape(grid.shape)
    isolated_index = 2
    if mirrored:
        fixed = fixed[:, :, ::-1].copy()
        initial = initial[:, :, ::-1].copy()
        values = values[:, :, ::-1].copy()
        kx = kx[:, :, ::-1].copy()
        isolated_index = 0
    with build_solver(grid=grid, kx=kx, ky=1., kz=1., fixed=fixed,
        values=values, source=np.zeros(grid.shape), initial=initial) as solver:
        head, info = solver.solve(device_picard=True, unconfined=True, max_cycles=7,
            krylov_restart=4, final_head_residual_tol=1e-8, abs_tol_min=1e-10)
    assert info["converged"]
    assert info["n_free_cells"] == 1
    assert info["n_isolated_cells_excluded"] == 1
    assert head.ravel()[isolated_index] == 7.
    assert head.ravel()[1] == pytest.approx(1., abs=1e-8)
    assert info["outer_history"][-1]["head_residual_rms"] == pytest.approx(
        info["budget"]["head_residual_rms"], abs=1e-12)



def test_head_change_diagnostics_distinguish_raw_clipped_and_accepted():
    driver = scalar_driver(starting_head=.2, correction=1000., residual_function=square_minus_one)
    current = dict(head=.96, flow=.96, relative=.96, finite=True)
    update = driver.backtrack(current=current)
    assert update["accepted"]
    assert update["proposed_max"] == 1000.
    assert update["clipped_max"] == pytest.approx(driver.controls.max_head_change_per_outer)
    assert update["dh_max"] < update["clipped_max"]
    assert update["dh_rms"] == pytest.approx(update["dh_max"])



def test_restart_accounting_uses_actual_allocated_workspace(physical_driver):
    driver = physical_driver
    s = driver.state
    driver.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
    driver.physical_residual(head=s.head_rel)
    combined = driver.dual_check(head_buffer=s.head_rel, candidate_buffer=s.head_rel)
    current = driver.residual_metrics(combined=combined)
    s.rhs.assign((s.residual.numpy()/s.scale.numpy()))
    s.delta.zero_()
    callback_blocks = []

    def count_restart(iteration, error, tolerance):
        callback_blocks.append((iteration, error, tolerance))

    driver.workspace.krylov._callback = count_restart
    result = driver.run_inner(initial_rms=current["head"], target_rms=1e-12)
    assert result["restart_blocks"] == len(callback_blocks)
    assert result["cycles"] <= driver.krylov_budget
