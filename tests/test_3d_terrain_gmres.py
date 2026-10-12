"""Terrain GMRES iteration-budget semantics: max_cycles is exact Krylov work."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
from DARCY_WARP_PACKAGE.solvers.terrain_fast_3d import TerrainGPUWorkspace3D
from test_3d_scientific_audit import device


def flat_workspace(*, restart, maxiter, tol=1e-11, atol=1e-12):
    grid = TerrainGrid3D(surface_elevation=np.zeros((4, 6)), zeta_faces=np.array([0., 1., 2.]), dx=2., dy=3.)
    reference = TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=grid), kx=1., ky=1., kz=.1)
    fixed = np.zeros(grid.shape, dtype=np.int32)
    fixed[:, :, 0] = fixed[:, :, -1] = 1
    values = np.zeros(grid.shape)
    values[:, :, 0], values[:, :, -1] = 3., 1.
    boundary = np.zeros((int(np.prod(grid.shape)), 6))
    workspace = TerrainGPUWorkspace3D(operator=reference, fixed=fixed, storage=np.zeros(grid.shape),
        boundary=boundary, device=device(), max_levels=3, min_coarse_n=2,
        restart=restart, maxiter=maxiter, rel_tol=tol, abs_tol=atol)
    rhs = np.zeros(grid.shape)
    rhs[fixed != 0] = values[fixed != 0]
    scale = np.ones(grid.shape)
    workspace.b.assign((rhs/scale).ravel())
    return workspace


def run_gmres(workspace):
    workspace.x.zero_()
    used, error, tolerance = workspace.krylov()
    return used, error, tolerance, workspace.x.numpy().reshape(workspace.shape)


@pytest.fixture()
def reference_solution():
    workspace = flat_workspace(restart=4, maxiter=400)
    used, error, tolerance, solution = run_gmres(workspace)
    assert error <= tolerance
    workspace.close()
    return used, tolerance, solution


def test_full_blocks_only_report_baseline(reference_solution):
    used, unused_tolerance, unused_solution = reference_solution
    # Convergence is only checked between restart blocks, so the unconstrained
    # budget always finishes on a whole block.
    assert used % 4 == 0 and used >= 8


def test_budget_below_minimum_nonconverges_exactly(reference_solution):
    used, tolerance, unused_solution = reference_solution
    workspace = flat_workspace(restart=4, maxiter=used-4)
    iterations, error, unused_tolerance, unused_solution = run_gmres(workspace)
    assert iterations == used-4
    assert error > tolerance
    workspace.close()


def test_shortened_final_block_matches_full_block_solution(reference_solution):
    used, tolerance, solution = reference_solution
    workspace = flat_workspace(restart=4, maxiter=used-1)
    iterations, error, unused_tolerance, shortened = run_gmres(workspace)
    assert iterations == used-1
    if error <= tolerance:
        np.testing.assert_allclose(shortened, solution, rtol=0, atol=1e-9)
    workspace.close()


def test_single_shortened_block_solves_and_reports_actual_iterations(reference_solution):
    unused_used, unused_tolerance, solution = reference_solution
    workspace = flat_workspace(restart=64, maxiter=13)
    iterations, error, unused_tolerance, value = run_gmres(workspace)
    assert iterations == 13
    if error <= workspace.krylov._atol**.5 + 1e-30:
        np.testing.assert_allclose(value, solution, rtol=0, atol=1e-9)
    workspace.close()


def test_zero_iteration_exit_and_converged_guess():
    workspace = flat_workspace(restart=4, maxiter=400)
    # The GMRES constructor coerces a zero budget to the system size, so an
    # exact zero budget is applied at the loop level (the terrain solve
    # skips the call entirely for max_cycles <= 0; this exercises the loop).
    workspace.krylov._maxiter = 0
    iterations, error, tolerance, unused_solution = run_gmres(workspace)
    assert iterations == 0 and error > tolerance
    exact = np.zeros(workspace.shape)
    ramp = np.linspace(3., 1., workspace.shape[2])
    exact[...] = ramp[None, None, :]
    workspace.x.assign(exact.ravel())
    iterations, error, tolerance = workspace.krylov()
    assert iterations == 0 and error <= tolerance
    workspace.close()


def test_shortened_final_block_keeps_single_graph_capture(reference_solution):
    used, unused_tolerance, unused_solution = reference_solution
    if not device().startswith("cuda"):
        pytest.skip("CUDA graph accounting requires a CUDA device")
    workspace = flat_workspace(restart=4, maxiter=used-1)
    iterations, error, unused_tolerance, unused_solution = run_gmres(workspace)
    assert iterations == used-1
    # One capture for the full-block prefix; the shortened final block runs
    # eagerly and neither captures nor corrupts the captured graph.
    assert workspace.krylov.graph_captures == 1
    assert workspace.krylov.graph_replays == (iterations+3)//4 - 1
    run_gmres(workspace)
    assert workspace.krylov.graph_captures == 1
    workspace.close()
