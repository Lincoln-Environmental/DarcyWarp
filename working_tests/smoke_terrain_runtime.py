"""Clean-install terrain GMRES smoke with an independent physical solution."""

import argparse
import json
import platform

import numpy as np
import warp as wp


def run_terrain_smoke(*, device: str, device_picard: bool) -> dict:
    from DARCY_WARP_PACKAGE.config import WP_FLOAT
    from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d
    from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import (
        TerrainReferenceOperator3D,
    )
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    if WP_FLOAT != wp.float64:
        raise RuntimeError("Set DARCY_FLOAT=float64 before importing DarcyWarp.")
    shape = (2, 1, 4)
    expected = np.broadcast_to(np.linspace(2.0, 4.0, shape[2]), shape).copy()
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = fixed[:, :, -1] = 1
    source = np.zeros(shape)
    with WarpDarcySolver3D(
        nx=shape[2],
        ny=shape[1],
        nz=shape[0],
        dx=1.0,
        device=device,
        implementation="fast",
    ) as solver:
        solver.build_from_terrain(
            surface_elevation=np.zeros(shape[1:]),
            zeta_faces=np.array([0.0, 1.0, 2.0]),
            kx_field=1.0,
            ky_field=0.6,
            kz_field=0.2,
            bc_mask=fixed,
            bc_values=expected,
            rhs=source,
            initial_head=np.ones(shape),
        )
        from DARCY_WARP_PACKAGE.solvers.terrain_gmres_3d import TerrainGMRES3D

        options = {
            "device_picard": device_picard,
            "unconfined": True,
            "max_cycles": 40,
            "krylov_restart": 4,
            "max_levels": 3,
            "min_coarse_n": 2,
            "unconfined_max_picard_iter": 100,
            "final_head_residual_tol": 1e-10,
            "final_budget_relative_tol": 1e-9,
            "abs_tol_min": 1e-12,
            "rel_tol": 1e-12,
        }
        first, first_info = solver.solve(**options)
        assert first_info["converged"], first_info
        np.testing.assert_allclose(first, expected, rtol=0.0, atol=1e-9)
        workspace = solver._session.workspace
        assert isinstance(workspace.krylov, TerrainGMRES3D)
        graph = getattr(workspace.krylov, "_cycle_graph", None)
        replays_before = getattr(workspace.krylov, "graph_replays", 0)
        head, info = solver.solve(**options)
        assert info["converged"] and info["workspace_reused"], info
        assert solver._session.workspace is workspace
        np.testing.assert_allclose(head, expected, rtol=0.0, atol=1e-9)
        operator = TerrainReferenceOperator3D(
            metrics=solver._session.metrics, kx=1.0, ky=0.6, kz=0.2, head=head
        )
        budget = compute_terrain_budget_3d(
            operator=operator,
            head=head,
            source=source,
            fixed=fixed,
            storage_change=np.zeros(shape),
            storage_diagonal=np.zeros(shape),
        )
        assert budget["head_residual_rms"] <= 1e-10, budget
        assert (
            budget["cell_residual_rms"] <= 1e-12
            or budget["relative_cell_imbalance"] <= 1e-9
        ), budget
        captures = getattr(workspace.krylov, "graph_captures", 0)
        replays = getattr(workspace.krylov, "graph_replays", 0)
        if wp.get_device(device).is_cuda:
            assert graph is not None and workspace.krylov._cycle_graph is graph
            assert replays > replays_before
        return {
            "python": platform.python_version(),
            "warp": wp.__version__,
            "precision": "float64",
            "device": str(wp.get_device(device)),
            "device_picard": device_picard,
            "max_head_error": float(np.max(np.abs(head - expected))),
            "krylov_iterations": info["krylov_iterations"],
            "budget": budget,
            "workspace_reused": info["workspace_reused"],
            "graph_captures": captures,
            "graph_replays": replays,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    options = parser.parse_args()
    selected_device = options.device
    driver_choices = (False, True)
    results = [
        run_terrain_smoke(device=selected_device, device_picard=driver)
        for driver in driver_choices
    ]
    print(json.dumps(results, indent=2))
