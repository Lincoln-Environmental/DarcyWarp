"""Physical plane flux through sloping lateral contacts with tangent offsets."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp
from test_terrain_contact_exchange import (
    refreshed_contact_flux,
    sloping_contact_case,
)

from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import (
    TerrainContact3D,
    TerrainReferenceOperator3D,
)
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


@pytest.fixture(params=["cpu", "cuda:0"])
def contact_backend(request):
    wp.init()
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


@pytest.mark.parametrize("direction", ["east", "west", "north", "south"])
@pytest.mark.parametrize("kind", ["head", "discharge_only"])
@pytest.mark.parametrize("fraction", [0.0, 0.3, 1.0])
@pytest.mark.parametrize("conductance", [0.0, 0.2, 3.0, None])
def test_lateral_offset_affine_physical_exchange(
    contact_backend, direction, kind, fraction, conductance
):
    base, _head, _natural, _flux = sloping_contact_case(
        direction="top", kind="head", conductance=None, fraction=fraction
    )
    grid = base.grid
    cell = {
        "east": (0, 1, 3),
        "west": (0, 1, 0),
        "north": (0, 2, 2),
        "south": (0, 0, 2),
    }[direction]
    x_normal = direction in ("east", "west")
    sign = 1 if direction in ("east", "north") else -1
    u, v = (0.0, 0.2) if x_normal else (0.3, 0.0)
    distance, area = 0.5, 1.3
    elevation = grid.elevations[cell] + 0.1
    y, x = np.mgrid[:3, :4]
    a, b, c = 0.3, -0.2, 0.7
    head = 4.0 + a * x[None] * grid.dx + b * y[None] * grid.dy + c * grid.elevations
    px = cell[2] * grid.dx + u + (sign * distance if x_normal else 0.0)
    py = cell[1] * grid.dy + v + (0.0 if x_normal else sign * distance)
    contact = TerrainContact3D(
        cell=cell,
        direction=direction,
        kind=kind,
        area=area,
        distance=distance,
        elevation=float(elevation),
        head=float(4.0 + a * px + b * py + c * elevation),
        conductance=conductance,
        x_offset=u,
        y_offset=v,
    )
    operator = TerrainReferenceOperator3D(
        metrics=base.metrics,
        kx=2.0,
        ky=0.6,
        kz=0.13,
        head=grid.bottoms + fraction * grid.thickness,
        min_sat=0.0,
        contacts=(contact,),
    )
    normal_k, gradient = (2.0, a) if x_normal else (0.6, b)
    normal_c = (
        normal_k * fraction * area / distance if conductance is None else conductance
    )
    expected = -sign * normal_c * distance * gradient
    if kind == "discharge_only":
        expected = max(expected, 0.0)
    actual, flux, _gates = refreshed_contact_flux(
        operator=operator, head=head, fraction=fraction, backend=contact_backend
    )
    np.testing.assert_allclose(
        actual[0], operator.contact_records[0][1:5], rtol=3e-13, atol=2e-14
    )
    assert operator.fluxes(head=head)[3][0] == pytest.approx(
        expected, rel=3e-13, abs=2e-14
    )
    assert flux[operator.contact_records[0][0]] == pytest.approx(
        expected, rel=4e-13, abs=2e-14
    )
    matrix, offset = operator.sparse_matrix(head_for_contacts=head)
    np.testing.assert_allclose(
        matrix @ head.ravel() + offset.ravel(),
        operator.divergence(fluxes=operator.fluxes(head=head)).ravel(),
        rtol=1e-12,
        atol=3e-14,
    )


@pytest.mark.parametrize("kind", ["head", "discharge_only"])
def test_lateral_offset_session_update_reuses_graph(contact_backend, kind):
    base, exact, _natural, _flux = sloping_contact_case(
        direction="top", kind="head", conductance=None, fraction=1.0
    )
    grid = base.grid
    exact = exact + 20.0
    cell = (0, 1, 0)
    elevation = float(grid.elevations[cell] + 0.1)
    contact = TerrainContact3D(
        cell=cell,
        direction="west",
        kind=kind,
        area=1.3,
        distance=0.5,
        elevation=elevation,
        head=24.0 + 0.3 * (-0.5) - 0.2 * grid.dy + 0.7 * elevation,
        conductance=0.2,
    )
    updated = replace(contact, y_offset=0.2, head=contact.head - 0.2 * 0.2)
    original = TerrainReferenceOperator3D(
        metrics=base.metrics, kx=2.0, ky=0.6, kz=0.13, contacts=(contact,)
    )
    refreshed = TerrainReferenceOperator3D(
        metrics=base.metrics, kx=2.0, ky=0.6, kz=0.13, contacts=(updated,)
    )
    # A physical affine gradient has the same plane flux at either centroid.
    assert original.fluxes(head=exact)[3][0] == pytest.approx(0.03, abs=3e-14)
    assert refreshed.fluxes(head=exact)[3][0] == pytest.approx(0.03, abs=3e-14)
    source = original.divergence(fluxes=original.fluxes(head=exact))
    fixed = np.zeros(grid.shape, dtype=np.int32)
    fixed[:, :, -1] = 1
    options = {
        "device_picard": True,
        "unconfined": True,
        "max_cycles": 120,
        "krylov_restart": 4,
        "unconfined_max_picard_iter": 100,
        "final_head_residual_tol": 1e-9,
        "final_budget_relative_tol": 1e-9,
        "abs_tol_min": 1e-12,
        "rel_tol": 1e-12,
    }
    results = []
    for initial_contacts in ((contact,), (updated,)):
        with WarpDarcySolver3D(
            nx=grid.shape[2],
            ny=grid.shape[1],
            nz=grid.shape[0],
            dx=grid.dx,
            dy=grid.dy,
            implementation="fast",
            device=contact_backend,
        ) as solver:
            solver.build_from_terrain(
                z_interfaces=grid.z_interfaces,
                kx_field=2.0,
                ky_field=0.6,
                kz_field=0.13,
                bc_mask=fixed,
                bc_values=exact,
                rhs=source,
                initial_head=exact - 0.2,
                contacts=initial_contacts,
            )
            head, info = solver.solve(**options)
            assert info["converged"], info
            if initial_contacts == (contact,):
                workspace = solver._session.workspace
                old_graph = getattr(workspace.krylov, "_cycle_graph", None)
                old_replays = getattr(workspace.krylov, "graph_replays", 0)
                solver._session.update_contacts(contacts=(updated,))
                head, info = solver.solve(**options)
                assert info["converged"] and info["workspace_reused"], info
                assert solver._session.workspace is workspace
                if contact_backend.startswith("cuda"):
                    assert (
                        workspace.krylov._cycle_graph is old_graph
                        and old_graph is not None
                    )
                    assert workspace.krylov.graph_replays > old_replays
            np.testing.assert_allclose(head, exact, rtol=0.0, atol=3e-9)
            budget = info["budget"]
            assert budget["head_residual_rms"] <= 1e-9
            assert (
                budget["cell_residual_rms"] <= 1e-12
                or budget["relative_cell_imbalance"] <= 1e-9
            )
            results.append(head)
    np.testing.assert_allclose(results[0], results[1], rtol=0.0, atol=3e-9)
