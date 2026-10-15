"""Local active stencils required by physical terrain contact reconstruction."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp
from test_terrain_contact_exchange import refreshed_contact_flux

from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import (
    TerrainContact3D,
    TerrainReferenceOperator3D,
)
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


@pytest.fixture(params=["cpu", "cuda:0"])
def stencil_backend(request):
    wp.init()
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def masked_contact_case(
    *, direction, tangent, neighbor=None, kind="head", conductance=1.0
):
    columns = np.zeros((3, 3), dtype=bool)
    if tangent == "y":
        columns[1, :] = True
    else:
        columns[:, 1] = True
    j, i = {
        "west": (1, 0),
        "east": (1, 2),
        "south": (0, 1),
        "north": (2, 1),
        "top": (1, 1),
        "bottom": (1, 1),
    }[direction]
    if neighbor is not None:
        columns[
            j + neighbor if tangent == "y" else j, i + neighbor if tangent == "x" else i
        ] = True
    grid = TerrainGrid3D(
        surface_elevation=np.full((3, 3), 2.0),
        zeta_faces=np.array([0.0, 1.0, 2.0]),
        active_columns=columns,
        dx=1.0,
        dy=1.0,
    )
    layer = int(direction == "bottom")
    cell = (layer, j, i)
    y, x = np.mgrid[:3, :3]
    head = np.broadcast_to(10.0 + (y if tangent == "y" else x), grid.shape).copy()
    elevation = {"top": 2.0, "bottom": 0.0}.get(direction, float(grid.elevations[cell]))
    contact = TerrainContact3D(
        cell=cell,
        direction=direction,
        area=1.0,
        distance=0.5,
        elevation=elevation,
        head=float(head[cell] - 0.25),
        kind=kind,
        conductance=conductance,
        x_offset=-0.25 if tangent == "x" else 0.0,
        y_offset=-0.25 if tangent == "y" else 0.0,
    )
    return grid, TerrainMetrics3D(grid=grid), contact, head


def contact_operator(*, metrics, contacts, fraction=None):
    grid = metrics.grid
    material_head = (
        None if fraction is None else grid.bottoms + fraction * grid.thickness
    )
    return TerrainReferenceOperator3D(
        metrics=metrics,
        kx=1.0,
        ky=1.0,
        kz=1.0,
        head=material_head,
        min_sat=0.0,
        contacts=contacts,
    )


def assert_stencil_error(*, exception, contact, tangent):
    message = str(exception.value)
    assert str(contact.cell) in message
    assert contact.direction in message
    assert f"{tangent} tangential" in message
    assert "active-neighbour stencil" in message
    assert "local active mask" in message


@pytest.mark.parametrize(
    "direction,tangent",
    [
        ("west", "y"),
        ("east", "y"),
        ("south", "x"),
        ("north", "x"),
        ("top", "x"),
        ("top", "y"),
        ("bottom", "x"),
        ("bottom", "y"),
    ],
)
@pytest.mark.parametrize("kind", ["head", "discharge_only"])
def test_masked_offset_requires_local_stencil(direction, tangent, kind):
    _grid, metrics, contact, _head = masked_contact_case(
        direction=direction, tangent=tangent, kind=kind
    )
    coefficients = metrics.x_coefficients if tangent == "x" else metrics.y_coefficients
    assert coefficients[1][contact.cell[1:]] == coefficients[2][contact.cell[1:]] == 0.0
    row = (metrics.gx if tangent == "x" else metrics.gy).getrow(
        np.ravel_multi_index(contact.cell, metrics.grid.shape)
    )
    assert row.nnz > 0 and np.count_nonzero(row.data) == 0
    with pytest.raises(ValueError) as exception:
        contact_operator(metrics=metrics, contacts=(contact,))
    assert_stencil_error(exception=exception, contact=contact, tangent=tangent)


@pytest.mark.parametrize(
    "direction,tangent",
    [
        ("west", "y"),
        ("east", "y"),
        ("south", "x"),
        ("north", "x"),
        ("top", "x"),
        ("top", "y"),
        ("bottom", "x"),
        ("bottom", "y"),
    ],
)
@pytest.mark.parametrize("kind", ["head", "discharge_only"])
@pytest.mark.parametrize("conductance", [1.0, None])
@pytest.mark.parametrize("fraction", [0.0, 0.5, 1.0])
def test_missing_stencil_rejected_independent_of_state(
    direction, tangent, kind, conductance, fraction
):
    grid, metrics, contact, _head = masked_contact_case(
        direction=direction, tangent=tangent, kind=kind, conductance=conductance
    )
    # Horizontally constant current heads and a matching boundary head give
    # zero flux/closed discharge gates, even though the offset is unsupported.
    current = grid.bottoms + fraction * grid.thickness
    contact = replace(contact, head=float(current[contact.cell]))
    with pytest.raises(ValueError) as exception:
        contact_operator(metrics=metrics, contacts=(contact,), fraction=fraction)
    assert_stencil_error(exception=exception, contact=contact, tangent=tangent)


@pytest.mark.parametrize(
    "direction,tangent",
    [
        ("west", "y"),
        ("east", "y"),
        ("south", "x"),
        ("north", "x"),
        ("top", "x"),
        ("top", "y"),
        ("bottom", "x"),
        ("bottom", "y"),
    ],
)
@pytest.mark.parametrize("neighbor", [-1, 1])
@pytest.mark.parametrize("kind", ["head", "discharge_only"])
@pytest.mark.parametrize("conductance", [1.0, None])
def test_one_sided_contact_reconstructs_physical_affine_flux(
    stencil_backend, direction, tangent, neighbor, kind, conductance
):
    _grid, metrics, contact, head = masked_contact_case(
        direction=direction,
        tangent=tangent,
        neighbor=neighbor,
        kind=kind,
        conductance=conductance,
    )
    coefficients = metrics.x_coefficients if tangent == "x" else metrics.y_coefficients
    column = contact.cell[1:]
    assert bool(coefficients[1][column]) == (neighbor == -1)
    assert bool(coefficients[2][column]) == (neighbor == 1)
    operator = contact_operator(metrics=metrics, contacts=(contact,), fraction=0.5)
    # No gradient normal to ANY of these flat faces: physical q is exactly zero.
    assert operator.fluxes(head=head)[3][0] == pytest.approx(0.0, abs=1e-14)
    rows, flux, gates = refreshed_contact_flux(
        operator=operator, head=head, fraction=0.5, backend=stencil_backend
    )
    np.testing.assert_allclose(
        rows[0], operator.contact_records[0][1:5], rtol=2e-13, atol=1e-14
    )
    assert flux[operator.contact_records[0][0]] == pytest.approx(0.0, abs=1e-14)
    assert gates[0] == int(kind == "head")


@pytest.mark.parametrize(
    "direction,tangent",
    [
        ("west", "y"),
        ("east", "y"),
        ("south", "x"),
        ("north", "x"),
        ("top", "x"),
        ("top", "y"),
        ("bottom", "x"),
        ("bottom", "y"),
    ],
)
def test_centered_contact_does_not_require_unused_horizontal_stencil(
    direction, tangent
):
    _grid, metrics, contact, head = masked_contact_case(
        direction=direction, tangent=tangent
    )
    contact = replace(
        contact, x_offset=0.0, y_offset=0.0, head=float(head[contact.cell])
    )
    operator = contact_operator(metrics=metrics, contacts=(contact,))
    assert operator.fluxes(head=head)[3][0] == pytest.approx(0.0, abs=1e-14)


@pytest.mark.parametrize("direction", ["top", "bottom"])
@pytest.mark.parametrize("available", ["x", "y"])
@pytest.mark.parametrize("conductance", [1.0, None])
def test_sloping_contact_requires_only_used_horizontal_axis(
    stencil_backend, direction, available, conductance
):
    missing = "y" if available == "x" else "x"
    grid, _metrics, contact, head = masked_contact_case(
        direction=direction, tangent=missing
    )
    y, x = np.mgrid[:3, :3]
    coordinate = x if available == "x" else y
    grid = TerrainGrid3D(
        surface_elevation=2.0 + 0.2 * coordinate,
        zeta_faces=np.array([0.0, 1.0, 2.0]),
        active_columns=grid.active_columns,
        dx=1.0,
        dy=1.0,
    )
    metrics = TerrainMetrics3D(grid=grid)
    offset_x, offset_y = (-0.25, 0.0) if available == "x" else (0.0, -0.25)
    cell = contact.cell
    elevation = float(
        grid.elevations[cell]
        + 0.2 * (offset_x + offset_y)
        + (0.5 if direction == "top" else -0.5)
    )
    head = np.broadcast_to(10.0 + coordinate, grid.shape).copy()
    contact = replace(
        contact,
        elevation=elevation,
        head=float(head[cell] - 0.25),
        x_offset=offset_x,
        y_offset=offset_y,
        conductance=conductance,
    )
    operator = contact_operator(metrics=metrics, contacts=(contact,))
    projected_area = 1.0 / np.sqrt(1.0 + 0.2**2)
    natural_conductance = projected_area * (1.0 + 0.2**2) / 0.5
    expected = (0.2 if direction == "top" else -0.2) * projected_area
    if conductance is not None:
        expected *= conductance / natural_conductance
    assert operator.fluxes(head=head)[3][0] == pytest.approx(expected, abs=1e-14)
    _rows, flux, _gates = refreshed_contact_flux(
        operator=operator, head=head, fraction=1.0, backend=stencil_backend
    )
    assert flux[operator.contact_records[0][0]] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize(
    "direction,tangent",
    [
        ("west", "y"),
        ("east", "y"),
        ("south", "x"),
        ("north", "x"),
        ("top", "x"),
        ("top", "y"),
        ("bottom", "x"),
        ("bottom", "y"),
    ],
)
@pytest.mark.parametrize("kind", ["head", "discharge_only"])
def test_zero_exchange_needs_no_offset_stencil(
    stencil_backend, direction, tangent, kind
):
    _grid, metrics, contact, head = masked_contact_case(
        direction=direction, tangent=tangent, kind=kind, conductance=0.0
    )
    operator = contact_operator(metrics=metrics, contacts=(contact,))
    absent = contact_operator(metrics=metrics, contacts=())
    actual_matrix, actual_offset = operator.sparse_matrix(head_for_contacts=head)
    expected_matrix, expected_offset = absent.sparse_matrix(head_for_contacts=head)
    np.testing.assert_array_equal(actual_matrix.toarray(), expected_matrix.toarray())
    np.testing.assert_array_equal(actual_offset, expected_offset)
    np.testing.assert_array_equal(operator.contact_records[0][1:5], 0.0)
    assert operator.fluxes(head=head)[3][0] == 0.0
    rows, flux, _gates = refreshed_contact_flux(
        operator=operator, head=head, fraction=1.0, backend=stencil_backend
    )
    np.testing.assert_array_equal(rows, 0.0)
    np.testing.assert_array_equal(flux, 0.0)


@pytest.mark.parametrize(
    "direction,tangent",
    [
        ("west", "y"),
        ("south", "x"),
        ("top", "x"),
        ("top", "y"),
        ("bottom", "x"),
        ("bottom", "y"),
    ],
)
def test_zero_exchange_offset_allowed_with_single_column_dimension(direction, tangent):
    grid = TerrainGrid3D(
        surface_elevation=np.full((1, 1), 2.0),
        zeta_faces=np.array([0.0, 1.0, 2.0]),
        dx=1.0,
        dy=1.0,
    )
    cell = (int(direction == "bottom"), 0, 0)
    contact = TerrainContact3D(
        cell=cell,
        direction=direction,
        area=1.0,
        distance=0.5,
        elevation={"top": 2.0, "bottom": 0.0}.get(
            direction, float(grid.elevations[cell])
        ),
        head=9.0,
        conductance=0.0,
        x_offset=-0.25 if tangent == "x" else 0.0,
        y_offset=-0.25 if tangent == "y" else 0.0,
    )
    operator = contact_operator(
        metrics=TerrainMetrics3D(grid=grid), contacts=(contact,)
    )
    np.testing.assert_array_equal(
        operator.fluxes(head=np.full(grid.shape, 10.0))[3], 0.0
    )


@pytest.mark.parametrize(
    "invalid", ["area", "offset", "elevation", "cell", "finite", "material"]
)
def test_zero_exchange_keeps_other_geometry_and_material_checks(invalid):
    _grid, metrics, contact, _head = masked_contact_case(
        direction="west", tangent="y", conductance=0.0
    )
    changes = {
        "area": {"area": 2.0},
        "offset": {"y_offset": -0.75},
        "elevation": {"elevation": 10.0},
        "cell": {"cell": (0, 0, 0)},
        "finite": {"head": np.nan},
        "material": {},
    }
    contact = replace(contact, **changes[invalid])
    with pytest.raises(ValueError):
        TerrainReferenceOperator3D(
            metrics=metrics,
            kx=0.0 if invalid == "material" else 1.0,
            ky=1.0,
            kz=1.0,
            contacts=(contact,),
        )


def test_vertical_reconstruction_requirement_remains_separate():
    grid = TerrainGrid3D(
        surface_elevation=np.full((1, 2), 1.0),
        zeta_faces=np.array([0.0, 1.0]),
        dx=1.0,
        dy=1.0,
    )
    contact = TerrainContact3D(
        cell=(0, 0, 0),
        direction="west",
        area=1.0,
        distance=0.5,
        elevation=0.25,
        head=10.0,
        conductance=0.0,
    )
    with pytest.raises(ValueError, match="enough cells to reconstruct"):
        contact_operator(metrics=TerrainMetrics3D(grid=grid), contacts=(contact,))


def make_session_solver(*, grid, backend, head, source, contacts):
    fixed = np.zeros(grid.shape, dtype=np.int32)
    fixed[:, 1, 2] = 1
    solver = WarpDarcySolver3D(
        nx=3, ny=3, nz=2, dx=1.0, implementation="fast", device=backend
    )
    solver.build_from_terrain(
        z_interfaces=grid.z_interfaces,
        active_columns=grid.active_columns,
        kx_field=1.0,
        ky_field=1.0,
        kz_field=1.0,
        bc_mask=fixed,
        bc_values=head,
        rhs=source,
        initial_head=head - 0.2,
        contacts=contacts,
    )
    return solver


def stencil_solve_controls(*, device_picard):
    return {
        "unconfined": True,
        "device_picard": device_picard,
        "max_cycles": 40,
        "krylov_restart": 2,
        "max_levels": 2,
        "min_coarse_n": 2,
        "unconfined_max_picard_iter": 100,
        "abs_tol_min": 1e-12,
        "rel_tol": 1e-12,
        "final_head_residual_tol": 1e-9,
        "final_budget_relative_tol": 1e-9,
    }


def assert_session_physics(*, solver, head, info, exact):
    session = solver._session
    np.testing.assert_allclose(
        head[session.grid.active != 0],
        exact[session.grid.active != 0],
        rtol=0.0,
        atol=3e-9,
    )
    assert info["converged"], info
    reference = contact_operator(metrics=session.metrics, contacts=session.contacts)
    budget = compute_terrain_budget_3d(
        operator=reference,
        head=head,
        source=session.rhs,
        fixed=session.bc_mask,
        storage_change=np.zeros(session.shape),
        storage_diagonal=np.zeros(session.shape),
    )
    assert budget["head_residual_rms"] <= 1e-9
    assert (
        budget["cell_residual_rms"] <= 1e-12
        or budget["relative_cell_imbalance"] <= 1e-9
    )


@pytest.mark.parametrize("device_picard", [False, True])
def test_session_contact_update_validates_mask_and_zero_exchange(
    stencil_backend, device_picard
):
    grid, _metrics, contact, _head = masked_contact_case(direction="west", tangent="y")
    # Constant heads solve a well-constrained problem even without contacts.
    exact = np.full(grid.shape, 11.0)
    with make_session_solver(
        grid=grid,
        backend=stencil_backend,
        head=exact,
        source=np.zeros(grid.shape),
        contacts=(),
    ) as solver:
        options = stencil_solve_controls(device_picard=device_picard)
        solved, info = solver.solve(**options)
        assert_session_physics(solver=solver, head=solved, info=info, exact=exact)
        with pytest.raises(ValueError) as exception:
            solver._session.update_contacts(contacts=(contact,))
        assert_stencil_error(exception=exception, contact=contact, tangent="y")
        assert solver._session.contacts == ()
        workspace = solver._session.workspace
        solver._session.update_contacts(contacts=(replace(contact, conductance=0.0),))
        solved, info = solver.solve(**options)
        assert info["workspace_reused"] and solver._session.workspace is workspace
        assert_session_physics(solver=solver, head=solved, info=info, exact=exact)
        np.testing.assert_allclose(
            info["budget"]["coastal_subface_outflows"], [0.0], atol=0.0
        )


@pytest.mark.parametrize("device_picard", [False, True])
@pytest.mark.parametrize("neighbor", [-1, 1])
def test_geometry_update_revalidates_contact_stencil(
    stencil_backend, device_picard, neighbor
):
    grid, metrics, contact, exact = masked_contact_case(
        direction="west", tangent="y", neighbor=neighbor
    )
    reference = contact_operator(metrics=metrics, contacts=(contact,))
    source = reference.divergence(fluxes=reference.fluxes(head=exact))
    with make_session_solver(
        grid=grid,
        backend=stencil_backend,
        head=exact,
        source=source,
        contacts=(contact,),
    ) as solver:
        options = stencil_solve_controls(device_picard=device_picard)
        solved, info = solver.solve(**options)
        assert_session_physics(solver=solver, head=solved, info=info, exact=exact)
        workspace = solver._session.workspace
        old_graph = getattr(workspace.krylov, "_cycle_graph", None)
        old_replays = getattr(workspace.krylov, "graph_replays", 0)
        contact = replace(contact, conductance=0.5)
        solver._session.update_contacts(contacts=(contact,))
        solved, info = solver.solve(**options)
        assert info["workspace_reused"] and solver._session.workspace is workspace
        assert_session_physics(solver=solver, head=solved, info=info, exact=exact)
        if stencil_backend.startswith("cuda"):
            assert old_graph is not None and workspace.krylov._cycle_graph is old_graph
            assert workspace.krylov.graph_replays > old_replays
        invalid_grid, _metrics, _contact, _head = masked_contact_case(
            direction="west", tangent="y"
        )
        with pytest.raises(ValueError) as exception:
            solver._session.update_geometry(grid=invalid_grid)
            solver.solve(**options)
        assert_stencil_error(exception=exception, contact=contact, tangent="y")
        restored_grid, restored_metrics, _contact, exact = masked_contact_case(
            direction="west", tangent="y", neighbor=-neighbor
        )
        solver._session.update_geometry(grid=restored_grid)
        refreshed = contact_operator(metrics=restored_metrics, contacts=(contact,))
        solver.update_rhs(rhs=refreshed.divergence(fluxes=refreshed.fluxes(head=exact)))
        solver.update_initial_head(initial_head=exact - 0.2)
        solved, info = solver.solve(**options)
        assert not info["workspace_reused"]
        assert_session_physics(solver=solver, head=solved, info=info, exact=exact)
