"""Physical exchange tests independent of sparse/device coefficient parity."""

from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import (
    TerrainContact3D,
    TerrainReferenceOperator3D,
)
from DARCY_WARP_PACKAGE.solvers.terrain_device_kernels import (
    terrain_contact_flux_kernel,
    terrain_contact_refresh_kernel,
    terrain_subface_gate_kernel,
)
from DARCY_WARP_PACKAGE.solvers.terrain_fast_3d import (
    pack_device_contact_refresh,
    pack_device_subfaces,
    pack_terrain_boundaries,
)
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


@pytest.fixture(params=["cpu", "cuda:0"])
def contact_backend(request):
    wp.init()
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def refreshed_contact_flux(*, operator, head, fraction, backend):
    """Execute the production constitutive, gating and physical-flux kernels."""

    def upload(values, dtype=wp.float64):
        return wp.array(np.ascontiguousarray(values), dtype=dtype, device=backend)

    rows, count = pack_device_subfaces(operator=operator, datum=0.0)
    cells = upload(values=rows[:, 0].astype(np.int32), dtype=wp.int32)
    coefficients = [upload(values=rows[:, col]) for col in range(1, 5)]
    endpoints = [
        upload(values=array)
        for array in pack_device_contact_refresh(operator=operator, datum=0.0)
    ]
    wp.launch(
        terrain_contact_refresh_kernel,
        dim=count,
        inputs=[
            cells,
            upload(values=np.broadcast_to(fraction, operator.shape).ravel()),
            *endpoints,
            *coefficients,
        ],
        device=backend,
    )
    gradients = [
        upload(values=gradient @ head.ravel())
        for gradient in (operator.metrics.gx, operator.metrics.gy, operator.metrics.gz)
    ]
    head_device = upload(values=head.ravel())
    hb = upload(values=rows[:, 5])
    gates = wp.zeros(count, dtype=wp.int32, device=backend)
    wp.launch(
        terrain_subface_gate_kernel,
        dim=count,
        inputs=[
            cells,
            *coefficients,
            hb,
            upload(values=rows[:, 6].astype(np.int32), dtype=wp.int32),
            *gradients,
            head_device,
            gates,
        ],
        device=backend,
    )
    net = wp.zeros(head.size, dtype=wp.float64, device=backend)
    gross = wp.zeros_like(net)
    wp.launch(
        terrain_contact_flux_kernel,
        dim=count,
        inputs=[cells, *coefficients, hb, gates, head_device, *gradients, net, gross],
        device=backend,
    )
    return (
        np.column_stack([array.numpy() for array in coefficients]),
        net.numpy(),
        gates.numpy(),
    )


@pytest.mark.parametrize("direction", ["top", "bottom"])
@pytest.mark.parametrize("kind", ["head", "discharge_only"])
@pytest.mark.parametrize("difference", [1.0, -1.0])
@pytest.mark.parametrize("conductance", [0.0, 0.1, 1.0, 2.0, None])
def test_flat_contact_physical_exchange(
    contact_backend, direction, kind, difference, conductance
):
    grid = TerrainGrid3D(
        surface_elevation=np.full((1, 1), 2.0),
        zeta_faces=np.array([0.0, 1.0, 2.0]),
        dx=1.0,
        dy=1.0,
    )
    layer = 0 if direction == "top" else 1
    cell = (layer, 0, 0)
    head = np.zeros(grid.shape)
    head[cell] = difference
    contact = TerrainContact3D(
        cell=cell,
        direction=direction,
        kind=kind,
        area=1.0,
        distance=0.5,
        elevation=2.0 if layer == 0 else 0.0,
        head=0.0,
        conductance=conductance,
    )
    operator = TerrainReferenceOperator3D(
        metrics=TerrainMetrics3D(grid=grid), kx=1.0, ky=1.0, kz=1.0, contacts=(contact,)
    )
    c = 2.0 if conductance is None else conductance
    expected = c * difference
    if kind == "discharge_only":
        expected = max(expected, 0.0)
    # Independent physical oracle; parity alone would preserve the original bug.
    actual, flux, gates = refreshed_contact_flux(
        operator=operator, head=head, fraction=1.0, backend=contact_backend
    )
    np.testing.assert_allclose(
        [
            operator.fluxes(head=head)[3][0],
            flux[np.ravel_multi_index(cell, grid.shape)],
        ],
        expected,
        rtol=0.0,
        atol=1e-14,
    )
    np.testing.assert_allclose(actual[0], [c, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-14)
    assert flux[np.ravel_multi_index(cell, grid.shape)] == pytest.approx(
        expected, abs=1e-14
    )
    assert gates[0] == int(kind == "head" or c * difference > 0.0)


def sloping_contact_case(*, direction, kind, conductance, fraction):
    y, x = np.mgrid[:3, :4]
    top = 10.0 + 0.3 * x + 0.15 * y
    middle = top - (2.0 + 0.1 * x + 0.07 * y)
    bottom = middle - (3.0 + 0.05 * x - 0.03 * y)
    grid = TerrainGrid3D(z_interfaces=np.stack((top, middle, bottom)), dx=2.0, dy=3.0)
    metrics = TerrainMetrics3D(grid=grid)
    layer = 0 if direction == "top" else 1
    cell = (layer, 1, 2)
    sx, sy = metrics.zx[cell], metrics.zy[cell]
    u, v, distance = 0.4, -0.3, 0.6
    sign = -1 if direction == "top" else 1
    elevation = grid.elevations[cell] + sx * u + sy * v - sign * distance
    a, b, c = 0.3, -0.2, 0.7
    head = 4.0 + a * x[None] * grid.dx + b * y[None] * grid.dy + c * grid.elevations
    hb = 4.0 + a * (x[1, 2] * grid.dx + u) + b * (y[1, 2] * grid.dy + v) + c * elevation
    contact = TerrainContact3D(
        cell=cell,
        direction=direction,
        kind=kind,
        area=1.3,
        distance=distance,
        elevation=float(elevation),
        head=float(hb),
        conductance=conductance,
        x_offset=u,
        y_offset=v,
    )
    operator = TerrainReferenceOperator3D(
        metrics=metrics,
        kx=2.0,
        ky=0.6,
        kz=0.13,
        head=grid.bottoms + fraction * grid.thickness,
        min_sat=0.0,
        contacts=(contact,),
    )
    interface = layer + int(direction == "bottom")
    ix = metrics.zx_interfaces[interface, 1, 2]
    iy = metrics.zy_interfaces[interface, 1, 2]
    projected = contact.area / np.sqrt(1.0 + ix * ix + iy * iy)
    normal = 0.13 + fraction * (2.0 * ix * ix + 0.6 * iy * iy)
    natural = projected * normal / distance
    flux = sign * projected * (0.13 * c - fraction * (2.0 * ix * a + 0.6 * iy * b))
    if conductance is not None:
        flux *= conductance / natural
    if kind == "discharge_only":
        flux = max(flux, 0.0)
    return operator, head, natural, flux


@pytest.mark.parametrize("direction", ["top", "bottom"])
@pytest.mark.parametrize("kind", ["head", "discharge_only"])
@pytest.mark.parametrize("fraction", [0.0, 0.2, 0.7, 1.0])
@pytest.mark.parametrize("conductance", [0.0, 0.1, 3.0, None])
def test_sloping_override_refresh_and_affine_flux(
    contact_backend, direction, kind, fraction, conductance
):
    operator, head, natural, expected = sloping_contact_case(
        direction=direction, kind=kind, conductance=conductance, fraction=fraction
    )
    actual, flux, _gates = refreshed_contact_flux(
        operator=operator, head=head, fraction=fraction, backend=contact_backend
    )
    assert operator.fluxes(head=head)[3][0] == pytest.approx(
        expected, rel=3e-13, abs=2e-14
    )
    np.testing.assert_allclose(
        actual[0], operator.contact_records[0][1:5], rtol=3e-13, atol=2e-14
    )
    assert flux[operator.contact_records[0][0]] == pytest.approx(
        expected, rel=4e-13, abs=3e-14
    )
    # Setting C to this state's natural conductance recovers the omitted law.
    original = operator.contacts[0]
    matching = replace(original, conductance=natural)
    natural_operator = TerrainReferenceOperator3D(
        metrics=operator.metrics,
        kx=operator.kx,
        ky=operator.ky,
        kz=operator.kz,
        head=operator.grid.bottoms + fraction * operator.grid.thickness,
        min_sat=0.0,
        contacts=(replace(original, conductance=None),),
    )
    matching_operator = TerrainReferenceOperator3D(
        metrics=operator.metrics,
        kx=operator.kx,
        ky=operator.ky,
        kz=operator.kz,
        head=operator.grid.bottoms + fraction * operator.grid.thickness,
        min_sat=0.0,
        contacts=(matching,),
    )
    np.testing.assert_allclose(
        matching_operator.contact_records[0][1:5],
        natural_operator.contact_records[0][1:5],
        rtol=3e-13,
        atol=2e-14,
    )
    matching_rows, matching_flux, _matching_gates = refreshed_contact_flux(
        operator=matching_operator,
        head=head,
        fraction=fraction,
        backend=contact_backend,
    )
    np.testing.assert_allclose(
        matching_rows[0],
        natural_operator.contact_records[0][1:5],
        rtol=3e-13,
        atol=2e-14,
    )
    assert matching_flux[operator.contact_records[0][0]] == pytest.approx(
        natural_operator.fluxes(head=head)[3][0], rel=4e-13, abs=3e-14
    )


@pytest.mark.parametrize("direction", ["top", "bottom"])
@pytest.mark.parametrize("kind", ["head", "discharge_only"])
@pytest.mark.parametrize("device_picard", [False, True])
def test_zero_contact_equals_absent_operator_solve_budget(
    contact_backend, direction, kind, device_picard
):
    operator, head, _natural, _flux = sloping_contact_case(
        direction=direction, kind=kind, conductance=0.0, fraction=1.0
    )
    absent = TerrainReferenceOperator3D(
        metrics=operator.metrics, kx=operator.kx, ky=operator.ky, kz=operator.kz
    )
    matrix, offset = operator.sparse_matrix(head_for_contacts=head)
    expected_matrix, expected_offset = absent.sparse_matrix(head_for_contacts=head)
    np.testing.assert_array_equal(matrix.toarray(), expected_matrix.toarray())
    np.testing.assert_array_equal(offset, expected_offset)
    boundary, rhs = pack_terrain_boundaries(operator=operator, head=head, datum=2.0)
    np.testing.assert_array_equal(boundary[:, :4], 0.0)
    np.testing.assert_array_equal(rhs, 0.0)
    assert operator.fluxes(head=head)[3][0] == 0.0
    grid = operator.grid
    fixed = np.zeros(grid.shape, dtype=np.int32)
    fixed[:, :, 0] = fixed[:, :, -1] = 1
    values = np.full(grid.shape, 20.0)
    values[:, :, -1] = 21.0
    initial = np.full(grid.shape, 20.0)
    source = np.zeros(grid.shape)
    results = []
    for contacts in ((), operator.contacts):
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
                bc_values=values,
                rhs=source,
                initial_head=initial,
                contacts=contacts,
            )
            result, info = solver.solve(
                device_picard=device_picard,
                unconfined=True,
                max_cycles=120,
                krylov_restart=4,
                unconfined_max_picard_iter=100,
                final_head_residual_tol=1e-9,
                final_budget_relative_tol=1e-9,
                abs_tol_min=1e-12,
                rel_tol=1e-12,
            )
            assert info["converged"], info
            reference = TerrainReferenceOperator3D(
                metrics=operator.metrics,
                kx=2.0,
                ky=0.6,
                kz=0.13,
                head=result,
                contacts=contacts,
            )
            budget = compute_terrain_budget_3d(
                operator=reference,
                head=result,
                source=source,
                fixed=fixed,
                storage_change=np.zeros(grid.shape),
                storage_diagonal=np.zeros(grid.shape),
            )
            assert budget["head_residual_rms"] <= 1e-9
            assert (
                budget["cell_residual_rms"] <= 1e-12
                or budget["relative_cell_imbalance"] <= 1e-9
            )
            if contact_backend.startswith("cuda"):
                assert solver._session.workspace.krylov.graph_captures > 0
                assert solver._session.workspace.krylov.graph_replays > 0
            results.append((result, budget))
    np.testing.assert_allclose(results[0][0], results[1][0], rtol=0.0, atol=2e-9)
    for field in (
        "boundary_net",
        "boundary_inflow",
        "boundary_outflow",
        "imbalance",
        "cell_residual_rms",
        "head_residual_rms",
    ):
        assert results[0][1][field] == pytest.approx(results[1][1][field], abs=2e-12)
