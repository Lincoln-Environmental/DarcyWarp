"""Independent regressions for the remaining fast-3D scientific defects."""
import numpy as np
import pytest
import warp as wp
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from DARCY_WARP_PACKAGE.solvers_3d import (
    build_7point_face_conductance_from_k, _validate_flat_layer_geometry,
)
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def device():
    wp.init()
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def independent_system(*, faces, active, bc, values, source, storage, previous):
    """Assemble balance by traversing each undirected face exactly once."""
    shape = active.shape
    indices = np.arange(active.size).reshape(shape)
    matrix = lil_matrix((active.size, active.size), dtype=np.float64)
    for axis, index in ((2, 0), (1, 2), (0, 4)):
        for here in np.ndindex(shape):
            there = list(here)
            there[axis] += 1
            if there[axis] >= shape[axis]:
                continue
            there = tuple(there)
            if not (active[here] and active[there]):
                continue
            c = faces[index][here]
            i, j = indices[here], indices[there]
            matrix[i, i] += c
            matrix[j, j] += c
            matrix[i, j] -= c
            matrix[j, i] -= c
    matrix.setdiag(matrix.diagonal() + storage.ravel())
    physical = matrix.tocsr()
    rhs = (source + storage * previous).ravel()
    solve_matrix = matrix.copy()
    solve_rhs = rhs.copy()
    for here in np.ndindex(shape):
        if bc[here] or not active[here]:
            index = indices[here]
            solve_matrix[index, :] = 0.0
            solve_matrix[index, index] = 1.0
            solve_rhs[index] = values[here] if bc[here] else 0.0
    exact = spsolve(solve_matrix.tocsr(), solve_rhs).reshape(shape)
    return exact, physical, rhs


@pytest.mark.parametrize("axis", [0, 1, 2])
@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("scale", [1.0, 1e-14, 1e-22, 1e-80])
@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_scaled_flow_all_six_directions(axis, direction, scale, implementation):
    shape = [1, 1, 1]
    shape[axis] = 7
    shape = tuple(shape)
    k = np.full(shape, scale)
    act = np.ones(shape, dtype=np.int32)
    bc = np.zeros(shape, dtype=np.int32)
    values = np.zeros(shape)
    first, last = [slice(None)] * 3, [slice(None)] * 3
    first[axis], last[axis] = 0, -1
    first, last = tuple(first), tuple(last)
    bc[first] = bc[last] = 1
    values[first], values[last] = (8.0, 2.0) if direction == 1 else (2.0, 8.0)
    source = np.full(shape, 0.2 * scale)
    faces = build_7point_face_conductance_from_k(kx_field=k, ky_field=k, kz_field=k,
                                               active=act, dx=1.0, dy=1.0, dz=1.0)
    zero = np.zeros(shape)
    exact, matrix, rhs = independent_system(faces=faces, active=act, bc=bc, values=values,
                                            source=source, storage=zero, previous=zero)
    with WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=1.0,
                          device=device(), implementation=implementation) as solver:
        solver.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k, active=act,
                                   bc_mask=bc, bc_values=values, rhs=source)
        head, info = solver.solve(max_levels=1, max_cycles=500, check_every_no=1,
                                  rel_tol=1e-11, abs_tol_min=0.0, dh_rms_tol=1e-10)
    assert info["converged"]
    np.testing.assert_allclose(head, exact, rtol=0.0, atol=2e-9)
    residual = (rhs - matrix @ head.ravel()).reshape(shape)
    assert np.max(np.abs(residual[bc == 0])) / scale < 3e-9
    assert info["budget"]["relative_imbalance"] < 1e-9
    flow = np.diff(np.moveaxis(head, axis, -1).ravel())
    assert np.all(direction * flow < 0.0)


def test_low_k_default_absolute_tolerance_cannot_accept_initial_head():
    shape = (1, 1, 3)
    k = np.full(shape, 1e-20)
    bc = np.array([[[1, 0, 1]]], dtype=np.int32)
    values = np.array([[[8.0, 0.0, 2.0]]])
    with WarpDarcySolver3D(nx=3, ny=1, nz=1, dx=1.0, device=device(), implementation="fast") as s:
        s.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k,
                              active=np.ones(shape, dtype=np.int32), bc_mask=bc,
                              bc_values=values, rhs=np.zeros(shape))
        head, info = s.solve()
    assert info["converged"] and info["final_physical_acceptance"]
    assert head[0, 0, 1] == pytest.approx(5.0, abs=1e-5)


def test_host_and_gpu_diagonal_use_only_actual_positive_connections():
    from DARCY_WARP_PACKAGE.solvers_3d import build_diag_preconditioner_7point
    from DARCY_WARP_PACKAGE.kernels_3d import build_diag_preconditioner_7point_kernel
    from DARCY_WARP_PACKAGE.config import WP_FLOAT

    shape = (1, 1, 4)
    faces = [np.full(shape, 99.0) for _ in range(6)]  # unused exterior slots
    faces[0][0, 0, :3] = [1e-20, 99.0, 99.0]
    faces[1][0, 0, 1:] = [1e-20, 99.0, 99.0]
    active = np.array([[[1, 1, 0, 1]]], dtype=np.int32)
    bc = np.zeros(shape, dtype=np.int32)
    storage = np.zeros(shape)
    host = build_diag_preconditioner_7point(tx_p=faces[0], tx_m=faces[1], ty_p=faces[2], ty_m=faces[3],
        tz_p=faces[4], tz_m=faces[5], active=active, bc_mask=bc, storage_diag=storage)
    dev = device()
    buffers = [wp.array(f, dtype=WP_FLOAT, device=dev) for f in faces]
    act = wp.array(active, dtype=wp.int32, device=dev)
    bcm = wp.array(bc, dtype=wp.int32, device=dev)
    sto = wp.array(storage, dtype=WP_FLOAT, device=dev)
    out = wp.zeros(shape, dtype=WP_FLOAT, device=dev)
    wp.launch(kernel=build_diag_preconditioner_7point_kernel, dim=shape,
              inputs=[*buffers, act, bcm, sto, out, 4, 1, 1], device=dev)
    np.testing.assert_allclose(host, [[[1e20, 1e20, 1.0, 0.0]]], rtol=1e-15, atol=0.0)
    np.testing.assert_allclose(out.numpy(), host, rtol=1e-15, atol=0.0)


@pytest.mark.parametrize("left,right", [(1e-200, 1e-200), (1e-200, 1e200), (1e200, 1e200), (0.0, 1e200)])
def test_stable_harmonic_extreme_contrasts(left, right):
    k = np.array([[[left, right]]])
    faces = build_7point_face_conductance_from_k(kx_field=k, ky_field=k, kz_field=k,
                                               active=np.ones(k.shape), dx=1.0)
    exact = 0.0 if min(left, right) == 0.0 else min(left, right) * (2.0 / (1.0 + min(left, right) / max(left, right)))
    assert faces[0][0, 0, 0] == pytest.approx(exact, rel=2e-15, abs=0.0)
    assert faces[1][0, 0, 1] == faces[0][0, 0, 0]


@pytest.mark.parametrize("implementation", ["classic", "fast"])
@pytest.mark.parametrize("source", [[1.0, 0.0], [1.0, -1.0], [0.0, 0.0]])
def test_connected_singular_group_requires_compatible_source_and_datum(implementation, source):
    shape = (1, 1, 2)
    k = np.ones(shape)
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=1.0, device=device(), implementation=implementation) as s:
        s.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k,
                              active=np.ones(shape, dtype=np.int32), bc_mask=np.zeros(shape, dtype=np.int32),
                              bc_values=np.zeros(shape), rhs=np.array(source).reshape(shape))
        diagnostic = "net source" if sum(source) else "undetermined head datum"
        with pytest.raises(ValueError, match=diagnostic):
            s.solve()
        head, info = s.solve(transient=True, storage_coeff=0.2, dt=1.0,
                             head_prev=np.full(shape, 5.0), rel_tol=1e-11, dh_rms_tol=1e-10,
                             check_every_no=1)
        assert info["converged"]
        expected = np.linalg.solve(np.array([[1.2, -1.0], [-1.0, 1.2]]), np.array(source) + 1.0)
        np.testing.assert_allclose(head.ravel(), expected, atol=1e-8, rtol=0.0)


@pytest.mark.parametrize("input_name", ["storage_coeff", "storage_diag"])
def test_mutable_storage_refresh_preserves_buffers_and_graph(input_name):
    shape = (2, 4, 6)
    k = np.ones(shape)
    act = np.ones(shape, dtype=np.int32)
    bc = np.zeros(shape, dtype=np.int32)
    bc[:, :, 0] = 1
    values = np.full(shape, 3.0)
    previous = np.full(shape, 7.0)
    source = np.full(shape, -0.1)
    storage_input = np.full(shape, 0.2)
    faces = build_7point_face_conductance_from_k(kx_field=k, ky_field=k, kz_field=k, active=act, dx=1.0)
    with WarpDarcySolver3D(nx=6, ny=4, nz=2, dx=1.0, device=device(), implementation="fast") as s:
        s.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k, active=act,
                              bc_mask=bc, bc_values=values, rhs=source, initial_head=previous)
        controls = dict(transient=True, storage_coeff=storage_input if input_name == "storage_coeff" else 0.0,
                        dt=2.0, head_prev=previous, rel_tol=1e-11, dh_rms_tol=1e-10,
                        check_every_no=1, max_cycles=500, min_coarse_n=1)
        if input_name == "storage_diag":
            controls[input_name] = storage_input
        _, first = s.solve(**controls)
        record = s._session._fast_workspace_holder["workspace"]
        graph = record["cycle_graph"]
        ids = [id(level["diag"]) for level in record["levels"]]
        for factor in (2.0, 1.0):
            storage_input *= factor
            head, info = s.solve(**controls)
            assert info["workspace_reused"]
            assert info["operator_refreshed_in_place"] == (factor != 1.0)
            assert s._session._fast_workspace_holder["workspace"]["cycle_graph"] is graph
            assert [id(level["diag"]) for level in record["levels"]] == ids
            storage = storage_input / 2.0 if input_name == "storage_coeff" else storage_input
            prev = previous if input_name == "storage_coeff" else np.zeros(shape)
            exact, matrix, rhs = independent_system(faces=faces, active=act, bc=bc, values=values,
                                                    source=source, storage=storage, previous=prev)
            np.testing.assert_allclose(head, exact, atol=1e-8, rtol=0.0)
            assert np.max(np.abs((rhs - matrix @ head.ravel()).reshape(shape)[bc == 0])) < 1e-8
            assert info["final_physical_acceptance"]
        controls["dt"] = 0.5
        _, changed_dt = s.solve(**controls)
        assert changed_dt["operator_refreshed_in_place"] == (input_name == "storage_coeff")
        s.update_boundary_values(bc_values=np.full(shape, 4.0))
        head, changed_bc = s.solve(**controls)
        assert changed_bc["workspace_reused"] and changed_bc["operator_refreshed_in_place"]
        np.testing.assert_array_equal(head[bc != 0], 4.0)
        assert changed_bc["component_topology_reused"]


@pytest.mark.parametrize("ztop_supplied", [False, True])
@pytest.mark.parametrize("bottoms", [[10.0, -1.0], [10.0, 1.0]])
def test_geometry_gap_and_overlap_never_bypass_validation(ztop_supplied, bottoms):
    zb = np.array(bottoms).reshape(2, 1, 1)
    with pytest.raises(ValueError, match="contiguous"):
        _validate_flat_layer_geometry(zbot=zb, ztop=zb + 10 if ztop_supplied else None,
                                      dz=10, shape=zb.shape)


@pytest.mark.parametrize("axis,index", [(2, 0), (1, 2), (0, 4)])
def test_public_faces_reject_asymmetry_and_inactive_connections(axis, index):
    shape = (2, 2, 2)
    act = np.ones(shape, dtype=np.int32)
    k = np.ones(shape)
    faces = list(build_7point_face_conductance_from_k(kx_field=k, ky_field=k, kz_field=k, active=act, dx=1.0))
    with WarpDarcySolver3D(nx=2, ny=2, nz=2, dx=1.0, device=device(), implementation="fast") as s:
        fields = dict(active=act, bc_mask=np.zeros(shape, dtype=np.int32), bc_values=np.zeros(shape), rhs=np.zeros(shape))
        faces[index][0, 0, 0] *= 1.01
        with pytest.raises(ValueError, match="at .*difference"):
            s.build_from_face_conductance(tx_p=faces[0], tx_m=faces[1], ty_p=faces[2], ty_m=faces[3],
                                          tz_p=faces[4], tz_m=faces[5], **fields)
        faces[index][0, 0, 0] /= 1.01
        act[0, 0, 0] = 0
        with pytest.raises(ValueError, match="inactive"):
            s.build_from_face_conductance(tx_p=faces[0], tx_m=faces[1], ty_p=faces[2], ty_m=faces[3],
                                          tz_p=faces[4], tz_m=faces[5], **fields)


def test_face_values_refresh_but_topology_rebuilds_and_float32_roundoff_is_canonical():
    shape = (1, 2, 6)
    k = np.ones(shape)
    active = np.ones(shape, dtype=np.int32)
    bc = np.zeros(shape, dtype=np.int32)
    bc[:, :, 0] = bc[:, :, -1] = 1
    values = np.zeros(shape)
    values[:, :, 0] = 5.0
    source = np.full(shape, 0.01)
    controls = dict(rel_tol=1e-10, dh_rms_tol=1e-9, check_every_no=1, min_coarse_n=1)
    faces = list(build_7point_face_conductance_from_k(kx_field=k, ky_field=k, kz_field=k,
        active=active, dx=1.0))
    faces = [f.astype(np.float32) for f in faces]
    faces[1][:, :, 1:] = np.nextafter(faces[1][:, :, 1:], np.float32(2.0))
    with WarpDarcySolver3D(nx=6, ny=2, nz=1, dx=1.0, device=device(), implementation="fast") as s:
        s.build_from_face_conductance(tx_p=faces[0], tx_m=faces[1], ty_p=faces[2], ty_m=faces[3],
            tz_p=faces[4], tz_m=faces[5], active=active, bc_mask=bc, bc_values=values, rhs=source)
        np.testing.assert_array_equal(s._tx_p[:, :, :-1], s._tx_m[:, :, 1:])
        _, first = s.solve(**controls)
        record = s._session._fast_workspace_holder["workspace"]
        graph = record["cycle_graph"]
        updated = tuple(f.copy() * 2.0 for f in s._session.faces)
        s.update_face_conductance(tx_p=updated[0], tx_m=updated[1], ty_p=updated[2], ty_m=updated[3],
                                  tz_p=updated[4], tz_m=updated[5])
        head, info = s.solve(**controls)
        assert info["workspace_reused"] and info["operator_refreshed_in_place"]
        assert s._session._fast_workspace_holder["workspace"]["cycle_graph"] is graph
        expected, _, _ = independent_system(faces=updated, active=active, bc=bc, values=values,
            source=source, storage=np.zeros(shape), previous=np.zeros(shape))
        np.testing.assert_allclose(head, expected, atol=1e-8, rtol=0.0)
        # Two valid components, each with its own fixed boundary.
        updated[0][:, :, 2] = 0.0
        updated[1][:, :, 3] = 0.0
        s.update_face_conductance(tx_p=updated[0], tx_m=updated[1], ty_p=updated[2], ty_m=updated[3],
                                  tz_p=updated[4], tz_m=updated[5])
        _, topology = s.solve(**controls)
        assert not topology["workspace_reused"]
        assert topology["n_fixed_head_components"] == 2
        assert topology["topology_revision"] == 1
