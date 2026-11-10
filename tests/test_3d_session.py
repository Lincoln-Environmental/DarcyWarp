import numpy as np

from DARCY_WARP_PACKAGE.solvers.session_3d import ThreeDOperatorSession
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def _session() -> ThreeDOperatorSession:
    shape = (2, 3, 4)
    faces = tuple(np.ones(shape, dtype=np.float64) for _ in range(6))
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[:, :, 0] = 1
    return ThreeDOperatorSession(
        faces=faces,
        active=np.ones(shape, dtype=np.int32),
        bc_mask=bc_mask,
        bc_values=np.zeros(shape, dtype=np.float64),
        rhs=np.zeros(shape, dtype=np.float64),
        initial_head=np.zeros(shape, dtype=np.float64),
        dx=1.0,
        dy=2.0,
        dz=3.0,
    )


def test_session_updates_are_in_place_and_versioned():
    session = _session()
    rhs = np.full(session.shape, 2.0)
    session.update_rhs(rhs)
    assert session.revision == 1
    assert np.all(session.rhs == 2.0)

    faces = tuple(np.full(session.shape, 3.0) for _ in range(6))
    session.update_face_conductance(faces)
    assert session.revision == 2
    assert all(np.all(face == 3.0) for face in session.faces)

    session._fast_workspace_holder["sentinel"] = object()
    boundary_values = np.full(session.shape, 4.0)
    original_identity = id(session.bc_values)
    session.update_boundary_values(boundary_values)
    assert session.revision == 3
    assert id(session.bc_values) == original_identity
    assert np.all(session.bc_values == 4.0)
    assert "sentinel" in session._fast_workspace_holder
    assert session.operator_revision == 2
    assert session.topology_revision == 0


def test_session_rejects_inconsistent_updates_and_closes():
    session = _session()
    with np.testing.assert_raises(ValueError):
        session.update_rhs(np.zeros((1, 1, 1)))
    with np.testing.assert_raises(ValueError):
        session.update_rhs(np.full(session.shape, np.nan))
    with np.testing.assert_raises(ValueError):
        session.update_boundary_values(np.zeros((1, 1, 1)))
    with np.testing.assert_raises(ValueError):
        session.update_boundary_values(np.full(session.shape, np.nan))
    with np.testing.assert_raises(ValueError):
        session.update_face_conductance(
            tuple(np.full(session.shape, np.nan) for _ in range(6))
        )
    with np.testing.assert_raises(ValueError):
        session.update_face_conductance(tuple(np.full(session.shape, -1.0) for _ in range(6)))
    with np.testing.assert_raises(ValueError):
        session.update_face_conductance(tuple(np.ones(session.shape) for _ in range(5)))
    with np.testing.assert_raises(ValueError):
        session.solve(solver="not-a-solver", device="cpu")
    session.close()
    with np.testing.assert_raises(RuntimeError):
        session.update_initial_head(None)


def test_session_rejects_negative_initial_faces_and_nonfinite_spacing():
    shape = (1, 1, 2)
    common = dict(
        active=np.ones(shape, dtype=np.int32),
        bc_mask=np.zeros(shape, dtype=np.int32),
        bc_values=np.zeros(shape), rhs=np.zeros(shape), initial_head=np.zeros(shape),
        dx=1.0, dy=1.0, dz=1.0,
    )
    with np.testing.assert_raises(ValueError):
        ThreeDOperatorSession(faces=tuple(np.array([[[-1.0, 0.0]]]) for _ in range(6)), **common)
    bad_spacing = dict(common)
    bad_spacing["dx"] = np.nan
    with np.testing.assert_raises(ValueError):
        ThreeDOperatorSession(faces=tuple(np.ones(shape) for _ in range(6)), **bad_spacing)


def test_wrapper_uses_session_owned_arrays_after_external_mutation():
    shape = (1, 1, 3)
    faces = [np.ones(shape) for _ in range(6)]
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape)
    rhs = np.zeros(shape)
    with WarpDarcySolver3D(nx=3, ny=1, nz=1, dx=1.0, device="cpu") as solver:
        solver.build_from_face_conductance(
            tx_p=faces[0], tx_m=faces[1], ty_p=faces[2], ty_m=faces[3],
            tz_p=faces[4], tz_m=faces[5], active=active, bc_mask=bc_mask,
            bc_values=bc_values, rhs=rhs,
        )
        faces[0][...] = 99.0
        rhs[...] = 7.0
        assert np.all(solver._tx_p != 99.0)
        assert np.all(solver._rhs == 0.0)
        assert solver._tx_p is solver._session.faces[0]
        assert solver._rhs is solver._session.rhs


def test_fast_backend_accepts_confined_transient_controls():
    shape = (1, 1, 3)
    faces = tuple(np.ones(shape) for _ in range(6))
    session = ThreeDOperatorSession(
        faces=faces, active=np.ones(shape, dtype=np.int32),
        bc_mask=np.zeros(shape, dtype=np.int32), bc_values=np.zeros(shape),
        rhs=np.zeros(shape), initial_head=np.zeros(shape), dx=1.0, dy=1.0, dz=1.0,
    )
    head, info = session.solve(
        solver="kcycle",
        implementation="fast",
        device="cpu",
        transient=True,
        storage_coeff=0.1,
        dt=1.0,
        head_prev=np.zeros(shape),
        max_cycles=2,
        max_levels=2,
        min_coarse_n=1,
        check_every_no=1,
    )
    assert head.shape == shape
    assert info["transient"] is True


def test_session_preserves_return_info_choice():
    session = _session()
    head = session.solve(
        solver="kcycle",
        device="cpu",
        return_info=False,
        max_cycles=2,
        max_levels=2,
        min_coarse_n=1,
        check_every_no=1,
    )
    assert head.shape == session.shape


def test_wrapper_preserves_return_info_choice():
    shape = (2, 2, 2)
    k = np.ones(shape, dtype=np.float64)
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_mask[0, 0, 0] = 1
    bc_values = np.zeros(shape, dtype=np.float64)
    with WarpDarcySolver3D(nx=2, ny=2, nz=2, dx=1.0, device="cpu") as solver:
        solver.build_from_K_fields(
            kx_field=k,
            ky_field=k,
            kz_field=k,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            rhs=np.zeros(shape, dtype=np.float64),
            initial_head=bc_values,
        )
        head = solver.solve(
            return_info=False,
            max_cycles=2,
            max_levels=2,
            min_coarse_n=1,
            check_every_no=1,
        )
    assert isinstance(head, np.ndarray)
    assert head.shape == shape
