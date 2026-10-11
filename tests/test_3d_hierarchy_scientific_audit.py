"""Implemented transfer matrices, coarse symmetry, and fine-grid acceptance."""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.config import WP_FLOAT
from DARCY_WARP_PACKAGE.kernels_3d import restrict_blockavg_axes_3d_kernel, prolong_bilinear_axes_3d_kernel
from DARCY_WARP_PACKAGE.solvers.fast_3d import _build_levels
from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from test_3d_scientific_audit import independent_system, device


def transfer_matrices(*, fine_shape, coarse_shape, active, bc, fy, fx):
    """Probe actual fast-solver kernels with basis vectors, including masks."""
    dev = device()
    nf, nc = int(np.prod(fine_shape)), int(np.prod(coarse_shape))
    rmat, pmat = np.zeros((nc, nf)), np.zeros((nf, nc))
    f = wp.zeros(fine_shape, dtype=WP_FLOAT, device=dev)
    c = wp.zeros(coarse_shape, dtype=WP_FLOAT, device=dev)
    act = wp.array(active, dtype=wp.int32, device=dev)
    bcm = wp.array(bc, dtype=wp.int32, device=dev)
    dims = [fine_shape[2], fine_shape[1], fine_shape[0], coarse_shape[2], coarse_shape[1], coarse_shape[0], fy, fx]
    for i in range(nf):
        basis = np.zeros(fine_shape)
        basis.ravel()[i] = 1.0
        f.assign(basis)
        wp.launch(kernel=restrict_blockavg_axes_3d_kernel, dim=coarse_shape,
                  inputs=[f, act, bcm, c, *dims], device=dev)
        rmat[:, i] = c.numpy().ravel()
    for i in range(nc):
        basis = np.zeros(coarse_shape)
        basis.ravel()[i] = 1.0
        c.assign(basis)
        wp.launch(kernel=prolong_bilinear_axes_3d_kernel, dim=fine_shape,
                  inputs=[c, f, *dims], device=dev)
        pmat[:, i] = f.numpy().ravel()
    return rmat, pmat


@pytest.mark.parametrize("irregular", [False, True])
@pytest.mark.parametrize("anisotropy", [1e-3, 10.0])
def test_actual_hierarchy_is_symmetric_approximation_not_exact_rap(irregular, anisotropy):
    shape = (2, 5, 7)
    rng = np.random.default_rng(37)
    k = np.exp(rng.uniform(-3.0, 3.0, size=shape))
    active = np.ones(shape, dtype=np.int32)
    if irregular:
        active[:, 2, 3:5] = 0
        active[0, 3, 1] = 0
    bc = np.zeros(shape, dtype=np.int32)
    bc[:, :, 0] = 1
    bc[0, 0, 1] = 1
    bc[active == 0] = 0
    values = np.full(shape, 5.0)
    values[:, :, 0] = 3.0
    source = np.full(shape, 0.01)
    faces = build_7point_face_conductance_from_k(kx_field=k, ky_field=k, kz_field=k * anisotropy,
                                               active=active, dx=1.0)
    levels = _build_levels(faces=faces, active=active, bc_mask=bc, bc_values=values,
                           rhs=source, storage_diag=np.zeros(shape), max_levels=3, min_coarse_n=1)
    coarse = levels[1]
    coarse_shape = coarse["active"].shape
    rmat, pmat = transfer_matrices(fine_shape=shape, coarse_shape=coarse_shape,
                                   active=active, bc=bc, fy=2, fx=2)
    np.testing.assert_allclose(rmat.sum(axis=1), np.where(rmat.sum(axis=1) > 0, 1, 0), atol=1e-15)
    np.testing.assert_allclose(pmat.sum(axis=1), 1.0, atol=1e-15)
    assert np.any(pmat == 0.5) or np.any(pmat == 0.25)
    assert np.all(rmat[:, (active == 0).ravel() | (bc != 0).ravel()] == 0.0)
    zero = np.zeros(shape)
    exact, af, _ = independent_system(faces=faces, active=active, bc=bc, values=values,
                                     source=source, storage=zero, previous=zero)
    _, ac, _ = independent_system(faces=coarse["faces"], active=coarse["active"], bc=coarse["bc_mask"],
                                  values=coarse["bc_values"], source=coarse["rhs"],
                                  storage=coarse["storage"], previous=np.zeros(coarse_shape))
    free = (active != 0) & (bc == 0)
    rap = rmat @ af @ (pmat * free.ravel()[:, None])
    assert np.linalg.norm(ac.toarray() - rap) > 1e-3
    np.testing.assert_allclose(ac.toarray(), ac.toarray().T, atol=1e-14, rtol=0.0)
    assert np.linalg.eigvalsh(ac.toarray()).min() >= -1e-12
    np.testing.assert_allclose(ac @ np.ones(ac.shape[0]), coarse["storage"].ravel(), atol=1e-13)
    with WarpDarcySolver3D(nx=7, ny=5, nz=2, dx=1.0, device=device(), implementation="fast") as s:
        s.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k * anisotropy, active=active,
                              bc_mask=bc, bc_values=values, rhs=source)
        head, info = s.solve(max_cycles=1500, min_coarse_n=1, max_levels=3,
                             rel_tol=1e-11, abs_tol_min=1e-12, dh_rms_tol=1e-10, check_every_no=1)
    if not info["converged"]:
        print("HIERARCHY_DIAGNOSTIC", info, "head_error", np.max(np.abs(head - exact)), flush=True)
    assert info["converged"] and info["final_physical_acceptance"]
    np.testing.assert_allclose(head[active != 0], exact[active != 0], atol=1e-8, rtol=0.0)
