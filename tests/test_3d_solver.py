# SPDX-License-Identifier: AGPL-3.0-only
"""
Tests for the 3D solver integration.
"""

from __future__ import annotations

import unittest

import numpy as np


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
        return True
    except ImportError:
        return False


def _warp_test_device() -> str:
    import warp as wp

    wp.init()
    if wp.is_cuda_available():
        return "cuda:0"
    return "cpu"


WARP_AVAILABLE = _warp_available()


@unittest.skipUnless(WARP_AVAILABLE, "warp not installed")
class Test3DImports(unittest.TestCase):
    """Smoke tests that the 3D modules and factory are importable."""

    def test_factory_import(self):
        from DARCY_WARP_PACKAGE.factory import create_solver

        self.assertTrue(callable(create_solver))

    def test_3d_class_import(self):
        from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

        self.assertTrue(callable(WarpDarcySolver3D))

    def test_kernels_3d_import(self):
        from DARCY_WARP_PACKAGE import kernels_3d

        self.assertIn("apply_A_7point_kernel", kernels_3d.__all__)

    def test_package_lazy_exports(self):
        import DARCY_WARP_PACKAGE as dwp

        self.assertTrue(callable(dwp.create_solver))
        self.assertTrue(callable(dwp.WarpDarcySolver3D))

    def test_3d_wrapper_retains_k_fields_for_unconfined(self):
        from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

        shape = (2, 3, 4)
        kx = np.ones(shape, dtype=np.float64)
        ky = np.full(shape, 2.0, dtype=np.float64)
        kz = np.full(shape, 3.0, dtype=np.float64)
        active = np.ones(shape, dtype=np.int32)
        bc_mask = np.zeros(shape, dtype=np.int32)
        bc_values = np.zeros(shape, dtype=np.float64)
        rhs = np.zeros(shape, dtype=np.float64)

        solver = WarpDarcySolver3D(nx=4, ny=3, nz=2, dx=1.0, dy=1.0, dz=1.0, device="cpu")
        solver.build_from_K_fields(
            kx_field=kx,
            ky_field=ky,
            kz_field=kz,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            rhs=rhs,
        )

        self.assertTrue(np.array_equal(solver._kx_field, kx))
        self.assertTrue(np.array_equal(solver._ky_field, ky))
        self.assertTrue(np.array_equal(solver._kz_field, kz))

    def test_3d_transient_unconfined_is_rejected(self):
        from DARCY_WARP_PACKAGE.solvers_3d import (
            solve_chebyshev_7point_3d,
            solve_multigrid_kcycle_7point_3d,
        )

        shape = (2, 2, 2)
        zeros = np.zeros(shape, dtype=np.float64)
        ones = np.ones(shape, dtype=np.float64)
        active = np.ones(shape, dtype=np.int32)
        bc_mask = np.zeros(shape, dtype=np.int32)

        for solve in (solve_chebyshev_7point_3d, solve_multigrid_kcycle_7point_3d):
            with self.assertRaises(NotImplementedError):
                solve(
                    tx_p=zeros,
                    tx_m=zeros,
                    ty_p=zeros,
                    ty_m=zeros,
                    tz_p=zeros,
                    tz_m=zeros,
                    rhs=zeros,
                    active=active,
                    bc_mask=bc_mask,
                    bc_values=zeros,
                    transient=True,
                    storage_coeff=1.0,
                    dt=1.0,
                    unconfined=True,
                    kx_field=ones,
                    ky_field=ones,
                    kz_field=ones,
                    zbot_field=zeros,
                    device="cpu",
                )

    def test_horizontal_coarsening_preserves_layers_for_benchmark_shape(self):
        from DARCY_WARP_PACKAGE.solvers_3d import _coarsen_max_edge_1x2x2

        mask = np.ones((2, 200, 1000), dtype=np.int32)
        shapes = [mask.shape]
        for _ in range(5):
            mask = _coarsen_max_edge_1x2x2(mask)
            shapes.append(mask.shape)

        self.assertEqual(
            shapes,
            [
                (2, 200, 1000),
                (2, 100, 500),
                (2, 50, 250),
                (2, 25, 125),
                (2, 13, 63),
                (2, 7, 32),
            ],
        )

    def test_kcycle_builds_multiple_horizontal_solver_levels(self):
        from DARCY_WARP_PACKAGE.solvers_3d import solve_multigrid_kcycle_7point_3d

        shape = (2, 8, 16)
        ones = np.ones(shape, dtype=np.float64)
        zeros = np.zeros(shape, dtype=np.float64)
        active = np.ones(shape, dtype=np.int32)
        bc_mask = np.zeros(shape, dtype=np.int32)
        bc_values = np.zeros(shape, dtype=np.float64)
        bc_mask[:, :, 0] = 1
        bc_values[:, :, 0] = 1.0

        _head, info = solve_multigrid_kcycle_7point_3d(
            tx_p=ones,
            tx_m=ones,
            ty_p=ones,
            ty_m=ones,
            tz_p=ones,
            tz_m=ones,
            rhs=zeros,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            initial_head=zeros,
            max_cycles=1,
            max_levels=6,
            min_coarse_n=2,
            device="cpu",
            return_info=True,
        )

        self.assertGreater(info["n_levels"], 1)
        self.assertEqual(info["coarsening_mode"], "horizontal")
        self.assertTrue(all(shape_l[0] == 2 for shape_l in info["level_shapes"]))

    def test_hybrid_chebyshev_vertical_line_smoother_runs(self):
        from DARCY_WARP_PACKAGE.solvers_3d import solve_multigrid_kcycle_7point_3d

        shape = (3, 8, 16)
        ones = np.ones(shape, dtype=np.float64)
        zeros = np.zeros(shape, dtype=np.float64)
        active = np.ones(shape, dtype=np.int32)
        bc_mask = np.zeros(shape, dtype=np.int32)
        bc_values = np.zeros(shape, dtype=np.float64)
        bc_mask[:, :, 0] = 1
        bc_values[:, :, 0] = 1.0

        _head, info = solve_multigrid_kcycle_7point_3d(
            tx_p=ones,
            tx_m=ones,
            ty_p=ones,
            ty_m=ones,
            tz_p=ones,
            tz_m=ones,
            rhs=zeros,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            initial_head=zeros,
            max_cycles=1,
            max_levels=6,
            min_coarse_n=2,
            smoother="chebyshev_vertical_line",
            nu_pre=2,
            nu_post=2,
            nu_coarse=1,
            line_omega=0.8,
            line_sweeps_pre=1,
            line_sweeps_post=1,
            line_sweeps_coarse=1,
            device="cpu",
            return_info=True,
        )

        self.assertEqual(info["smoother"], "chebyshev_vertical_line")
        self.assertEqual(info["line_omega"], 0.8)
        self.assertEqual(info["line_sweeps_pre"], 1)
        self.assertEqual(info["line_sweeps_post"], 1)
        self.assertEqual(info["line_sweeps_coarse"], 1)
        self.assertGreater(info["n_levels"], 1)
        self.assertEqual(info["coarsening_mode"], "horizontal")
        self.assertTrue(all(shape_l[0] == 3 for shape_l in info["level_shapes"]))

    def test_horizontal_transfer_kernels_do_not_mix_layers(self):
        import warp as wp
        from DARCY_WARP_PACKAGE.config import WP_FLOAT
        from DARCY_WARP_PACKAGE.kernels_3d import (
            prolong_bilinear_xy_3d_kernel,
            restrict_blockavg_xy_3d_kernel,
        )

        device = _warp_test_device()
        fine_shape = (2, 4, 4)
        coarse_shape = (2, 2, 2)

        residual = np.zeros(fine_shape, dtype=np.float64)
        residual[0, :, :] = 10.0
        residual[1, :, :] = 100.0
        active = np.ones(fine_shape, dtype=np.int32)
        bc_mask = np.zeros(fine_shape, dtype=np.int32)
        coarse_rhs = np.zeros(coarse_shape, dtype=np.float64)

        residual_wp = wp.array(residual, dtype=WP_FLOAT, device=device)
        active_wp = wp.array(active, dtype=wp.int32, device=device)
        bc_mask_wp = wp.array(bc_mask, dtype=wp.int32, device=device)
        coarse_rhs_wp = wp.array(coarse_rhs, dtype=WP_FLOAT, device=device)

        wp.launch(
            kernel=restrict_blockavg_xy_3d_kernel,
            dim=coarse_shape,
            inputs=[residual_wp, active_wp, bc_mask_wp, coarse_rhs_wp, 4, 4, 2, 2, 2, 2],
            device=device,
        )
        restricted = coarse_rhs_wp.numpy()
        self.assertTrue(np.allclose(restricted[0], 10.0))
        self.assertTrue(np.allclose(restricted[1], 100.0))

        coarse_corr = np.zeros(coarse_shape, dtype=np.float64)
        coarse_corr[0, :, :] = 1.0
        coarse_corr[1, :, :] = 9.0
        fine_corr = np.zeros(fine_shape, dtype=np.float64)
        coarse_corr_wp = wp.array(coarse_corr, dtype=WP_FLOAT, device=device)
        fine_corr_wp = wp.array(fine_corr, dtype=WP_FLOAT, device=device)

        wp.launch(
            kernel=prolong_bilinear_xy_3d_kernel,
            dim=fine_shape,
            inputs=[coarse_corr_wp, fine_corr_wp, 4, 4, 2, 2, 2, 2],
            device=device,
        )
        prolonged = fine_corr_wp.numpy()
        self.assertTrue(np.allclose(prolonged[0], 1.0))
        self.assertTrue(np.allclose(prolonged[1], 9.0))


class TestFactoryValidation(unittest.TestCase):
    """Argument validation should work even without a GPU."""

    def test_invalid_dim(self):
        from DARCY_WARP_PACKAGE.factory import create_solver

        with self.assertRaises(ValueError):
            create_solver(dim=1, nx=10, ny=10, dx=1.0)

    def test_3d_requires_nz(self):
        from DARCY_WARP_PACKAGE.factory import create_solver

        with self.assertRaises(ValueError):
            create_solver(dim=3, nx=10, ny=10, dx=1.0, solver="kcycle")

    def test_3d_requires_dz(self):
        from DARCY_WARP_PACKAGE.factory import create_solver

        with self.assertRaises(ValueError):
            create_solver(dim=3, nx=10, ny=10, nz=5, dx=1.0, solver="kcycle")

    def test_2d_solver_choice(self):
        from DARCY_WARP_PACKAGE.factory import create_solver

        with self.assertRaises(ValueError):
            create_solver(dim=2, nx=10, ny=10, dx=1.0, solver="invalid")


@unittest.skipUnless(WARP_AVAILABLE, "warp not installed")
class Test3DVerticalLineSmoother(unittest.TestCase):
    """Reference checks for one vertical-line relaxation sweep."""

    @staticmethod
    def _sanitize_initial(initial, active, bc_mask, bc_values):
        x_old = np.asarray(initial, dtype=np.float64).copy()
        x_old[bc_mask != 0] = bc_values[bc_mask != 0]
        x_old[active == 0] = 0.0
        return x_old

    @staticmethod
    def _face_conductance(field, active, k, j, i, kk, jj, ii):
        nz, ny, nx = active.shape
        if kk < 0 or kk >= nz or jj < 0 or jj >= ny or ii < 0 or ii >= nx:
            return 0.0
        if active[kk, jj, ii] == 0:
            return 0.0
        value = float(field[k, j, i])
        if value < 0.0:
            return 0.0
        return value

    @classmethod
    def _vertical_line_reference(
        cls,
        tx_p,
        tx_m,
        ty_p,
        ty_m,
        tz_p,
        tz_m,
        active,
        bc_mask,
        storage_diag,
        rhs,
        x_old,
        bc_values,
        omega,
    ):
        nz, ny, nx = rhs.shape
        out = np.zeros_like(x_old, dtype=np.float64)

        for j in range(ny):
            for i in range(nx):
                lower = np.zeros(nz, dtype=np.float64)
                diag = np.ones(nz, dtype=np.float64)
                upper = np.zeros(nz, dtype=np.float64)
                d = np.zeros(nz, dtype=np.float64)

                for k in range(nz):
                    if active[k, j, i] == 0:
                        d[k] = 0.0
                        continue
                    if bc_mask[k, j, i] != 0:
                        d[k] = bc_values[k, j, i]
                        continue

                    row_diag = max(float(storage_diag[k, j, i]), 0.0)
                    row_rhs = float(rhs[k, j, i])

                    for field, kk, jj, ii in (
                        (tx_p, k, j, i + 1),
                        (tx_m, k, j, i - 1),
                        (ty_p, k, j + 1, i),
                        (ty_m, k, j - 1, i),
                    ):
                        conductance = cls._face_conductance(field, active, k, j, i, kk, jj, ii)
                        if conductance <= 0.0:
                            continue
                        row_diag += conductance
                        if bc_mask[kk, jj, ii] != 0:
                            row_rhs += conductance * float(bc_values[kk, jj, ii])
                        else:
                            row_rhs += conductance * float(x_old[kk, jj, ii])

                    conductance = cls._face_conductance(tz_m, active, k, j, i, k - 1, j, i)
                    if conductance > 0.0:
                        row_diag += conductance
                        if bc_mask[k - 1, j, i] != 0:
                            row_rhs += conductance * float(bc_values[k - 1, j, i])
                        else:
                            lower[k] = -conductance

                    conductance = cls._face_conductance(tz_p, active, k, j, i, k + 1, j, i)
                    if conductance > 0.0:
                        row_diag += conductance
                        if bc_mask[k + 1, j, i] != 0:
                            row_rhs += conductance * float(bc_values[k + 1, j, i])
                        else:
                            upper[k] = -conductance

                    if row_diag < 1.0e-12:
                        diag[k] = 1.0
                        d[k] = row_rhs
                    else:
                        diag[k] = row_diag
                        d[k] = row_rhs

                c_prime = np.zeros(nz, dtype=np.float64)
                d_prime = np.zeros(nz, dtype=np.float64)
                c_prime[0] = upper[0] / diag[0]
                d_prime[0] = d[0] / diag[0]
                for k in range(1, nz):
                    denom = diag[k] - lower[k] * c_prime[k - 1]
                    c_prime[k] = upper[k] / denom
                    d_prime[k] = (d[k] - lower[k] * d_prime[k - 1]) / denom

                solved = np.zeros(nz, dtype=np.float64)
                solved[-1] = d_prime[-1]
                for k in range(nz - 2, -1, -1):
                    solved[k] = d_prime[k] - c_prime[k] * solved[k + 1]

                for k in range(nz):
                    if active[k, j, i] == 0:
                        out[k, j, i] = 0.0
                    elif bc_mask[k, j, i] != 0:
                        out[k, j, i] = bc_values[k, j, i]
                    else:
                        out[k, j, i] = x_old[k, j, i] + float(omega) * (solved[k] - x_old[k, j, i])

        return out

    def test_vertical_line_one_sweep_matches_numpy_reference(self):
        import warp as wp
        from DARCY_WARP_PACKAGE.config import WP_FLOAT
        from DARCY_WARP_PACKAGE.kernels_3d import vertical_line_relaxation_7point_kernel

        rng = np.random.default_rng(27)
        shape = (5, 3, 4)

        tx_p = rng.uniform(-0.25, 1.8, size=shape)
        tx_m = rng.uniform(-0.25, 1.8, size=shape)
        ty_p = rng.uniform(-0.25, 1.8, size=shape)
        ty_m = rng.uniform(-0.25, 1.8, size=shape)
        tz_p = rng.uniform(-0.25, 2.5, size=shape)
        tz_m = rng.uniform(-0.25, 2.5, size=shape)
        storage_diag = rng.uniform(-0.2, 1.0, size=shape) + 2.0
        rhs = rng.normal(size=shape)
        initial = rng.normal(size=shape)

        active = np.ones(shape, dtype=np.int32)
        active[2, 1, 2] = 0
        active[4, 2, 3] = 0

        bc_mask = np.zeros(shape, dtype=np.int32)
        bc_values = np.zeros(shape, dtype=np.float64)
        bc_mask[0, 1, 1] = 1
        bc_values[0, 1, 1] = 4.0
        bc_mask[3, 1, 1] = 1
        bc_values[3, 1, 1] = -2.0
        bc_mask[:, 0, 0] = 1
        bc_values[:, 0, 0] = 1.5
        bc_mask[active == 0] = 1
        bc_values[active == 0] = 99.0

        omega = 0.65
        x_old = self._sanitize_initial(initial, active, bc_mask, bc_values)
        expected = self._vertical_line_reference(
            tx_p,
            tx_m,
            ty_p,
            ty_m,
            tz_p,
            tz_m,
            active,
            bc_mask,
            storage_diag,
            rhs,
            x_old,
            bc_values,
            omega,
        )

        device = "cpu"
        x_new_wp = wp.zeros(shape, dtype=WP_FLOAT, device=device)
        wp.launch(
            kernel=vertical_line_relaxation_7point_kernel,
            dim=(shape[1] * shape[2],),
            inputs=[
                wp.array(tx_p, dtype=WP_FLOAT, device=device),
                wp.array(tx_m, dtype=WP_FLOAT, device=device),
                wp.array(ty_p, dtype=WP_FLOAT, device=device),
                wp.array(ty_m, dtype=WP_FLOAT, device=device),
                wp.array(tz_p, dtype=WP_FLOAT, device=device),
                wp.array(tz_m, dtype=WP_FLOAT, device=device),
                wp.array(active, dtype=wp.int32, device=device),
                wp.array(bc_mask, dtype=wp.int32, device=device),
                wp.array(storage_diag, dtype=WP_FLOAT, device=device),
                wp.array(rhs, dtype=WP_FLOAT, device=device),
                wp.array(x_old, dtype=WP_FLOAT, device=device),
                wp.array(bc_values, dtype=WP_FLOAT, device=device),
                float(omega),
                wp.zeros(shape, dtype=WP_FLOAT, device=device),
                wp.zeros(shape, dtype=WP_FLOAT, device=device),
                int(shape[2]),
                int(shape[1]),
                int(shape[0]),
                x_new_wp,
            ],
            device=device,
        )
        head = x_new_wp.numpy()

        self.assertTrue(np.allclose(head, expected, rtol=1.0e-6, atol=1.0e-6))
        self.assertTrue(np.all(head[active == 0] == 0.0))
        self.assertTrue(np.all(head[(active != 0) & (bc_mask != 0)] == bc_values[(active != 0) & (bc_mask != 0)]))


@unittest.skipUnless(WARP_AVAILABLE, "warp not installed")
class Test3DSparseReference(unittest.TestCase):
    """Compare 3D solver result to a scipy 7-point reference on a tiny grid."""

    def _build_3d_sparse_system(
        self,
        tx_p: np.ndarray,
        tx_m: np.ndarray,
        ty_p: np.ndarray,
        ty_m: np.ndarray,
        tz_p: np.ndarray,
        tz_m: np.ndarray,
        rhs: np.ndarray,
        active: np.ndarray,
        bc_mask: np.ndarray,
        bc_values: np.ndarray,
    ):
        """Assemble A h = b for the 7-point stencil used by the 3D Warp kernels."""
        from scipy.sparse import csr_matrix

        nz, ny, nx = rhs.shape
        n_cells = nz * ny * nx

        def idx(k: int, j: int, i: int) -> int:
            return (k * ny + j) * nx + i

        row = []
        col = []
        data = []
        b = np.zeros(n_cells, dtype=np.float64)

        for k in range(nz):
            for j in range(ny):
                for i in range(nx):
                    r = idx(k, j, i)
                    if active[k, j, i] == 0:
                        row.append(r)
                        col.append(r)
                        data.append(1.0)
                        b[r] = 0.0
                        continue
                    if bc_mask[k, j, i] != 0:
                        row.append(r)
                        col.append(r)
                        data.append(1.0)
                        b[r] = bc_values[k, j, i]
                        continue

                    cxp = float(tx_p[k, j, i])
                    cxm = float(tx_m[k, j, i])
                    cyp = float(ty_p[k, j, i])
                    cym = float(ty_m[k, j, i])
                    czp = float(tz_p[k, j, i])
                    czm = float(tz_m[k, j, i])

                    # Drop couplings to inactive neighbours.
                    if i + 1 >= nx or active[k, j, i + 1] == 0:
                        cxp = 0.0
                    if i - 1 < 0 or active[k, j, i - 1] == 0:
                        cxm = 0.0
                    if j + 1 >= ny or active[k, j + 1, i] == 0:
                        cyp = 0.0
                    if j - 1 < 0 or active[k, j - 1, i] == 0:
                        cym = 0.0
                    if k + 1 >= nz or active[k + 1, j, i] == 0:
                        czp = 0.0
                    if k - 1 < 0 or active[k - 1, j, i] == 0:
                        czm = 0.0

                    diag = cxp + cxm + cyp + cym + czp + czm
                    if diag < 1.0e-12:
                        row.append(r)
                        col.append(r)
                        data.append(1.0)
                        b[r] = 0.0
                        continue

                    row.append(r)
                    col.append(r)
                    data.append(diag)

                    if cxp > 0.0:
                        row.append(r)
                        col.append(idx(k, j, i + 1))
                        data.append(-cxp)
                    if cxm > 0.0:
                        row.append(r)
                        col.append(idx(k, j, i - 1))
                        data.append(-cxm)
                    if cyp > 0.0:
                        row.append(r)
                        col.append(idx(k, j + 1, i))
                        data.append(-cyp)
                    if cym > 0.0:
                        row.append(r)
                        col.append(idx(k, j - 1, i))
                        data.append(-cym)
                    if czp > 0.0:
                        row.append(r)
                        col.append(idx(k + 1, j, i))
                        data.append(-czp)
                    if czm > 0.0:
                        row.append(r)
                        col.append(idx(k - 1, j, i))
                        data.append(-czm)

                    b[r] = rhs[k, j, i]

        A = csr_matrix((data, (row, col)), shape=(n_cells, n_cells))
        return A, b

    def test_3d_solver_matches_scipy_reference(self):
        from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

        nz, ny, nx = 4, 5, 6
        dx = dy = dz = 1.0

        np.random.seed(0)
        kx = np.exp(np.random.randn(nz, ny, nx)) * 10.0
        ky = np.exp(np.random.randn(nz, ny, nx)) * 10.0
        kz = np.exp(np.random.randn(nz, ny, nx)) * 10.0

        active = np.ones((nz, ny, nx), dtype=np.int32)
        # carve a small inactive block
        active[1:3, 1:3, 1:3] = 0

        bc_mask = np.zeros((nz, ny, nx), dtype=np.int32)
        bc_values = np.zeros((nz, ny, nx), dtype=np.float64)
        # Dirichlet on two faces
        bc_mask[:, :, 0] = 1
        bc_values[:, :, 0] = 10.0
        bc_mask[:, -1, :] = 1
        bc_values[:, -1, :] = 5.0

        rhs = np.random.rand(nz, ny, nx).astype(np.float64)

        solver = WarpDarcySolver3D(
            nx=nx,
            ny=ny,
            nz=nz,
            dx=dx,
            dy=dy,
            dz=dz,
            device=_warp_test_device(),
            solver="chebyshev",
        )
        solver.build_from_K_fields(
            kx_field=kx,
            ky_field=ky,
            kz_field=kz,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            rhs=rhs,
        )
        head_warp, info = solver.solve(
            max_iter=200,
            rel_tol=1.0e-8,
            abs_tol_min=1.0e-10,
        )
        self.assertEqual(head_warp.shape, (nz, ny, nx))

        # Reference solve via scipy
        from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k

        tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = build_7point_face_conductance_from_k(
            kx_field=kx,
            ky_field=ky,
            kz_field=kz,
            active=active,
            dx=dx,
            dy=dy,
            dz=dz,
        )
        A, b = self._build_3d_sparse_system(
            tx_p, tx_m, ty_p, ty_m, tz_p, tz_m,
            rhs, active, bc_mask, bc_values,
        )
        from scipy.sparse.linalg import spsolve

        head_ref = spsolve(A, b).reshape(nz, ny, nx)

        free = (active != 0) & (bc_mask == 0)
        diff = np.abs(head_warp[free] - head_ref[free])
        self.assertLess(np.max(diff), 1.0e-4)


if __name__ == "__main__":
    unittest.main()
