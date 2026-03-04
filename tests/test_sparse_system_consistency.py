import sys
import unittest
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import spsolve

try:
    from DARCY_WARP_PACKAGE.CPU_FD import solve_darcy_fd_2d_matrix
    from DARCY_WARP_PACKAGE.sparse_operator import build_sparse_system_fd_like
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from DARCY_WARP_PACKAGE.CPU_FD import solve_darcy_fd_2d_matrix
    from DARCY_WARP_PACKAGE.sparse_operator import build_sparse_system_fd_like


class TestSparseSystemConsistency(unittest.TestCase):
    def _run_case(self, use_ghb: bool) -> None:
        rng = np.random.default_rng(12345)
        ny, nx = 9, 11
        dx = 50.0

        T_field = 500.0 + 4500.0 * rng.random((ny, nx))
        R_field = (rng.random((ny, nx)) - 0.5) * 2.0e-4

        active = np.ones((ny, nx), dtype=np.int32)
        active[2:4, 5] = 0
        active[6, 1:3] = 0

        bc_mask = np.zeros((ny, nx), dtype=np.int32)
        bc_mask[0, :] = 1
        bc_mask[-1, :] = 1
        bc_mask[:, -1] = 1
        bc_mask[active == 0] = 0

        bc_values = np.zeros((ny, nx), dtype=np.float64)
        bc_values[bc_mask != 0] = 40.0 + 5.0 * rng.random(np.count_nonzero(bc_mask))

        gh_mask = np.zeros((ny, nx), dtype=np.int32)
        gh_head = np.zeros((ny, nx), dtype=np.float64)
        gh_width = np.zeros((ny, nx), dtype=np.float64)
        if use_ghb:
            gh_mask[ny // 2, :] = 1
            gh_mask[(bc_mask != 0) | (active == 0)] = 0
            gh_head[:, :] = 38.0 + np.linspace(0.0, 4.0, nx, dtype=np.float64)[None, :]
            gh_width[gh_mask != 0] = 15.0 + 10.0 * rng.random(np.count_nonzero(gh_mask))

        T_field = T_field.astype(np.float64)
        R_field = R_field.astype(np.float64)
        T_field[active == 0] = 0.0
        R_field[active == 0] = 0.0

        A_csr, b, free_mask_flat = build_sparse_system_fd_like(
            T_field=T_field,
            R_field=R_field,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            dx=dx,
            gh_mask=gh_mask,
            gh_head=gh_head,
            gh_width=gh_width,
            gh_alpha=1.0,
            aq_thickness=12.0,
        )
        h_sparse = spsolve(A_csr, b).reshape(ny, nx)

        h_fd = solve_darcy_fd_2d_matrix(
            T_field=T_field,
            R_field=R_field,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            dx=dx,
            gh_mask=gh_mask,
            gh_head=gh_head,
            gh_width=gh_width,
            gh_alpha=1.0,
            aq_thickness=12.0,
        )

        expected_free = ((active != 0) & (bc_mask == 0)).reshape(-1)
        np.testing.assert_array_equal(free_mask_flat, expected_free)
        np.testing.assert_allclose(h_sparse, h_fd, rtol=0.0, atol=1.0e-10)

    def test_sparse_system_matches_fd_reference_steady(self) -> None:
        self._run_case(use_ghb=False)

    def test_sparse_system_matches_fd_reference_with_ghb(self) -> None:
        self._run_case(use_ghb=True)


if __name__ == "__main__":
    unittest.main()
