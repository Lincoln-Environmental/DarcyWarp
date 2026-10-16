# SPDX-License-Identifier: AGPL-3.0-only

from __future__ import annotations

import numpy as np


def build_sparse_system_fd_like(
    T_field: np.ndarray,
    R_field: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    dx: float,
    gh_mask: np.ndarray | None = None,
    gh_head: np.ndarray | None = None,
    gh_width: np.ndarray | None = None,
    gh_alpha: float = 1.0,
    aq_thickness: float = 1.0,
):
    """
    Assemble A h = b for the finite-volume FD operator used by Warp kernels.

    Returns
    -------
    A_csr : scipy.sparse.csr_matrix
    b : np.ndarray[float64]
    free_mask_flat : np.ndarray[bool]
    """
    from scipy.sparse import lil_matrix

    T_field = np.asarray(T_field, dtype=np.float64)
    R_field = np.asarray(R_field, dtype=np.float64)
    active = np.asarray(active, dtype=np.int32)
    bc_mask = np.asarray(bc_mask, dtype=np.int32)
    bc_values = np.asarray(bc_values, dtype=np.float64)

    ny, nx = T_field.shape
    n_cells = int(nx * ny)
    dx_f = float(dx)
    dx2 = dx_f * dx_f
    tiny = 1.0e-12

    if gh_mask is None:
        gh_mask = np.zeros((ny, nx), dtype=np.int32)
    else:
        gh_mask = np.asarray(gh_mask, dtype=np.int32)
    if gh_head is None:
        gh_head = np.zeros((ny, nx), dtype=np.float64)
    else:
        gh_head = np.asarray(gh_head, dtype=np.float64)
    if gh_width is None:
        gh_width = np.zeros((ny, nx), dtype=np.float64)
    else:
        gh_width = np.asarray(gh_width, dtype=np.float64)

    def idx(j: int, i: int) -> int:
        return int(j * nx + i)

    def harmonic(a_val: float, b_val: float) -> float:
        if a_val <= 0.0 or b_val <= 0.0:
            return 0.0
        return 2.0 * a_val * b_val / (a_val + b_val)

    A = lil_matrix((n_cells, n_cells), dtype=np.float64)
    b = np.zeros(n_cells, dtype=np.float64)

    for j in range(ny):
        for i in range(nx):
            k = idx(j, i)

            if active[j, i] == 0:
                A[k, k] = 1.0
                b[k] = 0.0
                continue

            if bc_mask[j, i] != 0:
                A[k, k] = 1.0
                b[k] = float(bc_values[j, i])
                continue

            T_c = float(T_field[j, i])

            T_e = 0.0
            T_w = 0.0
            T_n = 0.0
            T_s = 0.0

            if i + 1 < nx and active[j, i + 1] != 0:
                T_e = harmonic(T_c, float(T_field[j, i + 1]))
            if i - 1 >= 0 and active[j, i - 1] != 0:
                T_w = harmonic(T_c, float(T_field[j, i - 1]))
            if j - 1 >= 0 and active[j - 1, i] != 0:
                T_n = harmonic(T_c, float(T_field[j - 1, i]))
            if j + 1 < ny and active[j + 1, i] != 0:
                T_s = harmonic(T_c, float(T_field[j + 1, i]))

            C_gh = 0.0
            if gh_mask[j, i] != 0 and aq_thickness > 0.0:
                width = float(gh_width[j, i])
                if width > 0.0 and not np.isnan(width):
                    C_gh = float(gh_alpha) * T_c / float(aq_thickness) * width * dx_f

            diag = T_e + T_w + T_n + T_s + C_gh
            if diag < tiny:
                A[k, k] = 1.0
                b[k] = 0.0
                continue

            A[k, k] = diag
            if T_e > 0.0:
                A[k, idx(j, i + 1)] = -T_e
            if T_w > 0.0:
                A[k, idx(j, i - 1)] = -T_w
            if T_n > 0.0:
                A[k, idx(j - 1, i)] = -T_n
            if T_s > 0.0:
                A[k, idx(j + 1, i)] = -T_s

            rhs = float(R_field[j, i]) * dx2
            if C_gh > 0.0:
                rhs += C_gh * float(gh_head[j, i])
            b[k] = rhs

    free_mask = (active != 0) & (bc_mask == 0)
    return A.tocsr(), b, free_mask.reshape(-1)
