import time
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
from DARCY_WARP_PACKAGE.model_builder import build_truth_inputs


def solve_darcy_fd_2d_matrix(
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
        aq_thickness: float = 10.0, # controls GHB conductance
) -> np.ndarray:
    """
    Sparse FD reference solver for div(T grad h) with recharge and optional GHB.

    This is the system we mirror in the Warp CG implementation.

    :param T_field: transmissivity [ny, nx]
    :param R_field: recharge [ny, nx]
    :param active: 1 for active cells, 0 for inactive
    :param bc_mask: 1 for Dirichlet cells
    :param bc_values: Dirichlet head values
    :param dx: cell size
    :param gh_mask: 1 for general head boundary cells, 0 otherwise
    :param gh_head: external head for GHB cells [ny, nx]
    :param gh_width: effective boundary width for GHB cells [ny, nx]
    :param gh_alpha: scaling factor for GHB conductance
    :param aq_thickness: aquifer thickness for GHB conductance calculation
    :return: head field [ny, nx]

    """
    T_field = np.asarray(T_field, dtype=np.float64)
    R_field = np.asarray(R_field, dtype=np.float64)
    active = np.asarray(active, dtype=np.int32)
    bc_mask = np.asarray(bc_mask, dtype=np.int32)
    bc_values = np.asarray(bc_values, dtype=np.float64)

    ny, nx = T_field.shape
    n_cells = nx * ny

    if gh_mask is None:
        gh_mask = np.zeros_like(T_field, dtype=np.int32)
        gh_head = np.zeros_like(T_field, dtype=np.float64)
        gh_width = np.zeros_like(T_field, dtype=np.float64)
    else:
        gh_mask = np.asarray(gh_mask, dtype=np.int32)
        gh_head = np.asarray(gh_head, dtype=np.float64)
        gh_width = np.asarray(gh_width, dtype=np.float64)

    def idx(j, i):
        return j * nx + i

    A = lil_matrix((n_cells, n_cells), dtype=np.float64)
    b = np.zeros(n_cells, dtype=np.float64)

    dx2 = dx * dx
    tiny = 1.0e-12

    for j in range(ny):
        for i in range(nx):
            k = idx(j, i)

            if active[j, i] == 0:
                # inactive cell
                A[k, k] = 1.0
                b[k] = 0.0
                continue

            if bc_mask[j, i] != 0:
                # Dirichlet cell
                A[k, k] = 1.0
                b[k] = bc_values[j, i]
                continue

            T_c = T_field[j, i]

            def harmonic(a_val, b_val):
                if a_val <= 0.0 or b_val <= 0.0:
                    return 0.0
                return 2.0 * a_val * b_val / (a_val + b_val)

            T_e = 0.0
            T_w = 0.0
            T_n = 0.0
            T_s = 0.0

            if i + 1 < nx and active[j, i + 1] != 0:
                T_e = harmonic(T_c, T_field[j, i + 1])
            if i - 1 >= 0 and active[j, i - 1] != 0:
                T_w = harmonic(T_c, T_field[j, i - 1])
            if j - 1 >= 0 and active[j - 1, i] != 0:
                T_n = harmonic(T_c, T_field[j - 1, i])
            if j + 1 < ny and active[j + 1, i] != 0:
                T_s = harmonic(T_c, T_field[j + 1, i])

            sum_T = T_e + T_w + T_n + T_s

            # GHB conductance term, same form as Warp:
            # C_gh = gh_alpha * T_c * width / dx
            C_gh = 0.0
            if gh_mask[j, i] != 0:
                width = gh_width[j, i]
                if width > 0.0 and not np.isnan(width):
                    C_gh = gh_alpha * T_c/aq_thickness * width * dx

            total_diag = sum_T + C_gh

            if total_diag < tiny:
                A[k, k] = 1.0
                b[k] = 0.0
                continue

            # diagonal coefficient
            A[k, k] = total_diag

            # neighbors
            if T_e > 0.0:
                k_e = idx(j, i + 1)
                A[k, k_e] = -T_e
            if T_w > 0.0:
                k_w = idx(j, i - 1)
                A[k, k_w] = -T_w
            if T_n > 0.0:
                k_n = idx(j - 1, i)
                A[k, k_n] = -T_n
            if T_s > 0.0:
                k_s = idx(j + 1, i)
                A[k, k_s] = -T_s

            # RHS: recharge + GHB source term
            rhs = R_field[j, i] * dx2
            if C_gh > 0.0:
                rhs += C_gh * gh_head[j, i]

            b[k] = rhs

    A_csr = A.tocsr()
    h_flat = spsolve(A_csr, b)
    h = h_flat.reshape(ny, nx)

    return h



def run_fd_truth_forward(
        nx: int,
        ny: int,
        dx: float,
        T_truth: float | np.ndarray,
        R_truth: float | np.ndarray,
        use_ghb: bool = False,
        gh_alpha: float = 1.0,
        aq_thickness: float = 10.0, # controls GHB conductance
        width: float = 100.0,
) -> tuple:
    """
    Run the FD matrix solver on the same synthetic problem.

    :return: (head field, elapsed seconds)
    """
    (
        T_field,
        R_field,
        active,
        bc_mask,
        bc_values,
        gh_mask,
        gh_head,
        gh_width,
    ) = build_truth_inputs(
        nx=nx,
        ny=ny,
        dx=dx,
        T_truth=T_truth,
        R_truth=R_truth,
        use_ghb=use_ghb,
        width=width
    )

    t0 = time.perf_counter()
    head = solve_darcy_fd_2d_matrix(
        T_field=T_field,
        R_field=R_field,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        dx=dx,
        gh_mask=gh_mask,
        gh_head=gh_head,
        gh_width=gh_width,
        gh_alpha=gh_alpha,
        aq_thickness = aq_thickness,
    )
    t1 = time.perf_counter()
    elapsed = t1 - t0
    ny_loc, nx_loc = head.shape
    print(
        f"FD matrix forward truth: {elapsed:.4f} s for {ny_loc} x {nx_loc}"
    )
    return head, elapsed
