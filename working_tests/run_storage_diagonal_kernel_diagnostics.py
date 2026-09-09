#!/usr/bin/env python
"""
Storage-diagonal kernel consistency diagnostics.

For backward Euler the active non-Dirichlet frozen-coefficient transient
equation is::

    (A_diffusion + S) h = rhs + S * head_prev        rhs_eff = rhs + S * head_prev

where ``S = storage_diag``.  For a storage-only system (diffusion ~0) the exact
solution is ``h = head_prev`` and the residual at ``head_prev`` is exactly zero,
so the system is *more* diagonally dominant and must be *easier*, never harder.
If the storage-dominated solve diverges, a kernel is handling ``storage_diag``
inconsistently.

This script tests every operator component in isolation on tiny grids:

  * apply_A_kernel            -> Ah == (A_diff + diag(S)) h
  * apply_A_and_pAp_kernel    -> Ap == (A_diff + diag(S)) p        (PCG matvec)
  * init_pcg_with_A_kernel    -> r = rhs_eff - (A_diff + diag(S)) x  (PCG init)
  * compute_residual_kernel   -> r = rhs_eff - (A_diff + diag(S)) x
  * build_diag_preconditioner -> M_inv == 1 / (sum_face_C + S)
  * jacobi_applyA_fused_kernel-> one damped Jacobi update == reference
  * full-solver storage-only   -> solve returns ~head_prev

Run directly::

    python working_tests/run_storage_diagonal_kernel_diagnostics.py
    python working_tests/run_storage_diagonal_kernel_diagnostics.py --device cpu -v

Deterministic, no plotting, no seaborn, no lambdas, pathlib, ``__main__`` guard.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))
os.environ.setdefault("DARCY_FLOAT", "float64")


NA = "n/a"


def warp_available() -> bool:
    """Return True if the ``warp`` kernel package can be imported."""
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


# ---------------------------------------------------------------------------
# NumPy reference for (A_diffusion + diag(S)) and one Jacobi update
# ---------------------------------------------------------------------------
def _face_cond_x(T: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Harmonic-mean x-face conductance ``ce[:, j]`` between ``j`` and ``j+1``.

    :param T: transmissivity field (ny, nx).
    :param active: active-cell mask (ny, nx).
    :return: conductance field; last column and inactive faces are 0.
    """
    ce = np.zeros_like(T, dtype=np.float64)
    left = T[:, :-1]
    right = T[:, 1:]
    denom = left + right
    valid = (active[:, :-1] != 0) & (active[:, 1:] != 0) & (denom > 1.0e-12) & (left > 0) & (right > 0)
    vals = np.zeros_like(denom)
    vals[valid] = 2.0 * left[valid] * right[valid] / denom[valid]
    ce[:, :-1] = vals
    return ce


def _face_cond_y(T: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Harmonic-mean y-face conductance ``cn[i, :]`` between ``i`` and ``i+1``.

    :param T: transmissivity field (ny, nx).
    :param active: active-cell mask (ny, nx).
    :return: conductance field; last row and inactive faces are 0.
    """
    cn = np.zeros_like(T, dtype=np.float64)
    top = T[:-1, :]
    bottom = T[1:, :]
    denom = top + bottom
    valid = (active[:-1, :] != 0) & (active[1:, :] != 0) & (denom > 1.0e-12) & (top > 0) & (bottom > 0)
    vals = np.zeros_like(denom)
    vals[valid] = 2.0 * top[valid] * bottom[valid] / denom[valid]
    cn[:-1, :] = vals
    return cn


def ref_apply_A(T: np.ndarray, h: np.ndarray, S: np.ndarray, active: np.ndarray, bc: np.ndarray) -> np.ndarray:
    """Reference ``(A_diff + diag(S)) h`` matching the kernel stencil.

    :param T: transmissivity field.
    :param h: head field at which to apply the operator.
    :param S: storage diagonal field.
    :param active: active-cell mask.
    :param bc: Dirichlet-cell mask.
    :return: ``Ah`` field (Ah == h on inactive/Dirichlet cells, matching kernels).
    """
    h = np.asarray(h, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    Ah = np.zeros_like(h)
    ce = _face_cond_x(T, active)
    cn = _face_cond_y(T, active)
    outflow = np.zeros_like(h)  # sum_j C_ij (h_i - h_j)
    outflow[:, :-1] += ce[:, :-1] * (h[:, :-1] - h[:, 1:])
    outflow[:, 1:] += ce[:, :-1] * (h[:, 1:] - h[:, :-1])
    outflow[:-1, :] += cn[:-1, :] * (h[:-1, :] - h[1:, :])
    outflow[1:, :] += cn[:-1, :] * (h[1:, :] - h[:-1, :])
    Ah = outflow + S * h
    free = (active != 0) & (bc == 0)
    Ah[~free] = h[~free]  # kernels set Ah = h on inactive/Dirichlet cells
    return Ah


def ref_diag(T: np.ndarray, S: np.ndarray, active: np.ndarray, bc: np.ndarray) -> np.ndarray:
    """Reference diagonal ``sum_face_C + S`` per cell.

    :param T: transmissivity field.
    :param S: storage diagonal field.
    :param active: active-cell mask.
    :param bc: Dirichlet-cell mask.
    :return: diagonal field (1.0 on inactive/Dirichlet cells, matching the kernel).
    """
    S = np.asarray(S, dtype=np.float64)
    ce = _face_cond_x(T, active)
    cn = _face_cond_y(T, active)
    # Sum each face conductance into both cells it touches.
    contributions = np.zeros_like(T, dtype=np.float64)
    contributions[:, :-1] += ce[:, :-1]
    contributions[:, 1:] += ce[:, :-1]
    contributions[:-1, :] += cn[:-1, :]
    contributions[1:, :] += cn[:-1, :]
    diag = contributions + S
    free = (active != 0) & (bc == 0)
    diag[~free] = 1.0
    return diag


def ref_solve(T: np.ndarray, b: np.ndarray, S: np.ndarray, active: np.ndarray, bc: np.ndarray, bcv: np.ndarray) -> np.ndarray:
    """Solve the tiny reference system exactly with NumPy.

    :param T: transmissivity field.
    :param b: effective RHS field.
    :param S: storage diagonal field.
    :param active: active-cell mask.
    :param bc: Dirichlet-cell mask.
    :param bcv: Dirichlet values.
    :return: dense direct-solve head field.
    """
    T = np.asarray(T, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)
    active = np.asarray(active, dtype=np.int32)
    bc = np.asarray(bc, dtype=np.int32)
    bcv = np.asarray(bcv, dtype=np.float64)
    ny, nx = T.shape
    n = ny * nx
    matrix = np.zeros((n, n), dtype=np.float64)
    rhs = np.zeros(n, dtype=np.float64)
    ce = _face_cond_x(T, active)
    cn = _face_cond_y(T, active)

    for j in range(ny):
        for i in range(nx):
            row = j * nx + i
            if active[j, i] == 0:
                matrix[row, row] = 1.0
                rhs[row] = 0.0
                continue
            if bc[j, i] != 0:
                matrix[row, row] = 1.0
                rhs[row] = bcv[j, i]
                continue

            diag = float(S[j, i])
            if i + 1 < nx:
                c = float(ce[j, i])
                if c > 0.0:
                    diag += c
                    matrix[row, row + 1] -= c
            if i - 1 >= 0:
                c = float(ce[j, i - 1])
                if c > 0.0:
                    diag += c
                    matrix[row, row - 1] -= c
            if j + 1 < ny:
                c = float(cn[j, i])
                if c > 0.0:
                    diag += c
                    matrix[row, row + nx] -= c
            if j - 1 >= 0:
                c = float(cn[j - 1, i])
                if c > 0.0:
                    diag += c
                    matrix[row, row - nx] -= c

            matrix[row, row] = diag if diag > 1.0e-12 else 1.0
            rhs[row] = b[j, i] if diag > 1.0e-12 else 0.0

    return np.linalg.solve(matrix, rhs).reshape(ny, nx)


# ---------------------------------------------------------------------------
# Tiny-case factory
# ---------------------------------------------------------------------------
class StorageCase:
    """Holds the numpy fields for one storage/diffusion configuration."""

    def __init__(self, name: str, ny: int, nx: int, T_value: float, S_value: float, spatially_variable: bool = False):
        """Build a deterministic tiny case.

        :param name: case label.
        :param ny: rows.
        :param nx: columns.
        :param T_value: uniform transmissivity (use ~0 for storage-only).
        :param S_value: uniform storage diagonal magnitude.
        :param spatially_variable: if True, use a nonuniform S and head_prev.
        """
        self.name = name
        self.ny = ny
        self.nx = nx
        self.T = np.full((ny, nx), float(T_value), dtype=np.float64)
        if spatially_variable:
            S = np.empty((ny, nx), dtype=np.float64)
            head_prev = np.empty((ny, nx), dtype=np.float64)
            for j in range(ny):
                for i in range(nx):
                    S[j, i] = float(S_value) * (1.0 + 0.1 * (i + j))
                    head_prev[j, i] = 1.0 + 0.3 * i + 0.2 * j
        else:
            S = np.full((ny, nx), float(S_value), dtype=np.float64)
            head_prev = np.full((ny, nx), 3.0, dtype=np.float64)
            head_prev[:, :] = 1.0 + 0.3 * np.arange(nx)[None, :] + 0.2 * np.arange(ny)[:, None]
        self.S = S
        self.head_prev = head_prev
        self.active = np.ones((ny, nx), dtype=np.int32)
        self.bc = np.zeros((ny, nx), dtype=np.int32)
        self.bcv = np.zeros((ny, nx), dtype=np.float64)
        # rhs = 0 -> rhs_eff = S * head_prev (pure transient storage RHS)
        self.rhs = np.zeros((ny, nx), dtype=np.float64)
        self.rhs_eff = self.rhs + S * head_prev
        self.diff_diag = ref_diag(self.T, np.zeros_like(S), self.active, self.bc)


def build_cases() -> list[StorageCase]:
    """Return the six storage/diffusion ratio cases from the spec."""
    return [
        StorageCase("storage_only_uniform", 6, 8, 1.0e-8, 5.0),
        StorageCase("storage_only_spatially_variable", 6, 8, 1.0e-8, 5.0, spatially_variable=True),
        StorageCase("storage_plus_tiny_diffusion", 6, 8, 1.0e-3, 5.0),
        StorageCase("storage_plus_normal_diffusion", 6, 8, 10.0, 5.0),
        StorageCase("large_storage_small_diffusion", 6, 8, 1.0, 5000.0),
        StorageCase("small_storage_normal_diffusion", 6, 8, 10.0, 1.0e-3),
    ]


# ---------------------------------------------------------------------------
# Kernel-launch helpers
# ---------------------------------------------------------------------------
def _make_arrays(case: StorageCase, device: str):
    """Upload the case fields to warp arrays and return them in a dict.

    :param case: storage case.
    :param device: warp device string.
    :return: dict of warp arrays.
    """
    import warp as wp
    from DARCY_WARP_PACKAGE.config import NP_FLOAT, WP_FLOAT

    def wpa(arr):
        return wp.array(np.ascontiguousarray(np.asarray(arr, dtype=NP_FLOAT)), dtype=WP_FLOAT, device=device)

    def wpi(arr):
        return wp.array(np.ascontiguousarray(np.asarray(arr, dtype=np.int32)), dtype=wp.int32, device=device)

    return {
        "T": wpa(case.T),
        "active": wpi(case.active),
        "bc": wpi(case.bc),
        "gh_mask": wpi(np.zeros((case.ny, case.nx), dtype=np.int32)),
        "ghb_factor": wpa(np.zeros((case.ny, case.nx), dtype=np.float64)),
        "storage": wpa(case.S),
        "rhs_eff": wpa(case.rhs_eff),
        "bcv": wpa(case.bcv),
        "h": wpa(case.head_prev),
        "zero_field": wpa(np.zeros((case.ny, case.nx), dtype=np.float64)),
        "pAp_buf": wp.array(np.zeros(1, dtype=np.float64), dtype=wp.float64, device=device),
        "rho_buf": wp.array(np.zeros(1, dtype=np.float64), dtype=wp.float64, device=device),
        "rTr_buf": wp.array(np.zeros(1, dtype=np.float64), dtype=wp.float64, device=device),
        "device": device,
        "wp_float": WP_FLOAT,
    }


def _err(a: np.ndarray, b: np.ndarray, case: StorageCase) -> tuple[float, float, float]:
    """Return (max_abs_error, rms_error, rel_error) over free cells.

    :param a: kernel output.
    :param b: reference.
    :param case: storage case (for the free-cell mask).
    """
    free = (case.active != 0) & (case.bc == 0)
    diff = (np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))[free]
    if diff.size == 0:
        return float("nan"), float("nan"), float("nan")
    mx = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff * diff)))
    scale = float(np.max(np.abs(np.asarray(b, dtype=np.float64)[free]))) + 1.0e-30
    return mx, rms, mx / scale


# ---------------------------------------------------------------------------
# Per-component kernel tests
# ---------------------------------------------------------------------------
def test_apply_A(case: StorageCase, device: str) -> tuple[float, float, float, str]:
    """Test ``apply_A_kernel``: Ah == ref_apply_A(head_prev).

    :return: (max_abs, rms, rel, diagnosis).
    """
    import warp as wp
    from DARCY_WARP_PACKAGE.warped_darcy import apply_A_kernel

    a = _make_arrays(case, device)
    Ah_wp = a["zero_field"]
    wp.launch(
        apply_A_kernel, dim=(case.ny, case.nx),
        inputs=[a["T"], a["active"], a["bc"], a["gh_mask"], a["ghb_factor"], a["storage"], a["h"], Ah_wp, case.nx, case.ny],
        device=device,
    )
    Ah = Ah_wp.numpy()
    ref = ref_apply_A(case.T, case.head_prev, case.S, case.active, case.bc)
    mx, rms, rel = _err(Ah, ref, case)
    diag = "PASS" if rel < 1.0e-3 else "FAIL: apply_A storage mismatch"
    return mx, rms, rel, diag


def test_apply_A_and_pAp(case: StorageCase, device: str) -> tuple[float, float, float, str]:
    """Test ``apply_A_and_pAp_kernel`` (PCG matvec): Ap == ref_apply_A(head_prev).

    :return: (max_abs, rms, rel, diagnosis).
    """
    import warp as wp
    from DARCY_WARP_PACKAGE.warped_darcy import apply_A_and_pAp_kernel

    a = _make_arrays(case, device)
    Ap_wp = a["zero_field"]
    wp.launch(
        apply_A_and_pAp_kernel, dim=(case.ny, case.nx),
        inputs=[a["T"], a["active"], a["bc"], a["gh_mask"], a["ghb_factor"], a["storage"], a["h"], Ap_wp, a["pAp_buf"], case.nx, case.ny],
        device=device,
    )
    Ap = Ap_wp.numpy()
    ref = ref_apply_A(case.T, case.head_prev, case.S, case.active, case.bc)
    mx, rms, rel = _err(Ap, ref, case)
    diag = "PASS" if rel < 1.0e-3 else "FAIL: apply_A_and_pAp (PCG matvec) omits storage"
    return mx, rms, rel, diag


def test_compute_residual(case: StorageCase, device: str) -> tuple[float, float, float, str]:
    """Test ``compute_residual_kernel``: r == rhs_eff - ref_apply_A(head_prev).

    For storage-only head_prev, this is ~0.

    :return: (max_abs, rms, rel, diagnosis).
    """
    import warp as wp
    from DARCY_WARP_PACKAGE.warped_darcy import compute_residual_kernel

    a = _make_arrays(case, device)
    r_wp = a["zero_field"]
    wp.launch(
        compute_residual_kernel, dim=(case.ny, case.nx),
        inputs=[a["h"], a["rhs_eff"], a["T"], a["active"], a["bc"], a["gh_mask"], a["ghb_factor"], a["storage"], r_wp, a["rTr_buf"], case.nx, case.ny],
        device=device,
    )
    r = r_wp.numpy()
    ref_r = case.rhs_eff - ref_apply_A(case.T, case.head_prev, case.S, case.active, case.bc)
    mx, rms, rel = _err(r, ref_r, case)
    diag = "PASS" if mx < 1.0e-6 else "FAIL: residual storage mismatch"
    return mx, rms, rel, diag


def test_init_pcg_with_A(case: StorageCase, device: str) -> tuple[float, float, float, str]:
    """Test ``init_pcg_with_A_kernel``: r = rhs_eff - (A_diff+S) head_prev ~ 0.

    :return: (max_abs, rms, rel, diagnosis).
    """
    import warp as wp
    from DARCY_WARP_PACKAGE.warped_darcy import init_pcg_with_A_kernel
    from DARCY_WARP_PACKAGE.warped_darcy import build_diag_preconditioner

    a = _make_arrays(case, device)
    Minv_np = build_diag_preconditioner(case.T, case.active, case.bc, storage_diag=case.S)
    from DARCY_WARP_PACKAGE.config import NP_FLOAT, WP_FLOAT
    Minv_wp = wp.array(np.ascontiguousarray(Minv_np.astype(NP_FLOAT)), dtype=WP_FLOAT, device=device)
    Ap_wp = wp.array(np.zeros((case.ny, case.nx), dtype=NP_FLOAT), dtype=WP_FLOAT, device=device)
    r_wp = wp.array(np.zeros((case.ny, case.nx), dtype=NP_FLOAT), dtype=WP_FLOAT, device=device)
    z_wp = wp.array(np.zeros((case.ny, case.nx), dtype=NP_FLOAT), dtype=WP_FLOAT, device=device)
    p_wp = wp.array(np.zeros((case.ny, case.nx), dtype=NP_FLOAT), dtype=WP_FLOAT, device=device)
    wp.launch(
        init_pcg_with_A_kernel, dim=(case.ny, case.nx),
        inputs=[a["h"], a["rhs_eff"], a["T"], a["active"], a["bc"], a["gh_mask"], a["ghb_factor"], a["storage"],
                Minv_wp, Ap_wp, r_wp, z_wp, p_wp, a["rho_buf"], a["rTr_buf"], case.nx, case.ny],
        device=device,
    )
    r = r_wp.numpy()
    ref_r = case.rhs_eff - ref_apply_A(case.T, case.head_prev, case.S, case.active, case.bc)
    mx, rms, rel = _err(r, ref_r, case)
    diag = "PASS" if mx < 1.0e-6 else "FAIL: init_pcg_with_A (PCG init) omits storage"
    return mx, rms, rel, diag


def test_preconditioner(case: StorageCase, device: str) -> tuple[float, float, float, str]:
    """Test ``build_diag_preconditioner`` host: M_inv == 1/(sum_face_C + S).

    :return: (max_abs, rms, rel, diagnosis).
    """
    from DARCY_WARP_PACKAGE.warped_darcy import build_diag_preconditioner

    Minv = build_diag_preconditioner(case.T, case.active, case.bc, storage_diag=case.S)
    diag_total = ref_diag(case.T, case.S, case.active, case.bc)
    ref = np.where(diag_total > 1.0e-12, 1.0 / diag_total, 1.0)
    mx, rms, rel = _err(Minv, ref, case)
    diag = "PASS" if rel < 1.0e-4 else "FAIL: preconditioner omits/doubles storage"
    return mx, rms, rel, diag


def test_fused_smoother(case: StorageCase, device: str) -> tuple[float, float, float, str]:
    """Test one ``jacobi_applyA_fused_kernel`` update from zero vs the reference.

    Reference: x_new = 0 + omega * M_inv * (rhs_eff - A*0) = omega * M_inv * rhs_eff.

    :return: (max_abs, rms, rel, diagnosis).
    """
    import warp as wp
    from DARCY_WARP_PACKAGE.warped_darcy import jacobi_applyA_fused_kernel, build_diag_preconditioner
    from DARCY_WARP_PACKAGE.config import NP_FLOAT, WP_FLOAT

    a = _make_arrays(case, device)
    Minv_np = build_diag_preconditioner(case.T, case.active, case.bc, storage_diag=case.S)
    Minv_wp = wp.array(np.ascontiguousarray(Minv_np.astype(NP_FLOAT)), dtype=WP_FLOAT, device=device)
    x_in = a["zero_field"]  # start from 0
    x_out = wp.array(np.zeros((case.ny, case.nx), dtype=NP_FLOAT), dtype=WP_FLOAT, device=device)
    omega = 0.8
    wp.launch(
        jacobi_applyA_fused_kernel, dim=(case.ny, case.nx),
        inputs=[a["T"], a["active"], a["bc"], a["gh_mask"], a["ghb_factor"], a["storage"],
                a["rhs_eff"], x_in, Minv_wp, a["bcv"], float(omega), case.nx, case.ny, x_out],
        device=device,
    )
    out = x_out.numpy()
    diag_total = ref_diag(case.T, case.S, case.active, case.bc)
    Minv_ref = np.where(diag_total > 1.0e-12, 1.0 / diag_total, 1.0)
    ref = omega * Minv_ref * case.rhs_eff
    mx, rms, rel = _err(out, ref, case)
    diag = "PASS" if rel < 1.0e-3 else "FAIL: fused smoother storage mismatch"
    return mx, rms, rel, diag


def test_full_solver_storage_only(case: StorageCase, device: str, max_levels: int) -> tuple[float, float, str, bool]:
    """Run the full solver and compare it to a dense tiny-grid reference solve.

    :param max_levels: multigrid level cap.
    :return: (max_abs_error, rms_error, diagnosis, converged).
    """
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=case.nx, ny=case.ny, dx=1.0, device=device, use_ghb=False,
        solver_type="kcycle", diag_preconditioner_backend="host",
    )
    solver.build_from_fields(
        T_field=case.T, R_field=case.rhs, active=case.active, bc_mask=case.bc, bc_values=case.bcv,
    )
    initial = np.zeros((case.ny, case.nx), dtype=np.float64)
    try:
        h, info = solver.solve(
            formulation="confined", transient=True, storage_coeff=case.S, dt=1.0,
            head_prev=case.head_prev, initial_head=initial,
            max_cycles=60, max_levels=max_levels, min_coarse_cells=1, check_every_no=1, return_info=True,
        )
    except (FloatingPointError, ValueError, RuntimeError, OverflowError) as exc:
        solver.close()
        return float("inf"), float("inf"), f"EXC: {type(exc).__name__}", False
    converged = bool(info.get("converged", False))
    reference = ref_solve(
        T=case.T,
        b=case.rhs_eff,
        S=case.S,
        active=case.active,
        bc=case.bc,
        bcv=case.bcv,
    )
    free = (case.active != 0) & (case.bc == 0)
    diff = (np.asarray(h, dtype=np.float64) - reference)[free]
    mx = float(np.max(np.abs(diff))) if diff.size else float("nan")
    rms = float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan")
    solver.close()
    ok = converged and mx < 1.0e-3
    diag = "PASS" if ok else ("FAIL: stale storage array / PCG operator" if not math.isfinite(mx) or mx > 1.0 else "FAIL: full solve storage mismatch")
    return mx, rms, diag, converged


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
COMPONENT_TESTS = [
    ("apply_A", test_apply_A),
    ("compute_residual", test_compute_residual),
    ("apply_A_and_pAp (PCG matvec)", test_apply_A_and_pAp),
    ("init_pcg_with_A (PCG init)", test_init_pcg_with_A),
    ("diag_preconditioner", test_preconditioner),
    ("jacobi_applyA_fused (1 update)", test_fused_smoother),
]


def _fmt(v: float, spec: str) -> str:
    if v is None:
        return NA
    if isinstance(v, float) and (not math.isfinite(v)):
        return "inf" if v > 0 else "-inf"
    return format(float(v), spec)


def _join_row(row, widths) -> str:
    """Join a table row, left-justifying the first column, right-justifying the rest.

    :param row: sequence of cell strings.
    :param widths: sequence of column widths.
    :return: the formatted row string.
    """
    cells = []
    for idx, (cell, width) in enumerate(zip(row, widths)):
        text = str(cell)
        if len(text) > width:
            text = text[: max(0, width - 1)] + "…"
        cells.append(text.ljust(width) if idx == 0 else text.rjust(width))
    return "  ".join(cells)


def run_all(device: str, verbose: bool) -> int:
    """Run every component test on every case and print the diagnostic table.

    :param device: warp device string.
    :param verbose: print per-case detail.
    :return: number of failing rows.
    """
    cases = build_cases()
    print("Storage-diagonal kernel consistency diagnostics")
    print(f"device={device}  cases={len(cases)}  (deterministic; no plotting)\n")

    header_cols = (
        "case", "component", "S_min", "S_max", "diff_min", "diff_max",
        "S/diff_max", "max_abs_err", "rms_err", "rel_err", "verdict",
    )
    widths = (26, 28, 9, 9, 9, 9, 10, 12, 12, 10, 6)
    header = "  ".join(c.rjust(w) for c, w in zip(header_cols, widths))
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    n_fail = 0
    first_fail_diag = ""
    for case in cases:
        s_min, s_max = float(case.S.min()), float(case.S.max())
        d_min, d_max = float(case.diff_diag.min()), float(case.diff_diag.max())
        ratio = s_max / (d_max + 1.0e-30)
        for comp_name, fn in COMPONENT_TESTS:
            try:
                mx, rms, rel, diag = fn(case, device)
            except Exception as exc:  # pragma: no cover
                mx, rms, rel, diag = float("nan"), float("nan"), float("nan"), f"EXC: {type(exc).__name__}"
            verdict = diag.split(":")[0]
            if verdict != "PASS":
                n_fail += 1
                if not first_fail_diag:
                    first_fail_diag = diag
            row = (
                case.name[:26], comp_name[:28],
                _fmt(s_min, ".2e"), _fmt(s_max, ".2e"),
                _fmt(d_min, ".2e"), _fmt(d_max, ".2e"),
                _fmt(ratio, ".2e"),
                _fmt(mx, ".3e"), _fmt(rms, ".3e"), _fmt(rel, ".3e"),
                verdict,
            )
            print(_join_row(row, widths))
        # full-solver storage-only, single-level and multi-level
        for ml in (1, 3):
            mx, rms, diag, conv = test_full_solver_storage_only(case, device, ml)
            verdict = diag.split(":")[0] if diag.startswith(("PASS", "FAIL")) else "FAIL"
            if verdict != "PASS":
                n_fail += 1
                if not first_fail_diag:
                    first_fail_diag = diag
            row = (
                case.name[:26], f"full_solve (ml={ml})"[:28],
                _fmt(s_min, ".2e"), _fmt(s_max, ".2e"),
                _fmt(d_min, ".2e"), _fmt(d_max, ".2e"),
                _fmt(ratio, ".2e"),
                _fmt(mx, ".3e"), _fmt(rms, ".3e"), NA,
                verdict,
            )
            print(_join_row(row, widths))
        print(sep)

    print("\n=== DIAGNOSIS ===")
    if n_fail == 0:
        print("PASS: every operator component applies storage_diag consistently")
        print("      (apply_A, PCG matvec, PCG init, residual, preconditioner, smoother, full solve)")
    else:
        print(f"FAIL: {n_fail} component/case rows failed.")
        print(f"      First failing diagnosis: {first_fail_diag or 'UNKNOWN'}")
        print("      -> patch the named kernel/orchestration path before any solver-architecture change.")
    return n_fail


def parse_args(argv: list[str]) -> argparse.Namespace:
    """Parse command-line arguments.

    :param argv: argument list.
    :return: parsed namespace.
    """
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--device", default="cpu")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    :param argv: optional argument list.
    :return: 0 if all components pass, 1 otherwise.
    """
    args = parse_args(sys.argv[1:] if argv is None else argv)
    if not warp_available():
        print("ERROR: warp is not available.", file=sys.stderr)
        return 1
    n_fail = run_all(args.device, args.verbose)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
