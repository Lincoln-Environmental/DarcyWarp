# SPDX-License-Identifier: AGPL-3.0-only
"""Fast steady-confined K-cycle implementation (production, FP64 only).

This is the production adoption of the kernel-level improvements validated in
the mixed-precision campaign (``MIXED_PRECISION_CAMPAIGN.md``): face-conductance
precompute (no per-call FP64 divisions), block-reduced reductions (no
per-thread atomic serialization), Jacobi-block coarsest level, and CUDA-graph
capture of the fixed cycle — applied to the unchanged two-descent + per-level
Krylov K-cycle structure.

Scope and guards:

* steady confined 2D only — transient storage and unconfined formulations are
  rejected explicitly and keep using the classic backend;
* FP64 models only (``DARCY_FLOAT=float64``); ordinary FP32 production keeps
  the classic path;
* opt-in via ``implementation="fast"`` on the confined K-cycle backend; the
  classic implementation remains the default everywhere.

Numerical contract: same operator (face conductances use the identical
harmonic formula, validated bit-identical against the classic residual
kernel), same convergence criteria (initial-residual-relative tolerance,
head-change safeguards, check cadence, PCG divergence fallback), same info
fields plus ``implementation: "fast"``.  Iterates can differ from classic at
round-off level (different reduction order, Jacobi vs PCG coarsest), never in
the acceptance semantics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

_BLOCK = 256


# ---------------------------------------------------------------------------
# Kernels (explicit FP64; FP32 variants remain in solvers/mixed_fast_kernels.py)
# ---------------------------------------------------------------------------


@wp.kernel
def face_build_f64_kernel(
    T_field: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    ghb_factor: wp.array(ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    nx: int,
    ny: int,
):
    """Face conductances + diagonal in FP64 (identical harmonic formula to the
    classic stencil; GHB conductance added to the diagonal only; identity row
    for isolated cells)."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        Te[j, i] = wp.float64(0.0)
        Tw[j, i] = wp.float64(0.0)
        Tn[j, i] = wp.float64(0.0)
        Ts[j, i] = wp.float64(0.0)
        diag[j, i] = wp.float64(1.0)
        return

    tiny = wp.float64(1.0e-12)
    T_c = wp.float64(T_field[j, i])

    t_e = wp.float64(0.0)
    t_w = wp.float64(0.0)
    t_n = wp.float64(0.0)
    t_s = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float64(T_field[j, i + 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            t_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float64(T_field[j, i - 1])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            t_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float64(T_field[j - 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            t_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float64(T_field[j + 1, i])
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            t_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = t_e + t_w + t_n + t_s + C_gh
    if sum_T < tiny:
        Te[j, i] = wp.float64(0.0)
        Tw[j, i] = wp.float64(0.0)
        Tn[j, i] = wp.float64(0.0)
        Ts[j, i] = wp.float64(0.0)
        diag[j, i] = wp.float64(1.0)
    else:
        Te[j, i] = t_e
        Tw[j, i] = t_w
        Tn[j, i] = t_n
        Ts[j, i] = t_s
        diag[j, i] = sum_T


@wp.kernel
def face_jacobi_f64_kernel(
    b: wp.array(ndim=2),
    x_in: wp.array(ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values: wp.array(ndim=2),
    omega: float,
    nx: int,
    ny: int,
    x_out: wp.array(ndim=2),
):
    """Damped-Jacobi sweep, face-array form.  ALL arithmetic FP64; FMA-only
    except the diagonal divide; no reductions."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        x_out[j, i] = wp.float64(0.0)
        return

    if bc_mask[j, i] != 0:
        x_out[j, i] = wp.float64(bc_values[j, i])
        return

    hC = wp.float64(x_in[j, i])
    ax = wp.float64(diag[j, i]) * hC
    t_e = wp.float64(Te[j, i])
    t_w = wp.float64(Tw[j, i])
    t_n = wp.float64(Tn[j, i])
    t_s = wp.float64(Ts[j, i])
    if t_e > wp.float64(0.0):
        ax = ax - t_e * wp.float64(x_in[j, i + 1])
    if t_w > wp.float64(0.0):
        ax = ax - t_w * wp.float64(x_in[j, i - 1])
    if t_n > wp.float64(0.0):
        ax = ax - t_n * wp.float64(x_in[j - 1, i])
    if t_s > wp.float64(0.0):
        ax = ax - t_s * wp.float64(x_in[j + 1, i])

    r = wp.float64(b[j, i]) - ax
    x_out[j, i] = hC + wp.float64(omega) * r / wp.float64(diag[j, i])


@wp.kernel
def face_residual_f64_kernel(
    x: wp.array(ndim=2),
    b: wp.array(ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    r: wp.array(ndim=2),
    nx: int,
    ny: int,
):
    """Residual r = b - A x, face-array form.  ALL arithmetic FP64; no
    reduction.  Zero on inactive/Dirichlet cells."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        r[j, i] = wp.float64(0.0)
        return

    hC = wp.float64(x[j, i])
    ax = wp.float64(diag[j, i]) * hC
    t_e = wp.float64(Te[j, i])
    t_w = wp.float64(Tw[j, i])
    t_n = wp.float64(Tn[j, i])
    t_s = wp.float64(Ts[j, i])
    if t_e > wp.float64(0.0):
        ax = ax - t_e * wp.float64(x[j, i + 1])
    if t_w > wp.float64(0.0):
        ax = ax - t_w * wp.float64(x[j, i - 1])
    if t_n > wp.float64(0.0):
        ax = ax - t_n * wp.float64(x[j - 1, i])
    if t_s > wp.float64(0.0):
        ax = ax - t_s * wp.float64(x[j + 1, i])

    r[j, i] = wp.float64(b[j, i]) - ax


@wp.kernel
def dot_partials_f64_kernel(
    a: wp.array(ndim=2),
    b: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    partials: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    block_span: int,
):
    """Stage-1 dot product: FP64 load/product/partial, one FP64 atomic add
    per thread into block-indexed buckets (bucket = tid // block_span;
    launched 1-D, block_span is the launch block_dim)."""
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span
    acc = wp.float64(0.0)
    if active[j, i] != 0 and bc_mask[j, i] == 0:
        acc = wp.float64(a[j, i]) * wp.float64(b[j, i])
    wp.atomic_add(partials, block, acc)


@wp.kernel
def applyA_dot_partials_f64_kernel(
    x: wp.array(ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    partials: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    block_span: int,
):
    """Stage-1 of pAp = x . A x without storing A x.  ALL arithmetic FP64;
    one FP64 atomic add per thread into block-indexed buckets (bucket =
    tid // block_span)."""
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span
    acc = wp.float64(0.0)
    if active[j, i] != 0 and bc_mask[j, i] == 0:
        hC = wp.float64(x[j, i])
        ax = wp.float64(diag[j, i]) * hC
        t_e = wp.float64(Te[j, i])
        t_w = wp.float64(Tw[j, i])
        t_n = wp.float64(Tn[j, i])
        t_s = wp.float64(Ts[j, i])
        if t_e > wp.float64(0.0):
            ax = ax - t_e * wp.float64(x[j, i + 1])
        if t_w > wp.float64(0.0):
            ax = ax - t_w * wp.float64(x[j, i - 1])
        if t_n > wp.float64(0.0):
            ax = ax - t_n * wp.float64(x[j - 1, i])
        if t_s > wp.float64(0.0):
            ax = ax - t_s * wp.float64(x[j + 1, i])
        acc = hC * ax
    wp.atomic_add(partials, block, acc)


@wp.kernel
def combine_partials_kernel(
    partials: wp.array(dtype=wp.float64, ndim=1),
    out: wp.array(dtype=wp.float64, ndim=1),
    n_partials: int,
):
    """Stage-2: single-thread combine of per-block partials (also re-zeroes
    the partial buffer for reuse)."""
    t = wp.tid()
    if t > 0:
        return
    acc = wp.float64(0.0)
    for k in range(n_partials):
        acc = acc + partials[k]
        partials[k] = wp.float64(0.0)
    out[0] = acc


@wp.kernel
def combine_partials_max_kernel(
    partials: wp.array(dtype=wp.float64, ndim=1),
    out: wp.array(dtype=wp.float64, ndim=1),
    n_partials: int,
):
    """Stage-2 max combine (also re-zeroes the partial buffer for reuse)."""
    t = wp.tid()
    if t > 0:
        return
    mx = wp.float64(0.0)
    for k in range(n_partials):
        mx = wp.max(mx, partials[k])
        partials[k] = wp.float64(0.0)
    out[0] = mx


@wp.kernel
def face_check_dh_residual_f64_kernel(
    x: wp.array(ndim=2),
    x_prev: wp.array(ndim=2),
    b: wp.array(ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    partials_dh_sq: wp.array(dtype=wp.float64, ndim=1),
    partials_dh_max: wp.array(dtype=wp.float64, ndim=1),
    partials_rTr: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    block_span: int,
):
    """Convergence check, face-array form.  Mirrors
    ``kcycle_check_dh_and_residual_no_storage_kernel`` semantics (x_prev
    updated for ALL cells; dh stats and residual on free cells only) with
    per-block partial reductions instead of per-thread atomics."""
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span

    x_new = wp.float64(x[j, i])
    x_old = wp.float64(x_prev[j, i])
    x_prev[j, i] = x[j, i]

    sq = wp.float64(0.0)
    mx = wp.float64(0.0)
    rr = wp.float64(0.0)
    if active[j, i] != 0 and bc_mask[j, i] == 0:
        dh = x_new - x_old
        sq = dh * dh
        mx = wp.abs(dh)

        hC = x_new
        ax = wp.float64(diag[j, i]) * hC
        t_e = wp.float64(Te[j, i])
        t_w = wp.float64(Tw[j, i])
        t_n = wp.float64(Tn[j, i])
        t_s = wp.float64(Ts[j, i])
        if t_e > wp.float64(0.0):
            ax = ax - t_e * wp.float64(x[j, i + 1])
        if t_w > wp.float64(0.0):
            ax = ax - t_w * wp.float64(x[j, i - 1])
        if t_n > wp.float64(0.0):
            ax = ax - t_n * wp.float64(x[j - 1, i])
        if t_s > wp.float64(0.0):
            ax = ax - t_s * wp.float64(x[j + 1, i])
        rf = wp.float64(b[j, i]) - ax
        rr = rf * rf
    wp.atomic_add(partials_dh_sq, block, sq)
    wp.atomic_max(partials_dh_max, block, mx)
    wp.atomic_add(partials_rTr, block, rr)
