# SPDX-License-Identifier: AGPL-3.0-only
"""Production fast correction kernels for the mixed-precision solver.

Originated in Phase 3 of ``MIXED_PRECISION_CAMPAIGN.md`` and now support the
production confined ``mixed_fast`` path and transient FP32 corrections.

Why these kernels exist (Phase 1 profile evidence, 2000x1000):

1. Production row kernels accumulate every stencil row in FP64 — including
   four FP64 harmonic-mean *divisions* per cell per call — even when the
   hierarchy storage is FP32.  On a consumer GPU (FP64 throughput ≈ 1/64 of
   FP32) this makes the smoother/residual compute-bound, which is why FP32
   storage alone saved only 3–5 % per K-cycle.
2. Every reduction uses per-thread FP64 atomics into a single address
   (level-0 residual: 2.78 ms vs 0.47 ms for the same-traffic smoother).

This module provides, with explicitly auditable arithmetic precision:

* ``_mf3_build_faces_f32/f64`` — one-shot face-conductance arrays
  (Te/Tw/Tn/Ts + diagonal incl. GHB), removing all per-call divisions;
* ``_mf3_jacobi_f32/f64`` — face-array smoother, FMA-only, matching stated
  precision for loads, products, accumulation and stores;
* ``_mf3_residual_f32/f64`` — face-array residual, no atomic norm
  (fixed-work correction cycles never read the residual norm);
* two-stage reductions (per-block partials + single-block combine) for the
  few scalars the K-combination actually consumes on device;
* block-reduced FP64 outer kernels (true residual, correction accumulate)
  for the authoritative mixed-precision outer loop.

Precision contract per kernel is stated in its docstring.  The K-combination
dot products accumulate per-thread partials in FP64 (justified: consumed
on-device by the Krylov alpha; cost measured in the campaign doc).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import warp as wp

EXPERIMENTAL = False

from .face_kernels_f64 import (  # noqa: F401  (single source of truth, re-exported)
    applyA_dot_partials_f64_kernel as _mf3_applyA_dot_partials_f64,
    combine_partials_kernel as _mf3_combine_partials_kernel,
    combine_partials_max_kernel as _mf3_combine_partials_max_kernel,
    dot_partials_f64_kernel as _mf3_dot_partials_f64,
    face_build_f64_kernel as _mf3_build_faces_f64,
    face_jacobi_f64_kernel as _mf3_jacobi_f64,
    face_residual_f64_kernel as _mf3_residual_f64,
)


# ---------------------------------------------------------------------------
# Face-conductance construction (once per level; replaces per-call divisions)
# ---------------------------------------------------------------------------


@wp.kernel
def _mf3_build_faces_f32(
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
    """Face conductances + diagonal in FP32 arithmetic (FP32 output arrays).

    Identical formula to the FP64 variant; every load/product/accumulation
    here is FP32 (auditable: no wp.float64 appears in this kernel).
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        Te[j, i] = wp.float32(0.0)
        Tw[j, i] = wp.float32(0.0)
        Tn[j, i] = wp.float32(0.0)
        Ts[j, i] = wp.float32(0.0)
        diag[j, i] = wp.float32(1.0)
        return

    tiny = wp.float32(1.0e-12)
    T_c = wp.float32(T_field[j, i])

    t_e = wp.float32(0.0)
    t_w = wp.float32(0.0)
    t_n = wp.float32(0.0)
    t_s = wp.float32(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        T_nb = wp.float32(T_field[j, i + 1])
        if T_c > wp.float32(0.0) and T_nb > wp.float32(0.0):
            t_e = wp.float32(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        T_nb = wp.float32(T_field[j, i - 1])
        if T_c > wp.float32(0.0) and T_nb > wp.float32(0.0):
            t_w = wp.float32(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        T_nb = wp.float32(T_field[j - 1, i])
        if T_c > wp.float32(0.0) and T_nb > wp.float32(0.0):
            t_n = wp.float32(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    if j + 1 < ny and active[j + 1, i] != 0:
        T_nb = wp.float32(T_field[j + 1, i])
        if T_c > wp.float32(0.0) and T_nb > wp.float32(0.0):
            t_s = wp.float32(2.0) * T_c * T_nb / (T_c + T_nb + tiny)

    C_gh = wp.float32(0.0)
    if gh_mask[j, i] != 0:
        ghbf = wp.float32(ghb_factor[j, i])
        if ghbf > wp.float32(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = t_e + t_w + t_n + t_s + C_gh
    if sum_T < tiny:
        Te[j, i] = wp.float32(0.0)
        Tw[j, i] = wp.float32(0.0)
        Tn[j, i] = wp.float32(0.0)
        Ts[j, i] = wp.float32(0.0)
        diag[j, i] = wp.float32(1.0)
    else:
        Te[j, i] = t_e
        Tw[j, i] = t_w
        Tn[j, i] = t_n
        Ts[j, i] = t_s
        diag[j, i] = sum_T


# ---------------------------------------------------------------------------
# Face-array smoother (FMA-only; no divisions, no atomics)
# ---------------------------------------------------------------------------


@wp.kernel
def _mf3_jacobi_f32(
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
    """Damped-Jacobi sweep, face-array form.  ALL arithmetic FP32.

    storage FP32 | products FP32 | accumulation FP32 | output FP32 | no
    reductions.  Isolated cells (faces all zero) use diag = 1 from the face
    build, i.e. an identity row exactly like production.
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0:
        x_out[j, i] = wp.float32(0.0)
        return

    if bc_mask[j, i] != 0:
        x_out[j, i] = wp.float32(bc_values[j, i])
        return

    hC = wp.float32(x_in[j, i])
    ax = wp.float32(diag[j, i]) * hC
    t_e = wp.float32(Te[j, i])
    t_w = wp.float32(Tw[j, i])
    t_n = wp.float32(Tn[j, i])
    t_s = wp.float32(Ts[j, i])
    if t_e > wp.float32(0.0):
        ax = ax - t_e * wp.float32(x_in[j, i + 1])
    if t_w > wp.float32(0.0):
        ax = ax - t_w * wp.float32(x_in[j, i - 1])
    if t_n > wp.float32(0.0):
        ax = ax - t_n * wp.float32(x_in[j - 1, i])
    if t_s > wp.float32(0.0):
        ax = ax - t_s * wp.float32(x_in[j + 1, i])

    r = wp.float32(b[j, i]) - ax
    x_out[j, i] = hC + wp.float32(omega) * r / wp.float32(diag[j, i])


# ---------------------------------------------------------------------------
# Face-array residual (no atomic norm: fixed-work cycles never read it)
# ---------------------------------------------------------------------------


@wp.kernel
def _mf3_residual_f32(
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
    """Residual r = b - A x, face-array form.  ALL arithmetic FP32; no
    reduction (the residual norm is never consumed inside fixed-work
    correction cycles).  Zero on inactive/Dirichlet cells."""
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or bc_mask[j, i] != 0:
        r[j, i] = wp.float32(0.0)
        return

    hC = wp.float32(x[j, i])
    ax = wp.float32(diag[j, i]) * hC
    t_e = wp.float32(Te[j, i])
    t_w = wp.float32(Tw[j, i])
    t_n = wp.float32(Tn[j, i])
    t_s = wp.float32(Ts[j, i])
    if t_e > wp.float32(0.0):
        ax = ax - t_e * wp.float32(x[j, i + 1])
    if t_w > wp.float32(0.0):
        ax = ax - t_w * wp.float32(x[j, i - 1])
    if t_n > wp.float32(0.0):
        ax = ax - t_n * wp.float32(x[j - 1, i])
    if t_s > wp.float32(0.0):
        ax = ax - t_s * wp.float32(x[j + 1, i])

    r[j, i] = wp.float32(b[j, i]) - ax


# ---------------------------------------------------------------------------
# Two-stage reductions (per-block partials, then single-block combine)
# ---------------------------------------------------------------------------


@wp.kernel
def _mf3_dot_partials_f32(
    a: wp.array(ndim=2),
    b: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    partials: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    block_span: int,
):
    """Stage-1 dot product: FP32 loads, FP32 product, per-thread FP64
    partial, one FP64 atomic per *block* into partials[block_id].

    Launched 1-D over ny*nx with block_dim threads per block; block_span is
    the launch block_dim.  The FP64 partial/atomic is retained deliberately
    (consumed on-device by the Krylov alpha); cost is ~1 atomic per 256
    threads instead of per thread.
    """
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span
    acc = wp.float64(0.0)
    if active[j, i] != 0 and bc_mask[j, i] == 0:
        acc = wp.float64(wp.float32(a[j, i]) * wp.float32(b[j, i]))
    wp.atomic_add(partials, block, acc)


@wp.kernel
def _mf3_applyA_dot_partials_f32(
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
    """Stage-1 of pAp = x . A x without storing A x: FP32 stencil in FP32,
    per-thread FP64 partial, one FP64 atomic per block.

    Dirichlet rows contribute x*x (identity row), matching production
    apply_A_and_pAp semantics on pinned cells (x is zero there in the
    correction equation, so the contribution vanishes either way).
    """
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span
    acc = wp.float64(0.0)
    if active[j, i] != 0 and bc_mask[j, i] == 0:
        hC = wp.float32(x[j, i])
        ax = wp.float32(diag[j, i]) * hC
        t_e = wp.float32(Te[j, i])
        t_w = wp.float32(Tw[j, i])
        t_n = wp.float32(Tn[j, i])
        t_s = wp.float32(Ts[j, i])
        if t_e > wp.float32(0.0):
            ax = ax - t_e * wp.float32(x[j, i + 1])
        if t_w > wp.float32(0.0):
            ax = ax - t_w * wp.float32(x[j, i - 1])
        if t_n > wp.float32(0.0):
            ax = ax - t_n * wp.float32(x[j - 1, i])
        if t_s > wp.float32(0.0):
            ax = ax - t_s * wp.float32(x[j + 1, i])
        acc = wp.float64(hC * ax)
    wp.atomic_add(partials, block, acc)


# ---------------------------------------------------------------------------
# FP64 outer-loop kernels with block-reduced norms (authoritative state)
# ---------------------------------------------------------------------------


@wp.kernel
def _mf3_outer_residual_f64(
    x: wp.array(dtype=wp.float64, ndim=2),
    b: wp.array(dtype=wp.float64, ndim=2),
    Te: wp.array(ndim=2),
    Tw: wp.array(ndim=2),
    Tn: wp.array(ndim=2),
    Ts: wp.array(ndim=2),
    diag: wp.array(ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    r: wp.array(dtype=wp.float64, ndim=2),
    partials: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    block_span: int,
):
    """Authoritative FP64 true residual r64 = b64 - A(h64) with a
    block-reduced rTr (one FP64 atomic per block).

    Face arrays hold the conductances in FP64 (built once from the model's
    coefficient storage with the identical harmonic formula), so row values
    are bit-identical to the un-optimized FP64 residual kernel.
    """
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span

    acc = wp.float64(0.0)
    if active[j, i] == 0 or bc_mask[j, i] != 0:
        r[j, i] = wp.float64(0.0)
    else:
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
        rf = wp.float64(b[j, i]) - ax
        r[j, i] = rf
        acc = rf * rf
    wp.atomic_add(partials, block, acc)


@wp.kernel
def _mf3_accumulate_f64(
    h64: wp.array(dtype=wp.float64, ndim=2),
    delta32: wp.array(dtype=wp.float32, ndim=2),
    active: wp.array(dtype=wp.int32, ndim=2),
    bc_mask: wp.array(dtype=wp.int32, ndim=2),
    bc_values64: wp.array(dtype=wp.float64, ndim=2),
    partials_sq: wp.array(dtype=wp.float64, ndim=1),
    partials_max: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
    block_span: int,
):
    """h64 += delta32 (FP64 accumulate, FP32 correction load); Dirichlet
    re-pinned; dh_sq / dh_max reduced with one FP64 atomic per block."""
    t = wp.tid()
    n = nx * ny
    if t >= n:
        return
    j = t // nx
    i = t % nx
    block = t // block_span

    sq = wp.float64(0.0)
    mx = wp.float64(0.0)
    if active[j, i] == 0:
        h64[j, i] = wp.float64(0.0)
    elif bc_mask[j, i] != 0:
        h64[j, i] = bc_values64[j, i]
    else:
        dh = wp.float64(delta32[j, i])
        h64[j, i] = h64[j, i] + dh
        sq = dh * dh
        mx = wp.abs(dh)
    wp.atomic_add(partials_sq, block, sq)
    wp.atomic_max(partials_max, block, mx)




__all__ = ["EXPERIMENTAL"]
