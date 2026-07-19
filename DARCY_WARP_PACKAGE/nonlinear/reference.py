# SPDX-License-Identifier: AGPL-3.0-only
"""Host NumPy / SciPy reference for the 2D nonlinear operator.

This module is *test infrastructure*: it provides independent ground-truth
implementations of the same equation the device kernels evaluate, plus an
independent sparse-matrix assembly of the 5-point operator.  It is used to:

* prove the device operator matches a host assembly of the same maths
  (no silent host/device drift);
* prove the authoritative residual agrees with ``physics.storage_2d`` exact
  storage and with the production ``mf6_convertible_secant_sy`` Picard path;
* cross-check confined linear consistency against an independent sparse system.

Nothing here is on the hot solver path and no solver backend depends on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.sparse as sp

from DARCY_WARP_PACKAGE.physics import storage_2d


_TINY = 1.0e-12


def _flow_transmissivity(head: np.ndarray, ctx: Any) -> np.ndarray:
    """Host ``T = K * flow_sat(head)`` mirroring the device kernel."""
    h = np.asarray(head, dtype=np.float64)
    b = np.asarray(ctx.flow.zbot, dtype=np.float64)
    K = np.asarray(ctx.flow.K, dtype=np.float64)
    ms = float(ctx.flow.min_sat)
    sat = np.maximum(h - b, ms)
    if ctx.flow.ztop is not None:
        cap = np.maximum(np.asarray(ctx.flow.ztop, dtype=np.float64) - b, ms)
        sat = np.minimum(sat, cap)
    return K * sat


def _face_conductance_east_west(T: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Conductance of vertical faces between columns i and i+1, shape (ny, nx-1)."""
    Tc = T[:, :-1]
    Tn = T[:, 1:]
    act_c = active[:, :-1] != 0
    act_n = active[:, 1:] != 0
    C = np.zeros_like(Tc, dtype=np.float64)
    ok = act_c & act_n & (Tc > 0.0) & (Tn > 0.0)
    C[ok] = 2.0 * Tc[ok] * Tn[ok] / (Tc[ok] + Tn[ok] + _TINY)
    return C


def _face_conductance_north_south(T: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Conductance of horizontal faces between rows j and j+1, shape (ny-1, nx)."""
    Tc = T[:-1, :]
    Tn = T[1:, :]
    act_c = active[:-1, :] != 0
    act_n = active[1:, :] != 0
    C = np.zeros_like(Tc, dtype=np.float64)
    ok = act_c & act_n & (Tc > 0.0) & (Tn > 0.0)
    C[ok] = 2.0 * Tc[ok] * Tn[ok] / (Tc[ok] + Tn[ok] + _TINY)
    return C


def flow_operator_applied(head: np.ndarray, ctx: Any) -> np.ndarray:
    """Host ``flow_A(h) h`` (5-point + GHB diagonal, NO storage, NO sources)."""
    h = np.asarray(head, dtype=np.float64)
    T = _flow_transmissivity(h, ctx)
    active = np.asarray(ctx.boundaries.active, dtype=np.int32)

    Ah = np.zeros_like(h, dtype=np.float64)
    C_ew = _face_conductance_east_west(T, active)
    Ah[:, :-1] += C_ew * (h[:, :-1] - h[:, 1:])
    Ah[:, 1:] += C_ew * (h[:, 1:] - h[:, :-1])
    C_ns = _face_conductance_north_south(T, active)
    Ah[:-1, :] += C_ns * (h[:-1, :] - h[1:, :])
    Ah[1:, :] += C_ns * (h[1:, :] - h[:-1, :])

    gh_mask = np.asarray(ctx.boundaries.ghb_mask, dtype=np.int32) != 0
    ghb_factor = np.asarray(ctx.boundaries.ghb_factor, dtype=np.float64)
    C_gh = np.zeros_like(h, dtype=np.float64)
    ok = gh_mask & np.isfinite(ghb_factor) & (ghb_factor > 0.0)
    C_gh[ok] = T[ok] * ghb_factor[ok]
    Ah += C_gh * h

    # Mirror the device isolated-cell branch: if a free cell has no conductance
    # it degenerates to an identity row.
    sum_T = (
        np.pad(C_ew, ((0, 0), (0, 1)), constant_values=0.0)
        + np.pad(C_ew, ((0, 0), (1, 0)), constant_values=0.0)
        + np.pad(C_ns, ((0, 1), (0, 0)), constant_values=0.0)
        + np.pad(C_ns, ((1, 0), (0, 0)), constant_values=0.0)
        + C_gh
    )
    isolated = (sum_T < _TINY) & (active != 0)
    Ah[isolated] = h[isolated]
    return Ah


def exact_storage_terms_host(head: np.ndarray, ctx: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Host exact storage flux ``[L^3/T]`` from ``physics.storage_2d``.

    Returns ``(total, sy_term, ss_term)`` each already scaled by cell area.
    This is the authoritative host source the device
    :meth:`NonlinearOperator2D.exact_storage_terms` is validated against.
    """
    h = np.asarray(head, dtype=np.float64)
    sto = ctx.storage
    if not sto.transient:
        z = np.zeros_like(h, dtype=np.float64)
        return z, z, z
    hp = np.asarray(sto.head_prev, dtype=np.float64)
    total, sy_term, ss_term = storage_2d.exact_unconfined_storage_terms(
        head_new=h,
        head_old=hp,
        bottom=np.asarray(ctx.flow.zbot, dtype=np.float64),
        top=(
            np.asarray(ctx.flow.ztop, dtype=np.float64)
            if ctx.flow.ztop is not None
            else np.asarray(ctx.flow.zbot, dtype=np.float64)
        ),
        specific_yield=float(sto.sy),
        specific_storage=float(sto.ss),
        dt=float(sto.dt),
    )
    area = float(ctx.grid.area)
    total = total * area
    sy_term = sy_term * area
    ss_term = ss_term * area
    # Mirror the device kernel: non-free cells carry zero storage.
    free = ctx.free_mask
    total[~free] = 0.0
    sy_term[~free] = 0.0
    ss_term[~free] = 0.0
    return total, sy_term, ss_term


def nonlinear_residual_host(head: np.ndarray, ctx: Any) -> np.ndarray:
    """Host ``F(h) = A(h) h - b(h)`` (groundwater-balance form, exact storage).

    ``F = flow_A(h) + storage_exact(h) - sources(h)`` on free cells, zero on
    inactive / Dirichlet rows.  Independent of the device kernel; used to prove
    device == host.
    """
    h = np.asarray(head, dtype=np.float64)
    free = ctx.free_mask
    active = np.asarray(ctx.boundaries.active, dtype=np.int32)

    flow_Ah = flow_operator_applied(h, ctx)
    total_store, _sy, _ss = exact_storage_terms_host(h, ctx)

    R = np.asarray(ctx.sources.R_field, dtype=np.float64)
    area = float(ctx.grid.area)
    recharge = R * area

    gh_mask = np.asarray(ctx.boundaries.ghb_mask, dtype=np.int32) != 0
    ghb_factor = np.asarray(ctx.boundaries.ghb_factor, dtype=np.float64)
    T = _flow_transmissivity(h, ctx)
    gh_head = np.asarray(ctx.boundaries.ghb_external_head, dtype=np.float64)
    C_gh = np.zeros_like(h, dtype=np.float64)
    ok = gh_mask & np.isfinite(ghb_factor) & (ghb_factor > 0.0)
    C_gh[ok] = T[ok] * ghb_factor[ok]
    ghb_source = C_gh * gh_head

    F = flow_Ah + total_store - recharge - ghb_source
    out = np.zeros_like(h, dtype=np.float64)
    out[free] = F[free]
    return out


def frozen_picard_operator_host(head: np.ndarray, ctx: Any) -> dict[str, np.ndarray]:
    """Host replication of ``picard_unconfined._storage_from_picard_head``.

    Returns T(h), secant sy_coeff, secant ss_coeff (min_sat-floored Ss sat) and
    storage_diag, matching the production linearisation exactly.  Used to prove
    the device ``frozen_picard_operator`` reproduces the Picard coefficients.
    """
    h = np.asarray(head, dtype=np.float64)
    b = np.asarray(ctx.flow.zbot, dtype=np.float64)
    K = np.asarray(ctx.flow.K, dtype=np.float64)
    ms = float(ctx.flow.min_sat)
    shape = (ctx.ny, ctx.nx)
    free = ctx.free_mask

    sat_flow = np.maximum(h - b, ms)
    if ctx.flow.ztop is not None:
        full_thickness = np.maximum(np.asarray(ctx.flow.ztop, dtype=np.float64) - b, ms)
        sat_flow = np.minimum(sat_flow, full_thickness)
    else:
        full_thickness = np.full(shape, ms, dtype=np.float64)
    T = K * sat_flow

    sto = ctx.storage
    sy_coeff = np.zeros(shape, dtype=np.float64)
    ss_coeff = np.zeros(shape, dtype=np.float64)
    storage_diag = np.zeros(shape, dtype=np.float64)
    if sto.transient:
        hp = np.asarray(sto.head_prev, dtype=np.float64)
        sat_ref_zero = np.clip(h - b, 0.0, full_thickness)
        sat_old_zero = np.clip(hp - b, 0.0, full_thickness)
        sat_ref_ss = np.clip(h - b, ms, full_thickness)
        dh = h - hp
        moving = np.abs(dh) > 1.0e-12
        sy_coeff[moving] = float(sto.sy) * (sat_ref_zero[moving] - sat_old_zero[moving]) / dh[moving]
        if ctx.flow.ztop is not None:
            top = np.asarray(ctx.flow.ztop, dtype=np.float64)
            fallback = (~moving) & (h < top) & (h > b)
        else:
            fallback = (~moving) & (h > b)
        sy_coeff[fallback] = float(sto.sy)
        sy_coeff = np.clip(sy_coeff, 0.0, float(sto.sy))
        ss_coeff[:, :] = float(sto.ss) * sat_ref_ss
        area = float(ctx.grid.area)
        storage_diag = (sy_coeff + ss_coeff) * area / float(sto.dt)

    active = np.asarray(ctx.boundaries.active, dtype=np.int32) != 0
    T[~active] = 0.0
    sy_coeff[~free] = 0.0
    ss_coeff[~free] = 0.0
    storage_diag[~free] = 0.0
    return {
        "transmissivity": T,
        "sy_coeff": sy_coeff,
        "ss_coeff": ss_coeff,
        "storage_diag": storage_diag,
    }


def assemble_flow_operator_sparse(head: np.ndarray, ctx: Any) -> sp.csr_matrix:
    """Independent scipy CSR assembly of the head-dependent flow operator A(h).

    Built directly from the 5-point harmonic stencil with GHB diagonal; identity
    rows on inactive / Dirichlet cells.  Independent of both the device kernel
    and :func:`flow_operator_applied` (uses a different code path) so it is a
    genuine cross-check for confined linear consistency.
    """
    ny, nx = ctx.ny, ctx.nx
    n = ny * nx
    h = np.asarray(head, dtype=np.float64)
    T = _flow_transmissivity(h, ctx)
    active = np.asarray(ctx.boundaries.active, dtype=np.int32)
    dirichlet = np.asarray(ctx.boundaries.dirichlet_mask, dtype=np.int32) != 0
    gh_mask = np.asarray(ctx.boundaries.ghb_mask, dtype=np.int32) != 0
    ghb_factor = np.asarray(ctx.boundaries.ghb_factor, dtype=np.float64)

    def idx(j: int, i: int) -> int:
        return j * nx + i

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []

    for j in range(ny):
        for i in range(nx):
            k = idx(j, i)
            if active[j, i] == 0:
                rows.append(k); cols.append(k); vals.append(1.0)
                continue
            if dirichlet[j, i]:
                rows.append(k); cols.append(k); vals.append(1.0)
                continue
            diag = 0.0
            for dj, di, nj, ni in (
                (0, 1, j, i + 1), (0, -1, j, i - 1),
                (-1, 0, j - 1, i), (1, 0, j + 1, i),
            ):
                if 0 <= nj < ny and 0 <= ni < nx and active[nj, ni] != 0:
                    a = T[j, i]
                    bT = T[nj, ni]
                    if a > 0.0 and bT > 0.0:
                        C = 2.0 * a * bT / (a + bT + _TINY)
                        if C > 0.0:
                            diag += C
                            rows.append(k); cols.append(idx(nj, ni)); vals.append(-C)
            if gh_mask[j, i] and np.isfinite(ghb_factor[j, i]) and ghb_factor[j, i] > 0.0:
                diag += T[j, i] * ghb_factor[j, i]
            if diag < _TINY:
                rows.append(k); cols.append(k); vals.append(1.0)
            else:
                rows.append(k); cols.append(k); vals.append(diag)

    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
