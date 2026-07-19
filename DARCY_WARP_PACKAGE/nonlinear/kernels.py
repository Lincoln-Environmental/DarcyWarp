# SPDX-License-Identifier: AGPL-3.0-only
"""Authoritative 2D unconfined nonlinear operator kernels (Warp / device).

These kernels evaluate the *true* nonlinear groundwater equation directly from
hydraulic head.  They are deliberately self-contained: they do not call into
``warped_darcy`` and contain no Picard iteration, damping, relaxation,
acceleration, acceptance, fallback, or convergence logic.  They are the shared,
backend-neutral primitives that future nonlinear solvers
(``unconfined_semismooth_newton_kcycle``, ``unconfined_fas``) will consume.

Numerical conventions are replicated *exactly* from the trusted production
operator so the two agree bit-for-bit when transmissivity is head-independent:

* face conductance  ``T_face = 2 T_c T_nb / (T_c + T_nb + tiny)``  (harmonic mean,
  ``tiny = 1e-12``)  -- mirrors ``apply_A_kernel`` / ``compute_residual_kernel``
* GHB diagonal       ``C_gh = T_c * ghb_factor``                     -- mirrors ``apply_A_kernel``
* flow saturated thickness uses the positive ``min_sat`` ellipticity floor
  ``flow_sat = clip(head - bottom, min_sat, max(top - bottom, min_sat))``
* physical (storage) saturation uses the zero-to-full-thickness clipping of
  ``physics/storage_2d.py`` -- the ``min_sat`` flow floor is NEVER introduced
  into Sy or Ss storage.

The floating-point dtype is resolved from the same ``DARCY_FLOAT`` environment
variable as ``warped_darcy`` so device and host stay consistent.  All residual
arithmetic is carried out in ``wp.float64`` (matching the production residual
kernels) and only the stored field is cast to ``WP_FLOAT``.  Following the
repository convention there are no ``@wp.func`` helpers; storage/face maths are
inlined per kernel and validated against the host reference for consistency.
"""

from __future__ import annotations

import os

import numpy as np
import warp as wp


_float_env = os.environ.get("DARCY_FLOAT", "float64")
if _float_env == "float64":
    WP_FLOAT = wp.float64
    NP_FLOAT = np.float64
elif _float_env == "float32":
    WP_FLOAT = wp.float32
    NP_FLOAT = np.float32
else:
    raise ValueError("DARCY_FLOAT must be 'float32' or 'float64'.")


# Small regularization mirroring the production harmonic-mean denominator.
_NL_TINY = wp.float64(1.0e-12)


@wp.kernel
def nl_flow_transmissivity_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    K: wp.array(dtype=WP_FLOAT, ndim=2),
    zbot: wp.array(dtype=WP_FLOAT, ndim=2),
    has_ztop: int,
    ztop: wp.array(dtype=WP_FLOAT, ndim=2),
    min_sat: wp.float64,
    T_out: wp.array(dtype=WP_FLOAT, ndim=2),
    sat_out: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    """Populate the flow transmissivity ``T = K * flow_sat(head)``.

    ``flow_sat`` is the *flow* saturated thickness with the positive ``min_sat``
    ellipticity floor and (when a top is present) the ``max(top - bottom, min_sat)``
    cap.  This is the value used by the production Picard transmissivity update.
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    h = wp.float64(head[j, i])
    b = wp.float64(zbot[j, i])
    ms = wp.float64(min_sat)

    raw = h - b
    flow_sat = raw if raw > ms else ms
    if has_ztop != 0:
        thk_raw = wp.float64(ztop[j, i]) - b
        cap = thk_raw if thk_raw > ms else ms
        if flow_sat > cap:
            flow_sat = cap

    T_out[j, i] = WP_FLOAT(wp.float64(K[j, i]) * flow_sat)
    sat_out[j, i] = WP_FLOAT(flow_sat)


@wp.kernel
def nl_exact_storage_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    head_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    zbot: wp.array(dtype=WP_FLOAT, ndim=2),
    has_ztop: int,
    ztop: wp.array(dtype=WP_FLOAT, ndim=2),
    sy: wp.float64,
    ss: wp.float64,
    area: wp.float64,
    inv_dt: wp.float64,
    has_storage: int,
    free_mask: wp.array(dtype=wp.int32, ndim=2),
    total_out: wp.array(dtype=WP_FLOAT, ndim=2),
    sy_out: wp.array(dtype=WP_FLOAT, ndim=2),
    ss_out: wp.array(dtype=WP_FLOAT, ndim=2),
    sat_phys_out: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    """Authoritative exact convertible storage, mirroring ``storage_2d.py``.

    Outputs per-cell storage flux ``[L^3/T]``::

        sy_term = Sy * (sat_new - sat_old) * area / dt
        ss_term = (phi_new - phi_old) * area / dt
        total   = sy_term + ss_term

    where ``sat = clip(head - bottom, 0, max(top - bottom, 0))`` (zero floor) and
    ``phi`` is the exact specific-storage potential.  The ``min_sat`` flow floor
    is intentionally absent here.  Non-free cells write zero.
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if free_mask[j, i] == 0:
        total_out[j, i] = WP_FLOAT(0.0)
        sy_out[j, i] = WP_FLOAT(0.0)
        ss_out[j, i] = WP_FLOAT(0.0)
        sat_phys_out[j, i] = WP_FLOAT(0.0)
        return

    h = wp.float64(head[j, i])
    hp = wp.float64(head_prev[j, i])
    b = wp.float64(zbot[j, i])
    ss_f = wp.float64(ss)

    thk = wp.float64(0.0)
    if has_ztop != 0:
        thk_raw = wp.float64(ztop[j, i]) - b
        thk = thk_raw if thk_raw > wp.float64(0.0) else wp.float64(0.0)

    # Physical storage saturation: zero-to-full-thickness clipping.
    sat_new = h - b
    sat_old = hp - b
    if sat_new < wp.float64(0.0):
        sat_new = wp.float64(0.0)
    if sat_new > thk:
        sat_new = thk
    if sat_old < wp.float64(0.0):
        sat_old = wp.float64(0.0)
    if sat_old > thk:
        sat_old = thk

    sat_phys_out[j, i] = WP_FLOAT(sat_new)

    sy_term = wp.float64(0.0)
    ss_term = wp.float64(0.0)
    if has_storage != 0:
        sy_term = wp.float64(sy) * (sat_new - sat_old) * wp.float64(area) * wp.float64(inv_dt)
        # phi(h) - phi(h_prev), with phi the exact specific-storage potential.
        rel_new = h - b
        phi_new = wp.float64(0.0)
        if rel_new > wp.float64(0.0):
            if rel_new < thk:
                phi_new = wp.float64(0.5) * ss_f * rel_new * rel_new
            else:
                phi_new = wp.float64(0.5) * ss_f * thk * thk + ss_f * thk * (rel_new - thk)
        rel_old = hp - b
        phi_old = wp.float64(0.0)
        if rel_old > wp.float64(0.0):
            if rel_old < thk:
                phi_old = wp.float64(0.5) * ss_f * rel_old * rel_old
            else:
                phi_old = wp.float64(0.5) * ss_f * thk * thk + ss_f * thk * (rel_old - thk)
        ss_term = (phi_new - phi_old) * wp.float64(area) * wp.float64(inv_dt)

    sy_out[j, i] = WP_FLOAT(sy_term)
    ss_out[j, i] = WP_FLOAT(ss_term)
    total_out[j, i] = WP_FLOAT(sy_term + ss_term)


@wp.kernel
def nl_residual_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    K: wp.array(dtype=WP_FLOAT, ndim=2),
    zbot: wp.array(dtype=WP_FLOAT, ndim=2),
    has_ztop: int,
    ztop: wp.array(dtype=WP_FLOAT, ndim=2),
    min_sat: wp.float64,
    active: wp.array(dtype=wp.int32, ndim=2),
    dirichlet_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_head: wp.array(dtype=WP_FLOAT, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    R_field: wp.array(dtype=WP_FLOAT, ndim=2),
    head_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    sy: wp.float64,
    ss: wp.float64,
    area: wp.float64,
    inv_dt: wp.float64,
    has_storage: int,
    F_out: wp.array(dtype=WP_FLOAT, ndim=2),
    rTr_buf: wp.array(dtype=wp.float64, ndim=1),
    Fmax_buf: wp.array(dtype=wp.float64, ndim=1),
    nx: int,
    ny: int,
):
    """Authoritative free-cell nonlinear residual  ``F(h) = A(h) h - b(h)``.

    Written as the equivalent groundwater-balance residual on each free cell::

        F = flow_A(h) + storage_exact(h) - sources(h)

    where ``flow_A(h)`` is the head-dependent 5-point flow operator applied to
    ``h`` plus the GHB diagonal (NO storage diagonal), ``storage_exact(h)`` is
    the exact convertible storage flux, and
    ``sources(h) = R_field * area + C_gh(h) * gh_head``.  This is the negative of
    the production ``compute_residual_kernel`` residual ``r = b - A h``; the sign
    is documented and norm-invariant.

    Inactive and Dirichlet rows write ``F = 0`` so they are excluded from
    free-cell norms, matching the convention that those rows are identity rows.
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    if active[j, i] == 0 or dirichlet_mask[j, i] != 0:
        F_out[j, i] = WP_FLOAT(0.0)
        return

    ms = wp.float64(min_sat)

    hC = wp.float64(head[j, i])
    bC = wp.float64(zbot[j, i])
    KC = wp.float64(K[j, i])

    sat_c = hC - bC
    sat_c = sat_c if sat_c > ms else ms
    if has_ztop != 0:
        thk_raw_c = wp.float64(ztop[j, i]) - bC
        cap_c = thk_raw_c if thk_raw_c > ms else ms
        if sat_c > cap_c:
            sat_c = cap_c
    T_c = KC * sat_c

    # East / west / north / south face conductances (harmonic mean of T(h)).
    T_e = wp.float64(0.0)
    T_w = wp.float64(0.0)
    T_n = wp.float64(0.0)
    T_s = wp.float64(0.0)
    hE = wp.float64(0.0)
    hW = wp.float64(0.0)
    hN = wp.float64(0.0)
    hS = wp.float64(0.0)

    if i + 1 < nx and active[j, i + 1] != 0:
        hE = wp.float64(head[j, i + 1])
        b_nb = wp.float64(zbot[j, i + 1])
        sat_nb = hE - b_nb
        sat_nb = sat_nb if sat_nb > ms else ms
        if has_ztop != 0:
            thk_nb = wp.float64(ztop[j, i + 1]) - b_nb
            cap_nb = thk_nb if thk_nb > ms else ms
            if sat_nb > cap_nb:
                sat_nb = cap_nb
        T_nb = wp.float64(K[j, i + 1]) * sat_nb
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_e = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + _NL_TINY)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        hW = wp.float64(head[j, i - 1])
        b_nb = wp.float64(zbot[j, i - 1])
        sat_nb = hW - b_nb
        sat_nb = sat_nb if sat_nb > ms else ms
        if has_ztop != 0:
            thk_nb = wp.float64(ztop[j, i - 1]) - b_nb
            cap_nb = thk_nb if thk_nb > ms else ms
            if sat_nb > cap_nb:
                sat_nb = cap_nb
        T_nb = wp.float64(K[j, i - 1]) * sat_nb
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_w = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + _NL_TINY)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        hN = wp.float64(head[j - 1, i])
        b_nb = wp.float64(zbot[j - 1, i])
        sat_nb = hN - b_nb
        sat_nb = sat_nb if sat_nb > ms else ms
        if has_ztop != 0:
            thk_nb = wp.float64(ztop[j - 1, i]) - b_nb
            cap_nb = thk_nb if thk_nb > ms else ms
            if sat_nb > cap_nb:
                sat_nb = cap_nb
        T_nb = wp.float64(K[j - 1, i]) * sat_nb
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_n = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + _NL_TINY)

    if j + 1 < ny and active[j + 1, i] != 0:
        hS = wp.float64(head[j + 1, i])
        b_nb = wp.float64(zbot[j + 1, i])
        sat_nb = hS - b_nb
        sat_nb = sat_nb if sat_nb > ms else ms
        if has_ztop != 0:
            thk_nb = wp.float64(ztop[j + 1, i]) - b_nb
            cap_nb = thk_nb if thk_nb > ms else ms
            if sat_nb > cap_nb:
                sat_nb = cap_nb
        T_nb = wp.float64(K[j + 1, i]) * sat_nb
        if T_c > wp.float64(0.0) and T_nb > wp.float64(0.0):
            T_s = wp.float64(2.0) * T_c * T_nb / (T_c + T_nb + _NL_TINY)

    # GHB conductance term (diagonal side of the GHB row).
    C_gh = wp.float64(0.0)
    if gh_mask[j, i] != 0:
        ghbf = wp.float64(ghb_factor[j, i])
        if ghbf > wp.float64(0.0) and not wp.isnan(ghbf):
            C_gh = T_c * ghbf

    sum_T = T_e + T_w + T_n + T_s + C_gh

    # flow_A(h) (mirrors apply_A_kernel; isolated cell -> identity-like).
    flow_Ah = wp.float64(0.0)
    if sum_T < _NL_TINY:
        flow_Ah = hC
    else:
        flow_Ah = sum_T * hC
        if T_e > wp.float64(0.0):
            flow_Ah = flow_Ah - T_e * hE
        if T_w > wp.float64(0.0):
            flow_Ah = flow_Ah - T_w * hW
        if T_n > wp.float64(0.0):
            flow_Ah = flow_Ah - T_n * hN
        if T_s > wp.float64(0.0):
            flow_Ah = flow_Ah - T_s * hS

    # Exact convertible storage flux (zero-floor physical saturation).
    storage_flux = wp.float64(0.0)
    if has_storage != 0:
        hp = wp.float64(head_prev[j, i])
        ss_f = wp.float64(ss)
        thk = wp.float64(0.0)
        if has_ztop != 0:
            thk_raw = wp.float64(ztop[j, i]) - bC
            thk = thk_raw if thk_raw > wp.float64(0.0) else wp.float64(0.0)
        sat_new = hC - bC
        sat_old = hp - bC
        if sat_new < wp.float64(0.0):
            sat_new = wp.float64(0.0)
        if sat_new > thk:
            sat_new = thk
        if sat_old < wp.float64(0.0):
            sat_old = wp.float64(0.0)
        if sat_old > thk:
            sat_old = thk
        sy_term = wp.float64(sy) * (sat_new - sat_old) * wp.float64(area) * wp.float64(inv_dt)
        rel_new = hC - bC
        phi_new = wp.float64(0.0)
        if rel_new > wp.float64(0.0):
            if rel_new < thk:
                phi_new = wp.float64(0.5) * ss_f * rel_new * rel_new
            else:
                phi_new = wp.float64(0.5) * ss_f * thk * thk + ss_f * thk * (rel_new - thk)
        rel_old = hp - bC
        phi_old = wp.float64(0.0)
        if rel_old > wp.float64(0.0):
            if rel_old < thk:
                phi_old = wp.float64(0.5) * ss_f * rel_old * rel_old
            else:
                phi_old = wp.float64(0.5) * ss_f * thk * thk + ss_f * thk * (rel_old - thk)
        ss_term = (phi_new - phi_old) * wp.float64(area) * wp.float64(inv_dt)
        storage_flux = sy_term + ss_term

    # Sources: signed recharge/well field + GHB external-head inflow.
    recharge = wp.float64(R_field[j, i]) * wp.float64(area)
    ghb_source = C_gh * wp.float64(gh_head[j, i])

    F = flow_Ah + storage_flux - recharge - ghb_source
    F_out[j, i] = WP_FLOAT(F)
    wp.atomic_add(rTr_buf, 0, F * F)
    wp.atomic_max(Fmax_buf, 0, wp.abs(F))


@wp.kernel
def nl_picard_freeze_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    head_prev: wp.array(dtype=WP_FLOAT, ndim=2),
    K: wp.array(dtype=WP_FLOAT, ndim=2),
    zbot: wp.array(dtype=WP_FLOAT, ndim=2),
    has_ztop: int,
    ztop: wp.array(dtype=WP_FLOAT, ndim=2),
    min_sat: wp.float64,
    sy: wp.float64,
    ss: wp.float64,
    area: wp.float64,
    inv_dt: wp.float64,
    active: wp.array(dtype=wp.int32, ndim=2),
    free_mask: wp.array(dtype=wp.int32, ndim=2),
    T_out: wp.array(dtype=WP_FLOAT, ndim=2),
    sy_coeff_out: wp.array(dtype=WP_FLOAT, ndim=2),
    ss_coeff_out: wp.array(dtype=WP_FLOAT, ndim=2),
    storage_diag_out: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    """Freeze the production Picard linearisation at ``head`` (secant storage).

    Replicates ``picard_unconfined._storage_from_picard_head`` *exactly* so the
    coefficients returned here are the ones the trusted Picard backend would
    assemble at this head -- without invoking the Picard algorithm:

    * ``T = K * flow_sat(head)``                            (min_sat floor)
    * ``full_thickness = max(top - bottom, min_sat)``
    * ``sy_coeff = clip(Sy * (sat_ref_zero - sat_old_zero) / dh, 0, Sy)``  (zero-floor sat)
    * ``ss_coeff = Ss * clip(head - bottom, min_sat, full_thickness)``     (min_sat-floor sat)
    * ``storage_diag = (sy_coeff + ss_coeff) * area / dt``

    The Ss branch intentionally uses the ``min_sat`` floor (an ellipticity
    convenience inherited from the production linearisation); the authoritative
    ``nl_exact_storage_kernel`` uses the zero-floor physical potential instead.
    The two reconcile as ``dh -> 0`` (see module docstring and tests).
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return

    ms = wp.float64(min_sat)
    hC = wp.float64(head[j, i])
    bC = wp.float64(zbot[j, i])

    # Flow transmissivity (min_sat floor).
    sat_flow = hC - bC
    sat_flow = sat_flow if sat_flow > ms else ms
    full_thickness = ms
    top_val = wp.float64(0.0)
    if has_ztop != 0:
        top_val = wp.float64(ztop[j, i])
        thk_raw = top_val - bC
        full_thickness = thk_raw if thk_raw > ms else ms
        if sat_flow > full_thickness:
            sat_flow = full_thickness
    # T is zero on inactive cells (production zeros transmissivity on ~active);
    # Dirichlet cells keep their T because free neighbours face them.
    if active[j, i] == 0:
        T_out[j, i] = WP_FLOAT(0.0)
    else:
        T_out[j, i] = WP_FLOAT(wp.float64(K[j, i]) * sat_flow)

    if free_mask[j, i] == 0:
        sy_coeff_out[j, i] = WP_FLOAT(0.0)
        ss_coeff_out[j, i] = WP_FLOAT(0.0)
        storage_diag_out[j, i] = WP_FLOAT(0.0)
        return

    sy_f = wp.float64(sy)
    ss_f = wp.float64(ss)

    sat_ref_zero = hC - bC
    sat_old_zero = wp.float64(head_prev[j, i]) - bC
    if sat_ref_zero < wp.float64(0.0):
        sat_ref_zero = wp.float64(0.0)
    if sat_ref_zero > full_thickness:
        sat_ref_zero = full_thickness
    if sat_old_zero < wp.float64(0.0):
        sat_old_zero = wp.float64(0.0)
    if sat_old_zero > full_thickness:
        sat_old_zero = full_thickness

    sat_ref_ss = hC - bC
    if sat_ref_ss < ms:
        sat_ref_ss = ms
    if sat_ref_ss > full_thickness:
        sat_ref_ss = full_thickness

    dh = hC - wp.float64(head_prev[j, i])
    secant_eps = wp.float64(1.0e-12)
    sy_coeff = wp.float64(0.0)
    if wp.abs(dh) > secant_eps:
        sy_coeff = sy_f * (sat_ref_zero - sat_old_zero) / dh
    else:
        in_unsat = False
        if has_ztop != 0:
            if hC < top_val and hC > bC:
                in_unsat = True
        else:
            if hC > bC:
                in_unsat = True
        if in_unsat:
            sy_coeff = sy_f
    # clip to [0, Sy]
    if sy_coeff < wp.float64(0.0):
        sy_coeff = wp.float64(0.0)
    if sy_coeff > sy_f:
        sy_coeff = sy_f

    ss_coeff = ss_f * sat_ref_ss

    diag = (sy_coeff + ss_coeff) * wp.float64(area) * wp.float64(inv_dt)
    sy_coeff_out[j, i] = WP_FLOAT(sy_coeff)
    ss_coeff_out[j, i] = WP_FLOAT(ss_coeff)
    storage_diag_out[j, i] = WP_FLOAT(diag)


@wp.kernel
def nl_jacobian_vector_kernel(
    head: wp.array(dtype=WP_FLOAT, ndim=2),
    vector: wp.array(dtype=WP_FLOAT, ndim=2),
    K: wp.array(dtype=WP_FLOAT, ndim=2),
    zbot: wp.array(dtype=WP_FLOAT, ndim=2),
    has_ztop: int,
    ztop: wp.array(dtype=WP_FLOAT, ndim=2),
    min_sat: wp.float64,
    active: wp.array(dtype=wp.int32, ndim=2),
    dirichlet_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_mask: wp.array(dtype=wp.int32, ndim=2),
    gh_head: wp.array(dtype=WP_FLOAT, ndim=2),
    ghb_factor: wp.array(dtype=WP_FLOAT, ndim=2),
    sy: wp.float64,
    ss: wp.float64,
    area: wp.float64,
    inv_dt: wp.float64,
    has_storage: int,
    Jv_out: wp.array(dtype=WP_FLOAT, ndim=2),
    nx: int,
    ny: int,
):
    """Analytic generalized Jacobian action for :func:`nl_residual_kernel`.

    The clipping derivative is one strictly inside a clipping interval and
    zero outside it and exactly at either threshold.  Inactive and prescribed
    rows are correction rows and therefore return zero.  Each face includes
    both ``g * (v_i-v_j)`` and ``Dg[v] * (h_i-h_j)``.
    """
    j, i = wp.tid()
    if j >= ny or i >= nx:
        return
    if active[j, i] == 0 or dirichlet_mask[j, i] != 0:
        Jv_out[j, i] = WP_FLOAT(0.0)
        return

    ms = wp.float64(min_sat)
    h_c = wp.float64(head[j, i])
    v_c = wp.float64(vector[j, i])
    b_c = wp.float64(zbot[j, i])
    raw_c = h_c - b_c
    cap_c = wp.float64(1.0e300)
    if has_ztop != 0:
        thk_raw_c = wp.float64(ztop[j, i]) - b_c
        cap_c = thk_raw_c if thk_raw_c > ms else ms
    sat_c = raw_c
    if sat_c <= ms:
        sat_c = ms
    if sat_c >= cap_c:
        sat_c = cap_c
    ds_c = wp.float64(0.0)
    if raw_c > ms and raw_c < cap_c:
        ds_c = v_c
    T_c = wp.float64(K[j, i]) * sat_c
    dT_c = wp.float64(K[j, i]) * ds_c

    result = wp.float64(0.0)
    flow_sum = wp.float64(0.0)

    # The four face blocks deliberately remain explicit.  This mirrors the
    # authoritative residual launch and makes the active-neighbour semantics
    # identical without relying on Warp AD or a separate sparse matrix.
    if i + 1 < nx and active[j, i + 1] != 0:
        h_n = wp.float64(head[j, i + 1])
        v_n = wp.float64(vector[j, i + 1])
        b_n = wp.float64(zbot[j, i + 1])
        raw_n = h_n - b_n
        cap_n = wp.float64(1.0e300)
        if has_ztop != 0:
            thk_n = wp.float64(ztop[j, i + 1]) - b_n
            cap_n = thk_n if thk_n > ms else ms
        sat_n = raw_n
        if sat_n <= ms:
            sat_n = ms
        if sat_n >= cap_n:
            sat_n = cap_n
        ds_n = wp.float64(0.0)
        if raw_n > ms and raw_n < cap_n:
            ds_n = v_n
        T_n = wp.float64(K[j, i + 1]) * sat_n
        dT_n = wp.float64(K[j, i + 1]) * ds_n
        if T_c > wp.float64(0.0) and T_n > wp.float64(0.0):
            den = T_c + T_n + _NL_TINY
            g = wp.float64(2.0) * T_c * T_n / den
            flow_sum = flow_sum + g
            dg = (wp.float64(2.0) * T_n * (T_n + _NL_TINY) * dT_c +
                  wp.float64(2.0) * T_c * (T_c + _NL_TINY) * dT_n) / (den * den)
            result = result + g * (v_c - v_n) + dg * (h_c - h_n)

    if i - 1 >= 0 and active[j, i - 1] != 0:
        h_n = wp.float64(head[j, i - 1])
        v_n = wp.float64(vector[j, i - 1])
        b_n = wp.float64(zbot[j, i - 1])
        raw_n = h_n - b_n
        cap_n = wp.float64(1.0e300)
        if has_ztop != 0:
            thk_n = wp.float64(ztop[j, i - 1]) - b_n
            cap_n = thk_n if thk_n > ms else ms
        sat_n = raw_n
        if sat_n <= ms:
            sat_n = ms
        if sat_n >= cap_n:
            sat_n = cap_n
        ds_n = wp.float64(0.0)
        if raw_n > ms and raw_n < cap_n:
            ds_n = v_n
        T_n = wp.float64(K[j, i - 1]) * sat_n
        dT_n = wp.float64(K[j, i - 1]) * ds_n
        if T_c > wp.float64(0.0) and T_n > wp.float64(0.0):
            den = T_c + T_n + _NL_TINY
            g = wp.float64(2.0) * T_c * T_n / den
            flow_sum = flow_sum + g
            dg = (wp.float64(2.0) * T_n * (T_n + _NL_TINY) * dT_c +
                  wp.float64(2.0) * T_c * (T_c + _NL_TINY) * dT_n) / (den * den)
            result = result + g * (v_c - v_n) + dg * (h_c - h_n)

    if j - 1 >= 0 and active[j - 1, i] != 0:
        h_n = wp.float64(head[j - 1, i])
        v_n = wp.float64(vector[j - 1, i])
        b_n = wp.float64(zbot[j - 1, i])
        raw_n = h_n - b_n
        cap_n = wp.float64(1.0e300)
        if has_ztop != 0:
            thk_n = wp.float64(ztop[j - 1, i]) - b_n
            cap_n = thk_n if thk_n > ms else ms
        sat_n = raw_n
        if sat_n <= ms:
            sat_n = ms
        if sat_n >= cap_n:
            sat_n = cap_n
        ds_n = wp.float64(0.0)
        if raw_n > ms and raw_n < cap_n:
            ds_n = v_n
        T_n = wp.float64(K[j - 1, i]) * sat_n
        dT_n = wp.float64(K[j - 1, i]) * ds_n
        if T_c > wp.float64(0.0) and T_n > wp.float64(0.0):
            den = T_c + T_n + _NL_TINY
            g = wp.float64(2.0) * T_c * T_n / den
            flow_sum = flow_sum + g
            dg = (wp.float64(2.0) * T_n * (T_n + _NL_TINY) * dT_c +
                  wp.float64(2.0) * T_c * (T_c + _NL_TINY) * dT_n) / (den * den)
            result = result + g * (v_c - v_n) + dg * (h_c - h_n)

    if j + 1 < ny and active[j + 1, i] != 0:
        h_n = wp.float64(head[j + 1, i])
        v_n = wp.float64(vector[j + 1, i])
        b_n = wp.float64(zbot[j + 1, i])
        raw_n = h_n - b_n
        cap_n = wp.float64(1.0e300)
        if has_ztop != 0:
            thk_n = wp.float64(ztop[j + 1, i]) - b_n
            cap_n = thk_n if thk_n > ms else ms
        sat_n = raw_n
        if sat_n <= ms:
            sat_n = ms
        if sat_n >= cap_n:
            sat_n = cap_n
        ds_n = wp.float64(0.0)
        if raw_n > ms and raw_n < cap_n:
            ds_n = v_n
        T_n = wp.float64(K[j + 1, i]) * sat_n
        dT_n = wp.float64(K[j + 1, i]) * ds_n
        if T_c > wp.float64(0.0) and T_n > wp.float64(0.0):
            den = T_c + T_n + _NL_TINY
            g = wp.float64(2.0) * T_c * T_n / den
            flow_sum = flow_sum + g
            dg = (wp.float64(2.0) * T_n * (T_n + _NL_TINY) * dT_c +
                  wp.float64(2.0) * T_c * (T_c + _NL_TINY) * dT_n) / (den * den)
            result = result + g * (v_c - v_n) + dg * (h_c - h_n)

    # Head-dependent GHB conductance C_gh=T(h)*factor.
    if gh_mask[j, i] != 0:
        factor = wp.float64(ghb_factor[j, i])
        if factor > wp.float64(0.0) and not wp.isnan(factor):
            C = T_c * factor
            dC = dT_c * factor
            flow_sum = flow_sum + C
            result = result + C * v_c + dC * (h_c - wp.float64(gh_head[j, i]))

    if flow_sum < _NL_TINY:
        result = v_c

    # Exact convertible storage derivative.  Sy follows clipped physical
    # saturation.  The Ss potential remains linear above aquifer top, as in
    # the Stage-1 residual, so its derivative there is Ss*full_thickness.
    if has_storage != 0:
        thk = wp.float64(0.0)
        if has_ztop != 0:
            thk_raw = wp.float64(ztop[j, i]) - b_c
            thk = thk_raw if thk_raw > wp.float64(0.0) else wp.float64(0.0)
        rel = h_c - b_c
        ds_phys = wp.float64(0.0)
        if rel > wp.float64(0.0) and rel < thk:
            ds_phys = wp.float64(1.0)
        dphi = wp.float64(0.0)
        if rel > wp.float64(0.0) and rel < thk:
            dphi = wp.float64(ss) * rel
        elif rel > thk and thk > wp.float64(0.0):
            dphi = wp.float64(ss) * thk
        result = result + (wp.float64(sy) * ds_phys + dphi) * wp.float64(area) * wp.float64(inv_dt) * v_c

    Jv_out[j, i] = WP_FLOAT(result)
