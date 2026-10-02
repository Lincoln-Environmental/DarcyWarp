"""Zero-resistance head contact with an atmospheric complementarity active set.

Reaction is positive out of groundwater. The ordinary accepted FV residual,
including storage, evaluates it; this module contains no groundwater equation.
"""
from __future__ import annotations

import numpy as np


def solve_surface_contact_2d(*, model, contact_mask, surface_head,
                             ponded_mask, solve_arguments, maximum_iterations=32,
                             maximum_inflow_rate_m3_per_day=None):
    """Solve from the same previous head on every atmospheric active-set trial.

    Ponded, funded contact: H=Hsurface, signed reaction. With finite
    interval supply b: H<=Hsurface, q>=-b, (q+b)*(Hsurface-H)=0.
    Empty/no-supply dry contact has b=0. If supply is exhausted, impose
    the actual bounded FV inflow and release head equality. Genuine CHD
    fields and recharge accounting are never changed.
    """
    if solve_arguments.get("solver") != "unconfined_semismooth_newton_kcycle":
        raise ValueError("head contact requires the semismooth-Newton backend")
    if solve_arguments.get("transient", False) and solve_arguments.get("head_prev") is None:
        raise ValueError("transient contact requires an explicit accepted previous head")
    if maximum_iterations < 1:
        raise ValueError("maximum_iterations must be positive")
    mask = np.asarray(contact_mask, dtype=bool)
    stage = np.asarray(surface_head, dtype=np.float64)
    ponded = np.asarray(ponded_mask, dtype=bool)
    shape = (model.ny, model.nx)
    if mask.shape != shape or stage.shape != shape or ponded.shape != shape or not np.all(np.isfinite(stage)):
        raise ValueError("contact fields must be finite matching grids")
    cap = (np.where(ponded, np.inf, 0.) if maximum_inflow_rate_m3_per_day is None
           else np.asarray(maximum_inflow_rate_m3_per_day, dtype=np.float64))
    if cap.shape != shape or np.any(np.isnan(cap)) or np.any(cap < 0):
        raise ValueError("contact supply cap must be nonnegative matching field")
    constrained = mask.copy()
    bounded = np.zeros(shape, dtype=bool)
    arguments = dict(solve_arguments)
    for iteration in range(maximum_iterations):
        head, info = model.solve(**arguments, fixed_contact_mask=constrained,
                                 fixed_contact_head=stage,
                                 fixed_contact_flux=np.where(bounded, -cap, 0.))
        if not info.get("converged", False):
            return head, info
        reaction = info["contact_discharge_rate_array"]
        # Roundoff-scale switching only; integration/solver tolerances unchanged.
        release = constrained & (reaction < -cap-1e-10)
        activate = mask & ~constrained & (head > stage + 1e-12)
        updated = (constrained & ~release) | activate
        updated_bounded = (bounded | release) & ~activate
        if np.array_equal(updated, constrained) and np.array_equal(updated_bounded,bounded):
            info["contact_active_set_iterations"] = iteration + 1
            info["contact_supply_limited_mask"] = bounded.copy()
            return head, info
        bounded = updated_bounded
        constrained = updated
        # At the exact aquifer top Sy's generalized derivative is zero.
        # Seed released rows on the unsaturated side for Newton, without
        # altering the accepted previous head or storage law.
        initial = np.asarray(head).copy()
        initial[release] = np.minimum(initial[release], stage[release] - 1e-8)
        arguments["initial_head"] = initial
    raise RuntimeError("surface contact atmospheric active set did not converge")
