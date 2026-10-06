"""Spatial-recharge interval through a persistent production Picard session.

The replay driver and this boundary share one numerical engine. This boundary
reuses an assembled model and integrates native budgets over accepted adaptive
substeps. It never rebuilds an MF6 warm-start or uses cached model heads.
"""
from __future__ import annotations

import time
import numpy as np

from .transient_unconfined import TransientUnconfinedSession
from ..physics.budgets_2d import compute_mass_balance_budget, add_exact_storage_to_budget
from ..physics.storage_2d import exact_unconfined_storage_terms


class _AcceptedBudget:
    def __init__(self, *, model, initial_head, conductivity, bottom, top, sy, ss, min_sat):
        self.model = model
        self.previous = initial_head.copy()
        self.conductivity = conductivity
        self.bottom, self.top = bottom, top
        self.sy, self.ss, self.min_sat = sy, ss, min_sat
        self.volumes = {}
        self.duration_days = 0.

    def accumulate(self, *, head, duration_days):
        model = self.model
        flow_sat = np.clip(head-self.bottom, self.min_sat,
                           np.maximum(self.top-self.bottom, self.min_sat))
        transmissivity = self.conductivity*flow_sat
        transmissivity[np.asarray(model.active_host) == 0] = 0.
        budget = compute_mass_balance_budget(T_field=transmissivity,
            R_field=model.R_field_host, head=head, active=model.active_host,
            bc_mask=model.bc_mask_host, bc_values=model.bc_values_host, dx=float(model.dx),
            gh_mask=model.gh_mask_host if model.use_ghb else None,
            gh_head=model.gh_head_host if model.use_ghb else None,
            gh_width=model.gh_width_host if model.use_ghb else None,
            ghb_factor=model.ghb_factor_host if model.use_ghb else None,
            case='unconfined_picard_kcycle')
        terms = exact_unconfined_storage_terms(head_new=head, head_old=self.previous,
            bottom=self.bottom, top=self.top, specific_yield=self.sy,
            specific_storage=self.ss, dt=duration_days)[0]*float(model.dx)**2
        terms[(model.active_host == 0) | (model.bc_mask_host != 0)] = 0.
        budget = add_exact_storage_to_budget(budget=budget, storage_flux=terms)
        for name, rate in dict(budget.iloc[0]).items():
            if isinstance(rate, (int, float, np.number)):
                self.volumes[name] = self.volumes.get(name, 0.) + float(rate)*duration_days
        self.previous = head.copy()
        self.duration_days += duration_days

    def rates(self, *, duration_days):
        if not np.isclose(self.duration_days, duration_days, rtol=1e-10, atol=1e-12):
            raise RuntimeError('accepted Picard substeps do not cover requested interval')
        result = {name: volume/duration_days for name, volume in self.volumes.items()}
        total_in, total_out = result['total_in'], result['total_out']
        result['in_minus_out'] = total_in-total_out
        result['throughflow'] = .5*(total_in+total_out)
        result['percent_discrepancy'] = (100*(total_in-total_out)/(total_in+total_out)
                                         if total_in+total_out else 0.)
        result['imbalance_fraction'] = (2*(total_in-total_out)/(total_in+total_out)
                                       if total_in+total_out else 0.)
        return result


def solve_picard_transient_interval(*, model, **arguments):
    """Reuse the production session with explicit accepted head and recharge.

    The session cache owns numerical workspace only. Changes to static fields
    or solve controls create a new compatible session; coupling rollback does
    not restore or clone this cache.
    """
    started = time.perf_counter()
    controls = dict(arguments)
    return_info = bool(controls.pop('return_info', True))
    if not bool(controls.pop('transient', False)):
        raise ValueError('device transient interval requires transient=True')
    initial = np.asarray(controls.pop('initial_head'), dtype=np.float64)
    supplied_previous = controls.pop('head_prev', None)
    previous = initial if supplied_previous is None else np.asarray(supplied_previous, dtype=np.float64)
    if not np.array_equal(initial, previous):
        raise ValueError('device transient interval requires initial_head equal to accepted head_prev')
    conductivity = np.asarray(controls.pop('K_field'), dtype=np.float64)
    bottom = np.asarray(controls.pop('zbot_field'), dtype=np.float64)
    top = np.asarray(controls.pop('ztop_field'), dtype=np.float64)
    sy, ss = float(controls.pop('sy')), float(controls.pop('ss', 0.))
    dt = float(controls.pop('dt'))
    storage_mode = controls.pop('unconfined_storage_mode_2d', 'mf6_convertible_secant_sy')
    storage_reference = controls.pop('storage_reference', 'current_picard')
    controls.pop('storage_coeff', None)
    controls.pop('refresh_diag_with_transient_storage', None)
    if storage_mode != 'mf6_convertible_secant_sy' or storage_reference != 'current_picard':
        raise ValueError('device transient interval requires validated current-Picard secant Sy storage')
    if not controls.get('use_device_transient_fast_path', False):
        raise ValueError('device transient interval requires use_device_transient_fast_path=True')
    min_sat = float(controls.get('min_saturated_thickness', .1))
    parameters = dict(k_field=conductivity, zbot_field=bottom, ztop_field=top,
        sy=sy, ss=ss, active=model.active_host, bc_mask=model.bc_mask_host,
        bc_values=model.bc_values_host, gh_mask=model.gh_mask_host if model.use_ghb else None,
        gh_head=model.gh_head_host if model.use_ghb else None,
        gh_width=model.gh_width_host if model.use_ghb else None,
        gh_alpha=model.gh_alpha, aq_thickness=model.aq_thickness,
        storage_mode=storage_mode, storage_reference=storage_reference,
        solve_controls=controls, min_saturated_thickness=min_sat, reuse_model=True)
    session = getattr(model, '_production_transient_session', None)
    compatible = session is not None and session.solve_controls == controls
    if compatible:
        for name, value in parameters.items():
            if name in {'solve_controls', 'reuse_model'}:
                continue
            old = session._parameters[name]
            if not np.array_equal(old, value):
                compatible = False
                break
    if not compatible:
        if session is not None:
            session.invalidate()
        session = TransientUnconfinedSession(model=model, **parameters)
        model._production_transient_session = session
    ledger = _AcceptedBudget(model=model, initial_head=previous, conductivity=conductivity,
        bottom=bottom, top=top, sy=sy, ss=ss, min_sat=min_sat)
    head, info = session.advance(head_prev=previous,
        recharge_rate_m_per_day=np.asarray(model.R_field_host), dt=dt,
        accepted_substep_callback=ledger.accumulate)
    info.update(runtime_seconds=time.perf_counter()-started,
        budget_summary=ledger.rates(duration_days=dt),
        contact_discharge_rate_array=np.zeros_like(head), contact_active_set_iterations=0)
    return (head, info) if return_info else head
