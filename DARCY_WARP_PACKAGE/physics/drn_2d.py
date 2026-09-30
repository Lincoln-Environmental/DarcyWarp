"""MODFLOW-style DRN discharge and budget augmentation for accepted 2D heads."""

from __future__ import annotations

import numpy as np
import pandas as pd


def drn_discharge_2d(
    *,
    head: np.ndarray,
    elevation: np.ndarray,
    conductance: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Return non-negative per-cell aquifer outflow [L3/T].

    Mirrors the nonlinear operator's gated diagonal/source pair exactly:
    active cells have ``C * (head - elevation)`` for head above elevation,
    and zero otherwise. The caller supplies the sanitized DRN mask.
    """
    h = np.asarray(head, dtype=np.float64)
    elev = np.asarray(elevation, dtype=np.float64)
    cond = np.asarray(conductance, dtype=np.float64)
    enabled = np.asarray(mask, dtype=np.int32) != 0
    if h.ndim != 2:
        raise ValueError("head must be a 2D field")
    for name, field in (("elevation", elev), ("conductance", cond), ("mask", enabled)):
        if field.shape != h.shape:
            raise ValueError(f"{name} shape {field.shape} expected {h.shape}")
    if not np.all(np.isfinite(h[enabled])) or not np.all(np.isfinite(elev[enabled])):
        raise ValueError("active DRN head and elevation must be finite")
    if not np.all(np.isfinite(cond[enabled])) or np.any(cond[enabled] < 0.0):
        raise ValueError("active DRN conductance must be finite and non-negative")
    flux = np.zeros(h.shape, dtype=np.float64)
    flowing = enabled & (cond > 0.0) & (h > elev)
    flux[flowing] = cond[flowing] * (h[flowing] - elev[flowing])
    return flux


def add_drn_to_budget(budget: pd.DataFrame, discharge: np.ndarray) -> pd.DataFrame:
    """Add DRN gross outflow to one accepted groundwater budget row.

    This also works with a budget table that already has zero-filled DRN
    columns, without counting discharge twice.
    """
    if len(budget) != 1:
        raise ValueError("budget must contain exactly one row")
    q = np.asarray(discharge, dtype=np.float64)
    if q.ndim != 2 or not np.all(np.isfinite(q)) or np.any(q < 0.0):
        raise ValueError("discharge must be a finite, non-negative 2D field")
    result = budget.copy()
    row = result.index[0]
    old_out = float(result.loc[row, "drn_out"]) if "drn_out" in result.columns else 0.0
    new_out = float(q.sum())
    if old_out != 0.0 and not np.isclose(old_out, new_out, rtol=1.0e-10, atol=1.0e-12):
        raise ValueError("existing DRN budget conflicts with accepted discharge")
    result.loc[row, "drn_in"] = 0.0
    result.loc[row, "drn_out"] = new_out
    result.loc[row, "drn_net_out_positive"] = new_out
    delta = new_out - old_out
    total_in = float(result.loc[row, "total_in"])
    total_out = float(result.loc[row, "total_out"]) + delta
    imbalance = total_in - total_out
    denominator = abs(total_in) + abs(total_out)
    result.loc[row, "total_out"] = total_out
    result.loc[row, "in_minus_out"] = imbalance
    result.loc[row, "percent_discrepancy"] = 0.0 if denominator == 0.0 else 100.0 * imbalance / denominator
    result.loc[row, "throughflow"] = 0.5 * (total_in + total_out)
    result.loc[row, "imbalance_fraction"] = 0.0 if total_in + total_out == 0.0 else imbalance / (0.5 * (total_in + total_out))
    return result
