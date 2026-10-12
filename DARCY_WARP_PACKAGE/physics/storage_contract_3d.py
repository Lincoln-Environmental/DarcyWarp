"""Explicit storage input contracts for the 3D unconfined solvers."""
import numpy as np


def validate_unconfined_storage_inputs(*, transient, mode, sy, ss,
                                       storage_coeff, storage_diag, head_prev,
                                       initial_head, dt):
    """Reject unused inputs; never infer a different constitutive formulation.

    Secant/phreatic modes use Sy and Ss. Confined-volume mode uses only
    storage_coeff. Preassembled diagonals are deliberately unsupported for
    unconfined flow: their datum-dependent RHS cannot be inferred safely.
    """
    mode = str(mode).strip().lower()
    if mode not in {"secant_sy", "phreatic_sy", "confined_volume"}:
        raise ValueError("unconfined_storage_mode must be secant_sy, phreatic_sy, or confined_volume")
    if storage_diag is not None:
        raise ValueError("unconfined storage_diag is unsupported; supply sy/ss or explicit confined_volume storage_coeff")
    if mode in {"secant_sy", "phreatic_sy"} and storage_coeff is not None:
        raise ValueError(f"storage_coeff is incompatible with {mode}; select confined_volume explicitly")
    if mode == "confined_volume" and (sy is not None or ss is not None):
        raise ValueError("sy/ss are incompatible with confined_volume; use storage_coeff only")
    if not transient:
        if any(value is not None for value in (sy, ss, storage_coeff, head_prev, dt)):
            raise ValueError("storage inputs require transient=True")
        return mode
    if dt is None or not np.isfinite(float(dt)) or float(dt) <= 0:
        raise ValueError("transient unconfined storage requires dt > 0")
    if head_prev is None and initial_head is None:
        raise ValueError("transient unconfined storage requires head_prev or an explicit initial_head")
    if mode == "secant_sy" and sy is None and ss is None:
        raise ValueError("secant_sy requires sy and/or ss")
    if mode == "phreatic_sy" and sy is None:
        raise ValueError("phreatic_sy requires sy")
    if mode == "confined_volume" and storage_coeff is None:
        raise ValueError("confined_volume requires storage_coeff")
    return mode
