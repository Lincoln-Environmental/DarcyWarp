"""Opt-in FP64 fast steady Newton profile; transient defaults are untouched."""
from pathlib import Path
import subprocess
import hashlib


def production_unconfined_steady_controls(*, max_levels: int = 5) -> dict:
    """Return explicit controls for steady DRN/RIV/coastal unconfined Newton.

    Component-tested configuration, not a physical acceptance certificate.
    Callers supply the actual model, geometry, sources and initial head.
    Changing the hierarchy cap requires a fresh model/hierarchy.
    """
    if not isinstance(max_levels, int) or isinstance(max_levels, bool) or max_levels < 1:
        raise ValueError("max_levels must be a positive integer")
    return {
        "formulation": "unconfined",
        "solver": "unconfined_semismooth_newton_kcycle",
        "transient": False,
        "return_info": True,
        "inner_implementation": "fast",
        "max_levels": max_levels,
        "min_coarse_cells": 500,
        "nu_pre": 2, "nu_post": 2, "nu_coarse": 30,
        "smoother": "chebyshev", "omega": 0.7,
        "cheby_lambda_min": 0.1, "cheby_lambda_max": 2.0,
        "newton_preconditioner": "kcycle",
        "newton_preconditioner_kcycles": 2,
        "newton_coarse_residual_restriction": "average",
        "unconfined_startup_mode": "confined_pre_solve",
        # These three legacy names ARE consumed by confined active-set startup.
        "max_outer_iterations": 60, "rel_tol": 5e-7, "abs_tol_min": 5e-7,
        "min_saturated_thickness": 0.1,
        "newton_max_iterations": 100,
        "newton_residual_rms_tolerance": 1e-6,
        "newton_head_equivalent_rms_tolerance": 1e-4,
        "newton_fgmres_restart": 20, "newton_fgmres_max_iterations": 60,
        "newton_fgmres_absolute_tolerance": 1e-10,
        # Omit fixed relative tolerance: retain native adaptive true-residual forcing.
        "newton_inexact_forcing_enabled": True,
        "newton_forcing_eta_initial": 0.1, "newton_forcing_eta_min": 1e-7,
        "newton_forcing_eta_max": 0.5, "newton_forcing_eta_power": 1.0,
        "newton_forcing_safeguard_backtracks": 3,
        "newton_forcing_final_tighten_ratio": 100.0,
        "newton_max_backtracks": 10, "newton_min_step_length": 2.0**-10,
        "newton_armijo_coefficient": 1e-4, "newton_max_head_change": 10.0,
        "newton_gated_model_enabled": True,
        "newton_picard_warm_start_iterations": 0,
        "newton_fallback_to_picard": False,
    }


def steady_source_identity(*, model_file: str, solver_file: str) -> dict:
    """Read import provenance once after a solve, never inside GPU work loops."""
    model_path = Path(model_file).resolve()
    root = model_path.parent.parent
    package = model_path.parent
    files = [model_path, Path(solver_file).resolve(), Path(__file__).resolve(),
        package/'solvers/kcycle_preconditioner.py', package/'solvers/face_transient_f64.py',
        package/'solvers/fgmres.py']
    fingerprints = {str(file): hashlib.sha256(file.read_bytes()).hexdigest() for file in files}
    try:
        revision = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True, timeout=2).stdout.strip()
        branch = subprocess.run(["git", "-C", str(root), "branch", "--show-current"],
            capture_output=True, text=True, check=True, timeout=2).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--quiet"],
            capture_output=True, check=False, timeout=2).returncode != 0
        return dict(repository=str(root), sha=revision, branch=branch,
            tracked_dirty=dirty, file_sha256=fingerprints, model_file=str(model_path), solver_file=str(Path(solver_file).resolve()))
    except (OSError, subprocess.SubprocessError) as error:
        return dict(model_file=str(model_path), solver_file=str(Path(solver_file).resolve()),
            sha=None, provenance_error=str(error))
