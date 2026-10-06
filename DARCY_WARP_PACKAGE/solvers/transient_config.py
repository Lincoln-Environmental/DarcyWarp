"""Canonical validated 2D transient-unconfined production configuration.

Shared by public solver users and the transient replay harness.
These defaults are moved unchanged from working_tests/transient_replay_settings.py.
The warm-start artifact mode belongs to the replay harness; coupling
callers supply their own accepted head explicitly.
"""

DEFAULT_MIN_SAT = 0.1
STORAGE_REFERENCE_CURRENT_PICARD = "current_picard"


def default_solve_controls() -> dict:
    """Default Warp solve controls for the transient replay (kcycle)."""
    return {
        "max_cycles": 200,
        "max_levels": 4,
        "min_coarse_cells": 500,
        "nu_pre": 1,
        "nu_post": 1,
        "nu_coarse": 1,
        "check_every_no": 1,
        "max_outer_iterations": 100,
        "hclose": 1.0e-4,
        "rel_tol": 5.0e-7,
        "abs_tol_min": 5.0e-7,
        "dh_rms_tol": 1.0e-4,
        # Host-fallback-only knobs: residual_floor_tol, the chebyshev_* outer
        # acceleration controls, inner_forcing_eta,
        # transmissivity_relaxation_enabled, unconfined_pre_solve_iterations,
        # and initial_saturated_thickness are consumed by the host Picard solve
        # (use_device_transient_fast_path=False). The production device fast
        # path does not read them — it drives smoothing through "smoother" /
        # "cheby_lambda_*" and the adaptive_inner_* controller below.
        "residual_floor_tol": 1.0e-4,
        "smoother": "chebyshev",
        "omega": 0.7,
        "omega_min": 0.1,
        "omega_max": 0.9,
        "chebyshev_enabled": True,
        "chebyshev_order": 3,
        "cheby_lambda_min": 0.1,
        "cheby_lambda_max": 2.0,
        "chebyshev_reset_factor": 1.2,
        "chebyshev_rejection_factor": 1.2,
        "inner_forcing_eta": 0.10,
        "inner_head_residual_tol_min": 2.5e-6,
        "inner_head_residual_tol_max": 2.0e-4,
        "inner_picard_scale_max_fraction": 0.10,
        "transmissivity_relaxation_enabled": False,
        "unconfined_startup_mode": "confined_pre_solve",
        "unconfined_pre_solve_iterations": 3,
        "min_saturated_thickness": DEFAULT_MIN_SAT,
        "initial_saturated_thickness": 100.0,
        "max_head_change_per_outer_iteration": 10.0,
        "practical_picard_acceptance_enabled": True,
        "strict_head_residual_tol": 1.0e-6,
        # Practical acceptance is a FALLBACK, not the normal path: it may only
        # fire after strict Picard has had a genuine chance. Strict converges
        # in 10-12 outer iterations on the 1000x1000 production case (dh_max
        # contracts ~0.31x/iteration); the old value (8) short-circuited every
        # period ~3 iterations before strict success and was the root cause of
        # the 1M-cell accuracy failure (RMSE 0.0025 -> 5.5e-05 when fixed).
        "min_practical_outer_iterations": 20,
        "practical_head_residual_tol": 1.0e-5,
        "practical_residual_tol": 1.0e-5,
        "practical_dh_rms_tol": 3.0e-3,
        "practical_storage_diag_change_rms_tol": 30.0,
        "unconfined_inner_max_cycles_early": 10,
        "unconfined_inner_max_cycles_middle": 20,
        "unconfined_inner_max_cycles_late": 40,
        "unconfined_inner_middle_dh": 1.0,
        "unconfined_inner_late_dh": 1.0e-2,
        "adaptive_unconfined_inner_enabled": True,
        "adaptive_inner_initial_block_cycles": 5,
        "adaptive_inner_min_block_cycles": 5,
        "adaptive_inner_max_block_cycles": 20,
        "adaptive_inner_min_total_cycles": 5,
        "adaptive_inner_eta_initial": 0.05,
        "adaptive_inner_eta_min": 0.005,
        "adaptive_inner_eta_max": 0.10,
        "adaptive_inner_eta_gamma": 0.25,
        "adaptive_inner_eta_power": 1.5,
        "adaptive_inner_good_contraction_ratio": 0.40,
        "adaptive_inner_weak_contraction_ratio": 0.90,
        "adaptive_inner_stall_contraction_ratio": 0.9995,
        "adaptive_inner_divergence_contraction_ratio": 1.10,
        "adaptive_inner_stall_patience": 8,
        "adaptive_inner_minimum_usable_reduction_ratio": 0.10,
        "adaptive_inner_residual_floor": 1.0e-12,
        "adaptive_inner_relative_flow_residual_target": 1.0e-4,
        "adaptive_inner_save_block_history": False,
        "allow_unaccepted_transient_period": False,
        "use_device_transient_fast_path": True,
        # Incremental Picard: solve the inner system for the correction
        # A*delta = r^k (delta=0 on Dirichlet cells) instead of the full head.
        # Default False until 1M-cell validation confirms the improvement.
        "use_incremental_picard": False,
        # Adaptive dt is a MIKE-SHE-style safety net, not the primary path.
        # With the corrected strict budget below, full-dt strict Picard passes
        # in <=12 outer iterations on the production case, so the net is a
        # verified no-op there (single full-dt sub-step per period, identical
        # heads). It only shrinks dt when strict genuinely fails within the
        # budget, with practical acceptance as the floor at dt_min.
        "adaptive_dt_enabled": True,
        "adaptive_dt_min_fraction": 0.0625,
        "adaptive_dt_shrink_factor": 0.5,
        "adaptive_dt_grow_factor": 2.0,
        # Must exceed the natural full-dt strict iteration count (~12 max on
        # the 1000x1000 production case). The old value (6) sat below it and
        # guaranteed shrink/retry storms on every period.
        "adaptive_dt_strict_max_outer": 20,
        "adaptive_dt_max_growth_steps": 2,
        # Failure-path economics. Early shrink: once min_outer iterations give
        # a reliable dh contraction estimate, project iterations-to-hclose and
        # shrink immediately when the projection cannot make the remaining
        # budget (avoids paying the full budget on a doomed sub-step).
        # Extension: at budget exhaustion, if dh_max is within
        # extension_factor of hclose and still contracting, grant one extension
        # of extension_max_outer iterations instead of shrinking (cheaper than
        # a full retry). Neither fires on the production case (strict passes
        # well inside budget); they only price the genuinely-hard-case path.
        "adaptive_dt_early_shrink_enabled": True,
        "adaptive_dt_early_shrink_min_outer": 6,
        # Pessimistic-projection hysteresis: early-iteration contraction is
        # often pessimistic (it accelerates as the Picard iterate settles), so
        # the projection must say "won't make budget" on this many consecutive
        # checks before the driver actually shrinks dt. Prevents misfires on
        # hard-but-convergent periods (observed on the 1M hard-T case: strict
        # lands at 15-16 within budget 20 while the iteration-6 projection
        # still says no).
        "adaptive_dt_early_shrink_patience": 3,
        "adaptive_dt_extension_enabled": True,
        "adaptive_dt_extension_factor": 5.0,
        "adaptive_dt_extension_max_outer": 4,
        "adaptive_dt_extension_contraction_ratio": 0.8,
    }


def production_secant_sy_settings() -> dict:
    return {
        "solve_controls": default_solve_controls(),
        "unconfined_storage_mode": "mf6_convertible_secant_sy",
        "storage_reference": STORAGE_REFERENCE_CURRENT_PICARD,
        "warm_start_mode": "unconfined_steady_mf6",
    }
