#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Production 2D transient unconfined Warp-vs-MF6 replay support.

This module keeps working-test artifact loading, comparison, reporting, and
mass-balance orchestration around the production secant-Sy solver API in
``WarpDarcySolver.solve_transient_2d_unconfined``. Legacy replay experiments are
kept out of this wrapper; lower-level storage helper formulas remain tested in
their own working-test modules.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("DARCY_FLOAT", "float64")

from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from working_tests.transient_artifacts import (
    FORMULATION_UNCONFINED,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    WARM_START_ARTIFACT_INITIAL,
    WARM_START_UNCONFINED_STEADY_MF6,
    artifact_warm_start_provenance,
    default_artifact_path,
    load_transient_artifact,
    require_matching_artifact_formulation,
    select_artifact_warm_start,
    spatial_fields_from_artifact,
    validate_warm_start_comparability,
    validate_warm_start_head,
)
from working_tests.transient_replay_mass_balance import (
    annotate_mass_balance_classification,
    classify_replay_mass_balance,
    compute_replay_mass_balance,
    save_warp_storage_budget_terms,
    storage_budget_arrays_from_warp_result,
)
from working_tests.transient_replay_storage import (
    _initial_transmissivity,
)
from working_tests.transient_replay_metrics import (
    _field_stats,
    _head_metrics,
    _sat_ref_summary,
    _scalar,
    _summarize_last_info,
    _summarize_period_head_stats,
    _summarize_period_infos,
    compare_transient,
    save_summary,
)
from working_tests.transient_replay_reporting import (
    _print_production_report,
    build_performance_summary,
    build_production_acceptance,
    evaluate_head_accuracy,
    evaluate_method_settings,
    print_cumulative_mass_balance,
    print_mass_balance_table,
)
from working_tests.transient_replay_settings import (
    DIAGNOSTICS_RUN_MODE,
    PRODUCTION_RUN_MODE,
    STORAGE_ACTIVE_SET_NONE,
    DEFAULT_MIN_SAT,
    STORAGE_REFERENCE_CURRENT_PICARD,
    STORAGE_TOP_THRESHOLD_GE,
    default_run_config,
    default_solve_controls,
    production_secant_sy_settings,
)


def _warp_solver_class():
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    return WarpDarcySolver


def _warp_device(preferred: str = "auto") -> str:
    import warp as wp

    if preferred != "auto":
        return preferred
    try:
        return "cuda:0" if wp.is_cuda_available() else "cpu"
    except AttributeError:
        return "cuda:0"


def run_warp_transient_replay(
    spatial: dict,
    recharge_rates: np.ndarray,
    sy: float,
    dt: float,
    ss: float | None = None,
    n_periods: int | None = None,
    device: str = "auto",
    diag_preconditioner_backend: str = "auto",
    min_sat: float = DEFAULT_MIN_SAT,
    solve_controls: dict | None = None,
    warm_start_mode: str = WARM_START_ARTIFACT_INITIAL,
    warm_start_head: np.ndarray | None = None,
    formulation: str = FORMULATION_UNCONFINED,
    unconfined_storage_mode: str = UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    storage_reference: str = STORAGE_REFERENCE_CURRENT_PICARD,
    storage_top_threshold: str = STORAGE_TOP_THRESHOLD_GE,
    storage_active_set_strategy: str = STORAGE_ACTIVE_SET_NONE,
    storage_hysteresis_eps: float = 0.0,
    storage_freeze_after_stable_iterations: int = 0,
    storage_freeze_after_outer: int | None = None,
    storage_switch_fraction_tol: float = 0.0,
    _warp_solver_class_fn=None,
    _warp_device_fn=None,
) -> dict:
    """Run the production 2D unconfined transient replay solver path."""
    formulation = str(formulation).strip().lower()
    if formulation != FORMULATION_UNCONFINED:
        raise ValueError("transient replay support now only carries the 2D unconfined path")
    storage_reference = str(storage_reference).strip().lower()
    if storage_reference != STORAGE_REFERENCE_CURRENT_PICARD:
        raise ValueError("transient replay support now requires storage_reference='current_picard'")
    storage_top_threshold = str(storage_top_threshold).strip().lower()
    if storage_top_threshold != STORAGE_TOP_THRESHOLD_GE:
        raise ValueError("transient replay support now requires storage_top_threshold='ge'")

    warm_start_mode = str(warm_start_mode).strip().lower()
    allowed_warm_start_modes = {
        WARM_START_ARTIFACT_INITIAL,
        WARM_START_UNCONFINED_STEADY_MF6,
    }
    if warm_start_mode not in allowed_warm_start_modes:
        raise ValueError(
            "transient replay support now allows only artifact_initial synthetic starts "
            "or unconfined_steady_mf6 artifact replay starts"
        )

    unconfined_storage_mode = str(unconfined_storage_mode).strip().lower()
    if unconfined_storage_mode != UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY:
        raise ValueError(
            "transient replay support now requires "
            "unconfined_storage_mode='mf6_convertible_secant_sy'"
        )
    storage_active_set_strategy = str(storage_active_set_strategy).strip().lower()
    if storage_active_set_strategy != STORAGE_ACTIVE_SET_NONE:
        raise ValueError("transient replay support now requires storage_active_set_strategy='none'")

    controls = dict(default_solve_controls())
    if solve_controls:
        controls.update(solve_controls)
    controls.setdefault("save_transient_diagnostics", True)

    nx = int(spatial["nx"])
    ny = int(spatial["ny"])
    dx = float(spatial["dx"])
    active = np.asarray(spatial["active"], dtype=np.int32)
    bc_mask = np.asarray(spatial["bc_mask"], dtype=np.int32)
    bc_values = np.asarray(spatial["bc_values"], dtype=np.float64)
    top = np.asarray(spatial["top"], dtype=np.float64)
    bottom = np.asarray(spatial["bottom"], dtype=np.float64)
    k = np.asarray(spatial["k"], dtype=np.float64)
    initial_head = np.asarray(spatial["initial_head"], dtype=np.float64)

    rates = np.asarray(recharge_rates, dtype=np.float64).reshape(-1)
    n_periods = int(rates.shape[0]) if n_periods is None else int(n_periods)
    if n_periods < 1:
        raise ValueError("n_periods must be >= 1.")
    if n_periods > rates.shape[0]:
        raise ValueError(f"n_periods={n_periods} exceeds available recharge rates ({rates.shape[0]}).")

    if warm_start_head is None:
        if warm_start_mode != WARM_START_ARTIFACT_INITIAL:
            raise ValueError(
                f"warm_start_mode='{warm_start_mode}' requires warm_start_head. "
                "Use run_replay_from_artifact so it can read the artifact warm-start head."
            )
        replay_start_head = validate_warm_start_head(
            head=initial_head,
            spatial=spatial,
            label="artifact initial_head",
        )
        warm_start_used = WARM_START_ARTIFACT_INITIAL
    else:
        replay_start_head = validate_warm_start_head(
            head=warm_start_head,
            spatial=spatial,
            label="warm_start_head",
        )
        warm_start_used = warm_start_mode

    warp_device_fn = _warp_device if _warp_device_fn is None else _warp_device_fn
    device = warp_device_fn(device)
    initial_transmissivity = _initial_transmissivity(
        k=k,
        initial_head=replay_start_head,
        top=top,
        bottom=bottom,
        active=active,
        min_sat=min_sat,
    )
    recharge_field = np.zeros((ny, nx), dtype=np.float64)

    solver_class_factory = _warp_solver_class if _warp_solver_class_fn is None else _warp_solver_class_fn
    WarpDarcySolver = solver_class_factory()
    with WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=dx,
        device=device,
        solver_type="kcycle",
        diag_preconditioner_backend=diag_preconditioner_backend,
    ) as solver:
        solver.build_from_fields(
            T_field=initial_transmissivity,
            R_field=recharge_field,
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
        )
        heads_api, info_api = solver.solve_transient_2d_unconfined(
            initial_head=replay_start_head,
            recharge_rates=rates[:n_periods],
            k_field=k,
            zbot_field=bottom,
            ztop_field=top,
            sy=float(sy),
            ss=float(0.0 if ss is None else ss),
            dt=float(dt),
            active=active,
            bc_mask=bc_mask,
            bc_values=bc_values,
            storage_mode=unconfined_storage_mode,
            storage_reference=storage_reference,
            storage_top_threshold=storage_top_threshold,
            storage_active_set_strategy=storage_active_set_strategy,
            storage_hysteresis_eps=float(storage_hysteresis_eps),
            storage_freeze_after_stable_iterations=int(storage_freeze_after_stable_iterations),
            storage_freeze_after_outer=storage_freeze_after_outer,
            storage_switch_fraction_tol=float(storage_switch_fraction_tol),
            solve_controls=controls,
            min_saturated_thickness=float(min_sat),
            return_info=True,
        )

    heads_per_period = np.asarray(heads_api, dtype=np.float64)
    storage_coeffs = np.asarray(info_api["storage_coeffs_per_period"], dtype=np.float64)
    storativity_kind = (
        "sy_plus_ss_secant_saturated_thickness"
        if unconfined_storage_mode == UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY
        else "unconfined_storativity"
    )
    solve_controls_used = dict(controls)
    solve_controls_used.update(
        {
            "storage_active_set_strategy": storage_active_set_strategy,
            "storage_hysteresis_eps": float(storage_hysteresis_eps),
            "storage_freeze_after_stable_iterations": int(storage_freeze_after_stable_iterations),
            "storage_freeze_after_outer": storage_freeze_after_outer,
            "storage_switch_fraction_tol": float(storage_switch_fraction_tol),
        }
    )
    return {
        "heads_per_period": heads_per_period,
        "heads_old_per_period": np.asarray(info_api["heads_old_per_period"], dtype=np.float64),
        "heads_final": np.asarray(info_api["heads_final"], dtype=np.float64),
        "storage_reference_heads_per_period": np.asarray(info_api["storage_reference_heads_per_period"], dtype=np.float64),
        "storage_coeffs_per_period": storage_coeffs,
        "sy_storage_coeffs_per_period": np.asarray(info_api["sy_storage_coeffs_per_period"], dtype=np.float64),
        "ss_storage_coeffs_per_period": np.asarray(info_api["ss_storage_coeffs_per_period"], dtype=np.float64),
        "storage_terms_per_period": np.asarray(info_api["storage_terms_per_period"], dtype=np.float64),
        "sy_storage_terms_per_period": np.asarray(info_api["sy_storage_terms_per_period"], dtype=np.float64),
        "ss_storage_terms_per_period": np.asarray(info_api["ss_storage_terms_per_period"], dtype=np.float64),
        "sy_crossing_volume_terms_per_period": np.asarray(info_api["sy_crossing_volume_terms_per_period"], dtype=np.float64),
        "period_infos": info_api["period_infos"],
        "last_info": info_api["last_info"],
        "period_times": np.asarray(info_api["period_times"], dtype=np.float64),
        "total_time": float(info_api["total_time"]),
        "n_periods": n_periods,
        "device": device,
        "storativity": storage_coeffs[-1],
        "storativity_kind": storativity_kind,
        "include_specific_storage": True,
        "unconfined_storage_mode": unconfined_storage_mode,
        "saturated_thickness_reference": None,
        "saturated_thickness_reference_source": storage_reference,
        "storage_reference": storage_reference,
        "storage_top_threshold": storage_top_threshold,
        "storage_active_set_strategy": storage_active_set_strategy,
        "storage_hysteresis_eps": float(storage_hysteresis_eps),
        "storage_freeze_after_stable_iterations": int(storage_freeze_after_stable_iterations),
        "storage_freeze_after_outer": storage_freeze_after_outer,
        "storage_switch_fraction_tol": float(storage_switch_fraction_tol),
        "dt": float(dt),
        "formulation": formulation,
        "solve_controls": solve_controls_used,
        "warm_start_mode": warm_start_mode,
        "warm_start_used": warm_start_used,
        "warm_start_head": replay_start_head.copy(),
        "storage_coeff": storage_coeffs[-1],
        "storage_coeff_kind": storativity_kind,
    }


def run_replay_from_artifact(
    artifact_path: str | Path,
    workspace: str | Path | None = None,
    device: str = "auto",
    diag_preconditioner_backend: str = "auto",
    solve_controls: dict | None = None,
    warm_start_mode: str = WARM_START_UNCONFINED_STEADY_MF6,
    formulation: str = FORMULATION_UNCONFINED,
    unconfined_storage_mode: str = UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    storage_reference: str = STORAGE_REFERENCE_CURRENT_PICARD,
    storage_top_threshold: str = STORAGE_TOP_THRESHOLD_GE,
    storage_active_set_strategy: str = STORAGE_ACTIVE_SET_NONE,
    storage_hysteresis_eps: float = 0.0,
    storage_freeze_after_stable_iterations: int = 0,
    storage_freeze_after_outer: int | None = None,
    storage_switch_fraction_tol: float = 0.0,
    allow_warm_start_mismatch: bool = False,
    run_config: dict | None = None,
    load_transient_artifact_fn=None,
    run_warp_transient_replay_fn=None,
) -> dict:
    """
    Load the MF6 truth artifact, replay Warp through every period, compare, save.

    Returns the full summary dict and writes ``transient_replay_summary.json``
    plus ``warp_transient_heads.npz`` under the workspace.

    By default the replay starts from the artifact's own
    ``unconfined_steady_head`` (``unconfined_steady_mf6``) and builds an explicit
    MF6-like secant-Sy storativity field
    (``mf6_convertible_secant_sy``) using ``storage_reference=current_picard``,
    the closest tested head match to MF6. Starting from a different warm start
    than the artifact used is rejected unless
    ``allow_warm_start_mismatch=True``.
    """
    artifact_path = Path(artifact_path)
    if workspace is None:
        workspace = artifact_path.parent
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    artifact_loader = load_transient_artifact if load_transient_artifact_fn is None else load_transient_artifact_fn
    replay_runner = run_warp_transient_replay if run_warp_transient_replay_fn is None else run_warp_transient_replay_fn

    artifact = artifact_loader(artifact_path)
    spatial = spatial_fields_from_artifact(artifact)
    spatial["workspace"] = workspace
    formulation = str(formulation).strip().lower()
    if formulation != FORMULATION_UNCONFINED:
        raise ValueError("transient replay support now only carries formulation='unconfined'")
    artifact_mode = require_matching_artifact_formulation(
        artifact=artifact,
        requested_formulation=formulation,
        artifact_path=artifact_path,
    )
    warm_start_mode = str(warm_start_mode).strip().lower()
    if warm_start_mode != WARM_START_UNCONFINED_STEADY_MF6:
        raise ValueError(
            "transient replay support now requires "
            "warm_start_mode='unconfined_steady_mf6'"
        )
    unconfined_storage_mode = str(unconfined_storage_mode).strip().lower()
    if unconfined_storage_mode != UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY:
        raise ValueError(
            "transient replay support now requires "
            "unconfined_storage_mode='mf6_convertible_secant_sy'"
        )
    storage_reference = str(storage_reference).strip().lower()
    if storage_reference != STORAGE_REFERENCE_CURRENT_PICARD:
        raise ValueError("transient replay support now requires storage_reference='current_picard'")
    storage_top_threshold = str(storage_top_threshold).strip().lower()
    if storage_top_threshold != STORAGE_TOP_THRESHOLD_GE:
        raise ValueError("transient replay support now requires storage_top_threshold='ge'")
    storage_active_set_strategy = str(storage_active_set_strategy).strip().lower()
    if storage_active_set_strategy != STORAGE_ACTIVE_SET_NONE:
        raise ValueError("transient replay support now requires storage_active_set_strategy='none'")

    artifact_warm_start = artifact_warm_start_provenance(artifact)
    validate_warm_start_comparability(
        artifact_warm_start=artifact_warm_start,
        warp_warm_start_mode=warm_start_mode,
        allow_warm_start_mismatch=allow_warm_start_mismatch,
    )
    warm_start_head, warm_start_used = select_artifact_warm_start(
        artifact=artifact,
        spatial=spatial,
        warm_start_mode=warm_start_mode,
    )

    sy = float(artifact["sy"])
    ss = float(artifact["ss"])
    dt = float(artifact["dt_days"])
    recharge_rates = np.asarray(artifact["recharge_rates"], dtype=np.float64)
    n_periods = int(recharge_rates.shape[0])

    print(f"Transient replay: {spatial['nx']}x{spatial['ny']}, {n_periods} periods, dt={dt}")
    print(
        f"  formulation: unconfined; storativity S = secant(Sy crossing) + Ss*saturated_thickness "
        f"(Sy={sy}, Ss={ss}, reference={storage_reference}, "
        f"active_set_strategy={storage_active_set_strategy})"
    )
    print(f"  warm start: {warm_start_used} (artifact provenance: {artifact_warm_start})")
    print(f"  artifact: {artifact_path}")

    # Effective MF6 replay settings (caller overrides applied on top of the
    # declared defaults). Printed explicitly so the direct replay header and the
    # winning variant cannot silently drift apart again.
    effective_controls = dict(default_solve_controls())
    if solve_controls:
        effective_controls.update(solve_controls)
    effective_startup_mode = str(
        effective_controls.get("unconfined_startup_mode", "confined_pre_solve")
    )
    mass_balance_min_sat = float(effective_controls.get("min_saturated_thickness", DEFAULT_MIN_SAT))
    if run_config is None:
        run_config = default_run_config(device=device)
    else:
        merged = default_run_config(device=device)
        merged.update({str(k): v for k, v in run_config.items()})
        run_config = merged
    print(f"  run_mode={run_config['run_mode']} device={run_config['device']} "
          f"compute_mass_balance={run_config['compute_mass_balance']} "
          f"profile_performance={run_config['profile_performance']} "
          f"save_heavy_diagnostics={run_config['save_heavy_diagnostics']} "
          f"run_replay_matrix={run_config['run_replay_matrix']}")
    print("  MF6 replay settings:")
    print(f"    unconfined_storage_mode={unconfined_storage_mode}")
    print(f"    storage_reference={storage_reference}")
    print(f"    storage_top_threshold={storage_top_threshold}")
    print(f"    storage_active_set_strategy={storage_active_set_strategy}")
    print(f"    unconfined_startup_mode={effective_startup_mode}")
    print(f"    warm_start={warm_start_used}")

    warp_result = replay_runner(
        spatial=spatial,
        recharge_rates=recharge_rates,
        sy=sy,
        ss=ss,
        dt=dt,
        n_periods=n_periods,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
        solve_controls=solve_controls,
        warm_start_mode=warm_start_used,
        warm_start_head=warm_start_head,
        formulation=formulation,
        unconfined_storage_mode=unconfined_storage_mode,
        storage_reference=storage_reference,
        storage_top_threshold=storage_top_threshold,
        storage_active_set_strategy=storage_active_set_strategy,
        storage_hysteresis_eps=storage_hysteresis_eps,
        storage_freeze_after_stable_iterations=storage_freeze_after_stable_iterations,
        storage_freeze_after_outer=storage_freeze_after_outer,
        storage_switch_fraction_tol=storage_switch_fraction_tol,
    )

    comparison = compare_transient(
        warp_result,
        np.asarray(artifact["heads_per_period"], dtype=np.float64),
        np.asarray(artifact["heads_final"], dtype=np.float64),
        spatial["active"],
    )
    mass_balance_runtime: float | None = None
    if run_config.get("compute_mass_balance", True):
        mass_balance_t0 = time.perf_counter()
        mass_balance = compute_replay_mass_balance(
            spatial=spatial,
            recharge_rates=recharge_rates,
            sy=sy,
            dt=dt,
            formulation=formulation,
            unconfined_storage_mode=warp_result["unconfined_storage_mode"],
            warp_result=warp_result,
            min_sat=mass_balance_min_sat,
        )
        mass_balance_runtime = float(time.perf_counter() - mass_balance_t0)
        annotate_mass_balance_classification(mass_balance)
    else:
        mass_balance = {
            "warp_mass_balance_available": False,
            "per_period": [],
            "cumulative": {},
            "worst_period": {},
            "max_abs_percent_discrepancy": None,
            "mass_balance_class": "not_computed",
            "mass_balance_passed": False,
            "cumulative_percent_discrepancy": None,
            "mass_balance_warnings": ["compute_mass_balance=False; mass balance not computed"],
            "mass_balance_failures": [],
        }

    # Quantify how far the Warp warm start is from the artifact's initial_head
    # (the head MF6 started period 0 from). Near-zero when Warp reuses the
    # artifact warm start; nonzero when Warp re-solves its own warm start.
    warm_start_vs_initial = _head_metrics(
        np.asarray(warp_result["warm_start_head"], dtype=np.float64),
        np.asarray(artifact["initial_head"], dtype=np.float64),
        spatial["active"],
    )

    warp_npz = workspace.joinpath("warp_transient_heads.npz")
    sat_ref_field = warp_result["saturated_thickness_reference"]
    storage_budget_arrays = storage_budget_arrays_from_warp_result(warp_result=warp_result)
    period_infos = warp_result.get("period_infos")
    if not isinstance(period_infos, list):
        period_infos = [warp_result["last_info"]]
    np.savez_compressed(
        warp_npz,
        heads_per_period=warp_result["heads_per_period"],
        heads_old_per_period=storage_budget_arrays["heads_old_per_period"],
        heads_final=warp_result["heads_final"],
        total_time=np.asarray(warp_result["total_time"], dtype=np.float64),
        period_times=warp_result["period_times"],
        last_info=np.asarray(json.dumps(warp_result["last_info"], default=str)),
        period_infos=np.asarray(json.dumps(period_infos, default=str)),
        warp_storativity=(
            np.asarray(warp_result["storativity"], dtype=np.float64)
            if warp_result["storativity"] is not None
            else np.asarray([], dtype=np.float64)
        ),
        warp_storativity_kind=np.asarray(warp_result["storativity_kind"]),
        include_specific_storage=np.asarray(warp_result["include_specific_storage"]),
        unconfined_storage_mode=np.asarray(
            warp_result["unconfined_storage_mode"]
            if warp_result["unconfined_storage_mode"] is not None
            else "none"
        ),
        storage_reference=np.asarray(warp_result.get("storage_reference", STORAGE_REFERENCE_CURRENT_PICARD)),
        storage_top_threshold=np.asarray(warp_result.get("storage_top_threshold", STORAGE_TOP_THRESHOLD_GE)),
        storage_active_set_strategy=np.asarray(
            warp_result.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE)
        ),
        storage_hysteresis_eps=np.asarray(
            warp_result.get("storage_hysteresis_eps", 0.0), dtype=np.float64
        ),
        storage_freeze_after_stable_iterations=np.asarray(
            warp_result.get("storage_freeze_after_stable_iterations", 0), dtype=np.int32
        ),
        storage_freeze_after_outer=np.asarray(
            -1 if warp_result.get("storage_freeze_after_outer") is None else warp_result.get("storage_freeze_after_outer"),
            dtype=np.int32,
        ),
        storage_switch_fraction_tol=np.asarray(
            warp_result.get("storage_switch_fraction_tol", 0.0), dtype=np.float64
        ),
        storage_reference_heads_per_period=np.asarray(storage_budget_arrays["storage_reference_heads_per_period"], dtype=np.float64),
        storage_coeffs_per_period=np.asarray(storage_budget_arrays["storage_coeffs_per_period"], dtype=np.float64),
        sy_storage_coeffs_per_period=np.asarray(storage_budget_arrays["sy_storage_coeffs_per_period"], dtype=np.float64),
        ss_storage_coeffs_per_period=np.asarray(storage_budget_arrays["ss_storage_coeffs_per_period"], dtype=np.float64),
        storage_terms_per_period=np.asarray(storage_budget_arrays["storage_terms_per_period"], dtype=np.float64),
        sy_storage_terms_per_period=np.asarray(storage_budget_arrays["sy_storage_terms_per_period"], dtype=np.float64),
        ss_storage_terms_per_period=np.asarray(storage_budget_arrays["ss_storage_terms_per_period"], dtype=np.float64),
        sy_crossing_volume_terms_per_period=np.asarray(storage_budget_arrays["sy_crossing_volume_terms_per_period"], dtype=np.float64),
        saturated_thickness_reference=(
            np.asarray(sat_ref_field, dtype=np.float64)
            if sat_ref_field is not None
            else np.asarray([], dtype=np.float64)
        ),
        dt=np.asarray(warp_result["dt"], dtype=np.float64),
        formulation=np.asarray(warp_result["formulation"]),
        device=np.asarray(warp_result["device"]),
        warm_start_mode=np.asarray(warp_result["warm_start_mode"]),
        warm_start_used=np.asarray(warp_result["warm_start_used"]),
        warm_start_head=np.asarray(warp_result["warm_start_head"], dtype=np.float64),
        # Deprecated aliases:
        storage_coeff=(
            np.asarray(warp_result["storativity"], dtype=np.float64)
            if warp_result["storativity"] is not None
            else np.asarray([], dtype=np.float64)
        ),
        storage_coeff_kind=np.asarray(warp_result["storativity_kind"]),
    )

    provenance = artifact.get("provenance")
    if provenance is not None:
        provenance = str(np.asarray(provenance).reshape(()))
    mf6_transient_total_time = _scalar(artifact, "total_time")
    mf6_transient_engine_time = _scalar(artifact, "engine_time")
    mf6_confined_steady_engine_time = _scalar(artifact, "confined_steady_engine_time")
    mf6_unconfined_steady_engine_time = _scalar(artifact, "unconfined_steady_engine_time")
    mf6_combined_engine_time = None
    if mf6_transient_engine_time is not None:
        mf6_combined_engine_time = mf6_transient_engine_time
        if mf6_confined_steady_engine_time is not None and np.isfinite(mf6_confined_steady_engine_time):
            mf6_combined_engine_time += mf6_confined_steady_engine_time
        if mf6_unconfined_steady_engine_time is not None and np.isfinite(mf6_unconfined_steady_engine_time):
            mf6_combined_engine_time += mf6_unconfined_steady_engine_time

    # Storativity is zero on inactive/boundary cells by convention, so report
    # field statistics over active non-boundary (free) cells only.
    free_mask = (np.asarray(spatial["active"], dtype=np.int32) != 0) & (
        np.asarray(spatial["bc_mask"], dtype=np.int32) == 0
    )

    solve_settings_recorded = dict(warp_result["solve_controls"])
    solve_settings_recorded.update(
        {
            "unconfined_storage_mode": warp_result["unconfined_storage_mode"],
            "storage_reference": warp_result.get("storage_reference", STORAGE_REFERENCE_CURRENT_PICARD),
            "storage_top_threshold": warp_result.get("storage_top_threshold", STORAGE_TOP_THRESHOLD_GE),
            "storage_active_set_strategy": warp_result.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE),
            "warm_start": warp_result["warm_start_used"],
        }
    )

    summary = {
        "artifact_path": str(artifact_path),
        "artifact_formulation": artifact_mode,
        "grid": {"nx": spatial["nx"], "ny": spatial["ny"], "dx": spatial["dx"]},
        "n_periods": n_periods,
        "dt": dt,
        "formulation": formulation,
        # Consolidated, explicit MF6 replay settings. The direct replay and the
        # winning ``mf6_secant_sy_current_picard_none`` variant must agree on these.
        "mf6_replay_settings": {
            "unconfined_storage_mode": warp_result["unconfined_storage_mode"],
            "storage_reference": warp_result.get("storage_reference", STORAGE_REFERENCE_CURRENT_PICARD),
            "storage_top_threshold": warp_result.get("storage_top_threshold", STORAGE_TOP_THRESHOLD_GE),
            "storage_active_set_strategy": warp_result.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE),
            "unconfined_startup_mode": warp_result["solve_controls"].get(
                "unconfined_startup_mode", "confined_pre_solve"
            ),
            "warm_start": warp_result["warm_start_used"],
        },
        "storage": {
            "sy": sy,
            "ss": ss,
            "unconfined_storage_mode": warp_result["unconfined_storage_mode"],
            "storage_reference": warp_result.get("storage_reference", STORAGE_REFERENCE_CURRENT_PICARD),
            "storage_top_threshold": warp_result.get("storage_top_threshold", STORAGE_TOP_THRESHOLD_GE),
            "storage_active_set_strategy": warp_result.get("storage_active_set_strategy", STORAGE_ACTIVE_SET_NONE),
            "storage_hysteresis_eps": warp_result.get("storage_hysteresis_eps", 0.0),
            "storage_freeze_after_stable_iterations": warp_result.get(
                "storage_freeze_after_stable_iterations", 0
            ),
            "storage_freeze_after_outer": warp_result.get("storage_freeze_after_outer"),
            "storage_switch_fraction_tol": warp_result.get("storage_switch_fraction_tol", 0.0),
            "warp_storativity": _field_stats(warp_result["storativity"], free_mask),
            "warp_storativity_kind": warp_result["storativity_kind"],
            "include_specific_storage": warp_result["include_specific_storage"],
            "saturated_thickness_reference": _sat_ref_summary(
                warp_result["saturated_thickness_reference"],
                warp_result["saturated_thickness_reference_source"],
                free_mask,
            ),
            # Deprecated aliases (use warp_storativity* above):
            "warp_storage_coeff": _field_stats(warp_result["storativity"], free_mask),
            "warp_storage_coeff_kind": warp_result["storativity_kind"],
        },
        "warm_start": {
            "requested": warm_start_mode,
            "used": warp_result["warm_start_used"],
            "source": warm_start_used,
            "artifact_provenance": artifact_warm_start,
            "allow_warm_start_mismatch": bool(allow_warm_start_mismatch),
            "warm_start_vs_initial_head": warm_start_vs_initial,
        },
        "timing": {
            "warp_total_time": float(warp_result["total_time"]),
            "mf6_total_time": mf6_transient_total_time,
            "mf6_engine_time": mf6_transient_engine_time,
            "mf6_transient_total_time": mf6_transient_total_time,
            "mf6_transient_engine_time": mf6_transient_engine_time,
            "mf6_confined_steady_engine_time": mf6_confined_steady_engine_time,
            "mf6_unconfined_steady_engine_time": mf6_unconfined_steady_engine_time,
            "mf6_engine_time_including_warm_start": mf6_combined_engine_time,
            "warp_period_time_mean": float(np.mean(warp_result["period_times"])),
            "warp_period_time_max": float(np.max(warp_result["period_times"])),
            "warp_period_time_sum": float(np.sum(warp_result["period_times"])),
            "warp_period_times": [float(value) for value in np.asarray(warp_result["period_times"], dtype=np.float64)],
            "warp_period_1_time": (
                float(np.asarray(warp_result["period_times"], dtype=np.float64)[0])
                if int(np.asarray(warp_result["period_times"]).size) > 0
                else None
            ),
            "warp_period_time_mean_excluding_period_1": (
                float(np.mean(np.asarray(warp_result["period_times"], dtype=np.float64)[1:]))
                if int(np.asarray(warp_result["period_times"]).size) > 1
                else None
            ),
        },
        "convergence": _summarize_last_info(warp_result["last_info"]),
        "period_convergence": _summarize_period_infos(period_infos),
        "warp_head_stats": _summarize_period_head_stats(
            heads_per_period=warp_result["heads_per_period"],
            active=spatial["active"],
            bc_mask=spatial["bc_mask"],
        ),
        "solve_settings": solve_settings_recorded,
        "device": warp_result["device"],
        "diag_preconditioner_backend": diag_preconditioner_backend,
        "comparison": comparison,
        "mass_balance": mass_balance,
        "mf6_provenance": provenance,
    }
    # --- Production acceptance, performance summary, run configuration. ---
    method_settings = evaluate_method_settings(
        unconfined_storage_mode=warp_result["unconfined_storage_mode"],
        storage_reference=storage_reference,
        storage_top_threshold=storage_top_threshold,
        storage_active_set_strategy=storage_active_set_strategy,
        unconfined_startup_mode=effective_startup_mode,
        warm_start=warm_start_used,
    )
    head_accuracy = evaluate_head_accuracy(comparison)
    summary["run_config"] = run_config
    summary["head_accuracy"] = head_accuracy
    summary["method_settings"] = method_settings
    summary["production_acceptance"] = build_production_acceptance(
        method_settings=method_settings,
        head_accuracy=head_accuracy,
        mass_balance=mass_balance,
        period_convergence=summary["period_convergence"],
    )
    summary["performance"] = build_performance_summary(
        timing=summary["timing"],
        period_convergence=summary["period_convergence"],
        solve_settings=solve_settings_recorded,
        mass_balance_runtime=mass_balance_runtime,
        profile=None,
    )

    save_heavy = bool(
        run_config.get("save_heavy_diagnostics", False)
        or run_config.get("run_mode") == DIAGNOSTICS_RUN_MODE
    )
    if save_heavy:
        warp_storage_budget_path = workspace.joinpath("warp_storage_budget_terms.npz")
        save_warp_storage_budget_terms(
            path=warp_storage_budget_path,
            warp_result=warp_result,
        )
        summary["storage_budget_artifact"] = str(warp_storage_budget_path)
    else:
        summary["storage_budget_artifact"] = None

    summary_path = workspace.joinpath("transient_replay_summary.json")
    save_summary(summary_path, summary)

    print(f"Warp transient heads saved to {warp_npz}")
    print(f"Replay summary saved to {summary_path}")
    final_max = comparison["final"]["max_abs_diff"]
    final_rmse = comparison["final"]["rmse"]
    print(
        f"Final vs MF6: max_abs_diff={final_max:.6g} m, rmse={final_rmse:.6g} m "
        f"(worst period {comparison['worst_period_number_one_based']})"
    )
    if run_config.get("compute_mass_balance", True):
        print_mass_balance_table(mass_balance)
        print_cumulative_mass_balance(mass_balance)
    _print_production_report(summary=summary)
    return summary


def main(
    artifact_path: str | Path | None = None,
    workspace: str | Path | None = None,
    device: str = "auto",
    diag_preconditioner_backend: str = "auto",
    warm_start_mode: str = WARM_START_UNCONFINED_STEADY_MF6,
    formulation: str = FORMULATION_UNCONFINED,
    unconfined_storage_mode: str = UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    storage_reference: str = STORAGE_REFERENCE_CURRENT_PICARD,
    storage_top_threshold: str = STORAGE_TOP_THRESHOLD_GE,
    storage_active_set_strategy: str = STORAGE_ACTIVE_SET_NONE,
    storage_hysteresis_eps: float = 0.0,
    storage_freeze_after_stable_iterations: int = 0,
    storage_freeze_after_outer: int | None = None,
    storage_switch_fraction_tol: float = 0.0,
    allow_warm_start_mismatch: bool = False,
    run_config: dict | None = None,
) -> dict:
    """
    Run the default transient replay against the MF6 truth artifact.

    If the artifact is missing, prints instructions to generate it rather than
    failing, so importing this module never requires MF6 to be installed.
    """
    formulation = str(formulation).strip().lower()
    if formulation != FORMULATION_UNCONFINED:
        raise ValueError("transient replay support now only carries formulation='unconfined'")
    artifact_path = (
        Path(artifact_path)
        if artifact_path is not None
        else default_artifact_path(formulation=formulation)
    )
    if not artifact_path.exists():
        print(f"MF6 transient artifact not found at {artifact_path}.")
        print(
            "Generate it first with:  python working_tests/run_2d_transient_vs_mf6.py "
            f"(formulation={formulation})"
        )
        return {"artifact_path": str(artifact_path), "ran": False}
    return run_replay_from_artifact(
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
        warm_start_mode=warm_start_mode,
        formulation=formulation,
        unconfined_storage_mode=unconfined_storage_mode,
        storage_reference=storage_reference,
        storage_top_threshold=storage_top_threshold,
        storage_active_set_strategy=storage_active_set_strategy,
        storage_hysteresis_eps=storage_hysteresis_eps,
        storage_freeze_after_stable_iterations=storage_freeze_after_stable_iterations,
        storage_freeze_after_outer=storage_freeze_after_outer,
        storage_switch_fraction_tol=storage_switch_fraction_tol,
        allow_warm_start_mismatch=allow_warm_start_mismatch,
        run_config=run_config,
    )


if __name__ == "__main__":
    # Configuration parameters
    artifact_path = None                 # defaults to the standard MF6 truth artifact
    workspace = None                     # defaults to the artifact's parent directory
    device = "auto"
    diag_preconditioner_backend = "device"
    # Default to the artifact's own unconfined steady head so the Warp-vs-MF6
    # comparison starts from a steady water table. Override to a Warp-resolved
    # warm start only with allow_warm_start_mismatch=True.
    production_settings = production_secant_sy_settings()
    warm_start_mode = production_settings["warm_start_mode"]
    formulation = FORMULATION_UNCONFINED
    unconfined_storage_mode = production_settings["unconfined_storage_mode"]
    storage_reference = production_settings["storage_reference"]
    storage_top_threshold = production_settings["storage_top_threshold"]
    storage_active_set_strategy = production_settings["storage_active_set_strategy"]
    storage_hysteresis_eps = production_settings.get("storage_hysteresis_eps", 0.0)
    storage_freeze_after_stable_iterations = production_settings.get("storage_freeze_after_stable_iterations", 0)
    storage_freeze_after_outer = production_settings["storage_freeze_after_outer"]
    storage_switch_fraction_tol = production_settings.get("storage_switch_fraction_tol", 0.0)
    allow_warm_start_mismatch = False
    # Explicit production configuration variables (no argparse required).
    #   RUN_MODE:              production | benchmark | diagnostics
    #   COMPUTE_MASS_BALANCE:  report per-period/cumulative mass balance
    #   PROFILE_PERFORMANCE:   record optional category timing (not yet instrumented)
    #   SAVE_HEAVY_DIAGNOSTICS: write per-cell storage-budget NPZ artifacts
    #   RUN_REPLAY_MATRIX:     run the full comparison-variant matrix
    RUN_MODE = PRODUCTION_RUN_MODE
    DEVICE = device
    COMPUTE_MASS_BALANCE = True
    PROFILE_PERFORMANCE = False
    SAVE_HEAVY_DIAGNOSTICS = False
    RUN_REPLAY_MATRIX = False
    run_config = default_run_config(
        run_mode=RUN_MODE,
        device=DEVICE,
        compute_mass_balance=COMPUTE_MASS_BALANCE,
        profile_performance=PROFILE_PERFORMANCE,
        save_heavy_diagnostics=SAVE_HEAVY_DIAGNOSTICS,
        run_replay_matrix=RUN_REPLAY_MATRIX,
    )
    main(
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
        warm_start_mode=warm_start_mode,
        formulation=formulation,
        unconfined_storage_mode=unconfined_storage_mode,
        storage_reference=storage_reference,
        storage_top_threshold=storage_top_threshold,
        storage_active_set_strategy=storage_active_set_strategy,
        storage_hysteresis_eps=storage_hysteresis_eps,
        storage_freeze_after_stable_iterations=storage_freeze_after_stable_iterations,
        storage_freeze_after_outer=storage_freeze_after_outer,
        storage_switch_fraction_tol=storage_switch_fraction_tol,
        allow_warm_start_mismatch=allow_warm_start_mismatch,
        run_config=run_config,
    )
