#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Thin production harness for the 2D transient unconfined MF6 replay.

Case setup (grid, periods, storage, recharge, warm start, T field) is owned by
``build_case_setup()`` below; solve controls come from
``working_tests/transient_replay_settings.py`` via
``production_secant_sy_settings()`` and are intentionally not duplicated here.
The K-cycle tuning sweep lives in ``working_tests/optimize_2d_transient_kcycle.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from working_tests.transient_artifacts import (  # noqa: E402
    FORMULATION_CONFINED,
    FORMULATION_UNCONFINED,
    WARM_START_CONFINED_STEADY_MF6,
    validate_transient_artifact,
)
from working_tests.transient_replay_settings import (  # noqa: E402
    default_solve_controls,
    default_run_config,
    production_secant_sy_settings,
)
from working_tests.transient_replay_support import main as run_transient_replay  # noqa: E402


def build_grid_artifact_path(
    *,
    formulation: str,
    nx: int,
    ny: int,
    n_weeks: int,
    t_field_kind: str,
    t_field_seed: int,
    ghb_conductance_mode: str = "none",
) -> Path:
    """
    Build the grid-qualified MF6 truth artifact path used by the generator.

    Heterogeneous-T cases get a ``_ugly_t_s<seed>`` suffix so they never
    collide with the legacy homogeneous-artifact directories at the same grid.
    """
    if int(n_weeks) <= 0:
        raise ValueError("n_weeks must be positive.")
    t_field_kind = str(t_field_kind).strip().lower()
    suffix = f"_ugly_t_s{int(t_field_seed)}" if t_field_kind == "ugly_t" else ""
    ghb_mode = str(ghb_conductance_mode).strip().lower()
    if ghb_mode != "none":
        suffix += f"_ghb_{ghb_mode}"
    return data_store.joinpath(
        "working_tests",
        f"mf6_transient_2d_{formulation}_{int(nx)}x{int(ny)}_{int(n_weeks)}w{suffix}",
        "mf6_transient_heads.npz.lzma",
    )


def build_case_setup(
    *,
    formulation: str = FORMULATION_UNCONFINED,
    nx: int = 1000,
    ny: int = 1000,
    n_periods: int = 30,
    dx: float = 100.0,
    t_field_kind: str = "ugly_t",
    t_field_seed: int = 42,
    ghb_conductance_mode: str = "none",
) -> dict:
    """
    Single source of truth for the transient replay case.

    Defaults adopt the hard heterogeneous-T example from the confined
    steady-state benchmarks (``model_builder.make_ugly_T_field``, K = T / 100 m
    following the ``export_mf6_truth_npz`` convention). The generator script
    ``run_2d_transient_vs_mf6.py`` pulls this setup when run standalone, and
    ``ensure_case_artifact`` generates the MF6 artifact when it is missing.
    """
    formulation = str(formulation).strip().lower()
    if formulation not in {FORMULATION_CONFINED, FORMULATION_UNCONFINED}:
        raise ValueError("formulation must be 'confined' or 'unconfined'.")
    warm_start_mode = (
        WARM_START_CONFINED_STEADY_MF6
        if formulation == FORMULATION_CONFINED
        else "unconfined_steady_mf6"
    )
    setup = {
        "formulation": formulation,
        "nx": int(nx),
        "ny": int(ny),
        "dx": float(dx),
        "n_periods": int(n_periods),
        "dt_days": 7.0,
        "sy": 0.10,
        "ss": 1.0e-5,
        "annual_recharge_m": 0.3,
        "recharge_schedule_weeks": 52,
        "initial_saturated_thickness": 100.0,
        "t_field_kind": str(t_field_kind).strip().lower(),
        "t_field_seed": int(t_field_seed),
        "warm_start_mode": warm_start_mode,
        "ghb_conductance_mode": str(ghb_conductance_mode).strip().lower(),
    }
    setup["artifact_path"] = build_grid_artifact_path(
        formulation=formulation,
        nx=nx,
        ny=ny,
        n_weeks=n_periods,
        t_field_kind=setup["t_field_kind"],
        t_field_seed=setup["t_field_seed"],
        ghb_conductance_mode=setup["ghb_conductance_mode"],
    )
    return setup


def production_solver_backend(*, formulation: str) -> str:
    """Return the production backend paired with a replay formulation."""
    formulation = str(formulation).strip().lower()
    if formulation == FORMULATION_CONFINED:
        return "confined_kcycle"
    if formulation == FORMULATION_UNCONFINED:
        return "unconfined_picard_kcycle"
    raise ValueError("formulation must be 'confined' or 'unconfined'.")


def _artifact_scalar(artifact: dict, name: str) -> object:
    value = artifact.get(name)
    if value is None:
        return None
    return np.asarray(value).reshape(())


def case_setup_mismatches(artifact: dict, case_setup: dict) -> list[str]:
    """Return the ways an artifact's recorded physics differ from the request.

    The artifact path encodes only grid/periods/T-kind/seed/GHB mode, so the
    remaining equation inputs (dx, storage, recharge, initial thickness, warm
    start) must be compared against the artifact's own recorded metadata
    before reuse.  An empty list means the artifact matches the requested case.
    """
    mismatches: list[str] = []

    int_fields = ("nx", "ny", "n_periods", "t_field_seed", "recharge_schedule_weeks")
    for name in int_fields:
        if name not in case_setup:
            continue
        recorded = _artifact_scalar(artifact, name)
        if recorded is None or int(recorded) != int(case_setup[name]):
            mismatches.append(f"{name}: artifact={recorded} requested={case_setup[name]}")

    float_fields = ("dx", "sy", "ss", "dt_days", "initial_saturated_thickness")
    for name in float_fields:
        if name not in case_setup:
            continue
        recorded = _artifact_scalar(artifact, name)
        if recorded is None or float(recorded) != float(case_setup[name]):
            mismatches.append(f"{name}: artifact={recorded} requested={case_setup[name]}")

    str_fields = ("formulation", "t_field_kind", "warm_start_mode", "ghb_conductance_mode")
    for name in str_fields:
        if name not in case_setup:
            continue
        recorded = _artifact_scalar(artifact, name)
        recorded_str = None if recorded is None else str(recorded).strip().lower()
        requested_str = str(case_setup[name]).strip().lower()
        if recorded_str != requested_str:
            mismatches.append(f"{name}: artifact={recorded_str} requested={requested_str}")

    # annual_recharge_m is only recorded in the provenance JSON block.
    provenance = artifact.get("provenance")
    if provenance is not None and "annual_recharge_m" in case_setup:
        try:
            provenance_data = json.loads(str(np.asarray(provenance).reshape(())))
        except (TypeError, ValueError, json.JSONDecodeError):
            provenance_data = {}
        recorded_recharge = provenance_data.get("annual_recharge_m")
        if recorded_recharge is not None and float(recorded_recharge) != float(case_setup["annual_recharge_m"]):
            mismatches.append(
                f"annual_recharge_m: artifact={recorded_recharge} "
                f"requested={case_setup['annual_recharge_m']}"
            )
    return mismatches


def ensure_case_artifact(case_setup: dict) -> Path:
    """
    Return the case artifact path, generating the MF6 truth when missing.

    An existing artifact is reused only when it is internally valid AND its
    recorded physics match the requested setup (the path does not encode dx,
    storage, recharge, or the initial saturated thickness); otherwise it is
    regenerated on this automatic path.

    The generator is imported lazily so the replay stays Flopy-free until a
    truth artifact actually has to be built.
    """
    artifact_path = Path(case_setup["artifact_path"])
    if artifact_path.exists():
        try:
            artifact = validate_transient_artifact(artifact_path)
            mismatches = case_setup_mismatches(artifact, case_setup)
            if mismatches:
                raise ValueError(
                    "requested case differs from the artifact's recorded physics: "
                    + "; ".join(mismatches)
                )
            return artifact_path
        except (OSError, ValueError) as exc:
            print(f"Existing MF6 artifact failed validation and will be regenerated: {exc}")
    from working_tests import run_2d_transient_vs_mf6 as truth_generator

    print(f"MF6 truth artifact missing; generating (this can take a while):")
    print(f"  {artifact_path}")
    truth_generator.main(
        nx=case_setup["nx"],
        ny=case_setup["ny"],
        dx=case_setup["dx"],
        sy=case_setup["sy"],
        ss=case_setup["ss"],
        n_weeks=case_setup["n_periods"],
        annual_recharge_m=case_setup["annual_recharge_m"],
        recharge_schedule_weeks=case_setup["recharge_schedule_weeks"],
        initial_saturated_thickness=case_setup["initial_saturated_thickness"],
        t_field_kind=case_setup["t_field_kind"],
        t_field_seed=case_setup["t_field_seed"],
        ghb_conductance_mode=case_setup["ghb_conductance_mode"],
        out_path=artifact_path,
        reuse_existing_warm_start=True,
        warm_start_mode=case_setup["warm_start_mode"],
        formulation=case_setup["formulation"],
    )
    if not artifact_path.exists():
        raise RuntimeError(f"MF6 truth generation did not produce {artifact_path}")
    validate_transient_artifact(artifact_path)
    return artifact_path


def run_production_replay(
    *,
    formulation: str = FORMULATION_UNCONFINED,
    artifact_path: Path | None,
    workspace: Path | None,
    device: str,
    solve_control_overrides: dict | None = None,
    diag_preconditioner_backend: str = "device",
    allow_warm_start_mismatch: bool = False,
    solver_backend: str = "unconfined_picard_kcycle",
    implementation: str = "classic",
    use_device_transient_fast_path: bool | None = None,
    transient_face_operator_enabled: bool | None = None,
    transient_face_graphs_enabled: bool | None = None,
    transient_mixed_precision_enabled: bool | None = None,
) -> dict:
    """
    Run the production replay with the selected artifact.

    Controls are the locked production settings from
    ``production_secant_sy_settings()``; ``solve_control_overrides`` exists for
    genuine one-off experiments (e.g. the K-cycle tuning sweep) only.
    """
    formulation = str(formulation).strip().lower()
    if formulation == FORMULATION_UNCONFINED:
        production_settings = production_secant_sy_settings()
    elif formulation == FORMULATION_CONFINED:
        production_settings = {
            "solve_controls": default_solve_controls(),
            "unconfined_storage_mode": None,
            "storage_reference": None,
            "warm_start_mode": WARM_START_CONFINED_STEADY_MF6,
        }
    else:
        raise ValueError("formulation must be 'confined' or 'unconfined'.")
    if formulation == FORMULATION_CONFINED and solver_backend == "unconfined_picard_kcycle":
        solver_backend = "confined_kcycle"
    solve_controls = dict(production_settings["solve_controls"])
    solve_controls.update(solve_control_overrides or {})
    transient_switches = {
        "use_device_transient_fast_path": use_device_transient_fast_path,
        "transient_face_operator_enabled": transient_face_operator_enabled,
        "transient_face_graphs_enabled": transient_face_graphs_enabled,
        "transient_mixed_precision_enabled": transient_mixed_precision_enabled,
    }
    solve_controls.update({key: value for key, value in transient_switches.items() if value is not None})
    run_config = default_run_config(device=device)
    return run_transient_replay(
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
        warm_start_mode=production_settings["warm_start_mode"],
        formulation=formulation,
        solve_controls=solve_controls,
        unconfined_storage_mode=production_settings["unconfined_storage_mode"],
        storage_reference=production_settings["storage_reference"],
        allow_warm_start_mismatch=allow_warm_start_mismatch,
        solver_backend=solver_backend,
        implementation=implementation,
        run_config=run_config,
    )


def run_configured_replay(
    *,
    formulation: str,
    nx: int,
    ny: int,
    n_periods: int,
    t_field_kind: str,
    t_field_seed: int,
    artifact_path: Path | None,
    workspace: Path | None,
    device: str,
    solver_backend: str,
    implementation: str,
    use_device_transient_fast_path: bool,
    transient_face_operator_enabled: bool,
    transient_face_graphs_enabled: bool,
    transient_mixed_precision_enabled: bool,
) -> dict:
    """Build the requested case and run its production replay."""
    case_setup = build_case_setup(
        formulation=formulation,
        nx=nx,
        ny=ny,
        n_periods=n_periods,
        t_field_kind=t_field_kind,
        t_field_seed=t_field_seed,
    )
    selected_artifact_path = artifact_path or ensure_case_artifact(case_setup)
    return run_production_replay(
        formulation=formulation,
        artifact_path=selected_artifact_path,
        workspace=workspace,
        device=device,
        solver_backend=solver_backend,
        implementation=implementation,
        use_device_transient_fast_path=use_device_transient_fast_path,
        transient_face_operator_enabled=transient_face_operator_enabled,
        transient_face_graphs_enabled=transient_face_graphs_enabled,
        transient_mixed_precision_enabled=transient_mixed_precision_enabled,
    )


if __name__ == "__main__":
    # Adjust run settings here instead of passing command-line arguments.
    # Formulation options: "unconfined" or "confined".
    formulation = FORMULATION_UNCONFINED
    nx = 1000  # Number of grid cells in the x direction.
    ny = 1000  # Number of grid cells in the y direction.
    n_periods = 30  # Number of weekly transient periods to replay.

    # Available T fields: "ugly_t" (hard heterogeneous benchmark) or
    # "homogeneous" (legacy uniform K=100 m/day).
    t_field_kind = "ugly_t"
    t_field_seed = 42  # Random seed used by the "ugly_t" field.

    # None automatically validates/reuses or generates the matching MF6
    # artifact. Set an explicit Path to bypass automatic case-artifact lookup.
    artifact_path = None
    workspace = None  # None uses the replay's default workspace handling.

    # Device options: "auto", "cuda:0" (or another CUDA device), or "cpu".
    device = "auto"

    # Canonical solver backends:
    #   confined: "confined_kcycle" (transient-capable fixed-T solver)
    #   "unconfined_picard_kcycle" (production default)
    #   "unconfined_semismooth_newton_kcycle" (production alternative)
    #   "unconfined_fas" (experimental)
    # Legacy aliases include "picard", "picard_kcycle", and "kcycle".
    # Choose a backend compatible with the formulation above.
    solver_backend = production_solver_backend(formulation=formulation)

    # Confined transient implementation: "classic" or "fast".
    # "fast" uses the FP64 face-array transient K-cycle.
    implementation = "fast"

    # Unconfined transient production switches. The mixed path requires
    # DARCY_FLOAT=float64, the face operator, and the device fast path.
    use_device_transient_fast_path = True
    transient_face_operator_enabled = True
    transient_face_graphs_enabled = True
    transient_mixed_precision_enabled = True

    run_configured_replay(
        formulation=formulation,
        nx=nx,
        ny=ny,
        n_periods=n_periods,
        t_field_kind=t_field_kind,
        t_field_seed=t_field_seed,
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        solver_backend=solver_backend,
        implementation=implementation,
        use_device_transient_fast_path=use_device_transient_fast_path,
        transient_face_operator_enabled=transient_face_operator_enabled,
        transient_face_graphs_enabled=transient_face_graphs_enabled,
        transient_mixed_precision_enabled=transient_mixed_precision_enabled,
    )
