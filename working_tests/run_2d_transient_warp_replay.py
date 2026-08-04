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

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from working_tests.transient_artifacts import (  # noqa: E402
    FORMULATION_UNCONFINED,
    validate_transient_artifact,
)
from working_tests.transient_replay_settings import (  # noqa: E402
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
    formulation = FORMULATION_UNCONFINED
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
        "warm_start_mode": "unconfined_steady_mf6",
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
    artifact_path: Path | None,
    workspace: Path | None,
    device: str,
    solve_control_overrides: dict | None = None,
    diag_preconditioner_backend: str = "device",
    allow_warm_start_mismatch: bool = False,
    solver_backend: str = "unconfined_picard_kcycle",
) -> dict:
    """
    Run the production replay with the selected artifact.

    Controls are the locked production settings from
    ``production_secant_sy_settings()``; ``solve_control_overrides`` exists for
    genuine one-off experiments (e.g. the K-cycle tuning sweep) only.
    """
    production_settings = production_secant_sy_settings()
    solve_controls = dict(production_settings["solve_controls"])
    solve_controls.update(solve_control_overrides or {})
    run_config = default_run_config(device=device)
    return run_transient_replay(
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
        warm_start_mode=production_settings["warm_start_mode"],
        formulation=FORMULATION_UNCONFINED,
        solve_controls=solve_controls,
        unconfined_storage_mode=production_settings["unconfined_storage_mode"],
        storage_reference=production_settings["storage_reference"],
        allow_warm_start_mismatch=allow_warm_start_mismatch,
        solver_backend=solver_backend,
        run_config=run_config,
    )


def main() -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--t-field-kind",
        choices=("ugly_t", "homogeneous"),
        default="ugly_t",
        help="transmissivity field: 'ugly_t' = hard heterogeneous benchmark field "
             "(model_builder.make_ugly_T_field, K = T/100 m); "
             "'homogeneous' = legacy uniform K=100 m/day",
    )
    parser.add_argument("--t-field-seed", type=int, default=42, help="ugly_t field random seed")
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=1000)
    parser.add_argument("--n-periods", type=int, default=30)
    parser.add_argument("--artifact", default=None, help="explicit artifact path (bypasses the case setup)")
    parser.add_argument("--workspace", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--solver",
        default="unconfined_picard_kcycle",
        help="solver backend (e.g. unconfined_picard_kcycle, unconfined_fas, unconfined_semismooth_newton_kcycle)"
    )
    args = parser.parse_args()

    case_setup = build_case_setup(
        nx=args.nx,
        ny=args.ny,
        n_periods=args.n_periods,
        t_field_kind=args.t_field_kind,
        t_field_seed=args.t_field_seed,
    )
    artifact_path = Path(args.artifact) if args.artifact is not None else ensure_case_artifact(case_setup)
    workspace = Path(args.workspace) if args.workspace else None
    return run_production_replay(
        artifact_path=artifact_path,
        workspace=workspace,
        device=args.device,
        solver_backend=args.solver,
    )


if __name__ == "__main__":
    main()
