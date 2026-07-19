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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.project_base import data_store  # noqa: E402
from working_tests.transient_artifacts import FORMULATION_UNCONFINED  # noqa: E402
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
    }
    setup["artifact_path"] = build_grid_artifact_path(
        formulation=formulation,
        nx=nx,
        ny=ny,
        n_weeks=n_periods,
        t_field_kind=setup["t_field_kind"],
        t_field_seed=setup["t_field_seed"],
    )
    return setup


def ensure_case_artifact(case_setup: dict) -> Path:
    """
    Return the case artifact path, generating the MF6 truth when missing.

    The generator is imported lazily so the replay stays Flopy-free until a
    truth artifact actually has to be built.
    """
    artifact_path = Path(case_setup["artifact_path"])
    if artifact_path.exists():
        return artifact_path
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
        out_path=artifact_path,
        reuse_existing_warm_start=True,
        warm_start_mode=case_setup["warm_start_mode"],
        formulation=case_setup["formulation"],
    )
    if not artifact_path.exists():
        raise RuntimeError(f"MF6 truth generation did not produce {artifact_path}")
    return artifact_path


def run_production_replay(
    *,
    artifact_path: Path | None,
    workspace: Path | None,
    device: str,
    solve_control_overrides: dict | None = None,
    diag_preconditioner_backend: str = "device",
    allow_warm_start_mismatch: bool = False,
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
    )


if __name__ == "__main__":
    main()
