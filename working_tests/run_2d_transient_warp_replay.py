#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Thin production harness for the 2D transient unconfined MF6 replay."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from working_tests.transient_artifacts import FORMULATION_UNCONFINED  # noqa: E402
from working_tests.transient_replay_settings import (  # noqa: E402
    PRODUCTION_RUN_MODE,
    default_run_config,
    production_secant_sy_settings,
)
from working_tests.transient_replay_support import main as run_transient_replay  # noqa: E402


if __name__ == "__main__":
    artifact_path = None
    workspace = None
    device = "auto"
    diag_preconditioner_backend = "device"
    allow_warm_start_mismatch = False

    run_mode = PRODUCTION_RUN_MODE
    compute_mass_balance = True
    profile_performance = False
    save_heavy_diagnostics = False
    run_replay_matrix = False

    production_settings = production_secant_sy_settings()
    run_config = default_run_config(
        run_mode=run_mode,
        device=device,
        compute_mass_balance=compute_mass_balance,
        profile_performance=profile_performance,
        save_heavy_diagnostics=save_heavy_diagnostics,
        run_replay_matrix=run_replay_matrix,
    )

    run_transient_replay(
        artifact_path=artifact_path,
        workspace=workspace,
        device=device,
        diag_preconditioner_backend=diag_preconditioner_backend,
        warm_start_mode=production_settings["warm_start_mode"],
        formulation=FORMULATION_UNCONFINED,
        unconfined_storage_mode=production_settings["unconfined_storage_mode"],
        storage_reference=production_settings["storage_reference"],
        storage_top_threshold=production_settings["storage_top_threshold"],
        storage_active_set_strategy=production_settings["storage_active_set_strategy"],
        storage_freeze_after_outer=production_settings["storage_freeze_after_outer"],
        allow_warm_start_mismatch=allow_warm_start_mismatch,
        run_config=run_config,
    )
