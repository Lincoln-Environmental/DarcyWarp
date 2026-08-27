"""Small, dependency-light tests for transient truth identity and replay rules."""

from __future__ import annotations

import io
import lzma
from pathlib import Path

import numpy as np
import pytest

from working_tests.transient_artifacts import (
    ARTIFACT_SCHEMA_VERSION,
    compute_transient_case_fingerprint,
    validate_transient_artifact,
)
from working_tests.transient_replay_metrics import compare_transient
from working_tests.run_2d_transient_warp_replay import production_solver_backend
from working_tests.transient_artifacts import FORMULATION_CONFINED, FORMULATION_UNCONFINED


def test_production_solver_backend_matches_replay_formulation():
    assert production_solver_backend(
        formulation=FORMULATION_CONFINED,
    ) == "confined_kcycle"
    assert production_solver_backend(
        formulation=FORMULATION_UNCONFINED,
    ) == "unconfined_picard_kcycle"
    with pytest.raises(ValueError, match="formulation"):
        production_solver_backend(formulation="invalid")


def _write_artifact(path: Path, *, sy: float = 0.1, ss: float = 1.0e-5, dt: float = 7.0,
                    recharge: float = 1.0e-4, seed: int = 42, ghb_mode: str = "none") -> str:
    shape = (3, 2, 3)
    active = np.ones(shape[1:], dtype=np.int32)
    heads = np.arange(np.prod(shape), dtype=np.float64).reshape(shape) + 100.0
    payload = {
        "heads_per_period": heads,
        "heads_final": heads[-1],
        "nx": np.asarray(3, dtype=np.int32),
        "ny": np.asarray(2, dtype=np.int32),
        "dx": np.asarray(100.0),
        "initial_head": heads[0] - 1.0,
        "active": active,
        "bc_mask": np.zeros_like(active),
        "bc_values": np.zeros_like(active, dtype=np.float64),
        "top": np.full_like(active, 200.0, dtype=np.float64),
        "bottom": np.zeros_like(active, dtype=np.float64),
        "k_field": np.full_like(active, 10.0, dtype=np.float64),
        "t_field_kind": np.asarray("ugly_t"),
        "t_field_seed": np.asarray(seed, dtype=np.int32),
        "sy": np.asarray(sy),
        "ss": np.asarray(ss),
        "dt_days": np.asarray(dt),
        "recharge_rates": np.full(3, recharge),
        "n_periods": np.asarray(3, dtype=np.int32),
        "n_weeks": np.asarray(3, dtype=np.int32),
        "initial_saturated_thickness": np.asarray(100.0),
        "warm_start_mode": np.asarray("unconfined_steady_mf6"),
        "formulation": np.asarray("unconfined"),
        "ghb_conductance_mode": np.asarray(ghb_mode),
        "mf6_normal_termination": np.asarray(1, dtype=np.int8),
        "mf6_budget_discrepancy_max": np.asarray(0.01),
        "mf6_budget_discrepancy_tol": np.asarray(1.0),
        "artifact_schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int32),
    }
    fingerprint = compute_transient_case_fingerprint(payload)
    payload["case_fingerprint"] = np.asarray(fingerprint)
    buffer = io.BytesIO()
    np.savez(buffer, **payload)
    path.write_bytes(lzma.compress(buffer.getvalue()))
    return fingerprint


def test_exact_fingerprint_reuse_and_input_invalidation(tmp_path):
    artifact_path = tmp_path / "truth.npz.lzma"
    fingerprint = _write_artifact(artifact_path)
    loaded = validate_transient_artifact(artifact_path, expected_fingerprint=fingerprint)
    assert str(np.asarray(loaded["case_fingerprint"]).reshape(())) == fingerprint
    for kwargs in (
        {"sy": 0.11}, {"ss": 2.0e-5}, {"dt": 8.0}, {"recharge": 2.0e-4},
        {"seed": 7}, {"ghb_mode": "mf6_fixed_point"},
    ):
        changed = tmp_path / f"changed-{len(kwargs)}-{next(iter(kwargs))}.npz.lzma"
        changed_fp = _write_artifact(changed, **kwargs)
        assert changed_fp != fingerprint


def test_explicit_stale_artifact_is_refused(tmp_path):
    artifact_path = tmp_path / "truth.npz.lzma"
    fingerprint = _write_artifact(artifact_path)
    with pytest.raises(ValueError, match="requested case fingerprint"):
        validate_transient_artifact(artifact_path, expected_fingerprint=fingerprint + "stale")


def test_atomic_write_contract_leaves_no_staging_file(tmp_path):
    from working_tests.run_2d_transient_vs_mf6 import _save_compressed_npz

    artifact_path = tmp_path / "atomic.npz.lzma"
    _save_compressed_npz(artifact_path, {"value": np.asarray([1, 2, 3])})
    assert np.load(io.BytesIO(lzma.decompress(artifact_path.read_bytes()))) ["value"].tolist() == [1, 2, 3]
    assert not list(tmp_path.glob("*.staging-*"))


def test_truncated_replay_compares_selected_final_period():
    heads = np.zeros((3, 2, 2), dtype=np.float64)
    heads[1] = 1.0
    heads[2] = 2.0
    result = {"heads_per_period": heads[:2], "heads_final": heads[1]}
    comparison = compare_transient(
        result,
        heads[:2],
        heads[1],
        np.ones((2, 2), dtype=np.int32),
    )
    assert comparison["final"]["max_abs_diff"] == 0.0
