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
    head_history_sha256,
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
        "mf6_heads_sha256": np.asarray(head_history_sha256(heads)),
        "nx": np.asarray(3, dtype=np.int32),
        "ny": np.asarray(2, dtype=np.int32),
        "dx": np.asarray(100.0),
        "initial_head": heads[0] - 1.0,
        "unconfined_steady_head": heads[0] - 1.0,
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
        "mf6_version": np.asarray("mf6: 6.6.2 05/12/2025"),
        "mf6_sha256": np.asarray("9acdb07cae589555355b5d25b5ec1a7af7a0127ff95339ee14971ec7df6641ab"),
        "mf6_build_identity": np.asarray("official-linux-6.6.2"),
        "mf6_executable": np.asarray("/example/mf6"),
        "warm_start_source": np.asarray("native_steady"),
        "warm_start_fingerprint": np.asarray("a" * 64),
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


def test_native_cache_qualifies_each_physics_and_completed_warm_start(tmp_path, pinned_mf6):
    from concurrent.futures import ThreadPoolExecutor
    from working_tests.run_2d_transient_warp_replay import build_case_setup, ensure_case_artifact

    setup = build_case_setup(nx=16, ny=12, n_periods=2, t_field_kind="homogeneous")
    setup["artifact_path"] = tmp_path / "truth.npz.lzma"

    def generate(settings):
        return ensure_case_artifact(settings, executable=pinned_mf6)

    # Same caller-selected path; simultaneous producers must agree on complete siblings.
    with ThreadPoolExecutor(max_workers=2) as pool:
        first, second = pool.map(generate, [setup, setup])
    assert first == second
    stock = validate_transient_artifact(first, verify_native_files=True)
    assert Path(str(stock["mf6_executable"])) == pinned_mf6.resolve()
    assert str(stock["warm_start_source"]) == "native_steady"
    assert float(stock["unconfined_steady_engine_time"]) > 0

    # Changing only the transient storage requires fresh transient heads but
    # permits reuse of the exact same steady equation and native build.
    storage = validate_transient_artifact(generate({**setup, "sy": 0.12}), verify_native_files=True)
    assert float(storage["unconfined_steady_engine_time"]) == 0
    assert str(storage["warm_start_fingerprint"]) == str(stock["warm_start_fingerprint"])

    dx = validate_transient_artifact(generate({**setup, "dx": 2 * setup["dx"]}), verify_native_files=True)
    recharge = validate_transient_artifact(generate({**setup, "annual_recharge_m": 1.2}), verify_native_files=True)
    for changed in [dx, recharge]:
        assert float(changed["unconfined_steady_engine_time"]) > 0
        assert str(changed["warm_start_fingerprint"]) != str(stock["warm_start_fingerprint"])
        assert np.max(np.abs(changed["initial_head"] - stock["initial_head"])) > 1e-5

    warm_cache = tmp_path / "mf6_unconfined_steady_warm_start" / (str(stock["warm_start_fingerprint"]) + ".npz.lzma")
    warm_cache.write_bytes(b"interrupted")
    repaired = validate_transient_artifact(generate({**setup, "sy": 0.11}), verify_native_files=True)
    assert float(repaired["unconfined_steady_engine_time"]) > 0
    np.testing.assert_array_equal(repaired["initial_head"], stock["initial_head"])

    # A detached native budget must not survive under a trusted head artifact.
    (tmp_path / "mf6" / "tr2d_truth.cbb").write_bytes(b"stale native budget")
    recovered = generate(setup)
    assert recovered != first
    validate_transient_artifact(recovered, verify_native_files=True)


def test_transient_schema2_requires_explicit_historical_read(tmp_path):
    from working_tests.transient_artifacts import _load_compressed_npz
    from working_tests.run_2d_transient_vs_mf6 import _save_compressed_npz

    artifact = tmp_path / "legacy.npz.lzma"
    _write_artifact(artifact)
    old = _load_compressed_npz(artifact)
    for key in ("mf6_version", "mf6_executable", "mf6_sha256", "mf6_build_identity",
                "warm_start_source", "warm_start_fingerprint"):
        old.pop(key)
    old["artifact_schema_version"] = np.asarray(2)
    old["case_fingerprint"] = np.asarray(compute_transient_case_fingerprint(old, schema=2))
    _save_compressed_npz(artifact, old)
    with pytest.raises(ValueError, match="schema 2"):
        validate_transient_artifact(artifact)
    assert "mf6_sha256" not in validate_transient_artifact(artifact, allow_legacy=True)


def test_untrusted_build_cannot_produce_or_reuse_2d_qualification(tmp_path, pinned_mf6):
    from working_tests.run_2d_transient_warp_replay import build_case_setup, ensure_case_artifact
    import shutil

    setup = build_case_setup(nx=10, ny=8, n_periods=2, t_field_kind="homogeneous")
    setup["artifact_path"] = tmp_path / "truth.npz.lzma"
    trusted = ensure_case_artifact(setup, executable=pinned_mf6)
    modified = tmp_path / ("modified-" + pinned_mf6.name)
    shutil.copy2(pinned_mf6, modified)
    with modified.open("ab") as stream:
        stream.write(b"unapproved build")
    with pytest.raises(ValueError, match="pinned MF6 contract"):
        ensure_case_artifact(setup, executable=modified)
    assert ensure_case_artifact(setup, executable=pinned_mf6) == trusted


@pytest.mark.parametrize("field,message", [
    ("heads_per_period", "head history checksum"),
    ("heads_final", "final head"),
    ("unconfined_steady_head", "named steady head"),
    ("ghb_width", "fingerprint mismatch"),
])
def test_portable_reference_rejects_altered_arrays(tmp_path, field, message):
    from working_tests.transient_artifacts import _load_compressed_npz
    from working_tests.run_2d_transient_vs_mf6 import _save_compressed_npz

    path = tmp_path / "truth.npz.lzma"
    _write_artifact(path)
    payload = _load_compressed_npz(path)
    if field == "ghb_width":
        payload[field] = np.ones((2, 3))
    else:
        payload[field].flat[0] += 1
    _save_compressed_npz(path, payload)
    with pytest.raises(ValueError, match=message):
        validate_transient_artifact(path)


def test_both_direct_steady_runners_precheck_the_executable_hash(tmp_path, monkeypatch):
    from working_tests import run_2d_transient_vs_mf6 as generator
    from working_tests import transient_artifacts

    def forbid_process(*args, **kwargs):
        pytest.fail("unapproved executable was invoked for its version")

    monkeypatch.setattr(transient_artifacts.subprocess, "check_output", forbid_process)
    untrusted = tmp_path / "not-a-reference"
    untrusted.write_bytes(b"not a supported MF6")
    case = generator.build_transient_unconfined_case(nx=8, ny=6, n_weeks=2, workspace=tmp_path/"input")
    for runner in (generator.run_mf6_confined_steady_warm_start, generator.run_mf6_unconfined_steady_warm_start):
        with pytest.raises(ValueError, match="pinned MF6 contract"):
            runner(case=case, executable=untrusted, mf6_workspace=tmp_path / "steady")


def test_explicit_native_workspace_and_caller_supplied_initial_head(tmp_path, pinned_mf6):
    from working_tests.run_2d_transient_vs_mf6 import build_transient_unconfined_case, run_mf6_transient
    from working_tests.transient_artifacts import load_transient_artifact

    case = build_transient_unconfined_case(nx=8, ny=6, n_weeks=2, workspace=tmp_path / "fields")
    target = tmp_path / "bundle" / "truth.npz.lzma"
    raw = tmp_path / "custom_native"
    run_mf6_transient(case, out_path=target, mf6_workspace=raw, executable=pinned_mf6,
        warm_start_mode="unconfined_steady_mf6", unconfined_steady_head=case.initial_head)
    qualified = validate_transient_artifact(target, verify_native_files=True)
    assert str(qualified["mf6_native_workspace"]) == str(raw.resolve())
    assert str(qualified["warm_start_source"]) == "supplied_head"
    assert str(qualified["warm_start_mode"]) == "artifact_initial"
    assert str(load_transient_artifact(target)["warm_start_source"]) == "supplied_head"
