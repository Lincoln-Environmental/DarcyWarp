"""Production replay permits portable heads and binds every consumed native file."""
import json

import pytest

from test_transient_artifacts import _write_artifact
from working_tests.benchmark_2d_runtime import benchmark
from working_tests.transient_artifacts import file_sha256, validate_transient_artifact
from working_tests.transient_replay_support import run_replay_from_artifact, verify_replay_reference_files


def run_benchmark(path, *, native_workspace=None):
    return benchmark(artifact_path=path, output=path.parent / "out", variants={"production": {}},
                     repeats=1, device="cpu", audit=False, implementation="classic", baseline_heads=None,
                     default_overrides={}, mf6_workspace=native_workspace)


@pytest.mark.parametrize("consumer", ["benchmark", "replay"])
@pytest.mark.parametrize("filename", ["mf6_storage_budget_terms.npz", "mf6/tr2d_truth.hds"])
def test_present_unmatched_sidecar_stops_consumer_before_simulation(tmp_path, consumer, filename):
    artifact = tmp_path / "truth.npz.lzma"
    _write_artifact(artifact)
    sidecar = tmp_path / filename
    sidecar.parent.mkdir(exist_ok=True)
    sidecar.write_bytes(b"sidecar from another run")
    with pytest.raises(ValueError, match="Native output"):
        if consumer == "benchmark":
            run_benchmark(artifact)
        else:
            run_replay_from_artifact(artifact_path=artifact, workspace=tmp_path / "out", device="cpu")


def test_benchmark_requires_complete_explicit_native_timing_workspace(tmp_path):
    artifact = tmp_path / "truth.npz.lzma"
    _write_artifact(artifact)
    with pytest.raises(ValueError, match="Native output"):
        run_benchmark(artifact, native_workspace=tmp_path / "separate-timing-workspace")


def test_reference_report_names_only_native_files_actually_verified(tmp_path):
    path = tmp_path / "truth.npz.lzma"
    _write_artifact(path)
    artifact = validate_transient_artifact(path)
    report = verify_replay_reference_files(artifact=artifact, artifact_path=path)
    assert report["verified_sidecars"] == []
    assert report["mf6_sha256"] == str(artifact["mf6_sha256"])
    assert report["mf6_heads_sha256"] == str(artifact["mf6_heads_sha256"])

    native = tmp_path / "relocated-native-workspace"
    artifact["mf6_native_workspace"] = str(native)
    native.mkdir()
    head = native / "tr2d_truth.hds"
    head.write_bytes(b"sample head sidecar")
    storage = tmp_path / "mf6_storage_budget_terms.npz"
    storage.write_bytes(b"sample storage sidecar")
    artifact["native_output_sha256"] = json.dumps({head.name: file_sha256(head), storage.name: file_sha256(storage)})
    report = verify_replay_reference_files(artifact=artifact, artifact_path=path)
    assert set(report["verified_sidecars"]) == {str(head), str(storage)}


@pytest.mark.parametrize("reader", ["storage_comparison", "mass_balance"])
def test_analysis_rejects_a_substituted_native_storage_budget(tmp_path, reader):
    from working_tests import run_transient_unconfined_replay_analysis as analysis

    path = tmp_path / "truth.npz.lzma"
    _write_artifact(path)
    artifact = validate_transient_artifact(path)
    sidecar = tmp_path / "mf6_storage_budget_terms.npz"
    sidecar.write_bytes(b"original storage")
    artifact["native_output_sha256"] = json.dumps({sidecar.name: file_sha256(sidecar)})
    sidecar.write_bytes(b"replacement from another model")
    (tmp_path / "warp_storage_budget_terms.npz").touch()
    with pytest.raises(ValueError, match="Native output"):
        if reader == "storage_comparison":
            analysis.compare_storage_budgets(artifact=artifact, artifact_path=path, workspace=tmp_path)
        else:
            analysis.mf6_mass_balance_from_budget_artifact(artifact_path=path, artifact=artifact)
