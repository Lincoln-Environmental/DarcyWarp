"""Validation tests for the steady 2D unconfined MF6-vs-Warp runner.

Covers the runner hardening: artifact fingerprints, MF6 budget-discrepancy
gate, non-converged-Warp refusal, and the cold-cache GHB fixed-point path.
MF6-dependent tests use tiny grids and are skipped when flopy/MF6 or warp is
unavailable.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _flopy_available() -> bool:
    try:
        import flopy  # noqa: F401
    except Exception:
        return False
    return True


def _mf6_available() -> bool:
    if not _flopy_available():
        return False
    try:
        from DARCY_WARP_PACKAGE.project_base import require_mf6

        return Path(require_mf6()).exists()
    except Exception:
        return False


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


requires_mf6 = pytest.mark.skipif(not _mf6_available(), reason="flopy/MF6 binary not available")
requires_warp = pytest.mark.skipif(not _warp_available(), reason="warp is not available")

# The runner pins DARCY_FLOAT=float64 at import time; preserve the ambient
# setting so the rest of the test session is unaffected.
_PREV_DARCY_FLOAT = os.environ.get("DARCY_FLOAT")
R = pytest.importorskip("working_tests.run_2d_unconfined_warp_vs_mf6")
if _PREV_DARCY_FLOAT is None:
    os.environ.pop("DARCY_FLOAT", None)
else:
    os.environ["DARCY_FLOAT"] = _PREV_DARCY_FLOAT

_TINY = dict(
    nx=40,
    ny=30,
    hydraulic_conductivity=10.0,
    device="cpu",
    do_double_solve=False,
    check_every_no=5,
    inner_implementation="classic",
)


@pytest.fixture(scope="module")
def ghb_workspace(tmp_path_factory):
    """Cold-cache GHB case at a tiny grid (MF6 fixed point + Warp solve)."""
    ws = tmp_path_factory.mktemp("ghb_case")
    summary = R.run_case(
        **_TINY,
        use_ghb=True,
        ghb_conductance_mode="fixed_point",
        workspace=ws,
    )
    return ws, summary


@requires_mf6
@requires_warp
def test_cold_cache_ghb_fixed_point(ghb_workspace):
    """Cold workspace: MF6-side GHB fixed point runs, gates are recorded."""
    ws, summary = ghb_workspace
    fp = summary.get("ghb_fixed_point") or {}
    assert fp.get("converged") is True
    assert int(fp.get("iterations", 0)) >= 1
    assert float(fp["cumulative_engine_time"]) >= float(fp["terminal_run_engine_time"])
    assert float(fp["cumulative_total_time"]) >= float(fp["terminal_run_total_time"])
    assert summary.get("mf6_budget_discrepancy_max") is not None
    assert float(summary["mf6_budget_discrepancy_max"]) <= float(
        summary["mf6_budget_discrepancy_tol"]
    )
    assert summary.get("mf6_head_change_max_from_initial") is not None
    assert summary.get("comparison_ghb_cells"), "expected GHB-cell comparison metrics"
    assert summary.get("ghb_coupling_ratio"), "expected coupling-ratio metrics"
    assert bool(summary.get("solve2_converged")) is True
    with np.load(ws / "mf6_heads.npz", allow_pickle=False) as npz:
        assert int(np.asarray(npz["schema_version"]).reshape(())) == R.ARTIFACT_SCHEMA_VERSION
        stored = str(np.asarray(npz["case_fingerprint"]).reshape(()))
        assert float(np.asarray(npz["engine_time"]).reshape(())) == pytest.approx(
            float(fp["cumulative_engine_time"])
        )
    case = R.build_simple_unconfined_case(
        nx=_TINY["nx"], ny=_TINY["ny"],
        hydraulic_conductivity=_TINY["hydraulic_conductivity"],
        use_ghb=True, ghb_conductance_mode="fixed_point", workspace=ws,
    )
    assert stored == R.case_fingerprint(case)


@requires_mf6
@requires_warp
def test_artifact_reuse_and_fingerprint_mismatch(ghb_workspace):
    """Same case reuses the cache; a changed parameter is rejected."""
    ws, _ = ghb_workspace
    # Reuse path: do_run_mf6=False validates the fingerprint and reuses.
    summary = R.run_case(**_TINY, use_ghb=True, workspace=ws,
                         ghb_conductance_mode="fixed_point",
                         do_run_mf6=False, do_run_warp=False)
    assert summary.get("comparison"), "expected comparison from cached artifacts"
    # Fingerprint mismatch (different recharge) with regeneration disabled.
    with pytest.raises(RuntimeError, match="stale|fingerprint|does not match"):
        R.run_case(**{**_TINY, "recharge": 2.0e-4}, use_ghb=True, workspace=ws,
                   ghb_conductance_mode="fixed_point",
                   do_run_mf6=False, do_run_warp=False)


@requires_mf6
@requires_warp
def test_nonconverged_warp_refused(ghb_workspace, tmp_path, monkeypatch):
    """A non-converged Warp solve must be refused for comparison."""
    src_ws, _ = ghb_workspace
    ws = tmp_path / "refusal_case"
    ws.mkdir()
    shutil.copy(src_ws / "mf6_heads.npz", ws / "mf6_heads.npz")

    def _fake_warp(case, out_path=None, **kwargs):
        # Copy the real converged artifact but flip the convergence flags.
        with np.load(src_ws / "warp_heads.npz", allow_pickle=False) as npz:
            payload = {name: npz[name] for name in npz.files}
        for name in ("info", "info_solve2"):
            info = json.loads(str(np.asarray(payload[name]).reshape(())))
            info["converged"] = False
            info["strict_picard_convergence_passed"] = False
            payload[name] = np.asarray(json.dumps(info))
        payload["heads"] = np.asarray(payload["heads"], dtype=np.float64)
        np.savez_compressed(out_path, **payload)
        return out_path

    monkeypatch.setattr(R, "run_warp_unconfined", _fake_warp)
    with pytest.raises(RuntimeError, match="did not converge"):
        R.run_case(**_TINY, use_ghb=True, workspace=ws,
                   ghb_conductance_mode="fixed_point",
                   do_run_mf6=False, do_run_warp=True)


@requires_mf6
@requires_warp
def test_warp_matched_equation_equivalence_mode(tmp_path):
    """Default warp_matched mode: C_gh from Warp's converged head, gated on
    strict Warp convergence, fingerprint distinct from the fixed-point mode."""
    ws = tmp_path / "warp_matched_case"
    summary = R.run_case(**_TINY, use_ghb=True,
                         ghb_conductance_mode="warp_matched", workspace=ws)
    fp = summary.get("ghb_fixed_point") or {}
    assert fp.get("method") == "warp_matched_equation_equivalence"
    assert summary.get("ghb_conductance_mode") == "warp_matched"
    assert bool(summary.get("solve2_converged")) is True
    comp = summary.get("comparison") or {}
    assert comp.get("max_abs_diff") is not None
    # Equation-equivalence: same operator both sides -> tight agreement.
    assert float(comp["max_abs_diff"]) < R.DEFAULT_MF6_AGREEMENT_TOL
    with np.load(ws / "mf6_heads.npz", allow_pickle=False) as npz:
        assert "ghb_conductance" in npz
        assert "ghb_fixed_point" in npz
        stored_info = json.loads(str(np.asarray(npz["ghb_fixed_point"]).reshape(())))
    assert stored_info["method"] == "warp_matched_equation_equivalence"
    # The mode is part of the case fingerprint (no cross-mode cache reuse).
    case_fp = R.build_simple_unconfined_case(
        nx=_TINY["nx"], ny=_TINY["ny"], use_ghb=True,
        ghb_conductance_mode="fixed_point", workspace=tmp_path / "a",
    )
    case_wm = R.build_simple_unconfined_case(
        nx=_TINY["nx"], ny=_TINY["ny"], use_ghb=True,
        ghb_conductance_mode="warp_matched", workspace=tmp_path / "b",
    )
    assert R.case_fingerprint(case_fp) != R.case_fingerprint(case_wm)


@requires_mf6
def test_budget_discrepancy_gate(tmp_path, monkeypatch):
    """A huge parsed budget discrepancy must fail the MF6 run."""
    monkeypatch.setattr(
        R,
        "_parse_mf6_lst",
        lambda mf6_ws: {"budget_discrepancy_max": 250.0, "normal_termination": True},
    )
    case = R.build_simple_unconfined_case(nx=12, ny=10, workspace=tmp_path)
    with pytest.raises(RuntimeError, match="BUDGET DISCREPANCY"):
        R.run_mf6_unconfined(case)


def test_budget_parser_synthetic(tmp_path):
    """The .lst parser handles the real MF6 output format."""
    lst = tmp_path / "model.lst"
    lst.write_text(
        "some header\n"
        " PERCENT DISCREPANCY =           0.12     PERCENT DISCREPANCY =          -0.34\n"
        " Normal termination of simulation.\n"
    )
    stats = R._parse_mf6_lst(tmp_path)
    assert stats["normal_termination"] is True
    assert stats["budget_discrepancy_max"] == pytest.approx(0.34)


def test_budget_parser_missing_table(tmp_path):
    (tmp_path / "model.lst").write_text(" Normal termination of simulation.\n")
    stats = R._parse_mf6_lst(tmp_path)
    assert stats["normal_termination"] is True
    assert stats["budget_discrepancy_max"] is None
    with pytest.raises(RuntimeError, match="budget table"):
        R._check_mf6_budget_discrepancy(None, 1.0)


def test_runner_default_is_equation_equivalence():
    """The benchmark defaults to the requested same-operator comparison."""
    import inspect

    assert inspect.signature(R.run_case).parameters["ghb_conductance_mode"].default == "warp_matched"
    assert inspect.signature(R.run_grid_benchmark).parameters["ghb_conductance_mode"].default == "warp_matched"
    assert inspect.signature(R.main).parameters["ghb_conductance_mode"].default == "warp_matched"


def test_runner_uses_shared_sanity_grid_configuration():
    """The default grid sequence comes only from the shared spatial catalog."""
    from DARCY_WARP_PACKAGE.sanity_case_config import DEFAULT_GRID_LABELS, SPATIAL_GRID_CASES

    expected = tuple(
        (int(SPATIAL_GRID_CASES[label]["nx"]), int(SPATIAL_GRID_CASES[label]["ny"]))
        for label in DEFAULT_GRID_LABELS
    )
    assert R.BENCHMARK_GRID_SIZES == expected
    assert (3000, 3000) not in R.BENCHMARK_GRID_SIZES


@requires_mf6
@requires_warp
def test_validator_cold_cache_subprocess(tmp_path):
    """Subprocess smoke: the GHB/hard-T validator on an EMPTY workspace."""
    subprocess_env = dict(os.environ)
    subprocess_env["WARP_CACHE_PATH"] = str(tmp_path / "warp_cache")
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "working_tests" / "validate_unconfined_ghb_hardt.py"),
            "--workspace",
            str(tmp_path / "validator_ws"),
            "--nx",
            "64",
            "--ny",
            "48",
            "--combos",
            "uniform+ghb,ugly_t",
            "--skip-tight",
        ],
        capture_output=True,
        text=True,
        timeout=1200,
        cwd=str(REPO_ROOT),
        env=subprocess_env,
    )
    assert proc.returncode == 0, (
        f"validator failed (rc={proc.returncode})\nSTDOUT tail:\n{proc.stdout[-3000:]}\n"
        f"STDERR tail:\n{proc.stderr[-3000:]}"
    )
    assert "OVERALL: PASS" in proc.stdout
