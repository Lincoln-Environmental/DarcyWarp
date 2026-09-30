# SPDX-License-Identifier: AGPL-3.0-only
"""Live MF6 parity tests for the gated DRN/RIV boundary packages.

Drives ``working_tests.run_2d_unconfined_warp_vs_mf6.run_case`` with
``boundary_kind="drn"``/``"riv"`` on a tiny grid (live MF6 truth, one direct
MF6 run per case — DRN/RIV conductances are fixed and identical on both
sides, so no GHB conductance fixed point is involved).  For each case:

* Warp heads agree with MF6 to ``DEFAULT_MF6_AGREEMENT_TOL``;
* the host-side boundary-flux diagnostic agrees with the MF6 CBC budget;
* the gated on/off pattern matches MF6's (zero-flow DRN cells / constant-flux
  decoupled RIV cells in the CBC records).

Skipped when flopy/MF6 or warp is unavailable.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

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
os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))
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

# Boundary-flux totals must agree with the MF6 CBC budget to this relative
# tolerance; the gate pattern must match exactly.
_FLUX_RTOL = 1.0e-3


@pytest.fixture(scope="module")
def drn_all_active_workspace(tmp_path_factory):
    """Low drain elevation: every centre-row drain is active at convergence."""
    ws = tmp_path_factory.mktemp("drn_all_active")
    summary = R.run_case(
        **_TINY, boundary_kind="drn",
        drn_elevation=20.0, drn_conductance=200.0, workspace=ws,
    )
    return summary


@pytest.fixture(scope="module")
def drn_partially_gated_workspace(tmp_path_factory):
    """High drain elevation: part of the row gates off at the converged head.

    The no-drain equilibrium on the centre row spans 220-300 m of saturated
    thickness; with the drain elevation at 250 m above the bottom the western
    (high-bottom) cells stay below the drain and gate off.
    """
    ws = tmp_path_factory.mktemp("drn_partially_gated")
    summary = R.run_case(
        **_TINY, boundary_kind="drn",
        drn_elevation=250.0, drn_conductance=200.0, workspace=ws,
    )
    return summary


@pytest.fixture(scope="module")
def riv_fully_coupled_workspace(tmp_path_factory):
    """Stage near the DEM and a low riverbed bottom: all cells coupled."""
    ws = tmp_path_factory.mktemp("riv_fully_coupled")
    summary = R.run_case(
        **_TINY, boundary_kind="riv",
        riv_stage_elevation=290.0, riv_rbot_elevation=5.0,
        riv_conductance=200.0, workspace=ws,
    )
    return summary


@pytest.fixture(scope="module")
def riv_partially_decoupled_workspace(tmp_path_factory):
    """High riverbed bottom (still below stage): part of the row decouples."""
    ws = tmp_path_factory.mktemp("riv_partially_decoupled")
    summary = R.run_case(
        **_TINY, boundary_kind="riv",
        riv_stage_elevation=290.0, riv_rbot_elevation=250.0,
        riv_conductance=200.0, workspace=ws,
    )
    return summary


def _check_parity(summary, *, expect_gated_off: bool) -> dict:
    comp = summary.get("comparison") or {}
    assert comp.get("max_abs_diff") is not None
    assert float(comp["max_abs_diff"]) <= R.DEFAULT_MF6_AGREEMENT_TOL

    bf = comp.get("boundary_flux_comparison") or {}
    assert bf, "expected boundary_flux_comparison in the comparison results"
    assert int(bf["n_boundary_cells"]) > 0
    assert float(bf["relative_total_diff"]) < _FLUX_RTOL
    assert bf["gate_pattern_matches_mf6"] is True
    assert bool(summary.get("solve2_converged")) is True
    if expect_gated_off:
        assert int(bf["n_gated_off_warp"]) >= 1
        assert int(bf["n_gated_off_warp"]) < int(bf["n_boundary_cells"])
    else:
        assert int(bf["n_gated_off_warp"]) == 0
        assert int(bf["n_gated_off_mf6"]) == 0
    return bf


@requires_mf6
@requires_warp
def test_drn_all_active_parity(drn_all_active_workspace):
    _check_parity(drn_all_active_workspace, expect_gated_off=False)


@requires_mf6
@requires_warp
def test_drn_partially_gated_parity(drn_partially_gated_workspace):
    _check_parity(drn_partially_gated_workspace, expect_gated_off=True)


@requires_mf6
@requires_warp
def test_riv_fully_coupled_parity(riv_fully_coupled_workspace):
    _check_parity(riv_fully_coupled_workspace, expect_gated_off=False)


@requires_mf6
@requires_warp
def test_riv_partially_decoupled_parity(riv_partially_decoupled_workspace):
    _check_parity(riv_partially_decoupled_workspace, expect_gated_off=True)
