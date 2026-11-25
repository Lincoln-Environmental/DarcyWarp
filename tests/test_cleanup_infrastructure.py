"""Regression tests for consolidated plotting and benchmark helpers."""

import subprocess
import sys

import numpy as np

from DARCY_WARP_PACKAGE.canterbury_case_study.map_helpers import _build_edges


def test_north_first_map_edges():
    """The shared helper preserves the north-first raster orientation."""
    x, y = _build_edges((10.0, 14.0, 20.0, 26.0), 2, 3)
    np.testing.assert_array_equal(x, [10.0, 12.0, 14.0])
    np.testing.assert_array_equal(y, [26.0, 24.0, 22.0, 20.0])


def test_project_base_import_is_quiet():
    """Importing benchmark paths does not print host or network-share status."""
    completed = subprocess.run(
        [sys.executable, "-c", "import DARCY_WARP_PACKAGE.project_base"],
        capture_output=True, text=True, check=True,
    )
    assert completed.stdout == ""


def test_budget_reader_accepts_current_and_archived_records(tmp_path):
    """Match current listings by content and never guess a cached reference's budget.

    :param tmp_path: Temporary directory for the synthetic benchmark inputs.
    """
    import hashlib
    import json
    from DARCY_WARP_PACKAGE.process_mass_balance import compare_mass_balance

    listing_text = "IN:\n RCHA = 5\n TOTAL IN = 5\nOUT:\n CHD = 5\n TOTAL OUT = 5\n IN - OUT = 0\n PERCENT DISCREPANCY = 0\n"
    current_listing = tmp_path / "current.lst"
    current_listing.write_text(listing_text)
    digest = hashlib.sha256(current_listing.read_bytes()).hexdigest()
    for case in ("2x2", "3x3"):
        workspace = tmp_path / f"Paper_mf6_truth_{case}"
        workspace.mkdir()
        (workspace / "model_truth.lst").write_text(listing_text)
    budget = {"total_in": 5, "total_out": 5, "percent_discrepancy": 0}
    source = tmp_path / "budgets.json"
    source.write_text(json.dumps({
        "2x2": {"mass_balance_budget_kcycle": budget},
        "3x3": {"mass_balance_budget_kcycle": budget, "mf6_listing_path": None},
        "4x4": {"mass_balance_budget_kcycle": budget, "mf6_listing_path": str(current_listing), "mf6_listing_sha256": digest},
        "5x5": {"mass_balance_budget_kcycle": budget, "mf6_listing_path": str(current_listing), "mf6_listing_sha256": "stale"},
    }))
    result = compare_mass_balance(source, tmp_path)
    assert result["case"].tolist() == ["2x2", "3x3", "4x4", "5x5"]
    assert result["mf6_found"].tolist() == [True, False, True, False]
    assert result.iloc[0]["diff_total_in"] == 0.0
    assert np.isnan(result.iloc[1]["diff_total_in"])
    assert result.iloc[2]["diff_total_out"] == 0.0
    assert np.isnan(result.iloc[3]["diff_total_out"])
