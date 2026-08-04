"""Contract tests for the shared spatial sanity catalog."""

from __future__ import annotations

from DARCY_WARP_PACKAGE.sanity_case_config import (
    GRID_CASES,
    SPATIAL_GRID_CASES,
    STEADY_SANITY_LABELS,
    TRANSIENT_CAPACITY_LABELS,
    TRANSIENT_PRODUCTION_LABELS,
    TRANSIENT_SCALE_LABELS,
    TRANSIENT_SHAPE_LABELS,
    TRANSIENT_SMOKE_LABELS,
)


def test_selected_labels_exist_in_spatial_catalog():
    labels = (
        STEADY_SANITY_LABELS
        + TRANSIENT_SMOKE_LABELS
        + TRANSIENT_SHAPE_LABELS
        + TRANSIENT_PRODUCTION_LABELS
        + TRANSIENT_SCALE_LABELS
        + TRANSIENT_CAPACITY_LABELS
    )
    assert set(labels) <= set(SPATIAL_GRID_CASES)


def test_grid_cases_is_steady_backward_compatible_view():
    assert tuple(GRID_CASES) == STEADY_SANITY_LABELS
    assert all(set(case) == {"nx", "ny"} for case in GRID_CASES.values())
    assert "3000x1999" not in GRID_CASES
    assert all(not SPATIAL_GRID_CASES[label]["manual_only"] for label in GRID_CASES)


def test_rectangular_and_odd_geometry_is_exact():
    expected = {
        "100x250": (100, 250),
        "100x1000": (100, 1000),
        "1000x1001": (1000, 1001),
        "3000x111": (3000, 111),
        "3000x223": (3000, 223),
        "3000x333": (3000, 333),
        "3000x999": (3000, 999),
    }
    for label, (nx, ny) in expected.items():
        assert (SPATIAL_GRID_CASES[label]["nx"], SPATIAL_GRID_CASES[label]["ny"]) == (nx, ny)


def test_capacity_cases_are_manual_only_and_not_selected():
    for label in TRANSIENT_CAPACITY_LABELS:
        assert SPATIAL_GRID_CASES[label]["manual_only"] is True
        assert label not in set(
            TRANSIENT_SMOKE_LABELS
            + TRANSIENT_SHAPE_LABELS
            + TRANSIENT_PRODUCTION_LABELS
            + TRANSIENT_SCALE_LABELS
        )
