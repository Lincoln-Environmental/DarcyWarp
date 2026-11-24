"""Canterbury boundary inputs distinguish missing stages, zero heads and overlap."""

import numpy as np

from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import (
    load_case_inputs, resolve_fixed_heads,
)


def test_canterbury_source_masks_preserve_zero_stages_and_apply_sea_precedence():
    north = np.array([[np.nan, 0., 0.]])
    south = np.array([[0., 0., 1.3]])
    sea = np.array([[0., 0., 0.1]])
    fixed, stages, audit = resolve_fixed_heads(active=np.ones_like(north, dtype=bool),
        stages=dict(north=north, south=south, sea=sea),
        masks=dict(north=np.array([[True, True, False]])), policy="sea_precedence")
    np.testing.assert_array_equal(fixed, [[0, 1, 1]])
    np.testing.assert_allclose(stages, [[0.0, 0.0, 0.1]])
    assert audit["missing_stages"] == [dict(row=0, column=0, source="north")]
    assert audit["overlaps"][0]["selected_source"] == "sea"


def test_archive_boundary_provenance_and_free_zero_stage_ghb():
    case = load_case_inputs(grid_size=100)
    missing = [(row["row"], row["column"]) for row in case.boundary_audit["missing_stages"]]
    assert missing == [(58, 834), (59, 841), (59, 842), (59, 843), (60, 852)]
    assert all(case.bc_mask[row, col] == 0 for row, col in missing)
    assert case.gh_mask[60, 852] == 1 and case.gh_head[60, 852] == 0.0
    assert case.bc_mask[625, 455] == 1
    np.testing.assert_allclose(case.bc_values[625, 455], 0.1)
