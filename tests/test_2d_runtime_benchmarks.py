"""Acceptance protections for the portable 2D performance comparisons."""
import pytest

from working_tests.benchmark_2d_runtime import evaluate_heads, matching_time_grid, variant_controls
from working_tests.transient_replay_settings import HEAD_ACCURACY_CRITERIA


def test_runtime_variant_rejects_unqualified_ghb_before_running():
    with pytest.raises(ValueError, match='not qualified for GHB'):
        variant_controls(defaults={},
            variants={'runtime': {'supports_ghb': False, 'solve_controls': {'omega': 1.}}},
            use_ghb=True)


def test_head_gate_rejects_nonfinite_cells_even_when_metrics_mask_them():
    metric = {'rmse': 0., 'max_abs_diff': 0., 'percent_within_0_01m': 100.}
    comparison = {'per_period': [metric], 'final': metric}
    assert not evaluate_heads(comparison=comparison, finite=False, gates=HEAD_ACCURACY_CRITERIA)


def test_fixed_step_parity_rejects_substeps_even_with_same_end_time():
    assert matching_time_grid(infos=[{'adaptive_dt_substep_dts': [7.]}], dt=7.)
    assert not matching_time_grid(infos=[{'adaptive_dt_substep_dts': [3.5, 3.5]}], dt=7.)
