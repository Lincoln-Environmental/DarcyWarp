# SPDX-License-Identifier: AGPL-3.0-only
"""Numerical-regression helpers for solver extraction checkpoints."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


VOLATILE_DIAGNOSTIC_KEYS = frozenset(
    {
        "total_time",
        "period_times",
        "cuda_graph_built_this_call",
        "cuda_graph_reused",
        "gpu_scalar_synchronization_count",
        "profile",
        "timing",
    }
)


def normalize_diagnostics(value: Any) -> Any:
    """Copy diagnostics while removing only runtime/graph-observation fields."""
    if isinstance(value, dict):
        return {
            key: normalize_diagnostics(item)
            for key, item in value.items()
            if key not in VOLATILE_DIAGNOSTIC_KEYS
        }
    if isinstance(value, list):
        return [normalize_diagnostics(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_diagnostics(item) for item in value)
    if isinstance(value, np.ndarray):
        return np.asarray(value).copy()
    return deepcopy(value)


def compare_heads(*, actual: np.ndarray, expected: np.ndarray, dtype: np.dtype) -> None:
    """Use the extraction tolerance appropriate to the configured precision."""
    atol = 1.0e-12 if np.dtype(dtype) == np.dtype(np.float64) else 2.0e-6
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=atol)


def assert_diagnostic_schema_and_values(*, actual: dict[str, Any], expected: dict[str, Any]) -> None:
    """Assert diagnostic equivalence after documented volatile-key removal."""
    actual_normalized = normalize_diagnostics(actual)
    expected_normalized = normalize_diagnostics(expected)
    assert actual_normalized.keys() == expected_normalized.keys()
    _assert_nested_equivalent(actual=actual_normalized, expected=expected_normalized)


def _assert_nested_equivalent(*, actual: Any, expected: Any) -> None:
    if isinstance(expected, dict):
        assert isinstance(actual, dict)
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equivalent(actual=actual[key], expected=expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert isinstance(actual, type(expected))
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected, strict=True):
            _assert_nested_equivalent(actual=actual_item, expected=expected_item)
        return
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(actual, expected)
        return
    if isinstance(expected, float) and np.isnan(expected):
        assert isinstance(actual, float) and np.isnan(actual)
        return
    assert actual == expected
