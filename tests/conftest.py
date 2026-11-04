"""Declare unavailable CUDA coverage explicitly for parametrized 3D checks."""
from __future__ import annotations

import importlib.util

import pytest


def pytest_collection_modifyitems(items):
    cuda_items = []
    for item in items:
        filename = item.path.name
        is_3d = any(token in filename for token in ("3d", "terrain", "npf662"))
        parameters = getattr(getattr(item, "callspec", None), "params", {})
        requested = (parameters.get("device"), parameters.get("backend"))
        if is_3d and any(isinstance(value, str) and value.startswith("cuda") for value in requested):
            cuda_items.append(item)
    if not cuda_items or importlib.util.find_spec("warp") is None:
        return
    import warp as wp

    if wp.is_cuda_available():
        return
    unavailable = pytest.mark.skip(reason="CUDA hardware unavailable; CPU results do not validate CUDA execution")
    for item in cuda_items:
        item.add_marker(unavailable)
