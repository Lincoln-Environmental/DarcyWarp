# SPDX-License-Identifier: AGPL-3.0-only
"""Persist solver-extraction golden results produced by a caller-supplied case."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from DARCY_WARP_PACKAGE.solvers.regression import normalize_diagnostics


def save_solver_baseline(
    *,
    output_path: Path,
    head: np.ndarray,
    diagnostics: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """Save a stable numeric head array and JSON-safe normalized diagnostics."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, head=np.asarray(head, dtype=np.float64))
    diagnostics_path = output_path.with_suffix(".json")
    diagnostics_payload = {
        "metadata": metadata,
        "diagnostics": normalize_diagnostics(diagnostics),
    }
    diagnostics_path.write_text(json.dumps(diagnostics_payload, indent=2, default=_json_default))


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"not JSON serializable: {type(value).__name__}")


if __name__ == "__main__":
    output_path = Path("solver_baseline.npz")
    head = np.zeros((1, 1), dtype=np.float64)
    diagnostics: dict[str, Any] = {}
    metadata = {"note": "Import save_solver_baseline from a concrete regression case."}
    save_solver_baseline(
        output_path=output_path,
        head=head,
        diagnostics=diagnostics,
        metadata=metadata,
    )
