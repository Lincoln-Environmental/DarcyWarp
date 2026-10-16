# SPDX-License-Identifier: AGPL-3.0-only
"""
Self-contained, compressed truth fixtures for the 2D unconfined Warp-vs-MF6
benchmark.

A truth artifact bundles, for one benchmark grid:

* the MF6 reference heads (the expensive ground truth we do not want to
  recompute),
* the deterministic case inputs needed to rebuild the Warp solve
  (active/bc/K/top/bottom/initial_head, plus the derived initial transmissivity
  and recharge RHS),
* the exact scalar solver settings and constructor settings used to generate
  the truth, and
* a small provenance dict (when/how it was made).

Container format: an uncompressed ``np.savez`` stream compressed as one blob
with the standard-library ``lzma`` codec at preset 9.  Heads are stored as
float64, so the artifact is **lossless** (full regression sensitivity) while
the largest grid (3000x3000) stays well under 40 MB.  The loader needs only
``numpy`` + stdlib ``lzma`` / ``json`` — no optional dependencies.

Files use the ``.npz.lzma`` extension to make the payload obvious.
"""

from __future__ import annotations

import io
import json
import lzma
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "TRUTH_FILE_SUFFIX",
    "save_truth_artifact",
    "load_truth_artifact",
]

TRUTH_FILE_SUFFIX = ".npz.lzma"

# Array-valued solve() kwargs that are stored as named arrays in the artifact
# rather than inside the settings dict (they do not survive JSON round-trips).
_ARRAY_SOLVE_KEYS = frozenset({"K_field", "zbot_field", "ztop_field", "initial_head"})


def save_truth_artifact(
    out_path: str | Path,
    *,
    heads: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    top: np.ndarray,
    bottom: np.ndarray,
    k_field: np.ndarray,
    recharge: np.ndarray,
    initial_head: np.ndarray,
    initial_transmissivity: np.ndarray,
    rhs_recharge: np.ndarray,
    solve_settings: dict[str, Any],
    constructor_settings: dict[str, Any],
    provenance: dict[str, Any] | None = None,
    preset: int = 9,
) -> Path:
    """
    Write a compressed, self-contained truth fixture for one grid.

    All array inputs are stored as float64 (int masks are kept as int32).
    ``solve_settings`` must contain only JSON-serialisable scalar values — the
    array-valued solve kwargs (``K_field``/``zbot_field``/``ztop_field``/
    ``initial_head``) are provided as named arrays instead.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    stray_arrays = _ARRAY_SOLVE_KEYS.intersection(solve_settings)
    if stray_arrays:
        raise ValueError(
            f"solve_settings must not contain array keys {sorted(stray_arrays)}; "
            "pass those as the corresponding named array arguments instead."
        )

    arrays: dict[str, np.ndarray] = {
        "heads": np.asarray(heads, dtype=np.float64),
        "active": np.asarray(active, dtype=np.int32),
        "bc_mask": np.asarray(bc_mask, dtype=np.int32),
        "bc_values": np.asarray(bc_values, dtype=np.float64),
        "top": np.asarray(top, dtype=np.float64),
        "bottom": np.asarray(bottom, dtype=np.float64),
        "k_field": np.asarray(k_field, dtype=np.float64),
        "recharge": np.asarray(recharge, dtype=np.float64),
        "initial_head": np.asarray(initial_head, dtype=np.float64),
        "initial_transmissivity": np.asarray(initial_transmissivity, dtype=np.float64),
        "rhs_recharge": np.asarray(rhs_recharge, dtype=np.float64),
        "solve_settings": np.asarray(json.dumps(solve_settings, default=str)),
        "constructor_settings": np.asarray(json.dumps(constructor_settings, default=str)),
        "provenance": np.asarray(json.dumps(provenance or {}, default=str)),
    }

    buffer = io.BytesIO()
    np.savez(buffer, **arrays)
    compressed = lzma.compress(buffer.getvalue(), preset=preset)
    out_path.write_bytes(compressed)
    return out_path


def load_truth_artifact(path: str | Path) -> dict[str, Any]:
    """
    Load a truth fixture and return a dict with parsed arrays and settings.

    Returns keys: ``heads``, ``active``, ``bc_mask``, ``bc_values``, ``top``,
    ``bottom``, ``k_field``, ``recharge``, ``initial_head``,
    ``initial_transmissivity``, ``rhs_recharge``, ``solve_settings`` (dict),
    ``constructor_settings`` (dict), ``provenance`` (dict).
    """
    path = Path(path)
    raw = lzma.decompress(path.read_bytes())
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        arrays = {k: data[k] for k in data.files}

    def _json_field(name: str) -> dict[str, Any]:
        return json.loads(str(np.asarray(arrays[name]).reshape(())))

    return {
        "heads": arrays["heads"],
        "active": arrays["active"],
        "bc_mask": arrays["bc_mask"],
        "bc_values": arrays["bc_values"],
        "top": arrays["top"],
        "bottom": arrays["bottom"],
        "k_field": arrays["k_field"],
        "recharge": arrays["recharge"],
        "initial_head": arrays["initial_head"],
        "initial_transmissivity": arrays["initial_transmissivity"],
        "rhs_recharge": arrays["rhs_recharge"],
        "solve_settings": _json_field("solve_settings"),
        "constructor_settings": _json_field("constructor_settings"),
        "provenance": _json_field("provenance"),
    }


def build_solve_kwargs(artifact: dict[str, Any]) -> dict[str, Any]:
    """
    Assemble the full ``solver.solve(...)`` kwargs for a re-run: the stored
    scalar settings merged with the array fields pulled from the artifact.
    """
    kwargs = dict(artifact["solve_settings"])
    kwargs["K_field"] = artifact["k_field"]
    kwargs["zbot_field"] = artifact["bottom"]
    kwargs["ztop_field"] = artifact["top"]
    kwargs["initial_head"] = artifact["initial_head"].copy()
    return kwargs
