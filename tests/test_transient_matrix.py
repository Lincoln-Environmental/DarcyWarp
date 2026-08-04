"""Resumability and selection checks for the transient sanity matrix."""

from __future__ import annotations

import json

import numpy as np

from working_tests.run_2d_transient_sanity_matrix import (
    MATRIX_SCHEMA_VERSION,
    _matrix_row_identity,
    _row_key,
    _apply_case_parity,
    atomic_write_json,
    _selected_labels,
)


def test_matrix_selection_uses_shared_catalog():
    assert _selected_labels("smoke") == ("100x100", "100x250")
    assert "1000x1001" in _selected_labels("shape")
    assert "3000x3000" in _selected_labels("capacity")


def test_matrix_json_write_is_atomic_and_resumable(tmp_path):
    path = tmp_path / "matrix.json"
    atomic_write_json(path, {"rows": {"100x100:face_graph_fp64": {"done": True}}})
    assert json.loads(path.read_text())["rows"]["100x100:face_graph_fp64"]["done"] is True
    assert not list(tmp_path.glob("*.staging-*"))


def test_matrix_resume_key_includes_execution_identity():
    base = _matrix_row_identity(
        label="100x100",
        variant="face_graph_fp64",
        physical_fingerprint="physical-a",
        solver_control_fingerprint="controls-a",
        precision_mode="fp64",
        commit="commit-a",
        device="cuda:0",
        warp_version="1.11.0",
    )
    changed_device = dict(base, device="cpu")
    changed_controls = dict(base, solver_control_fingerprint="controls-b")
    assert base["schema_version"] == MATRIX_SCHEMA_VERSION
    assert _row_key(base) != _row_key(changed_device)
    assert _row_key(base) != _row_key(changed_controls)


def test_matrix_persists_parity_for_reference_and_graph_rows(tmp_path):
    classic_path = tmp_path / "classic.npz"
    eager_path = tmp_path / "eager.npz"
    graph_path = tmp_path / "graph.npz"
    heads = np.ones((3, 4, 5), dtype=np.float64)
    np.savez(classic_path, heads_per_period=heads)
    np.savez(eager_path, heads_per_period=heads)
    np.savez(graph_path, heads_per_period=heads)
    rows = {
        "classic": {
            "case_label": "100x100",
            "variant": "classic_device_fp64",
            "completed": True,
            "head_artifact_path": str(classic_path),
        },
        "graph": {
            "case_label": "100x100",
            "variant": "face_graph_fp64",
            "completed": True,
            "head_artifact_path": str(graph_path),
        },
        "eager": {
            "case_label": "100x100",
            "variant": "face_eager_fp64",
            "completed": True,
            "head_artifact_path": str(eager_path),
        },
    }

    _apply_case_parity(
        rows=rows,
        label="100x100",
        variants=("classic_device_fp64", "face_eager_fp64", "face_graph_fp64"),
    )

    assert rows["classic"]["head_parity"]["comparison"] == "reference_self"
    assert rows["classic"]["head_parity"]["passed"] is True
    assert rows["graph"]["head_parity"]["comparison"] == "face_graph_vs_face_eager"
    assert rows["graph"]["head_parity"]["passed"] is True
