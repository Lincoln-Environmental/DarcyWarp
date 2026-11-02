# SPDX-License-Identifier: AGPL-3.0-or-later
"""The documented 3D example must execute, verify and save real model results."""
import json
import numpy as np

from examples.layered_confined_3d import main


def test_layered_confined_example_is_reproducible_on_cpu(tmp_path):
    result = main(device="cpu", output_dir=tmp_path)
    assert result["converged"]
    assert result["max_head_error_m"] < 1e-7
    assert result["relative_cell_imbalance"] < 1e-8
    assert json.loads((tmp_path / "summary.json").read_text()) == result
    with np.load(tmp_path / "layered_confined_3d.npz", allow_pickle=False) as data:
        assert data["head_m"].shape == (9, 4, 5)
        np.testing.assert_allclose(data["head_m"], data["analytical_head_m"], atol=1e-7, rtol=0.0)
