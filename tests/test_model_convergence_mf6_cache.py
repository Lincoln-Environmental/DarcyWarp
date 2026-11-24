"""Steady reference caching uses physical inputs and exact reference identity."""

from pathlib import Path

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.model_builder import _build_domain
from DARCY_WARP_PACKAGE.validation.steady_truth_2d import load_or_run_mf6_truth, steady_transmissivity


@pytest.mark.parametrize("isotropic", [True, False])
def test_reference_cache_tracks_the_actual_field_and_writer(tmp_path: Path, isotropic):
    pytest.importorskip("flopy")
    nx, ny = 9, 8
    domain = _build_domain(nx=nx, ny=ny)
    t_field = steady_transmissivity(domain=domain, isotropic=isotropic, isotropic_value=3000.0, seed=42)
    conductivity = t_field / 300.0
    recharge = np.full((ny, nx), 1.0e-4)
    heads = np.arange(nx * ny, dtype=float).reshape(ny, nx)
    truth_path = tmp_path / "historical.npz"
    np.savez(truth_path, heads=-heads)
    executable = tmp_path / "mf6"
    executable.write_bytes(b"first identity")
    received = []

    def fake_reference(**kwargs):
        received.append(kwargs)
        return heads, 0.75

    options = dict(truth_path=truth_path, workspace=tmp_path / "workspace", label="9x8",
        nx=nx, ny=ny, dx=100., ghb=True, isotropic=isotropic, t_isotropic_value=3000.,
        thickness=300., width=100., recharge_rate=1e-4, recharge_field=recharge, seed=42,
        hk_field=conductivity, output_dtype=np.dtype(np.float64), mf6_runner=fake_reference,
        domain=domain, executable=executable)
    result, seconds, source = load_or_run_mf6_truth(**options)
    assert source == "generated" and seconds == 0.75
    assert received[0]["hk"] is conductivity and received[0]["recharge"] is recharge
    assert (np.unique(conductivity[domain != 0]).size == 1) is isotropic
    np.testing.assert_array_equal(result, heads)
    assert load_or_run_mf6_truth(**options)[2] == "cache"
    assert len(received) == 1

    conductivity[4, 4] *= 1.1
    assert load_or_run_mf6_truth(**options)[2] == "generated"
    executable.write_bytes(b"second identity")
    assert load_or_run_mf6_truth(**options)[2] == "generated"
    assert load_or_run_mf6_truth(**options)[2] == "cache"
    assert len(received) == 3
    with np.load(truth_path) as archived:
        np.testing.assert_array_equal(archived["heads"], -heads)
