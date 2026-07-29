from __future__ import annotations

from pathlib import Path

import numpy as np

from DARCY_WARP_PACKAGE.model_convergence_and_sanity_tests import (
    load_or_run_mf6_truth,
    save_mf6_truth,
)


def build_case_arrays(nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heads = np.arange(nx * ny, dtype=np.float64).reshape(ny, nx)
    recharge = np.full((ny, nx), 1.0e-4, dtype=np.float64)
    hk = np.full((ny, nx), 10.0, dtype=np.float64)
    return heads, recharge, hk


def test_load_or_run_mf6_truth_uses_matching_cache(tmp_path: Path) -> None:
    nx = 4
    ny = 3
    heads, recharge, hk = build_case_arrays(nx=nx, ny=ny)
    truth_path = tmp_path.joinpath("truth.npz")
    save_mf6_truth(
        truth_path=truth_path,
        heads=heads,
        label="4x3",
        nx=nx,
        ny=ny,
        dx=100.0,
        ghb=True,
        isotropic=False,
        t_isotropic_value=3000.0,
        thickness=300.0,
        width=100.0,
        recharge=1.0e-4,
        seed=123,
        mf6_seconds=1.25,
        output_dtype=np.dtype(np.float32),
    )

    def fail_if_called(**_kwargs):
        raise AssertionError("MF6 must not run for a matching cache artifact")

    loaded_heads, mf6_seconds, source = load_or_run_mf6_truth(
        truth_path=truth_path,
        workspace=tmp_path.joinpath("workspace"),
        label="4x3",
        nx=nx,
        ny=ny,
        dx=100.0,
        ghb=True,
        isotropic=False,
        t_isotropic_value=3000.0,
        thickness=300.0,
        width=100.0,
        recharge_rate=1.0e-4,
        recharge_field=recharge,
        seed=123,
        hk_field=hk,
        output_dtype=np.dtype(np.float32),
        mf6_runner=fail_if_called,
    )

    np.testing.assert_allclose(loaded_heads, heads, rtol=0.0, atol=0.0)
    assert mf6_seconds == 1.25
    assert source == "cache"


def test_load_or_run_mf6_truth_populates_missing_cache(tmp_path: Path) -> None:
    nx = 4
    ny = 3
    heads, recharge, hk = build_case_arrays(nx=nx, ny=ny)
    truth_path = tmp_path.joinpath("truth.npz")
    call_count = 0

    def fake_mf6_runner(**kwargs):
        nonlocal call_count
        call_count += 1
        assert kwargs["nx"] == nx
        assert kwargs["ny"] == ny
        np.testing.assert_array_equal(kwargs["recharge"], recharge)
        return heads, 0.75

    generated_heads, generated_seconds, generated_source = load_or_run_mf6_truth(
        truth_path=truth_path,
        workspace=tmp_path.joinpath("workspace"),
        label="4x3",
        nx=nx,
        ny=ny,
        dx=100.0,
        ghb=True,
        isotropic=False,
        t_isotropic_value=3000.0,
        thickness=300.0,
        width=100.0,
        recharge_rate=1.0e-4,
        recharge_field=recharge,
        seed=123,
        hk_field=hk,
        output_dtype=np.dtype(np.float32),
        mf6_runner=fake_mf6_runner,
    )

    assert call_count == 1
    assert truth_path.exists()
    assert generated_seconds == 0.75
    assert generated_source == "generated"
    np.testing.assert_array_equal(generated_heads, heads)

    def fail_if_called(**_kwargs):
        raise AssertionError("MF6 must not rerun after populating the cache")

    cached_heads, cached_seconds, cached_source = load_or_run_mf6_truth(
        truth_path=truth_path,
        workspace=tmp_path.joinpath("workspace"),
        label="4x3",
        nx=nx,
        ny=ny,
        dx=100.0,
        ghb=True,
        isotropic=False,
        t_isotropic_value=3000.0,
        thickness=300.0,
        width=100.0,
        recharge_rate=1.0e-4,
        recharge_field=recharge,
        seed=123,
        hk_field=hk,
        output_dtype=np.dtype(np.float32),
        mf6_runner=fail_if_called,
    )

    assert call_count == 1
    assert cached_seconds == 0.75
    assert cached_source == "cache"
    np.testing.assert_allclose(cached_heads, heads, rtol=0.0, atol=0.0)
