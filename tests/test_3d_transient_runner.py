from __future__ import annotations

import importlib.util
import py_compile
import sys
from pathlib import Path

import numpy as np
import pytest


def _load_run_3d_module():
    script_path = Path("working_tests/run_3d_warp_vs_mf6.py").resolve()
    spec = importlib.util.spec_from_file_location("run_3d_warp_vs_mf6", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_run_3d_warp_vs_mf6_script_compiles():
    script_path = Path("working_tests/run_3d_warp_vs_mf6.py")

    py_compile.compile(
        file=str(script_path),
        doraise=True,
    )


def test_transient_helper_metrics_and_comparison_do_not_require_engines(tmp_path):
    pytest.importorskip("flopy")
    run_3d = _load_run_3d_module()

    rates = run_3d.seasonal_recharge_rates(
        recharge_base=1.0e-4,
        n_periods=4,
    )
    assert rates.shape == (4,)
    assert np.all(rates > 0.0)
    assert float(rates.max()) > float(rates.min())

    mf6_heads = np.zeros((2, 2, 3, 4), dtype=np.float64)
    warp_heads = mf6_heads.copy()
    warp_heads[0, 0, 1, 1] = 0.10
    warp_heads[1, 1, 2, 2] = -0.20
    active = np.ones((2, 3, 4), dtype=np.int32)
    active[:, :, 0] = 0

    period_metrics = run_3d._transient_head_metrics(
        warp_heads=warp_heads[0],
        mf6_heads=mf6_heads[0],
        active=active,
    )
    assert period_metrics["n_active"] == int(active.sum())
    assert period_metrics["max_abs_diff"] == pytest.approx(0.10)
    assert period_metrics["rmse"] > 0.0

    mf6_path = tmp_path / "mf6_transient_heads.npz"
    warp_path = tmp_path / "warp_transient_heads.npz"
    np.savez_compressed(
        mf6_path,
        heads_per_period=mf6_heads,
        heads_final=mf6_heads[-1],
        engine_time=np.asarray(1.25, dtype=np.float64),
        total_time=np.asarray(1.50, dtype=np.float64),
    )
    np.savez_compressed(
        warp_path,
        heads_per_period=warp_heads,
        heads_final=warp_heads[-1],
        total_time=np.asarray(0.75, dtype=np.float64),
    )

    summary = run_3d.compare_transient_results(
        mf6_path=mf6_path,
        warp_path=warp_path,
        active_3d=active,
    )

    assert summary["n_periods"] == 2
    assert summary["worst_period"] == 1
    assert summary["final"]["max_abs_diff"] == pytest.approx(0.20)
    assert summary["timing"]["mf6_engine_time"] == pytest.approx(1.25)
    assert summary["timing"]["warp_total_time"] == pytest.approx(0.75)
