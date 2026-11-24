"""Reference generation and qualification checks for the f8e7683 audit."""

from multiprocessing.shared_memory import SharedMemory

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.validation.time_grid import same_step_times
from working_tests.transient_replay_metrics import compare_transient
from working_tests.transient_replay_reporting import evaluate_head_accuracy


def test_tenth_day_saved_time_grid():
    expected = 0.1 * np.arange(1, 11)
    native = np.cumsum(np.full(10, 0.1))
    assert native[-1] != expected[-1]
    assert same_step_times(saved=native, expected=expected)
    assert not same_step_times(saved=native[:-1], expected=expected)
    assert not same_step_times(saved=native[::-1], expected=expected)
    native[7] += 1e-8
    assert not same_step_times(saved=native, expected=expected)


@pytest.mark.parametrize("bad", [None, np.nan, np.inf, -np.inf])
def test_replay_scores_exact_array_and_counts_invalid_active_heads(bad):
    history = np.ones((2, 1, 3))
    active = np.array([[1, 1, 0]])
    truth = history.copy()
    history[:, 0, 2] = np.nan
    if bad is not None:
        history[0, 0, 0] = bad
        truth[1, 0, 1] = bad
    comparison = compare_transient(
        warp_result=dict(heads_per_period=history, heads_final=history[-1]),
        mf6_heads_per_period=truth, mf6_heads_final=truth[-1], active=active,
    )
    assert all(row["n_active"] == 2 for row in comparison["per_period"])
    assert evaluate_head_accuracy(comparison=comparison)["passed"] is (bad is None)
    assert comparison["final"]["rmse"] == 0.
    assert comparison["per_period"][0]["invalid_warp_active_heads"] == int(bad is not None)
    assert comparison["per_period"][1]["invalid_mf6_active_heads"] == int(bad is not None)


def test_persistent_ensemble_matches_fresh_heads_and_packages(tmp_path, pinned_mf6, monkeypatch):
    import flopy
    from DARCY_WARP_PACKAGE import modflow_truth
    from DARCY_WARP_PACKAGE.model_benchmarking_T_change import run_mf6_persistent_worker_batch_shm_T
    from DARCY_WARP_PACKAGE.model_builder import build_base_fields

    width = 30
    active, _, _, recharge = build_base_fields(nx=width, ny=width, dx=100.0)
    recharge = np.asarray(recharge, dtype=np.float64)
    conductivities = np.stack([np.full((width, width), k) for k in (10., 20., 15.)])
    t_fields = 300.0 * conductivities
    t_memory = SharedMemory(create=True, size=t_fields.nbytes)
    r_memory = SharedMemory(create=True, size=recharge.nbytes)
    saved = []
    original_extract = modflow_truth._extract_heads

    def read_package_flows(*, workspace):
        with flopy.utils.CellBudgetFile(str(workspace / "model_truth.cbb")) as budget:
            return {name: budget.get_data(text=name)[-1]["q"].copy() for name in ("CHD", "GHB", "RCHA")}

    def capture_realisation(path):
        head = original_extract(path)
        saved.append((head.copy(), read_package_flows(workspace=path.parent)))
        return head

    try:
        np.ndarray(t_fields.shape, dtype=t_fields.dtype, buffer=t_memory.buf)[:] = t_fields
        np.ndarray(recharge.shape, dtype=recharge.dtype, buffer=r_memory.buf)[:] = recharge
        monkeypatch.setattr(modflow_truth, "_extract_heads", capture_realisation)
        run_mf6_persistent_worker_batch_shm_T(
            worker_id=0, nx=width, ny=width, dx=100., T_shm_name=t_memory.name,
            T_shape=t_fields.shape, T_dtype=str(t_fields.dtype), R_shm_name=r_memory.name,
            R_shape=recharge.shape, R_dtype=str(recharge.dtype), case_start=0, case_end=len(t_fields),
            run_root=tmp_path, ghb=True, extract_heads=True,
        )
        monkeypatch.setattr(modflow_truth, "_extract_heads", original_extract)
        assert len(saved) == len(t_fields)
        for index, conductivity in enumerate(conductivities):
            workspace = tmp_path / f"fresh-{index}"
            fresh_head, _ = modflow_truth.make_mf_model(nx=width, ny=width, grid_size=100.,
                workspace=workspace, hk=conductivity, recharge=recharge, use_ghb=True, executable=pinned_mf6)
            head, flows = saved[index]
            np.testing.assert_allclose(head[active != 0], fresh_head[active != 0], rtol=0., atol=1e-5)
            for name, expected in read_package_flows(workspace=workspace).items():
                np.testing.assert_allclose(flows[name], expected, rtol=1e-9, atol=1e-7)
    finally:
        t_memory.close()
        t_memory.unlink()
        r_memory.close()
        r_memory.unlink()


def test_replay_keeps_indices_after_a_period_with_no_valid_heads():
    native = np.zeros((3, 1, 2))
    warp = native.copy()
    warp[0] = np.nan
    warp[1] = 0.5
    warp[2] = 0.1
    result = compare_transient(
        warp_result=dict(heads_per_period=warp, heads_final=warp[-1]),
        mf6_heads_per_period=native, mf6_heads_final=native[-1], active=np.ones((1, 2)),
    )
    assert result["per_period"][0]["n_active"] == 2
    assert result["per_period"][0]["invalid_warp_active_heads"] == 2
    assert result["worst_period_number_one_based"] == 2
    assert not evaluate_head_accuracy(comparison=result)["passed"]
