"""Production-scale validation of all 2D unconfined solver backends.

Runs the 500x500, 52-week homogeneous transient case (uniform K=100 m/day)
against the MF6 truth artifact for every unconfined backend:

* ``unconfined_picard_kcycle`` — production default, full replay harness
  (production acceptance, head accuracy, mass balance, runtime),
* ``unconfined_semismooth_newton_kcycle`` — production alternative via the
  alternate nonlinear multi-period driver (head accuracy, budget closure,
  runtime),
* ``unconfined_fas`` — experimental, through the same compatibility driver.

The MF6 truth artifact is generated on first use by
``run_2d_transient_warp_replay.ensure_case_artifact`` and cached under
``DARCY_WARP_PACKAGE/data/working_tests/``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("WARP_CACHE_PATH", "/tmp/darcywarp-warp-cache")

wp = pytest.importorskip("warp")

from working_tests.run_2d_transient_warp_replay import (  # noqa: E402
    build_case_setup,
    ensure_case_artifact,
    run_production_replay,
)
from working_tests.transient_artifacts import (  # noqa: E402
    WARM_START_UNCONFINED_STEADY_MF6,
    load_transient_artifact,
    select_artifact_warm_start,
    spatial_fields_from_artifact,
)
from working_tests.transient_replay_metrics import compare_transient  # noqa: E402
from working_tests.transient_replay_reporting import evaluate_head_accuracy  # noqa: E402

GRID_NX = 500
GRID_NY = 500
N_PERIODS = 52
T_FIELD_KIND = "homogeneous"

ALTERNATE_NONLINEAR_BACKENDS = (
    "unconfined_semismooth_newton_kcycle",
    "unconfined_fas",
)


@pytest.fixture(scope="module")
def case_artifact(tmp_path_factory) -> Path:
    setup = build_case_setup(
        nx=GRID_NX,
        ny=GRID_NY,
        n_periods=N_PERIODS,
        t_field_kind=T_FIELD_KIND,
    )
    original = setup["artifact_path"]
    if original.is_file():
        from working_tests.run_2d_transient_vs_mf6 import MF6_MODEL_NAME
        base = original.parent / "mf6" / MF6_MODEL_NAME
        if not all(base.with_suffix(ext).is_file() for ext in (".cbb", ".hds", ".dis.grb")):
            # A portable heads-only copy cannot verify native flows. Generate a
            # matching complete run without replacing the existing head fixture.
            setup["artifact_path"] = tmp_path_factory.mktemp("mf6_face_reference") / original.name
    return ensure_case_artifact(setup)


@pytest.fixture(scope="module")
def loaded_case(case_artifact):
    artifact = load_transient_artifact(case_artifact)
    spatial = spatial_fields_from_artifact(artifact)
    warm_head, _ = select_artifact_warm_start(
        artifact=artifact,
        spatial=spatial,
        warm_start_mode=WARM_START_UNCONFINED_STEADY_MF6,
    )
    return {
        "artifact": artifact,
        "folder": case_artifact.parent,
        "spatial": spatial,
        "warm_head": warm_head,
        "sy": float(artifact["sy"]),
        "ss": float(artifact["ss"]),
        "dt": float(artifact["dt_days"]),
        "rates": np.asarray(artifact["recharge_rates"], dtype=np.float64),
        "mf6_engine_time": float(artifact["engine_time"]) if "engine_time" in artifact else None,
    }


def _device() -> str:
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _report_perf(name: str, warp_seconds: float, mf6_engine_time: float | None) -> None:
    if mf6_engine_time and mf6_engine_time > 0.0:
        print(f"\n[{name}] warp total {warp_seconds:.2f} s vs MF6 engine {mf6_engine_time:.2f} s "
              f"(speedup {mf6_engine_time / warp_seconds:.2f}x)")
    else:
        print(f"\n[{name}] warp total {warp_seconds:.2f} s")


def _check_native_faces(*, loaded_case, heads, backend, period_infos):
    from working_tests.transient_face_flow_audit import compare_native_face_flows

    assert len(period_infos) == len(heads)
    step_key = ("adaptive_dt_substep_dts" if backend == "unconfined_picard_kcycle"
                else "experimental_substep_dts")
    for period, row in enumerate(period_infos, start=1):
        assert row["solver_backend"] == backend, (period, row["solver_backend"])
        assert row["converged"], (period, backend)
        assert row[step_key] == [loaded_case["dt"]], (period, row[step_key])
        if backend != "unconfined_picard_kcycle":
            assert row["experimental_fallback_timestep_count"] == 0, (period, backend)
    folder = loaded_case["folder"]
    report = compare_native_face_flows(native_workspace=folder / "mf6", heads=heads,
        artifact=loaded_case["artifact"])
    report["solver_backend"] = backend
    for metrics, row in zip(report["periods"], period_infos, strict=True):
        metrics["accepted_step_days"] = row[step_key]
    path = folder / f"{backend}_face_flows.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{backend}: compared {report['compared_faces']} faces at {len(report['periods'])} endpoints; "
          f"max relative L2 {report['worst_relative_l2']:.6g}. {path}")
    assert len(report["periods"]) == N_PERIODS
    assert report["worst_relative_l2"] <= 0.01


def test_picard_production_replay_500x500(case_artifact, loaded_case):
    """Production backend: full acceptance, head accuracy, mass balance, runtime."""
    summary = run_production_replay(
        artifact_path=case_artifact,
        workspace=None,
        device=_device(),
    )
    timing = summary.get("timing") or {}
    warp_total = float(timing.get("warp_total_time", float("nan")))
    _report_perf("unconfined_picard_kcycle", warp_total, (summary.get("timing") or {}).get("mf6_engine_time"))

    production = summary.get("production_acceptance") or {}
    head_accuracy = summary.get("head_accuracy") or {}
    mass_balance = summary.get("mass_balance") or {}
    assert head_accuracy.get("passed", False), head_accuracy
    assert mass_balance.get("mass_balance_passed", False), mass_balance
    assert str(mass_balance.get("mass_balance_class", "")).strip().lower() in {
        "excellent", "good", "acceptable",
    }
    assert production.get("production_acceptance_passed", False), production
    with np.load(case_artifact.parent / "warp_transient_heads.npz") as saved:
        _check_native_faces(loaded_case=loaded_case, heads=saved["heads_per_period"], backend="unconfined_picard_kcycle",
            period_infos=json.loads(str(saved["period_infos"])))
    assert np.isfinite(warp_total) and warp_total > 0.0


@pytest.mark.parametrize("backend", ALTERNATE_NONLINEAR_BACKENDS)
def test_alternate_nonlinear_backend_500x500(loaded_case, backend):
    """Alternate nonlinear backends run complete multi-period simulations."""
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    spatial = loaded_case["spatial"]
    artifact = loaded_case["artifact"]
    sy = loaded_case["sy"]
    ss = loaded_case["ss"]
    dt = loaded_case["dt"]
    rates = loaded_case["rates"]

    # This reference uses one backward-Euler step per week. Give Newton
    # enough Krylov iterations and a tight residual to close the budget
    # on that step. A different backend or a shorter step cannot qualify it.
    solve_controls = {
        "save_transient_diagnostics": True,
        "experimental_retry_enabled": False,
    }
    if backend == "unconfined_semismooth_newton_kcycle":
        solve_controls.update({
            "newton_head_equivalent_rms_tolerance": 1.0e-8,
            "newton_residual_rms_tolerance": 1.0e-7,
            "newton_fgmres_max_iterations": 240,
            "newton_fallback_to_picard": False,
        })
    else:
        solve_controls["fas_fallback_enabled"] = False

    solver = WarpDarcySolver(
        nx=int(spatial["nx"]),
        ny=int(spatial["ny"]),
        dx=float(spatial["dx"]),
        device=_device(),
        solver_type="kcycle",
        diag_preconditioner_backend="device",
    )
    try:
        t0 = time.perf_counter()
        heads, info = solver.solve_transient_2d_unconfined(
            solver=backend,
            initial_head=loaded_case["warm_head"],
            recharge_rates=rates,
            k_field=spatial["k"],
            zbot_field=spatial["bottom"],
            ztop_field=spatial["top"],
            sy=sy,
            ss=ss,
            dt=dt,
            active=spatial["active"],
            bc_mask=spatial["bc_mask"],
            bc_values=spatial["bc_values"],
            storage_mode="mf6_convertible_secant_sy",
            storage_reference="current_picard",
            solve_controls=solve_controls,
            return_info=True,
        )
        try:
            wp.synchronize_device(str(solver.device_str))
        except Exception:
            pass
        warp_total = time.perf_counter() - t0
    finally:
        solver.close()
    _report_perf(backend, warp_total, loaded_case["mf6_engine_time"])

    assert info["solver_backend"] == backend
    assert heads.shape == artifact["heads_per_period"].shape
    assert np.all(np.isfinite(heads))
    # Every stress period was attempted and accepted (no forced accepts, no
    # hard failure) with per-timestep budget closure.
    assert len(info["period_infos"]) == N_PERIODS
    assert all(p["converged"] for p in info["period_infos"])
    assert info["transient_replay_counters"]["experimental_forced_accept_count"] == 0
    assert info["transient_replay_counters"]["experimental_fallback_timestep_count"] == 0
    for period_budget in info["experimental_period_budgets"]:
        assert abs(period_budget["percent_discrepancy"]) < 0.1

    comparison = compare_transient(
        {"heads_per_period": heads, "heads_final": heads[-1]},
        artifact["heads_per_period"],
        artifact["heads_final"],
        spatial["active"],
    )
    accuracy = evaluate_head_accuracy(comparison)
    print(f"[{backend}] final RMSE {accuracy['final_rmse']:.3g} m, "
          f"worst-period RMSE {accuracy['worst_period_rmse']:.3g} m, "
          f"retries {info['transient_replay_counters']['experimental_retry_count']}, "
          f"fallbacks {info['transient_replay_counters']['experimental_fallback_timestep_count']}")
    assert accuracy["passed"], accuracy["checks"]
    _check_native_faces(loaded_case=loaded_case, heads=heads, backend=backend, period_infos=info["period_infos"])
    assert np.isfinite(warp_total) and warp_total > 0.0
