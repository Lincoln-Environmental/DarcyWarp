from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

from working_tests import transient_artifacts as artifact_helpers
from working_tests import transient_replay_settings as replay_settings
from working_tests import transient_replay_storage as storage_helpers


def _load_replay_module():
    return importlib.import_module("working_tests.transient_replay_support")


def _load_mf6_module():
    pytest.importorskip("flopy")
    script_path = Path("working_tests/run_2d_transient_vs_mf6.py").resolve()
    spec = importlib.util.spec_from_file_location("run_2d_transient_vs_mf6", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_artifact_confined_steady_warm_start_is_selected():
    spatial = artifact_helpers.build_synthetic_spatial_fields(nx=8, ny=6)
    warm_head = np.asarray(spatial["initial_head"], dtype=np.float64).copy()
    active = spatial["active"] != 0
    warm_head[active] += 0.25
    warm_head[spatial["bc_mask"] != 0] = spatial["bc_values"][spatial["bc_mask"] != 0]

    selected, used = artifact_helpers.select_artifact_warm_start(
        artifact={
            "initial_head": spatial["initial_head"],
            "confined_steady_head": warm_head,
        },
        spatial=spatial,
        warm_start_mode=artifact_helpers.WARM_START_CONFINED_STEADY_MF6,
    )

    assert used == artifact_helpers.WARM_START_CONFINED_STEADY_MF6
    np.testing.assert_allclose(selected, warm_head)


def test_missing_confined_steady_warm_start_errors_clearly():
    spatial = artifact_helpers.build_synthetic_spatial_fields(nx=8, ny=6)

    with pytest.raises(KeyError, match="confined_steady_head"):
        artifact_helpers.select_artifact_warm_start(
            artifact={"initial_head": spatial["initial_head"]},
            spatial=spatial,
            warm_start_mode=artifact_helpers.WARM_START_CONFINED_STEADY_MF6,
        )


def test_artifact_unconfined_steady_warm_start_is_selected():
    spatial = artifact_helpers.build_synthetic_spatial_fields(nx=8, ny=6)
    warm_head = np.asarray(spatial["initial_head"], dtype=np.float64).copy()
    active = spatial["active"] != 0
    warm_head[active] += 0.5
    warm_head[spatial["bc_mask"] != 0] = spatial["bc_values"][spatial["bc_mask"] != 0]

    selected, used = artifact_helpers.select_artifact_warm_start(
        artifact={
            "initial_head": spatial["initial_head"],
            "unconfined_steady_head": warm_head,
        },
        spatial=spatial,
        warm_start_mode=artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6,
    )

    assert used == artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6
    np.testing.assert_allclose(selected, warm_head)


def test_missing_unconfined_steady_warm_start_errors_clearly():
    spatial = artifact_helpers.build_synthetic_spatial_fields(nx=8, ny=6)

    with pytest.raises(KeyError, match="unconfined_steady_head"):
        artifact_helpers.select_artifact_warm_start(
            artifact={"initial_head": spatial["initial_head"]},
            spatial=spatial,
            warm_start_mode=artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6,
        )


def test_default_solve_controls_include_full_kcycle_and_unconfined_settings():
    controls = replay_settings.default_solve_controls()

    expected = {
        "max_cycles": 200,
        "max_levels": 4,
        "min_coarse_cells": 500,
        "nu_pre": 1,
        "nu_post": 1,
        "nu_coarse": 1,
        "check_every_no": 1,
        "max_outer_iterations": 100,
        "hclose": 1.0e-4,
        "rel_tol": 5.0e-7,
        "abs_tol_min": 5.0e-7,
        "dh_rms_tol": 1.0e-4,
        "residual_floor_tol": 1.0e-4,
        "smoother": "chebyshev",
        "omega": 0.7,
        "omega_min": 0.1,
        "omega_max": 0.9,
        "chebyshev_enabled": True,
        "chebyshev_order": 3,
        "cheby_lambda_min": 0.1,
        "cheby_lambda_max": 2.0,
        "chebyshev_reset_factor": 1.2,
        "chebyshev_rejection_factor": 1.2,
        "inner_forcing_eta": 0.10,
        "inner_head_residual_tol_min": 2.5e-6,
        "inner_head_residual_tol_max": 2.0e-4,
        "inner_picard_scale_max_fraction": 0.10,
        "transmissivity_relaxation_enabled": False,
        "unconfined_startup_mode": "confined_pre_solve",
        "unconfined_pre_solve_iterations": 3,
        "min_saturated_thickness": 0.1,
        "initial_saturated_thickness": 100.0,
        "max_head_change_per_outer_iteration": 10.0,
        "practical_picard_acceptance_enabled": True,
        "strict_head_residual_tol": 1.0e-6,
        "min_practical_outer_iterations": 8,
        "practical_head_residual_tol": 1.0e-5,
        "practical_residual_tol": 1.0e-5,
        "practical_dh_rms_tol": 3.0e-3,
        "practical_storage_diag_change_rms_tol": 30.0,
        "unconfined_inner_max_cycles_early": 10,
        "unconfined_inner_max_cycles_middle": 20,
        "unconfined_inner_max_cycles_late": 40,
        "unconfined_inner_middle_dh": 1.0,
        "unconfined_inner_late_dh": 1.0e-2,
        "adaptive_unconfined_inner_enabled": True,
        "adaptive_inner_initial_block_cycles": 5,
        "adaptive_inner_min_block_cycles": 5,
        "adaptive_inner_max_block_cycles": 20,
        "adaptive_inner_min_total_cycles": 5,
        "adaptive_inner_eta_initial": 0.05,
        "adaptive_inner_eta_min": 0.005,
        "adaptive_inner_eta_max": 0.10,
        "adaptive_inner_eta_gamma": 0.25,
        "adaptive_inner_eta_power": 1.5,
        "adaptive_inner_good_contraction_ratio": 0.40,
        "adaptive_inner_weak_contraction_ratio": 0.90,
        "adaptive_inner_stall_contraction_ratio": 0.9995,
        "adaptive_inner_divergence_contraction_ratio": 1.10,
        "adaptive_inner_stall_patience": 8,
        "adaptive_inner_minimum_usable_reduction_ratio": 0.10,
        "adaptive_inner_residual_floor": 1.0e-12,
        "adaptive_inner_relative_flow_residual_target": 1.0e-4,
        "adaptive_inner_save_block_history": False,
        "allow_unaccepted_transient_period": False,
        "use_device_transient_fast_path": True,
    }
    assert controls == expected


def test_default_artifact_path_is_formulation_specific():
    confined_path = artifact_helpers.default_artifact_path(
        formulation=artifact_helpers.FORMULATION_CONFINED,
    )
    unconfined_path = artifact_helpers.default_artifact_path(
        formulation=artifact_helpers.FORMULATION_UNCONFINED,
    )

    assert "mf6_transient_2d_confined" in str(confined_path)
    assert "mf6_transient_2d_unconfined" in str(unconfined_path)
    assert confined_path.name == artifact_helpers.DEFAULT_ARTIFACT_NAME
    assert unconfined_path.name == artifact_helpers.DEFAULT_ARTIFACT_NAME


def test_artifact_formulation_is_read_from_legacy_provenance():
    formulation = artifact_helpers.artifact_formulation(
        artifact={
            "provenance": np.asarray(
                '{"kind": "2d_unconfined_transient_mf6_truth"}',
            ),
        },
    )

    assert formulation == artifact_helpers.FORMULATION_UNCONFINED


def test_mismatched_artifact_formulation_errors_clearly(tmp_path):
    artifact_path = tmp_path / "mf6_transient_heads.npz.lzma"
    with pytest.raises(ValueError, match="does not match MF6 artifact formulation"):
        artifact_helpers.require_matching_artifact_formulation(
            artifact={"formulation": np.asarray(artifact_helpers.FORMULATION_UNCONFINED)},
            requested_formulation=artifact_helpers.FORMULATION_CONFINED,
            artifact_path=artifact_path,
        )


def test_mf6_truth_main_forwards_formulation_switch(monkeypatch, tmp_path):
    mf6_runner = _load_mf6_module()
    captured = {}

    def fake_run_mf6_transient(**kwargs):
        captured.update(kwargs)
        return tmp_path / "truth.npz.lzma"

    monkeypatch.setattr(mf6_runner, "run_mf6_transient", fake_run_mf6_transient)

    out = mf6_runner.main(
        nx=8,
        ny=6,
        n_weeks=2,
        out_path=tmp_path / "truth.npz.lzma",
        warm_start_mode="artifact_initial",
        formulation=mf6_runner.FORMULATION_CONFINED,
    )

    assert out == tmp_path / "truth.npz.lzma"
    assert captured["formulation"] == mf6_runner.FORMULATION_CONFINED
    assert captured["warm_start_mode"] == "artifact_initial"


# ---------------------------------------------------------------------------
# Storativity semantics (dimensionless S) for unconfined transient replay.
# ---------------------------------------------------------------------------

def _spatial_8x6():
    replay = _load_replay_module()
    return replay, artifact_helpers.build_synthetic_spatial_fields(nx=8, ny=6)


def test_build_unconfined_storativity_secant_sy_adds_ss_term():
    spatial = artifact_helpers.build_synthetic_spatial_fields(nx=8, ny=6)
    active = spatial["active"]
    bc_mask = spatial["bc_mask"]
    top = spatial["top"]
    bottom = spatial["bottom"]
    # Use a water-table head strictly below the model top so the secant-Sy
    # fallback recovers the full specific yield.
    head_ref = bottom + 50.0
    head_ref[bc_mask != 0] = spatial["bc_values"][bc_mask != 0]
    head_ref[active == 0] = 0.0
    free = (active != 0) & (bc_mask == 0)

    sy, ss = 0.2, 1.0e-5
    storativity, sat_ref = storage_helpers.build_unconfined_storativity(
        sy=sy,
        ss=ss,
        head_ref=head_ref,
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        include_specific_storage=True,
    )

    full_thickness = np.maximum(top - bottom, replay_settings.DEFAULT_MIN_SAT)
    expected_sat = np.clip(head_ref - bottom, replay_settings.DEFAULT_MIN_SAT, full_thickness)
    expected = sy + ss * expected_sat

    np.testing.assert_allclose(storativity[free], expected[free])
    np.testing.assert_allclose(sat_ref[free], expected_sat[free])
    assert np.all(storativity[~free] == 0.0)
    assert np.all(sat_ref[~free] == 0.0)


def test_build_unconfined_storativity_uses_physical_saturated_thickness():
    ny, nx = 1, 3
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bottom = np.full((ny, nx), 10.0)
    top = np.full((ny, nx), 110.0)  # full thickness = 100
    # dry (below bottom), water-table, confined (above top)
    head_ref = np.array([[5.0, 60.0, 200.0]])

    _, sat_ref = storage_helpers.build_unconfined_storativity(
        sy=0.2,
        ss=1.0e-5,
        head_ref=head_ref,
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        include_specific_storage=True,
    )

    np.testing.assert_allclose(sat_ref[0, 0], 0.0)                     # physically dry
    np.testing.assert_allclose(sat_ref[0, 1], 50.0)                    # head - bottom
    np.testing.assert_allclose(sat_ref[0, 2], 100.0)                   # clipped to top-bottom


def test_build_unconfined_secant_sy_below_to_below_recovers_sy():
    active = np.ones((1, 1), dtype=np.int32)
    bc_mask = np.zeros((1, 1), dtype=np.int32)
    bottom = np.array([[10.0]], dtype=np.float64)
    top = np.array([[110.0]], dtype=np.float64)

    components = storage_helpers.compute_unconfined_storage_components(
        sy=0.2,
        ss=1.0e-5,
        head_old=np.array([[60.0]], dtype=np.float64),
        head_ref=np.array([[70.0]], dtype=np.float64),
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=replay_settings.DEFAULT_MIN_SAT,
        storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    )

    np.testing.assert_allclose(components["sy_coeff"][0, 0], 0.2)


def test_build_unconfined_secant_sy_above_to_above_zeroes_sy():
    active = np.ones((1, 1), dtype=np.int32)
    bc_mask = np.zeros((1, 1), dtype=np.int32)
    bottom = np.array([[10.0]], dtype=np.float64)
    top = np.array([[110.0]], dtype=np.float64)

    components = storage_helpers.compute_unconfined_storage_components(
        sy=0.2,
        ss=1.0e-5,
        head_old=np.array([[120.0]], dtype=np.float64),
        head_ref=np.array([[130.0]], dtype=np.float64),
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=replay_settings.DEFAULT_MIN_SAT,
        storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    )

    np.testing.assert_allclose(components["sy_coeff"][0, 0], 0.0)


def test_build_unconfined_secant_sy_below_to_above_is_fractional():
    active = np.ones((1, 1), dtype=np.int32)
    bc_mask = np.zeros((1, 1), dtype=np.int32)
    bottom = np.array([[10.0]], dtype=np.float64)
    top = np.array([[110.0]], dtype=np.float64)
    head_old = np.array([[100.0]], dtype=np.float64)
    head_ref = np.array([[120.0]], dtype=np.float64)

    components = storage_helpers.compute_unconfined_storage_components(
        sy=0.2,
        ss=1.0e-5,
        head_old=head_old,
        head_ref=head_ref,
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=replay_settings.DEFAULT_MIN_SAT,
        storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    )

    expected = 0.2 * (110.0 - 100.0) / (120.0 - 100.0)
    np.testing.assert_allclose(components["sy_coeff"][0, 0], expected)
    assert 0.0 < components["sy_coeff"][0, 0] < 0.2


def test_build_unconfined_secant_sy_above_to_below_is_fractional():
    active = np.ones((1, 1), dtype=np.int32)
    bc_mask = np.zeros((1, 1), dtype=np.int32)
    bottom = np.array([[10.0]], dtype=np.float64)
    top = np.array([[110.0]], dtype=np.float64)
    head_old = np.array([[120.0]], dtype=np.float64)
    head_ref = np.array([[100.0]], dtype=np.float64)

    components = storage_helpers.compute_unconfined_storage_components(
        sy=0.2,
        ss=1.0e-5,
        head_old=head_old,
        head_ref=head_ref,
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=replay_settings.DEFAULT_MIN_SAT,
        storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    )

    expected = 0.2 * (100.0 - 110.0) / (100.0 - 120.0)
    np.testing.assert_allclose(components["sy_coeff"][0, 0], expected)
    assert 0.0 < components["sy_coeff"][0, 0] < 0.2


def test_build_unconfined_secant_sy_no_change_below_uses_sy():
    active = np.ones((1, 1), dtype=np.int32)
    bc_mask = np.zeros((1, 1), dtype=np.int32)
    bottom = np.array([[10.0]], dtype=np.float64)
    top = np.array([[110.0]], dtype=np.float64)

    components = storage_helpers.compute_unconfined_storage_components(
        sy=0.2,
        ss=1.0e-5,
        head_old=np.array([[60.0]], dtype=np.float64),
        head_ref=np.array([[60.0]], dtype=np.float64),
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=replay_settings.DEFAULT_MIN_SAT,
        storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    )

    np.testing.assert_allclose(components["sy_coeff"][0, 0], 0.2)


def test_build_unconfined_secant_sy_no_change_above_uses_zero_sy():
    active = np.ones((1, 1), dtype=np.int32)
    bc_mask = np.zeros((1, 1), dtype=np.int32)
    bottom = np.array([[10.0]], dtype=np.float64)
    top = np.array([[110.0]], dtype=np.float64)

    components = storage_helpers.compute_unconfined_storage_components(
        sy=0.2,
        ss=1.0e-5,
        head_old=np.array([[120.0]], dtype=np.float64),
        head_ref=np.array([[120.0]], dtype=np.float64),
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=replay_settings.DEFAULT_MIN_SAT,
        storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    )

    np.testing.assert_allclose(components["sy_coeff"][0, 0], 0.0)


def test_build_unconfined_secant_sy_zeroes_inactive_and_dirichlet():
    active = np.array([[1, 1, 0]], dtype=np.int32)
    bc_mask = np.array([[1, 0, 0]], dtype=np.int32)
    bottom = np.full((1, 3), 10.0, dtype=np.float64)
    top = np.full((1, 3), 110.0, dtype=np.float64)

    components = storage_helpers.compute_unconfined_storage_components(
        sy=0.2,
        ss=1.0e-5,
        head_old=np.array([[100.0, 60.0, 60.0]], dtype=np.float64),
        head_ref=np.array([[120.0, 70.0, 70.0]], dtype=np.float64),
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=replay_settings.DEFAULT_MIN_SAT,
        storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    )

    np.testing.assert_allclose(components["storage_coeff"][0, 0], 0.0)
    np.testing.assert_allclose(components["storage_coeff"][0, 2], 0.0)


def test_build_unconfined_storativity_zeroes_inactive_and_boundary():
    active = np.array([[1, 1, 0]], dtype=np.int32)   # last cell inactive
    bc_mask = np.array([[1, 0, 0]], dtype=np.int32)  # first cell boundary
    bottom = np.full((1, 3), 10.0, dtype=np.float64)
    top = np.full((1, 3), 110.0, dtype=np.float64)
    head_ref = np.full((1, 3), 60.0, dtype=np.float64)
    storativity, _ = storage_helpers.build_unconfined_storativity(
        sy=0.3,
        active=active,
        bc_mask=bc_mask,
        bottom=bottom,
        top=top,
        head_ref=head_ref,
        include_specific_storage=False,
    )
    assert storativity[0, 0] == 0.0  # boundary
    assert storativity[0, 1] == 0.3  # free
    assert storativity[0, 2] == 0.0  # inactive


# ---------------------------------------------------------------------------
# Warm-start provenance comparability.
# ---------------------------------------------------------------------------

def test_artifact_warm_start_provenance_reads_provenance():
    import json
    artifact = {
        "provenance": np.asarray(json.dumps({
            "kind": "2d_unconfined_transient_mf6_truth",
            "warm_start_mode": "confined_steady_mf6",
        })),
    }
    assert artifact_helpers.artifact_warm_start_provenance(artifact) == "confined_steady_mf6"

    assert artifact_helpers.artifact_warm_start_provenance({"provenance": None}) is None
    assert artifact_helpers.artifact_warm_start_provenance({}) is None
    assert artifact_helpers.artifact_warm_start_provenance(
        {"provenance": np.asarray("not-json")}
    ) is None


def test_validate_warm_start_comparability_errors_on_mismatch():
    with pytest.raises(ValueError, match="warm-start provenance"):
        artifact_helpers.validate_warm_start_comparability(
            artifact_warm_start="confined_steady_mf6",
            warp_warm_start_mode=artifact_helpers.WARM_START_CONFINED_STEADY_WARP,
            allow_warm_start_mismatch=False,
        )

    # artifact_initial provenance is equally incomparable with a Warp re-solve.
    with pytest.raises(ValueError, match="warm-start provenance"):
        artifact_helpers.validate_warm_start_comparability(
            artifact_warm_start="artifact_initial",
            warp_warm_start_mode=artifact_helpers.WARM_START_CONFINED_STEADY_WARP,
            allow_warm_start_mismatch=False,
        )


def test_validate_warm_start_comparability_override_must_be_explicit():
    replay = _load_replay_module()

    # Explicit override suppresses the error.
    artifact_helpers.validate_warm_start_comparability(
        artifact_warm_start="confined_steady_mf6",
        warp_warm_start_mode=artifact_helpers.WARM_START_CONFINED_STEADY_WARP,
        allow_warm_start_mismatch=True,
    )

    # Comparable Warp modes never raise, regardless of provenance.
    artifact_helpers.validate_warm_start_comparability(
        artifact_warm_start="confined_steady_mf6",
        warp_warm_start_mode=artifact_helpers.WARM_START_CONFINED_STEADY_MF6,
    )
    artifact_helpers.validate_warm_start_comparability(
        artifact_warm_start="confined_steady_mf6",
        warp_warm_start_mode=artifact_helpers.WARM_START_ARTIFACT_INITIAL,
    )
    # No provenance -> never blocks.
    artifact_helpers.validate_warm_start_comparability(
        artifact_warm_start=None,
        warp_warm_start_mode=artifact_helpers.WARM_START_CONFINED_STEADY_WARP,
    )


def _fake_artifact(spatial, formulation, warm_start_mode):
    import json
    import numpy as np
    return {
        "formulation": np.asarray(formulation),
        "provenance": np.asarray(json.dumps({
            "kind": f"2d_{formulation}_transient_mf6_truth",
            "formulation": formulation,
            "warm_start_mode": warm_start_mode,
        })),
        "initial_head": spatial["initial_head"],
        "confined_steady_head": spatial["initial_head"],
        "unconfined_steady_head": spatial["initial_head"],
        "active": spatial["active"],
        "bc_mask": spatial["bc_mask"],
        "bc_values": spatial["bc_values"],
        "top": spatial["top"],
        "bottom": spatial["bottom"],
        "k_field": spatial["k"],
        "nx": np.asarray(8, dtype=np.int32),
        "ny": np.asarray(6, dtype=np.int32),
        "dx": np.asarray(100.0, dtype=np.float64),
        "dt_days": np.asarray(7.0, dtype=np.float64),
        "sy": np.asarray(0.2),
        "ss": np.asarray(1.0e-5),
        "recharge_rates": np.array([1.0e-4], dtype=np.float64),
        "heads_per_period": spatial["initial_head"][None, :, :].copy(),
        "heads_final": spatial["initial_head"].copy(),
    }


def test_run_replay_from_artifact_rejects_nonproduction_warm_start(monkeypatch, tmp_path):
    replay, spatial = _spatial_8x6()
    artifact = _fake_artifact(spatial, "unconfined", "confined_steady_mf6")
    artifact_path = tmp_path / "mf6_transient_heads.npz.lzma"
    artifact_path.write_bytes(b"placeholder")

    monkeypatch.setattr(replay, "load_transient_artifact", lambda path: artifact)

    def boom(**kwargs):
        raise AssertionError("run_warp_transient_replay must not run on a mismatch")
    monkeypatch.setattr(replay, "run_warp_transient_replay", boom)

    with pytest.raises(ValueError, match="warm_start_mode='unconfined_steady_mf6'"):
        replay.run_replay_from_artifact(
            artifact_path=artifact_path,
            workspace=tmp_path / "ws",
            device="cpu",
            warm_start_mode=artifact_helpers.WARM_START_CONFINED_STEADY_WARP,
            formulation=artifact_helpers.FORMULATION_UNCONFINED,
            allow_warm_start_mismatch=False,
        )


def test_run_replay_from_artifact_override_emits_provenance_and_diff(monkeypatch, tmp_path):
    replay, spatial = _spatial_8x6()
    artifact = _fake_artifact(spatial, "unconfined", "unconfined_steady_mf6")
    artifact_path = tmp_path / "mf6_transient_heads.npz.lzma"
    artifact_path.write_bytes(b"placeholder")
    monkeypatch.setattr(replay, "load_transient_artifact", lambda path: artifact)

    warm_head = spatial["initial_head"] + 0.0
    captured = {}

    def fake_core(**kwargs):
        captured.update(kwargs)
        return {
            "heads_per_period": spatial["initial_head"][None, :, :].copy(),
            "heads_final": spatial["initial_head"].copy(),
            "period_times": np.array([0.0]),
            "total_time": 0.0,
            "last_info": {"converged": True},
            "storativity": np.full((6, 8), 0.2),
            "storativity_kind": "sy_plus_ss_secant_saturated_thickness",
            "include_specific_storage": True,
            "unconfined_storage_mode": artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
            "saturated_thickness_reference": None,
            "saturated_thickness_reference_source": replay_settings.STORAGE_REFERENCE_CURRENT_PICARD,
            "dt": 7.0,
            "formulation": artifact_helpers.FORMULATION_UNCONFINED,
            "solve_controls": {"unconfined_startup_mode": "confined_pre_solve"},
            "storage_reference": replay_settings.STORAGE_REFERENCE_CURRENT_PICARD,
            "warm_start_mode": artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6,
            "warm_start_used": artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6,
            "warm_start_head": warm_head.copy(),
            "device": "cpu",
        }

    monkeypatch.setattr(replay, "run_warp_transient_replay", fake_core)

    summary = replay.run_replay_from_artifact(
        artifact_path=artifact_path,
        workspace=tmp_path / "ws",
        device="cpu",
        warm_start_mode=artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6,
        formulation=artifact_helpers.FORMULATION_UNCONFINED,
        unconfined_storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    )

    assert captured["warm_start_mode"] == artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6
    assert captured["unconfined_storage_mode"] == artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY
    ws = summary["warm_start"]
    assert ws["allow_warm_start_mismatch"] is False
    assert ws["artifact_provenance"] == "unconfined_steady_mf6"
    assert ws["used"] == artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6
    assert "warm_start_vs_initial_head" in ws
    storage = summary["storage"]
    assert storage["warp_storativity_kind"] == "sy_plus_ss_secant_saturated_thickness"
    assert storage["include_specific_storage"] is True
    assert storage["unconfined_storage_mode"] == artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY
    assert "warp_storativity" in storage


def test_run_replay_from_artifact_defaults_align_with_mf6():
    import inspect
    replay = _load_replay_module()

    sig = inspect.signature(replay.run_replay_from_artifact)
    assert sig.parameters["warm_start_mode"].default == artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6
    assert sig.parameters["unconfined_storage_mode"].default == artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY
    assert sig.parameters["storage_reference"].default == replay_settings.STORAGE_REFERENCE_CURRENT_PICARD
    assert sig.parameters["allow_warm_start_mismatch"].default is False

    main_sig = inspect.signature(replay.main)
    assert main_sig.parameters["warm_start_mode"].default == artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6
    assert main_sig.parameters["unconfined_storage_mode"].default == artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY
    assert main_sig.parameters["storage_reference"].default == replay_settings.STORAGE_REFERENCE_CURRENT_PICARD


def _load_analysis_module():
    """Load the replay-analysis script as a module (no MF6/solver import at top level)."""
    script_path = Path("working_tests/run_transient_unconfined_replay_analysis.py").resolve()
    spec = importlib.util.spec_from_file_location("run_transient_unconfined_replay_analysis", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Task 8: regression tests for startup-mode propagation and JSON recording.
# ---------------------------------------------------------------------------

def test_direct_replay_consistent_with_winning_variant():
    """The direct MF6 replay defaults and the winning variant must agree on all material settings (Task 2)."""
    analysis = _load_analysis_module()
    report = analysis.check_direct_vs_winning_variant()
    assert report["consistent"] is True
    assert report["mismatches"] == []
    assert report["winning_variant"] == "production_secant_sy"
    assert report["direct_settings"]["unconfined_startup_mode"] == "confined_pre_solve"
    assert report["winning_variant_settings"]["unconfined_startup_mode"] == "confined_pre_solve"
    assert report["direct_settings"]["warm_start"] == "unconfined_steady_mf6"
    assert report["direct_settings"]["unconfined_storage_mode"] == analysis.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY


def test_production_secant_sy_settings_match_validated_defaults():
    """Production helper preserves the validated MF6-compatible replay settings."""
    replay = _load_replay_module()

    settings = replay_settings.production_secant_sy_settings()

    assert settings["unconfined_storage_mode"] == artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY
    assert settings["storage_reference"] == replay_settings.STORAGE_REFERENCE_CURRENT_PICARD
    assert settings["warm_start_mode"] == artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6
    assert settings["solve_controls"]["unconfined_startup_mode"] == "confined_pre_solve"
    assert settings["solve_controls"]["nu_pre"] == 1
    assert settings["solve_controls"]["nu_post"] == 1
    assert settings["solve_controls"]["nu_coarse"] == 1
    assert settings["solve_controls"]["max_levels"] == 4


def test_fastest_speed_variant_accepts_json_truthy_production_flag():
    """Speed-sweep recommendation accepts bool-like values from JSON summaries."""
    analysis = _load_analysis_module()

    rows = [
        {
            "variant_name": "slow",
            "runtime": 20.0,
            "production_acceptance_passed": True,
            "final_rmse": 5.0e-5,
            "final_max_abs_diff": 2.0e-4,
            "worst_period_rmse": 7.0e-5,
            "worst_period_max_abs_diff": 3.0e-4,
            "mass_balance_class": "excellent",
        },
        {
            "variant_name": "secant_sy_nu_3_coarse_1_max_levels_4",
            "runtime": 14.0,
            "production_acceptance_passed": 1,
            "final_rmse": 6.0e-5,
            "final_max_abs_diff": 3.0e-4,
            "worst_period_rmse": 7.0e-5,
            "worst_period_max_abs_diff": 3.0e-4,
            "mass_balance_class": "excellent",
        },
    ]

    fastest = analysis.fastest_speed_variant_summary(rows=rows)

    assert fastest is not None
    assert fastest["variant_name"] == "secant_sy_nu_3_coarse_1_max_levels_4"


def test_legacy_storage_variants_not_called_by_normal_execution(monkeypatch):
    """The normal variant matrix must not call the manual-only legacy matrix."""
    analysis = _load_analysis_module()

    def fail_if_called(variant_set="full"):
        raise AssertionError("legacy matrix should be manual-only")

    monkeypatch.setattr(analysis, "legacy_storage_variant_matrix_for_manual_debug_only", fail_if_called)
    configs = analysis._replay_variant_configs()

    assert "production_secant_sy" in configs


def test_compare_transient_reports_one_based_and_zero_based_worst_period():
    replay = _load_replay_module()

    warp_result = {
        "heads_per_period": np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[1.0, 3.0], [3.0, 4.0]],
            ],
            dtype=np.float64,
        ),
        "heads_final": np.asarray([[1.0, 3.0], [3.0, 4.0]], dtype=np.float64),
    }
    mf6_heads_per_period = np.asarray(
        [
            [[1.0, 2.0], [3.0, 4.0]],
            [[1.0, 2.0], [3.0, 4.0]],
        ],
        dtype=np.float64,
    )
    active = np.ones((2, 2), dtype=np.int32)

    comparison = replay.compare_transient(
        warp_result=warp_result,
        mf6_heads_per_period=mf6_heads_per_period,
        mf6_heads_final=mf6_heads_per_period[-1],
        active=active,
    )

    assert comparison["worst_period_index_zero_based"] == 1
    assert comparison["worst_period_number_one_based"] == 2
    assert comparison["worst_period"] == 2


def test_diagnose_reports_unknown_storage_budget_without_failing_head_targets():
    analysis = _load_analysis_module()

    variant_results = {
        analysis.WINNING_VARIANT_NAME: {
            "final": {"max_abs_diff": 0.001, "rmse": 0.0003},
            "period_error": [
                {"period": 1, "max_abs_diff": 0.0075, "rmse": 0.0012},
                {"period": 2, "max_abs_diff": 0.0010, "rmse": 0.0003},
            ],
            "period_convergence": {
                "periods": [
                    {
                        "period": 1,
                        "converged": True,
                        "outer_iterations": 4,
                        "picard_max_iter": 100,
                        "final_max_abs_head_change": 1.0e-5,
                    },
                    {
                        "period": 2,
                        "converged": True,
                        "outer_iterations": 3,
                        "picard_max_iter": 100,
                        "final_max_abs_head_change": 1.0e-5,
                    },
                ]
            },
            "storage_budget": {},
        }
    }

    diagnosis = analysis.diagnose(
        default_final={"max_abs_diff": 0.001, "rmse": 0.0003},
        pattern={"worst_period": 1},
        worst_cells=[],
        variant_results=variant_results,
    )

    assert diagnosis["final_period_practical_target_passed"] is True
    assert diagnosis["all_period_practical_target_passed"] is True
    assert diagnosis["nonlinear_convergence_passed"] is True
    assert diagnosis["storage_budget_diagnostics_available"] is False
    assert diagnosis["storage_budget_practical_target_passed"] is None
    assert "PASS: final-period MF6 practical target met" in diagnosis["labels"]
    assert "PASS: all-period practical target met" in diagnosis["labels"]
    assert "UNKNOWN: storage/water-budget target not evaluated" in diagnosis["labels"]


def test_variant_full_metrics_include_timing_and_storage_flags():
    analysis = _load_analysis_module()

    result = {
        "variant_name": analysis.WINNING_VARIANT_NAME,
        "final": {
            "max_abs_diff": 0.001,
            "rmse": 0.0003,
            "mean_bias_warp_minus_mf6": -1.0e-4,
            "percent_within_0_01m": 99.0,
            "percent_within_0_1m": 100.0,
        },
        "period_error": [
            {"period": 1, "max_abs_diff": 0.0075, "rmse": 0.0012, "mean_diff": 0.0},
            {"period": 2, "max_abs_diff": 0.0010, "rmse": 0.0003, "mean_diff": 0.0},
        ],
        "period_convergence": {
            "periods": [
                {"period": 1, "converged": True, "outer_iterations": 4},
                {"period": 2, "converged": True, "outer_iterations": 3},
            ]
        },
        "convergence": {},
        "timing": {
            "warp_period_time_mean": 1.25,
            "warp_period_time_max": 2.0,
        },
        "runtime": 2.5,
        "storage_budget": {
            "available": True,
            "final_storage_rmse": 0.1,
            "final_storage_max_abs": 0.2,
            "worst_period_storage_rmse": 0.15,
            "worst_period_storage_max_abs": 0.25,
            "error_by_crossing_class": {"below_to_above": {"n_cells": 3, "storage_rmse": 0.1}},
            "storage_sign_used": "warp_storage_multiplied_by_+1_before_mf6_minus_warp_comparison",
        },
    }

    metrics = analysis.variant_full_metrics(result)

    assert metrics["variant_name"] == analysis.WINNING_VARIANT_NAME
    assert metrics["final_period_practical_target_passed"] is True
    assert metrics["all_period_practical_target_passed"] is True
    assert metrics["worst_period_number"] == 1
    assert metrics["worst_period_index_zero_based"] == 0
    assert metrics["runtime"] == 2.5
    assert metrics["mean_period_runtime"] == 1.25
    assert metrics["max_period_runtime"] == 2.0
    assert metrics["storage_budget_diagnostics_available"] is True
    assert metrics["storage_sign_used"] == "warp_storage_multiplied_by_+1_before_mf6_minus_warp_comparison"


def test_compare_storage_budgets_available_with_synthetic_budget_artifacts(tmp_path):
    analysis = _load_analysis_module()

    artifact_path = tmp_path / "mf6_transient_heads.npz.lzma"
    artifact_path.write_bytes(b"placeholder")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    artifact = {
        "active": np.ones((2, 2), dtype=np.int32),
        "bc_mask": np.zeros((2, 2), dtype=np.int32),
        "top": np.full((2, 2), 10.0, dtype=np.float64),
    }

    np.savez_compressed(
        artifact_path.with_name("mf6_storage_budget_terms.npz"),
        unique_record_names=np.asarray(["STO-SS", "STO-SY"], dtype=object),
        selected_storage_record_name=np.asarray("STO-SS+STO-SY", dtype=object),
        storage_total_per_period=np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64),
    )
    np.savez_compressed(
        workspace / "warp_storage_budget_terms.npz",
        storage_terms_per_period=np.asarray([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float64),
        sy_storage_terms_per_period=np.asarray([[[0.5, 1.0], [1.5, 2.0]]], dtype=np.float64),
        ss_storage_terms_per_period=np.asarray([[[0.5, 1.0], [1.5, 2.0]]], dtype=np.float64),
        heads_old_per_period=np.asarray([[[9.0, 9.0], [9.0, 9.0]]], dtype=np.float64),
        heads_new_per_period=np.asarray([[[9.5, 9.5], [9.5, 9.5]]], dtype=np.float64),
    )

    storage_budget = analysis.compare_storage_budgets(
        artifact=artifact,
        artifact_path=artifact_path,
        workspace=workspace,
    )

    assert storage_budget["available"] is True
    assert storage_budget["storage_budget_diagnostics_available"] is True
    assert storage_budget["selected_mf6_storage_record"] == "STO-SS+STO-SY"
    assert storage_budget["storage_sign_used"] == "warp_storage_multiplied_by_+1_before_mf6_minus_warp_comparison"
    assert storage_budget["worst_period"] == 1
    assert storage_budget["worst_period_index_zero_based"] == 0
    assert storage_budget["rows"][0]["n_compared_cells"] == 4


def test_period_convergence_reports_practical_acceptance_separately_from_strict():
    replay = _load_replay_module()

    summary = replay._summarize_period_infos(
        [
            {
                "converged": True,
                "strict_picard_convergence_passed": False,
                "practical_picard_acceptance_passed": True,
                "production_acceptance_passed": True,
                "outer_iterations": 8,
            }
        ]
    )

    assert summary["all_converged"] is True
    assert summary["strict_all_converged"] is False
    assert summary["practical_all_accepted"] is True
    assert summary["production_accepted"] is True
    assert summary["first_strict_nonconverged_period"] == 1
    assert summary["first_practical_nonaccepted_period"] is None


def test_compute_replay_mass_balance_contains_all_periods_and_cumulative_totals():
    replay = _load_replay_module()

    spatial = {
        "nx": 2,
        "ny": 2,
        "dx": 1.0,
        "active": np.ones((2, 2), dtype=np.int32),
        "bc_mask": np.zeros((2, 2), dtype=np.int32),
        "bc_values": np.zeros((2, 2), dtype=np.float64),
        "top": np.full((2, 2), 10.0, dtype=np.float64),
        "bottom": np.zeros((2, 2), dtype=np.float64),
        "k": np.zeros((2, 2), dtype=np.float64),
    }
    warp_result = {
        "heads_per_period": np.asarray(
            [
                np.full((2, 2), 5.1, dtype=np.float64),
                np.full((2, 2), 4.9, dtype=np.float64),
            ]
        ),
        "warm_start_head": np.full((2, 2), 5.0, dtype=np.float64),
        "dt": 1.0,
        "unconfined_storage_mode": artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
        "storage_coeffs_per_period": np.asarray(
            [
                np.full((2, 2), 0.2, dtype=np.float64),
                np.full((2, 2), 0.2, dtype=np.float64),
            ]
        ),
        "sy_storage_coeffs_per_period": np.asarray(
            [
                np.full((2, 2), 0.2, dtype=np.float64),
                np.full((2, 2), 0.2, dtype=np.float64),
            ]
        ),
        "ss_storage_coeffs_per_period": np.zeros((2, 2, 2), dtype=np.float64),
        "storage_terms_per_period": np.asarray(
            [
                np.full((2, 2), 0.2 * 0.1, dtype=np.float64),
                np.full((2, 2), 0.2 * -0.2, dtype=np.float64),
            ]
        ),
    }

    mass_balance = replay.compute_replay_mass_balance(
        spatial=spatial,
        recharge_rates=np.asarray([0.0, 0.0], dtype=np.float64),
        sy=0.2,
        dt=1.0,
        formulation=artifact_helpers.FORMULATION_UNCONFINED,
        unconfined_storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
        warp_result=warp_result,
        min_sat=0.1,
    )

    assert mass_balance["warp_mass_balance_available"] is True
    assert len(mass_balance["per_period"]) == 2
    cumulative = mass_balance["cumulative"]
    sum_total_in = sum(float(row["total_in"]) for row in mass_balance["per_period"])
    sum_total_out = sum(float(row["total_out"]) for row in mass_balance["per_period"])
    assert cumulative["total_in_total"] == pytest.approx(sum_total_in)
    assert cumulative["total_out_total"] == pytest.approx(sum_total_out)


def test_secant_sy_mass_balance_storage_release_sign_and_small_dh_consistency():
    replay = _load_replay_module()

    spatial = {
        "nx": 1,
        "ny": 1,
        "dx": 1.0,
        "active": np.ones((1, 1), dtype=np.int32),
        "bc_mask": np.zeros((1, 1), dtype=np.int32),
        "bc_values": np.zeros((1, 1), dtype=np.float64),
        "top": np.full((1, 1), 10.0, dtype=np.float64),
        "bottom": np.zeros((1, 1), dtype=np.float64),
        "k": np.zeros((1, 1), dtype=np.float64),
    }
    warp_result = {
        "heads_per_period": np.asarray([[[5.001]], [[4.999]]], dtype=np.float64),
        "warm_start_head": np.asarray([[5.0]], dtype=np.float64),
        "dt": 1.0,
        "unconfined_storage_mode": artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
        "storage_coeffs_per_period": np.asarray([[[0.2]], [[0.2]]], dtype=np.float64),
        "sy_storage_coeffs_per_period": np.asarray([[[0.2]], [[0.2]]], dtype=np.float64),
        "ss_storage_coeffs_per_period": np.zeros((2, 1, 1), dtype=np.float64),
        "storage_terms_per_period": np.asarray([[[0.0002]], [[-0.0004]]], dtype=np.float64),
    }

    mass_balance = replay.compute_replay_mass_balance(
        spatial=spatial,
        recharge_rates=np.asarray([0.0, 0.0], dtype=np.float64),
        sy=0.2,
        dt=1.0,
        formulation=artifact_helpers.FORMULATION_UNCONFINED,
        unconfined_storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
        warp_result=warp_result,
        min_sat=0.1,
    )

    first_row = mass_balance["mass_balance_volume_sy"]["per_period"][0]
    second_row = mass_balance["mass_balance_volume_sy"]["per_period"][1]
    assert first_row["storage_out"] > 0.0
    assert first_row["storage_in"] == pytest.approx(0.0)
    assert second_row["storage_in"] > 0.0
    assert second_row["storage_out"] == pytest.approx(0.0)

    linearized_first = mass_balance["mass_balance_linearized"]["per_period"][0]["storage_release_total"]
    volume_first = mass_balance["mass_balance_volume_sy"]["per_period"][0]["storage_release_total"]
    assert linearized_first == pytest.approx(volume_first, rel=1.0e-9, abs=1.0e-12)


# ---------------------------------------------------------------------------
# Tiered mass-balance classification + production acceptance + performance
# summary (reporting/acceptance layer).
# ---------------------------------------------------------------------------

def _abs_percent_key(row):
    return abs(float(row.get("percent_discrepancy", 0.0) or 0.0))


def _mb_row(period, percent_discrepancy):
    return {
        "period": int(period),
        "recharge_in": 1.0, "recharge_out": 0.0,
        "chd_in": 1.0, "chd_out": 1.0,
        "ghb_in": 0.0, "ghb_out": 0.0,
        "storage_in": 0.0, "storage_out": 0.0,
        "total_in": 2.0, "total_out": 2.0,
        "in_minus_out": 0.0,
        "percent_discrepancy": float(percent_discrepancy),
        "throughflow": 2.0,
        "imbalance_fraction": float(percent_discrepancy) / 100.0,
    }


def _synth_mass_balance(discrepancies, cumulative_pct):
    rows = [_mb_row(i + 1, d) for i, d in enumerate(discrepancies)]
    return {
        "warp_mass_balance_available": True,
        "per_period": rows,
        "cumulative": {"percent_discrepancy": float(cumulative_pct), "n_periods": len(rows)},
        "worst_period": max(rows, key=_abs_percent_key),
        "max_abs_percent_discrepancy": max(abs(float(d)) for d in discrepancies),
    }


def _period_conv(strict_all, practical_all, first_strict_fail=1):
    return {
        "strict_all_converged": bool(strict_all),
        "practical_all_accepted": bool(practical_all),
        "production_accepted": bool(practical_all),
        "first_strict_nonconverged_period": first_strict_fail if not strict_all else None,
        "first_practical_nonaccepted_period": None if practical_all else 1,
        "periods": [],
    }


def test_mass_balance_startup_warning_passes_for_period_1_elevated():
    replay = _load_replay_module()
    # Period 1 = 0.10566% (elevated), periods 2-10 tiny, cumulative 0.0094%.
    discrepancies = [0.10566] + [0.0001] * 9
    mb = _synth_mass_balance(discrepancies, cumulative_pct=0.0094)
    classification = replay.classify_replay_mass_balance(mb)
    assert classification["mass_balance_class"] == "startup_warning"
    assert classification["mass_balance_passed"] is True
    assert classification["warnings"], "startup warning text should be emitted"


def test_mass_balance_fails_when_cumulative_discrepancy_too_large():
    replay = _load_replay_module()
    mb = _synth_mass_balance([0.0001] * 10, cumulative_pct=0.15)  # cumulative >= 0.1%
    classification = replay.classify_replay_mass_balance(mb)
    assert classification["mass_balance_class"] == "fail"
    assert classification["mass_balance_passed"] is False


def test_mass_balance_fails_when_startup_discrepancy_ge_0p2():
    replay = _load_replay_module()
    mb = _synth_mass_balance([0.25] + [0.0001] * 9, cumulative_pct=0.0094)
    classification = replay.classify_replay_mass_balance(mb)
    assert classification["mass_balance_class"] == "fail"
    assert classification["mass_balance_passed"] is False


def test_mass_balance_fails_when_nonstartup_discrepancy_ge_0p01():
    replay = _load_replay_module()
    mb = _synth_mass_balance([0.0001, 0.02] + [0.0001] * 8, cumulative_pct=0.0094)
    classification = replay.classify_replay_mass_balance(mb)
    assert classification["mass_balance_class"] == "fail"
    assert classification["mass_balance_passed"] is False


def test_mass_balance_excellent_when_all_periods_tight():
    replay = _load_replay_module()
    mb = _synth_mass_balance([0.0001] * 10, cumulative_pct=0.0002)
    classification = replay.classify_replay_mass_balance(mb)
    assert classification["mass_balance_class"] == "excellent"
    assert classification["mass_balance_passed"] is True


def test_annotate_mass_balance_classification_adds_per_period_class():
    replay = _load_replay_module()
    mb = _synth_mass_balance([0.10566, 0.0001, 0.00005], cumulative_pct=0.0094)
    replay.annotate_mass_balance_classification(mb)
    assert mb["mass_balance_class"] == "startup_warning"
    assert mb["mass_balance_passed"] is True
    assert mb["cumulative_percent_discrepancy"] == pytest.approx(0.0094)
    assert mb["per_period"][0]["mass_balance_class"] == "startup_warning"
    assert mb["per_period"][1]["mass_balance_class"] == "excellent"


def test_evaluate_head_accuracy_pass_and_fail():
    replay = _load_replay_module()
    good = {
        "final": {"rmse": 0.0003, "max_abs_diff": 0.0013},
        "per_period": [
            {"rmse": 0.0012, "max_abs_diff": 0.0067, "percent_within_0_01m": 100.0},
            {"rmse": 0.0003, "max_abs_diff": 0.0013, "percent_within_0_01m": 100.0},
        ],
    }
    assert replay.evaluate_head_accuracy(good)["passed"] is True
    bad = {
        "final": {"rmse": 0.002, "max_abs_diff": 0.0013},  # final rmse too large
        "per_period": [
            {"rmse": 0.0012, "max_abs_diff": 0.0067, "percent_within_0_01m": 100.0},
        ],
    }
    assert replay.evaluate_head_accuracy(bad)["passed"] is False


def test_evaluate_method_settings_validated_vs_deviation():
    replay = _load_replay_module()
    valid = replay.evaluate_method_settings(
        unconfined_storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
        storage_reference=replay_settings.STORAGE_REFERENCE_CURRENT_PICARD,
        unconfined_startup_mode="confined_pre_solve",
        warm_start=artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6,
    )
    assert valid["passed"] is True
    deviated = replay.evaluate_method_settings(
        unconfined_storage_mode=artifact_helpers.UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
        storage_reference="previous_period",
        unconfined_startup_mode="confined_pre_solve",
        warm_start=artifact_helpers.WARM_START_UNCONFINED_STEADY_MF6,
    )
    assert deviated["passed"] is False
    assert "storage_reference" in deviated["mismatches"]


def test_production_acceptance_passes_with_strict_period1_warning():
    replay = _load_replay_module()
    method = {"passed": True, "settings": {}, "mismatches": {}}
    head = {"passed": True}
    mb = _synth_mass_balance([0.10566] + [0.0001] * 9, cumulative_pct=0.0094)
    replay.annotate_mass_balance_classification(mb)
    # strict fails in period 1, practical passes -> production still accepted.
    acceptance = replay.build_production_acceptance(
        method_settings=method,
        head_accuracy=head,
        mass_balance=mb,
        period_convergence=_period_conv(strict_all=False, practical_all=True, first_strict_fail=1),
    )
    assert acceptance["strict_picard_convergence_passed"] is False
    assert acceptance["practical_picard_acceptance_passed"] is True
    assert acceptance["mass_balance_passed"] is True
    assert acceptance["production_acceptance_passed"] is True
    assert any("strict Picard convergence failed" in w for w in acceptance["warnings"])


def test_production_acceptance_fails_when_mass_balance_fails():
    replay = _load_replay_module()
    method = {"passed": True, "settings": {}, "mismatches": {}}
    head = {"passed": True}
    mb = _synth_mass_balance([0.25] + [0.0001] * 9, cumulative_pct=0.0094)  # startup >= 0.2%
    replay.annotate_mass_balance_classification(mb)
    acceptance = replay.build_production_acceptance(
        method_settings=method,
        head_accuracy=head,
        mass_balance=mb,
        period_convergence=_period_conv(strict_all=False, practical_all=True),
    )
    assert acceptance["mass_balance_passed"] is False
    assert acceptance["production_acceptance_passed"] is False


def test_production_acceptance_fails_when_head_target_fails():
    replay = _load_replay_module()
    method = {"passed": True, "settings": {}, "mismatches": {}}
    head = {"passed": False}
    mb = _synth_mass_balance([0.0001] * 10, cumulative_pct=0.0002)
    replay.annotate_mass_balance_classification(mb)
    acceptance = replay.build_production_acceptance(
        method_settings=method,
        head_accuracy=head,
        mass_balance=mb,
        period_convergence=_period_conv(strict_all=True, practical_all=True),
    )
    assert acceptance["head_accuracy_passed"] is False
    assert acceptance["production_acceptance_passed"] is False


def test_performance_summary_contains_required_keys():
    replay = _load_replay_module()
    timing = {
        "warp_total_time": 27.0,
        "mf6_transient_total_time": 105.0,
        "mf6_engine_time_including_warm_start": 139.0,
        "warp_period_1_time": 3.1,
        "warp_period_time_mean": 2.7,
        "warp_period_time_mean_excluding_period_1": 2.6,
        "warp_period_time_max": 3.1,
        "warp_period_time_sum": 27.0,
    }
    period_conv = {
        "periods": [
            {"outer_iterations": 8},
            {"outer_iterations": 6},
            {"outer_iterations": 6},
        ],
    }
    solve_settings = {"nu_pre": 15, "nu_post": 15, "nu_coarse": 3, "omega": 0.7, "max_cycles": 200, "smoother": "chebyshev"}
    perf = replay.build_performance_summary(
        timing=timing,
        period_convergence=period_conv,
        solve_settings=solve_settings,
        mass_balance_runtime=0.4,
        profile=None,
    )
    for key in ("warp_total_time", "period_1_runtime", "speedup_vs_mf6_transient", "total_outer_iterations"):
        assert key in perf
    assert perf["speedup_vs_mf6_transient"] == pytest.approx(105.0 / 27.0, rel=1.0e-6)
    assert perf["total_outer_iterations"] == 20
    assert perf["period_1_outer_iterations"] == 8
    assert perf["profile_available"] is False
    assert perf["profile_reason"] == "category timing not yet instrumented"


def test_default_run_config_records_production_defaults():
    replay = _load_replay_module()
    cfg = replay_settings.default_run_config(device="cuda:0")
    assert cfg["run_mode"] == "production"
    assert cfg["compute_mass_balance"] is True
    assert cfg["profile_performance"] is False
    assert cfg["save_heavy_diagnostics"] is False
    assert cfg["run_replay_matrix"] is False
    assert cfg["device"] == "cuda:0"
