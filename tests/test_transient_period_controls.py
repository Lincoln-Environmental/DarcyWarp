"""Physical controls and published storage for alternate 2D period backends."""
import numpy as np
import pytest

pytest.importorskip("warp")

from DARCY_WARP_PACKAGE import create_solver

NEWTON = "unconfined_semismooth_newton_kcycle"
FAS = "unconfined_fas"
PICARD = "unconfined_picard_kcycle"


def _period_inputs(initial, *, top=10.0, sy=0.2, ss=0.0):
    ones = np.ones((4, 4))
    zeros = np.zeros_like(ones)
    return dict(
        initial_head=initial * ones, recharge_rates=[0.0], dt=1.0,
        k_field=ones, zbot_field=zeros, ztop_field=top * ones, sy=sy, ss=ss,
        active=np.ones((4, 4), dtype=np.int32), bc_mask=np.zeros((4, 4), dtype=np.int32),
        bc_values=zeros.copy(),
    )


def _build_uniform_ghb(model, inputs, stage, factor):
    ones = np.ones_like(inputs["initial_head"])
    model.build_from_fields(
        T_field=ones, R_field=np.zeros_like(ones), active=inputs["active"],
        bc_mask=inputs["bc_mask"], bc_values=inputs["bc_values"],
        gh_mask=ones, gh_head=stage * ones, gh_width=ones, gh_alpha=factor, aq_thickness=1.0,
    )


@pytest.mark.parametrize("backend", [NEWTON, FAS])
@pytest.mark.parametrize("floor_in_controls", [False, True])
def test_period_thickness_floor_matches_direct_solve(backend, floor_in_controls):
    inputs = _period_inputs(0.25)
    controls = {"fas_fallback_enabled": False} if backend == FAS else {}
    with create_solver(dim=2, nx=4, ny=4, dx=1.0, device="cpu", use_ghb=True) as model:
        _build_uniform_ghb(model, inputs, stage=0.5, factor=0.2)
        direct_head, direct = model.solve(
            formulation="unconfined", solver=backend, initial_head=inputs["initial_head"],
            K_field=inputs["k_field"], zbot_field=inputs["zbot_field"], ztop_field=inputs["ztop_field"],
            sy=inputs["sy"], ss=inputs["ss"], transient=True, dt=1.0,
            head_prev=inputs["initial_head"], min_saturated_thickness=1.0, **controls,
        )
        if floor_in_controls:
            controls["min_saturated_thickness"] = 1.0
        else:
            inputs["min_saturated_thickness"] = 1.0
        heads, info = model.solve_transient_2d_unconfined(solver=backend, solve_controls=controls, **inputs)
    # Per cell: 0.2*(h-0.25) = 0.2*(0.5-h) while the flow thickness stays at 1.
    assert direct["converged"] and info["period_infos"][0]["converged"]
    np.testing.assert_allclose(heads[0], 0.375, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(heads[0], direct_head, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(info["last_info"]["transmissivity_array"], 1.0)
    assert info["period_infos"][0]["budget_summary"]["ghb_in"] == pytest.approx(0.4, abs=1e-6)


def test_newton_keeps_thickness_floor_after_timestep_rejection():
    inputs = _period_inputs(0.25)
    with create_solver(dim=2, nx=4, ny=4, dx=1.0, device="cpu", use_ghb=True) as model:
        _build_uniform_ghb(model, inputs, stage=0.5, factor=0.2)
        heads, info = model.solve_transient_2d_unconfined(
            solver=NEWTON, min_saturated_thickness=1.0, **inputs,
            solve_controls=dict(newton_max_iterations=1, newton_max_head_change=0.06,
                                experimental_dt_min_fraction=0.25),
        )
    records = info["experimental_timestep_records"]
    assert not records[0]["accepted"]
    old = 0.25
    accepted_time = 0.0
    for record in records:
        if record["accepted"]:
            dt = record["dt"]
            old = (old + dt * 0.5) / (1.0 + dt)
            np.testing.assert_allclose(record["accepted_head"], old, rtol=0.0, atol=1e-6)
            accepted_time += dt
    assert accepted_time == pytest.approx(1.0)
    np.testing.assert_allclose(heads[-1], old, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize("backend", [NEWTON, FAS])
@pytest.mark.parametrize("ss", [0.0, 0.025])
def test_period_storage_maps_match_free_cell_budget(backend, ss):
    inputs = _period_inputs(5.0, ss=ss)
    inputs["bc_mask"][:, 0] = 1
    inputs["bc_values"][:, 0] = 5.0
    inputs["recharge_rates"] = [0.0, 0.0]
    sequence = np.stack([inputs["bc_values"], inputs["bc_values"]])
    sequence[1, :, 0] = 6.0
    if ss:
        inputs["active"][0, -1] = 0
    # Both precisions can represent these tolerances on the changing-head step.
    controls = {
        "fas_residual_rms_tolerance": 1e-5,
        "fas_head_equivalent_rms_tolerance": 1e-6,
        "fas_fallback_enabled": False,
    } if backend == FAS else {
        "newton_residual_rms_tolerance": 1e-5,
        "newton_head_equivalent_rms_tolerance": 1e-6,
        "newton_fallback_to_picard": False,
    }
    with create_solver(dim=2, nx=4, ny=4, dx=1.0, device="cpu") as model:
        _, info = model.solve_transient_2d_unconfined(
            solver=backend, **inputs, bc_values_per_period=sequence, solve_controls=controls,
        )
    non_storage = (inputs["bc_mask"] != 0) | (inputs["active"] == 0)
    for field in ("storage_terms_per_period", "sy_storage_terms_per_period", "ss_storage_terms_per_period",
                  "sy_crossing_volume_terms_per_period"):
        np.testing.assert_array_equal(info[field][:, non_storage], 0.0)
    total = info["storage_terms_per_period"]
    np.testing.assert_allclose(total, info["sy_storage_terms_per_period"] + info["ss_storage_terms_per_period"])
    for period, row in enumerate(info["period_infos"]):
        assert row["converged"]
        budget = row["budget_summary"]
        assert np.sum(total[period]) == pytest.approx(budget["storage_out"] - budget["storage_in"], abs=3e-6)
    if ss == 0:
        assert np.sum(total[1]) == pytest.approx(2.076623, abs=5e-5)


def test_fas_picard_recovery_keeps_convertible_storage():
    inputs = _period_inputs(10.0, top=8.0, sy=0.4, ss=0.025)
    with create_solver(dim=2, nx=4, ny=4, dx=1.0, device="cpu", use_ghb=True) as model:
        _build_uniform_ghb(model, inputs, stage=11.0, factor=0.0125)
        heads, info = model.solve_transient_2d_unconfined(
            solver=FAS, **inputs,
            solve_controls=dict(
                fas_max_cycles=1, fas_pre_smoothing_sweeps=0, fas_post_smoothing_sweeps=0,
                fas_coarse_smoothing_sweeps=0, fas_fallback_order=(PICARD,), fas_fallback_enabled=True,
            ),
        )
    # Above the aquifer top only Ss contributes. S=0.025*8=0.2 and C=0.0125*8=0.1.
    assert [r["backend_used"] for r in info["experimental_timestep_records"]] == [PICARD]
    assert info["period_infos"][0]["converged"]
    np.testing.assert_allclose(heads[0], (0.2 * 10.0 + 0.1 * 11.0) / 0.3, rtol=0.0, atol=4e-5)


def test_changing_chd_preserves_shared_boundary_and_bottom_input():
    inputs = _period_inputs(5.0)
    inputs["bc_mask"][:, 0] = 1
    inputs["bc_values"] = inputs["zbot_field"]
    inputs["recharge_rates"] = [0.0, 0.0]
    sequence = np.zeros((2, 4, 4))
    sequence[0, :, 0] = 5.0
    sequence[1, :, 0] = 6.0
    with create_solver(dim=2, nx=4, ny=4, dx=1.0, device="cpu") as model:
        heads, info = model.solve_transient_2d_unconfined(
            solver=NEWTON, **inputs, bc_values_per_period=sequence,
            solve_controls=dict(newton_residual_rms_tolerance=1e-5,
                                newton_head_equivalent_rms_tolerance=1e-6),
        )
    np.testing.assert_array_equal(inputs["zbot_field"], 0.0)
    np.testing.assert_allclose(heads[-1, 0], [6.0, 5.91283649, 5.85563357, 5.82730818],
                               rtol=0.0, atol=2e-5)
    assert info["period_infos"][-1]["budget_summary"]["storage_out"] == pytest.approx(2.076623, abs=5e-5)
