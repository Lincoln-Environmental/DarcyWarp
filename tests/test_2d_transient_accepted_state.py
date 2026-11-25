"""Accepted-state ownership regressions for the 2D transient Picard driver.

On the fixed-step FP64 Warp CPU device path the downloaded trial head can be
a view of the live ``h_iter_wp`` solver buffer. An accepted period must
commit an owned head so that the next period's physical budget check compares
distinct head fields, and a later rejected trial cannot leak into
``heads_final``. The analytical case below has a no-flow perimeter and a
spatially uniform response, so ``dh = recharge * dt / Sy`` is exact and the
storage release follows from the budget convention documented in
``evaluate_picard_transient_balance``. No MF6 installation is required.
"""

import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))

NX = 8
NY = 6
CELL_COUNT = NX * NY
DX = 10.0
CELL_AREA = DX * DX
SY = 0.2
SS = 0.0
K_VALUE = 1.0
DT_DAYS = 2.0
INITIAL_HEAD = 5.0
BOTTOM_ELEVATION = 0.0
TOP_ELEVATION = 10.0
RECHARGE_FILL = 0.001
RECHARGE_DRAIN = -0.0005
RECHARGE_IMPOSSIBLE = -1.0
DAY_TWO_HEAD = INITIAL_HEAD + RECHARGE_FILL * DT_DAYS / SY
DAY_FOUR_HEAD = DAY_TWO_HEAD + RECHARGE_DRAIN * DT_DAYS / SY
HEAD_ATOL = 1.0e-6
STORAGE_ATOL = 1.0e-3
# Documented convention of evaluate_picard_transient_balance: the storage
# term is negative when the aquifer drains, and that negative storage_volume
# is reported as storage_in (release from storage into the cells).
EXPECTED_FILL_STORAGE_OUT = SY * (DAY_TWO_HEAD - INITIAL_HEAD) / DT_DAYS * CELL_AREA * CELL_COUNT
EXPECTED_DRAIN_STORAGE_IN = SY * (DAY_TWO_HEAD - DAY_FOUR_HEAD) / DT_DAYS * CELL_AREA * CELL_COUNT

VARIANT_CONTROLS = {
    "host_picard": {"use_device_transient_fast_path": False},
    "classic_device": {
        "use_device_transient_fast_path": True,
        "transient_face_operator_enabled": False,
    },
    "face": {
        "use_device_transient_fast_path": True,
        "transient_face_operator_enabled": True,
    },
    "face_incremental": {
        "use_device_transient_fast_path": True,
        "transient_face_operator_enabled": True,
        "use_incremental_picard": True,
    },
    "face_mixed_precision": {
        "use_device_transient_fast_path": True,
        "transient_face_operator_enabled": True,
        "transient_mixed_precision_enabled": True,
    },
}
DEVICE_VARIANT_IDS = ["classic_device", "face", "face_incremental", "face_mixed_precision"]


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _warp_available(), reason="warp is not available")


def _skip_unless_fp64_fields() -> None:
    """Skip device-path cases when the model fields are not FP64.

    The transient face, mixed-precision and graph kernels require
    ``DARCY_FLOAT=float64``; the FP32 fields raise on those variants before
    the accepted-state path is even reached.
    """
    import warp as wp

    from DARCY_WARP_PACKAGE import warped_darcy

    if warped_darcy.WP_FLOAT != wp.float64:
        pytest.skip("device fast path requires DARCY_FLOAT=float64")


def build_case_solver():
    """Build the fully active no-flow solver for the analytical case.

    The grid is 6 rows x 8 columns of 10 m square cells with a no-flow
    perimeter, no prescribed heads, bottom at 0 m, top at 10 m, uniform
    initial head 5 m, K = 1 m/day, Sy = 0.2 and Ss = 0.

    :returns: Tuple of the solver, the hydraulic conductivity field, the
        bottom elevation field, the top elevation field, and the uniform
        initial head field.
    """
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=NX, ny=NY, dx=DX, device="cpu",
        use_ghb=False, solver_type="kcycle", diag_preconditioner_backend="host",
    )
    active = np.ones((NY, NX), dtype=np.int32)
    bc_mask = np.zeros((NY, NX), dtype=np.int32)
    bc_values = np.zeros((NY, NX), dtype=np.float64)
    solver.build_from_fields(
        T_field=np.full((NY, NX), K_VALUE * INITIAL_HEAD, dtype=np.float64),
        R_field=np.zeros((NY, NX), dtype=np.float64),
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
    )
    k_field = np.full((NY, NX), K_VALUE, dtype=np.float64)
    zbot_field = np.full((NY, NX), BOTTOM_ELEVATION, dtype=np.float64)
    ztop_field = np.full((NY, NX), TOP_ELEVATION, dtype=np.float64)
    initial_head = np.full((NY, NX), INITIAL_HEAD, dtype=np.float64)
    return solver, k_field, zbot_field, ztop_field, initial_head


def solve_two_periods(*, recharge_rates, controls_override, allow_rejected_trial):
    """Run the public transient entry point with adaptive timestepping off.

    :param recharge_rates: One recharge rate in m/day per fixed stress period.
    :param controls_override: Solve controls merged over the production
        defaults to select the device variant under test.
    :param allow_rejected_trial: Whether an unaccepted period is reported as
        a rejected trial instead of raising.
    :returns: Tuple of the per-period head array and the driver info mapping
        returned by ``solve_transient_2d_unconfined``.
    """
    from working_tests.transient_replay_settings import default_solve_controls

    solver, k_field, zbot_field, ztop_field, initial_head = build_case_solver()
    solve_controls = default_solve_controls()
    solve_controls["adaptive_dt_enabled"] = False
    solve_controls["allow_unaccepted_transient_period"] = bool(allow_rejected_trial)
    solve_controls.update(controls_override)
    heads, info = solver.solve_transient_2d_unconfined(
        initial_head=initial_head,
        recharge_rates=np.asarray(recharge_rates, dtype=np.float64),
        k_field=k_field,
        zbot_field=zbot_field,
        ztop_field=ztop_field,
        sy=SY,
        ss=SS,
        dt=DT_DAYS,
        storage_mode="mf6_convertible_secant_sy",
        storage_reference="current_picard",
        solve_controls=solve_controls,
        return_info=True,
    )
    return heads, info


def assert_variant_engaged(*, variant_id, info):
    """Assert the requested device variant actually ran the periods.

    :param variant_id: Identifier from ``VARIANT_CONTROLS``.
    :param info: Driver info mapping returned by the public entry point.
    """
    fast_path_counter = int(
        info["transient_replay_counters"]["device_side_picard_fast_path_active"]
    )
    if variant_id == "host_picard":
        assert fast_path_counter == 0
        return
    assert fast_path_counter == 1
    first_period = info["period_infos"][0]
    assert first_period["device_side_picard_fast_path_active"] is True
    assert first_period["transient_face_operator"] is (
        variant_id != "classic_device"
    )
    if variant_id == "face_incremental":
        assert first_period["incremental_picard_enabled"] is True
    if variant_id == "face_mixed_precision":
        assert first_period["transient_mixed_precision"] is True


def assert_two_period_analytical_outcome(*, heads, info):
    """Assert the committed two-period analytical outcome.

    Day 2 must be 5.010 m, day 4 must be 5.005 m, both periods accepted,
    ``accepted_time`` 4 days, and ``heads_final`` the day-4 head. The second
    period must release 2.4 m^3/day from storage with a closed budget, per
    the signed convention of ``evaluate_picard_transient_balance``.

    :param heads: Per-period head array returned by the public entry point.
    :param info: Driver info mapping returned by the public entry point.
    """
    period_infos = info["period_infos"]
    assert len(period_infos) == 2
    assert period_infos[0]["state"] == "accepted"
    assert period_infos[1]["state"] == "accepted"
    assert period_infos[0]["converged"] is True
    assert period_infos[1]["converged"] is True
    assert info["accepted_periods"] == 2
    assert info["accepted_time"] == pytest.approx(2 * DT_DAYS)

    np.testing.assert_allclose(heads[0], DAY_TWO_HEAD, atol=HEAD_ATOL)
    np.testing.assert_allclose(heads[1], DAY_FOUR_HEAD, atol=HEAD_ATOL)
    heads_final = np.asarray(info["heads_final"], dtype=np.float64)
    np.testing.assert_allclose(heads_final, DAY_FOUR_HEAD, atol=HEAD_ATOL)
    np.testing.assert_allclose(heads_final, heads[1], atol=0.0)

    fill_balance = period_infos[0]["physical_period_balance"]
    drain_balance = period_infos[1]["physical_period_balance"]
    assert fill_balance["accepted"] is True
    assert drain_balance["accepted"] is True
    assert abs(float(fill_balance["percent_discrepancy"])) < 0.01
    assert abs(float(drain_balance["percent_discrepancy"])) < 0.01
    assert float(fill_balance["storage_in"]) == pytest.approx(0.0, abs=STORAGE_ATOL)
    assert float(fill_balance["storage_out"]) == pytest.approx(
        EXPECTED_FILL_STORAGE_OUT, abs=STORAGE_ATOL
    )
    assert float(drain_balance["storage_in"]) == pytest.approx(
        EXPECTED_DRAIN_STORAGE_IN, abs=STORAGE_ATOL
    )
    assert float(drain_balance["storage_out"]) == pytest.approx(0.0, abs=STORAGE_ATOL)


@pytest.mark.parametrize("variant_id", ["host_picard"])
def test_host_picard_accepts_analytical_two_period_storage_response(variant_id):
    """Check the host Picard control commits both analytical periods.

    :param variant_id: Identifier from ``VARIANT_CONTROLS``.
    """
    heads, info = solve_two_periods(
        recharge_rates=(RECHARGE_FILL, RECHARGE_DRAIN),
        controls_override=VARIANT_CONTROLS[variant_id],
        allow_rejected_trial=False,
    )
    assert_variant_engaged(variant_id=variant_id, info=info)
    assert_two_period_analytical_outcome(heads=heads, info=info)


@pytest.mark.parametrize("variant_id", DEVICE_VARIANT_IDS)
def test_device_fast_path_accepts_analytical_two_period_storage_response(variant_id):
    """Check the FP64 Warp CPU device variants commit both analytical periods.

    Covers the classic device operator, the face operator, the incremental
    face variant, and the fast mixed-precision implementation.

    :param variant_id: Identifier from ``VARIANT_CONTROLS``.
    """
    _skip_unless_fp64_fields()
    heads, info = solve_two_periods(
        recharge_rates=(RECHARGE_FILL, RECHARGE_DRAIN),
        controls_override=VARIANT_CONTROLS[variant_id],
        allow_rejected_trial=False,
    )
    assert_variant_engaged(variant_id=variant_id, info=info)
    assert_two_period_analytical_outcome(heads=heads, info=info)


@pytest.mark.parametrize("variant_id", list(VARIANT_CONTROLS))
def test_rejected_trial_keeps_first_period_committed_state(variant_id):
    """Check a genuinely failing second period cannot move the committed state.

    Period one fills to the analytical day-2 head and is accepted. Period two
    demands more water than storage can supply (recharge −1 m/day with the
    dry-cell conductance floor), so strict acceptance genuinely fails within
    the fixed budget. With rejected-trial reporting enabled the trial is
    reported, ``accepted_periods``/``accepted_time`` still refer to period
    one, and ``heads_final`` equals the saved first-period head. The case
    runs the real Warp buffers through the public entry point.

    :param variant_id: Identifier from ``VARIANT_CONTROLS``.
    """
    if variant_id != "host_picard":
        _skip_unless_fp64_fields()
    controls_override = dict(VARIANT_CONTROLS[variant_id])
    controls_override["practical_picard_acceptance_enabled"] = False
    heads, info = solve_two_periods(
        recharge_rates=(RECHARGE_FILL, RECHARGE_IMPOSSIBLE),
        controls_override=controls_override,
        allow_rejected_trial=True,
    )
    period_infos = info["period_infos"]
    assert len(period_infos) == 2
    assert period_infos[0]["state"] == "accepted"
    assert period_infos[0]["converged"] is True
    assert period_infos[1]["state"] == "rejected_trial"
    assert period_infos[1]["converged"] is False
    assert info["accepted_periods"] == 1
    assert info["accepted_time"] == pytest.approx(DT_DAYS)

    np.testing.assert_allclose(heads[0], DAY_TWO_HEAD, atol=HEAD_ATOL)
    # The trial head is a genuinely different field and stays available for
    # rejected-trial diagnostics.
    assert np.all(np.isfinite(heads[1]))
    assert float(np.max(heads[1])) < INITIAL_HEAD - 100.0

    heads_final = np.asarray(info["heads_final"], dtype=np.float64)
    np.testing.assert_allclose(heads_final, heads[0], atol=0.0)
    np.testing.assert_allclose(heads_final, DAY_TWO_HEAD, atol=HEAD_ATOL)
