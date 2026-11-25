"""Verify extracted transient preparation and accepted-state assembly."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.solvers.transient_unconfined import (
    _assemble_transient_result, _prepare_transient_controls, _prepare_transient_fields,
)


class Grid:
    nx = 3
    ny = 2


def test_prepare_controls_does_not_mutate_input():
    """Host and device controls are separated without changing user inputs."""
    original = {"adaptive_dt_enabled": True, "max_cycles": 3, "min_saturated_thickness": 0.5}
    controls, device, save, trial, floor = _prepare_transient_controls(
        solve_controls=original, save_diagnostics=False, return_info=True, min_saturated_thickness=0.1,
    )
    assert controls == {"max_cycles": 3, "min_saturated_thickness": 0.5}
    assert device == original
    assert original["adaptive_dt_enabled"] is True
    assert not save and not trial and floor == 0.5


def test_prepare_fields_preserves_shape_and_precision():
    """Grid fields are normalised to the requested execution precision."""
    shape = (2, 3)
    head = np.ones(shape)
    options = dict(model=Grid(), initial_head=head, k_field=head,
                   zbot_field=head, ztop_field=head, active=None, bc_mask=None,
                   bc_values=None, recharge_rates=[0.0, 0.1], dt=1.0, dtype=np.float32)
    h, k, bottom, top, active, fixed, fixed_heads, rates, dt = _prepare_transient_fields(**options)
    for field in (h, k, bottom, top, fixed_heads, rates):
        assert field.dtype == np.float32
    np.testing.assert_array_equal(active, np.ones(shape))
    np.testing.assert_array_equal(fixed, np.zeros(shape))
    assert dt == 1.0
    with pytest.raises(ValueError, match="dt"):
        _prepare_transient_fields(**dict(options, dt=0.0))


def test_result_keeps_accepted_head_after_rejected_trial():
    """A rejected trial is returned diagnostically but never becomes final head."""
    import time

    model = Grid()
    heads = np.stack([np.ones((2, 3)), np.full((2, 3), 2.0), np.zeros((2, 3))])
    active = np.ones((2, 3), dtype=np.int32)
    fixed = np.zeros_like(active)
    fixed[0, 0] = 1
    active[1, 0] = 0
    histories = {
        "heads_old_per_period": heads.copy(),
        "storage_reference_heads_per_period": heads.copy(),
        "storage_terms_per_period": np.ones_like(heads),
    }
    records = [{"state": "accepted"}, {"state": "rejected_trial"}]
    result, info = _assemble_transient_result(
        model=model, heads_per_period=heads, head_prev=heads[0],
        accepted_periods=1, n_periods=3, period_infos=records, last_info=records[-1],
        period_times=np.array([1.0, 2.0, 0.0]), total_t0=time.perf_counter(),
        dt_f=1.0, storage_reference="current_picard", controls={},
        counters={"head_downloads": 2}, active_i=active, bc_i=fixed,
        return_info=True, histories=histories,
    )
    np.testing.assert_array_equal(result, heads[:2])
    np.testing.assert_array_equal(info["heads_final"], heads[0])
    assert info["accepted_periods"] == 1 and info["n_periods"] == 2
    np.testing.assert_array_equal(info["storage_terms_per_period"][:, :, 0], 0.0)
    np.testing.assert_array_equal(info["heads_old_per_period"], heads[:2])


def test_device_selection_preserves_precision_and_graph_controls(monkeypatch):
    """Default FP64 graphs, explicit opt-outs and FP32 GHB fallback stay intact.

    :param monkeypatch: Pytest fixture isolating process environment overrides.
    """
    from DARCY_WARP_PACKAGE.solvers.device_picard_periods import select_device_picard_options

    for name in ("DARCY_TRANSIENT_FACE_OPERATOR", "DARCY_TRANSIENT_FACE_GRAPHS", "DARCY_TRANSIENT_MIXED"):
        monkeypatch.delenv(name, raising=False)
    controls = {"use_device_transient_fast_path": True}
    common = dict(fast_path_controls=controls, fp64_fields=True, use_ghb=False,
                  storage_mode="mf6_convertible_secant_sy", storage_reference="current_picard")
    options = select_device_picard_options(**common)
    assert options.enabled and options.face_operator and options.face_graphs
    assert not options.incremental and not options.mixed_precision
    disabled = dict(controls, transient_face_graphs_enabled=False)
    assert not select_device_picard_options(**dict(common, fast_path_controls=disabled)).face_graphs
    options = select_device_picard_options(**dict(common, fp64_fields=False, use_ghb=True))
    assert not options.enabled and not options.face_operator
    with pytest.raises(ValueError, match="require DARCY_FLOAT"):
        select_device_picard_options(**dict(common, fp64_fields=False,
            fast_path_controls=dict(controls, transient_face_operator_enabled=True)))
    with pytest.raises(ValueError, match="requires transient_face_operator_enabled"):
        select_device_picard_options(**dict(common, fast_path_controls=dict(controls,
            transient_face_operator_enabled=False, transient_mixed_precision_enabled=True)))
