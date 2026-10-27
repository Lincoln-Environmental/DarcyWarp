"""DRN extraction from accepted semismooth transient states."""

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.physics.drn_2d import drn_discharge_2d
from DARCY_WARP_PACKAGE.physics.drn_2d import add_drn_to_budget
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver


def test_drn_gating_sign_and_mask():
    head = np.array([[1.0, 2.0, 3.0, 4.0]])
    elevation = np.array([[1.0, 2.5, 2.0, 2.0]])
    flux = drn_discharge_2d(
        head=head,
        elevation=elevation,
        conductance=np.full((1, 4), 5.0),
        mask=np.array([[1, 1, 1, 0]], dtype=np.int32),
    )
    np.testing.assert_array_equal(flux, [[0.0, 0.0, 5.0, 0.0]])
    assert np.all(flux >= 0.0)


def test_drn_budget_augmentation_works_without_preexisting_drn_columns():
    import pandas as pd

    budget = pd.DataFrame([{
        "total_in": 2.0, "total_out": 1.0,
        "in_minus_out": 1.0, "percent_discrepancy": 100.0 / 3.0,
        "throughflow": 1.5, "imbalance_fraction": 2.0 / 3.0,
    }])
    augmented = add_drn_to_budget(budget, np.array([[0.5]]))
    assert augmented.iloc[0]["drn_in"] == 0.0
    assert augmented.iloc[0]["drn_out"] == 0.5
    assert augmented.iloc[0]["total_out"] == 1.5
    assert augmented.iloc[0]["in_minus_out"] == 0.5


def test_accepted_flux_uses_returned_head_and_requires_authoritative_state():
    solver = WarpDarcySolver.__new__(WarpDarcySolver)
    solver.use_drn = True
    solver.ny, solver.nx = 1, 2
    solver.drn_mask_host = np.ones((1, 2), dtype=np.int32)
    solver.drn_elev_host = np.ones((1, 2))
    solver.drn_cond_host = np.full((1, 2), 10.0)
    solver.T_field_host = np.full((1, 2), 999.0)  # deliberately stale
    solver.x_wp = None  # deliberately unavailable
    head = np.array([[2.0, 0.5]])
    info = {
        "converged": True,
        "solver_backend": "unconfined_semismooth_newton_kcycle",
        "transmissivity_array": np.full((1, 2), 2.0),
        "saturated_thickness_array": np.full((1, 2), 1.0),
        "drn_discharge_rate_array": np.array([[10.0, 0.0]]),
    }
    np.testing.assert_array_equal(
        solver.accepted_drn_discharge_2d(head=head, info=info), [[10.0, 0.0]]
    )
    with pytest.raises(RuntimeError, match="lacks transmissivity_array"):
        solver.accepted_drn_discharge_2d(
            head=head,
            info={key: value for key, value in info.items() if key != "transmissivity_array"},
        )
    with pytest.raises(RuntimeError, match="converged"):
        solver.accepted_drn_discharge_2d(head=head, info={**info, "converged": False})
    with pytest.raises(RuntimeError, match="does not match"):
        solver.accepted_drn_discharge_2d(head=np.array([[1.5, 0.5]]), info=info)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_transient_budget_contains_accepted_drn_volume(device: str):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA is unavailable")
    shape = (5, 5)
    ground = np.full(shape, 9.0)
    initial_head = np.full(shape, 8.99)
    conductivity = np.ones(shape)
    with WarpDarcySolver(
        nx=5, ny=5, dx=10.0, device=device, use_drn=True,
        solver_type="kcycle", diag_preconditioner_backend="device",
    ) as solver:
        solver.build_from_fields(
            T_field=conductivity * initial_head,
            R_field=np.full(shape, 2.0),
            active=np.ones(shape, dtype=np.int32),
            bc_mask=np.zeros(shape, dtype=np.int32),
            bc_values=initial_head,
            drn_mask=np.ones(shape, dtype=np.int32),
            drn_elev=ground,
            drn_cond=np.full(shape, 100.0),
        )
        dt_days = 0.04
        head, info = solver.solve(
            formulation="unconfined", solver="unconfined_semismooth_newton_kcycle",
            initial_head=initial_head, K_field=conductivity,
            zbot_field=np.zeros(shape), ztop_field=ground,
            transient=True, sy=0.1, ss=1.0e-5, dt=dt_days,
            head_prev=initial_head, return_info=True,
            newton_fallback_to_picard=False, max_levels=2, min_coarse_cells=1,
        )
        assert info["converged"], info.get("newton_failure_reason")
        flux = solver.accepted_drn_discharge_2d(head=head, info=info)
        assert flux.sum() > 0.0
        assert info["budget_summary"]["drn_in"] == 0.0
        assert info["budget_summary"]["drn_out"] == pytest.approx(float(flux.sum()))
        assert info["drn_volume"] == pytest.approx(float(flux.sum()) * dt_days)
