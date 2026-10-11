"""Quantify the floor's face-flow perturbation, separately from conservation."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.dry_cell_3d import quantify_dry_cell_transmission
from DARCY_WARP_PACKAGE.physics.budget_3d import compute_3d_flow_budget
from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_unconfined_from_sat
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from test_3d_scientific_audit import device


@pytest.mark.parametrize("thickness", [10.0, 0.05])
@pytest.mark.parametrize("contrast", [1.0, 1e-14])
@pytest.mark.parametrize("saturation", [1.0, 0.05, 0.0])
def test_floor_and_dry_vertical_transmission_against_face_resistances(thickness, contrast, saturation):
    shape = (2, 1, 2)
    bottom = np.broadcast_to(np.array([thickness, 0.0])[:, None, None], shape).copy()
    # Unequal heads probe flow without claiming they solve a zero-source problem.
    head = bottom + min(saturation, thickness)
    head[:, :, 1] -= 0.01
    k = np.ones(shape)
    k[1] = contrast
    active = np.ones(shape, dtype=np.int32)
    bc = np.zeros(shape, dtype=np.int32)
    source = np.array([0.01, -0.02, 0.0, 0.0]).reshape(shape)
    true_sat = np.clip(head - bottom, 0.0, thickness)
    actual_sat = np.clip(head - bottom, min(0.1, thickness), thickness)
    faces = build_7point_face_conductance_unconfined_from_sat(kx_field=k, ky_field=k, kz_field=k,
        sat=actual_sat, active=active, dx=1.0, dz=thickness)
    reference = build_7point_face_conductance_unconfined_from_sat(kx_field=k, ky_field=k, kz_field=k,
        sat=true_sat, active=active, dx=1.0, dz=thickness)
    budget = compute_3d_flow_budget(head=head, source=source, faces=faces, active=active, bc_mask=bc)
    diagnostic = quantify_dry_cell_transmission(head=head, bottoms=bottom, tops=bottom + thickness,
        actual_faces=faces, unfloored_faces=reference, active=active, bc_mask=bc, source=source, budget=budget)
    # Independent half-cell resistance formula, and count horizontal faces once.
    expected = 0.0
    for layer in range(2):
        b0, b1 = actual_sat[layer, 0]
        actual = 2.0 / (1.0 / (k[layer, 0, 0] * b0) + 1.0 / (k[layer, 0, 1] * b1))
        s0, s1 = true_sat[layer, 0]
        physical = 0.0 if min(s0, s1) == 0.0 else 2.0 / (1.0 / (k[layer, 0, 0] * s0) + 1.0 / (k[layer, 0, 1] * s1))
        expected += (actual - physical) * 0.01
    assert diagnostic["horizontal_floor_excess_flow"] == pytest.approx(expected, abs=1e-16)
    assert diagnostic["horizontal_floor_to_recharge"] == pytest.approx(expected / 0.01, abs=1e-14)
    assert diagnostic["horizontal_floor_to_abstraction"] == pytest.approx(expected / 0.02, abs=1e-14)
    if saturation == 0.0:
        cz = 2.0 / (thickness / 1.0 + thickness / contrast)
        assert diagnostic["vertical_dry_interface_flow"] == pytest.approx(2 * cz * thickness, rel=1e-14)
    else:
        assert diagnostic["vertical_dry_interface_flow"] == 0.0


@pytest.mark.parametrize("source_rate", [0.01, -0.01])
def test_live_dry_floor_budget_can_conserve_while_artificial_flow_is_large(source_rate):
    shape = (1, 1, 3)
    k = np.ones(shape)
    bc = np.array([[[1, 0, 1]]], dtype=np.int32)
    values = np.array([[[-1.0, 0.0, -2.0]]])
    source = np.array([[[0.0, source_rate, 0.0]]])
    with WarpDarcySolver3D(nx=3, ny=1, nz=1, dx=1.0, dz=1.0,
                          device=device(), implementation="fast") as s:
        s.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k, active=np.ones(shape, dtype=np.int32),
                              bc_mask=bc, bc_values=values, rhs=source, initial_head=np.full(shape, -1.5))
        head, info = s.solve(unconfined=True, zbot_field=np.zeros(shape), chebyshev_enabled=False,
                             unconfined_head_tol=1e-9, final_head_residual_tol=1e-9,
                             rel_tol=1e-11, dh_rms_tol=1e-10, check_every_no=1)
        assert info["converged"] and info["final_physical_acceptance"]
        assert head[0, 0, 1] == pytest.approx(-1.5 + source_rate / 0.2, abs=1e-8)
        assert info["budget"]["relative_imbalance"] < 1e-8
        diagnostic = info["dry_cell_approximation"]
        assert diagnostic["horizontal_floor_excess_flow"] == pytest.approx(0.1, abs=1e-9)
        ratio = diagnostic["horizontal_floor_to_recharge" if source_rate > 0 else "horizontal_floor_to_abstraction"]
        assert ratio == pytest.approx(10.0, abs=1e-7)
        assert diagnostic["horizontal_floor_to_budget_throughput"] == pytest.approx(0.1 / 0.11, abs=1e-7)
        # Rewet the same cells by changing only boundary values and forcing.
        s.update_rhs(rhs=np.zeros(shape))
        s.update_boundary_values(bc_values=np.array([[[2.0, 0.0, 1.0]]]))
        s.update_initial_head(initial_head=head)
        wet, wet_info = s.solve(unconfined=True, zbot_field=np.zeros(shape),
                                unconfined_head_tol=1e-9, final_head_residual_tol=1e-9,
                                rel_tol=1e-11, dh_rms_tol=1e-10, check_every_no=1,
                                chebyshev_enabled=False)
        assert wet_info["converged"]
        assert wet[0, 0, 1] == pytest.approx(1.5, abs=1e-8)
        assert wet_info["dry_cell_approximation"]["horizontal_floor_excess_flow"] == 0.0


def test_default_floor_cannot_exceed_thin_layer_thickness():
    shape = (1, 1, 3)
    k = np.ones(shape)
    bc = np.array([[[1, 0, 1]]], dtype=np.int32)
    values = np.array([[[-1.0, 0.0, -2.0]]])
    with WarpDarcySolver3D(nx=3, ny=1, nz=1, dx=1.0, dz=0.05,
                          device=device(), implementation="fast") as s:
        s.build_from_K_fields(kx_field=k, ky_field=k, kz_field=k, active=np.ones(shape, dtype=np.int32),
                              bc_mask=bc, bc_values=values, rhs=np.zeros(shape), initial_head=np.full(shape, -1.5))
        head, info = s.solve(unconfined=True, zbot_field=np.zeros(shape),
                             unconfined_head_tol=1e-9, final_head_residual_tol=1e-9)
    assert info["converged"]
    assert info["requested_min_saturated_thickness"] == 0.1
    assert info["min_saturated_thickness"] == 0.05
    assert head[0, 0, 1] == pytest.approx(-1.5, abs=1e-7)
    assert info["budget"]["boundary_inflow"] == pytest.approx(0.025, abs=1e-9)
