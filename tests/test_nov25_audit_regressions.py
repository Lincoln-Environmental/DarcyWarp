"""Public regressions for the 25 November DarcyWarp audit."""

import os

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE import create_solver
from DARCY_WARP_PACKAGE.config import NP_FLOAT as NP_FLOAT_3D
from DARCY_WARP_PACKAGE.warped_darcy import NP_FLOAT as NP_FLOAT_2D


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param == "cuda:0" and (
        os.environ.get("DARCYWARP_TEST_CPU_ONLY") == "1" or not wp.is_cuda_available()
    ):
        pytest.skip("CUDA required")
    return request.param


@pytest.mark.parametrize("backend", ["unconfined_semismooth_newton_kcycle", "unconfined_fas"])
def test_closed_basins_store_each_period_of_recharge(device, backend):
    shape = (4, 9)
    fixed = np.zeros(shape, dtype=int)
    active = np.ones(shape, dtype=int)
    active[:, 4] = 0
    rate = np.zeros(shape)
    rate[:, :4] = 0.2 * 0.292 / 364
    rate[:, 5:] = -rate[0, 0]
    rates = np.broadcast_to(rate, (52, *shape))
    start = np.full(shape, 5.0)
    dx = 0.01
    with create_solver(dim=2, nx=shape[1], ny=shape[0], dx=dx, device=device) as model:
        heads, info = model.solve_transient_2d_unconfined(
            solver=backend, initial_head=start, recharge_rates=np.zeros(len(rates)),
            source_fields_per_period=rates, k_field=np.ones(shape), zbot_field=np.zeros(shape),
            ztop_field=np.full(shape, 10.0), sy=0.2, ss=0.0, dt=7.0,
            active=active, bc_mask=fixed, bc_values=np.zeros(shape),
        )
    assert all(period["converged"] for period in info["period_infos"])
    records = info["experimental_timestep_records"]
    assert len(records) == 52
    for record in records:
        assert record["accepted"]
        balance = record["physical_period_balance"]
        assert balance["accepted"] and len(balance["component_balances"]) == 2
        assert all(abs(row["imbalance"]) <= row["tolerance"] for row in balance["component_balances"])
    for columns, final_head in ((slice(0, 4), 5.292), (slice(5, None), 4.708)):
        stored = info["storage_terms_per_period"][:, :, columns]
        supplied = rates[:, :, columns]
        np.testing.assert_allclose(stored.sum(axis=(1, 2)), supplied.sum(axis=(1, 2)), rtol=1e-3)
        np.testing.assert_allclose(stored.sum(), supplied.sum(), rtol=1e-3)
        np.testing.assert_allclose(heads[-1, :, columns], final_head, rtol=0., atol=2.92e-4)


@pytest.mark.parametrize("backend", ["unconfined_semismooth_newton_kcycle", "unconfined_fas"])
@pytest.mark.parametrize("cap", [0.48, 0.01])
def test_experimental_step_cap_includes_remainder(device, backend, cap):
    shape = (4, 4)
    with create_solver(dim=2, nx=4, ny=4, dx=1., device=device) as model:
        _, info = model.solve_transient_2d_unconfined(
            solver=backend, initial_head=np.full(shape, 5.), k_field=np.ones(shape),
            zbot_field=np.zeros(shape), ztop_field=np.full(shape, 10.),
            recharge_rates=[0.], dt=1., sy=0.2, ss=0.,
            active=np.ones(shape, dtype=int), bc_mask=np.zeros(shape, dtype=int), bc_values=np.zeros(shape),
            solve_controls={"experimental_max_dt": cap},
        )
    steps = info["period_infos"][0]["experimental_substep_dts"]
    assert max(steps) <= cap
    assert sum(steps) == pytest.approx(1.0)


def fields_2d(*, t, r):
    mask = np.zeros(t.shape, dtype=int)
    mask[:, (0, -1)] = 1
    stages = np.zeros(t.shape)
    stages[:, 0] = 2.
    return dict(T_field=t, R_field=r, active=np.ones(t.shape, dtype=int), bc_mask=mask, bc_values=stages)


@pytest.mark.parametrize("same_buffer", [True, False])
def test_updates_own_shared_and_overlapping_buffers(device, same_buffer):
    base = np.full(21, 0.5, dtype=NP_FLOAT_2D)
    t = base[:20].reshape(4, 5)
    r = t if same_buffer else base[1:].reshape(4, 5)
    original = base.copy()
    outputs = []
    for transmissivity, recharge in ((t.copy(), r.copy()), (t, r)):
        fields = fields_2d(t=transmissivity, r=recharge)
        with create_solver(dim=2, nx=5, ny=4, dx=1., device=device) as model:
            model.build_from_fields(**fields)
            model.update_R_in_place(R_truth=np.zeros_like(r))
            model.update_T_in_place(T_truth=3.0 * transmissivity)
            np.testing.assert_array_equal(model.T_field_host, model.T_wp.numpy())
            np.testing.assert_array_equal(model.R_field_host, model.R_wp.numpy())
            head, info = model.solve()
            assert info["converged"]
            outputs.append(head)
    np.testing.assert_array_equal(base, original)
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0., atol=1e-9)


@pytest.mark.parametrize("stage", [np.nan, np.inf, 1e20, -1e30])
def test_rejected_2d_prescribed_heads_preserve_solvable_model(device, stage):
    fields = fields_2d(t=np.ones((4, 5)), r=np.zeros((4, 5)))
    fields["bc_values"][0, 1] = -1e30
    invalid = fields["bc_values"].copy()
    invalid[0, 0] = stage
    with create_solver(dim=2, nx=5, ny=4, dx=1., device=device) as model:
        model.build_from_fields(**fields)
        expected, _ = model.solve()
        with pytest.raises(ValueError, match="prescribed head"):
            model.build_from_fields(**dict(fields, bc_values=invalid))
        with pytest.raises(ValueError, match="prescribed head"):
            model.update_bc_in_place(bc_values=invalid)
        np.testing.assert_array_equal(model.solve()[0], expected)


@pytest.mark.parametrize("width", [np.nan, np.inf, -1., -1e30])
def test_rejected_2d_ghb_width_preserves_solvable_model(device, width):
    shape = (4, 5)
    fields = fields_2d(t=np.ones(shape), r=np.zeros(shape))
    gh_mask = np.zeros(shape)
    gh_mask[1, 1:3] = 1
    gh_width = np.full(shape, np.nan)
    gh_width[1, 1:3] = [0.0, 1.0]
    stage = np.full(shape, -1e30)
    stage[1, 2] = 0.0
    ghb = dict(gh_mask=gh_mask, gh_head=stage, gh_width=gh_width)
    with create_solver(dim=2, nx=5, ny=4, dx=1., device=device, use_ghb=True, aq_thickness=1.) as model:
        model.build_from_fields(**fields, **ghb)
        expected, _ = model.solve()
        invalid = gh_width.copy()
        invalid[1, 2] = width
        with pytest.raises(ValueError, match="GHB width"):
            model.build_from_fields(**fields, **dict(ghb, gh_width=invalid))
        np.testing.assert_array_equal(model.solve()[0], expected)


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_low_positive_2d_transmissivity_preserves_linear_head(device, implementation):
    if implementation == "fast" and NP_FLOAT_2D != np.float64:
        pytest.skip("Fast 2D requires FP64")
    shape = (4, 5)
    fields = fields_2d(t=np.full(shape, 1e-13), r=np.zeros(shape))
    with create_solver(dim=2, nx=5, ny=4, dx=1., device=device) as model:
        model.build_from_fields(**fields)
        head, info = model.solve(implementation=implementation)
    assert info["converged"]
    np.testing.assert_allclose(head, np.broadcast_to(np.linspace(2., 0., 5), shape), rtol=0., atol=1e-5)


def fields_3d(*, k, stage):
    shape = k.shape
    fixed = np.zeros(shape, dtype=int)
    fixed[..., 0] = 1
    return dict(kx_field=k, ky_field=k, kz_field=k, active=np.ones(shape, dtype=int),
                bc_mask=fixed, bc_values=np.full(shape, stage), rhs=np.zeros(shape))


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_rejected_cartesian_previous_head_keeps_model_usable(device, implementation):
    if implementation == "fast" and NP_FLOAT_3D != np.float64:
        pytest.skip("Fast 3D requires FP64")
    shape = (1, 1, 3)
    fields = fields_3d(k=np.ones(shape), stage=5.)
    fields["active"][..., -1] = 0
    prev = np.array([[[1.e30, 5., -1.e30]]])
    with create_solver(dim=3, nx=3, ny=1, nz=1, dx=1., dz=1., device=device,
                       implementation=implementation) as model:
        model.build_from_K_fields(**fields)
        good, _ = model.solve(transient=True, dt=1., storage_coeff=0.2, head_prev=prev)
        with pytest.raises(ValueError, match="head_prev"):
            model.solve(transient=True, dt=1., storage_coeff=0.2, head_prev=np.full(shape, 1.e30))
        np.testing.assert_array_equal(model.solve(transient=True, dt=1., storage_coeff=0.2, head_prev=prev)[0], good)


def test_cartesian_fp32_forms_face_before_narrowing(device):
    if NP_FLOAT_3D != np.float32:
        pytest.skip("FP32 underflow contract")
    shape = (1, 1, 2)
    k = np.full(shape, 1.e-50)
    fields = fields_3d(k=k, stage=5.)
    with create_solver(dim=3, nx=2, ny=1, nz=1, dx=1., dy=1.e12, dz=1.e12, device=device) as model:
        model.build_from_K_fields(**fields)
        assert model._session.faces[0][0, 0, 0] == pytest.approx(1.e-26, rel=1.e-6, abs=0.)
        model.update_K_fields(kx_field=2 * k, ky_field=k, kz_field=k)
        assert model._session.faces[0][0, 0, 0] == pytest.approx(2.e-26, rel=1.e-6, abs=0.)
        too_small = np.full(shape, 1.e-90)
        with pytest.raises(ValueError, match="(?i)unrepresentable"):
            model.update_K_fields(kx_field=too_small, ky_field=k, kz_field=k)
        with pytest.raises(ValueError, match="(?i)unrepresentable"):
            model.build_from_K_fields(**dict(fields, kx_field=too_small))
        assert model._session.faces[0][0, 0, 0] == pytest.approx(2.e-26, rel=1.e-6, abs=0.)


def test_initial_transmissivity_can_share_conductivity(device):
    shape = (4, 5)
    conductivity = np.full(shape, 1., dtype=NP_FLOAT_2D)
    original = conductivity.copy()
    results = []
    for initial_t in (conductivity.copy(), conductivity):
        fields = fields_2d(t=initial_t, r=np.zeros(shape))
        fields["bc_values"][:, 0] = 8.
        fields["bc_values"][:, -1] = 5.
        with create_solver(dim=2, nx=5, ny=4, dx=1., device=device) as model:
            model.build_from_fields(**fields)
            head, info = model.solve(solver="unconfined_picard_kcycle", formulation="unconfined",
                K_field=conductivity, zbot_field=np.zeros(shape), initial_head=np.full(shape, 6.5))
            assert info["converged"]
            np.testing.assert_array_equal(model.T_field_host, model.T_wp.numpy())
            results.append(head)
    np.testing.assert_array_equal(conductivity, original)
    np.testing.assert_array_equal(results[0], results[1])
