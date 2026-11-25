"""Independent steady confined acceptance and mixed-workspace regressions."""

import weakref

import numpy as np
import pytest

from DARCY_WARP_PACKAGE import create_solver
from DARCY_WARP_PACKAGE.config import NP_FLOAT


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("time_scale", [1.0, 86400.0], ids=["seconds", "days"])
def test_pcg_closes_recharged_strip(device, time_scale):
    if NP_FLOAT != np.float64:
        pytest.skip("This budget control uses FP64 heads.")
    ny, nx = 4, 101
    grid = np.ones((ny, nx))
    fixed = np.zeros((ny, nx), dtype=int)
    fixed[:, [0, -1]] = 1
    t = 0.001 * time_scale
    r = 1.0e-9 * time_scale
    with create_solver(dim=2, nx=nx, ny=ny, dx=10.0, device=device) as model:
        model.build_from_fields(
            T_field=t * grid, R_field=r * grid, active=grid, bc_mask=fixed, bc_values=10.0 * grid,
        )
        h, info = model.solve(solver="confined_pcg", initial_head=10.0 * grid)
    assert info["converged"]
    assert h[2, 50] == pytest.approx(10.125, abs=1e-4)
    outflow = t * np.sum(h[:, 1] - h[:, 0] + h[:, -2] - h[:, -1])
    assert outflow == pytest.approx(ny * (nx - 2) * r * 100.0, rel=0.001)


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("nx,expected_convergence", [(65, True), (8, False)])
def test_pcg_fp32_verifies_emitted_heads(device, nx, expected_convergence):
    if NP_FLOAT != np.float32:
        pytest.skip("This grid probes FP32 rounding at a 100 m datum.")
    shape = (8, nx)
    fixed = np.zeros(shape, dtype=int)
    fixed[:, [0, -1]] = 1
    boundary = np.zeros(shape)
    boundary[:, 0], boundary[:, -1] = 101.0, 102.0
    with create_solver(dim=2, nx=nx, ny=8, dx=100.0, device=device) as model:
        model.build_from_fields(
            T_field=np.full(shape, 25.0), R_field=np.zeros(shape),
            active=np.ones(shape), bc_mask=fixed, bc_values=boundary,
        )
        h, info = model.solve(solver="confined_pcg", initial_head=np.full(shape, 100.0),
                              pcg_max_iter=350, history_every=10)
    h = h.astype(np.float64)
    interior = h[:, 1:-1]
    flow = 25.0 * (2.0 * interior - h[:, :-2] - h[:, 2:])
    flow[1:] += 25.0 * (interior[1:] - interior[:-1])
    flow[:-1] += 25.0 * (interior[:-1] - interior[1:])
    actual_rms = np.sqrt(np.sum(flow**2) / h.size)
    assert info["rms_res_final_phys"] == pytest.approx(actual_rms, abs=1e-14)
    assert info["history"][-1]["rms_res_phys"] == pytest.approx(actual_rms, abs=1e-14)
    assert info["converged"] is expected_convergence
    assert bool(actual_rms <= info["tol_abs_phys"]) is expected_convergence


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
def test_mixed_workspace_tracks_mutated_and_replaced_forcing(device):
    if NP_FLOAT != np.float32:
        pytest.skip("The mixed driver requires an FP32 model.")
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import get_mixed_fast_session, solve_mixed_fast

    ny, nx = 4, 9
    shape = (ny, nx)
    ones = np.ones(shape)
    fixed = np.zeros(shape, dtype=int)
    fixed[:, [0, -1]] = 1
    ghb = np.zeros_like(fixed)
    ghb[:, nx // 2] = 1
    zero = np.zeros(shape)
    stage = zero.copy()
    boundary = zero.copy()
    recharge = zero.copy()
    with create_solver(dim=2, nx=nx, ny=ny, dx=10.0, device=device,
                       use_ghb=True, aq_thickness=1.0) as model:
        model.build_from_fields(T_field=ones, R_field=recharge, active=ones, bc_mask=fixed,
                                bc_values=boundary, gh_mask=ghb, gh_width=ones, gh_head=stage)
        fields = dict(bc_values_f64=boundary, gh_head_f64=stage, R_f64=recharge)
        initial, info = solve_mixed_fast(model, zero, **fields)
        assert info["converged"]
        np.testing.assert_array_equal(initial, zero)
        workspace = weakref.ref(model._mixed_fast_workspace)
        ptrs = (workspace().h64.ptr, workspace().b64.ptr, workspace().bc_values64.ptr)

        # Prescribe the GHB at the exact head so its exchange is zero.
        # The uniform recharge strip then has an exact discrete parabola.
        recharge.fill(0.001)
        model.update_R_in_place(recharge)
        columns = np.arange(nx)
        profile = 0.05 * columns * (nx - 1 - columns)
        for datum in (0.0, 10.0):
            boundary[fixed != 0] = datum
            model.update_bc_in_place(boundary)
            stage[ghb != 0] = datum + profile[nx // 2]
            head, info = solve_mixed_fast(model, zero, **fields)
            assert info["converged"]
            np.testing.assert_allclose(head, datum + np.broadcast_to(profile, shape), atol=1e-4, rtol=0)

        # Change only the GHB stage, using replacement arrays and unchanged
        # model operator. By symmetry the centre drains through both ends.
        fields = {name: array.copy() for name, array in fields.items()}
        fields["gh_head_f64"][ghb != 0] += 1.0
        head, info = solve_mixed_fast(model, zero, **fields)
        assert info["converged"]
        conductance = float(model.ghb_factor_host[0, nx // 2])
        centre_rise = conductance / (conductance + 2.0 / (nx // 2))
        assert head[2, nx // 2] == pytest.approx(10.8 + centre_rise, abs=1e-4)
        assert workspace() is model._mixed_fast_workspace
        assert (workspace().h64.ptr, workspace().b64.ptr, workspace().bc_values64.ptr) == ptrs
        # Explicitly retained sessions may coexist without shared forcing.
        first = get_mixed_fast_session(model, **fields)
        second_fields = dict(fields, R_f64=fields["R_f64"].copy())
        second = get_mixed_fast_session(model, **second_fields)
        assert first is not second
        assert get_mixed_fast_session(model, **fields) is first
        first_ref, second_ref = weakref.ref(first), weakref.ref(second)
        del first, second
        assert second_ref() is None
    assert first_ref() is None
    assert workspace() is None
