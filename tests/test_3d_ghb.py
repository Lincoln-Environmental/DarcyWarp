"""Independent analytic GHB laws, boundary updates and physical conservation."""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls


@pytest.fixture(params=["cpu", "cuda:0"])
def device(request):
    if request.param.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    return request.param


def single_cell_fields(*, conductance, boundary_head, source, initial):
    shape = (1, 1, 1)
    return dict(kx_field=np.ones(shape=shape), ky_field=np.ones(shape=shape),
                kz_field=np.ones(shape=shape), active=np.ones(shape=shape, dtype=np.int32),
                bc_mask=np.zeros(shape=shape, dtype=np.int32), bc_values=np.zeros(shape=shape),
                rhs=np.full(shape=shape, fill_value=source),
                initial_head=np.full(shape=shape, fill_value=initial),
                ghb_conductance=conductance, ghb_head=boundary_head)


@pytest.mark.parametrize("implementation", ["classic", "fast"])
@pytest.mark.parametrize("formulation", ["legacy", "mf6_6_6_2_npf"])
@pytest.mark.parametrize("source", [-3.0, 0.0, 3.0])
def test_ghb_alone_grounds_cell_and_obeys_scalar_law(device, implementation, formulation, source):
    fields = single_cell_fields(conductance=4.0, boundary_head=2.0, source=source, initial=0.0)
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
                           implementation=implementation, formulation=formulation) as model:
        model.build_from_K_fields(**fields)
        head, info = model.solve(max_cycles=100, rel_tol=1e-13, abs_tol_min=1e-13)
        assert info["converged"], info
        np.testing.assert_allclose(actual=head, desired=2.0 + source / 4.0, rtol=0.0, atol=2e-14)
        model.update_ghb(ghb_conductance=2.0, ghb_head=5.0)
        changed, info = model.solve(max_cycles=100, rel_tol=1e-13, abs_tol_min=1e-13)
        assert info["converged"], info
        np.testing.assert_allclose(actual=changed, desired=5.0 + source / 2.0, rtol=0.0, atol=4e-14)


@pytest.mark.parametrize("implementation", ["classic", "fast"])
def test_cartesian_transient_ghb_is_not_counted_as_storage(device, implementation):
    fields = single_cell_fields(conductance=2.0, boundary_head=12.0, source=0.75, initial=9.0)
    previous = fields["initial_head"].copy()
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
                           implementation=implementation) as model:
        model.build_from_K_fields(**fields)
        head, info = model.solve(transient=True, dt=2.0, storage_coeff=0.25,
                                 head_prev=previous, max_cycles=100, rel_tol=1e-13, abs_tol_min=1e-13)
        expected = (0.75 + 2.0 * 12.0 + 0.125 * 9.0) / (2.0 + 0.125)
        assert info["converged"], info
        np.testing.assert_allclose(actual=head, desired=expected, rtol=0.0, atol=5e-14)
        assert info["budget"]["storage_uptake"] == pytest.approx(0.125 * (expected - 9.0), abs=1e-13)
        assert info["budget"]["ghb_net"] == pytest.approx(2.0 * (12.0 - expected), abs=1e-13)
        assert abs(info["budget"]["domain_imbalance"]) < 1e-13
        np.testing.assert_array_equal(actual=previous, desired=fields["initial_head"])


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("convertible", [False, True])
@pytest.mark.parametrize("preconditioner", ["diagonal", "kcycle"])
def test_npf_ghb_newton_storage_and_cached_updates(device, native, convertible, preconditioner):
    fields = single_cell_fields(conductance=2.0, boundary_head=0.5, source=0.125, initial=0.25)
    previous = fields["initial_head"].copy()
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
                           implementation="fast", formulation="mf6_6_6_2_npf") as model:
        model.build_from_K_fields(**fields)
        controls = NPF662NewtonControls(tiled_reductions=True, reuse_trial_state=True)
        for conductance, boundary_head in ((2.0, 0.5), (4.0, 0.75), (0.0, 0.5)):
            model.update_ghb(ghb_conductance=conductance, ghb_head=boundary_head)
            head, info = model.solve(transient=True, dt=2.0, head_prev=previous,
                ss=0.0 if convertible else 0.5, sy=0.5 if convertible else 0.0,
                unconfined=convertible, top_elevation=1.0, npf_device_native=native,
                npf_preconditioner=preconditioner, npf_newton_controls=controls,
                max_cycles=100, max_outer=10)
            # In the interior MF6's quadratic Newton smoothing has slope
            # 1/(1-epsilon); its constant offset cancels between timesteps.
            storage_slope = 0.25 / (1.0 - 1e-6) if convertible else 0.25
            expected = (0.125 + conductance * boundary_head + storage_slope * 0.25) / (conductance + storage_slope)
            assert info["converged"], info
            np.testing.assert_allclose(actual=head, desired=expected, rtol=0.0, atol=1e-13)
            assert info["ghb_budget"]["ghb_net"] == pytest.approx(conductance*(boundary_head-expected), abs=1e-13)
            np.testing.assert_array_equal(actual=previous, desired=fields["initial_head"])


@pytest.mark.parametrize("preconditioner", ["diagonal", "kcycle"])
def test_npf_steady_ghb_newton_grounding(device, preconditioner):
    fields = single_cell_fields(conductance=2.0, boundary_head=0.5, source=0.125, initial=0.25)
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
                           implementation="fast", formulation="mf6_6_6_2_npf") as model:
        model.build_from_K_fields(**fields)
        head, info = model.solve(unconfined=True, top_elevation=1.0,
            npf_preconditioner=preconditioner, max_cycles=100, max_outer=10)
        assert info["converged"], info
        np.testing.assert_allclose(actual=head, desired=0.5625, rtol=0.0, atol=1e-13)


@pytest.mark.parametrize("native", [False, True])
def test_add_ghb_after_captured_no_ghb_solve(device, native):
    fields = single_cell_fields(conductance=None,boundary_head=None,source=0.125,initial=0.25)
    with WarpDarcySolver3D(nx=1,ny=1,nz=1,dx=1.0,device=device,
                           implementation="fast",formulation="mf6_6_6_2_npf") as model:
        model.build_from_K_fields(**fields)
        controls = NPF662NewtonControls(tiled_reductions=True,reuse_trial_state=True)
        options = dict(transient=True,dt=2.0,head_prev=fields["initial_head"],ss=0.5,sy=0.0,
            npf_device_native=native,npf_preconditioner="kcycle",npf_newton_controls=controls,
            max_cycles=100,max_outer=10)
        head,info = model.solve(**options)
        assert info["converged"],info
        np.testing.assert_allclose(actual=head,desired=0.75,rtol=0.0,atol=1e-13)
        model.update_ghb(ghb_conductance=2.0,ghb_head=0.5)
        head,info = model.solve(**options)
        assert info["converged"],info
        np.testing.assert_allclose(actual=head,desired=(0.125+1.0+0.0625)/2.25,rtol=0.0,atol=1e-13)


@pytest.mark.parametrize("invalid", [
    dict(ghb_conductance=-1.0,ghb_head=0.5),
    dict(ghb_conductance=np.nan,ghb_head=0.5),
    dict(ghb_conductance=1.0,ghb_head=np.inf),
    dict(ghb_conductance=1.0,ghb_head=None),
    dict(ghb_conductance=1.0,ghb_head=0.5,ghb_mask=np.full((1,1,1),0.9)),
    dict(ghb_conductance=np.ones((1,1,1)),ghb_head=0.5,ghb_mask=np.zeros((1,1,1))),
])
def test_failed_ghb_update_is_atomic(invalid):
    fields = single_cell_fields(conductance=2.0,boundary_head=0.5,source=0.0,initial=0.25)
    with WarpDarcySolver3D(nx=1,ny=1,nz=1,dx=1.0,device="cpu",implementation="fast") as model:
        model.build_from_K_fields(**fields)
        revision = model._session.operator_revision
        with pytest.raises(ValueError):
            model.update_ghb(**invalid)
        assert model._session.operator_revision == revision
        head,info = model.solve(rel_tol=1e-13,abs_tol_min=1e-13)
        assert info["converged"],info
        np.testing.assert_allclose(actual=head,desired=0.5,rtol=0.0,atol=1e-13)


@pytest.mark.parametrize("inactive,fixed", [(True,False),(False,True)])
def test_ghb_rejects_invalid_cell_ownership(inactive, fixed):
    fields=single_cell_fields(conductance=2.0,boundary_head=0.5,source=0.0,initial=0.25)
    fields["active"][:]=int(not inactive)
    fields["bc_mask"][:]=int(fixed)
    with WarpDarcySolver3D(nx=1,ny=1,nz=1,dx=1.0,device="cpu",implementation="fast") as model:
        with pytest.raises(ValueError,match="GHB cells"):
            model.build_from_K_fields(**fields)


@pytest.mark.parametrize("implementation", ["classic", "fast"])
@pytest.mark.parametrize("transient", [False,True])
def test_cartesian_picard_ghb_analytic(device, implementation, transient):
    fields=single_cell_fields(conductance=2.0,boundary_head=0.5,source=0.125,initial=0.25)
    with WarpDarcySolver3D(nx=1,ny=1,nz=1,dx=1.0,device=device,implementation=implementation) as model:
        model.build_from_K_fields(**fields)
        options=dict(transient=True,dt=2.0,head_prev=fields["initial_head"],ss=0.0,sy=0.5) if transient else {}
        head,info=model.solve(unconfined=True,zbot_field=np.zeros((1,1,1)),ztop_field=np.ones((1,1,1)),unconfined_head_tol=1e-12,
            unconfined_max_picard_iter=80,final_head_residual_tol=1e-12,rel_tol=1e-13,abs_tol_min=1e-13,**options)
        expected=(0.125+1.0+0.0625)/2.25 if transient else 0.5625
        assert info["converged"],info
        np.testing.assert_allclose(actual=head,desired=expected,rtol=0.0,atol=2e-12)
