"""Live pinned-MF6 GHB comparisons: native heads, exchange and STO histories."""
import hashlib
from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case, NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.validation.npf662_transient_mf6 import run_transient_mf6
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls


def ghb_case(*, convertible):
    shape = (3, 4, 7)
    z, y, x = np.indices(dimensions=shape)
    conductivity = 1.0 + 0.1*x + 0.05*y + 0.2*z
    active = np.ones(shape=shape, dtype=int)
    fixed = np.zeros_like(a=active)
    fixed[:, :, 0] = 1
    values = np.full(shape=shape, fill_value=2.4 if convertible else 4.0)
    source = np.zeros(shape=shape)
    source[0, 1, 3] = 0.025
    source[1, 2, 4] = -0.01
    conductance = np.zeros(shape=shape)
    conductance[:, :, -1] = 0.7
    conductance[1, 2, 2] = 0.25  # Native GHB may lie inside the grid.
    boundary_head = np.full(shape=shape, fill_value=2.1 if convertible else 3.5)
    boundary_head[1, 2, 2] = 2.6 if convertible else 4.2
    return NPF662Case(kx=conductivity,ky=0.6*conductivity,kz=0.15*conductivity,
        active=active,fixed=fixed,values=values,source=source,initial=values.copy(),
        dx=2.0,dy=3.0,dz=1.0,top=3.0,cell_type=int(convertible),
        ghb_conductance=conductance,ghb_head=boundary_head)


def build_case(*, case, device, formulation):
    nz,ny,nx = case.active.shape
    model = WarpDarcySolver3D(nx=nx,ny=ny,nz=nz,dx=case.dx,dy=case.dy,dz=case.dz,
        device=device,implementation="fast",formulation=formulation)
    model.build_from_K_fields(kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
        active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,
        initial_head=case.initial,ghb_conductance=case.ghb_conductance,ghb_head=case.ghb_head)
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("convertible", [False, True])
def test_live_npf_ghb_steady_and_transient(tmp_path, pinned_mf6, device, convertible):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    case = ghb_case(convertible=convertible)
    reference = NPF662Reference(case=case)
    native, native_info, native_fields = run_reference_mf6(case=case,directory=tmp_path/"steady",
        executable=pinned_mf6,head_tolerance_m=1e-12,
        native_solver_options=dict(rcloserecord=[1e-12,"STRICT"],under_relaxation="NONE"))
    assert native_info["native_success"] and native_info["independent_conservation"], native_info
    assert native_info["ghb_law_valid"], native_info
    controls = NPF662NewtonControls(inner_forcing=1e-3,tiled_reductions=True,reuse_trial_state=True)
    with build_case(case=case,device=device,formulation="mf6_6_6_2_npf") as model:
        head,info = model.solve(unconfined=convertible,top_elevation=case.top,npf_preconditioner="kcycle",
            npf_final_relative_tolerance=1e-12,npf_newton_controls=controls,max_cycles=500,max_outer=50)
        assert info["converged"], info
        np.testing.assert_allclose(actual=head,desired=native,rtol=0.0,atol=2e-11)
        checks,fields = reference.evaluate(head=head)
        assert checks["independent_accepted"], checks
        np.testing.assert_allclose(actual=-fields["ghb_outflow"],desired=native_fields["ghb_native_inflow"],rtol=0.0,atol=2e-11)
    factors = [1.0,0.5,1.5]
    native_history,native_transient_info = run_transient_mf6(case=case,directory=tmp_path/"transient",
        executable=pinned_mf6,executable_sha256=hashlib.sha256(pinned_mf6.read_bytes()).hexdigest(),
        source_factors=factors,dt=2.0,ss=1e-3,sy=0.15,
        head_tolerance=1e-12,residual_tolerance=1e-12)
    assert native_transient_info["passed"], native_transient_info
    with build_case(case=case,device=device,formulation="mf6_6_6_2_npf") as model:
        heads,info = model.solve_transient_npf(source_factors=factors,dt=2.0,
            head_prev=case.initial,ss=1e-3,sy=0.15,unconfined=convertible,top_elevation=case.top,
            npf_final_relative_tolerance=1e-12,npf_newton_controls=controls,max_cycles=500,max_outer=50)
        assert info["converged"],info
        np.testing.assert_allclose(actual=heads,desired=native_history["heads"],rtol=0.0,atol=2e-11)
        previous = case.initial
        for index,head in enumerate(heads):
            step = replace(case,source=case.source*factors[index])
            auditor = NPF662TransientReference(case=step,previous=previous,dt=2.0,ss=1e-3,sy=0.15)
            checks,fields = auditor.evaluate(head=head)
            assert checks["independent_accepted"],checks
            np.testing.assert_allclose(actual=-fields["ghb_outflow"],desired=native_history["ghb_inflows"][index],rtol=0.0,atol=2e-11)
            np.testing.assert_allclose(actual=fields["storage_rate"],desired=native_history["storage_rates"][index],rtol=0.0,atol=2e-11)
            previous = head


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("acceleration", ["kcycle", "fgmres"])
def test_live_cartesian_ghb_steady_and_transient(tmp_path, pinned_mf6, device, acceleration):
    if device.startswith("cuda") and not wp.is_cuda_available():
        pytest.skip("CUDA unavailable")
    case = ghb_case(convertible=False)
    native,info,unused = run_reference_mf6(case=case,directory=tmp_path/"steady",executable=pinned_mf6,
        head_tolerance_m=1e-12,native_solver_options=dict(rcloserecord=[1e-12,"STRICT"],under_relaxation="NONE"))
    assert info["native_success"] and info["independent_conservation"],info
    with build_case(case=case,device=device,formulation="legacy") as model:
        head,info = model.solve(max_cycles=1000,rel_tol=1e-14,abs_tol_min=1e-13,final_budget_relative_tol=1e-12,
            linear_acceleration=acceleration)
        assert info["converged"],info
        np.testing.assert_allclose(actual=head,desired=native,rtol=0.0,atol=2e-11)
    factors = [1.0,0.5,1.5]
    native_history,native_info = run_transient_mf6(case=case,directory=tmp_path/"transient",executable=pinned_mf6,
        executable_sha256=hashlib.sha256(pinned_mf6.read_bytes()).hexdigest(),source_factors=factors,
        dt=2.0,ss=1e-3,sy=0.0,head_tolerance=1e-12,residual_tolerance=1e-12)
    assert native_info["passed"],native_info
    with build_case(case=case,device=device,formulation="legacy") as model:
        previous=case.initial.copy()
        for index,factor in enumerate(factors):
            model.update_rhs(rhs=case.source*factor)
            model.update_initial_head(initial_head=previous)
            head,info=model.solve(transient=True,dt=2.0,storage_coeff=1e-3,head_prev=previous,
                max_cycles=1000,rel_tol=1e-14,abs_tol_min=1e-13,final_budget_relative_tol=1e-12,
                linear_acceleration=acceleration)
            assert info["converged"],info
            np.testing.assert_allclose(actual=head,desired=native_history["heads"][index],rtol=0.0,atol=2e-11)
            previous=head
