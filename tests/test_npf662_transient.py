"""Transient NPF storage, independent tangents, analytic roots and step ownership."""
from dataclasses import replace
import numpy as np
import pytest
import warp as wp
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable; transient GPU coverage not executed')
    return request.param


def test_confined_analytic_step_and_reused_previous_state(device):
    shape = (1, 1, 2)
    one = np.ones(shape)
    case = NPF662LayeredCase(kx=one, ky=one, kz=one, active=one.astype(int),
        fixed=np.array([0, 1]).reshape(shape), values=2.*one, source=np.array([.4, 0.]).reshape(shape),
        initial=np.array([.5, 2.]).reshape(shape), dx=1., dy=1.,
        z_interfaces=np.stack([3.*one[0], 0.*one[0]]), cell_type=0)
    previous = case.initial.copy()
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=1., dy=1., device=device,
            formulation='mf6_6_6_2_npf', implementation='fast') as model:
        model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx, ky_field=case.ky,
            kz_field=case.kz, active=case.active, bc_mask=case.fixed, bc_values=case.values,
            rhs=case.source, initial_head=case.initial)
        for step, dt in enumerate([7., 7., 14.]):
            old = previous.copy()
            capacity = .2*3./dt
            expected = (.4+3.*2.+capacity*old[0, 0, 0])/(3.+capacity)
            model.update_initial_head(initial_head=previous)
            head, info = model.solve(transient=True, dt=dt, head_prev=previous, ss=.2, sy=0.,
                npf_preconditioner='kcycle', npf_kcycle_controls=NPF662KCycleControls(use_cuda_graph=False),
                max_outer=30, max_cycles=100, krylov_restart=4)
            assert info['converged'], info
            np.testing.assert_allclose(head[0, 0, 0], expected, rtol=0., atol=1e-9)
            np.testing.assert_array_equal(previous, old)
            assert info['effective_options']['storage'] and info['dt'] == dt
            if step == 1:
                assert info['workspace_reused'] and info['multigrid']['hierarchy_reused']
            if step == 2:
                assert info['workspace_reused'] and not info['multigrid']['hierarchy_reused']
            previous = head.copy()
        # A following steady solve must not inherit the previous STO terms.
        head, info = model.solve(npf_preconditioner='kcycle',
            npf_kcycle_controls=NPF662KCycleControls(use_cuda_graph=False), max_outer=30, max_cycles=100)
        assert info['converged'] and not info['effective_options']['storage']
        np.testing.assert_allclose(head[0, 0, 0], 2.+.4/3., rtol=0., atol=1e-9)


@pytest.mark.parametrize('fraction', [-.1, .3e-6, .45, 1.-.3e-6, 1.2])
def test_storage_and_full_device_jacobian_against_finite_differences(fraction, device):
    shape = (2, 2, 3)
    k, j, i = np.indices(shape)
    one = np.ones(shape)
    interfaces = np.broadcast_to(np.array([6., 3., 0.])[:, None, None], (3, 2, 3)).copy()
    fixed = np.zeros(shape, dtype=int)
    fixed[:, :, 0] = 1
    values = 7.*one
    previous = interfaces[1:]+.2*3.
    case = NPF662LayeredCase(kx=one, ky=.7*one, kz=.2*one, active=one.astype(int), fixed=fixed,
        values=values, source=np.zeros(shape), initial=interfaces[1:]+fraction*3.,
        dx=2., dy=3., z_interfaces=interfaces, cell_type=1)
    case.initial[fixed != 0] = values[fixed != 0]
    reference = NPF662TransientReference(case=case, previous=previous, dt=7., ss=1e-3, sy=.15)
    direction = np.random.default_rng(521).normal(size=shape)
    direction[~reference.free] = 0.
    expected = (reference.jacobian(head=case.initial)@direction.ravel()).reshape(shape)
    errors = []
    for step in [1e-6, 1e-8, 1e-10]:
        unused, plus = reference.evaluate(head=case.initial+step*direction)
        unused, minus = reference.evaluate(head=case.initial-step*direction)
        finite = -(plus['residual']-minus['residual'])/(2.*step)
        errors.append(np.linalg.norm((finite-expected)[reference.free])/np.linalg.norm(expected[reference.free]))
    assert min(errors) < 1e-6
    with WarpDarcySolver3D(nx=3, ny=2, nz=2, dx=2., dy=3., device=device,
            formulation='mf6_6_6_2_npf', implementation='fast') as model:
        model.build_from_layers(z_interfaces=interfaces, kx_field=case.kx, ky_field=case.ky,
            kz_field=case.kz, active=case.active, bc_mask=fixed, bc_values=values,
            rhs=case.source, initial_head=case.initial)
        workspace = NPF662Workspace(model=model, case=case, reference=reference, restart=4, krylov_budget=100)
        x = wp.array(direction.ravel(), dtype=wp.float64, device=device)
        out = wp.zeros_like(x)
        workspace.matvec(x, out, out, 1., 0.)
        np.testing.assert_allclose((out.numpy()*workspace.scale.numpy()).reshape(shape)[reference.free],
            expected[reference.free], rtol=3e-12, atol=1e-12)
        rate, tangent = reference.storage(head=case.initial)
        np.testing.assert_allclose(workspace.storage_rate.numpy().reshape(shape), rate, rtol=3e-12, atol=1e-12)
        np.testing.assert_allclose(workspace.storage_tangent.numpy().reshape(shape), tangent, rtol=3e-12, atol=1e-12)
        workspace.close()


def test_storage_validation_rejects_invalid_inputs():
    shape = (1, 1, 2)
    one = np.ones(shape)
    case = NPF662LayeredCase(kx=one, ky=one, kz=one, active=one.astype(int), fixed=np.array([0, 1]).reshape(shape),
        values=one, source=np.zeros(shape), initial=one, dx=1., dy=1.,
        z_interfaces=np.stack([2.*one[0], 0.*one[0]]), cell_type=1)
    for invalid in [dict(dt=0.), dict(ss=-1.), dict(sy=1.5), dict(previous=one*np.nan)]:
        values = dict(case=case, previous=one, dt=7., ss=1e-5, sy=.15)
        values.update(invalid)
        with pytest.raises(ValueError):
            NPF662TransientReference(**values)


def test_storage_anchors_an_isolated_transient_cell(device):
    shape = (1, 1, 1)
    one = np.ones(shape)
    case = NPF662LayeredCase(kx=one, ky=one, kz=one, active=one.astype(int),
        fixed=np.zeros(shape, dtype=int), values=np.zeros(shape), source=.2*one,
        initial=one, dx=1., dy=1., z_interfaces=np.stack([3.*one[0], 0.*one[0]]), cell_type=0)
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1., dy=1., device=device,
            formulation='mf6_6_6_2_npf', implementation='fast') as model:
        model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx, ky_field=case.ky,
            kz_field=case.kz, active=case.active, bc_mask=case.fixed, bc_values=case.values,
            rhs=case.source, initial_head=case.initial)
        head, info = model.solve(transient=True, dt=7., head_prev=case.initial, ss=.2, sy=0.,
            npf_preconditioner='kcycle', npf_kcycle_controls=NPF662KCycleControls(use_cuda_graph=False),
            max_outer=30, max_cycles=100, krylov_restart=4)
        assert info['converged'], info
        np.testing.assert_allclose(head, 1.+.2*7./(.2*3.), rtol=0., atol=1e-9)
