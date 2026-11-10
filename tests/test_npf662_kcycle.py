"""Flexible K-cycle scaling, nonlinear physics, safeguards and owned resources."""
from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace, NPF662NewtonControls
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls, NPF662KCycleWorkspace
from DARCY_WARP_PACKAGE.solvers.npf662_fgmres import RestartedFGMRES3D, masked_axpy
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference
from validate_npf662_confined import make_gpu_model
from validate_npf662_dry_start import make_dry_case
from test_npf662_layers import layered_case
from test_npf662_newton import unconfined_case


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('Local CUDA unavailable')
    return request.param


def test_manufactured_scaled_linear_and_fixed_mask(device):
    case = layered_case(masked=True, shape=(5, 7, 9))
    reference = NPF662Reference(case=case)
    with make_gpu_model(case=case, device=device) as model:
        ws = NPF662Workspace(model=model, case=case, restart=12, krylov_budget=200)
        mg = NPF662KCycleWorkspace(faces=model._session.faces, active=case.active, fixed=case.fixed,
                                 device=device, controls=NPF662KCycleControls())
        gmres = RestartedFGMRES3D(workspace=ws, restart=12)
        exact = np.random.default_rng(seed=991).normal(size=case.active.size)
        exact[~reference.free.ravel()] = 0.
        physical = reference.jacobian(head=case.initial) @ exact
        scaled = physical / ws.scale.numpy()
        ws.rhs.assign(scaled)
        used, error, target = gmres.solve(preconditioner=mg, max_iterations=200, absolute_tolerance=1e-11)
        actual = ws.delta.numpy()
        true = scaled - (reference.jacobian(head=case.initial) @ actual) / ws.scale.numpy()
        assert error <= target and used <= 200, gmres.last_info
        np.testing.assert_allclose(actual=actual, desired=exact, rtol=0., atol=2e-8)
        assert np.linalg.norm(true[reference.free.ravel()]) <= target * 1.01
        np.testing.assert_array_equal(actual[~reference.free.ravel()], 0.)
        assert mg.applications == used
        assert [level['dim'][0] for level in mg.levels] == [5] * len(mg.levels)
        # Repeated application resets all corrections; affine CHD values never enter.
        vector = wp.array(data=scaled, dtype=wp.float64, device=device)
        mg.apply(vector=vector, out=ws.delta, diagonal=ws.scale, free=ws.free)
        once = ws.delta.numpy()
        mg.apply(vector=vector, out=ws.delta, diagonal=ws.scale, free=ws.free)
        np.testing.assert_allclose(actual=ws.delta.numpy(), desired=once, rtol=1e-13, atol=1e-13)
        if device.startswith('cuda'):
            assert mg.graph_captures == 1 and mg.graph_replays == used + 2 and not mg.graph_failed
        mg.close(); mg.close(); gmres.close(); gmres.close(); ws.close()


@pytest.mark.parametrize('initial', ['wet', 'dry', 'mixed'])
def test_nonlinear_matches_independent_root(device, initial):
    case = unconfined_case(initial=initial, heterogeneous=True)
    expected, checks, unused = NPF662Reference(case=case).solve(max_outer=200)
    assert checks['independent_accepted']
    with make_gpu_model(case=case, device=device) as model:
        head, info = model.solve(unconfined=True, top_elevation=case.top, npf_cell_type=case.cell_types,
            npf_preconditioner='kcycle', krylov_restart=12, max_outer=200, max_cycles=200)
        assert info['converged'] and info['independent_acceptance']['independent_accepted'], info
        np.testing.assert_allclose(actual=head, desired=expected, rtol=0., atol=1e-7)
        assert info['krylov_method'] == 'FGMRES'
        assert info['multigrid']['applications'] == info['krylov_iterations']
        assert info['multigrid']['inner_scalar_readbacks'] == 0
        assert all(row['krylov'] <= 200 for row in info['outer_history'])


def test_reuse_source_values_material_geometry_and_close(device):
    case = layered_case(masked=True, shape=(5, 5, 7))
    model = make_gpu_model(case=case, device=device)
    options = dict(npf_preconditioner='kcycle', krylov_restart=12, max_cycles=200, max_outer=80)
    first_head, first = model.solve(**options)
    assert first['converged'], first
    ws = model._session._npf662_workspace_holder['workspace']
    mg, gmres = ws.multigrid, ws.flexible
    case.values += .2
    case.source *= 2.
    model.update_boundary_values(bc_values=case.values)
    model.update_rhs(rhs=case.source)
    second_head, second = model.solve(**options)
    assert second['converged'] and second['multigrid']['hierarchy_reused']
    assert ws.multigrid is mg and ws.flexible is gmres
    expected, checks, unused = NPF662Reference(case=case).solve()
    np.testing.assert_allclose(actual=second_head[case.active != 0], desired=expected[case.active != 0], rtol=0., atol=1e-7)
    changed_faces = tuple(value * 1.1 for value in model._session.faces)
    assert not mg.compatible(faces=changed_faces, active=case.active, fixed=case.fixed,
                             device=device, controls=NPF662KCycleControls())
    changed_fixed = case.fixed.copy(); changed_fixed[:, 0, 1] = 1
    assert not mg.compatible(faces=model._session.faces, active=case.active, fixed=changed_fixed,
                             device=device, controls=NPF662KCycleControls())
    for changed_case in (replace(case, kx=case.kx * 1.1),
                         replace(case, z_interfaces=case.z_interfaces * 1.02)):
        model.build_from_layers(z_interfaces=changed_case.z_interfaces, kx_field=changed_case.kx,
            ky_field=changed_case.ky, kz_field=changed_case.kz, active=changed_case.active,
            bc_mask=changed_case.fixed, bc_values=changed_case.values,
            rhs=changed_case.source, initial_head=changed_case.initial)
        assert mg.closed and gmres.closed and ws.closed
        head, info = model.solve(**options)
        assert info['converged'] and not info['multigrid']['hierarchy_reused'], info
        expected, checks, unused = NPF662Reference(case=changed_case).solve()
        np.testing.assert_allclose(actual=head[case.active != 0], desired=expected[case.active != 0], rtol=0., atol=1e-7)
        ws = model._session._npf662_workspace_holder['workspace']; mg = ws.multigrid; gmres = ws.flexible
    model.close(); model.close()
    assert mg.closed and gmres.closed and ws.closed


def test_zero_rows_and_shared_exhausted_budgets(device):
    case = make_dry_case(initial=-.2, source=1.)
    with make_gpu_model(case=case, device=device) as model:
        head, info = model.solve(unconfined=True, top_elevation=4., npf_preconditioner='kcycle',
                                 krylov_restart=4, max_cycles=24, max_outer=30)
        assert info['converged'] and info['full_saturated_safeguard_solves'] > 0, info
        assert info['outer_history'][0]['attempts'][0]['stopping'] == 'zero_row_nonzero_rhs'
        head, failed = model.solve(unconfined=True, top_elevation=4., npf_preconditioner='kcycle',
                                   krylov_restart=4, max_cycles=0, max_outer=1)
        assert not failed['converged'] and failed['krylov_iterations'] == 0
        case.initial[:] = .001
        model.update_initial_head(initial_head=case.initial)
        head, failed = model.solve(unconfined=True, top_elevation=4., npf_preconditioner='kcycle',
            krylov_restart=4, max_cycles=5, max_outer=1,
            npf_newton_controls=NPF662NewtonControls(line_search_steps=1, recovery_growth=1.))
        assert all(row['krylov'] <= 5 for row in failed['outer_history'])


def test_nonfinite_and_zero_operator_breakdowns(device):
    case = layered_case(shape=(2, 3, 5))
    with make_gpu_model(case=case, device=device) as model:
        ws = NPF662Workspace(model=model, case=case, restart=4, krylov_budget=5)
        mg = NPF662KCycleWorkspace(faces=model._session.faces, active=case.active,
            fixed=case.fixed, device=device, controls=NPF662KCycleControls())
        gmres = RestartedFGMRES3D(workspace=ws, restart=4)
        ws.rhs.fill_(np.nan)
        used, error, target = gmres.solve(preconditioner=mg, max_iterations=5, absolute_tolerance=1e-12)
        assert used == 0 and not gmres.last_info['converged']
        ws.rhs.fill_(1.)

        def zero_action(x, y, z, alpha, beta):
            z.zero_()

        ws.matvec = zero_action
        used, error, target = gmres.solve(preconditioner=mg, max_iterations=5, absolute_tolerance=1e-12)
        assert used == 1 and not gmres.last_info['converged']
        assert gmres.last_info['stopping'] == 'arnoldi_breakdown'
        mg.close(); gmres.close(); ws.close()


class AlternatingPreconditioner:
    """Deliberately nonstationary action to detect incorrect V reconstruction."""
    def __init__(self, *, multigrid):
        self.multigrid = multigrid
        self.device_seconds = 0.
        self.applications = 0

    def apply(self, *, vector, out, diagonal, free):
        self.multigrid.apply(vector=vector, out=out, diagonal=diagonal, free=free)
        self.applications += 1
        factor = .5 if self.applications % 2 else 1.5
        wp.launch(kernel=masked_axpy, dim=vector.size, inputs=[out, out, free, factor-1.],
                  device=self.multigrid.device)

    def collect_timing(self):
        self.multigrid.collect_timing()
        self.device_seconds = self.multigrid.device_seconds


def test_flexible_basis_reconstruction_and_short_restart(device):
    case = layered_case(masked=True, shape=(3, 5, 7))
    reference = NPF662Reference(case=case)
    with make_gpu_model(case=case, device=device) as model:
        ws = NPF662Workspace(model=model, case=case, restart=4, krylov_budget=200)
        mg = NPF662KCycleWorkspace(faces=model._session.faces, active=case.active,
            fixed=case.fixed, device=device, controls=NPF662KCycleControls())
        variable = AlternatingPreconditioner(multigrid=mg)
        gmres = RestartedFGMRES3D(workspace=ws, restart=4)
        exact = np.random.default_rng(seed=61).normal(size=case.active.size)
        exact[~reference.free.ravel()] = 0.
        rhs = (reference.jacobian(head=case.initial) @ exact) / ws.scale.numpy()
        ws.rhs.assign(rhs)
        used, error, target = gmres.solve(preconditioner=variable, max_iterations=200, absolute_tolerance=1e-10)
        assert error <= target and used > 4, gmres.last_info
        np.testing.assert_allclose(actual=ws.delta.numpy(), desired=exact, rtol=0., atol=1e-7)
        used, error, target = gmres.solve(preconditioner=variable, max_iterations=5, absolute_tolerance=1e-30)
        assert used == 5 and gmres.last_info['blocks'] == 2 and not gmres.last_info['converged']
        mg.close(); gmres.close(); ws.close()


def test_lazy_selection_switching_and_shared_operator_buffers(device):
    case = layered_case(masked=True, shape=(5, 5, 7))
    with make_gpu_model(case=case, device=device) as model:
        options = dict(krylov_restart=12, max_cycles=200, max_outer=80)
        head, first = model.solve(npf_preconditioner='kcycle', **options)
        assert first['converged']
        ws = model._session._npf662_workspace_holder['workspace']
        assert ws.krylov is None and ws.flexible is not None
        mg, flexible = ws.multigrid, ws.flexible
        pointers = {name: getattr(ws, name).ptr for name in ('c0', 'c', 'dl', 'dr', 'scale', 'rhs')}
        head, diagnostic = model.solve(npf_preconditioner='diagonal', npf_krylov_method='fgmres', **options)
        assert diagnostic['converged'] and diagnostic['krylov_method'] == 'FGMRES'
        assert diagnostic['preconditioner_applications'] == diagnostic['krylov_iterations']
        assert diagnostic['preconditioner_cycles'] == 0 and 'multigrid' not in diagnostic
        assert ws.flexible is flexible and ws.multigrid is mg and ws.krylov is None
        head, diagonal = model.solve(**options)
        assert diagonal['converged'] and diagonal['krylov_method'] == 'GMRES'
        assert flexible.closed and ws.flexible is None and ws.krylov is not None
        legacy = weakref.ref(ws.krylov)
        head, final = model.solve(npf_preconditioner='kcycle', **options)
        gc.collect()
        assert legacy() is None and ws.krylov is None
        assert final['converged'] and final['multigrid']['hierarchy_reused']
        assert ws.multigrid is mg and ws.flexible is not flexible
        assert pointers == {name: getattr(ws, name).ptr for name in pointers}
        assert final['krylov_iterations'] == first['krylov_iterations']
        with pytest.raises(ValueError, match='flexible'):
            model.solve(npf_preconditioner='kcycle', npf_krylov_method='gmres')


@pytest.mark.parametrize('initial', ['wet', 'dry', 'mixed'])
def test_diagonal_fgmres_matches_kcycle_on_identical_nonlinear_problem(device, initial):
    case = unconfined_case(initial=initial, heterogeneous=True)
    expected, checks, unused = NPF662Reference(case=case).solve(max_outer=200)
    assert checks['independent_accepted']
    with make_gpu_model(case=case, device=device) as model:
        for preconditioner in ('diagonal', 'kcycle'):
            head, info = model.solve(unconfined=True, top_elevation=case.top, npf_cell_type=case.cell_types,
                npf_krylov_method='fgmres', npf_preconditioner=preconditioner,
                krylov_restart=12, max_cycles=200, max_outer=200)
            assert info['converged'] and info['krylov_method'] == 'FGMRES', info
            np.testing.assert_allclose(actual=head, desired=expected, rtol=0., atol=1e-7)
            assert info['preconditioner_applications'] == info['krylov_iterations']
            assert model._session._npf662_workspace_holder['workspace'].krylov is None


def test_diagonal_fgmres_linear_reference_and_value_updates(device):
    from DARCY_WARP_PACKAGE.solvers.npf662_fgmres import DiagonalFGMRESPreconditioner
    case = layered_case(masked=True, shape=(3, 5, 7))
    reference = NPF662Reference(case=case)
    with make_gpu_model(case=case, device=device) as model:
        ws = NPF662Workspace(model=model, case=case, restart=12, krylov_budget=200)
        assert ws.krylov is None and ws.flexible is None
        ws.select_krylov(method='fgmres', krylov_budget=200)
        preconditioner = DiagonalFGMRESPreconditioner(workspace=ws)
        exact = np.random.default_rng(seed=222).normal(size=case.active.size)
        exact[~reference.free.ravel()] = 0.
        rhs = (reference.jacobian(head=case.initial) @ exact)/ws.scale.numpy()
        ws.rhs.assign(rhs)
        used, error, target = ws.flexible.solve(preconditioner=preconditioner,
                                               max_iterations=200, absolute_tolerance=1e-11)
        assert error <= target and used == preconditioner.applications
        np.testing.assert_allclose(actual=ws.delta.numpy(), desired=exact, rtol=0., atol=2e-8)
        preconditioner.close(); ws.close()
        options = dict(npf_preconditioner='diagonal', npf_krylov_method='fgmres',
                       krylov_restart=12, max_cycles=200, max_outer=80)
        head, info = model.solve(**options)
        owned = model._session._npf662_workspace_holder['workspace']
        basis_pointer = owned.flexible.V[0].ptr
        case.values += .1; case.source *= 2.
        model.update_boundary_values(bc_values=case.values)
        model.update_rhs(rhs=case.source)
        head, info = model.solve(**options)
        assert info['converged'] and info['krylov_workspace_reused']
        assert owned.flexible.V[0].ptr == basis_pointer and owned.krylov is None
        expected, checks, unused = NPF662Reference(case=case).solve()
        np.testing.assert_allclose(actual=head[case.active != 0], desired=expected[case.active != 0], rtol=0., atol=1e-7)
