"""Native transient acceptance and pointer-stable conductivity refresh."""
from dataclasses import replace

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase
from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference


@pytest.fixture(params=['cpu', 'cuda:0'])
def device(request):
    if request.param.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    return request.param


def make_case(*, convertible, shape=(2, 4, 6)):
    k, j, i = np.indices(dimensions=shape)
    conductivity = 1.+.1*i+.2*j+.3*k
    fixed = np.zeros(shape, dtype=int)
    fixed[:, :, 0] = 1
    initial = 4.+.1*i
    initial[fixed != 0] = 4.
    types = np.zeros(shape, dtype=int)
    if convertible:
        types[0] = 1
    return NPF662LayeredCase(kx=conductivity, ky=.7*conductivity, kz=.2*conductivity,
        active=np.ones(shape, dtype=int), fixed=fixed, values=np.full(shape, 4.),
        source=np.where(fixed == 0, .01, 0.), initial=initial, dx=2., dy=3.,
        z_interfaces=np.broadcast_to(np.array([6., 3., 0.])[:, None, None], (3, *shape[1:])).copy(),
        cell_type=types)


def make_model(*, case, device):
    nz, ny, nx = case.active.shape
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
        device=device, implementation='fast', formulation='mf6_6_6_2_npf')
    model.build_from_layers(z_interfaces=case.z_interfaces, kx_field=case.kx, ky_field=case.ky,
        kz_field=case.kz, active=case.active, bc_mask=case.fixed, bc_values=case.values,
        rhs=case.source, initial_head=case.initial)
    return model


def solve_options(*, case, previous, dt):
    return dict(transient=True, dt=dt, head_prev=previous, ss=1e-3, sy=.15,
        unconfined=bool(np.any(case.cell_types)), npf_cell_type=case.cell_types,
        npf_device_native=True, npf_preconditioner='kcycle',
        npf_kcycle_controls=NPF662KCycleControls(levels=3, coarse_sweeps=10),
        max_outer=50, max_cycles=500, krylov_restart=12)


def fail_sparse_reference(*args, **kwargs):
    raise AssertionError('Native positive-storage solve rebuilt the sparse reference')


@pytest.mark.parametrize('convertible', [False, True])
@pytest.mark.parametrize('shape', [(2, 4, 6), (2, 17, 19)])
def test_positive_storage_native_matches_independent_without_sparse_build(device, convertible, shape, monkeypatch):
    case = make_case(convertible=convertible, shape=shape)
    previous = case.initial.copy()
    oracle = NPF662TransientReference(case=case, previous=previous, dt=7., ss=1e-3, sy=.15)
    with make_model(case=case, device=device) as model:
        monkeypatch.setattr(NPF662TransientReference, '__init__', fail_sparse_reference)
        head, info = model.solve(**solve_options(case=case, previous=previous, dt=7.))
    checks, fields = oracle.evaluate(head=head)
    assert info['converged'] and checks['independent_accepted']
    assert not info['independent_audit_performed']
    assert info['independent_acceptance'] is None
    assert info['physical_acceptance']['all_free_cells_storage_anchored']
    np.testing.assert_array_equal(actual=previous, desired=case.initial)


def test_conductivity_and_dt_refresh_preserve_levels_and_graph(device):
    case = make_case(convertible=True)
    with make_model(case=case, device=device) as model:
        first, info = model.solve(**solve_options(case=case, previous=case.initial, dt=7.))
        assert info['converged']
        workspace = model._session._npf662_workspace_holder['workspace']
        hierarchy = workspace.multigrid
        graph = hierarchy.graph
        pointers = [tuple(array.ptr for array in level['faces']) for level in hierarchy.levels]
        changed = replace(case, kx=case.kx*2., ky=case.ky*1.5, kz=case.kz*.8)
        model.update_K_fields(kx_field=changed.kx, ky_field=changed.ky, kz_field=changed.kz)
        model.update_initial_head(initial_head=first)
        head, updated = model.solve(**solve_options(case=changed, previous=first, dt=3.5))
        assert updated['converged']
        assert workspace.multigrid is hierarchy
        assert hierarchy.graph is graph
        assert [tuple(array.ptr for array in level['faces']) for level in hierarchy.levels] == pointers
        assert updated['multigrid']['coefficients_refreshed']
        assert updated['multigrid']['hierarchy_reused']
    checks, fields = NPF662TransientReference(case=changed, previous=first, dt=3.5,
        ss=1e-3, sy=.15).evaluate(head=head)
    assert checks['independent_accepted']


def test_previous_storage_snapshot_is_distinct_from_iterative_tangent(device):
    case = make_case(convertible=True)
    previous = case.initial.copy()
    oracle = NPF662TransientReference(case=case, previous=previous, dt=7., ss=1e-3, sy=.15)
    with make_model(case=case, device=device) as model:
        model.update_initial_head(initial_head=case.initial+.2)
        head, info = model.solve(**solve_options(case=case, previous=previous, dt=7.))
        assert info['converged']
        state = model._session._npf662_workspace_holder['native_state']
        workspace = model._session._npf662_workspace_holder['workspace']
        np.testing.assert_allclose(actual=state.preconditioner_storage,
            desired=oracle.preconditioner_storage, rtol=1e-12, atol=1e-14)
        assert not np.shares_memory(state.preconditioner_storage, workspace.storage_tangent.numpy())
        np.testing.assert_array_equal(actual=state.previous, desired=previous)


def test_dry_zero_storage_falls_back_and_rejects_ungrounded_state(device):
    one = np.ones((1, 1, 1))
    case = NPF662LayeredCase(kx=one, ky=one, kz=one, active=one.astype(int),
        fixed=np.zeros_like(one, dtype=int), values=np.zeros_like(one), source=np.zeros_like(one),
        initial=-one, dx=1., dy=1., z_interfaces=np.array([[[3.]], [[0.]]]), cell_type=1)
    with make_model(case=case, device=device) as model:
        head, info = model.solve(transient=True, dt=7., head_prev=case.initial, ss=0., sy=.15,
            unconfined=True, npf_cell_type=1, npf_device_native=True, max_outer=0, max_cycles=0)
    assert not info['converged']
    assert info['independent_audit_performed']
    assert info['independent_acceptance']['ungrounded_current_components'] == 1


@pytest.mark.parametrize('shape', [(2, 5, 7), (2, 1, 7)])
def test_device_storage_hierarchy_matches_fresh_after_material_updates(device, shape):
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleWorkspace
    case = make_case(convertible=True, shape=shape)
    case.active[:, -1, -1] = 0
    case.source[:, -1, -1] = 0.
    controls = NPF662KCycleControls(levels=4)
    storage = np.where((case.active != 0)&(case.fixed == 0), .03, 0.)
    with make_model(case=case, device=device) as model:
        hierarchy = NPF662KCycleWorkspace(faces=model._session.faces, active=case.active,
            fixed=case.fixed, device=device, controls=controls, storage_diag=storage)
        try:
            graph = hierarchy.graph
            for step in range(3):
                if step == 1:
                    model.update_K_fields(kx_field=case.kx*1.7, ky_field=case.ky*.6, kz_field=case.kz*2.)
                changed = storage*(step+2)
                hierarchy.refresh(faces=model._session.faces, active=case.active,
                    fixed=case.fixed, storage_diag=changed)
                fresh = NPF662KCycleWorkspace(faces=model._session.faces, active=case.active,
                    fixed=case.fixed, device=device, controls=controls, storage_diag=changed)
                try:
                    for updated, expected in zip(hierarchy.levels, fresh.levels):
                        for actual_face, expected_face in zip(updated['faces'], expected['faces']):
                            np.testing.assert_allclose(actual=actual_face.numpy(),
                                desired=expected_face.numpy(), rtol=2e-14, atol=1e-14)
                        for name in ('storage', 'diag', 'boundary_springs'):
                            np.testing.assert_allclose(actual=updated[name].numpy(),
                                desired=expected[name].numpy(), rtol=2e-14, atol=1e-14)
                finally:
                    fresh.close()
                assert hierarchy.graph is graph
            assert hierarchy.device_storage_refresh_count == 3
        finally:
            hierarchy.close()


@pytest.mark.parametrize('restart', [3, 12])
def test_device_arnoldi_preserves_solution_and_reduces_readbacks(device, restart):
    case = make_case(convertible=True)
    results = []
    for native in (False, True):
        with make_model(case=case, device=device) as model:
            options = solve_options(case=case, previous=case.initial, dt=7.)
            options.update(npf_device_native=native, krylov_restart=restart)
            results.append(model.solve(**options))
    baseline, accelerated = results
    assert baseline[1]['converged'] and accelerated[1]['converged']
    np.testing.assert_allclose(actual=accelerated[0], desired=baseline[0], rtol=0., atol=1e-11)
    assert baseline[1]['krylov_iterations'] == accelerated[1]['krylov_iterations']
    assert accelerated[1]['krylov_scalar_readbacks'] < baseline[1]['krylov_scalar_readbacks']
    assert all(row['device_arnoldi'] for row in accelerated[1]['linear_history'])
    if device.startswith('cuda'):
        assert all(row['arnoldi_graphs'] > 0 and not row['arnoldi_graph_fallback']
            for row in accelerated[1]['linear_history'])


def test_arnoldi_capture_failure_uses_valid_eager_fallback(monkeypatch):
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    case = make_case(convertible=True)

    def reject_capture(*args, **kwargs):
        raise RuntimeError('Injected capture failure')

    with make_model(case=case, device='cuda:0') as model:
        options = solve_options(case=case, previous=case.initial, dt=7.)
        expected, first = model.solve(**options)
        assert first['converged']
        workspace = model._session._npf662_workspace_holder['workspace']
        workspace.flexible.arnoldi_graphs.clear()
        monkeypatch.setattr(wp, 'ScopedCapture', reject_capture)
        actual, info = model.solve(**options)
    assert info['converged']
    assert all(row['arnoldi_graph_fallback'] for row in info['linear_history'])
    np.testing.assert_allclose(actual=actual, desired=expected, rtol=0., atol=1e-11)
