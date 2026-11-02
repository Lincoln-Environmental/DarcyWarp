"""Public 3D state, mask, control and independently evaluated budget regressions."""
from dataclasses import replace
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.physics.budget_3d import compute_3d_flow_budget, is_3d_budget_conserved
from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls


def inputs(*, shape=(2, 3, 5)):
    fixed = np.zeros(shape, dtype=int)
    fixed[:, :, 0] = fixed[:, :, -1] = 1
    values = np.broadcast_to(np.linspace(1., 2., shape[2]), shape).copy()
    return dict(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
                active=np.ones(shape, dtype=int), bc_mask=fixed, bc_values=values,
                rhs=np.zeros(shape), initial_head=values.copy())


def model_from(*, data, formulation='legacy', geometry='cartesian'):
    nz, ny, nx = data['active'].shape
    model = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=1., device='cpu',
                              implementation='fast', formulation=formulation)
    build(model=model, data=data, geometry=geometry)
    return model


def build(*, model, data, geometry):
    if geometry == 'terrain':
        kwargs = dict(data)
        kwargs.pop('active')
        model.build_from_terrain(z_interfaces=np.broadcast_to(
            -np.arange(model.nz+1)[:, None, None], (model.nz+1, model.ny, model.nx)), **kwargs)
    elif geometry == 'layered':
        model.build_from_layers(z_interfaces=np.broadcast_to(
            -np.arange(model.nz+1)[:, None, None], (model.nz+1, model.ny, model.nx)), **data)
    else:
        model.build_from_K_fields(**data)


@pytest.mark.parametrize('geometry', ['cartesian', 'layered', 'terrain'])
@pytest.mark.parametrize('field', ['active', 'bc_mask'])
@pytest.mark.parametrize('value', [.9, 1.9, -.1, np.nan, np.inf])
def test_masks_validated_before_conversion(geometry, field, value):
    if geometry == 'terrain' and field == 'active':
        with pytest.raises(ValueError):
            TerrainGrid3D(z_interfaces=np.broadcast_to(-np.arange(3.)[:, None, None], (3, 2, 2)), dx=1., dy=1.,
                          active_columns=np.full((2, 2), value))
        return
    data = inputs()
    data[field] = data[field].astype(float)
    data[field][0, 0, 1] = value
    with pytest.raises(ValueError):
        model_from(data=data, geometry=geometry,
                   formulation='mf6_6_6_2_npf' if geometry == 'layered' else 'legacy')


@pytest.mark.parametrize('geometry', ['cartesian', 'layered', 'terrain'])
@pytest.mark.parametrize('defect', ['shape', 'nonfinite', 'mask'])
def test_failed_rebuild_preserves_usable_model(geometry, defect):
    data = inputs()
    formulation = 'mf6_6_6_2_npf' if geometry == 'layered' else 'legacy'
    with model_from(data=data, formulation=formulation, geometry=geometry) as model:
        old_session, old_geometry = model._session, model.geometry
        expected, unused = model.solve()
        bad = dict(data)
        if defect == 'shape': bad['rhs'] = np.zeros((1, 1, 2))
        if defect == 'nonfinite': bad['rhs'] = np.full(model.shape, np.nan)
        if defect == 'mask': bad['bc_mask'] = np.full(model.shape, 2)
        with pytest.raises(ValueError):
            build(model=model, data=bad, geometry=geometry)
        assert model._session is old_session and model.geometry == old_geometry
        actual, info = model.solve()
        assert info['converged']
        np.testing.assert_allclose(actual=actual, desired=expected, rtol=0., atol=1e-10)


def test_physical_model_rejects_raw_face_update():
    with model_from(data=inputs()) as model:
        with pytest.raises(ValueError, match='physical|conductivity|K_fields'):
            model.update_face_conductance(**dict(zip(
                ('tx_p', 'tx_m', 'ty_p', 'ty_m', 'tz_p', 'tz_m'), model._session.faces)))


@pytest.mark.parametrize('option', [dict(rel_tol=1e-5), dict(max_levels=3), dict(min_coarse_n=3)])
def test_uniform_npf_newton_dispatch_rejects_legacy_overrides(option):
    with model_from(data=inputs(), formulation='mf6_6_6_2_npf') as model:
        with pytest.raises(ValueError, match='npf_'):
            model.solve(npf_preconditioner='kcycle', **option)


def test_explicit_gmres_is_honoured_and_unused_controls_rejected():
    with model_from(data=inputs(), formulation='mf6_6_6_2_npf') as model:
        unused, info = model.solve(npf_krylov_method='gmres')
        assert info['krylov_method'] == 'GMRES'
        with pytest.raises(ValueError, match='npf_'):
            model.solve(npf_newton_controls=object())


@pytest.mark.parametrize('scale', [1e-200, 1., 1e200])
def test_cartesian_budget_stable_residual_rms(scale):
    shape = (1, 1, 3)
    faces = tuple(np.zeros(shape) for unused in range(6))
    source = np.array([scale, -scale, 0.]).reshape(shape)
    budget = compute_3d_flow_budget(head=np.zeros(shape), source=source, faces=faces,
        active=np.ones(shape), bc_mask=np.zeros(shape))
    assert budget['cell_residual_rms'] == pytest.approx(scale*np.sqrt(2./3.), rel=1e-14, abs=0.)
    assert not is_3d_budget_conserved(budget=budget, absolute_tolerance=0., relative_tolerance=1e-8)


def test_terrain_overflow_cannot_pass_physical_acceptance():
    shape = (2, 2, 5)
    grid = TerrainGrid3D(z_interfaces=np.broadcast_to(-np.arange(3.)[:, None, None], (3, 2, 5)), dx=1., dy=1.)
    operator = TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=grid), kx=1e307, ky=1., kz=1.)
    fixed = np.zeros(shape, dtype=int)
    fixed[:, :, 0] = fixed[:, :, -1] = 1
    head = np.broadcast_to(np.array([0., 10., 10., 10., 0.]), shape).copy()
    with np.errstate(over='ignore', invalid='ignore'):
        budget = compute_terrain_budget_3d(operator=operator, head=head, source=np.zeros(shape), fixed=fixed,
            storage_change=np.zeros(shape), storage_diagonal=np.zeros(shape))
    assert not is_3d_budget_conserved(budget=budget, absolute_tolerance=1e-12, relative_tolerance=1e-8)
    assert not budget['physical_metrics_valid']


def test_returned_heads_remain_owned_across_cpu_workspace_reuse():
    for formulation in ('legacy', 'mf6_6_6_2_npf'):
        with model_from(data=inputs(), formulation=formulation) as model:
            options = dict(npf_preconditioner='kcycle') if formulation != 'legacy' else {}
            first, unused = model.solve(**options)
            snapshot = first.copy()
            model.update_boundary_values(bc_values=model._bc_values+.2)
            unused, info = model.solve(**options)
            assert info['converged']
            np.testing.assert_array_equal(actual=first, desired=snapshot)


@pytest.mark.parametrize('formulation,unconfined', [('legacy', False), ('legacy', True),
    ('mf6_6_6_2_npf', False), ('mf6_6_6_2_npf', True)])
def test_updated_physical_model_matches_fresh_and_repeated(formulation, unconfined):
    data = inputs()
    data['rhs'][0, 1, 2] = .005
    options = dict(unconfined=unconfined)
    if formulation == 'legacy':
        if unconfined:
            options.update(zbot_field=np.broadcast_to(-np.arange(1, 3.)[:, None, None], data['active'].shape),
                           unconfined_max_picard_iter=100, unconfined_head_tol=1e-8, final_budget_relative_tol=1e-8)
    else:
        options.update(npf_preconditioner='kcycle')
        if unconfined: options.update(top_elevation=1.)
    changed = dict(data)
    changed['kx_field'] = np.broadcast_to(np.linspace(.3, 2., 5), data['active'].shape).copy()
    changed['ky_field'] = .4*data['ky_field']
    changed['kz_field'] = 1.4*data['kz_field']
    with model_from(data=data, formulation=formulation) as model:
        unused, unused_info = model.solve(**options)
        model.update_K_fields(kx_field=changed['kx_field'], ky_field=changed['ky_field'], kz_field=changed['kz_field'])
        actual, info = model.solve(**options)
        repeat, repeated_info = model.solve(**options)
        assert info['converged'] and repeated_info['converged']
        with model_from(data=changed, formulation=formulation) as fresh:
            expected, fresh_info = fresh.solve(**options)
            assert fresh_info['converged']
        np.testing.assert_allclose(actual=actual, desired=expected, rtol=0., atol=1e-8)
        np.testing.assert_allclose(actual=actual, desired=repeat, rtol=0., atol=1e-10)
        if formulation != 'legacy':
            assert repeated_info['reference_reused']
            assert repeated_info['multigrid']['hierarchy_reused']
            assert info['effective_solver_settings']['hierarchy']['actual_levels'] >= 1
            for key in ('physical_rms', 'relative_cell_imbalance', 'head_residual_rms'):
                assert info['independent_acceptance'][key] == pytest.approx(fresh_info['independent_acceptance'][key], abs=1e-14)


@pytest.mark.parametrize('geometry', ['cartesian', 'layered', 'terrain'])
def test_replacement_constructor_failure_rolls_back(geometry, monkeypatch):
    data = inputs()
    formulation = 'mf6_6_6_2_npf' if geometry == 'layered' else 'legacy'
    with model_from(data=data, geometry=geometry, formulation=formulation) as model:
        original = model._session
        before, unused = model.solve()

        def reject_replacement(*args, **kwargs):
            raise RuntimeError('injected construction failure')

        target = ('DARCY_WARP_PACKAGE.solvers.terrain_fast_3d.TerrainOperatorSession3D' if geometry == 'terrain'
                  else 'DARCY_WARP_PACKAGE.warped_darcy_3d.ThreeDOperatorSession')
        with monkeypatch.context() as patch:
            patch.setattr(target, reject_replacement)
            with pytest.raises(RuntimeError, match='injected'):
                build(model=model, data=data, geometry=geometry)
        assert model._session is original
        after, info = model.solve()
        assert info['converged']
        np.testing.assert_allclose(actual=after, desired=before, rtol=0., atol=1e-12)


def test_invalid_contact_rebuild_leaves_model_usable():
    from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
    data = inputs()
    with model_from(data=data, geometry='terrain') as model:
        original = model._session
        bad = dict(data)
        bad.pop('active')
        contact = TerrainContact3D(cell=(0, 1, 2), direction='bad', head=0., area=1., distance=.5, elevation=0.)
        with pytest.raises(ValueError):
            model.build_from_terrain(z_interfaces=original.grid.z_interfaces, contacts=(contact,), **bad)
        assert model._session is original
        unused, info = model.solve()
        assert info['converged']


def test_valid_boolean_and_exact_binary_masks():
    for dtype in (bool, int, float):
        data = inputs()
        data['active'] = data['active'].astype(dtype)
        data['bc_mask'] = data['bc_mask'].astype(dtype)
        with model_from(data=data) as model:
            unused, info = model.solve()
            assert info['converged']


def test_hierarchy_controls_and_contradictions():
    with model_from(data=inputs(), formulation='mf6_6_6_2_npf') as model:
        unused, info = model.solve(npf_preconditioner='kcycle',
            npf_kcycle_controls=NPF662KCycleControls(levels=2, min_coarse_n=1), krylov_restart=5, max_outer=7)
        assert info['effective_solver_settings']['restart'] == 5
        assert info['multigrid']['controls']['levels'] == 2
        for options in (dict(npf_preconditioner='kcycle', npf_krylov_method='gmres'),
                        dict(npf_preconditioner='diagonal', npf_kcycle_controls=NPF662KCycleControls()),
                        dict(max_outer=3), dict(krylov_restart=5)):
            with pytest.raises(ValueError):
                model.solve(**options)






def test_terrain_counts_actual_preconditioner_calls(monkeypatch):
    from DARCY_WARP_PACKAGE.solvers.terrain_fast_3d import TerrainGPUWorkspace3D
    original = TerrainGPUWorkspace3D.precondition
    calls = []

    def counted_precondition(self, x, y, z, alpha, beta):
        calls.append(1)
        return original(self=self, x=x, y=y, z=z, alpha=alpha, beta=beta)

    monkeypatch.setattr(TerrainGPUWorkspace3D, 'precondition', counted_precondition)
    data = inputs()
    data['rhs'][0, 1, 2] = .1
    with model_from(data=data, geometry='terrain') as model:
        unused, info = model.solve(krylov_restart=4, max_cycles=40)
    assert info['converged'] and len(calls) > 0
    assert info['preconditioner_applications'] == len(calls)
    assert info['multigrid_cycles'] == len(calls)
    assert info['multigrid_cycles'] > info['gmres_restart_blocks']


def test_independent_terrain_gate_honours_invalid_physical_aggregates():
    from DARCY_WARP_PACKAGE.validation.terrain_cpu import accepted, CPUControls
    budget = dict(head_residual_rms=0., cell_residual_rms=0., relative_cell_imbalance=0.,
                  throughput=1., physical_metrics_valid=False)
    assert not accepted(head=np.zeros((1, 1, 1)), budget=budget, controls=CPUControls())


def test_public_timing_includes_independent_acceptance():
    with model_from(data=inputs(), formulation='mf6_6_6_2_npf') as model:
        unused, info = model.solve(npf_krylov_method='gmres')
    assert info['public_total_seconds'] >= info['seconds'] >= 0.
    assert info['setup_seconds'] >= 0. and info['reference_setup_seconds'] >= 0.
    assert 'independent final acceptance' in info['public_timing_scope']


def test_independent_face_model_rejects_implicit_physical_mode_change():
    data = inputs()
    with model_from(data=data) as model:
        faces = tuple(value.copy() for value in model._session.faces)
        fields = {name: data[name] for name in ('active', 'bc_mask', 'bc_values', 'rhs', 'initial_head')}
        model.build_from_face_conductance(**dict(zip(('tx_p','tx_m','ty_p','ty_m','tz_p','tz_m'), faces)), **fields)
        with pytest.raises(ValueError, match='build_from_K_fields'):
            model.solve(unconfined=True, kx_field=data['kx_field'], ky_field=data['ky_field'], kz_field=data['kz_field'])
        unused, info = model.solve()
        assert info['converged']
