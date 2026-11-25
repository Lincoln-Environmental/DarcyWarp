"""Independent 2D conservation, model replacement, and heads-only regressions."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE import create_solver


NEWTON = 'unconfined_semismooth_newton_kcycle'
FAS = 'unconfined_fas'


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('backend', [NEWTON, FAS])
def test_recharged_strip_uses_si_volumetric_forcing(device, backend):
    grid = np.ones((4, 101))
    fixed = np.zeros_like(grid, dtype=int)
    fixed[:, [0, -1]] = 1
    with create_solver(dim=2, nx=101, ny=4, dx=10.0, device=device) as model:
        model.build_from_fields(T_field=0.001 * grid, R_field=1.e-9 * grid,
                                active=grid, bc_mask=fixed, bc_values=10.0 * grid)
        head, info = model.solve(formulation='unconfined', solver=backend, initial_head=10.0 * grid,
                                 K_field=2.e-4 * grid, zbot_field=0.0 * grid, ztop_field=5.0 * grid)
    # Fully saturated: T stays constant and the quadratic solution is exact on the grid.
    assert info['converged']
    assert head[2, 50] == pytest.approx(10.125, abs=0.001)
    outflow = 0.001 * np.sum(head[:, 1] - head[:, 0] + head[:, -2] - head[:, -1])
    assert outflow == pytest.approx(4 * 99 * 1.e-9 * 100., rel=0.001)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('backend', [NEWTON, FAS])
def test_local_storage_across_connected_aquitard(device, backend):
    # Four active cells in the centre column, the first between external CHDs.
    active = np.zeros((4, 3), dtype=int)
    active[0, :] = active[:, 1] = 1
    fixed = np.zeros_like(active)
    fixed[0, [0, 2]] = 1
    head = np.full(active.shape, 10.)
    head[0] = [11., 10., 9.]
    head[active == 0] = 0.
    day = 86400.
    conductivity = np.full_like(head, 100. / day)
    conductivity[1, 1] = 1.e-6 / day
    source = np.zeros_like(head)
    source[2:, 1] = 8.e-6 / (100. * day)
    with create_solver(dim=2, nx=3, ny=4, dx=10., device=device) as model:
        heads, info = model.solve_transient_2d_unconfined(
            solver=backend, initial_head=head, recharge_rates=[0., 0., 0.],
            source_fields_per_period=np.stack([source, source, source]), k_field=conductivity,
            zbot_field=np.full_like(head, -2.), ztop_field=np.full_like(head, 8.),
            sy=0., ss=1.e-5, dt=365. * day, active=active, bc_mask=fixed, bc_values=head * fixed,
        )
    # Frozen T throughout. Build the four free equations independently in days.
    matrix = np.diag([2000., 0., 0., 0.])
    neck = 2. / (1./1000. + 1./1.e-5)
    for i, conductance in enumerate([neck, neck, 1000.]):
        matrix[i, i] += conductance
        matrix[i + 1, i + 1] += conductance
        matrix[i, i + 1] -= conductance
        matrix[i + 1, i] -= conductance
    forcing = np.array([20000., 0., 8.e-6, 8.e-6])
    expected = np.full(4, 10.)
    for solved, period in zip(heads, info['period_infos'], strict=True):
        assert period['converged']
        for dt in period['experimental_substep_dts']:
            storage = 1.e-5 * 1000. / (dt / day)
            expected = np.linalg.solve(matrix + np.eye(4) * storage, forcing + storage * expected)
        np.testing.assert_allclose(solved[:, 1], expected, rtol=0, atol=1.e-4)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('package', ['drn', 'riv'])
def test_rejected_gated_package_preserves_usable_model(device, package):
    grid = np.ones((4, 4))
    zeros = np.zeros_like(grid)
    fields = dict(T_field=grid, R_field=zeros, active=grid, bc_mask=zeros, bc_values=zeros,
                  gh_mask=grid, gh_width=grid, gh_alpha=1., aq_thickness=1.)
    if package == 'drn':
        fields.update(drn_mask=grid, drn_elev=5.*grid, drn_cond=grid)
    else:
        fields.update(riv_mask=grid, riv_stage=5.*grid, riv_rbot=4.*grid, riv_cond=grid)
    parameters = dict(formulation='unconfined', solver=NEWTON, initial_head=7.5*grid,
                      K_field=grid, zbot_field=zeros, ztop_field=grid)
    with create_solver(dim=2, nx=4, ny=4, dx=1., device=device, use_ghb=True,
                       use_drn=package == 'drn', use_riv=package == 'riv') as model:
        model.build_from_fields(**fields, gh_head=10.*grid)
        original, info = model.solve(**parameters)
        assert info['converged']
        assert info['budget_summary'][f'{package}_out'] == pytest.approx(40.)
        fields[f'{package}_cond'] = -grid
        with pytest.raises(ValueError, match=f'{package}_cond'):
            model.build_from_fields(**fields, gh_head=20.*grid)
        repeated, info = model.solve(**parameters)
    assert info['converged']
    assert info['budget_summary'][f'{package}_out'] == pytest.approx(40.)
    np.testing.assert_allclose(repeated, original, rtol=0, atol=1.e-6)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
def test_heads_only_does_not_allocate_discarded_diagnostic_cubes(device, monkeypatch):
    grid = np.ones((4, 4))
    zeros = np.zeros_like(grid)
    original = np.zeros_like

    def tracked_allocation(prototype, *args, **kwargs):
        assert prototype.shape != (3, 4, 4), 'unrequested period history'
        return original(prototype, *args, **kwargs)

    monkeypatch.setattr(np, 'zeros_like', tracked_allocation)
    with create_solver(dim=2, nx=4, ny=4, dx=1., device=device) as model:
        heads = model.solve_transient_2d_unconfined(
            solver=NEWTON, initial_head=grid, recharge_rates=[0., 0., 0.], k_field=grid,
            zbot_field=zeros, ztop_field=10.*grid, sy=0.2, ss=0., dt=1., active=grid,
            bc_mask=zeros, bc_values=zeros, return_info=False,
        )
    assert heads.shape == (3, 4, 4)
    np.testing.assert_array_equal(heads, 1.)
