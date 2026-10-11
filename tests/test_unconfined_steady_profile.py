"""Actual fast steady execution, gated physics and documented effective controls."""
import numpy as np
import pytest


def test_profile_is_opt_in_independent_and_selects_native_backend():
    from DARCY_WARP_PACKAGE.solvers.steady_config import production_unconfined_steady_controls
    from DARCY_WARP_PACKAGE.solvers.registry import select_backend
    from DARCY_WARP_PACKAGE.solvers.transient_config import default_solve_controls
    transient_before = default_solve_controls()
    controls = production_unconfined_steady_controls()
    backend = select_backend(solver=controls['solver'], formulation=controls['formulation'],
        transient=controls['transient'], default='kcycle')
    assert backend.name == 'unconfined_semismooth_newton_kcycle'
    assert controls['inner_implementation'] == 'fast'
    assert 'newton_fgmres_relative_tolerance' not in controls
    controls['nu_pre'] = 99
    assert production_unconfined_steady_controls()['nu_pre'] == 2
    assert default_solve_controls() == transient_before
    assert transient_before['max_levels'] == 4
    assert transient_before['adaptive_dt_enabled']
    with pytest.raises(ValueError):
        production_unconfined_steady_controls(max_levels=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('kind', ['drn', 'riv'])
def test_fast_steady_profile_preserves_gates_budget_and_true_residual(device, kind):
    import warp as wp
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    from DARCY_WARP_PACKAGE.solvers.steady_config import production_unconfined_steady_controls
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    from DARCY_WARP_PACKAGE.nonlinear.coastal import CoastalSubfaces2D
    shape = (40, 60)
    active = np.ones(shape, int)
    zeros = np.zeros(shape)
    top = np.ones(shape)
    recharge = np.full(shape, 0.3/365.25)
    mask = active.copy()
    mask[0, 0] = 0
    conductance = np.full(shape, 250.)
    coast = CoastalSubfaces2D(rows=np.array([0, 0]), columns=np.array([0, 0]),
        head_m=np.array([0., 1.]), conductance_m2_per_time=np.array([.1, .2]),
        submerged=np.array([1, 0]))
    fields = dict(T_field=np.full(shape, 50.), R_field=recharge, active=active,
        bc_mask=np.zeros(shape, int), bc_values=zeros, coastal_boundary=coast)
    if kind == 'drn':
        fields.update(drn_mask=mask, drn_elev=top, drn_cond=conductance)
    else:
        fields.update(riv_mask=mask, riv_stage=top, riv_rbot=zeros, riv_cond=conductance)
    model = WarpDarcySolver(nx=shape[1], ny=shape[0], dx=5., device=device,
        use_drn=kind == 'drn', use_riv=kind == 'riv', diag_preconditioner_backend='device')
    try:
        model.build_from_fields(**fields)
        controls = production_unconfined_steady_controls()
        head, info = model.solve(initial_head=top.copy(),
            reference_transmissivity_field=np.full(shape, 50.),
            zbot_field=np.full(shape, -50.), ztop_field=top, max_cycles=1, check_every_no=123, **controls)
        assert info['solver_backend'] == controls['solver']
        assert info['preconditioner_implementation'] == 'fast'
        assert info['converged'], info['newton_failure_reason']
        assert not info['newton_fallback_used']
        effective = info['effective_configuration']
        assert effective['precision'] == 'float64'
        assert effective['ignored_supplied_controls'] == ['check_every_no', 'max_cycles']
        assert not effective['mixed_precision']
        assert effective['actual_level_shapes'] == [(40, 60), (20, 30)]
        assert effective['fgmres_restart'] == 20
        assert effective['fgmres_max_iterations'] == 60
        assert effective['forcing']['eta_min'] == 1e-7
        assert effective['nonlinear_residual_rms_tolerance'] == 1e-6
        assert effective['head_equivalent_rms_tolerance'] == 1e-4
        assert effective['preconditioner_controls']['max_cycles'] == 2
        assert effective['preconditioner_controls']['nu_coarse'] == 30
        assert effective['startup']['relative_tolerance'] == 5e-7
        assert effective['startup']['max_active_set_iterations'] == 60
        assert effective['startup']['accepted']
        assert info['true_nonlinear_residual_rms'] <= 1e-6
        assert info['head_equivalent_residual_rms'] <= 1e-4
        independent_gate = conductance*mask*np.maximum(head-top, 0.) if kind == 'drn' else conductance*mask*(head-top)
        native_coastal = coast.fluxes(head=head)
        independent_residual = float(np.sum(recharge)*25.-np.sum(independent_gate)-np.sum(native_coastal))
        gross_input = float(np.sum(recharge)*25.+np.maximum(-native_coastal, 0.).sum()+np.maximum(-independent_gate, 0.).sum())
        assert abs(independent_residual) <= max(1e-6, 1e-6*gross_input)
        assert info['budget_summary'][kind+'_out'] == pytest.approx(float(np.maximum(independent_gate, 0.).sum()), abs=1e-7)
        assert info['budget_summary']['in_minus_out'] == pytest.approx(independent_residual, abs=1e-9)
        assert np.all(np.isfinite(head))
        assert np.all(head > -50.)
        histories = info['linear_history'] + info['confined_startup'].get('confined_history', [])
        for row in histories:
            assert row.get('converged', row.get('linear_converged', False))
            final = row.get('final_true_residual', row.get('linear_final_true_residual'))
            initial = row.get('initial_true_residual', row.get('linear_initial_true_residual'))
            assert final <= max(row['requested_absolute_tolerance'],
                row['requested_relative_tolerance']*initial)
        executed = info['kcycle_graph_execution']
        if device.startswith('cuda'):
            assert executed['graph_launches'] > 0
            assert executed['graph_launches'] == info['kcycle_preconditioner_cycles']
            assert executed['eager_cycles'] == 0
        else:
            assert executed['graph_launches'] == 0
            assert executed['eager_cycles'] == info['kcycle_preconditioner_cycles']
        assert info['source_identity']['sha'] is not None
    finally:
        model.close()


@pytest.mark.parametrize('drain_everywhere', [True, False])
def test_executable_example_exports_only_independently_accepted_heads(tmp_path, drain_everywhere):
    from working_tests.run_steady_production_profile import solve_and_export
    shape = (4, 6)
    top = np.ones(shape)
    mask = np.ones(shape, int) if drain_everywhere else np.zeros(shape, int)
    mask[:, -1] = 1
    input_archive = tmp_path/'inputs.npz'
    output_archive = tmp_path/'accepted.npz'
    np.savez(input_archive, active=np.ones(shape, bool), dem_m=top,
        darcy_bottom_m=np.full(shape, -50.), drainage_mask=mask,
        drain_conductance_m2_day=np.full(shape, 250.),
        reference_transmissivity_m2_day=np.full(shape, 50.),
        coastal_rows=np.empty(0, int), coastal_columns=np.empty(0, int),
        coastal_subface_head_m=np.empty(0), coastal_subface_conductance_m2_day=np.empty(0),
        coastal_subface_submerged=np.empty(0, int))
    options = dict(input_archive=input_archive, output_archive=output_archive,
        recharge_mm_year=300., days_per_year=365.25, cell_size_m=5., max_levels=5,
        device='cpu', budget_relative=1e-6, budget_absolute=1e-6, head_tolerance_m=1e-6)
    if drain_everywhere:
        metadata = solve_and_export(**options)
        assert abs(metadata['budget']['residual_m3_day']) < 1e-6
        with np.load(output_archive, allow_pickle=False) as archive:
            np.testing.assert_allclose(archive['head_m'], top+.3/365.25*25./250., atol=1e-9, rtol=0.)
    else:
        with pytest.raises(RuntimeError, match='Physical hot-start rejection'):
            solve_and_export(**options)
        assert not output_archive.exists()
