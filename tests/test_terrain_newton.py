"""Discrete terrain Newton derivative, state safety and public API checks."""
import numpy as np
import pytest
import warp as wp

from test_terrain_device_picard import physical_driver
from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import TerrainJacobian3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D


@pytest.fixture(autouse=True, params=['cpu','cuda:0'])
def newton_test_device(request, monkeypatch):
    wp.init()
    backend=request.param
    if backend.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA is unavailable')

    def selected_device():
        return backend

    monkeypatch.setattr('test_terrain_device_picard.device',selected_device)
    monkeypatch.setattr('test_3d_terrain_gpu.device',selected_device)
    monkeypatch.setattr('test_3d_scientific_audit.device',selected_device)
    return backend


def reference_residual(*, driver, head):
    session = driver.session
    op = TerrainReferenceOperator3D(metrics=session.metrics, kx=session.kx, ky=session.ky,
        kz=session.kz, head=head, min_sat=driver.min_sat, contacts=session.contacts)
    residual = session.rhs-op.divergence(fluxes=op.fluxes(head=head))
    residual[driver.state.free.numpy().reshape(session.shape) == 0] = 0.
    return residual


def evaluate_jv(*, driver, head, direction):
    s = driver.state
    s.update_head(host_head=head)
    driver.transient = False
    driver.evaluate(head_buffer=s.head, rel_buffer=s.head_rel)
    driver.physical_residual(head=s.head_rel)
    j = TerrainJacobian3D(driver=driver)
    j.prepare()
    vector = wp.array(data=direction.ravel(), dtype=wp.float64, device=driver.device)
    out = wp.zeros_like(src=vector)
    arrays = [s.head, s.head_rel, s.coefficients, s.scale, s.rhs, s.residual, s.boundary,
        s.sub_open, s.sub_c, s.sub_bz, s.sub_bx, s.sub_by]
    for level in driver.workspace.levels:
        arrays.extend([*level['faces'], level['storage'], level['diag']])
    before = [array.numpy().copy() for array in arrays]
    j.matvec(x=vector, y=out, z=out, alpha=1., beta=0.)
    for array, expected in zip(arrays,before):
        np.testing.assert_array_equal(actual=array.numpy(),desired=expected)
    return out.numpy().reshape(head.shape), j


@pytest.mark.parametrize('regime', ['saturated','thin','floor','mixed'])
def test_analytic_jv_matches_independent_host_differences(physical_driver, regime):
    driver = physical_driver
    grid = driver.session.grid
    head = grid.bottoms+.2*grid.thickness
    if regime == 'saturated':
        head = grid.tops+2.
    if regime == 'thin':
        head = grid.bottoms+.03
    if regime == 'floor':
        head = grid.bottoms-1.
    if regime == 'mixed':
        head.ravel()[::3] = grid.bottoms.ravel()[::3]-1.
        head.ravel()[1::3] = grid.tops.ravel()[1::3]+1.
    fixed = driver.session.bc_mask != 0
    head[fixed] = driver.session.bc_values[fixed]
    rng = np.random.default_rng(seed=234)
    v = rng.normal(size=head.shape)
    v[fixed] = 0.
    jv, j = evaluate_jv(driver=driver,head=head,direction=v)
    expected = reference_residual(driver=driver,head=head)
    np.testing.assert_allclose(actual=driver.state.residual.numpy().reshape(head.shape),desired=expected,rtol=1e-11,atol=1e-11)
    errors=[]
    for eps in (1e-3,1e-4,1e-5,1e-6):
        fd = -(reference_residual(driver=driver,head=head+eps*v)-reference_residual(driver=driver,head=head-eps*v))/(2*eps)
        errors.append(np.linalg.norm(x=(jv-fd)[~fixed])/np.linalg.norm(x=jv[~fixed]))
    assert min(errors) < 2e-7, errors
    assert errors[1] < max(errors[0]*.05, 2e-8), errors
    # Face incidence conserves its derivative before exterior/fixed exchanges.
    qx=j.qx.numpy().reshape(head.shape)
    qy=j.qy.numpy().reshape(head.shape)
    qz=j.qz.numpy().reshape(head.shape)
    divergence=np.zeros_like(a=head)
    divergence[:,:,:-1] += qx[:,:,:-1]
    divergence[:,:,1:] -= qx[:,:,:-1]
    divergence[:,:-1,:] += qy[:,:-1,:]
    divergence[:,1:,:] -= qy[:,:-1,:]
    divergence[:-1] += qz[:-1]
    divergence[1:] -= qz[:-1]
    assert abs(np.sum(a=divergence)) < 1e-10


@pytest.mark.parametrize('threshold', ['floor','top'])
def test_clamp_threshold_constant_branch_one_sided(physical_driver,threshold):
    d=physical_driver
    grid=d.session.grid
    d.min_sat=.125
    head=grid.bottoms+(d.min_sat if threshold=='floor' else grid.thickness)
    fixed=d.session.bc_mask!=0
    head[fixed]=d.session.bc_values[fixed]
    v=np.ones_like(a=head)*(-1. if threshold=='floor' else 1.)
    v[fixed]=0.
    jv, unused=evaluate_jv(driver=d,head=head,direction=v)
    r=reference_residual(driver=d,head=head)
    fd=-(reference_residual(driver=d,head=head+1e-6*v)-r)/1e-6
    np.testing.assert_allclose(actual=jv[~fixed],desired=fd[~fixed],rtol=2e-6,atol=1e-6)


def reject_all(*, current, candidate):
    return False


def test_newton_rejected_trial_restores_all_accepted_fields(physical_driver, monkeypatch):
    d=physical_driver
    d.transient=False
    s=d.state
    d.evaluate(head_buffer=s.head,rel_buffer=s.head_rel)
    d.physical_residual(head=s.head_rel)
    current=d.residual_metrics(combined=d.dual_check(head_buffer=s.head_rel,candidate_buffer=s.head_rel))
    arrays=[s.head,s.head_rel,s.residual,s.rhs,s.coefficients,s.boundary,s.scale,s.sub_open,
        s.sub_c,s.sub_bx,s.sub_by,s.sub_bz,s.storage]
    for level in d.workspace.levels:
        arrays.extend([*level['faces'],level['storage'],level['diag']])
    snapshots=[a.numpy().copy() for a in arrays]
    s.delta.assign(src=np.linspace(start=.1,stop=1.,num=s.n))
    monkeypatch.setattr(d,'candidate_improves',reject_all)
    result=d.backtrack(current=current,initial_relaxation=1.)
    assert not result['accepted']
    assert len(result['trials'])==12
    for a,expected in zip(arrays,snapshots):
        np.testing.assert_allclose(actual=a.numpy(),desired=expected,rtol=2e-13,atol=2e-13)


def test_fixed_direction_projection_preserves_boundary_flux_derivatives(physical_driver):
    d=physical_driver
    head=d.state.head.numpy().reshape(d.session.shape)
    fixed=d.session.bc_mask!=0
    v=np.ones_like(a=head)
    projected=v.copy()
    projected[fixed]=0.
    a,unused=evaluate_jv(driver=d,head=head,direction=v)
    b,unused=evaluate_jv(driver=d,head=head,direction=projected)
    np.testing.assert_allclose(actual=a[~fixed],desired=b[~fixed],rtol=0.,atol=0.)
    # Identity constraint rows retain x; the solved RHS there is zero.
    np.testing.assert_allclose(actual=a[fixed],desired=1.,rtol=0.,atol=0.)
    assert np.linalg.norm(x=b[~fixed])>0


def make_newton(*, driver, budget):
    from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import DeviceTerrainNewton
    return DeviceTerrainNewton(session=driver.session,state=driver.state,workspace=driver.workspace,
        device=driver.device,controls=driver.controls,datum=driver.datum,restart=driver.restart,
        krylov_budget=budget,unconfined=True,transient=False,dt=None,abs_tol_min=driver.abs_tol_min,
        head_tolerance=driver.head_tolerance,budget_relative_tol=driver.budget_relative_tol,
        rel_tol=driver.rel_tol,min_sat=driver.min_sat,outer_limit=3,unconfined_relax=.7)


@pytest.mark.parametrize('budget',[0,1,13,25])
def test_shared_budget_and_failure_integrity(physical_driver,monkeypatch,budget):
    d=make_newton(driver=physical_driver,budget=budget)
    s=d.state
    initial=s.head.numpy().copy()
    work=[]

    def failing_block(*,budget,target_rms):
        work.append(budget)
        s.delta.fill_(value=.1)
        return budget, np.inf, target_rms

    monkeypatch.setattr(d,'krylov_block',failing_block)
    monkeypatch.setattr(d,'candidate_improves',reject_all)
    d.controls.nonlinear_growth_limit=0.
    result=d.run(max_outer=3)
    assert not result['converged']
    assert result['stop_reason'] in ('line_search_failed','linear_solve_failed')
    assert result['krylov_iterations']==sum(work)==budget
    assert result['outer_iterations']==1
    row=result['history'][0]
    assert row['newton_krylov_iterations']+row['picard_krylov_iterations']==sum(work)
    if budget>1:
        assert row['newton_krylov_iterations']>0
        assert row['picard_krylov_iterations']>0
    np.testing.assert_array_equal(actual=s.head.numpy(),desired=initial)
    expected=reference_residual(driver=d,head=initial.reshape(s.shape))
    np.testing.assert_allclose(actual=s.residual.numpy().reshape(s.shape),desired=expected,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('method',['picard','newton'])
def test_public_solver_session_handoff_and_acceptance(method):
    from test_3d_terrain_gpu import build_solver
    from test_3d_scientific_audit import device
    from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
    grid=TerrainGrid3D(surface_elevation=np.zeros(shape=(3,5)),zeta_faces=np.array(object=[0.,1.,3.]),dx=2.,dy=3.)
    fixed=np.zeros(shape=grid.shape,dtype=np.int32)
    fixed[:,:,0]=1
    initial=np.full(shape=grid.shape,fill_value=-.4)
    source=np.full(shape=grid.shape,fill_value=.001)
    with build_solver(grid=grid,kx=1.,ky=.6,kz=.15,fixed=fixed,values=initial,source=source,initial=initial) as solver:
        head,info=solver.solve(unconfined=True,terrain_nonlinear_method=method,device_picard=True,
            max_cycles=200,krylov_restart=8,unconfined_max_picard_iter=100,max_levels=3,min_coarse_n=2)
        assert info['converged'],info
        # Independent physical budget on the actual returned head.
        from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d
        session=solver._session
        op=TerrainReferenceOperator3D(metrics=session.metrics,kx=session.kx,ky=session.ky,kz=session.kz,head=head)
        budget=compute_terrain_budget_3d(operator=op,head=head,source=source,fixed=fixed,
            storage_change=np.zeros_like(a=head),storage_diagonal=np.zeros_like(a=head))
        assert budget['head_residual_rms']<=1e-6
        assert budget['cell_residual_rms']<=1e-12 or budget['relative_cell_imbalance']<=1e-8
        for key in ('head_residual_rms','cell_residual_rms','relative_cell_imbalance'):
            assert info['budget'][key]==pytest.approx(expected=budget[key],abs=1e-13,rel=1e-9)
        # Newton -> Picard and Picard -> Newton on the same normal session.
        other='picard' if method=='newton' else 'newton'
        second,second_info=solver.solve(unconfined=True,terrain_nonlinear_method=other,device_picard=True,
            initial_head=head,max_cycles=200,krylov_restart=8,unconfined_max_picard_iter=100,max_levels=3,min_coarse_n=2)
        assert second_info['converged']
        np.testing.assert_allclose(actual=second,desired=head,rtol=0.,atol=1e-8)


@pytest.mark.parametrize('branch',['open','closed','threshold'])
@pytest.mark.parametrize('override',[None,.7])
def test_discharge_gate_active_set(branch,override):
    from test_3d_terrain_gpu import build_solver
    from test_3d_scientific_audit import device
    from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
    from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
    from DARCY_WARP_PACKAGE.solvers.terrain_device_3d import DeviceTerrainState,DeviceTerrainPicard,DevicePicardControls
    from DARCY_WARP_PACKAGE.solvers.terrain_fast_3d import pack_device_subfaces,pack_device_contact_refresh
    grid=TerrainGrid3D(surface_elevation=np.full(shape=(1,3),fill_value=2.),zeta_faces=np.array(object=[0.,2.]),dx=1.,dy=1.)
    head=np.full(shape=grid.shape,fill_value=.5)
    hb=dict(open=.4,closed=.6,threshold=.5)[branch]
    contact=TerrainContact3D(cell=(0,0,0),direction='west',area=1.,distance=.5,elevation=1.,head=hb,
        kind='discharge_only',conductance=override)
    fixed=np.zeros(shape=grid.shape,dtype=np.int32)
    fixed[:,:,-1]=1
    source=np.zeros_like(a=head)
    with build_solver(grid=grid,kx=2.,ky=.6,kz=.15,fixed=fixed,values=head,source=source,
            initial=head,contacts=(contact,)) as solver:
        solver.solve(max_cycles=0)
        session=solver._session
        op=TerrainReferenceOperator3D(metrics=session.metrics,kx=session.kx,ky=session.ky,kz=session.kz,
            head=head,contacts=(contact,))
        subfaces,unused=pack_device_subfaces(operator=op,datum=2.)
        state=DeviceTerrainState(session=session,device=device(),workspace=session.workspace)
        state.upload_static(source=source,sy=source,ss=source,free=(fixed==0),subfaces=subfaces,
            contact_refresh=pack_device_contact_refresh(operator=op,datum=2.))
        state.update_head(host_head=head)
        d=DeviceTerrainPicard(session=session,state=state,workspace=session.workspace,device=device(),
            controls=DevicePicardControls(),datum=2.,restart=4,krylov_budget=40,unconfined=True,transient=False,
            dt=None,abs_tol_min=1e-12,head_tolerance=1e-6,budget_relative_tol=1e-8,rel_tol=1e-12,
            min_sat=.01,outer_limit=10,unconfined_relax=.7)
        v=np.array(object=[-1.,0.,0.]).reshape(head.shape)
        jv,j=evaluate_jv(driver=d,head=head,direction=v)
        assert state.sub_open.numpy()[0] == (branch=='open')
        eps=1e-6
        if branch=='threshold':
            fd=-(reference_residual(driver=d,head=head+eps*v)-reference_residual(driver=d,head=head))/eps
        else:
            fd=-(reference_residual(driver=d,head=head+eps*v)-reference_residual(driver=d,head=head-eps*v))/(2*eps)
        np.testing.assert_allclose(actual=jv,desired=fd,rtol=5e-6,atol=1e-6)
        if branch!='open':
            assert j.contact.numpy()[0]==0.
        state.close()


def accept_all(*,current,candidate):
    return True


def test_usable_newton_state_retained_when_picard_linear_fallback_is_unusable(physical_driver,monkeypatch):
    from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import TerrainNewtonControls
    d=make_newton(driver=physical_driver,budget=40)
    d.newton_controls=TerrainNewtonControls(adequate_merit_ratio=1e-10)
    s=d.state
    original=s.head.numpy().copy()
    free=s.free.numpy()!=0
    calls=[]

    def inner(*,initial_rms,target_rms):
        calls.append(target_rms)
        usable=len(calls)==1
        correction=np.zeros(shape=s.n)
        correction[free]=.001 if usable else 0.
        s.delta.assign(src=correction)
        return dict(cycles=4,restart_blocks=1,reached=usable,usable=usable,retried=False,
            stagnated=False,diverged=False,head_rms=target_rms if usable else initial_rms,target_rms=target_rms)

    monkeypatch.setattr(d,'run_inner',inner)
    monkeypatch.setattr(d,'candidate_improves',accept_all)
    result=d.run(max_outer=1)
    assert len(calls)==2
    assert result['history'][0]['direction']=='newton'
    expected=original.copy()
    expected[free]+=.001
    np.testing.assert_array_equal(actual=s.head.numpy(),desired=expected)
    residual=reference_residual(driver=d,head=expected.reshape(s.shape))
    np.testing.assert_allclose(actual=s.residual.numpy().reshape(s.shape),desired=residual,atol=1e-12,rtol=1e-12)
    d.close()


@pytest.mark.parametrize('value',[0.,1.,-1.,np.nan,np.inf])
def test_invalid_newton_controls_rejected(value):
    from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import TerrainNewtonControls
    with pytest.raises(ValueError,match='physical_forcing'):
        TerrainNewtonControls(physical_forcing=value).validate()
