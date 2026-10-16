"""Matched-law derivatives, active sets, state safety, budgets and live roots."""
from pathlib import Path
import numpy as np
import pytest
import warp as wp
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case,NPF662Reference,quadratic_saturation_derivative
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace,NPF662NewtonControls,search_direction
from DARCY_WARP_PACKAGE.validation.mf6_tpfa_reference import tpfa_fluxes
from validate_npf662_confined import make_gpu_model
from validate_terrain_scalar_roots import scalar_case
from validate_npf662_dry_start import make_dry_case


def unconfined_case(*,initial='mixed',heterogeneous=False):
    shape=(3,3,5)
    rng=np.random.default_rng(seed=1937)
    k=10.**rng.uniform(low=-2.,high=2.,size=shape) if heterogeneous else np.ones(shape)
    active=np.ones(shape,dtype=int);fixed=np.zeros(shape,dtype=int)
    fixed[:,:,0]=1;fixed[:,:,-1]=1
    values=np.broadcast_to(np.linspace(2.8,3.2,shape[2]),shape).copy()
    bottoms=np.broadcast_to(np.array([2.,0.,-2.])[:,None,None],shape)
    head=np.full(shape,5.) if initial=='wet' else bottoms-.2
    if initial=='mixed':head=np.where(np.indices(shape).sum(axis=0)%2,head,values)
    source=np.zeros(shape);source[0,1,2]=.004
    types=np.ones(shape,dtype=int);types[1]=0
    return NPF662Case(kx=k,ky=.6*k,kz=.15*k,active=active,fixed=fixed,values=values,
        source=source,initial=head,dx=2.,dy=3.,dz=2.,top=4.,cell_type=types)


def scalar_npf_case(*,initial):
    terrain=scalar_case(thickness=4.,high_head=2.,low_head=-1.,source=.8,kx=1.,kz=4.,floor=.01,initial=initial)
    a=terrain.arrays
    return NPF662Case(kx=a['kx'],ky=a['ky'],kz=a['kz'],active=terrain.grid().active.astype(int),
        fixed=a['fixed'],values=a['values'],source=a['source'],initial=a['initial'],dx=1.,dy=1.,dz=4.,top=4.,cell_type=1)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('regime',['full','thin','dry','smooth'])
def test_analytic_jv_and_actual_seven_point_support(device,regime):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('Local CUDA unavailable')
    case=unconfined_case(heterogeneous=True)
    rng=np.random.default_rng(seed=17)
    offsets={'full':3.,'thin':.0005,'dry':-.2,'smooth':.7}
    variation=.05 if regime=='smooth' else .0001 if regime=='thin' else 0.
    head=case.bottoms+offsets[regime]+variation*rng.random(size=case.active.shape)
    head[case.fixed!=0]=case.values[case.fixed!=0]
    case.initial=head.copy()
    reference=NPF662Reference(case=case)
    v=rng.normal(size=case.active.shape);v[~reference.free]=0.
    J=reference.jacobian(head=head)
    if regime=='smooth':assert (J-J.T).nnz>0
    expected=(J@v.ravel()).reshape(case.active.shape)
    row,column=J.nonzero()
    coordinates=np.stack(np.unravel_index(np.arange(case.active.size),case.active.shape),axis=1)
    assert np.all(np.sum(np.abs(coordinates[row]-coordinates[column]),axis=1)<=1)
    assert np.max(np.diff(J.indptr))<=7
    errors=[]
    for epsilon in (1e-3,1e-4,1e-5,1e-6,1e-7,1e-8):
        unused,plus=reference.evaluate(head=head+epsilon*v)
        unused,minus=reference.evaluate(head=head-epsilon*v)
        fd=-(plus['residual']-minus['residual'])/(2*epsilon)
        errors.append(float(np.linalg.norm((fd-expected)[reference.free])/np.linalg.norm(expected[reference.free])))
    assert min(errors)<2e-7,errors
    with make_gpu_model(case=case,device=device) as model:
        workspace=NPF662Workspace(model=model,case=case,restart=8,krylov_budget=100)
        x=wp.array(data=v.ravel(),dtype=wp.float64,device=device);z=wp.zeros_like(src=x)
        workspace.matvec(x=x,y=z,z=z,alpha=1.,beta=0.)
        actual=z.numpy()*workspace.scale.numpy()
        np.testing.assert_allclose(actual[reference.free.ravel()],expected.ravel()[reference.free.ravel()],rtol=3e-12,atol=2e-11)


@pytest.mark.parametrize('threshold',[0.,1e-6,1.-1e-6,1.])
def test_saturation_threshold_one_sided_derivatives(threshold):
    h=np.array([threshold]);bottom=np.array([0.]);types=np.array([1])
    value,derivative=quadratic_saturation_derivative(head=h,bottom=bottom,thickness=1.,cell_type=types)
    errors=[]
    for step in (1e-8,1e-9,1e-10):
        plus,unused=quadratic_saturation_derivative(head=h+step,bottom=bottom,thickness=1.,cell_type=types)
        minus,unused=quadratic_saturation_derivative(head=h-step,bottom=bottom,thickness=1.,cell_type=types)
        errors.append(max(abs(float((plus-value)[0]/step)-derivative[0]),abs(float((value-minus)[0]/step)-derivative[0])))
    assert errors[-1]<2e-4,errors
    assert errors[-1]<errors[0]*.1,errors


@pytest.mark.parametrize('axis',[1,2])
def test_upstream_tie_uses_mf6_user_node_branch(axis):
    shape=(1,2,1) if axis==1 else (1,1,2)
    fixed=np.array([0,1]).reshape(shape);types=np.array([1,0]).reshape(shape)
    one=np.ones(shape);head=2.*one
    case=NPF662Case(kx=one,ky=one,kz=one,active=one.astype(int),fixed=fixed,values=head,
        source=np.zeros(shape),initial=head.copy(),dx=1.,dy=1.,dz=4.,top=4.,cell_type=types)
    reference=NPF662Reference(case=case)
    C,dl,dr,sat=reference.face_terms(head=head)
    expected=2. if axis==1 else 4.
    assert C[0]==expected
    v=np.zeros(shape);v.ravel()[0]=1. if axis==1 else -1.
    derivative=(reference.jacobian(head=head)@v.ravel())[0]
    unused,at=reference.evaluate(head=head)
    for epsilon in (1e-5,1e-6):
        unused,trial=reference.evaluate(head=head+epsilon*v)
        fd=-(trial['residual'].ravel()[0]-at['residual'].ravel()[0])/epsilon
        assert abs(fd-derivative)<2e-5


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('initial',['wet','dry','mixed'])
def test_cold_cpu_gpu_and_native_mf6_solutions(tmp_path,device,initial):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('Local CUDA unavailable')
    case=unconfined_case(initial=initial,heterogeneous=True)
    reference=NPF662Reference(case=case)
    cpu,ci,fields=reference.solve(max_outer=200)
    assert ci['independent_accepted'],ci
    with make_gpu_model(case=case,device=device) as model:
        head,info=model.solve(unconfined=True,top_elevation=case.top,npf_cell_type=case.cell_types,max_outer=200)
        assert info['converged'],info
        np.testing.assert_allclose(head[reference.free],cpu[reference.free],rtol=0.,atol=1e-7)
        assert all(row['krylov']<=2000 for row in info['outer_history'])
    if not Path('/bin/modflow/mf6').is_file():pytest.skip('Pinned native MF6 unavailable')
    native,mi,mf=run_reference_mf6(case=case,directory=tmp_path/'mf6',executable=Path('/bin/modflow/mf6'))
    assert mi['native_success'] and mi['independent_conservation'],mi
    np.testing.assert_allclose(native[reference.free],cpu[reference.free],rtol=0.,atol=1e-7)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_scalar_reference_root_and_preserved_old_control(tmp_path,device):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('Local CUDA unavailable')
    case=scalar_npf_case(initial=-.2)
    reference=NPF662Reference(case=case)
    cpu,ci,fields=reference.solve()
    assert ci['independent_accepted']
    assert abs(cpu.ravel()[0]-3.8/3.)<1e-10
    with make_gpu_model(case=case,device=device) as model:
        gpu,gi=model.solve(unconfined=True,top_elevation=4.,max_outer=30)
        assert gi['converged'],gi
        assert abs(gpu.ravel()[0]-3.8/3.)<1e-10
    native,mi,mf=run_reference_mf6(case=case,directory=tmp_path/'mf6_scalar',executable=Path('/bin/modflow/mf6'))
    assert mi['native_success'] and mi['independent_conservation'],mi
    assert abs(native.ravel()[0]-3.8/3.)<1e-10


def test_rejected_trial_does_not_mutate_accepted_state():
    case=unconfined_case()
    case.initial=NPF662Reference(case=case).solve()[0]
    with make_gpu_model(case=case,device='cpu') as model:
        workspace=NPF662Workspace(model=model,case=case,restart=8,krylov_budget=100)
        metrics=workspace.evaluate(trial=False,linearize=True)
        names=('head','rhs','q','c','dl','dr','scale')
        before={name:getattr(workspace,name).numpy().copy() for name in names}
        workspace.delta.fill_(1e6)
        trial=search_direction(workspace=workspace,current=metrics,
            controls=NPF662NewtonControls(line_search_steps=1,max_head_change_m=1e6,recovery_growth=1.))
        assert not trial['accepted']
        for name in names:np.testing.assert_array_equal(getattr(workspace,name).numpy(),before[name])
        workspace.delta.fill_(wp.float64(np.inf))
        invalid=search_direction(workspace=workspace,current=metrics,controls=NPF662NewtonControls())
        assert not invalid['accepted']
        for name in names:np.testing.assert_array_equal(getattr(workspace,name).numpy(),before[name])


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_exact_dry_zero_row_safeguard_and_shared_budget(device):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('Local CUDA unavailable')
    case=make_dry_case(initial=-.2,source=1.)
    reference=NPF662Reference(case=case)
    cpu,ci,unused=reference.solve(max_outer=30)
    assert ci['independent_accepted'],ci
    a=1./(1.-1e-6)
    root=float(np.max(np.roots([a,2.*(1.-a),-1.])))
    assert abs(cpu.ravel()[0]-root)<1e-9
    with make_gpu_model(case=case,device=device) as model:
        head,info=model.solve(unconfined=True,top_elevation=4.,max_cycles=24,max_outer=30)
        assert info['converged'] and info['full_saturated_safeguard_solves']>0,info
        assert abs(head.ravel()[0]-root)<1e-9
        first=info['outer_history'][0]['attempts']
        assert first[0]['stopping']=='zero_row_nonzero_rhs' and first[0]['krylov']==0
        assert all(row['krylov']<=24 for row in info['outer_history'])
        failed,fi=model.solve(unconfined=True,top_elevation=4.,max_cycles=1,max_outer=1)
        assert not fi['converged']
        assert fi['krylov_iterations']<=1


def test_both_attempts_and_safeguard_share_five_krylov_iterations():
    case=make_dry_case(initial=.001,source=1.)
    with make_gpu_model(case=case,device='cpu') as model:
        head,info=model.solve(unconfined=True,top_elevation=4.,max_cycles=5,max_outer=1,
            npf_newton_controls=NPF662NewtonControls(line_search_steps=1,max_head_change_m=50.,recovery_growth=1.))
        assert not info['converged']
        row=info['outer_history'][0]
        assert model._session._npf662_workspace_holder['workspace'].mode.numpy()[0]==1
        assert len(row['attempts'])==3
        assert row['krylov']==5 and sum(a['krylov'] for a in row['attempts'])==5


def test_balanced_undetermined_dry_component_is_not_accepted():
    case=make_dry_case(initial=-.2,source=0.)
    result,unused=NPF662Reference(case=case).evaluate(head=case.initial)
    assert result['physical_rms']==0.
    assert result['ungrounded_current_components']==1
    assert not result['independent_accepted']
    with make_gpu_model(case=case,device='cpu') as model:
        head,info=model.solve(unconfined=True,top_elevation=4.,max_outer=2)
        assert not info['converged'] and info['stop_reason']=='ungrounded_saturation_components'


def test_shared_budget_exhaustion_and_cold_workspace_reset():
    case=scalar_npf_case(initial=-.2)
    with make_gpu_model(case=case,device='cpu') as model:
        head,info=model.solve(unconfined=True,top_elevation=4.,max_cycles=0,max_outer=3)
        assert not info['converged'] and info['stop_reason']=='krylov_budget' and info['krylov_iterations']==0
        a,first=model.solve(unconfined=True,top_elevation=4.,max_outer=30)
        b,second=model.solve(unconfined=True,top_elevation=4.,max_outer=30)
        assert first['converged'] and second['converged'] and second['workspace_reused']
        np.testing.assert_array_equal(a,b)
        assert first['krylov_iterations']==second['krylov_iterations']
        for invalid in ({'unconfined_min_sat':.01},{'variablecv':True},{'transient':True}):
            with pytest.raises(ValueError,match='Unsupported'):model.solve(unconfined=True,top_elevation=4.,**invalid)


def test_same_state_native_fluxes_and_old_tpfa_subset(tmp_path):
    case=scalar_npf_case(initial=.5)
    reference=NPF662Reference(case=case)
    head=case.values.copy();head.ravel()[0]=.75
    native,info,mf=run_reference_mf6(case=case,directory=tmp_path/'supplied',executable=Path('/bin/modflow/mf6'),prescribed_state=head)
    unused,expected=reference.evaluate(head=head)
    assert info['native_success']
    np.testing.assert_allclose(mf['flux_native'],expected['flux'],rtol=3e-12,atol=2e-11)
    np.testing.assert_allclose(mf['native_residual'],expected['residual'],rtol=3e-12,atol=2e-11)
    original=scalar_case(thickness=4.,high_head=2.,low_head=-1.,source=.8,kx=1.,kz=4.,floor=.01,initial=.5)
    qx,qy,qz=tpfa_fluxes(case=original,head=head)
    assert abs(qx.ravel()[0]-expected['flux'][0])<1e-12


@pytest.mark.parametrize('head_value',[2e-6,4.-2e-6])
def test_live_native_smoothing_buffers_have_no_terrain_floor(tmp_path,head_value):
    case=make_dry_case(initial=head_value,source=1.)
    head=case.initial.copy()
    native,info,fields=run_reference_mf6(case=case,directory=tmp_path/'buffer',executable=Path('/bin/modflow/mf6'),prescribed_state=head)
    unused,expected=NPF662Reference(case=case).evaluate(head=head)
    assert info['native_success'] and info['saturation_max_error']<1e-13
    np.testing.assert_allclose(fields['flux_native'],expected['flux'],rtol=3e-12,atol=0.)
    if head_value<1e-5:
        assert fields['native_saturation'].ravel()[0]<1e-6
        assert fields['flux_native'][0]<1e-10


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_newton_value_refresh_and_topology_reset(device):
    if device.startswith('cuda') and not wp.is_cuda_available():pytest.skip('Local CUDA unavailable')
    case=unconfined_case()
    with make_gpu_model(case=case,device=device) as model:
        a,first=model.solve(unconfined=True,top_elevation=case.top,npf_cell_type=case.cell_types,max_outer=50)
        workspace=model._session._npf662_workspace_holder['workspace']
        case.values+=.1
        model.update_boundary_values(bc_values=case.values)
        case.source*=2.
        model.update_rhs(rhs=case.source)
        b,second=model.solve(unconfined=True,top_elevation=case.top,npf_cell_type=case.cell_types,max_outer=50)
        assert first['converged'] and second['converged'] and second['workspace_reused']
        assert model._session._npf662_workspace_holder['workspace'] is workspace
        cpu,ci,unused=NPF662Reference(case=case).solve()
        np.testing.assert_allclose(b,cpu,rtol=0.,atol=1e-7)
        case.active[:,1,2]=0;case.source[:,1,2]=0.
        model.build_from_K_fields(kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,active=case.active,
            bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
        c,third=model.solve(unconfined=True,top_elevation=case.top,npf_cell_type=case.cell_types,max_outer=50)
        assert third['converged'] and not third['workspace_reused']
        assert model._session._npf662_workspace_holder['workspace'] is not workspace
        assert workspace.closed and workspace.krylov is None and workspace.head is None


def test_native_boundary_admissibility_is_enforced_without_repair(tmp_path):
    case=scalar_npf_case(initial=-.2)
    case.values[0,0,1]=-.1
    original=case.values.copy()
    with pytest.raises(ValueError,match='Convertible CHD head below cell bottom'):
        NPF662Reference(case=case)
    with make_gpu_model(case=case,device='cpu') as model:
        with pytest.raises(ValueError,match='Convertible CHD head below cell bottom'):
            model.solve(unconfined=True,top_elevation=4.)
    with pytest.raises(ValueError,match='Convertible CHD head below cell bottom'):
        run_reference_mf6(case=case,directory=tmp_path/'invalid',executable=Path('/bin/modflow/mf6'))
    np.testing.assert_array_equal(case.values,original)
    assert not (tmp_path/'invalid').exists()
    case=scalar_npf_case(initial=.5)
    case.top=np.zeros(case.active.shape)
    with pytest.raises(ValueError,match='flat scalar top'):NPF662Reference(case=case)
