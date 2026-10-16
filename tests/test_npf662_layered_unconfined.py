"""Owning-thickness Newton law, active branches, live fluxes and resets."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.validation.npf662_reference import (
    NPF662LayeredCase, NPF662Reference, quadratic_saturation_derivative)
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace
from DARCY_WARP_PACKAGE.solvers.npf662_kernels import saturation_pair
from test_npf662_layers import layered_case
from validate_npf662_confined import make_gpu_model
from validate_npf662_layered import make_layered_dry_source
from test_npf662_newton import unconfined_case


def valid_unconfined_case(*,masked=False):
    """A distinct valid control; the confined fixture's CHD can lie below bottom."""
    case=layered_case(masked=masked)
    case.values=case.values+1.
    case.initial=case.values.copy()
    case.cell_type=1
    return case


@wp.kernel
def probe_saturation(head:wp.array(dtype=wp.float64), bottom:wp.array(dtype=wp.float64),
    thickness:wp.array(dtype=wp.float64), saturation:wp.array(dtype=wp.float64), derivative:wp.array(dtype=wp.float64)):
    i=wp.tid()
    pair=saturation_pair(head[i],bottom[i],thickness[i],1)
    saturation[i]=pair[0];derivative[i]=pair[1]


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('regime',['full','smooth','lower_buffer','upper_buffer','dry'])
def test_owning_thickness_derivatives_and_actual_support(device,regime):
    case=valid_unconfined_case(masked=True)
    rng=np.random.default_rng(seed=188)
    indices=np.indices(case.active.shape)
    case.cell_type=((indices[0]+indices[1]+indices[2])%3!=0).astype(int)
    fraction={'full':1.2,'smooth':.45,'lower_buffer':.4e-6,'upper_buffer':1.-.4e-6,'dry':-.2}[regime]
    jitter=.03 if regime=='smooth' else .03e-6 if 'buffer' in regime else .001
    head=case.bottoms+(fraction+jitter*rng.random(size=case.active.shape))*case.thicknesses
    head[case.fixed!=0]=case.values[case.fixed!=0]
    case.initial=head.copy()
    reference=NPF662Reference(case=case)
    v=rng.normal(size=head.shape);v[~reference.free]=0.
    J=reference.jacobian(head=head)
    row,col=J.nonzero()
    positions=np.stack(np.unravel_index(np.arange(head.size),head.shape),axis=1)
    assert np.all(np.abs(positions[row]-positions[col]).sum(axis=1)<=1)
    assert np.max(np.diff(J.indptr))<=7
    if regime=='smooth':assert (J-J.T).nnz>0
    expected=(J@v.ravel()).reshape(head.shape)
    errors=[]
    for epsilon in (1e-4,1e-5,1e-6,1e-7,1e-8,1e-9,1e-10):
        unused,plus=reference.evaluate(head=head+epsilon*v)
        unused,minus=reference.evaluate(head=head-epsilon*v)
        fd=-(plus['residual']-minus['residual'])/(2*epsilon)
        errors.append(float(np.linalg.norm((fd-expected)[reference.free])/np.linalg.norm(expected[reference.free])))
    assert min(errors)<2e-7,errors
    with make_gpu_model(case=case,device=device) as model:
        workspace=NPF662Workspace(model=model,case=case,restart=8,krylov_budget=100)
        x=wp.array(data=v.ravel(),dtype=wp.float64,device=device);z=wp.zeros_like(src=x)
        workspace.matvec(x=x,y=z,z=z,alpha=1.,beta=0.)
        np.testing.assert_allclose((z.numpy()*workspace.scale.numpy())[reference.free.ravel()],
            expected.ravel()[reference.free.ravel()],rtol=3e-12,atol=2e-11)
        unused,fields=reference.evaluate(head=head)
        np.testing.assert_allclose(workspace.residual.numpy().reshape(head.shape),fields['residual'],rtol=3e-12,atol=2e-11)
        np.testing.assert_array_equal(np.asarray(reference.incidence.sum(axis=0)),0.)
        workspace.close()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_owning_thickness_thresholds_on_device(device):
    thickness=np.repeat(np.array([.13,1.7,9.]),4)
    fraction=np.tile(np.array([0.,1e-6,1.-1e-6,1.]),3)
    bottom=np.full(thickness.shape,2.5)
    head=bottom+fraction*thickness
    expected,derivative=quadratic_saturation_derivative(head=head,bottom=bottom,thickness=thickness,cell_type=1)
    inputs=[wp.array(data=value,dtype=wp.float64,device=device) for value in (head,bottom,thickness)]
    sat=wp.zeros_like(src=inputs[0]);ds=wp.zeros_like(src=sat)
    wp.launch(kernel=probe_saturation,dim=head.size,inputs=[*inputs,sat,ds],device=device)
    np.testing.assert_allclose(sat.numpy(),expected,rtol=3e-13,atol=0.)
    np.testing.assert_allclose(ds.numpy(),derivative,rtol=3e-13,atol=0.)
    for step_fraction in (1e-8,1e-9,1e-10):
        step=step_fraction*thickness
        plus,unused=quadratic_saturation_derivative(head=head+step,bottom=bottom,thickness=thickness,cell_type=1)
        minus,unused=quadratic_saturation_derivative(head=head-step,bottom=bottom,thickness=thickness,cell_type=1)
        error=np.maximum(np.abs((plus-expected)/step-derivative),np.abs((expected-minus)/step-derivative))
    assert np.all(error*thickness<2e-4),error


@pytest.mark.parametrize('device',['cpu','cuda:0'])
@pytest.mark.parametrize('axis',[1,2])
def test_equal_head_different_convertible_saturation_selected_tie(device,axis,tmp_path):
    shape=(1,2,1) if axis==1 else (1,1,2)
    one=np.ones(shape)
    interfaces=np.stack([np.array([4.,7.]).reshape(shape)[0],np.array([0.,1.]).reshape(shape)[0]])
    head=2.*one
    case=NPF662LayeredCase(kx=one,ky=one,kz=one,active=one.astype(int),fixed=np.array([0,1]).reshape(shape),
        values=head.copy(),source=np.zeros(shape),initial=head.copy(),dx=1.,dy=1.,z_interfaces=interfaces,cell_type=1)
    reference=NPF662Reference(case=case)
    sign=1. if axis==1 else -1.
    v=np.zeros(shape);v.ravel()[0]=sign
    expected=(reference.jacobian(head=head)@v.ravel())[0]
    fraction=.5 if axis==1 else 1./6.
    saturation=(fraction-.5e-6)/(1.-1e-6)
    assert expected==pytest.approx(sign*4.8*saturation,rel=3e-13)
    with make_gpu_model(case=case,device=device) as model:
        workspace=NPF662Workspace(model=model,case=case,restart=4,krylov_budget=24)
        x=wp.array(data=v.ravel(),dtype=wp.float64,device=device);z=wp.zeros_like(src=x)
        workspace.matvec(x=x,y=z,z=z,alpha=1.,beta=0.)
        actual=z.numpy()[0]*workspace.scale.numpy()[0]
        np.testing.assert_allclose(actual,expected,rtol=3e-12,atol=2e-11)
        workspace.close()
    for epsilon in (1e-4,1e-6,1e-8):
        unused,trial=reference.evaluate(head=head+epsilon*v)
        fd=-trial['residual'].ravel()[0]/epsilon
    assert abs(fd-expected)<2e-7
    state=head+1e-6*v
    native,info,fields=run_reference_mf6(case=case,directory=tmp_path/'selected_branch',executable=Path('/bin/modflow/mf6'),prescribed_state=state)
    unused,expected_fields=reference.evaluate(head=state)
    assert info['native_success']
    np.testing.assert_allclose(fields['flux_native'],expected_fields['flux'],rtol=3e-12,atol=2e-11)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_layered_dry_source_safeguard_and_policy(device):
    case=make_layered_dry_source(initial=-.2)
    reference=NPF662Reference(case=case)
    cpu,ci,unused=reference.solve(max_outer=100)
    with make_gpu_model(case=case,device=device) as model:
        head,info=model.solve(unconfined=True,max_outer=100,max_cycles=2000)
        assert info['converged'] and ci['independent_accepted'],info
        assert info['full_saturated_safeguard_solves']>0
        assert all(row['krylov']<=2000 for row in info['outer_history'])
        np.testing.assert_allclose(head[reference.free],cpu[reference.free],rtol=0.,atol=1e-7)
        with pytest.raises(ValueError,match='fixed by V1'):model.solve(unconfined=True,abs_tol_min=1e-14)


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_layered_unconfined_numerical_reuse_geometry_rebuild_and_cold_reset(device):
    case=valid_unconfined_case(masked=True)
    with make_gpu_model(case=case,device=device) as model:
        first,fi=model.solve(unconfined=True,max_outer=100)
        assert fi['converged'],fi
        old=model._session._npf662_workspace_holder['workspace']
        initial=model._session.initial_head.copy()
        case.source*=1.1;case.values[case.fixed!=0]+=.01
        model.update_rhs(rhs=case.source);model.update_boundary_values(bc_values=case.values)
        head,second=model.solve(unconfined=True,max_outer=100)
        assert second['converged'] and second['workspace_reused']
        np.testing.assert_array_equal(model._session.initial_head,initial)
        with make_gpu_model(case=case,device=device) as fresh:
            expected,ei=fresh.solve(unconfined=True,max_outer=100)
        np.testing.assert_allclose(head,expected,rtol=0.,atol=1e-10)
        case.z_interfaces[-1]-=.4
        model.build_from_layers(z_interfaces=case.z_interfaces,kx_field=case.kx,ky_field=case.ky,kz_field=case.kz,
            active=case.active,bc_mask=case.fixed,bc_values=case.values,rhs=case.source,initial_head=case.initial)
        head,third=model.solve(unconfined=True,max_outer=100)
        cpu,ci,unused=NPF662Reference(case=case).solve(max_outer=100)
        assert old.closed and third['converged'] and not third['workspace_reused']
        np.testing.assert_allclose(head[case.active!=0],cpu[case.active!=0],rtol=0.,atol=1e-7)


def test_invalid_convertible_chd_and_invalid_supplied_state_are_not_repaired(tmp_path):
    case=valid_unconfined_case()
    case.values[0,0,0]=case.bottoms[0,0,0]-.1
    values=case.values.copy()
    with make_gpu_model(case=case,device='cpu') as model:
        with pytest.raises(ValueError,match='below cell bottom'):model.solve(unconfined=True)
    np.testing.assert_array_equal(case.values,values)
    case=valid_unconfined_case()
    with pytest.raises(ValueError,match='below a convertible bottom'):
        run_reference_mf6(case=case,directory=tmp_path/'invalid',executable=Path('/bin/modflow/mf6'),
            prescribed_state=case.bottoms-.1)
    assert not (tmp_path/'invalid').exists()


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_uniform_unconfined_explicit_interfaces_preserve_v1(device):
    uniform=unconfined_case(initial='mixed',heterogeneous=True)
    names=('kx','ky','kz','active','fixed','values','source','initial','dx','dy','cell_type')
    explicit=NPF662LayeredCase(**{name:getattr(uniform,name) for name in names},
        z_interfaces=np.concatenate([uniform.tops[:1],uniform.bottoms],axis=0))
    a,b=NPF662Reference(case=uniform),NPF662Reference(case=explicit)
    head=uniform.bottoms+.45*uniform.thicknesses
    head[uniform.fixed!=0]=uniform.values[uniform.fixed!=0]
    unused,af=a.evaluate(head=head);unused,bf=b.evaluate(head=head)
    np.testing.assert_allclose(a.c0,b.c0,rtol=3e-13,atol=0.)
    for name in ('saturation','conductance','dflux_left','dflux_right','flux','residual'):
        np.testing.assert_allclose(af[name],bf[name],rtol=3e-12,atol=2e-11)
    v=np.random.default_rng(seed=91).normal(size=head.size)
    np.testing.assert_allclose(a.jacobian(head=head)@v,b.jacobian(head=head)@v,rtol=3e-12,atol=2e-11)
    with make_gpu_model(case=uniform,device=device) as old,make_gpu_model(case=explicit,device=device) as new:
        hu,ui=old.solve(unconfined=True,top_elevation=uniform.top,max_outer=100)
        hl,li=new.solve(unconfined=True,max_outer=100)
        assert ui['converged'] and li['converged']
        np.testing.assert_allclose(hu,hl,rtol=0.,atol=1e-7)
        uc,unused=a.evaluate(head=hu);lc,unused=b.evaluate(head=hl)
        assert uc['independent_accepted'] and lc['independent_accepted']


def test_fully_prescribed_layered_domain_native_empty_free_subset(tmp_path):
    case=valid_unconfined_case(masked=True);case.fixed=case.active.copy()
    checks,unused=NPF662Reference(case=case).evaluate(head=case.values)
    assert checks['independent_accepted'] and checks['free_cells']==0
    assert checks['physical_rms']==0. and checks['gross_throughput']==0.
    native,info,fields=run_reference_mf6(case=case,directory=tmp_path/'all_prescribed',executable=Path('/bin/modflow/mf6'))
    assert info['native_success'] and info['independent_conservation']
    assert info['independent']['independent_accepted']
    np.testing.assert_array_equal(native[case.active!=0],case.values[case.active!=0])


def test_layered_cpu_nonlinear_reference_runs_with_warp_import_forbidden():
    code='''import builtins
original=builtins.__import__
def reject_warp(name,globals=None,locals=None,fromlist=(),level=0):
    if name.startswith('warp'):raise ImportError('Warp forbidden')
    return original(name,globals,locals,fromlist,level)

if __name__ == "__main__":
    builtins.__import__=reject_warp
    import numpy as np
    from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase,NPF662Reference
    shape=(1,1,2)
    one=np.ones(shape)
    case=NPF662LayeredCase(kx=one,ky=.6*one,kz=.15*one,active=one.astype(int),
        fixed=np.array([0,1]).reshape(shape),values=np.zeros(shape),source=np.array([1.,0.]).reshape(shape),
        initial=np.full(shape,-.2),dx=1.,dy=1.,z_interfaces=np.array([[[2.,4.]],[[0.,0.]]]),cell_type=1)
    head,checks,fields=NPF662Reference(case=case).solve(max_outer=100)
    assert checks['independent_accepted']
    assert abs(head[0,0,0]-.86602547)<1e-6
'''
    subprocess.run([sys.executable,'-c',code],check=True)
