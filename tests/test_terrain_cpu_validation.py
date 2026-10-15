"""Independent CPU equations, serialization, derivatives and adapters."""
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,mf6_cell,internal_cell
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference,CPUControls,solve_cpu
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D


def small_case(*, slope=.02, contact=False, override=None):
    y,x=np.mgrid[:3,:5]
    top=10.+slope*(x+2*y)
    z=np.stack([top,top-(1.+.02*x),top-(3.+.03*y)])
    grid=TerrainGrid3D(z_interfaces=z,dx=2.,dy=3.)
    shape=grid.shape
    fixed=np.zeros(shape,dtype=int)
    fixed[:,:,-1]=1
    h=grid.bottoms+.4*grid.thickness
    values=h.copy()
    source=np.full(shape,.001)
    source[fixed!=0]=0.
    contacts=()
    if contact:
        contacts=(TerrainContact3D(cell=(0,1,0),direction='west',area=.3,distance=1.,
            elevation=float(grid.elevations[0,1,0]),head=6.,kind='discharge_only',conductance=override),)
    k=np.ones(shape)
    k[-1]=.001
    return TerrainCase(case_id='small',arrays=dict(interfaces=z,active=np.ones((3,5),dtype=bool),fixed=fixed,
        values=values,initial=h,source=source,kx=k,ky=.6*k,kz=.15*k),
        physical=dict(dx=2.,dy=3.,floor=.01,steady=True,storage=0),controls={},contacts=contacts)


@pytest.mark.parametrize('regime',['full','thin','floor','mixed'])
@pytest.mark.parametrize('override',[None,.7])
def test_independent_sparse_jacobian(regime,override):
    case=small_case(contact=True,override=override)
    ref=CPUReference(case=case)
    g=ref.grid
    h=g.bottoms+.25*g.thickness
    if regime=='full':h=g.tops+2.
    if regime=='thin':h=g.bottoms+.03
    if regime=='floor':h=g.bottoms-1.
    if regime=='mixed':h.ravel()[::3]=g.bottoms.ravel()[::3]-1.
    rng=np.random.default_rng(seed=32)
    v=rng.normal(size=h.shape)
    v[~ref.free]=0.
    op,r,b=ref.evaluate(head=h)
    J=ref.jacobian(operator=op,head=h)
    jv=(J@v.ravel()).reshape(h.shape)
    errors=[]
    for step in [1e-3,1e-4,1e-5,1e-6]:
        unused,rp,unused=ref.evaluate(head=h+step*v)
        unused,rm,unused=ref.evaluate(head=h-step*v)
        fd=-(rp-rm)/(2*step)
        errors.append(np.linalg.norm((jv-fd)[ref.free])/np.linalg.norm(jv[ref.free]))
    assert min(errors)<2e-7,errors
    assert J.nnz<50*h.size


@pytest.mark.parametrize('threshold',['floor','top'])
def test_cpu_clamp_one_sided_selected_derivative(threshold):
    case=small_case(slope=0.)
    case.physical['floor']=.125
    ref=CPUReference(case=case)
    # Force exactly representable clamp state on uniform interfaces.
    case.arrays['interfaces'][:]=np.array([10.,9.,7.])[:,None,None]
    ref=CPUReference(case=case)
    h=ref.grid.bottoms+.125 if threshold=='floor' else ref.grid.tops.copy()
    v=np.full(h.shape,-1. if threshold=='floor' else 1.)
    v[~ref.free]=0.
    op,r,b=ref.evaluate(head=h)
    jv=(ref.jacobian(operator=op,head=h)@v.ravel()).reshape(h.shape)
    unused,rp,unused=ref.evaluate(head=h+1e-6*v)
    np.testing.assert_allclose(jv[ref.free],(-(rp-r)/1e-6)[ref.free],rtol=2e-7,atol=2e-8)


def test_case_hash_and_roundtrip(tmp_path):
    case=small_case(contact=True)
    directory=tmp_path/'case'
    case.save(directory=directory)
    loaded=TerrainCase.load(directory=directory)
    assert loaded.digest()==case.digest()
    with np.load(directory/'inputs.npz') as data:
        arrays={k:data[k] for k in data.files}
    arrays['source'][0,0,0]+=.01
    np.savez_compressed(directory/'inputs.npz',**arrays)
    with pytest.raises(ValueError,match='hash mismatch'):
        TerrainCase.load(directory=directory)


def test_index_roundtrip_and_mf6_one_based_number():
    shape=(4,3,5)
    for k,j,i in np.ndindex(shape):
        mapped=mf6_cell(cell=(k,j,i),shape=shape)
        assert internal_cell(cell=mapped,shape=shape)==(k,j,i)
        assert np.ravel_multi_index(mapped,shape)+1==k*15+(2-j)*5+i+1


@pytest.mark.parametrize('method',['picard','newton'])
def test_complete_cpu_cold_solve_and_reset(method):
    case=small_case(slope=0.)
    controls=CPUControls(method=method,max_outer=300)
    before=case.arrays['initial'].copy()
    h,info=solve_cpu(case=case,controls=controls)
    assert info['converged'],info
    second,other=solve_cpu(case=case,controls=controls)
    np.testing.assert_array_equal(case.arrays['initial'],before)
    np.testing.assert_array_equal(h,second)
    assert other['outers']==info['outers']
    assert max(t.get('backward_error',0.) for row in info['history'] for t in row['trials'])<1e-12


def test_cpu_solver_does_not_import_warp(tmp_path):
    case=small_case(slope=0.)
    case.save(directory=tmp_path/'case')
    code='''import sys
class RejectWarp:
    def find_spec(self,fullname,path=None,target=None):
        if fullname=="warp" or fullname.startswith("warp."):
            raise RuntimeError("CPU reference imported Warp")
sys.meta_path.insert(0,RejectWarp())
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import solve_cpu,CPUControls
case=TerrainCase.load(directory=Path(sys.argv[1]))
head,result=solve_cpu(case=case,controls=CPUControls(max_outer=300))
assert result["converged"]
assert "warp" not in sys.modules
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path/'case')],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_cpu_honest_outer_budget():
    case=small_case()
    head,info=solve_cpu(case=case,controls=CPUControls(max_outer=0))
    assert not info['converged']
    assert info['stop_reason']=='outer_budget'


@pytest.mark.parametrize('net',[0.,-.01])
def test_cpu_steady_component_datum_and_compatibility(net):
    case=small_case(slope=0.,contact=True,override=1.)
    case.arrays['fixed'][:]=0
    case.arrays['source'][:]=net
    with pytest.raises(ValueError,match='datum|negative source'):
        solve_cpu(case=case,controls=CPUControls(max_outer=10))


def test_restart_registry_refuses_stale_and_partial(tmp_path):
    from DARCY_WARP_PACKAGE.validation.records import start_record,check_cached,finish_record
    from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
    case=small_case()
    directory=tmp_path/'run'
    assert not check_cached(directory=directory,case=case,kind='cpu',settings={})
    directory.mkdir()
    with pytest.raises(ValueError,match='incomplete'):
        check_cached(directory=directory,case=case,kind='cpu',settings={})
    start_record(directory=directory,case=case,kind='cpu',settings={})
    atomic_json(path=directory/'summary.json',value=dict(input_hash=case.digest()))
    np.savez_compressed(directory/'head.npz',head=case.arrays['initial'])
    finish_record(directory=directory)
    assert check_cached(directory=directory,case=case,kind='cpu',settings={})
    with pytest.raises(ValueError,match='stale'):
        check_cached(directory=directory,case=case,kind='cpu',settings=dict(max_outer=9))
    (directory/'head.npz').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt'):
        check_cached(directory=directory,case=case,kind='cpu',settings={})


@pytest.mark.parametrize('branch',['closed','threshold'])
def test_cpu_discharge_closed_active_set_one_sided(branch):
    from dataclasses import replace
    case=small_case(slope=0.,contact=True)
    ref=CPUReference(case=case)
    h=ref.grid.bottoms+.25*ref.grid.thickness
    c=case.contacts[0]
    hb=float(h[c.cell])+(1. if branch=='closed' else 0.)
    case.contacts=(replace(c,head=hb),)
    ref=CPUReference(case=case)
    v=np.zeros(case.shape)
    v[c.cell]=-1.
    op,r,b=ref.evaluate(head=h)
    assert op.fluxes(head=h)[3][0]==0.
    jv=(ref.jacobian(operator=op,head=h)@v.ravel()).reshape(h.shape)
    unused,rp,unused=ref.evaluate(head=h+1e-6*v)
    np.testing.assert_allclose(jv[ref.free],(-(rp-r)/1e-6)[ref.free],rtol=2e-6,atol=1e-8)


def test_cpu_acceptance_requires_finite_physical_state():
    from DARCY_WARP_PACKAGE.validation.terrain_cpu import accepted
    budget=dict(head_residual_rms=0.,cell_residual_rms=0.,relative_cell_imbalance=0.,throughput=1.,finite_physical_state=False)
    assert not accepted(head=np.zeros((1,1,1)),budget=budget,controls=CPUControls())
    budget['finite_physical_state']=True
    budget['throughput']=np.inf
    assert not accepted(head=np.zeros((1,1,1)),budget=budget,controls=CPUControls())


def test_compact_reduced_case_retains_distinct_independent_roots():
    directory=Path(__file__).parent/'fixtures'/'terrain_multiple_roots'
    case=TerrainCase.load(directory=directory)
    seeds=np.load(directory/'seeds.npz')
    rng=np.random.default_rng(seed=1966)
    perturbation=rng.normal(size=case.shape)*1e-4
    perturbation[case.arrays['fixed']!=0]=0.
    controls=CPUControls(max_outer=30,head_tol=1e-10,flow_tol=1e-12,relative_tol=1e-11,fallback=False,recovery_growth=1.)
    roots=[]
    for label in ('first','second'):
        head,info=solve_cpu(case=case,controls=controls,initial=seeds[label]+perturbation)
        assert info['converged'],info
        assert info['outers']>0
        assert info['budget']['relative_cell_imbalance']<=1e-11
        roots.append(head)
    assert np.max(np.abs(roots[0]-roots[1]))>.3
    assert np.sum(case.arrays['fixed']==0)==196


def test_harmonic_thickness_is_a_converging_face_area_approximation():
    from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
    from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainMetrics3D
    errors=[]
    for nx in (8,16):
        dx=20./nx
        x=(np.arange(nx)+.5)*dx
        top=np.full((2,nx),10.)
        middle=top-(2.+.2*x[None,:]/20.)
        bottom=np.full((2,nx),5.)
        grid=TerrainGrid3D(z_interfaces=np.stack([top,middle,bottom]),dx=dx,dy=2.)
        op=TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=grid),kx=1.,ky=1.,kz=1.)
        head=np.broadcast_to(200.+.03*x[None,None,:],grid.shape)
        qx=op.fluxes(head=head)[0]
        left,right=grid.thickness[:,:,:-1],grid.thickness[:,:,1:]
        harmonic=2*left*right/(left+right)
        physical_midpoint=.5*(left+right)
        np.testing.assert_allclose(qx,-.03*grid.dy*harmonic,rtol=1e-11,atol=1e-12)
        middle_face=nx//2-1
        error=abs(qx[0,0,middle_face]/grid.dy+.03*physical_midpoint[0,0,middle_face])
        assert error>0.
        errors.append(error)
    assert errors[0]/errors[1]==pytest.approx(4.,rel=1e-6)


def test_scalar_control_has_two_smooth_wet_roots_and_a_floor_root():
    shape=(2,1,2)
    z=np.broadcast_to(np.array([4.,0.,-4.])[:,None,None],(3,1,2)).copy()
    fixed=np.ones(shape,dtype=int);fixed[0,0,0]=0
    values=np.full(shape,-1.);values[0,0,1]=2.
    source=np.zeros(shape);source[0,0,0]=.8
    case=TerrainCase(case_id='analytic_scalar',arrays=dict(interfaces=z,active=np.ones((1,2),dtype=bool),fixed=fixed,
        values=values,initial=values.copy(),source=source,kx=np.ones(shape),ky=np.full(shape,.6),kz=np.full(shape,4.)),
        physical=dict(dx=1.,dy=1.,floor=.01,steady=True,storage=0),controls={},contacts=())
    wet=np.sort(np.roots([5.,-5.8,.4]))
    c_floor=4*.01/(.01+2.)
    dry=(.8-1+2*c_floor)/(1+c_floor)
    ref=CPUReference(case=case)
    signs=[]
    for value in (dry,*wet):
        head=values.copy();head[0,0,0]=value
        op,r,b=ref.evaluate(head=head)
        assert abs(r[0,0,0])<1e-14
        signs.append(np.sign(ref.jacobian(operator=op,head=head)[0,0]))
        head[0,0,0]+=.001
        solved,info=solve_cpu(case=case,controls=CPUControls(max_outer=20,fallback=False),initial=head)
        assert info['converged'],info
        assert solved[0,0,0]==pytest.approx(value,abs=1e-9)
    assert np.all(wet>.01) and np.all(wet<4.)
    assert signs==[1.,-1.,1.]
