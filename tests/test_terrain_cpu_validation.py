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
