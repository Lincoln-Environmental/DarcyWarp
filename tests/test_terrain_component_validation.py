"""Package-level regressions for component constraints, fallback and counts."""
import numpy as np
import pytest
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import DeviceTerrainNewton
from test_3d_terrain_gpu import build_solver


@pytest.mark.parametrize('initial',[1.,3.])
@pytest.mark.parametrize('source',[[.5,-.5],[-.5,-.5]])
def test_device_steady_discharge_component_requires_compatible_datum(initial,source):
    grid=TerrainGrid3D(surface_elevation=np.zeros((1,2)),zeta_faces=np.array([0.,1.]),dx=1.,dy=1.)
    contact=TerrainContact3D(cell=(0,0,0),direction='west',area=1.,distance=.5,elevation=-.5,
        head=2.,kind='discharge_only',conductance=3.)
    h=np.full(grid.shape,initial)
    with build_solver(grid=grid,kx=1.,ky=1.,kz=1.,fixed=np.zeros(grid.shape,dtype=int),values=h,
        source=np.array(source).reshape(grid.shape),initial=h,contacts=(contact,)) as solver:
        with pytest.raises(ValueError,match='datum|no valid steady'):
            solver.solve(unconfined=True,device_picard=True,max_cycles=0)


@pytest.mark.parametrize('initial',[1.,3.])
def test_device_transient_real_storage_constrains_discharge_component(initial):
    grid=TerrainGrid3D(surface_elevation=np.zeros((1,2)),zeta_faces=np.array([0.,1.]),dx=1.,dy=1.)
    contact=TerrainContact3D(cell=(0,0,0),direction='west',area=1.,distance=.5,elevation=-.5,
        head=2.,kind='discharge_only',conductance=3.)
    h=np.full(grid.shape,initial)
    with build_solver(grid=grid,kx=1.,ky=1.,kz=1.,fixed=np.zeros(grid.shape,dtype=int),values=h,
        source=np.full(grid.shape,-.01),initial=h,contacts=(contact,)) as solver:
        head,info=solver.solve(unconfined=True,device_picard=True,transient=True,dt=1.,ss=.1,head_prev=h,
            max_cycles=100,unconfined_max_picard_iter=200)
    assert info['converged'],info
    assert np.all(np.isfinite(head))




@pytest.mark.parametrize('initial',[1.,3.])
def test_device_positive_source_opens_or_retains_real_discharge_constraint(initial):
    grid=TerrainGrid3D(surface_elevation=np.zeros((1,2)),zeta_faces=np.array([0.,1.]),dx=1.,dy=1.)
    contact=TerrainContact3D(cell=(0,0,0),direction='west',area=1.,distance=.5,elevation=-.5,
        head=2.,kind='discharge_only',conductance=3.)
    h=np.full(grid.shape,initial)
    with build_solver(grid=grid,kx=1.,ky=1.,kz=1.,fixed=np.zeros(grid.shape,dtype=int),values=h,
        source=np.full(grid.shape,.5),initial=h,contacts=(contact,)) as solver:
        head,info=solver.solve(unconfined=True,device_picard=True,max_cycles=100,unconfined_max_picard_iter=200)
    assert info['converged'],info
    np.testing.assert_allclose(head.ravel(),[2+1/3,2+1/3+.5],rtol=0.,atol=2e-7)
