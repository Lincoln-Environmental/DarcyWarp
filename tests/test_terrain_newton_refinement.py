"""Package-level regressions for component constraints, fallback and counts."""
import numpy as np
import pytest
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import DeviceTerrainNewton
from test_3d_terrain_gpu import build_solver


def refuse_candidate(self,*,current,candidate):
    return False


def test_real_picard_fallback_refines_remaining_budget(monkeypatch):
    grid=TerrainGrid3D(surface_elevation=np.zeros((3,6)),zeta_faces=np.array([0.,1.,3.]),dx=1.,dy=1.)
    fixed=np.zeros(grid.shape,dtype=int)
    fixed[:,:,-1]=1
    h=np.full(grid.shape,-.4)
    calls=[]
    original_init=DeviceTerrainNewton.__init__

    def configure(self,**kwargs):
        original_init(self,**kwargs)
        self.controls.nonlinear_growth_limit=0.

    monkeypatch.setattr(DeviceTerrainNewton,'__init__',configure)
    original=DeviceTerrainNewton.refine_inner

    def record_refinement(self,*,outcome,initial_rms,target_rms):
        calls.append(dict(used=outcome['cycles'],allocation=self.krylov_budget))
        return original(self,outcome=outcome,initial_rms=initial_rms,target_rms=target_rms)

    monkeypatch.setattr(DeviceTerrainNewton,'candidate_improves',refuse_candidate)
    monkeypatch.setattr(DeviceTerrainNewton,'refine_inner',record_refinement)
    with build_solver(grid=grid,kx=1.,ky=.6,kz=.15,fixed=fixed,values=h,source=np.full(grid.shape,.01),initial=h) as solver:
        head,info=solver.solve(unconfined=True,terrain_nonlinear_method='newton',max_cycles=200,
            krylov_restart=8,unconfined_max_picard_iter=3)
    assert not info['converged']
    assert calls,'usable Picard correction rejected nonlinearly must receive original remaining-budget refinement'
    assert info['krylov_iterations']<=200
    np.testing.assert_array_equal(head,h)
