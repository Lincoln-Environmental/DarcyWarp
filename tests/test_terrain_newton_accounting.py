"""Package-level regressions for component constraints, fallback and counts."""
import numpy as np
import pytest
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D
from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import DeviceTerrainNewton
from test_3d_terrain_gpu import build_solver


def refuse_candidate(*,current,candidate):
    return False


def test_public_newton_rejected_rows_are_not_accepted_steps(monkeypatch):
    grid=TerrainGrid3D(surface_elevation=np.zeros((2,4)),zeta_faces=np.array([0.,1.,3.]),dx=1.,dy=1.)
    fixed=np.zeros(grid.shape,dtype=int)
    fixed[:,:,-1]=1
    h=np.full(grid.shape,-.4)
    with build_solver(grid=grid,kx=1.,ky=.6,kz=.15,fixed=fixed,values=h,source=np.full(grid.shape,.01),initial=h) as solver:
        head,info=solver.solve(unconfined=True,terrain_nonlinear_method='newton',max_cycles=0,unconfined_max_picard_iter=3)
    assert not info['converged']
    assert info['newton_accepted_steps']==0
    assert not any(row['accepted'] for row in info['outer_history'])


