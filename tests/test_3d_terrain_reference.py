"""Analytical physical fluxes and independently assembled terrain equations."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D, storage_potential_3d
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D, TerrainContact3D


def terrain_case(*, irregular=False, shift=0):
    y,x = np.mgrid[:4,:5]
    surface = shift+25+1.3*x+0.7*y+0.15*x*y
    active = np.ones(surface.shape, dtype=bool)
    if irregular:
        active[0,0] = active[2,3] = False
    grid = TerrainGrid3D(surface_elevation=surface, zeta_faces=np.array([0,0.2,1.0,3.0]),
                          dx=2, dy=3, active_columns=active)
    return grid, TerrainMetrics3D(grid=grid)


@pytest.mark.parametrize("gradient", [(0,0,0),(0,0,1),(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,-1),(.3,-.2,.7)])
@pytest.mark.parametrize("irregular", [False,True])
def test_affine_physical_head_fluxes(gradient,irregular):
    grid,metrics = terrain_case(irregular=irregular)
    a,b,c = gradient
    y,x = np.mgrid[:4,:5]
    head = 9+a*x[None]*grid.dx+b*y[None]*grid.dy+c*grid.elevations
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=3,ky=5,kz=0.02)
    qx,qy,qz,boundary = operator.fluxes(head=head)
    validx = grid.active_columns[:,:-1]&grid.active_columns[:,1:]
    validy = grid.active_columns[:-1]&grid.active_columns[1:]
    np.testing.assert_allclose(qx,-3*a*grid.dy*grid.thickness*validx[None],atol=2e-12)
    np.testing.assert_allclose(qy,-5*b*grid.dx*grid.thickness*validy[None],atol=2e-12)
    expected = (-3*a*metrics.sx-5*b*metrics.sy+0.02*c)*grid.dx*grid.dy*grid.active_columns
    np.testing.assert_allclose(qz,np.broadcast_to(expected,qz.shape),atol=4e-12)
    assert abs(operator.divergence(fluxes=(qx,qy,qz,boundary)).sum()) < 5e-13


@pytest.mark.parametrize("irregular", [False,True])
def test_independent_sparse_matches_face_divergence(irregular):
    grid,metrics = terrain_case(irregular=irregular)
    rng = np.random.default_rng(47)
    k = np.exp(rng.uniform(-25,4,grid.shape))
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=2*k,kz=0.001*k)
    head = rng.normal(size=grid.shape)
    matrix,offset = operator.sparse_matrix()
    np.testing.assert_allclose((matrix @ head.ravel()).reshape(grid.shape)+offset,
        operator.divergence(fluxes=operator.fluxes(head=head)),rtol=1e-13,atol=1e-13)
    np.testing.assert_allclose(matrix @ np.ones(head.size),0,atol=1e-13)
    np.testing.assert_allclose(np.ones(head.size) @ matrix,0,atol=1e-13)


def test_datum_shift_and_unit_jacobian():
    grid,metrics = terrain_case()
    shifted,sm = terrain_case(shift=10000)
    h = grid.elevations+np.arange(5)[None,None,:]
    first = TerrainReferenceOperator3D(metrics=metrics,kx=3,ky=2,kz=0.1)
    second = TerrainReferenceOperator3D(metrics=sm,kx=3,ky=2,kz=0.1)
    for q1,q2 in zip(first.fluxes(head=h),second.fluxes(head=h+10000)):
        np.testing.assert_allclose(q1,q2,rtol=0,atol=3e-10)
    np.testing.assert_array_equal(grid.volumes,shifted.volumes)
    np.testing.assert_allclose(grid.volumes.sum(),grid.dx*grid.dy*20*3)
    with pytest.raises(ValueError):
        grid.zeta_faces[1] = 0.1


@pytest.mark.parametrize("interfaces", [np.array([0,1,1]),np.array([0,2,1]),np.array([1,2]),np.ones((4,5,3))])
def test_invalid_depth_geometry_fails(interfaces):
    with pytest.raises(ValueError,match="zeta_faces"):
        TerrainGrid3D(surface_elevation=np.zeros((4,5)),zeta_faces=interfaces,dx=1,dy=1)


def test_distinct_coastal_subfaces_and_discharge_sign():
    grid = TerrainGrid3D(surface_elevation=np.zeros((1,2)),zeta_faces=np.array([0,1,2]),dx=2,dy=2)
    metrics = TerrainMetrics3D(grid=grid)
    contacts = (TerrainContact3D(cell=(0,0,0),direction="west",area=1,distance=1,elevation=-.2,head=2,conductance=3),
        TerrainContact3D(cell=(0,0,0),direction="west",area=1,distance=1,elevation=-.8,head=4,kind="discharge_only",conductance=5))
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=1,ky=1,kz=1,contacts=contacts)
    for h,expected in ((1,[-3,0]),(3,[3,0]),(5,[9,5])):
        head = np.full(grid.shape,h)
        flux = operator.fluxes(head=head)
        np.testing.assert_array_equal(flux[3],expected)
        matrix,offset = operator.sparse_matrix(head_for_contacts=head)
        np.testing.assert_allclose(matrix @ head.ravel()+offset.ravel(),operator.divergence(fluxes=flux).ravel())
    assert operator.contacts[0].elevation != operator.contacts[1].elevation


def test_storage_uses_actual_nonuniform_layers_and_no_floor():
    grid,metrics = terrain_case()
    sy = np.broadcast_to(np.array([.1,.2,.3])[:,None,None],grid.shape)
    h = grid.bottoms-1
    np.testing.assert_array_equal(storage_potential_3d(head=h,bottoms=grid.bottoms,thickness=grid.thickness,sy=sy,ss=1e-4),0)
    wet = storage_potential_3d(head=grid.tops+.5,bottoms=grid.bottoms,thickness=grid.thickness,sy=sy,ss=1e-4)
    expected = sy*grid.thickness+1e-4*(.5*grid.thickness**2+.5*grid.thickness)
    np.testing.assert_array_equal(wet,expected)
