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
    row,col = matrix.nonzero()
    allowed = {0,1,-1,5,-5,20,-20,19,-19,21,-21,15,-15,25,-25}
    assert set((col-row).tolist()).issubset(allowed)
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


@pytest.mark.parametrize("direction", ["west","east","south","north","top","bottom"])
def test_analytic_exterior_total_head_and_transformed_flux_sign(direction):
    surface = 20+.4*np.arange(3)[None,:]+.6*np.arange(3)[:,None]
    grid = TerrainGrid3D(surface_elevation=surface,zeta_faces=np.array([0,1,2]),dx=2,dy=3)
    metrics = TerrainMetrics3D(grid=grid)
    sides = {"west": ((0,1,0),2,-1),"east": ((0,1,2),2,1),
        "south": ((0,0,1),1,-1),"north": ((0,2,1),1,1),
        "top": ((0,1,1),0,-1),"bottom": ((1,1,1),0,1)}
    cell,axis,sign = sides[direction]
    a,b,c = .3,-.2,.7
    y,x = np.mgrid[:3,:3]
    h = 9+a*x[None]*2+b*y[None]*3+c*grid.elevations
    sx,sy = metrics.sx[cell[1:]],metrics.sy[cell[1:]]
    distance = (2 if axis == 2 else 3 if axis == 1 else 1)/2
    area = 3 if axis == 2 else 2 if axis == 1 else 6
    derivative = a+c*sx if axis == 2 else b+c*sy if axis == 1 else -c
    boundary_head = h[cell]+sign*distance*derivative
    elevation = grid.elevations[cell]+sign*distance*(sx if axis == 2 else sy if axis == 1 else -1)
    physical_area = area*np.sqrt(1+sx**2+sy**2) if axis == 0 else area
    contact = TerrainContact3D(cell=cell,direction=direction,area=physical_area,distance=distance,
                              elevation=elevation,head=boundary_head)
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=3,ky=5,kz=.02,contacts=(contact,))
    flux = operator.fluxes(head=h)[3][0]
    physical = -3*a if axis == 2 else -5*b if axis == 1 else (-3*a*sx-5*b*sy+.02*c)
    assert flux == pytest.approx(sign*physical*area,abs=2e-12)


def test_flat_operator_exact_cartesian_reduction():
    from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k,build_7point_face_conductance_unconfined_from_sat
    grid = TerrainGrid3D(surface_elevation=np.full((3,5),20.),zeta_faces=np.arange(4)*2.,dx=3,dy=4)
    metrics = TerrainMetrics3D(grid=grid)
    rng = np.random.default_rng(18)
    k = np.exp(rng.normal(size=grid.shape))
    for head in (None,grid.bottoms+rng.uniform(-1,3,grid.shape)):
        terrain = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=2*k,kz=.1*k,head=head)
        if head is None:
            faces = build_7point_face_conductance_from_k(kx_field=k,ky_field=2*k,kz_field=.1*k,
                active=grid.active,dx=grid.dx,dy=grid.dy,dz=2.)
        else:
            faces = build_7point_face_conductance_unconfined_from_sat(kx_field=k,ky_field=2*k,kz_field=.1*k,
                sat=np.clip(head-grid.bottoms,.01,2),active=grid.active,dx=grid.dx,dy=grid.dy,dz=2.)
        for expected,actual in zip(faces,terrain.principal_faces()):
            np.testing.assert_allclose(actual,expected,rtol=5e-16,atol=0)


def test_continuous_manufactured_source_converges_under_refinement():
    """Independent calculus reference, rather than source=discrete A*h."""
    errors = []
    for nx in (16,32,64):
        spacing = 1/nx
        x = (np.arange(nx)+.5)*spacing
        surface = 900*np.sin(x)[None,:]
        grid = TerrainGrid3D(surface_elevation=surface,zeta_faces=np.array([0,.4,1,2]),dx=spacing,dy=1)
        operator = TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=grid),kx=3,ky=2,kz=.01)
        h = np.broadcast_to(np.sin(2*x)[None,None,:],grid.shape).copy()
        discrete = operator.divergence(fluxes=operator.fluxes(head=h))
        # Integral -Kx*h_xx over each physical cell: the unit Jacobian
        # removes terrain relief from the volume measure exactly.
        exact = -3*(2*np.cos(2*(x+spacing/2))-2*np.cos(2*(x-spacing/2)))*grid.thickness
        errors.append(float(np.max(np.abs(discrete[1,:,1:-1]-exact[1,:,1:-1]))/np.max(np.abs(exact[1,:,1:-1]))))
    assert errors[0] < .002
    assert errors[1] < .3*errors[0]
    assert errors[2] < .3*errors[1]


def test_tiny_confined_elastic_increment_is_not_lost_in_yield_potential():
    from DARCY_WARP_PACKAGE.physics.storage_3d import storage_increment_3d
    old = np.array([10.])
    new = np.nextafter(old,np.inf)
    increment = storage_increment_3d(head=new,previous=old,bottoms=0.,thickness=10.,sy=.2,ss=1e-4)
    expected = 1e-4*10*(new-old)
    np.testing.assert_array_equal(increment,expected)
    # Subtracting the two full stored-water potentials would lose this change.
    assert (storage_potential_3d(head=new,bottoms=0,thickness=10,sy=.2,ss=1e-4)-
            storage_potential_3d(head=old,bottoms=0,thickness=10,sy=.2,ss=1e-4)).item() == 0
