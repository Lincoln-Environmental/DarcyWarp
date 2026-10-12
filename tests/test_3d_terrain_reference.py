"""Analytical physical fluxes and independently assembled terrain equations."""
import numpy as np
import pytest
from scipy.sparse.linalg import spsolve

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
    face_thickness_x = .5*(grid.thickness[:,:,:-1]+grid.thickness[:,:,1:])
    face_thickness_y = .5*(grid.thickness[:,:-1]+grid.thickness[:,1:])
    np.testing.assert_allclose(qx,-3*a*grid.dy*face_thickness_x*validx[None],atol=2e-12)
    np.testing.assert_allclose(qy,-5*b*grid.dx*face_thickness_y*validy[None],atol=2e-12)
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
    # The metric-defect terms reach two layers through the one-sided
    # vertical gradient, so +-2*ncol (=+-40) is expected alongside +-ncol.
    allowed = {0,1,-1,5,-5,20,-20,40,-40,19,-19,21,-21,15,-15,25,-25}
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
    np.testing.assert_allclose(grid.volumes,shifted.volumes,rtol=0,atol=1e-9)
    # Unit Jacobian: volumes are dx*dy*depth-thickness exactly for the
    # legacy parallel form, independent of the DEM datum.
    np.testing.assert_allclose(grid.volumes,
        grid.dx*grid.dy*np.broadcast_to(np.diff(grid.zeta_faces)[:,None,None],grid.shape),rtol=0,atol=1e-12)
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


def test_vertical_flow_has_no_lateral_flux_at_distinct_coastal_elevations():
    grid,metrics = terrain_case()
    cell = (0,2,0)
    sx = metrics.sx[cell[1:]]
    face_centre = grid.elevations[cell]-sx*grid.dx/2
    contacts = tuple(TerrainContact3D(cell=cell,direction="west",area=.1,distance=grid.dx/2,
        elevation=face_centre-depth_offset,head=face_centre-depth_offset,conductance=1)
        for depth_offset in (-.03,.04))
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=1,ky=2,kz=.01,contacts=contacts)
    np.testing.assert_allclose(operator.fluxes(head=grid.elevations)[3],0,atol=2e-12)


def test_contact_geometry_discrepancies_are_rejected():
    grid,metrics = terrain_case()
    invalid = TerrainContact3D(cell=(0,2,2),direction="top",area=.1,distance=.1,
        elevation=grid.elevations[0,2,2],head=3)
    with pytest.raises(ValueError,match="elevation must match"):
        TerrainReferenceOperator3D(metrics=metrics,kx=1,ky=1,kz=1,contacts=(invalid,))


def variable_interface_case(*, shift=0, thin_middle=False):
    """Independently curving geological interfaces with no shared morphology."""
    y,x = np.mgrid[:4,:5]
    surface = shift+25+1.3*x+0.7*y+0.15*x*y
    topsoil = surface-0.3-0.1*np.sin(x)                     # 0.2..0.4 m topsoil
    aquifer = topsoil-5-2*np.cos(y)+0.2*x                   # aquifer base, own gradient
    basement = aquifer-(0.02 if thin_middle else 6)-0.001*x
    interfaces = np.stack((surface,topsoil,aquifer,basement))
    active = np.ones(surface.shape,dtype=bool)
    grid = TerrainGrid3D(z_interfaces=interfaces,dx=2,dy=3,active_columns=active)
    return grid,TerrainMetrics3D(grid=grid)


def test_variable_interface_construction_and_validation():
    grid,metrics = variable_interface_case()
    thickness = grid.thickness
    assert thickness.shape == grid.shape
    assert np.all(thickness > 0)
    np.testing.assert_allclose(grid.tops,grid.z_interfaces[:-1])
    np.testing.assert_allclose(grid.bottoms,grid.z_interfaces[1:])
    assert grid.surface_elevation is not grid.z_interfaces[0]
    # Independent morphologies: interface slopes differ from the DEM's.
    assert not np.allclose(metrics.zx_interfaces[2],metrics.zx_interfaces[0])
    assert not np.allclose(metrics.zxf[1],metrics.sxf)
    # Legacy construction still reproduces the same parallel interfaces.
    parallel = TerrainGrid3D(surface_elevation=grid.z_interfaces[0],
        zeta_faces=np.array([0.,.5,5.,12.]),dx=2,dy=3)
    assert parallel.zeta_faces is not None
    assert parallel.thickness.shape == parallel.shape


@pytest.mark.parametrize("interfaces", [
    np.stack((np.zeros((4,5)),np.ones((4,5)))),                       # inverted
    np.stack((np.zeros((4,5)),np.zeros((4,5)))),                      # zero thickness
    np.stack((np.zeros((4,5)),np.full((4,5),-1.),np.full((4,5),-1.+.5),np.full((4,5),-2.))),  # crossing
    np.stack((np.zeros((4,5)),np.full((4,5),np.nan))),
])
def test_invalid_variable_interfaces_are_rejected(interfaces):
    with pytest.raises(ValueError):
        TerrainGrid3D(z_interfaces=interfaces,dx=1,dy=1)


def test_variable_interface_affine_exactness():
    grid,metrics = variable_interface_case()
    thickness = grid.thickness
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=3,ky=5,kz=.02)
    face_tx = .5*(thickness[:,:,:-1]+thickness[:,:,1:])
    face_ty = .5*(thickness[:,:-1]+thickness[:,1:])
    y,x = np.mgrid[:4,:5]
    for flux in operator.fluxes(head=np.full(grid.shape,9.)):
        np.testing.assert_allclose(flux,0,atol=1e-12)
    # Pure physical vertical gradient: no spurious lateral flux on any of
    # the independently sloping interfaces, and the vertical flux is exactly
    # Kz per map area (the interface-consistent face scheme cancels the
    # metric terms of affine fields exactly).
    vertical = operator.fluxes(head=grid.elevations)
    np.testing.assert_allclose(vertical[0],0,atol=1e-11)
    np.testing.assert_allclose(vertical[1],0,atol=1e-11)
    np.testing.assert_allclose(vertical[2],.02*grid.dx*grid.dy,rtol=0,atol=1e-11)
    # Independent horizontal physical gradients act through the harmonic
    # series of the face transmissivities (exact variable-thickness result).
    for a,b,c in ((1.,0.,0),(0.,-1.,0),(.3,-.2,.7)):
        head = 9+a*x[None]*grid.dx+b*y[None]*grid.dy+c*grid.elevations
        qx,qy,qz,boundary = operator.fluxes(head=head)
        tx,ty = 3*thickness,5*thickness
        harmonic_x = 2*tx[:,:,:-1]*tx[:,:,1:]/(tx[:,:,:-1]+tx[:,:,1:])
        harmonic_y = 2*ty[:,:-1]*ty[:,1:]/(ty[:,:-1]+ty[:,1:])
        np.testing.assert_allclose(qx,-a*grid.dy*harmonic_x,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(qy,-b*grid.dx*harmonic_y,rtol=1e-12,atol=1e-12)
        expected_qz = (c*.02-3*a*metrics.zx_interfaces[1:-1]-5*b*metrics.zy_interfaces[1:-1])*grid.dx*grid.dy
        np.testing.assert_allclose(qz,expected_qz,rtol=0,atol=2e-11)
        assert abs(operator.divergence(fluxes=(qx,qy,qz,boundary)).sum()) < 5e-12


def test_variable_interface_sparse_matches_flux_divergence():
    grid,metrics = variable_interface_case()
    rng = np.random.default_rng(11)
    k = np.exp(rng.uniform(-20,3,grid.shape))
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=2*k,kz=.01*k)
    head = rng.normal(size=grid.shape)
    matrix,offset = operator.sparse_matrix()
    np.testing.assert_allclose((matrix @ head.ravel()).reshape(grid.shape)+offset,
        operator.divergence(fluxes=operator.fluxes(head=head)),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(matrix @ np.ones(head.size),0,atol=1e-12)


def test_thin_strictly_positive_layer_is_represented():
    grid,metrics = variable_interface_case(thin_middle=True)
    assert np.all(grid.thickness > 0)
    assert grid.thickness[2].min() < .05
    rng = np.random.default_rng(5)
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=1.,ky=1.,kz=1.)
    head = rng.normal(size=grid.shape)
    matrix,offset = operator.sparse_matrix()
    np.testing.assert_allclose((matrix @ head.ravel()).reshape(grid.shape)+offset,
        operator.divergence(fluxes=operator.fluxes(head=head)),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(matrix @ np.ones(head.size),0,atol=1e-12)


def test_nonparallel_contact_elevations_are_interface_exact():
    grid,metrics = variable_interface_case()
    cell = (2,2,4)
    # A coastal subface pinned to the true basement-interface elevation at
    # an offset point must validate against the reconstructed interface.
    elevation = float(grid.z_interfaces[3][2,4]+metrics.zx_interfaces[3][2,4]*grid.dx/2)
    contact = TerrainContact3D(cell=cell,direction="east",area=1.,distance=grid.dx/2,
        elevation=elevation,head=elevation)
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=1.,ky=1.,kz=1.,contacts=(contact,))
    np.testing.assert_allclose(operator.fluxes(head=grid.elevations)[3],0,atol=1e-12)
    with pytest.raises(ValueError,match="outside its layer"):
        TerrainReferenceOperator3D(metrics=metrics,kx=1.,ky=1.,kz=1.,contacts=(
            TerrainContact3D(cell=cell,direction="east",area=1.,distance=grid.dx/2,
                elevation=float(grid.z_interfaces[3][2,4]-1.),head=0.),))


def test_manufactured_solution_on_curved_interfaces_converges():
    """Continuous calculus source, independently curved interface family.

    Interfaces: z0 = sin-based rolling surface, z1 offset by a differing
    thickness function, z2 a differently sloping base. The source is derived
    from -div(K grad h) with h analytic, NOT from the discrete operator.
    """
    from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import storage_potential_3d  # noqa: F401  (import guard)
    rates = []
    for n in (16, 32, 64):
        spacing = 1.0/n
        x = (np.arange(n)+.5)*spacing
        y = (np.arange(n)+.5)*spacing
        X,Y = np.meshgrid(x,y,indexing="ij")
        surface = 40+8*np.sin(2*X)*np.cos(3*Y)
        mid = surface-(4+1.5*np.sin(X+2*Y))          # independent thickness + own curvature
        base = mid-(9-2*np.cos(2*X-Y))               # differently sloping bottom
        interfaces = np.stack((surface,mid,base))
        grid = TerrainGrid3D(z_interfaces=interfaces,dx=spacing,dy=spacing)
        metrics = TerrainMetrics3D(grid=grid)
        operator = TerrainReferenceOperator3D(metrics=metrics,kx=3.,ky=2.,kz=.5)
        h = (1+np.sin(3*X)*np.sin(2*Y))[None]+grid.elevations*.05
        # Continuous source: -Kx h_xx - Ky h_yy - Kz h_zz with h = f(x,y) + .05 z
        # and z measured through the exact linear-in-s layer mapping.
        hx = 3*np.cos(3*X)*np.sin(2*Y)
        hxx = -9*np.sin(3*X)*np.sin(2*Y)
        hyy = -4*np.sin(3*X)*np.sin(2*Y)
        # Volume-integrated continuous source; the .05*z part contributes
        # no second derivative and the interface metric cancels exactly.
        source = -(3*hxx+2*hyy)[None]*grid.volumes
        discrete = operator.divergence(fluxes=operator.fluxes(head=h))
        error = float(np.max(np.abs((discrete-source)[1,2:-2,2:-2])))
        rates.append(error)
    scale = float(np.max(np.abs(-(3*hxx+2*hyy)))*16*4*.5)
    assert rates[0]/scale < .05
    assert rates[1] < .3*rates[0]
    assert rates[2] < .3*rates[1]


def opposite_slope_case():
    """Aquifer base rising where the ground surface rises."""
    y,x = np.mgrid[:4,:5]
    surface = 30+2.*x+.5*y
    mid = surface-(6-1.2*x)          # thickens where terrain rises, then thins
    base = mid-(8-.8*x+.6*y)
    grid = TerrainGrid3D(z_interfaces=np.stack((surface,mid,base)),dx=2.,dy=3.)
    return grid,TerrainMetrics3D(grid=grid)


def test_opposite_sloping_interface_exactness_and_conservatism():
    grid,metrics = opposite_slope_case()
    assert np.all(grid.thickness > 0)
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=2.,ky=3.,kz=.05)
    rng = np.random.default_rng(64)
    head = rng.normal(size=grid.shape)
    matrix,offset = operator.sparse_matrix()
    np.testing.assert_allclose((matrix @ head.ravel()).reshape(grid.shape)+offset,
        operator.divergence(fluxes=operator.fluxes(head=head)),rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(matrix @ np.ones(head.size),0,atol=1e-12)
    # Affine fields remain exact on the opposite-sloping family.
    y,x = np.mgrid[:4,:5]
    for a,b,c in ((.7,0.,0),(.4,-.6,.9)):
        h = 9+a*x[None]*grid.dx+b*y[None]*grid.dy+c*grid.elevations
        qx,qy,qz,boundary = operator.fluxes(head=h)
        t_x,t_y = 2*grid.thickness,3*grid.thickness
        harmonic_x = 2*t_x[:,:,:-1]*t_x[:,:,1:]/(t_x[:,:,:-1]+t_x[:,:,1:])
        harmonic_y = 2*t_y[:,:-1]*t_y[:,1:]/(t_y[:,:-1]+t_y[:,1:])
        np.testing.assert_allclose(qx,-a*grid.dy*harmonic_x,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(qy,-b*grid.dx*harmonic_y,rtol=1e-12,atol=1e-12)
        expected_qz = (c*.05-2*a*metrics.zx_interfaces[1:-1]-3*b*metrics.zy_interfaces[1:-1])*grid.dx*grid.dy
        np.testing.assert_allclose(qz,expected_qz,rtol=0,atol=2e-11)


def test_variable_geometry_datum_shift_invariance():
    grid,metrics = variable_interface_case()
    shifted,sshift = variable_interface_case(shift=7500.)
    rng = np.random.default_rng(31)
    h = grid.elevations+rng.normal(scale=.5,size=grid.shape)
    first = TerrainReferenceOperator3D(metrics=metrics,kx=1.5,ky=2.5,kz=.2)
    second = TerrainReferenceOperator3D(metrics=sshift,kx=1.5,ky=2.5,kz=.2)
    for q1,q2 in zip(first.fluxes(head=h),second.fluxes(head=h+7500.)):
        np.testing.assert_allclose(q1,q2,rtol=0,atol=1e-8)


def test_contact_flux_continuity_across_contrasting_layers():
    """One face, one flux: contrasting K across a nonparallel contact."""
    grid,metrics = variable_interface_case()
    k = np.full(grid.shape,5.)
    k[1] = .02                                     # weathered aquitard layer
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=k,kz=k)
    rng = np.random.default_rng(17)
    head = rng.normal(size=grid.shape)
    fluxes = operator.fluxes(head=head)
    divergence = operator.divergence(fluxes=fluxes)
    row_sums = operator.sparse_matrix()[0] @ np.ones(head.size)
    np.testing.assert_allclose(row_sums,0,atol=1e-12)
    # The face between the aquifer and the aquitard contributes exactly
    # opposite amounts to its two cells: implied by conservation of the
    # whole face set, asserted here through the balanced divergence.
    assert abs(divergence.sum()) < 1e-12


def test_low_k_basement_series_resistance():
    """Vertical flow through aquifer over low-K basement: series harmonic."""
    n = 24
    surface = np.full((1,n),20.)
    mid = surface-4.
    base = mid-2.
    grid = TerrainGrid3D(z_interfaces=np.stack((surface,mid,base)),dx=1/n,dy=1.)
    metrics = TerrainMetrics3D(grid=grid)
    k = np.full(grid.shape,1.)
    k[1] = .001
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=k,kz=k)
    h_top = grid.elevations[0]+1.
    h_base = grid.elevations[1]-1.
    # Drive top and bottom layers with fixed heads via the sparse assembly.
    matrix,offset = operator.sparse_matrix()
    ids = np.arange(int(np.prod(grid.shape))).reshape(grid.shape)
    A = matrix.tolil(); b = offset.ravel().copy()
    ncol = grid.shape[1]*grid.shape[2]
    for layer,value in ((0,h_top),(1,h_base)):
        for index in ids[layer].ravel():
            A[index,:] = 0; A[index,index] = 1; b[index] = value.ravel()[index-layer*ncol]
    head = spsolve(A.tocsr(),b).reshape(grid.shape)
    flux_down = operator.fluxes(head=head)[2][0]        # aquifer->basement flux
    resistance = 4./(2*1.)+2./(2*.001)                  # half-layers in series per unit area
    expected = (h_top-h_base).mean()/resistance*grid.dx*grid.dy   # per-face volume
    interior = flux_down[:,2:-2]
    np.testing.assert_allclose(interior,expected,rtol=2e-3)
