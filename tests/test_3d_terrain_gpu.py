"""GPU face, matrix-vector, solution, storage and cache acceptance tests."""
import numpy as np
import pytest
import warp as wp
from scipy import sparse
from scipy.sparse.linalg import spsolve

from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D, storage_potential_3d
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D, TerrainContact3D
from DARCY_WARP_PACKAGE.solvers.terrain_fast_3d import TerrainGPUWorkspace3D, pack_terrain_boundaries
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from test_3d_scientific_audit import device
from test_3d_terrain_reference import terrain_case


@pytest.mark.parametrize("irregular", [False,True])
@pytest.mark.parametrize("heterogeneous", [False,True])
@pytest.mark.parametrize("unconfined", [False,True])
def test_gpu_full_matvec_and_faces_against_independent_sparse(irregular,heterogeneous,unconfined):
    grid,metrics = terrain_case(irregular=irregular)
    rng = np.random.default_rng(15)
    k = np.exp(rng.uniform(-28,3,grid.shape)) if heterogeneous else np.ones(grid.shape)
    reference = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=2*k,kz=.003*k,
        head=grid.bottoms+rng.uniform(-.1,1,grid.shape) if unconfined else None)
    fixed = np.zeros(grid.shape,dtype=np.int32)
    fixed[:,:,0] = grid.active[:,:,0]
    storage = .1*grid.volumes
    h = rng.normal(size=grid.shape)
    h[grid.active == 0] = 0
    boundary,unused = pack_terrain_boundaries(operator=reference,head=h,datum=0)
    workspace = TerrainGPUWorkspace3D(operator=reference,fixed=fixed,storage=storage,boundary=boundary,
        device=device(),max_levels=3,min_coarse_n=2,restart=12,maxiter=120,rel_tol=1e-10,abs_tol=1e-12)
    value = wp.array(h.ravel(),dtype=wp.float64,device=device())
    result = wp.zeros_like(value)
    workspace.matvec(value,result,result,alpha=1,beta=0)
    matrix,offset = reference.sparse_matrix()
    exact = (matrix @ h.ravel()).reshape(grid.shape)+storage*h
    exact[(fixed != 0)|(grid.active == 0)] = h[(fixed != 0)|(grid.active == 0)]
    np.testing.assert_allclose(result.numpy().reshape(grid.shape),exact,rtol=3e-13,atol=2e-13)
    for axis,devflux,expected in zip((2,1,0),(workspace.qx,workspace.qy,workspace.qz),reference.fluxes(head=h)[:3]):
        selection = [slice(None)]*3
        selection[axis] = slice(None,-1)
        np.testing.assert_allclose(devflux.numpy().reshape(grid.shape)[tuple(selection)],expected,rtol=3e-13,atol=2e-13)


def build_solver(*, grid, kx, ky, kz, fixed, values, source, initial, contacts=()):
    solver = WarpDarcySolver3D(nx=grid.shape[2],ny=grid.shape[1],nz=grid.shape[0],
        dx=grid.dx,dy=grid.dy,implementation="fast",device=device())
    solver.build_from_terrain(surface_elevation=grid.surface_elevation,zeta_faces=grid.zeta_faces,
        active_columns=grid.active_columns,kx_field=kx,ky_field=ky,kz_field=kz,
        bc_mask=fixed,bc_values=values,rhs=source,initial_head=initial,contacts=contacts)
    return solver


@pytest.mark.parametrize("irregular", [False,True])
@pytest.mark.parametrize("basement", [1,1e-14])
@pytest.mark.parametrize("anisotropy", [.05,1e-4])
def test_manufactured_solution_fine_equations_with_actual_hierarchy(irregular,basement,anisotropy):
    grid,metrics = terrain_case(irregular=irregular)
    y,x = np.mgrid[:4,:5]
    exact = 10+.03*x[None]+.05*y[None]+.1*grid.elevations
    k = np.broadcast_to((1+.2*x+.1*y)[None],grid.shape).copy()
    k[-1] *= basement
    reference = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=.6*k,kz=anisotropy*k)
    source = reference.divergence(fluxes=reference.fluxes(head=exact))
    fixed = np.zeros(grid.shape,dtype=np.int32)
    fixed[:,:,0] = grid.active[:,:,0]
    fixed[:,:,-1] = grid.active[:,:,-1]
    with build_solver(grid=grid,kx=k,ky=.6*k,kz=anisotropy*k,fixed=fixed,values=exact,source=source,initial=exact-.2) as solver:
        h,info = solver.solve(max_levels=4,max_cycles=600,abs_tol_min=1e-16,rel_tol=1e-13,dh_rms_tol=1e-9)
    free = (grid.active != 0)&(fixed == 0)
    assert info["converged"], info
    np.testing.assert_allclose(h[free],exact[free],rtol=0,atol=2e-7)
    residual = source-reference.divergence(fluxes=reference.fluxes(head=h))
    assert np.sqrt(np.mean((residual[free]/sum(reference.principal_faces())[free])**2)) < 1e-9
    assert info["budget"]["relative_cell_imbalance"] < 1e-8


def test_flat_nonuniform_confined_transient_mutable_storage_and_graph_reuse():
    grid = TerrainGrid3D(surface_elevation=np.zeros((3,5)),zeta_faces=np.array([0,.2,1,3]),dx=2,dy=3)
    metrics = TerrainMetrics3D(grid=grid)
    k = np.ones(grid.shape)
    k[-1] = 1e-14
    fixed = np.zeros(grid.shape,dtype=np.int32)
    fixed[:,:,0] = fixed[:,:,-1] = 1
    values = np.zeros(grid.shape)
    values[:,:,0],values[:,:,-1] = 4,2
    previous = np.full(grid.shape,3.0)
    source = np.full(grid.shape,.001)
    storage = np.full(grid.shape,.1)
    reference = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=k,kz=.1*k)
    matrix,offset = reference.sparse_matrix()
    with build_solver(grid=grid,kx=k,ky=k,kz=.1*k,fixed=fixed,values=values,source=source,initial=previous) as solver:
        workspace = None
        for step in range(3):
            if step == 2:
                storage[...] *= 2
            h,info = solver.solve(transient=True,storage_coeff=storage,dt=2,head_prev=previous,
                                 rel_tol=1e-12,dh_rms_tol=1e-9)
            diagonal = storage*grid.volumes/2
            assembled = (matrix+sparse.diags(diagonal.ravel())).tolil()
            rhs = (source+diagonal*previous).ravel()
            for index in np.flatnonzero(fixed.ravel()):
                assembled[index,:] = 0
                assembled[index,index] = 1
                rhs[index] = values.ravel()[index]
            independent = spsolve(assembled.tocsr(),rhs).reshape(grid.shape)
            np.testing.assert_allclose(h,independent,rtol=0,atol=2e-8)
            assert info["converged"]
            if workspace is not None:
                assert solver._session.workspace is workspace
                assert info["cuda_graph_reused"] == device().startswith("cuda")
            workspace = solver._session.workspace


def test_storage_only_column_crossings_and_rewetting_exact_volume():
    grid = TerrainGrid3D(surface_elevation=np.array([[20.]]),zeta_faces=np.array([0,10,20]),dx=10,dy=10)
    fixed = np.zeros(grid.shape,dtype=np.int32)
    previous = np.full(grid.shape,12.)
    sy = np.array([.2,.35]).reshape(grid.shape)
    source = np.array([0.,-215.]).reshape(grid.shape)
    with build_solver(grid=grid,kx=1,ky=1,kz=1,fixed=fixed,values=np.zeros(grid.shape),source=source,initial=previous) as solver:
        for forcing,expected in ((-215.,np.array([9.,5.]).reshape(grid.shape)),(235.,np.array([13.,19.]).reshape(grid.shape))):
            source[1] = forcing
            solver.update_rhs(rhs=source)
            solver.update_initial_head(initial_head=previous)
            head,info = solver.solve(unconfined=True,transient=True,sy=sy,dt=1,head_prev=previous,
                unconfined_head_tol=1e-8,final_head_residual_tol=1e-8,unconfined_max_picard_iter=180,
                abs_tol_min=1e-11,rel_tol=1e-12)
            assert info["converged"],info
            np.testing.assert_allclose(head,expected,rtol=0,atol=2e-7)
            change = 100*sy*(np.clip(head-grid.bottoms,0,grid.thickness)-np.clip(previous-grid.bottoms,0,grid.thickness))
            assert change.sum() == pytest.approx(forcing,abs=2e-6)
            previous = head


def test_steep_hydrostatic_zero_cycles_and_geometry_metric_refresh():
    grid,metrics = terrain_case()
    head = np.full(grid.shape,100.)
    fixed = np.zeros(grid.shape,dtype=np.int32)
    fixed[:,:,0] = 1
    with build_solver(grid=grid,kx=1,ky=1,kz=.01,fixed=fixed,values=head,source=np.zeros(grid.shape),initial=head) as solver:
        result,info = solver.solve(max_cycles=0)
        assert info["converged"] and info["krylov_iterations"] == 0
        workspace = solver._session.workspace
        solver.update_terrain_geometry(surface_elevation=grid.surface_elevation*2,zeta_faces=grid.zeta_faces)
        result,info = solver.solve(max_cycles=0)
        assert info["converged"]
        assert solver._session.workspace is workspace
        assert info["geometry_revision"] == 1


def test_coastal_gpu_subfaces_and_discharge_gate():
    grid = TerrainGrid3D(surface_elevation=np.zeros((1,2)),zeta_faces=np.array([0,1,2]),dx=2,dy=2)
    contacts = (TerrainContact3D(cell=(0,0,0),direction="west",area=1,distance=1,elevation=-.2,head=2,conductance=3),
        TerrainContact3D(cell=(0,0,0),direction="west",area=1,distance=1,elevation=-.8,head=4,kind="discharge_only",conductance=5))
    fixed = np.zeros(grid.shape,dtype=np.int32)
    with build_solver(grid=grid,kx=1,ky=1,kz=1,fixed=fixed,values=np.zeros(grid.shape),source=np.full(grid.shape,1.),initial=np.full(grid.shape,3.),contacts=contacts) as solver:
        h,info = solver.solve(unconfined_head_tol=1e-9,rel_tol=1e-12,final_budget_relative_tol=1e-11)
    assert info["converged"]
    assert sum(info["budget"]["coastal_subface_outflows"]) == pytest.approx(4,abs=1e-8)
    assert info["budget"]["coastal_subface_outflows"][1] == 0


@pytest.mark.parametrize("scale", [1,1e-14,1e-80])
def test_terrain_steady_scale_invariance(scale):
    grid,metrics = terrain_case()
    exact = grid.elevations*.02+np.arange(5)[None,None,:]*.1
    reference = TerrainReferenceOperator3D(metrics=metrics,kx=scale,ky=2*scale,kz=.1*scale)
    source = reference.divergence(fluxes=reference.fluxes(head=exact))
    fixed = np.zeros(grid.shape,dtype=np.int32)
    fixed[:,:,0] = 1
    with build_solver(grid=grid,kx=scale,ky=2*scale,kz=.1*scale,fixed=fixed,values=exact,
                      source=source,initial=np.zeros(grid.shape)) as solver:
        h,info = solver.solve(abs_tol_min=0,rel_tol=1e-13,dh_rms_tol=1e-9,max_cycles=600)
    assert info["converged"],info
    np.testing.assert_allclose(h,exact,rtol=0,atol=2e-8)


@pytest.mark.parametrize("old,new", [(2,2.5),(3.9,5),(5,3.9),(.2,0),(-1,.3)])
def test_storage_only_exact_potential_drying_rewetting_and_elastic(old,new):
    grid = TerrainGrid3D(surface_elevation=np.array([[4.]]),zeta_faces=np.array([0,4.]),dx=2,dy=2)
    sy,ss = .3,1e-3
    previous = np.full(grid.shape,old)
    expected = np.full(grid.shape,new)
    # Independent analytic integrated capacity [volume], no solver potential.
    old_wet,new_wet = np.clip(old,0,4),np.clip(new,0,4)
    volume = 4*(sy*(new_wet-old_wet)+ss*(.5*(new_wet**2-old_wet**2)+4*(max(new-4,0)-max(old-4,0))))
    with build_solver(grid=grid,kx=0,ky=0,kz=0,fixed=np.zeros(grid.shape,dtype=np.int32),
        values=previous,source=np.full(grid.shape,volume),initial=previous) as solver:
        h,info = solver.solve(unconfined=True,transient=True,head_prev=previous,dt=1,sy=sy,ss=ss,
            rel_tol=1e-12,abs_tol_min=1e-13,unconfined_head_tol=1e-9,final_head_residual_tol=1e-9,
            unconfined_max_picard_iter=400)
    assert info["converged"],info
    if new > 0:
        np.testing.assert_allclose(h,expected,rtol=0,atol=2e-8)
    else:
        assert np.max(h) < 1e-9
    assert info["budget"]["storage_uptake"] == pytest.approx(volume,abs=2e-9)


@pytest.mark.parametrize("net", [0,1])
def test_terrain_connected_unconstrained_component_rejected(net):
    grid = TerrainGrid3D(surface_elevation=np.zeros((1,2)),zeta_faces=np.array([0,1]),dx=1,dy=1)
    with build_solver(grid=grid,kx=1,ky=1,kz=1,fixed=np.zeros(grid.shape,dtype=np.int32),
        values=np.zeros(grid.shape),source=np.array([1,-1+net]).reshape(grid.shape),initial=np.zeros(grid.shape)) as solver:
        with pytest.raises(ValueError,match="datum|No valid steady"):
            solver.solve()


@pytest.mark.parametrize("floor", [.1,.01,.001])
@pytest.mark.parametrize("contrast", [1,1e-14])
def test_terrain_floor_sensitivity_has_no_floor_storage(floor,contrast):
    from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d,terrain_dry_diagnostics_3d
    grid = TerrainGrid3D(surface_elevation=np.array([[0.,.2]]),zeta_faces=np.array([0,.05,1.05]),dx=1,dy=1)
    metrics = TerrainMetrics3D(grid=grid)
    h = grid.elevations-2
    k = np.broadcast_to(np.array([1,contrast])[:,None,None],grid.shape)
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=k,ky=k,kz=.01*k,head=h,min_sat=floor)
    fixed = np.zeros(grid.shape,dtype=np.int32)
    source = np.array([.01,-.02,0,0]).reshape(grid.shape)
    budget = compute_terrain_budget_3d(operator=operator,head=h,source=source,fixed=fixed,
        storage_change=np.zeros(grid.shape),storage_diagonal=np.zeros(grid.shape))
    diagnostic = terrain_dry_diagnostics_3d(operator=operator,head=h,source=source,fixed=fixed,budget=budget)
    # In a wholly dry column, the metric-consistent physical vertical field
    # has zero lateral physical gradient despite the lateral head difference.
    assert diagnostic["horizontal_floor_excess_flow"] < 1e-15
    assert diagnostic["vertical_dry_interface_flow"] > 0
    np.testing.assert_array_equal(storage_potential_3d(head=h,bottoms=grid.bottoms,
        thickness=grid.thickness,sy=.2,ss=1e-4),0)
    assert np.all(operator.regularized_sat <= grid.thickness)


def test_terrain_material_values_refresh_and_mask_topology_rebuild():
    grid,metrics = terrain_case()
    h = np.full(grid.shape,50.)
    fixed = np.zeros(grid.shape,dtype=np.int32)
    fixed[:,:,0] = 1
    with build_solver(grid=grid,kx=1,ky=1,kz=.1,fixed=fixed,values=h,
        source=np.zeros(grid.shape),initial=h) as solver:
        head,info = solver.solve(max_cycles=0)
        workspace = solver._session.workspace
        solver.update_terrain_K_fields(kx_field=2,ky_field=3,kz_field=.2)
        head,info = solver.solve(max_cycles=0)
        assert info["converged"] and solver._session.workspace is workspace
        active = grid.active_columns.copy()
        active[1,2] = False
        solver.update_terrain_geometry(surface_elevation=grid.surface_elevation,zeta_faces=grid.zeta_faces,active_columns=active)
        head,info = solver.solve(max_cycles=0)
        assert info["converged"] and solver._session.workspace is not workspace
        assert not info["workspace_reused"]


def test_terrain_rejected_storage_roundoff_cannot_hide_sources():
    grid = TerrainGrid3D(surface_elevation=np.zeros((1,2)),zeta_faces=np.array([0,1]),dx=1,dy=1)
    previous = np.full(grid.shape,5.)
    source = np.array([1.,-1.]).reshape(grid.shape)
    with build_solver(grid=grid,kx=1,ky=1,kz=1,fixed=np.zeros(grid.shape,dtype=np.int32),
        values=previous,source=source,initial=previous) as solver:
        head,info = solver.solve(transient=True,storage_coeff=1e16,dt=1,head_prev=previous)
    assert not info["converged"]
    assert info["budget"]["domain_imbalance"] == 0
    assert info["budget"]["cell_residual_max"] == 1
    assert info["budget"]["relative_cell_imbalance"] == 1
