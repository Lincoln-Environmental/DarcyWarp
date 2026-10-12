"""Independent hydro-diagnostics: water table, coastal split, discharge potential."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D, TerrainContact3D
from DARCY_WARP_PACKAGE.physics.terrain_shanz_diagnostics import (
    water_table_profile, coastal_exchange, surface_discharge_potential)


def layered_grid():
    n = 4
    x = np.arange(n)*2.0
    surface = np.full((1, n), 20.)
    mid = surface-4.
    base = mid-5.
    grid = TerrainGrid3D(z_interfaces=np.stack((surface,mid,base)),dx=2.,dy=2.)
    return grid, TerrainMetrics3D(grid=grid)


def test_water_table_phreatic_potentiometric_and_dry_columns():
    grid, unused = layered_grid()
    head = np.zeros(grid.shape)
    head[0, 0, 0] = 18.5      # inside layer 0: water table at 18.5
    head[:, 0, 1] = 30.       # fully saturated + pressurised: potentiometric
    head[0, 0, 2] = 14.       # layer 0 dry; water table inside layer 1 at 14
    head[1, 0, 2] = 14.
    head[:, 0, 3] = 10.       # at the layer-0 bottom exactly: dry column
    elevation, kind, depth = water_table_profile(head=head,
        z_interfaces=grid.z_interfaces, surface=grid.surface_elevation)
    assert kind[0,0] == "water_table" and elevation[0,0] == 18.5
    assert kind[0,1] == "potentiometric" and np.isnan(elevation[0,1])
    assert kind[0,2] == "water_table" and elevation[0,2] == 14.
    assert kind[0,3] == "dry" and np.isnan(elevation[0,3])
    np.testing.assert_array_equal(depth[0], grid.surface_elevation[0]-elevation[0])


def test_coastal_exchange_split_and_net():
    grid, metrics = layered_grid()
    fixed = np.zeros(grid.shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    values = np.zeros(grid.shape)
    values[:, :, 0] = 12.
    operator = TerrainReferenceOperator3D(metrics=metrics,kx=1.,ky=1.,kz=1.)
    h = np.full(grid.shape, 15.)
    h[:, :, 0] = 12.
    # Independent face-walk: gross exchange across free-fixed faces.
    fluxes = operator.fluxes(head=h)
    expected_in, expected_out = 0., 0.
    fm = fixed != 0
    for axis, flux in zip((2,1,0),fluxes[:3]):
        left,right = [slice(None)]*3,[slice(None)]*3
        left[axis],right[axis] = slice(None,-1),slice(1,None)
        left,right = tuple(left),tuple(right)
        to_fixed = np.where(fm[right]&~fm[left],flux,0.)
        to_free = np.where(fm[left]&~fm[right],-flux,0.)
        expected_out += float(np.sum(np.maximum(to_fixed,0)))+float(np.sum(np.maximum(to_free,0)))
        expected_in += float(np.sum(np.maximum(-to_fixed,0)))+float(np.sum(np.maximum(-to_free,0)))
    spring = TerrainContact3D(cell=(0,0,3),direction="east",area=1.,distance=1.,
        elevation=float(grid.bottoms[0,0,3]),head=0.,kind="discharge_only",conductance=.5)
    spring_op = TerrainReferenceOperator3D(metrics=metrics,kx=1.,ky=1.,kz=1.,contacts=(spring,))
    spring_out = float(np.sum(np.maximum(spring_op.fluxes(head=h)[3],0.)))
    split = coastal_exchange(operator=operator,head=h,fixed=fixed,spring_outflows=[spring_out])
    assert split["fixed_head_inflow"] == pytest.approx(expected_in,abs=1e-12)
    assert split["fixed_head_outflow"] == pytest.approx(expected_out,abs=1e-12)
    assert split["spring_outflow"] == pytest.approx(spring_out,abs=1e-12)
    assert split["net_coastal_exchange"] == pytest.approx(expected_out+spring_out-expected_in,abs=1e-12)


def test_surface_discharge_potential_separation():
    grid, unused = layered_grid()
    head = np.zeros(grid.shape)
    head[:, 0, 0] = 25.       # fully saturated, head above ground: artesian
    head[0, 0, 1] = 20.       # head exactly at ground: water table at surface
    head[0, 0, 2] = 15.       # water table below ground, deep head above it
    head[1, 0, 2] = 21.
    head[:, 0, 3] = 12.       # ordinary
    result = surface_discharge_potential(head=head, z_interfaces=grid.z_interfaces,
        surface=grid.surface_elevation)
    assert result["artesian_cells"] == 1
    assert result["water_table_at_surface_cells"] == 1
    assert result["deep_head_above_surface_cells"] == 1
