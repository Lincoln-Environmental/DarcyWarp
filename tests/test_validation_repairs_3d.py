"""Boundary, inactive-marker and empty-domain regression reproductions."""
from pathlib import Path
import numpy as np
import pytest
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_compare import evaluate_saved_head, compare_fields
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import build_mf6
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D


def flat_case(*, shape=(2, 2, 3), masked=False):
    nz, ny, nx = shape
    interfaces = np.broadcast_to(np.arange(nz + 1, dtype=float)[::-1, None, None], (nz + 1, ny, nx)).copy()
    active = np.ones((ny, nx), dtype=bool)
    if masked:
        active[0, 1] = False
    fixed = np.zeros(shape, dtype=int)
    fixed[:, :, 0] = 1
    fixed[:, :, -1] = 1
    fixed[:, ~active] = 0
    values = np.broadcast_to(np.linspace(2., 4., nx), shape).copy()
    if masked:
        values[:] = 4.
    one = np.ones(shape)
    return TerrainCase(case_id='flat_validation_repair', arrays=dict(interfaces=interfaces,
        active=active, fixed=fixed, values=values, initial=values.copy(), source=np.zeros(shape),
        kx=one, ky=one.copy(), kz=one.copy()), physical=dict(dx=1., dy=1., floor=.01,
        steady=True, storage=0), controls={}, contacts=())


def test_shifted_prescribed_heads_are_rejected():
    case = flat_case()
    shifted = case.arrays['values'] + 10.
    result, fields = evaluate_saved_head(case=case, head=shifted)
    assert result['budget']['cell_residual_rms'] == 0.
    assert not result['independent_accepted']
    assert not result['boundary_condition_valid']
    assert result['prescribed_cells_checked'] == 8
    assert result['prescribed_head_error_max_m'] == 10.
    assert result['prescribed_head_tolerance_m'] <= 1e-9
    np.testing.assert_array_equal(fields['head'], shifted)


@pytest.mark.parametrize('marker', [np.nan, 1e30])
def test_inactive_markers_are_preserved_and_excluded(marker):
    case = flat_case(masked=True)
    head = case.arrays['values'].copy()
    head[:, 0, 1] = marker
    result, fields = evaluate_saved_head(case=case, head=head)
    assert result['independent_accepted'], result
    assert result['invalid_active_heads'] == 0
    assert result['inactive_marker_cells'] == 2
    np.testing.assert_array_equal(fields['head'], head)
    assert np.all(np.isfinite(fields['evaluation_head']))
    head[0, 1, 1] = marker
    invalid, fields = evaluate_saved_head(case=case, head=head)
    assert not invalid['independent_accepted']
    assert invalid['invalid_active_heads'] == 1


def test_empty_free_layers_and_comparison_subsets():
    case = flat_case()
    case.arrays['fixed'][0] = 1
    result, fields = evaluate_saved_head(case=case, head=case.arrays['values'])
    assert result['independent_accepted']
    assert result['layers'][0]['free_cells'] == 0
    assert result['layers'][0]['head_min'] is None
    assert result['layers'][0]['head_max'] is None
    comparison = compare_fields(case=case, left=fields, right=fields)
    assert comparison['layerwise_head_rms'][0] is None
    case.arrays['fixed'][:] = 1
    result, fields = evaluate_saved_head(case=case, head=case.arrays['values'])
    comparison = compare_fields(case=case, left=fields, right=fields)
    assert comparison['free_cells'] == 0
    assert comparison['head_max'] is None and comparison['head_rms'] is None


def test_prescribed_head_check_uses_absolute_tolerance_at_large_datum():
    case=flat_case()
    case.arrays['values']+=1e9
    head=case.arrays['values']+.001
    result,unused=evaluate_saved_head(case=case,head=head)
    assert not result['boundary_condition_valid']


def test_native_contacts_require_an_explicit_identity(tmp_path):
    case=flat_case()
    case.arrays['fixed'][:,:,0]=0
    case.contacts=(TerrainContact3D(cell=(0,1,0),direction='west',area=1.,distance=.5,
        elevation=1.5,head=0.,kind='discharge_only'),)
    with pytest.raises(ValueError,match='explicit contact_mode'):
        build_mf6(case=case,directory=tmp_path,executable=Path('/bin/modflow/mf6'),xt3d=False,convertible=True)


def test_live_masked_mf6_output_and_node_mapping(tmp_path):
    executable = Path('/bin/modflow/mf6')
    if not executable.is_file():
        pytest.skip('MF6 executable unavailable; live mapping unverified')
    case = flat_case(shape=(2, 3, 5), masked=True)
    case.arrays['source'][0, 0, 3] = .01234567891234567
    directory = tmp_path / 'masked'
    result = run_mf6(case=case, directory=directory, executable=executable,
        xt3d=False, convertible=False, drains=False, contact_mode='none',head_tol=1e-12,residual_tol=1e-12)
    assert result['native_success'] and result['independent_conservation'], result
    assert result['invalid_active_heads'] == 0
    assert result['inactive_marker_cells'] == 2
    assert result['source_allocation_max_error'] < 1e-15
    data = np.load(directory / 'extracted.npz')
    assert not np.isfinite(data['raw_head'][:, 0, 1]).all() or np.all(np.abs(data['raw_head'][:, 0, 1]) > 1e20)
    np.testing.assert_array_equal(data['source_realised'], case.arrays['source'])
    assert len(np.unique(data['node_to_user'])) == len(data['node_to_user'])
    checked, fields = evaluate_saved_head(case=case, head=data['head'])
    assert checked['independent_accepted'], checked
