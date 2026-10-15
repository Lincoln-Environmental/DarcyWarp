"""Live native MF6 analytic and adapter validation; skipped is unavailable."""
from pathlib import Path
import json
import numpy as np
import pytest
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference,CPUControls,solve_cpu
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6,extract_mf6
from test_terrain_cpu_validation import small_case


@pytest.mark.parametrize('xt3d',[False,True])
def test_live_mf6_layered_affine_exact_flow(tmp_path,xt3d):
    executable=Path('/bin/modflow/mf6')
    if not executable.is_file():pytest.skip('MF6 executable unavailable')
    case=small_case(slope=0.)
    a=case.arrays
    a['interfaces'][:]=np.array([10.,9.,7.])[:,None,None]
    a['fixed'][:]=0
    a['fixed'][:,:,0]=1
    a['fixed'][:,:,-1]=1
    a['values'][:]=np.linspace(20.,19.,case.shape[2])[None,None,:]
    a['source'][:]=0.
    a['initial'][:]=20.
    path=tmp_path/'mf6'
    result=run_mf6(case=case,directory=path,executable=executable,xt3d=xt3d,convertible=False,drains=False)
    assert result['native_success'],result
    assert result['independent_conservation'],result
    assert result['reciprocity_max']<1e-13
    data=np.load(path/'extracted.npz')
    np.testing.assert_allclose(data['head'],a['values'],rtol=0.,atol=1e-8)
    expected=a['kx'][:,:,:-1]*np.array([1.,2.])[:,None,None]*3./2./4.
    np.testing.assert_allclose(data['qx'],expected,rtol=1e-7,atol=1e-10)
    np.testing.assert_allclose(data['qz'],0.,rtol=0.,atol=1e-10)
    cpu,info=solve_cpu(case=case,controls=CPUControls(max_outer=20))
    assert info['converged']
    np.testing.assert_allclose(cpu,data['head'],rtol=0.,atol=1e-8)
    refused=extract_mf6(case=case,directory=path,native_success=False)
    assert not refused['independent_conservation']


def test_missing_output_never_passes(tmp_path):
    case=small_case()
    result=extract_mf6(case=case,directory=tmp_path,native_success=True)
    assert not result['independent_conservation']
    assert result['extraction']=='missing_output'


def test_live_mf6_exact_source_precision_and_row_mapping(tmp_path):
    executable=Path('/bin/modflow/mf6')
    if not executable.is_file():pytest.skip('MF6 executable unavailable')
    case=small_case(slope=0.)
    a=case.arrays
    a['interfaces'][:]=np.array([10.,9.,7.])[:,None,None]
    a['values'][:]=20.
    a['initial'][:]=20.
    a['source'][:]=0.
    a['source'][0,0,1]=.1234567891234567
    result=run_mf6(case=case,directory=tmp_path/'model',executable=executable,xt3d=False,convertible=False,drains=False)
    assert result['native_success'] and result['independent_conservation'],result
    assert result['source_allocation_max_error']<1e-15
    fields=np.load(tmp_path/'model'/'extracted.npz')
    np.testing.assert_array_equal(fields['source_realised'],a['source'])
