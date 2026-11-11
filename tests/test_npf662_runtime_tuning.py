"""Saved-reference controls and validity must survive runtime experiments."""
import hashlib
import json

import numpy as np
import pytest

from benchmark_npf662_vs_mf6 import suite_settings
from tune_npf_gpu_runtime import load_reference
from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import generate_case


def cached_case(*, folder):
    definition = suite_settings(sizes=[4], variants=['convertible'])[0]
    settings = dict(definition['geology'], layer_k=np.asarray(definition['geology']['layer_k']))
    case = generate_case(settings=settings, convertible=definition['convertible'])
    factors = np.ones(2)
    options = dict(dt=7., ss=1e-5, sy=.15)
    native_options = dict(outer_dvclose=1e-6, inner_dvclose=1e-8)
    np.savez(file=folder/'reference.npz', heads=np.stack([case.initial, case.initial]))
    (folder/'native').mkdir()
    (folder/'native/stdout.log').write_text('Normal termination of simulation\n')
    physical = dict(input_valid=True, boundary_condition_valid=True, physical_metrics_valid=True)
    manifest = dict(accepted=False, fingerprint='fixture', inputs=dict(
        case_sha256=case.digest(), factors=factors.tolist(), native_options=native_options,
        mode='transient', family='npf', **options), native=dict(periods=[dict(
            source_allocation_valid=True, physical=physical, native_relative_cell_imbalance=1e-7)]*2),
        sha256={'reference.npz':hashlib.sha256((folder/'reference.npz').read_bytes()).hexdigest()})
    (folder/'manifest.json').write_text(json.dumps(manifest))
    return case, factors, options, native_options, manifest


def test_old_strict_audit_flag_does_not_replace_shared_sanity_gate(tmp_path):
    case, factors, options, native_options, manifest = cached_case(folder=tmp_path)
    heads, unused, path = load_reference(folder=tmp_path, case=case, factors=factors,
        options=options, native_options=native_options)
    assert path == tmp_path and heads.shape == (2, *case.active.shape)


@pytest.mark.parametrize('change', ['ims', 'hash', 'termination', 'physical', 'source_allocation'])
def test_incompatible_or_invalid_truth_is_rejected(tmp_path, change):
    case, factors, options, native_options, manifest = cached_case(folder=tmp_path)
    if change == 'ims':
        native_options = dict(native_options, outer_dvclose=1e-3)
    elif change == 'hash':
        manifest['sha256']['reference.npz'] = 'mismatch'
    elif change == 'termination':
        (tmp_path/'native/stdout.log').write_text('Simulation failed\n')
    elif change == 'physical':
        manifest['native']['periods'][0]['physical']['physical_metrics_valid'] = False
    else:
        manifest['native']['periods'][0]['source_allocation_valid'] = False
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        load_reference(folder=tmp_path, case=case, factors=factors, options=options, native_options=native_options)
