"""Benchmark controls must track the 2D sanity source, including residual type."""
from pathlib import Path
import shutil

import pytest

from sanity_3d_controls import read_profiles


def copy_sources(*, root, destination):
    profiles = read_profiles(root=root)
    for relative in profiles['source_sha256']:
        target = destination/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src=root/relative, dst=target)
    return profiles


def test_profile_follows_2d_source_edits_without_importing_workflow(tmp_path):
    original = copy_sources(root=Path('.'), destination=tmp_path)
    path = tmp_path/'DARCY_WARP_PACKAGE/model_convergence_and_sanity_tests.py'
    text = path.read_text()
    assert 'rel_tol=5.0e-7' in text
    path.write_text(data=text.replace('rel_tol=5.0e-7', 'rel_tol=7.5e-7', 1)
                    + '\nraise RuntimeError("The benchmark must not import this workflow")\n')
    changed = read_profiles(root=tmp_path)
    for implementation in ('classic', 'fast'):
        assert changed['steady'][implementation]['rel_tol'] == 7.5e-7
    relative = str(path.relative_to(tmp_path))
    assert changed['source_sha256'][relative] != original['source_sha256'][relative]
    assert changed['mf6']['transient']['rcloserecord'][1] == 'RELATIVE_RCLOSE'


def test_unresolved_native_profile_fails_instead_of_omitting_settings(tmp_path):
    copy_sources(root=Path('.'), destination=tmp_path)
    path = tmp_path/'DARCY_WARP_PACKAGE/modflow_truth.py'
    text = path.read_text()
    assert 'rclose = 1.0e-6' in text
    path.write_text(data=text.replace('rclose = 1.0e-6', 'rclose = unresolved_setting()', 1))
    with pytest.raises(ValueError, match='complete native sanity profile'):
        read_profiles(root=tmp_path)
