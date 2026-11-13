"""Keep failed native references distinct from successful benchmark samples."""
import json
import subprocess

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.validation import npf662_mf6
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase


@pytest.mark.parametrize('outcome,reason,returncode,native_success', [
    ('timeout', 'wall_timeout', None, False),
    ('failure', 'native_failure_or_interrupted', 1, False),
    ('missing_output', 'native_missing_output', 0, True),
])
def test_native_failure_reason_survives_missing_output(
        tmp_path, pinned_mf6, monkeypatch, outcome, reason, returncode, native_success):
    shape = (1, 1, 2)
    one = np.ones(shape=shape)
    case = NPF662LayeredCase(kx=one, ky=one, kz=one,
        active=one.astype(int), fixed=np.asarray(a=[0, 1]).reshape(shape),
        values=2.*one, source=np.zeros(shape=shape), initial=2.*one,
        dx=1., dy=1., z_interfaces=np.stack(arrays=[3.*one[0], 0.*one[0]]),
        cell_type=0)
    directory = tmp_path / 'native'
    original_run = subprocess.run

    def simulate_native_run(args, **options):
        # Let the executable's real version probe and identity checks run.
        if options.get('cwd') != directory:
            return original_run(args=args, **options)
        assert options['timeout'] == .1
        if outcome == 'timeout':
            raise subprocess.TimeoutExpired(cmd=args, timeout=.1, output=b'partial native output\n')
        stdout = 'Normal termination of simulation\n' if outcome == 'missing_output' else 'native failed\n'
        return subprocess.CompletedProcess(args=args, returncode=returncode, stdout=stdout, stderr='')

    monkeypatch.setattr(npf662_mf6.subprocess, 'run', simulate_native_run)
    head, info, fields = npf662_mf6.run_reference_mf6(
        case=case, directory=directory, executable=pinned_mf6, timeout_seconds=.1)
    assert head is None and fields is None
    assert info['stop_reason'] == reason
    assert info['returncode'] == returncode
    assert info['native_success'] is native_success
    assert not info['independent_conservation']
    saved = json.loads(s=(directory / 'summary.json').read_text())
    assert saved['stop_reason'] == reason
    assert not saved['independent_conservation']
    assert (directory / 'stdout.log').read_text()
