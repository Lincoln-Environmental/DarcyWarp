import pytest

from working_tests.diagnose_3d_confined_mf6 import select_confined_case, save_snapshot


@pytest.mark.parametrize('changes,message', [
    ({'variant': 'convertible'}, 'requires a confined problem'),
    ({'gpu_physical_accepted': False}, 'GPU physical checks must pass'),
    ({'native_reference_accepted': True}, 'missing or invalid native references only'),
])
def test_confined_fallback_refuses_ineligible_cases(changes, message):
    row = dict(variant='confined', gpu_physical_accepted=True,
               native_reference_accepted=False, fingerprint='example')
    row.update(changes)
    with pytest.raises(ValueError, match=message):
        select_confined_case(baseline=dict(complete=True, selected=[row]),
                             selection=dict(fingerprint='example'))


def test_confined_fallback_refuses_ambiguous_selection():
    row = dict(variant='confined', gpu_physical_accepted=True,
               native_reference_accepted=False)
    with pytest.raises(ValueError, match='exactly one'):
        select_confined_case(baseline=dict(complete=True, selected=[row, row.copy()]), selection={})


def test_changed_driver_cannot_overwrite_an_executed_snapshot(tmp_path):
    source = tmp_path / 'source.py'
    saved = tmp_path / 'saved.py'
    source.write_text(data='original source\n')
    save_snapshot(source=source, destination=saved)
    source.write_text(data='changed source\n')
    with pytest.raises(ValueError, match='choose a new output directory'):
        save_snapshot(source=source, destination=saved)
    assert saved.read_text() == 'original source\n'
