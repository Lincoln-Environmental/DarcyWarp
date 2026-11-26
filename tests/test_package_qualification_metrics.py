"""Head and rate agreement cannot substitute for accepted volume agreement."""
import pytest

from working_tests.compare_package_qualification import compare_accepted_boundary_volumes


@pytest.mark.parametrize('accepted_outflow,accepted', [(6., True), (3., False)])
def test_volume_comparison_uses_driver_ledger(accepted_outflow, accepted):
    """A three-unit rate over two days is six units, even with correct heads."""
    inputs = dict(accepted_totals={'drn': dict(inflow=0., outflow=accepted_outflow, net=-accepted_outflow)},
        native_periods=[dict(dt=2., volumes={'DRN': dict(inflow=0., outflow=6., net=-6.)})],
        entry_counts={'DRN': 1}, rate_gate=1.e-8)
    if accepted:
        result = compare_accepted_boundary_volumes(**inputs)
        assert result['DRN']['errors_m3'] == dict(inflow=0., outflow=0., net=0.)
    else:
        with pytest.raises(AssertionError):
            compare_accepted_boundary_volumes(**inputs)
