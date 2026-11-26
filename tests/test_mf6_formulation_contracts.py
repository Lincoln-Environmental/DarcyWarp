"""Native formulation options change the physical unconfined face law."""
import numpy as np
import pytest

from working_tests.check_mf6_face_formulations import measure_native_face


@pytest.mark.parametrize('newton', [False, True])
def test_native_unconfined_face_formulation(pinned_mf6, tmp_path, newton):
    """All prescribed heads isolate MF6's conductance law from solution error."""
    result = measure_native_face(executable=pinned_mf6, workspace=tmp_path,
        newton=newton, heads=np.array([4., 2.]), conductivity=np.array([1., 4.]),
        top=10., bottom=0., dx=1., dy=1., smoothing_fraction=1.e-6, flow_tolerance=1.e-12,
        ims_controls=dict(complexity='COMPLEX', linear_acceleration='BICGSTAB',
            outer_maximum=200, outer_dvclose=1.e-12, inner_maximum=200,
            inner_dvclose=1.e-12, rcloserecord=[1.e-12, 'strict']))
    assert result['mf6_newton'] is newton
    assert result['matched_law_absolute_error_m3_day'] <= 1.e-12
