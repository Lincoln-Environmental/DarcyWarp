"""The fast 2D reductions must overwrite outputs and clear reusable partials."""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.solvers.face_kernels_f64 import launch_combine_partials


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('count', [0, 1, 17, 255, 256, 257, 3907])
@pytest.mark.parametrize('maximum', [False, True])
def test_partial_combine_matches_numpy_and_clears_only_inputs(device, count, maximum):
    if device.startswith('cuda') and not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    rng = np.random.default_rng(seed=17)
    values = rng.normal(size=count+5)
    partials = wp.array(data=values, dtype=wp.float64, device=device)
    out = wp.full(shape=1, value=wp.float64(-999.), dtype=wp.float64, device=device)
    launch_combine_partials(partials=partials, out=out, n_partials=count,
        device=device, maximum=maximum)
    expected = max(0., float(np.max(values[:count]))) if maximum and count else float(np.sum(values[:count]))
    np.testing.assert_allclose(actual=out.numpy()[0], desired=expected, rtol=1e-13, atol=1e-12)
    np.testing.assert_array_equal(actual=partials.numpy()[:count], desired=np.zeros(count))
    np.testing.assert_array_equal(actual=partials.numpy()[count:], desired=values[count:])
    launch_combine_partials(partials=partials, out=out, n_partials=count,
        device=device, maximum=maximum)
    assert out.numpy()[0] == 0.


@pytest.mark.parametrize('maximum', [False, True])
def test_partial_combine_graph_replays_current_values(maximum):
    if not wp.is_cuda_available():
        pytest.skip('CUDA unavailable')
    device = 'cuda:0'
    count = 3907
    source = wp.ones(shape=count, dtype=wp.float64, device=device)
    partials = wp.zeros(shape=count, dtype=wp.float64, device=device)
    out = wp.zeros(shape=1, dtype=wp.float64, device=device)
    launch_combine_partials(partials=partials, out=out, n_partials=count,
        device=device, maximum=maximum)
    with wp.ScopedCapture(device=device) as capture:
        wp.copy(dest=partials, src=source)
        launch_combine_partials(partials=partials, out=out, n_partials=count,
            device=device, maximum=maximum)
    for value in (2., 5.):
        source.fill_(value=value)
        wp.capture_launch(graph=capture.graph)
        assert out.numpy()[0] == (value if maximum else value * count)
        np.testing.assert_array_equal(actual=partials.numpy(), desired=np.zeros(count))
