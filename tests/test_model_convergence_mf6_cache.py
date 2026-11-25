"""Steady reference caching uses physical inputs and exact reference identity."""

from pathlib import Path

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.model_builder import _build_domain
from DARCY_WARP_PACKAGE.validation.steady_truth_2d import load_or_run_mf6_truth, steady_transmissivity


@pytest.mark.parametrize("isotropic", [True, False])
def test_reference_cache_tracks_the_actual_field_and_writer(tmp_path: Path, isotropic):
    pytest.importorskip("flopy")
    nx, ny = 9, 8
    domain = _build_domain(nx=nx, ny=ny)
    t_field = steady_transmissivity(domain=domain, isotropic=isotropic, isotropic_value=3000.0, seed=42)
    conductivity = t_field / 300.0
    recharge = np.full((ny, nx), 1.0e-4)
    heads = np.arange(nx * ny, dtype=float).reshape(ny, nx)
    truth_path = tmp_path / "historical.npz"
    np.savez(truth_path, heads=-heads)
    executable = tmp_path / "mf6"
    executable.write_bytes(b"first identity")
    received = []

    def fake_reference(**kwargs):
        received.append(kwargs)
        return heads, 0.75

    options = dict(truth_path=truth_path, workspace=tmp_path / "workspace", label="9x8",
        nx=nx, ny=ny, dx=100., ghb=True, isotropic=isotropic, t_isotropic_value=3000.,
        thickness=300., width=100., recharge_rate=1e-4, recharge_field=recharge, seed=42,
        hk_field=conductivity, output_dtype=np.dtype(np.float64), mf6_runner=fake_reference,
        domain=domain, executable=executable)
    result, seconds, source = load_or_run_mf6_truth(**options)
    assert source == "generated" and seconds == 0.75
    assert received[0]["hk"] is conductivity and received[0]["recharge"] is recharge
    assert (np.unique(conductivity[domain != 0]).size == 1) is isotropic
    np.testing.assert_array_equal(result, heads)
    assert load_or_run_mf6_truth(**options)[2] == "cache"
    assert len(received) == 1

    conductivity[4, 4] *= 1.1
    assert load_or_run_mf6_truth(**options)[2] == "generated"
    executable.write_bytes(b"second identity")
    assert load_or_run_mf6_truth(**options)[2] == "generated"
    assert load_or_run_mf6_truth(**options)[2] == "cache"
    assert len(received) == 3
    with np.load(truth_path) as archived:
        np.testing.assert_array_equal(archived["heads"], -heads)


def test_masked_native_reference_and_reported_artifact(tmp_path, pinned_mf6):
    import hashlib
    import flopy
    from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model

    shape = (6, 6)
    active = np.ones(shape, dtype=np.int8)
    active[3, 3] = 0
    archive = tmp_path / "legacy.npz"
    np.savez(archive, heads=np.ones(shape))
    report = {}
    options = dict(truth_path=archive, workspace=tmp_path / "native", label="6x6", nx=6, ny=6,
        dx=100., ghb=True, isotropic=True, t_isotropic_value=3000., thickness=300., width=100.,
        recharge_rate=1e-4, recharge_field=np.full(shape, 1e-4), seed=42, hk_field=np.full(shape, 10.),
        output_dtype=np.float64, mf6_runner=make_mf_model, domain=active, executable=pinned_mf6,
        provenance=report)
    heads, _, source = load_or_run_mf6_truth(**options)
    sim = flopy.mf6.MFSimulation.load(sim_ws=options["workspace"], verbosity_level=0)
    np.testing.assert_array_equal(sim.get_model().dis.idomain.array[0], active)
    assert source == "generated" and np.isnan(heads[3, 3])
    actual = Path(report["path"])
    assert actual != archive and actual.exists()
    assert hashlib.sha256(actual.read_bytes()).hexdigest() == report["sha256"]
    with np.load(actual) as stored:
        assert str(stored["reference_fingerprint"]) == report["fingerprint"]
    assert load_or_run_mf6_truth(**options)[2] == "cache"


@pytest.mark.parametrize('marker', [-1e30, 1e30, np.nan])
def test_reference_rejects_invalid_active_heads_on_creation_and_reuse(tmp_path, marker):
    from DARCY_WARP_PACKAGE.validation.steady_truth_2d import load_cached_mf6_truth
    mask = np.ones((3, 3), dtype=int)
    mask[0, 0] = 0
    heads = np.ones(mask.shape)
    heads[0, 0] = marker
    exe = tmp_path/'mf6'
    exe.write_bytes(b'fixture')
    path = tmp_path/'ref.npz'
    provenance = {}

    def runner(**kwargs):
        return heads.copy(), 1.

    opts = dict(truth_path=path, workspace=tmp_path, label='3x3', nx=3, ny=3, dx=1.,
        ghb=False, isotropic=True, t_isotropic_value=1., thickness=1., width=1.,
        recharge_rate=0., recharge_field=np.zeros(mask.shape), seed=1,
        hk_field=np.ones(mask.shape), output_dtype=np.float64, mf6_runner=runner,
        domain=mask, executable=exe, provenance=provenance)
    load_or_run_mf6_truth(**opts)
    assert load_or_run_mf6_truth(**opts)[2] == 'cache'
    with np.load(path) as saved:
        fields = {name: saved[name] for name in saved.files}
    heads[1, 1] = marker
    fields["heads"] = heads
    np.savez(path, **fields)
    cached = load_cached_mf6_truth(path, '3x3', 3, 3, 1., False, True, 1., 1., 1., 0., 1,
        fingerprint=provenance['fingerprint'], active=mask)
    assert cached is None
    with pytest.raises(ValueError, match='active domain'):
        load_or_run_mf6_truth(**opts)


def test_legacy_custom_writer_can_only_qualify_its_default_domain(tmp_path):
    def old_writer(*, nx, ny, grid_size, nper, workspace, hk, recharge, run, use_ghb,
                   ghb_width, aquifer_thickness, executable):
        return np.ones((ny, nx)), 1.

    shape = (3, 4)
    exe = tmp_path/'mf6'
    exe.write_bytes(b'fixture')
    opts = dict(truth_path=tmp_path/'ref.npz', workspace=tmp_path, label='4x3',
        nx=4, ny=3, dx=1., ghb=False, isotropic=True, t_isotropic_value=1., thickness=1.,
        width=1., recharge_rate=0., recharge_field=np.zeros(shape), seed=1,
        hk_field=np.ones(shape), output_dtype=np.float64, mf6_runner=old_writer, executable=exe)
    assert load_or_run_mf6_truth(**opts)[2] == 'generated'
    mask = np.ones(shape)
    mask[1, 1] = 0
    with pytest.raises(ValueError, match='cannot represent a custom domain'):
        load_or_run_mf6_truth(**opts, domain=mask)
