# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the MF6 input adapter and its binary output."""

import numpy as np
import pytest

flopy = pytest.importorskip("flopy")

from DARCY_WARP_PACKAGE.mf6_import import (
    UnsupportedMF6Model, import_simulation, run_mf6_model,
    timestep_lengths, _schedules, _stress_period, _add_array_recharge,
)


def make_simulation(workspace, *, xt3d=False, nonuniform=False):
    """Create a native structured model with mixed stress-period behaviour.

    :param workspace: MF6 workspace path.
    :param xt3d: Enable an unsupported option.
    :param nonuniform: Construct an unsupported variable-spacing grid.
    """
    sim = flopy.mf6.MFSimulation(sim_name="import", sim_ws=str(workspace))
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=3,
        perioddata=[(1.0, 1, 1.0), (3.0, 2, 2.0), (1.0, 1, 1.0)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="aquifer", save_flows=True)
    flopy.mf6.ModflowIms(sim, complexity="COMPLEX", outer_dvclose=1e-10,
                        inner_dvclose=1e-12, rcloserecord=[1e-10, "STRICT"],
                        linear_acceleration="BICGSTAB")
    flopy.mf6.ModflowGwfdis(gwf, nlay=2, nrow=3, ncol=5, top=10.0,
        botm=[5.0, 0.0], delr=[10, 10, 11, 10, 10] if nonuniform else 10.0,
        delc=20.0, length_units="METERS")
    flopy.mf6.ModflowGwfic(gwf, strt=9.0)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=0, k=2.0, k33=0.5,
                          xt3doptions=True if xt3d else None)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=0.001, sy=0.0,
                          steady_state={0: True}, transient={1: True})
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data={
        0: [((k, j, 0), 9.0) for k in range(2) for j in range(3)]})
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=[
        ((0, 1, 4), 8.0, 2.0), ((0, 1, 4), 9.0, 1.0)])
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={
        0: [((1, 1, 3), -0.25), ((1, 1, 3), 0.05)], 2: []})
    flopy.mf6.ModflowGwfrcha(gwf, recharge={0: 0.001, 1: 0.002}, fixed_cell=True)
    flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=[((0, 0, 3), 8.5, 0.1)])
    flopy.mf6.ModflowGwfriv(gwf, stress_period_data=[((0, 2, 3), 8.7, 0.2, 7.0)])
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="aquifer.hds",
                         budget_filerecord="aquifer.cbc",
                         saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    return sim


def test_timestep_sequence():
    """TDIS geometric timesteps preserve duration and ratio."""
    np.testing.assert_allclose(timestep_lengths(7, 3, 2), [1, 2, 4])
    np.testing.assert_allclose(timestep_lengths(7, 3, 0.5), [4, 2, 1])
    np.testing.assert_allclose(timestep_lengths(6, 3, 1), [2, 2, 2])
    with pytest.raises(UnsupportedMF6Model):
        timestep_lengths(1, 0, 1)


def test_import_translation(tmp_path):
    """Read actual MF6 files, with inherited and explicitly cleared lists."""
    make_simulation(tmp_path)
    sim = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_path), verbosity_level=0)
    model = import_simulation(sim)
    assert model.shape == (2, 3, 5)
    assert (model.dx, model.dy) == (10, 20)
    assert [period[0] for period in model.periods] == [False, True, True]
    np.testing.assert_allclose(model.periods[1][1], [1.0, 2.0])
    np.testing.assert_allclose(model.ky, model.kx)
    np.testing.assert_allclose(model.kz, 0.5)
    schedules = _schedules(model)
    first = _stress_period(model, schedules, 0)
    second = _stress_period(model, schedules, 1)
    third = _stress_period(model, schedules, 2)
    assert len(first["wel_entries"]) == len(second["wel_entries"]) == 2
    assert third["wel_entries"] is None
    assert first["ghb_conductance"][0, 1, 4] == 3.0
    assert first["ghb_head"][0, 1, 4] == pytest.approx(25 / 3)
    assert first["drn_entries"] == [((0, 0, 3), 0.1, 8.5)]
    assert first["riv_entries"] == [((0, 2, 3), 0.2, 8.7, 7.0)]
    assert first["rhs"].sum() == pytest.approx(2.4)
    assert second["rhs"].sum() == third["rhs"].sum() == pytest.approx(4.8)


@pytest.mark.parametrize("options", [{"xt3d": True}, {"nonuniform": True}])
def test_unsupported_options(tmp_path, options):
    """Reject unsupported physics and geometry before solving."""
    sim = make_simulation(tmp_path, **options)
    with pytest.raises(UnsupportedMF6Model):
        import_simulation(sim)


def test_import_run(tmp_path, solver_device):
    """Execute the imported model and read every accepted binary HEAD record."""
    workspace = tmp_path / "mf6"
    make_simulation(workspace)
    output = tmp_path / "dw"
    report = run_mf6_model(workspace, output, device=solver_device)
    assert report["accepted_steps"] == 4
    heads = flopy.utils.HeadFile(report["head_file"], precision="double")
    try:
        np.testing.assert_allclose(heads.get_times(), [1, 2, 4, 5])
        assert heads.get_alldata().shape == (4, 2, 3, 5)
        np.testing.assert_allclose(heads.get_alldata()[:, :, :, 0], 9.0)
    finally:
        heads.close()


def test_native_parity(tmp_path, solver_device, pinned_mf6):
    """Compare an independently executed MF6 workspace at each time step."""
    workspace = tmp_path / "mf6"
    sim = make_simulation(workspace)
    sim.exe_name = str(pinned_mf6)
    success, output = sim.run_simulation(silent=True, report=True)
    assert success, output
    report = run_mf6_model(workspace, tmp_path / "dw", device=solver_device,
                           reference_head_file=workspace / "aquifer.hds")
    assert len(report["head_comparison"]) == 4
    for comparison in report["head_comparison"]:
        assert comparison["max_abs"] < 1e-6, comparison


def test_array_recharge_idomain_and_irch(tmp_path):
    """Excluded IDOMAIN and fixed-head cells do not receive array recharge."""
    sim = make_simulation(tmp_path)
    model = import_simulation(sim)
    model.active[0, 1, 1] = False
    fixed = np.zeros(model.shape, dtype=bool)
    fixed[0, :, 0] = True
    rhs = np.zeros(model.shape)
    _add_array_recharge(rhs, model, 0.01, 0, fixed)
    assert rhs[0, 1, 1] == rhs[1, 1, 1] == 0.0
    assert rhs.sum() == pytest.approx(22.0)
    rhs.fill(0)
    irch = np.zeros(model.shape[1:], dtype=int)
    irch[1, 1] = 1
    _add_array_recharge(rhs, model, 0.01, irch, fixed)
    assert rhs[1, 1, 1] == pytest.approx(2.0)
    assert rhs.sum() == pytest.approx(24.0)
