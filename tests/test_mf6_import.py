# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the MF6 input adapter and its binary output."""

import os
import subprocess
import sys

import numpy as np
import pytest

flopy = pytest.importorskip("flopy")

from DARCY_WARP_PACKAGE.mf6_import import (
    UnsupportedMF6Model, import_simulation, run_mf6_model,
    timestep_lengths, _schedules, _stress_period, _add_array_recharge,
    _prepare_interfaces, _prepare_run_outputs,
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


def test_excluded_geometry_preserves_active_interfaces():
    """Repair excluded columns, end cells and internal gaps without changing live cells."""
    # Columns: entirely inactive, top excluded, bottom excluded, internal gap.
    original = np.array([
        [0, 5, 10, 10],
        [0, 5, 8, 8],
        [0, 3, 4, 9],
        [0, 0, 4, 4],
        [0, -1, 4, 0],
    ], dtype=float)[:, None, :]
    active = np.array([
        [0, 0, 1, 1],
        [0, 1, 1, 0],
        [0, 1, 0, 0],
        [0, 1, 0, 1],
    ], dtype=bool)[:, None, :]
    original_copy = original.copy()
    geometry = _prepare_interfaces(original, active)
    assert np.all(geometry[:-1] > geometry[1:])
    np.testing.assert_array_equal(geometry[:-1][active], original[:-1][active])
    np.testing.assert_array_equal(geometry[1:][active], original[1:][active])
    np.testing.assert_array_equal(original, original_copy)


def test_active_zero_thickness_rejected():
    """A real active-cell pinch-out must not be silently changed."""
    interfaces = np.array([10., 10., 0.])[:, None, None]
    with pytest.raises(UnsupportedMF6Model, match="active cells"):
        _prepare_interfaces(interfaces, np.ones((2, 1, 1), dtype=bool))


def test_unrepresentable_inactive_internal_gap_rejected():
    """Both bounding active elevations must remain fixed."""
    interfaces = np.array([10., 5., 5., 0.])[:, None, None]
    active = np.array([True, False, True])[:, None, None]
    with pytest.raises(UnsupportedMF6Model, match="inactive internal gap"):
        _prepare_interfaces(interfaces, active)


def test_single_layer_excluded_geometry(tmp_path):
    """Read a native DIS with a completely excluded zero-thickness column."""
    sim = flopy.mf6.MFSimulation(sim_name="single", sim_ws=str(tmp_path))
    flopy.mf6.ModflowTdis(sim)
    gwf = flopy.mf6.ModflowGwf(sim, modelname="aquifer")
    flopy.mf6.ModflowIms(sim)
    flopy.mf6.ModflowGwfdis(gwf, nlay=1, nrow=1, ncol=3, delr=10,
                          delc=10, top=[[10., 0., 10.]],
                          botm=[[[0., 0., 0.]]], idomain=[[[1, 0, 1]]])
    flopy.mf6.ModflowGwfic(gwf, strt=5.)
    flopy.mf6.ModflowGwfnpf(gwf, k=1.)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[
        ((0, 0, 0), 5.), ((0, 0, 2), 5.)])
    sim.write_simulation(silent=True)
    loaded = flopy.mf6.MFSimulation.load(sim_ws=str(tmp_path), verbosity_level=0)
    model = import_simulation(loaded)
    assert model.active.sum() == 2
    assert np.all(model.interfaces[:-1] > model.interfaces[1:])
    np.testing.assert_array_equal(model.interfaces[:, 0, 0], [10., 0.])


def test_excluded_geometry_native_parity(tmp_path, solver_device, pinned_mf6):
    """Import inactive zero-thickness cells and compare untouched native MF6."""
    workspace = tmp_path / "mf6"
    sim = make_simulation(workspace)
    dis = sim.get_model("aquifer").dis
    active = np.ones((2, 3, 5), dtype=int)
    active[0, 0, 1] = 0
    active[:, 0, 2] = 0
    top = np.full((3, 5), 10.)
    top[0, 1] = 5.
    top[0, 2] = 0.
    bottom = np.broadcast_to(np.array([5., 0.])[:, None, None], active.shape).copy()
    bottom[:, 0, 2] = 0.
    dis.idomain.set_data(active)
    dis.top.set_data(top)
    dis.botm.set_data(bottom)
    sim.write_simulation(silent=True)
    sim.exe_name = str(pinned_mf6)
    success, output = sim.run_simulation(silent=True, report=True)
    assert success, output
    report = run_mf6_model(workspace, tmp_path / "dw", device=solver_device,
                           reference_head_file=workspace / "aquifer.hds")
    assert len(report["head_comparison"]) == 4
    for comparison in report["head_comparison"]:
        assert comparison["max_abs"] < 1e-6, comparison


@pytest.mark.parametrize("configured_precision", [None, "float32", "float64"])
def test_mf6_import_precision_independent(tmp_path, configured_precision):
    """Run an actual imported model after the process precision is initialised."""
    workspace = tmp_path / "mf6"
    make_simulation(workspace)
    env = os.environ.copy()
    if configured_precision is None:
        env.pop("DARCY_FLOAT", None)
    else:
        env["DARCY_FLOAT"] = configured_precision
    code = (
        "import os, sys, numpy as np\n"
        "from DARCY_WARP_PACKAGE.config import NP_FLOAT\n"
        "from DARCY_WARP_PACKAGE.mf6_import import run_mf6_model\n"
        "configured = os.environ.get('DARCY_FLOAT')\n"
        "dtype = np.dtype(NP_FLOAT)\n"
        "report = run_mf6_model(sys.argv[1], sys.argv[2], device='cpu')\n"
        "assert report['accepted_steps'] == 4\n"
        "assert report['precision_profile'] == 'mixed_fast'\n"
        "assert all(s['head_precision'] == 'float64' and "
        "s['preconditioner_precision'] == 'float32' for s in report['steps'])\n"
        "assert os.environ.get('DARCY_FLOAT') == configured\n"
        "assert np.dtype(NP_FLOAT) == dtype\n"
    )
    subprocess.run([sys.executable, "-c", code, str(workspace),
                    str(tmp_path / "dw")], env=env, check=True)


def test_mf6_import_full_precision_override(tmp_path, solver_device):
    """An explicit full-FP64 profile retains the same accepted physical solution."""
    workspace = tmp_path / "mf6"
    make_simulation(workspace)
    mixed = run_mf6_model(workspace, tmp_path / "mixed", device=solver_device)
    full = run_mf6_model(workspace, tmp_path / "full", device=solver_device,
                         precision="float64", reference_head_file=mixed["head_file"])
    assert full["precision_profile"] == "float64"
    assert all(step["preconditioner_precision"] == "float64" for step in full["steps"])
    assert all(step["head_precision"] == "float64" for step in full["steps"])
    assert all(row["max_abs"] < 1e-6 for row in full["head_comparison"])


def test_uniform_fast_npf_in_fp32_process():
    """The direct confined fast path also owns its FP64 face and head fields."""
    env = dict(os.environ, DARCY_FLOAT="float32")
    code = (
        "import numpy as np\n"
        "from DARCY_WARP_PACKAGE.config import NP_FLOAT\n"
        "from DARCY_WARP_PACKAGE import WarpDarcySolver3D\n"
        "shape = (1, 3, 5)\n"
        "ones = np.ones(shape)\n"
        "fixed = np.zeros(shape, dtype=bool)\n"
        "fixed[:, :, 0] = fixed[:, :, -1] = True\n"
        "expected = np.broadcast_to(np.linspace(10., 6., 5), shape).copy()\n"
        "with WarpDarcySolver3D(nx=5, ny=3, nz=1, dx=10., dz=2., "
        "device='cpu', formulation='mf6_6_6_2_npf', implementation='fast') as model:\n"
        "    model.build_from_K_fields(kx_field=ones, ky_field=ones, kz_field=ones, "
        "active=ones, bc_mask=fixed, bc_values=expected, initial_head=ones*8.)\n"
        "    assert all(face.dtype == np.float64 for face in model._session.faces)\n"
        "    head, info = model.solve()\n"
        "    assert info['converged'], info\n"
        "    assert head.dtype == np.float64\n"
        "    np.testing.assert_allclose(head, expected, atol=1e-6, rtol=0)\n"
        "assert np.dtype(NP_FLOAT) == np.dtype(np.float32)\n"
    )
    subprocess.run([sys.executable, "-c", code], env=env, check=True)



def test_chd_overlaps_native_parity(tmp_path, solver_device, pinned_mf6):
    """MF6 suppresses GHB, WEL, DRN and RIV on prescribed-head cells."""
    workspace = tmp_path / "mf6"
    sim = make_simulation(workspace)
    gwf = sim.get_model("aquifer")
    fixed = (0, 1, 0)
    ghb = gwf.get_package("ghb")
    ghb.maxbound.set_data(3)
    ghb.stress_period_data.set_data({
        0: [((0, 1, 4), 8., 2.), ((0, 1, 4), 9., 1.), (fixed, 4., 500.)],
        2: [],
    })
    wel = gwf.get_package("wel")
    wel.maxbound.set_data(3)
    wel.stress_period_data.set_data({
        0: [((1, 1, 3), -.25), ((1, 1, 3), .05), (fixed, -100.)],
        2: [],
    })
    drn = gwf.get_package("drn")
    drn.maxbound.set_data(2)
    drn.stress_period_data.set_data({0: [((0, 0, 3), 8.5, .1), (fixed, 6., 100.)]})
    riv = gwf.get_package("riv")
    riv.maxbound.set_data(2)
    riv.stress_period_data.set_data({0: [((0, 2, 3), 8.7, .2, 7.), (fixed, 7., 100., 6.)]})
    sim.write_simulation(silent=True)
    sim.exe_name = str(pinned_mf6)
    success, output = sim.run_simulation(silent=True, report=True)
    assert success, output
    report = run_mf6_model(workspace, tmp_path / "dw", device=solver_device,
                           reference_head_file=workspace / "aquifer.hds")
    assert len(report["head_comparison"]) == 4
    assert all(row["max_abs"] < 1e-6 for row in report["head_comparison"])
    periods = report["excluded_boundary_entries"]
    for kind in ("GHB", "WEL", "DRN", "RIV"):
        assert periods[0]["packages"][kind]["chd"]["count"] == 1
        assert periods[1]["packages"][kind]["chd"]["first_cell"] == list(fixed)
    assert "GHB" not in periods[2]["packages"]


def test_inactive_list_entries_require_explicit_opt_in(tmp_path):
    """Never conflate native CHD suppression with invalid inactive list data."""
    sim = make_simulation(tmp_path)
    model = import_simulation(sim)
    model.active[0, 1, 4] = False
    schedules = _schedules(model)
    with pytest.raises(UnsupportedMF6Model, match="IDOMAIN=0"):
        _stress_period(model, schedules, 0)
    diagnostics = {}
    with pytest.warns(UserWarning, match="omitting list entries"):
        forcing = _stress_period(model, schedules, 0, ignore_inactive_boundaries=True,
                                 diagnostics=diagnostics)
    assert forcing["ghb_conductance"] is None
    assert diagnostics["GHB"]["inactive"]["count"] == 2
    assert diagnostics["GHB"]["inactive"]["first_cell"] == [0, 1, 4]


def test_overwrite_replaces_only_run_files(tmp_path, solver_device):
    """Reuse a failed or completed output workspace without deleting unrelated files."""
    workspace = tmp_path / "mf6"
    make_simulation(workspace)
    output = tmp_path / "dw"
    output.mkdir()
    unrelated = output / "keep.txt"
    unrelated.write_text("retain")
    partial = output / "darcywarp.hds.partial"
    partial.write_bytes(b"incomplete")
    with pytest.raises(FileExistsError, match="overwrite=True"):
        run_mf6_model(workspace, output, device=solver_device)
    assert partial.read_bytes() == b"incomplete"
    first = run_mf6_model(workspace, output, device=solver_device, overwrite=True)
    second = run_mf6_model(workspace, output, device=solver_device, overwrite=True)
    assert first["accepted_steps"] == second["accepted_steps"] == 4
    assert unrelated.read_text() == "retain"
    assert not partial.exists()
    with pytest.raises(ValueError, match="Reference head file"):
        _prepare_run_outputs(output, overwrite=True,
                             reference_head_file=output / "darcywarp.hds")
    assert (output / "darcywarp.hds").is_file()
