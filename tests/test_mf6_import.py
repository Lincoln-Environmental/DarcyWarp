# SPDX-License-Identifier: AGPL-3.0-or-later
"""Regression tests for the MF6 input adapter and its binary output."""

import os
import subprocess
import sys

import numpy as np
import pytest

flopy = pytest.importorskip("flopy")

from DARCY_WARP_PACKAGE.mf6_import import (
    UnsupportedMF6Model, import_simulation, load_mf6_workspace, run_mf6_model,
    timestep_lengths, _schedules, _stress_period, _add_array_recharge,
    _prepare_interfaces, _prepare_run_outputs, _import_solver_options,
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


def test_mxiter_after_ims_keeps_archived_inputs(tmp_path):
    """FloPy can load a native-valid trailing MXITER without editing input."""
    workspace = tmp_path / "archive"
    original = make_simulation(workspace=workspace)
    original.get_model().npf.k.store_as_external_file(external_file_path="conductivity.txt")
    original.write_simulation(silent=True)
    namefile = workspace / "mfsim.nam"
    lines = namefile.read_text().splitlines(keepends=True)
    ims = next(index for index, line in enumerate(lines) if line.lstrip().upper().startswith("IMS6 "))
    lines.insert(ims + 1, "  MXITER 1\n")
    namefile.write_text("".join(lines))
    archived = namefile.read_bytes()

    with pytest.warns(UserWarning, match="MXITER before IMS6"):
        loaded = import_simulation(simulation=load_mf6_workspace(directory=workspace))
    assert namefile.read_bytes() == archived
    np.testing.assert_array_equal(loaded.kx[loaded.active], 2.)


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



def test_import_benchmark_profile_and_overrides(tmp_path):
    """The importer must use the checked-in qualification settings, not ad hoc defaults."""
    import json
    from pathlib import Path
    from dataclasses import asdict

    model = import_simulation(make_simulation(tmp_path))
    profile, options = _import_solver_options(model=model, device="cuda:0",
                                             precision="mixed_fast", solver_options=None)
    assert profile == "npf_ghb_five_level_controls.json"
    root = Path(__file__).resolve().parents[1]
    expected = json.loads((root / "working_tests" / profile).read_text())
    for name, value in expected["solve"].items():
        assert options[name] == value
    for name, value in expected["newton"].items():
        assert asdict(options["npf_newton_controls"])[name] == value
    for name, value in expected["multigrid"].items():
        assert asdict(options["npf_kcycle_controls"])[name] == value
    _, overridden = _import_solver_options(
        model=model, device="cpu", precision="float64",
        solver_options={"max_outer": 100, "krylov_restart": 16})
    assert overridden["max_outer"] == 100
    assert overridden["krylov_restart"] == 16
    assert not overridden["npf_device_native"]
    assert overridden["npf_kcycle_controls"].preconditioner_precision == "float64"


def test_import_progress_and_effective_controls(tmp_path, solver_device, capsys):
    """Expose progress and actual native settings without altering acceptance."""
    workspace = tmp_path / "mf6"
    make_simulation(workspace)
    report = run_mf6_model(workspace, tmp_path / "dw", device=solver_device, verbose=True)
    output = capsys.readouterr().out
    for phrase in ("Reading MF6 workspace", "Imported model", "Benchmark profile",
                   "Operator ready", "Solving step", "Preparing Newton workspace",
                   "Preparing K-cycle hierarchy", "Solver setup complete",
                   "Final physical acceptance", "Step accepted", "Run complete"):
        assert phrase in output
    assert report["benchmark_profile"] == "npf_ghb_five_level_controls.json"
    assert report["resolved_solver_controls"]["max_outer"] == 80
    for step in report["steps"]:
        effective = step["effective_solver_settings"]
        assert effective["restart"] == 8
        assert effective["krylov_budget_per_outer"] == 4000
        assert effective["max_outer"] == 80
        assert effective["convergence"]["relative_cell_imbalance"] == 1e-7
    quiet = run_mf6_model(workspace, tmp_path / "quiet", device=solver_device,
                          verbose=False)
    assert quiet["accepted_steps"] == report["accepted_steps"]
    assert "[DarcyWarp MF6" not in capsys.readouterr().out



def test_failed_import_writes_solver_diagnostics(tmp_path, solver_device):
    """A rejected solve leaves diagnostic JSON but no completed head history."""
    import json

    workspace = tmp_path / "mf6"
    make_simulation(workspace)
    output = tmp_path / "dw"
    with pytest.raises(RuntimeError, match="diagnostics"):
        run_mf6_model(workspace, output, device=solver_device,
                      solver_options={"max_outer": 0}, verbose=False)
    report = json.loads((output / "darcywarp_report.json").read_text())
    assert report["converged"] is False
    assert report["accepted_steps"] == 0
    assert report["failed_step"]["kper"] == 1
    assert "final_metrics" in report["failed_step"]
    assert not (output / "darcywarp.hds").exists()


def test_multilayer_convertible_newton_matches_native_mf6(tmp_path, solver_device, pinned_mf6):
    """Verify the full 3D correction operator and the independent MF6 endpoint."""
    import json
    import warp as wp

    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model, independent_reference
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleWorkspace
    from working_tests.benchmark_regional_mf6 import audit

    # A genuinely layered problem: nonparallel interfaces, anisotropic K,
    # two convertible layers, one confined layer, irregular active domain,
    # vertical exchange and head-dependent boundary conditions.
    shape = (3, 5, 9)
    nz, ny, nx = shape
    j, i = np.indices((ny, nx))
    top = 35. + 0.2 * i + 0.1 * j
    bottom = np.stack((top - 14. - 0.05 * j,
                       top - 28. - 0.1 * i,
                       top - 46. - 0.1 * j))
    active = np.ones(shape, dtype=int)
    active[:, 2, 4] = 0
    active[2, 0, 5] = 0
    types = np.ones(shape, dtype=int)
    types[2] = 0
    k = np.stack((2. + 0.7 * i + 0.1 * j,
                  1. + 0.4 * j + 0.2 * i,
                  5. + 0.3 * i + 0.2 * j))
    initial = np.stack((top - 5. - 0.3 * i,
                        top - 17. - 0.2 * j,
                        top - 23. - 0.1 * i))
    directory = tmp_path / "mf6_layered"
    sim = flopy.mf6.MFSimulation(sim_name="layered", sim_ws=str(directory),
                                 exe_name=str(pinned_mf6))
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=1, perioddata=[(1., 1, 1.)])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="aquifer", save_flows=True,
                              newtonoptions="UNDER_RELAXATION")
    flopy.mf6.ModflowIms(sim, complexity="COMPLEX", outer_maximum=200,
        inner_maximum=500, outer_dvclose=1e-10, inner_dvclose=1e-12,
        rcloserecord=[1e-9, "STRICT"], linear_acceleration="BICGSTAB")
    flopy.mf6.ModflowGwfdis(gwf, nlay=nz, nrow=ny, ncol=nx,
        top=top, botm=bottom, idomain=active, delr=20., delc=15., length_units="METERS")
    flopy.mf6.ModflowGwfic(gwf, strt=initial)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=types, k=k, k22=0.6*k,
                            k33=0.05*k, save_flows=True, save_saturation=True)
    chd = [((layer, row, col), float(value))
           for layer, heads in enumerate(((30., 27.), (23., 19.), (20., 17.)))
           for row in range(ny) for col, value in zip((0, nx-1), heads)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=chd, save_flows=True)
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=[
        ((0, 1, 4), 26., 3.), ((2, 3, 6), 18., 5.)], save_flows=True)
    flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=[
        ((0, 1, 3), 27., 1.), ((0, 3, 5), 28., 2.)], save_flows=True)
    irch = np.zeros((ny, nx), dtype=int)
    irch[1, 1] = 1
    flopy.mf6.ModflowGwfrcha(gwf, fixed_cell=True, irch=irch,
                            recharge=0.001, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="aquifer.hds",
        budget_filerecord="aquifer.cbc", saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    sim.write_simulation(silent=True)
    success, messages = sim.run_simulation(silent=True, report=True)
    assert success, messages

    # Inspect the real correction operator at the original heads, where the
    # saturation derivative is nonzero, rather than at an already converged
    # or confined linear solution.
    model = import_simulation(simulation=load_mf6_workspace(directory=directory))
    profile, options = _import_solver_options(model=model, device=solver_device,
                                              precision="mixed_fast", solver_options=None)
    shift = options["npf_newton_controls"].steady_initial_shift
    assert shift > 0.
    forcing = _stress_period(model=model, schedules=_schedules(model=model), kper=0)
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=model.dx, dy=model.dy,
                           device=solver_device, formulation="mf6_6_6_2_npf",
                           implementation="fast", solver="kcycle") as solver:
        solver.build_from_layers(z_interfaces=model.interfaces, kx_field=model.kx,
            ky_field=model.ky, kz_field=model.kz, active=model.active,
            initial_head=model.initial, **forcing)
        case = case_from_model(model=solver, top=0., cell_type=model.cell_type)
        reference, unused = independent_reference(model=solver, case=case)
        state = NPF662Workspace(model=solver, case=case, restart=16,
                                krylov_budget=500, reference=reference)
        hierarchy = None
        try:
            start = state.head.numpy().reshape(shape).copy()
            jacobian = reference.jacobian(head=start)
            free = reference.free.ravel()
            diagonal = jacobian.diagonal()
            _, physical = reference.evaluate(head=start)
            # Prove the case exercises the missing Newton derivative.
            assert np.max(np.abs(diagonal[free] -
                          physical["principal_diagonal"].ravel()[free])) > 0.1
            original_residual = state.residual.numpy().copy()
            hierarchy = NPF662KCycleWorkspace(faces=solver._session.faces,
                active=case.active, fixed=case.fixed, device=solver_device,
                controls=options["npf_kcycle_controls"],
                storage_diag=state.operator_tangent.numpy().reshape(shape))
            direction = np.where(free, np.linspace(-0.4, 0.7, state.n), 0.)
            x = wp.array(direction, dtype=wp.float64, device=solver_device)
            y = wp.zeros_like(x)
            state.mode.fill_(1)
            # Check both the regularised correction and its removal.
            for value in (shift, 0.):
                state.set_linear_shift(value=value)
                hierarchy.refresh_current_operator(workspace=state, mode=1)
                actual = hierarchy.levels[0]["diag"].numpy().ravel()
                np.testing.assert_allclose(actual[free], diagonal[free]+value,
                                           rtol=1e-12, atol=1e-10)
                state.matvec(x, y, y, 1., 0.)
                expected = (jacobian @ direction + value*direction)/state.scale.numpy()
                np.testing.assert_allclose(y.numpy()[free], expected[free],
                                           rtol=1e-12, atol=1e-12)
                np.testing.assert_array_equal(state.residual.numpy(), original_residual)
            # A dry but vertically connected convertible cell has a physical
            # head, not an arbitrary zero row. The damping must not clip it.
            cell = (0, 1, 2)
            delta = np.zeros(shape)
            target = case.bottoms[cell] - 1.
            delta[cell] = target - start[cell]
            state.delta.assign(delta.ravel())
            state.damp_drying = True
            state.candidate(alpha=1.)
            assert state.trial.numpy().reshape(shape)[cell] == pytest.approx(target)
        finally:
            if hierarchy is not None:
                hierarchy.close()
            state.close()

    output = tmp_path / "layered_dw"
    # Native equation parity must pass before attempting the DarcyWarp solve.
    tolerances = dict(head_gate=1e-6, head_rmse_gate=1e-7,
                      saturation_head_gate=1e-6, flow_atol=1e-6, write_maps=False)
    audit(archive=directory, directory=directory, output=output,
          device=solver_device, **tolerances)
    report = run_mf6_model(directory, output, device=solver_device,
                           reference_head_file=directory / "aquifer.hds", verbose=False)
    assert report["accepted_steps"] == 1
    assert report["head_comparison"][0]["max_abs"] < 1e-6
    audit(archive=directory, directory=directory, output=output,
          device=solver_device, use_darcywarp_heads=True, **tolerances)
    receipt = json.loads((output / "solution.json").read_text())
    assert receipt["gates_passed"] and receipt["cpu"]["independent_accepted"]
