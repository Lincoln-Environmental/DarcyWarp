"""Public DarcyWarp solves compared with independent, live MODFLOW 6.6.2 inputs."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def native_model(directory, binary, *, shape, dx, dy, top, bottom, k, kz, start,
                 durations=(1.,), active=1, convertible=False, newton=False):
    """Write the DIS geometry and material directly from physical model inputs."""
    import flopy

    sim = flopy.mf6.MFSimulation(sim_name="parity", sim_ws=directory, exe_name=str(binary),
                                 verbosity_level=0)
    sim.simulation_data.float_precision = 17
    sim.simulation_data.float_characters = 25
    flopy.mf6.ModflowTdis(sim, time_units="DAYS", nper=len(durations),
                          perioddata=[(dt, 1, 1.) for dt in durations])
    flopy.mf6.ModflowIms(sim, complexity="COMPLEX", linear_acceleration="BICGSTAB",
                         outer_maximum=200, inner_maximum=200, outer_dvclose=1.e-12,
                         inner_dvclose=1.e-12, rcloserecord=[1.e-12, "STRICT"])
    gwf = flopy.mf6.ModflowGwf(sim, modelname="parity", save_flows=True,
                               newtonoptions="UNDER_RELAXATION" if newton else None)
    nlay, nrow, ncol = shape
    flopy.mf6.ModflowGwfdis(gwf, nlay=nlay, nrow=nrow, ncol=ncol, delr=dx, delc=dy,
                            top=top, botm=bottom, idomain=active)
    flopy.mf6.ModflowGwfic(gwf, strt=start)
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=int(convertible), k=k, k33=kz, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord="parity.hds", budget_filerecord="parity.cbc",
                           saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    return sim, gwf


def native_results(sim, directory, durations, packages):
    """Require convergence and a saved head and CBC record at every requested endpoint."""
    import flopy

    sim.write_simulation(silent=True)
    success, log = sim.run_simulation(silent=True, report=True)
    assert success and any("Normal termination" in line for line in log), log[-20:]
    assert not any("failed to converge" in line.lower() for line in log), log[-20:]
    with flopy.utils.HeadFile(directory / "parity.hds", precision="double") as reader:
        np.testing.assert_allclose(reader.get_times(), np.cumsum(durations), rtol=0, atol=1.e-12)
        heads = reader.get_alldata()
    data = {}
    with flopy.utils.CellBudgetFile(directory / "parity.cbc", precision="double") as reader:
        np.testing.assert_allclose(reader.get_times(), np.cumsum(durations), rtol=0, atol=1.e-12)
        for name in packages:
            periods = []
            for i in range(len(durations)):
                records = reader.get_data(text=name, kstpkper=(0, i))
                assert len(records) == 1, (name, i)
                record = records[0]
                values = record["q"] if record.dtype.names and "q" in record.dtype.names else record
                periods.append(values.copy())
            data[name] = periods
    return heads, data


def assert_budget(actual, native_q, *, atol=1.e-10):
    native_q = np.asarray(native_q)
    assert actual["inflow"] == pytest.approx(np.maximum(native_q, 0.).sum(), rel=0, abs=atol)
    assert actual["outflow"] == pytest.approx(np.maximum(-native_q, 0.).sum(), rel=0, abs=atol)
    assert actual["net"] == pytest.approx(native_q.sum(), rel=0, abs=atol)


def assert_npf(info):
    assert info["converged"], info.get("stop_reason")
    assert info["npf_preconditioner"] == "kcycle"
    assert info["krylov_method"] == "FGMRES"
    assert info["preconditioner_cycles"] > 0


@pytest.fixture(scope="module")
def steady_2d(pinned_mf6, solver_device, tmp_path_factory):
    """Single-layer T=50 has K=5, thickness=10; wells have volumetric input rates."""
    import flopy

    directory = tmp_path_factory.mktemp("steady_2d")
    ny, nx = 5, 6
    wells = [((3, 2), -900.), ((3, 2), 25.), ((1, 4), 30.)]
    sim, gwf = native_model(directory, pinned_mf6, shape=(1, ny, nx), dx=100., dy=100.,
                            top=10., bottom=0., k=5., kz=5., start=12.)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[((0, r, 0), 12.) for r in range(ny)])
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data=[((0, r, c), .0005)
                          for r in range(ny) for c in range(1, nx)])
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[((0, r, c), q) for (r, c), q in wells])
    reference, cbc = native_results(sim, directory, [1.], ["WEL", "RCH"])
    fixed = np.zeros((ny, nx), dtype=np.int32)
    fixed[:, 0] = 1
    with WarpDarcySolver(nx=nx, ny=ny, dx=100., device=solver_device) as solver:
        solver.build_from_fields(T_field=np.full((ny, nx), 50.), R_field=.0005,
                                 active=np.ones_like(fixed), bc_mask=fixed,
                                 bc_values=np.full((ny, nx), 12.), wel_entries=wells)
        head, info = solver.solve(solver="confined_kcycle", implementation="fast",
            initial_head=np.full((ny, nx), 12.), max_cycles=200, min_coarse_cells=2,
            check_every_no=1, nu_coarse=10, smoother="jacobi", fallback_to_pcg=False,
            rel_tol=1.e-12, abs_tol_min=1.e-12)
        assert info["converged"] and info["implementation"] == "fast"
        entries = solver.accepted_wel_rates(head=head, info=info)
    return head, info, entries, reference[0, 0], cbc


def test_confined_2d_drawdown_field_matches_native(steady_2d):
    head, _, _, native, _ = steady_2d
    assert native[3, 2] < native[3, 0] - 1.
    np.testing.assert_allclose(head, native, atol=1.e-6, rtol=0.)


def test_2d_duplicate_well_and_areal_recharge_budgets_match_native(steady_2d):
    _, info, entries, _, cbc = steady_2d
    np.testing.assert_array_equal(entries, cbc["WEL"][0])
    assert_budget(info["source_budgets"]["wel"], cbc["WEL"][0])
    assert_budget(info["source_budgets"]["recharge"], cbc["RCH"][0])


@pytest.fixture(scope="module")
def water_table_history(pinned_mf6, solver_device, tmp_path_factory):
    """Uniform unconfined storage isolates specific yield from intercell flux."""
    import flopy

    directory = tmp_path_factory.mktemp("water_table")
    shape = (6, 6)
    initial = np.full(shape, 5.)
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros_like(active)
    cells = list(np.ndindex(shape))
    sim, gwf = native_model(directory, pinned_mf6, shape=(1, *shape), dx=10., dy=10.,
        top=10., bottom=0., k=1., kz=1., start=5., durations=[1., 1.], convertible=True)
    flopy.mf6.ModflowGwfsto(gwf, iconvert=1, ss=0., sy=.2, transient={0: True}, save_flows=True)
    flopy.mf6.ModflowGwfrch(gwf, stress_period_data={i: [((0, *c), r) for c in cells]
                                                   for i, r in enumerate([.01, .002])})
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={i: [((0, *c), q) for c in cells]
                                                   for i, q in enumerate([-.5, -1.5])})
    reference, cbc = native_results(sim, directory, [1., 1.], ["STO-SY", "WEL", "RCH"])
    with WarpDarcySolver(nx=6, ny=6, dx=10., device=solver_device) as solver:
        solver.build_from_fields(T_field=initial, R_field=0., active=active, bc_mask=fixed,
                                 bc_values=initial)
        heads, info = solver.solve_transient_2d_unconfined(solver="unconfined_picard_kcycle",
            initial_head=initial, recharge_rates=[.01, .002], k_field=np.ones(shape),
            wel_rates=[[(c, -.5) for c in cells], [(c, -1.5) for c in cells]],
            zbot_field=np.zeros(shape), ztop_field=np.full(shape, 10.), sy=.2, ss=0., dt=1.,
            active=active, bc_mask=fixed, bc_values=initial, solve_controls=dict(
                unconfined_head_tol=1.e-10, strict_head_residual_tol=1.e-10, unconfined_relax=1.,
                omega=1., omega_max=1., use_device_transient_fast_path=True,
                transient_face_operator_enabled=True, transient_face_graphs_enabled=True))
        assert all(p["converged"] and p["device_side_picard_fast_path_active"]
                   for p in info["period_infos"])
    return heads, reference[:, 0], cbc


def test_2d_water_table_rise_then_decline_matches_mf6(water_table_history):
    heads, native, _ = water_table_history
    np.testing.assert_allclose(native[:, 0, 0], [5.025, 4.96], rtol=0, atol=1.e-9)
    np.testing.assert_allclose(heads, native, rtol=0., atol=1.e-8)


def test_2d_unconfined_storage_release_matches_native_specific_yield(water_table_history):
    heads, _, cbc = water_table_history
    previous = np.full((6, 6), 5.)
    for head, release in zip(heads, cbc["STO-SY"], strict=True):
        np.testing.assert_allclose(-.2 * 100. * (head - previous), release[0], atol=3.e-8, rtol=0.)
        previous = head


@pytest.fixture(scope="module")
def layered_history(pinned_mf6, solver_device, tmp_path_factory):
    """Rectangular cells with a missing top cell and independently scheduled pumping."""
    import flopy

    directory = tmp_path_factory.mktemp("layered_history")
    shape = (2, 2, 3)
    active = np.ones(shape, dtype=np.int32)
    active[0, 0, 2] = 0
    fixed = np.zeros_like(active)
    fixed[:, :, 0] = 1
    dt = [.75, 2.25, .5]
    recharge = [.05, -.01, .03]
    well = (1, 1, 2)
    schedule = [[(well, -4.), (well, 1.)], [(well, 2.)], [(well, -3.)]]
    sim, gwf = native_model(directory, pinned_mf6, shape=shape, dx=2., dy=3.,
        top=10., bottom=[5., 0.], k=5., kz=.7, start=8., durations=dt, active=active)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[((k, r, 0), 8.)
                                                   for k in range(2) for r in range(2)])
    flopy.mf6.ModflowGwfsto(gwf, iconvert=0, ss=.01, sy=0., transient={0: True}, save_flows=True)
    receivers = [(0, 0, 1), (1, 0, 2), (0, 1, 1), (0, 1, 2)]
    flopy.mf6.ModflowGwfrch(gwf, fixed_cell=True, stress_period_data={
        i: [(c, rate) for c in receivers] for i, rate in enumerate(recharge)})
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=dict(enumerate(schedule)))
    reference, cbc = native_results(sim, directory, dt, ["WEL", "RCH", "STO-SS", "CHD"])
    with WarpDarcySolver3D(nx=3, ny=2, nz=2, dx=2., dy=3., dz=5., device=solver_device,
                          solver="kcycle", implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_K_fields(kx_field=np.full(shape, 5.), ky_field=np.full(shape, 5.),
            kz_field=np.full(shape, .7), active=active, bc_mask=fixed, bc_values=np.full(shape, 8.),
            initial_head=np.full(shape, 8.), rhs=np.zeros(shape, order="F"))
        heads, info = solver.solve_transient_npf(dt=dt, head_prev=np.full(shape, 8.),
            ss=.01, sy=0., wel_rates=schedule, recharge_rates=recharge, top_elevation=10.,
            npf_preconditioner="kcycle")
        assert info["accepted_steps"] == len(dt)
        for period in info["periods"]:
            assert_npf(period)
        rates = [solver.accepted_wel_rates(head=h, info=p) for h, p in zip(heads, info["periods"], strict=True)]
    return heads, info, rates, reference, cbc, active != 0, fixed != 0, dt


def test_layered_history_heads_match_native_at_each_physical_time(layered_history):
    heads, info, _, native, _, active, _, dt = layered_history
    assert [row["dt"] for row in info["periods"]] == dt
    np.testing.assert_allclose(heads[:, active], native[:, active], rtol=0., atol=1.e-8)


def test_scheduled_native_wells_and_lower_layer_recharge(layered_history):
    _, info, rates, _, cbc, _, _, _ = layered_history
    for i, period in enumerate(info["periods"]):
        np.testing.assert_array_equal(rates[i], cbc["WEL"][i])
        assert_budget(period["source_budgets"]["recharge"], cbc["RCH"][i])


def test_three_dimensional_storage_uses_actual_volume_and_duration(layered_history):
    heads, _, _, _, cbc, active, fixed, dt = layered_history
    previous = np.full(active.shape, 8.)
    for i, head in enumerate(heads):
        free = active & ~fixed
        release = -.01 * (2. * 3. * 5.) * (head[free] - previous[free]) / dt[i]
        np.testing.assert_allclose(release, cbc["STO-SS"][i][free], rtol=0., atol=1.e-8)
        previous = head


def test_accepted_history_gross_volumes_match_native_time_integration(layered_history):
    _, info, _, _, cbc, _, _, dt = layered_history
    for source, name in (("wel", "WEL"), ("recharge", "RCH")):
        volumes = []
        for i, period in enumerate(info["periods"]):
            volume = cbc[name][i] * dt[i]
            volumes.append(volume)
            assert_budget(period["source_volume_budgets"][source], volume)
        assert_budget(info["accepted_exchange_volume_totals"]["sources"][source], np.concatenate(volumes))


@pytest.fixture(scope="module")
def switching(pinned_mf6, solver_device, tmp_path_factory):
    """A buried cell is attached to three fixed faces and several switching packages."""
    import flopy

    directory = tmp_path_factory.mktemp("switching")
    shape = (2, 3, 4)
    cell = (1, 2, 2)
    active = np.ones(shape, dtype=np.int32)
    active[0, 2, 2] = 0
    fixed = active.copy()
    fixed[cell] = 0
    wells = [4., -6., -40., 4.]
    rivers = [[(2., 8., 4.), (3., 7.5, 5.)], [(2., 8., 4.), (3., 7.5, 5.)],
              [(2., 8., 4.), (3., 7.5, 5.), (.5, 6., 3.)], [(2., 8.5, 4.), (3., 7.9, 5.)]]
    sim, gwf = native_model(directory, pinned_mf6, shape=shape, dx=1., dy=1., top=10.,
                            bottom=[9., 8.], k=1., kz=1., start=8., durations=[1.]*4, active=active)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[(tuple(c), 8.) for c in np.argwhere(fixed)])
    flopy.mf6.ModflowGwfghb(gwf, stress_period_data=[(cell, 8., 1.)])
    flopy.mf6.ModflowGwfdrn(gwf, stress_period_data=[(cell, 7., .7)])
    flopy.mf6.ModflowGwfriv(gwf, stress_period_data={
        i: [(cell, stage, c, bed) for c, stage, bed in period] for i, period in enumerate(rivers)})
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data={i: [(cell, q)] for i, q in enumerate(wells)})
    flopy.mf6.ModflowGwfrch(gwf, fixed_cell=True, stress_period_data=[(cell, .1)])
    ref, cbc = native_results(sim, directory, [1.]*4, ["DRN", "RIV", "GHB", "CHD"])
    ghb = np.zeros(shape)
    ghb[cell] = 1.
    heads, infos, entries = [], [], []
    with WarpDarcySolver3D(nx=4, ny=3, nz=2, dx=1., dy=1., dz=1., device=solver_device,
                          solver="kcycle", implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
            active=active, bc_mask=fixed, bc_values=np.full(shape, 8.), initial_head=np.full(shape, 8.),
            R_field=.1, ghb_conductance=ghb, ghb_head=np.full(shape, 8.),
            drn_entries=[(cell, .7, 7.)])
        for i, q in enumerate(wells):
            solver.update_wel(entries=[(cell, q)])
            solver.update_riv_entries(entries=[(cell, *entry) for entry in rivers[i]])
            head, info = solver.solve(top_elevation=10., npf_preconditioner="kcycle",
                                      npf_roundoff_acceptance=True)
            assert_npf(info)
            heads.append(head)
            infos.append(info)
            solver.update_initial_head(initial_head=head)
        for h, info in zip(heads, infos, strict=True):
            entries.append(solver.accepted_boundary_entry_rates(package="RIV", head=h, info=info))
    return np.array(heads)[:, *cell], infos, entries, ref[:, *cell], cbc


def test_buried_cell_switching_heads_match_four_independent_roots(switching):
    heads, _, _, native, _ = switching
    # In period 2 all rivers supply constant leakage and the drain is closed.
    analytical = [79.5 / 9.7, 69.5 / 9.7, 9.1 / 4., 81.7 / 9.7]
    np.testing.assert_allclose(native, analytical, atol=2.e-10, rtol=0.)
    np.testing.assert_allclose(heads, analytical, atol=2.e-12, rtol=0.)


def test_historical_river_entry_outputs_match_native_after_replacement(switching):
    _, _, entry_rates, _, cbc = switching
    assert [len(rates) for rates in entry_rates] == [2, 2, 3, 2]
    for outflow, inflow in zip(entry_rates, cbc["RIV"], strict=True):
        np.testing.assert_allclose(outflow, -inflow, atol=2.e-10, rtol=0.)


def test_drains_shut_off_and_reopen_and_rivers_retain_gross_directions(switching):
    _, infos, _, _, cbc = switching
    assert cbc["DRN"][2][0] == 0. and cbc["DRN"][3][0] < 0.
    assert np.min(cbc["RIV"][3]) < 0. < np.max(cbc["RIV"][3])
    for i, info in enumerate(infos):
        assert_budget(info["drn_budget"], cbc["DRN"][i])
        assert_budget(info["riv_budget"], cbc["RIV"][i])


def test_three_dimensional_reservoir_exchange_matches_native(switching):
    _, infos, _, _, cbc = switching
    for i, info in enumerate(infos):
        for name in ("GHB", "CHD"):
            actual = info[name.lower() + "_budget"]
            q = cbc[name][i]
            assert actual[name.lower() + "_inflow"] == pytest.approx(np.maximum(q, 0.).sum(), abs=2.e-10)
            assert actual[name.lower() + "_outflow"] == pytest.approx(np.maximum(-q, 0.).sum(), abs=2.e-10)


@pytest.fixture(scope="module")
def convertible(pinned_mf6, solver_device, tmp_path_factory):
    """Two prescribed stages force an interior Newton/upstream face-law test."""
    import flopy

    directory = tmp_path_factory.mktemp("convertible")
    shape = (1, 1, 3)
    sim, gwf = native_model(directory, pinned_mf6, shape=shape, dx=2., dy=3.,
                           top=10., bottom=0., k=5., kz=1., start=6., convertible=True, newton=True)
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=[((0, 0, 0), 9.), ((0, 0, 2), 4.)])
    flopy.mf6.ModflowGwfwel(gwf, stress_period_data=[((0, 0, 1), -2.)])
    native, cbc = native_results(sim, directory, [1.], ["CHD"])
    with WarpDarcySolver3D(nx=3, ny=1, nz=1, dx=2., dy=3., dz=10., device=solver_device,
                          solver="kcycle", implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_K_fields(kx_field=np.full(shape, 5.), ky_field=np.full(shape, 5.),
            kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32),
            bc_mask=np.array([[[1, 0, 1]]]), bc_values=np.array([[[9., 0., 4.]]]),
            initial_head=np.array([[[9., 6., 4.]]]), wel_entries=[((0, 0, 1), -2.)])
        head, info = solver.solve(unconfined=True, top_elevation=10., npf_preconditioner="kcycle")
        assert_npf(info)
    return head.ravel(), info, native.ravel(), cbc["CHD"][0]


def test_convertible_well_drawdown_matches_mf6_newton(convertible):
    head, _, native, _ = convertible
    np.testing.assert_allclose(head, native, atol=2.e-8, rtol=0.)
    assert 4. < head[1] < 9.


def test_convertible_upstream_face_flows_balance_well_withdrawal(convertible):
    h, info, _, q = convertible
    # Saturated thickness on each horizontal connection belongs to its upstream cell.
    # All three cells are within the linear interior of the MF6 Newton saturation
    # ramp. Its end smoothing (omega = 1e-6) slightly changes the interior slope.
    omega = 1.e-6
    saturated_thickness = (h - .5 * 10. * omega) / (1. - omega)
    left = 5. * 3. / 2. * saturated_thickness[0] * (h[0] - h[1])
    right = 5. * 3. / 2. * saturated_thickness[1] * (h[2] - h[1])
    np.testing.assert_allclose([left, right], q, atol=2.e-7, rtol=0.)
    assert info["chd_budget"]["chd_inflow"] == pytest.approx(q[0], abs=2.e-7)
    assert info["chd_budget"]["chd_outflow"] == pytest.approx(-q[1], abs=2.e-7)
    assert sum(q) == pytest.approx(2., abs=1.e-8)


def test_multilayer_convertible_newton_matches_native_mf6(tmp_path, solver_device, pinned_mf6):
    """Verify the full 3D correction operator and the independent MF6 endpoint."""
    import flopy
    import warp as wp
    from flopy.mf6.utils import MfGrdFile

    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model, independent_reference
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace, NPF662NewtonControls
    from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleWorkspace, NPF662KCycleControls

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
    # Build the same physical model through the production public API. The
    # maintenance release deliberately does not include the MF6 importer.
    interfaces = np.concatenate((top[None], bottom), axis=0)
    fixed = np.zeros(shape, dtype=int)
    values = np.zeros(shape)
    for cell, head_value in chd:
        fixed[cell] = 1
        values[cell] = head_value
    ghb_c = np.zeros(shape)
    ghb_h = np.zeros(shape)
    ghb_c[0, 1, 4], ghb_h[0, 1, 4] = 3., 26.
    ghb_c[2, 3, 6], ghb_h[2, 3, 6] = 5., 18.
    source = np.zeros(shape)
    receiving = (active[irch, j, i] != 0) & (fixed[irch, j, i] == 0)
    source[irch, j, i] = np.where(receiving, 0.001*20.*15., 0.)
    selected = (active != 0) & (types != 0)
    thickness = (interfaces[:-1]-interfaces[1:])[selected]
    cx = k[selected]*thickness*15./20.
    cy = 0.6*k[selected]*thickness*20./15.
    shift = 0.08 * float(np.median(0.5*cx + 0.5*cy))
    controls = NPF662NewtonControls(
        steady_initial_shift=shift, max_head_change_m=5.*float(np.median(thickness)),
        inner_forcing=0.01, tiled_reductions=True, reuse_trial_state=True)
    multigrid = NPF662KCycleControls(levels=5, coarse_sweeps=2,
        cached_line_factors=True, device_hierarchy_build=True,
        preconditioner_precision="float32", coefficient_state="current")
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=20., dy=15.,
                           device=solver_device, formulation="mf6_6_6_2_npf",
                           implementation="fast", solver="kcycle") as solver:
        solver.build_from_layers(z_interfaces=interfaces, kx_field=k,
            ky_field=0.6*k, kz_field=0.05*k, active=active, bc_mask=fixed,
            bc_values=values, rhs=source, initial_head=initial,
            ghb_conductance=ghb_c, ghb_head=ghb_h,
            drn_entries=[((0, 1, 3), 1., 27.), ((0, 3, 5), 2., 28.)])
        case = case_from_model(model=solver, top=0., cell_type=types)
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
                controls=multigrid,
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

        head, info = solver.solve(unconfined=True, npf_cell_type=types,
            npf_newton_controls=controls, npf_kcycle_controls=multigrid,
            max_outer=80, max_cycles=500, krylov_restart=16,
            npf_final_relative_tolerance=1e-7)
        assert_npf(info)
        assert info["physical_acceptance"]["independent_accepted"]
        _, physical = reference.evaluate(head=head, relative_tolerance=1e-7)
        with flopy.utils.HeadFile(directory / "aquifer.hds", precision="double") as reader:
            native = reader.get_data()
        selected = active != 0
        np.testing.assert_allclose(head[selected], native[selected], atol=1e-6, rtol=0.)
        # Compare real native face and package budgets. The GRB may use full
        # user-grid CSR or reduced-node CSR, so explicitly map both.
        grid = MfGrdFile(next(directory.glob("*.grb")))
        mapping = (np.arange(active.size) if grid.ia.size == active.size+1
                   else np.flatnonzero(selected))
        assert grid.ia.size == mapping.size+1
        rows = np.repeat(mapping, np.diff(grid.ia))
        columns = mapping[grid.ja]
        faces = rows != columns
        keys = rows[faces]*active.size + columns[faces]
        order = np.argsort(keys)
        expected_keys = reference.left*active.size + reference.right
        indices = np.searchsorted(keys[order], expected_keys)
        np.testing.assert_array_equal(keys[order][indices], expected_keys)
        with flopy.utils.CellBudgetFile(directory / "aquifer.cbc", precision="double") as reader:
            native_faces = reader.get_data(text="FLOW-JA-FACE")[-1].ravel()[faces]
            np.testing.assert_allclose(physical["flux"], -native_faces[order][indices],
                                       atol=1e-5, rtol=1e-9)
            for package, field, sign in (("CHD", "chd_cell_rate", 1.),
                                         ("GHB", "ghb_outflow", -1.),
                                         ("DRN", "drn_outflow", -1.)):
                records = reader.get_data(text=package)[-1]
                native_grid = np.bincount(records["node"]-1, weights=records["q"],
                                          minlength=active.size)
                np.testing.assert_allclose(sign*physical[field].ravel(), native_grid,
                                           atol=1e-5, rtol=1e-9)
