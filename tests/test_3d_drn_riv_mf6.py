"""Matched two-cell MF6 checks for native confined 3D DRN and RIV."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.project_base import require_mf6
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def _run_mf6_package_case(*, workspace, package):
    """Run one two-cell confined model with an MF6 DRN or RIV boundary.

    :param workspace: Writable directory for the simulation files.
    :param package: Boundary package name, either ``DRN`` or ``RIV``.
    :return: MF6 free-cell head and signed package rate from its cell budget.
    :raises RuntimeError: If MF6 fails to terminate normally.
    """
    import flopy

    name = "dw_drn_riv"
    simulation = flopy.mf6.MFSimulation(sim_name=name, exe_name=str(require_mf6()),
        version="mf6", sim_ws=str(workspace))
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=1,
        perioddata=[(1.0, 1, 1.0)])
    model = flopy.mf6.ModflowGwf(simulation, modelname=name, save_flows=True)
    flopy.mf6.ModflowIms(simulation, complexity="COMPLEX", linear_acceleration="BICGSTAB",
        outer_maximum=100, outer_dvclose=1.0e-12, inner_maximum=100,
        inner_dvclose=1.0e-12, rcloserecord=[1.0e-12, "strict"])
    flopy.mf6.ModflowGwfdis(model, nlay=1, nrow=1, ncol=2, delr=1.0, delc=1.0,
        top=10.0, botm=0.0)
    flopy.mf6.ModflowGwfic(model, strt=8.0)
    flopy.mf6.ModflowGwfnpf(model, icelltype=0, k=1.0, k33=1.0, save_flows=True)
    flopy.mf6.ModflowGwfchd(model, maxbound=1,
        stress_period_data=[((0, 0, 1), 8.0)], save_flows=True)
    flopy.mf6.ModflowGwfwel(model, maxbound=1,
        stress_period_data=[((0, 0, 0), 3.0)], save_flows=True)
    if package == "DRN":
        flopy.mf6.ModflowGwfdrn(model, maxbound=1,
            stress_period_data=[((0, 0, 0), 4.0, 2.0)], save_flows=True)
    else:
        flopy.mf6.ModflowGwfriv(model, maxbound=1,
            stress_period_data=[((0, 0, 0), 7.0, 2.0, 2.0)], save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc", saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    simulation.write_simulation()
    success, output = simulation.run_simulation(silent=True, report=True)
    if not success or not any("Normal termination" in line for line in output):
        raise RuntimeError(f"MF6 failed: {output}")
    head_file = flopy.utils.HeadFile(str(workspace / f"{name}.hds"))
    budget_file = flopy.utils.CellBudgetFile(str(workspace / f"{name}.cbc"), precision="double")
    record = budget_file.get_data(text=package, totim=1.0)[0]
    return float(head_file.get_data(totim=1.0)[0, 0, 0]), float(record["q"].sum())


def _run_mf6_transient_package_case(*, workspace, package):
    """Run a one-cell two-period MF6 STO model with a switching package.

    :param workspace: Writable directory for the simulation files.
    :param package: Boundary package name, either ``DRN`` or ``RIV``.
    :return: Time-indexed heads and signed package rates from the cell budget.
    :raises RuntimeError: If MF6 fails to terminate normally.
    """
    import flopy

    name = "dwdrntr"
    simulation = flopy.mf6.MFSimulation(sim_name=name, exe_name=str(require_mf6()),
        version="mf6", sim_ws=str(workspace))
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=2,
        perioddata=[(1.0, 1, 1.0), (1.0, 1, 1.0)])
    model = flopy.mf6.ModflowGwf(simulation, modelname=name, save_flows=True)
    flopy.mf6.ModflowIms(simulation, complexity="COMPLEX", linear_acceleration="BICGSTAB",
        outer_maximum=100, outer_dvclose=1.0e-12, inner_maximum=100,
        inner_dvclose=1.0e-12, rcloserecord=[1.0e-12, "strict"])
    flopy.mf6.ModflowGwfdis(model, nlay=1, nrow=1, ncol=1, delr=1.0, delc=1.0,
        top=10.0, botm=0.0)
    flopy.mf6.ModflowGwfic(model, strt=0.0)
    flopy.mf6.ModflowGwfnpf(model, icelltype=0, k=1.0, k33=1.0, save_flows=True)
    flopy.mf6.ModflowGwfsto(model, iconvert=0, ss=1.0, sy=0.0,
        steady_state={}, transient={0: True}, save_flows=True)
    flopy.mf6.ModflowGwfwel(model, maxbound=1,
        stress_period_data={0: [((0, 0, 0), 30.0)], 1: [((0, 0, 0), 0.0)]}, save_flows=True)
    if package == "DRN":
        flopy.mf6.ModflowGwfdrn(model, maxbound=1,
            stress_period_data=[((0, 0, 0), 2.0, 1.0)], save_flows=True)
    else:
        flopy.mf6.ModflowGwfriv(model, maxbound=1,
            stress_period_data=[((0, 0, 0), 3.0, 1.0, 1.0)], save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc", saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    simulation.write_simulation()
    success, output = simulation.run_simulation(silent=True, report=True)
    if not success or not any("Normal termination" in line for line in output):
        raise RuntimeError(f"MF6 failed: {output}")
    head_file = flopy.utils.HeadFile(str(workspace / f"{name}.hds"))
    budget_file = flopy.utils.CellBudgetFile(str(workspace / f"{name}.cbc"), precision="double")
    times = head_file.get_times()
    heads = np.asarray([head_file.get_data(totim=time)[0, 0, 0] for time in times])
    rates = np.asarray([budget_file.get_data(text=package, totim=time)[0]["q"].sum()
        for time in times])
    return heads, rates


def _run_mf6_layered_mixed_case(*, workspace):
    """Run a two-layer MF6 case with co-located DRN/RIV capability.

    :param workspace: Writable directory for the simulation files.
    :return: Full head grid and individual signed DRN/RIV rates.
    :raises RuntimeError: If MF6 fails to terminate normally.
    """
    import flopy

    name = "dwmix3d"
    simulation = flopy.mf6.MFSimulation(sim_name=name, exe_name=str(require_mf6()),
        version="mf6", sim_ws=str(workspace))
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=1,
        perioddata=[(1.0, 1, 1.0)])
    model = flopy.mf6.ModflowGwf(simulation, modelname=name, save_flows=True)
    flopy.mf6.ModflowIms(simulation, complexity="COMPLEX", linear_acceleration="BICGSTAB",
        outer_maximum=100, outer_dvclose=1.0e-12, inner_maximum=100,
        inner_dvclose=1.0e-12, rcloserecord=[1.0e-12, "strict"])
    flopy.mf6.ModflowGwfdis(model, nlay=2, nrow=1, ncol=3, delr=1.0, delc=1.0,
        top=12.0, botm=[6.0, 0.0])
    flopy.mf6.ModflowGwfic(model, strt=11.0)
    flopy.mf6.ModflowGwfnpf(model, icelltype=0, k=np.array([[[10.0] * 3], [[2.0] * 3]]),
        k33=np.array([[[2.0] * 3], [[1.0] * 3]]), save_flows=True)
    flopy.mf6.ModflowGwfchd(model, maxbound=4,
        stress_period_data=[((layer, 0, col), 11.0)
            for layer in range(2) for col in (0, 2)], save_flows=True)
    flopy.mf6.ModflowGwfwel(model, maxbound=2,
        stress_period_data=[((0, 0, 1), 2.0), ((1, 0, 1), -1.0)], save_flows=True)
    flopy.mf6.ModflowGwfdrn(model, maxbound=1,
        stress_period_data=[((1, 0, 1), 9.0, 1.0)], save_flows=True)
    flopy.mf6.ModflowGwfriv(model, maxbound=1,
        stress_period_data=[((0, 0, 1), 10.5, 2.0, 9.5)], save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc", saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    simulation.write_simulation()
    success, output = simulation.run_simulation(silent=True, report=True)
    if not success or not any("Normal termination" in line for line in output):
        raise RuntimeError(f"MF6 failed: {output}")
    head_file = flopy.utils.HeadFile(str(workspace / f"{name}.hds"))
    budget_file = flopy.utils.CellBudgetFile(str(workspace / f"{name}.cbc"), precision="double")
    head = head_file.get_data(totim=1.0)
    drn = budget_file.get_data(text="DRN", totim=1.0)[0]
    riv = budget_file.get_data(text="RIV", totim=1.0)[0]
    return head, float(drn["q"].sum()), float(riv["q"].sum())


def _run_mf6_convertible_mixed_case(*, workspace):
    """Run MF6 Newton NPF with convertible cells and co-located DRN/RIV.

    :param workspace: Writable directory for the simulation files.
    :return: Full head grid and signed DRN/RIV package totals.
    :raises RuntimeError: If MF6 fails to terminate normally.
    """
    import flopy

    name = "dwcv3d"
    simulation = flopy.mf6.MFSimulation(sim_name=name, exe_name=str(require_mf6()),
        version="mf6", sim_ws=str(workspace))
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=1,
        perioddata=[(1.0, 1, 1.0)])
    model = flopy.mf6.ModflowGwf(simulation, modelname=name, newtonoptions="NEWTON",
        save_flows=True)
    flopy.mf6.ModflowIms(simulation, complexity="COMPLEX", linear_acceleration="BICGSTAB",
        outer_maximum=200, outer_dvclose=1.0e-10, inner_maximum=100,
        inner_dvclose=1.0e-10, rcloserecord=[1.0e-10, "strict"])
    flopy.mf6.ModflowGwfdis(model, nlay=1, nrow=1, ncol=3, delr=1.0, delc=1.0,
        top=10.0, botm=0.0)
    flopy.mf6.ModflowGwfic(model, strt=8.0)
    flopy.mf6.ModflowGwfnpf(model, icelltype=1, k=1.0, k33=1.0, save_flows=True)
    flopy.mf6.ModflowGwfchd(model, maxbound=2,
        stress_period_data=[((0, 0, 0), 8.0), ((0, 0, 2), 8.0)], save_flows=True)
    flopy.mf6.ModflowGwfwel(model, maxbound=1,
        stress_period_data=[((0, 0, 1), 4.0)], save_flows=True)
    flopy.mf6.ModflowGwfdrn(model, maxbound=1,
        stress_period_data=[((0, 0, 1), 6.0, 1.0)], save_flows=True)
    flopy.mf6.ModflowGwfriv(model, maxbound=1,
        stress_period_data=[((0, 0, 1), 8.0, 2.0, 5.0)], save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc", saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    simulation.write_simulation()
    success, output = simulation.run_simulation(silent=True, report=True)
    if not success or not any("Normal termination" in line for line in output):
        raise RuntimeError(f"MF6 failed: {output}")
    head_file = flopy.utils.HeadFile(str(workspace / f"{name}.hds"))
    budget_file = flopy.utils.CellBudgetFile(str(workspace / f"{name}.cbc"), precision="double")
    head = head_file.get_data(totim=1.0)
    drn = budget_file.get_data(text="DRN", totim=1.0)[0]
    riv = budget_file.get_data(text="RIV", totim=1.0)[0]
    return head, float(drn["q"].sum()), float(riv["q"].sum())


def _run_mf6_transient_layered_case(*, workspace):
    """Run two-layer confined STO with distinct DRN and RIV cells.

    :param workspace: Writable directory for the simulation files.
    :return: Time-indexed head fields and signed package-rate totals.
    :raises RuntimeError: If MF6 fails to terminate normally.
    """
    import flopy

    name = "dwtrlyr"
    simulation = flopy.mf6.MFSimulation(sim_name=name, exe_name=str(require_mf6()),
        version="mf6", sim_ws=str(workspace))
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=2,
        perioddata=[(1.0, 1, 1.0), (1.0, 1, 1.0)])
    model = flopy.mf6.ModflowGwf(simulation, modelname=name, save_flows=True)
    flopy.mf6.ModflowIms(simulation, complexity="COMPLEX", linear_acceleration="BICGSTAB",
        outer_maximum=200, outer_dvclose=1.0e-10, inner_maximum=100,
        inner_dvclose=1.0e-10, rcloserecord=[1.0e-10, "strict"])
    flopy.mf6.ModflowGwfdis(model, nlay=2, nrow=1, ncol=3, delr=1.0, delc=1.0,
        top=12.0, botm=[6.0, 0.0])
    flopy.mf6.ModflowGwfic(model, strt=11.0)
    kx = np.array([[[10.0] * 3], [[2.0] * 3]])
    kz = np.array([[[2.0] * 3], [[1.0] * 3]])
    flopy.mf6.ModflowGwfnpf(model, icelltype=0, k=kx, k33=kz, save_flows=True)
    flopy.mf6.ModflowGwfsto(model, iconvert=0, ss=1.0, sy=0.0,
        steady_state={}, transient={0: True}, save_flows=True)
    flopy.mf6.ModflowGwfchd(model, maxbound=4,
        stress_period_data=[((layer, 0, col), 11.0)
            for layer in range(2) for col in (0, 2)], save_flows=True)
    flopy.mf6.ModflowGwfwel(model, maxbound=1,
        stress_period_data={0: [((0, 0, 1), 0.0)], 1: [((0, 0, 1), -100.0)]},
        save_flows=True)
    flopy.mf6.ModflowGwfdrn(model, maxbound=1,
        stress_period_data=[((1, 0, 1), 9.0, 1.0)], save_flows=True)
    flopy.mf6.ModflowGwfriv(model, maxbound=1,
        stress_period_data=[((0, 0, 1), 10.5, 2.0, 9.5)], save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc", saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    simulation.write_simulation()
    success, output = simulation.run_simulation(silent=True, report=True)
    if not success or not any("Normal termination" in line for line in output):
        raise RuntimeError(f"MF6 failed: {output}")
    head_file = flopy.utils.HeadFile(str(workspace / f"{name}.hds"))
    budget_file = flopy.utils.CellBudgetFile(str(workspace / f"{name}.cbc"), precision="double")
    times = head_file.get_times()
    heads = np.stack([head_file.get_data(totim=time) for time in times])
    package_rates = []
    for time in times:
        drn = budget_file.get_data(text="DRN", totim=time)[0]
        riv = budget_file.get_data(text="RIV", totim=time)[0]
        package_rates.append((float(drn["q"].sum()), float(riv["q"].sum())))
    return heads, np.asarray(package_rates)


@pytest.mark.parametrize("package", ["DRN", "RIV"])
def test_confined_native_package_matches_mf6_head_and_exchange(tmp_path, package):
    """Compare steady DarcyWarp head and package exchange with actual MF6.

    :param tmp_path: Pytest-provided temporary directory.
    :param package: Boundary package name, either ``DRN`` or ``RIV``.
    :return: None.
    """
    pytest.importorskip("flopy")
    require_mf6()
    fields = dict(
        kx_field=np.ones((1, 1, 2)),
        ky_field=np.ones((1, 1, 2)),
        kz_field=np.ones((1, 1, 2)),
        active=np.ones((1, 1, 2), dtype=np.int32),
        bc_mask=np.array([[[0, 1]]], dtype=np.int32),
        bc_values=np.array([[[0.0, 8.0]]]),
        rhs=np.array([[[3.0, 0.0]]]),
    )
    if package == "DRN":
        fields.update(drn_conductance=2.0, drn_elevation=4.0,
            drn_mask=np.array([[[1, 0]]], dtype=np.int32))
    else:
        fields.update(riv_conductance=2.0, riv_stage=7.0, riv_bottom=2.0,
            riv_mask=np.array([[[1, 0]]], dtype=np.int32))
    with WarpDarcySolver3D(nx=2, ny=1, nz=1, dx=1.0, dy=1.0, dz=10.0,
            device="cuda:0", implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_K_fields(**fields)
        head, info = solver.solve(max_outer=30, max_cycles=100)
        if package == "DRN":
            package_q = solver.accepted_drn_discharge(head=head, info=info)[0, 0, 0]
        else:
            package_q = solver.accepted_riv_discharge(head=head, info=info)[0, 0, 0]
    mf6_head, mf6_package_q = _run_mf6_package_case(workspace=tmp_path, package=package)
    assert info["converged"]
    np.testing.assert_allclose(head[0, 0, 0], mf6_head, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(package_q, -mf6_package_q, rtol=1e-11, atol=1e-11)


@pytest.mark.parametrize("package", ["DRN", "RIV"])
def test_transient_native_package_matches_mf6_heads_and_rates(tmp_path, package):
    """Compare transient threshold and disconnected leakage behavior with MF6.

    :param tmp_path: Pytest-provided temporary directory.
    :param package: Boundary package name, either ``DRN`` or ``RIV``.
    :return: None.
    """
    pytest.importorskip("flopy")
    require_mf6()
    shape = (1, 1, 1)
    package_fields = ({"drn_conductance": 1.0, "drn_elevation": 2.0}
        if package == "DRN" else {"riv_conductance": 1.0, "riv_stage": 3.0, "riv_bottom": 1.0})
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, dy=1.0, dz=10.0,
            device="cuda:0", implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32),
            bc_mask=np.zeros(shape, dtype=np.int32), bc_values=np.zeros(shape),
            rhs=np.full(shape, 30.0), **package_fields)
        heads, info = solver.solve_transient_npf(source_factors=[1.0, 0.0], dt=[1.0, 1.0],
            head_prev=np.zeros(shape), ss=1.0, sy=0.0, max_outer=30, max_cycles=100)
        if package == "DRN":
            rates = [solver.accepted_drn_discharge(head=head, info=period)[0, 0, 0]
                for head, period in zip(heads, info["periods"])]
        else:
            rates = [solver.accepted_riv_discharge(head=head, info=period)[0, 0, 0]
                for head, period in zip(heads, info["periods"])]
    mf6_heads, mf6_rates = _run_mf6_transient_package_case(workspace=tmp_path, package=package)
    assert info["converged"]
    np.testing.assert_allclose(heads[:, 0, 0, 0], mf6_heads, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(rates, -mf6_rates, rtol=1e-11, atol=1e-11)


def test_layered_npf_with_mixed_drn_riv_matches_mf6(tmp_path):
    """Compare layered heads and simultaneous DRN/RIV budgets with MF6.

    :param tmp_path: Pytest-provided temporary directory.
    :return: None.
    """
    pytest.importorskip("flopy")
    require_mf6()
    shape = (2, 1, 3)
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    fixed[:, :, 2] = 1
    values = np.full(shape, 11.0)
    values[fixed == 0] = 0.0
    initial = np.full(shape, 11.0)
    source = np.zeros(shape)
    source[0, 0, 1] = 2.0
    source[1, 0, 1] = -1.0
    drn_mask = np.zeros(shape, dtype=np.int32)
    drn_mask[1, 0, 1] = 1
    riv_mask = np.zeros(shape, dtype=np.int32)
    riv_mask[0, 0, 1] = 1
    interfaces = np.broadcast_to(np.array([12.0, 6.0, 0.0])[:, None, None],
        (3, 1, 3)).copy()
    with WarpDarcySolver3D(nx=3, ny=1, nz=2, dx=1.0, dy=1.0, dz=1.0,
            device="cuda:0", implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_layers(z_interfaces=interfaces, kx_field=np.broadcast_to(
            np.array([10.0, 2.0])[:, None, None], shape).copy(),
            ky_field=np.broadcast_to(np.array([10.0, 2.0])[:, None, None], shape).copy(),
            kz_field=np.broadcast_to(np.array([2.0, 1.0])[:, None, None], shape).copy(),
            active=active, bc_mask=fixed, bc_values=values, rhs=source, initial_head=initial,
            drn_conductance=1.0, drn_elevation=9.0, drn_mask=drn_mask,
            riv_conductance=2.0, riv_stage=10.5, riv_bottom=9.5, riv_mask=riv_mask)
        head, info = solver.solve(max_outer=30, max_cycles=100)
        drn_rate = solver.accepted_drn_discharge(head=head, info=info)[1, 0, 1]
        riv_rate = solver.accepted_riv_discharge(head=head, info=info)[0, 0, 1]
    mf6_head, mf6_drn_rate, mf6_riv_rate = _run_mf6_layered_mixed_case(workspace=tmp_path)
    assert info["converged"]
    np.testing.assert_allclose(head, mf6_head, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(drn_rate, -mf6_drn_rate, rtol=1e-11, atol=1e-11)
    np.testing.assert_allclose(riv_rate, -mf6_riv_rate, rtol=1e-11, atol=1e-11)


def test_convertible_npf_with_mixed_drn_riv_matches_mf6(tmp_path):
    """Compare convertible heads and co-located package rates with MF6.

    :param tmp_path: Pytest-provided temporary directory.
    :return: None.
    """
    pytest.importorskip("flopy")
    require_mf6()
    shape = (1, 1, 3)
    fixed = np.array([[[1, 0, 1]]], dtype=np.int32)
    values = np.array([[[8.0, 0.0, 8.0]]])
    source = np.array([[[0.0, 4.0, 0.0]]])
    package_mask = np.array([[[0, 1, 0]]], dtype=np.int32)
    with WarpDarcySolver3D(nx=3, ny=1, nz=1, dx=1.0, dy=1.0, dz=10.0,
            device="cuda:0", implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32),
            bc_mask=fixed, bc_values=values, rhs=source, initial_head=np.full(shape, 8.0),
            drn_conductance=1.0, drn_elevation=6.0, drn_mask=package_mask,
            riv_conductance=2.0, riv_stage=8.0, riv_bottom=5.0, riv_mask=package_mask)
        head, info = solver.solve(unconfined=True, top_elevation=10.0,
            npf_cell_type=1, max_outer=40, max_cycles=150)
        drn_rate = solver.accepted_drn_discharge(head=head, info=info)[0, 0, 1]
        riv_rate = solver.accepted_riv_discharge(head=head, info=info)[0, 0, 1]
    mf6_head, mf6_drn_rate, mf6_riv_rate = _run_mf6_convertible_mixed_case(workspace=tmp_path)
    assert info["converged"], info
    np.testing.assert_allclose(head, mf6_head, rtol=0.0, atol=2e-8)
    np.testing.assert_allclose(drn_rate, -mf6_drn_rate, rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(riv_rate, -mf6_riv_rate, rtol=1e-9, atol=1e-9)


def test_transient_layered_npf_mixed_drn_riv_matches_mf6(tmp_path):
    """Compare two-layer STO heads and DRN/RIV rates against MF6 by period.

    :param tmp_path: Pytest-provided temporary directory.
    :return: None.
    """
    pytest.importorskip("flopy")
    require_mf6()
    shape = (2, 1, 3)
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    fixed[:, :, 2] = 1
    values = np.full(shape, 11.0)
    values[fixed == 0] = 0.0
    initial = np.full(shape, 11.0)
    source = np.zeros(shape)
    source[0, 0, 1] = -100.0
    drn_mask = np.zeros(shape, dtype=np.int32)
    drn_mask[1, 0, 1] = 1
    riv_mask = np.zeros(shape, dtype=np.int32)
    riv_mask[0, 0, 1] = 1
    interfaces = np.broadcast_to(np.array([12.0, 6.0, 0.0])[:, None, None],
        (3, 1, 3)).copy()
    with WarpDarcySolver3D(nx=3, ny=1, nz=2, dx=1.0, dy=1.0, dz=1.0,
            device="cuda:0", implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_layers(z_interfaces=interfaces,
            kx_field=np.broadcast_to(np.array([10.0, 2.0])[:, None, None], shape).copy(),
            ky_field=np.broadcast_to(np.array([10.0, 2.0])[:, None, None], shape).copy(),
            kz_field=np.broadcast_to(np.array([2.0, 1.0])[:, None, None], shape).copy(),
            active=active, bc_mask=fixed, bc_values=values, rhs=source, initial_head=initial,
            drn_conductance=1.0, drn_elevation=9.0, drn_mask=drn_mask,
            riv_conductance=2.0, riv_stage=10.5, riv_bottom=9.5, riv_mask=riv_mask)
        heads, info = solver.solve_transient_npf(source_factors=[0.0, 1.0], dt=[1.0, 1.0],
            head_prev=initial, ss=1.0, sy=0.0, max_outer=40, max_cycles=150)
        rates = []
        for head, period in zip(heads, info["periods"]):
            rates.append((solver.accepted_drn_discharge(head=head, info=period)[1, 0, 1],
                solver.accepted_riv_discharge(head=head, info=period)[0, 0, 1]))
    mf6_heads, mf6_rates = _run_mf6_transient_layered_case(workspace=tmp_path)
    assert info["converged"]
    assert rates[0][1] > 0.0
    assert rates[1][1] < 0.0
    np.testing.assert_allclose(heads, mf6_heads, rtol=0.0, atol=2e-8)
    np.testing.assert_allclose(rates, -mf6_rates, rtol=1e-9, atol=1e-9)
