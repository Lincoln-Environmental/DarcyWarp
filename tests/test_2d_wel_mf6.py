# SPDX-License-Identifier: AGPL-3.0-or-later
"""Actual MF6 6.6.2 comparisons for the 2D WEL package.

A confined two-dimensional model with explicit sparse wells reproduces a
matched MF6 DIS/NPF/WEL/CHD simulation on the same grid, properties and
forcing.
"""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.project_base import require_mf6
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

DEVICES = ["cpu", "cuda:0"]


def _cuda_available(device):
    if not device.startswith("cuda"):
        return True
    wp.init()
    return wp.is_cuda_available()


def _run_mf6_2d_wel(*, workspace, wells, recharge=0.0):
    """Run a matched single-layer MF6 model with explicit WEL bounds.

    :param workspace: Writable directory for the simulation.
    :param wells: Mapping of ``(row, col)`` cells to volumetric rates [L3/T].
    :param recharge: Uniform areal recharge [L/T] on every cell.
    :return: Final head grid.
    """
    import flopy

    from DARCY_WARP_PACKAGE.project_base import require_mf6 as require

    name = "dw2dwel"
    nrow, ncol = 5, 6
    simulation = flopy.mf6.MFSimulation(sim_name=name, exe_name=str(require_mf6()),
        version="mf6", sim_ws=str(workspace))
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=1,
        perioddata=[(1.0, 1, 1.0)])
    model = flopy.mf6.ModflowGwf(simulation, modelname=name, save_flows=True)
    flopy.mf6.ModflowIms(simulation, complexity="COMPLEX", linear_acceleration="BICGSTAB",
        outer_maximum=100, outer_dvclose=1.0e-12, inner_maximum=100,
        inner_dvclose=1.0e-12, rcloserecord=[1.0e-12, "strict"])
    flopy.mf6.ModflowGwfdis(model, nlay=1, nrow=nrow, ncol=ncol, delr=100.0, delc=100.0,
        top=10.0, botm=0.0)
    flopy.mf6.ModflowGwfic(model, strt=12.0)
    # The 2D DarcyWarp model takes transmissivity directly; MF6 NPF
    # multiplies conductivity by the 10 m layer thickness, so k = 5 gives
    # the matched T = 50.
    flopy.mf6.ModflowGwfnpf(model, icelltype=0, k=5.0, save_flows=True)
    if recharge:
        flopy.mf6.ModflowGwfrch(model, maxbound=nrow * ncol,
            stress_period_data=[((0, r, c), recharge)
                for r in range(nrow) for c in range(ncol)], save_flows=True)
    flopy.mf6.ModflowGwfwel(model, maxbound=len(wells),
        stress_period_data=[((0, row, col), float(rate))
                            for (row, col), rate in wells.items()], save_flows=True)
    flopy.mf6.ModflowGwfchd(model, maxbound=nrow,
        stress_period_data=[((0, r, 0), 12.0) for r in range(nrow)], save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc", saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    simulation.write_simulation()
    success, output = simulation.run_simulation(silent=True, report=True)
    if not success or not any("Normal termination" in line for line in output):
        raise RuntimeError(f"MF6 failed: {output}")
    import flopy.utils.binaryfile as bf

    heads = bf.HeadFile(str(workspace / f"{name}.hds"), precision="double")
    return heads.get_data(totim=heads.get_times()[-1])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend,implementation", [("confined_pcg", "classic"),
    ("confined_kcycle", "classic"), ("confined_kcycle", "fast")])
def test_2d_confined_wel_matches_mf6(tmp_path, device, backend, implementation):
    """Matched confined pumping agrees with MF6 on the same grid."""
    pytest.importorskip("flopy")
    require_mf6()
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    ny, nx, dx = 5, 6, 100.0
    wells = {(2, 3): -400.0, (1, 4): -250.0}
    solver = WarpDarcySolver(nx=nx, ny=ny, dx=dx, device=device)
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    solver.build_from_fields(T_field=np.full((ny, nx), 50.0), R_field=0.0,
        active=active, bc_mask=bc_mask, bc_values=np.full((ny, nx), 12.0),
        wel_entries=[((2, 3), -400.0), ((1, 4), -250.0)])
    try:
        options = dict(initial_head=np.full((ny, nx), 12.), rel_tol=1.e-12, abs_tol_min=1.e-12)
        if backend == "confined_kcycle":
            options.update(implementation=implementation, max_cycles=200, min_coarse_cells=2,
                check_every_no=1, nu_coarse=10, smoother="jacobi", fallback_to_pcg=False)
        head, info = solver.solve(solver=backend, **options)
        assert info["converged"]
        assert info["solver_backend"] == backend
        if implementation == "fast":
            assert info["implementation"] == "fast"
            if device.startswith("cuda"):
                assert info["cuda_graph_built_this_call"] or info["cuda_graph_reused"]
        mf6_heads = _run_mf6_2d_wel(workspace=tmp_path, wells=wells)
        # The 2D confined PCG solve closes at its documented head tolerance;
        # agreement at this grid sits near 1e-7 m.
        np.testing.assert_allclose(head, mf6_heads[0], rtol=0.0, atol=1e-6)
        assert solver.wel_budget()["outflow"] == pytest.approx(650.0)
    finally:
        solver.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("backend,implementation", [("confined_pcg", "classic"),
    ("confined_kcycle", "classic"), ("confined_kcycle", "fast")])
def test_2d_recharge_with_wel_matches_mf6(tmp_path, device, backend, implementation):
    """Combined recharge and pumping agree with MF6 including budgets."""
    pytest.importorskip("flopy")
    require_mf6()
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    ny, nx, dx = 5, 6, 100.0
    wells = {(3, 2): -900.0}
    solver = WarpDarcySolver(nx=nx, ny=ny, dx=dx, device=device)
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    solver.build_from_fields(T_field=np.full((ny, nx), 50.0), R_field=5e-4,
        active=active, bc_mask=bc_mask, bc_values=np.full((ny, nx), 12.0),
        wel_entries=[((3, 2), -900.0)])
    try:
        options = dict(initial_head=np.full((ny, nx), 12.), rel_tol=1.e-12, abs_tol_min=1.e-12)
        if backend == "confined_kcycle":
            options.update(implementation=implementation, max_cycles=200, min_coarse_cells=2,
                check_every_no=1, nu_coarse=10, smoother="jacobi", fallback_to_pcg=False)
        head, info = solver.solve(solver=backend, **options)
        assert info["converged"]
        assert info["solver_backend"] == backend
        if implementation == "fast":
            assert info["implementation"] == "fast"
            if device.startswith("cuda"):
                assert info["cuda_graph_built_this_call"] or info["cuda_graph_reused"]
        mf6_heads = _run_mf6_2d_wel(workspace=tmp_path, wells=wells, recharge=5e-4)
        np.testing.assert_allclose(head, mf6_heads[0], rtol=0.0, atol=1e-6)
        assert solver.wel_budget()["outflow"] == pytest.approx(900.0)
    finally:
        solver.close()
