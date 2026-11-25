"""Compare the realistic WEL/boundary benchmark case against actual MF6.

Builds the matched MF6 6.6.2 twin of the small heterogeneous benchmark
grid: identical grids, properties, receiving-cell recharge, per-entry
WEL/DRN/RIV bounds (including shared cells) and timesteps, then compares
final heads and gross package rates.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_3d_wel_boundaries import heterogeneous_fields  # noqa: E402

RESULT_DIR = Path(__file__).resolve().parents[1] / "benchmark_results" / "wel_boundaries_3d"


def receiving_cells(active, fixed):
    """Uppermost active non-fixed cell of each column, C-order flat indices."""
    nz, ny, nx = active.shape
    layers = np.argmax(active, axis=0)
    rows, columns = np.ogrid[:ny, :nx]
    receives = (active[layers, rows, columns] != 0) & (fixed[layers, rows, columns] == 0)
    rows, columns = np.nonzero(receives)
    return np.ravel_multi_index((layers[rows, columns], rows, columns), (nz, ny, nx))


def run_mf6_twin(fields, shape, workspace):
    """Run the matched MF6 model for the benchmark fields.

    :param fields: The heterogeneous benchmark field mapping.
    :param shape: (nz, ny, nx) grid shape.
    :param workspace: Writable directory for the simulation.
    :return: Final head grid and gross package rate totals.
    """
    import flopy

    from DARCY_WARP_PACKAGE.project_base import require_mf6

    nz, ny, nx = shape
    dx = dy = 40.0
    dz = 10.0
    name = "dwwelbound"
    simulation = flopy.mf6.MFSimulation(sim_name=name, exe_name=str(require_mf6()),
        version="mf6", sim_ws=str(workspace))
    flopy.mf6.ModflowTdis(simulation, time_units="DAYS", nper=1,
        perioddata=[(1.0, 1, 1.0)])
    model = flopy.mf6.ModflowGwf(simulation, modelname=name, save_flows=True)
    flopy.mf6.ModflowIms(simulation, complexity="COMPLEX", linear_acceleration="BICGSTAB",
        outer_maximum=200, outer_dvclose=1.0e-11, inner_maximum=200,
        inner_dvclose=1.0e-12, rcloserecord=[1.0e-12, "strict"])
    botm = np.array([40.0 - dz * (layer + 1) for layer in range(nz)])
    flopy.mf6.ModflowGwfdis(model, nlay=nz, nrow=ny, ncol=nx, delr=dx, delc=dy,
        top=40.0, botm=botm)
    flopy.mf6.ModflowGwfic(model, strt=40.0)
    flopy.mf6.ModflowGwfnpf(model, icelltype=0, k=fields["kx_field"],
        k22=fields["ky_field"], k33=fields["kz_field"], save_flows=True)
    receivers = receiving_cells(fields["active"], fields["bc_mask"])
    layers, rows, columns = np.unravel_index(receivers, shape)
    flat_rate = np.broadcast_to(np.asarray(fields["R_field"], dtype=float), (ny, nx)).ravel()
    recharge_bounds = [((int(l), int(r), int(c)), float(flat_rate[r * nx + c]))
                       for l, r, c in zip(layers, rows, columns)]
    flopy.mf6.ModflowGwfrch(model, maxbound=len(recharge_bounds),
        stress_period_data=recharge_bounds, save_flows=True)
    from DARCY_WARP_PACKAGE.physics.drn_riv_3d import validate_boundary_entries
    from DARCY_WARP_PACKAGE.physics.wel import validate_well_entries

    wel = validate_well_entries(entries=fields["wel_entries"], shape=shape,
                                active=fields["active"])
    wel_bounds = [((int(c[0]), int(c[1]), int(c[2])), float(rate))
                  for c, rate in zip(wel.cells.tolist(), wel.rates.tolist())]
    flopy.mf6.ModflowGwfwel(model, maxbound=len(wel_bounds),
        stress_period_data=wel_bounds, save_flows=True)
    drn = validate_boundary_entries(entries=fields["drn_entries"], package="DRN",
        shape=shape, active=fields["active"], fixed=fields["bc_mask"])
    drn_bounds = [((int(c[0]), int(c[1]), int(c[2])), float(e), float(cnd))
                  for c, cnd, e in zip(drn.cells.tolist(), drn.conductance.tolist(),
                                       drn.first_level.tolist())]
    flopy.mf6.ModflowGwfdrn(model, maxbound=len(drn_bounds),
        stress_period_data=drn_bounds, save_flows=True)
    riv = validate_boundary_entries(entries=fields["riv_entries"], package="RIV",
        shape=shape, active=fields["active"], fixed=fields["bc_mask"])
    riv_bounds = [((int(c[0]), int(c[1]), int(c[2])), float(s), float(cnd), float(b))
                  for c, cnd, s, b in zip(riv.cells.tolist(), riv.conductance.tolist(),
                                          riv.second_level.tolist(), riv.third_level.tolist())]
    flopy.mf6.ModflowGwfriv(model, maxbound=len(riv_bounds),
        stress_period_data=riv_bounds, save_flows=True)
    fixed = fields["bc_mask"]
    chd_bounds = [((int(l), int(r), int(c)), float(fields["bc_values"][l, r, c]))
                  for l, r, c in zip(*np.nonzero(fixed))]
    flopy.mf6.ModflowGwfchd(model, maxbound=len(chd_bounds),
        stress_period_data=chd_bounds, save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord=f"{name}.hds",
        budget_filerecord=f"{name}.cbc", saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")])
    simulation.write_simulation()
    success, output = simulation.run_simulation(silent=True, report=True)
    if not success or not any("Normal termination" in line for line in output):
        raise RuntimeError(f"MF6 failed: {output}")
    import flopy.utils.binaryfile as bf

    listing = (workspace / f"{name}.lst").read_text()
    heads = bf.HeadFile(str(workspace / f"{name}.hds"), precision="double")
    budget = bf.CellBudgetFile(str(workspace / f"{name}.cbc"), precision="double")
    final = heads.get_data(totim=heads.get_times()[-1])
    record = budget.get_data(text="WEL", totim=budget.get_times()[-1])[0]
    wel_total = float(sum(float(item["q"]) for item in record))
    record = budget.get_data(text=" DRN", totim=budget.get_times()[-1])[0]
    drn_total = float(sum(float(item["q"]) for item in record))
    record = budget.get_data(text=" RIV", totim=budget.get_times()[-1])[0]
    riv_total = float(sum(float(item["q"]) for item in record))
    return final, dict(wel=wel_total, drn=drn_total, riv=riv_total,
        percent_discrepancy=float(listing.split("PERCENT DISCREPANCY =")[-1].split()[0]))


def main(device="cuda:0"):
    """Run the CUDA solve and the matched MF6 twin, then compare.

    :param device: Warp device for the DarcyWarp solve.
    :return: Comparison mapping written next to the benchmark artifact.
    """
    import warp as wp

    wp.init()
    shape = (10, 112, 90)
    fields = heterogeneous_fields(shape=shape)
    fields.pop("dx"); fields.pop("dy"); fields.pop("dz")
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    model = WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=40.0, dy=40.0, dz=10.0,
        formulation="mf6_6_6_2_npf", solver="kcycle", implementation="fast", device=device)
    model.build_from_K_fields(**fields, initial_head=np.full(shape, 40.0))
    head, info = model.solve()
    assert info["converged"], info["stop_reason"]
    budgets = info["source_budgets"]
    drn_budget = info["drn_budget"]
    riv_budget = info["riv_budget"]
    model.close()
    mf6_heads, mf6_rates = run_mf6_twin(fields, shape, Path(tempfile.mkdtemp()))
    comparison = dict(
        cells=int(np.prod(shape)),
        head_rmse=float(np.sqrt(np.mean((head - mf6_heads) ** 2))),
        head_max_abs=float(np.max(np.abs(head - mf6_heads))),
        wel_rate_darcywarp=-budgets["wel"]["net"],
        wel_rate_mf6=-mf6_rates["wel"],
        drn_rate_darcywarp=drn_budget["net"],
        drn_rate_mf6=mf6_rates["drn"],
        riv_rate_darcywarp=riv_budget["net"],
        riv_rate_mf6=mf6_rates["riv"],
        mf6_percent_discrepancy=mf6_rates["percent_discrepancy"],
    )
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    (RESULT_DIR / "mf6_parity_small.json").write_text(json.dumps(comparison, indent=2))
    print(json.dumps(comparison, indent=2))
    return comparison


if __name__ == "__main__":
    main()
