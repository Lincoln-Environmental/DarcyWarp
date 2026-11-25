"""Manually runnable 3D WEL examples covering pumping, schedules and updates.

Run from the repository root with the project environment active:

.. code-block:: console

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/example_3d_wel.py
"""
from __future__ import annotations

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def steady_multilayer_pumping(device: str = "cuda:0") -> None:
    """Pump two layers through explicit sparse well entries.

    :param device: Warp device identifier.
    :return: None.
    """
    shape = (2, 6, 6)
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    values = np.where(fixed != 0, 20.0, 0.0)
    model = WarpDarcySolver3D(nx=6, ny=6, nz=2, dx=50.0, dy=50.0, dz=10.0,
        formulation="mf6_6_6_2_npf", solver="kcycle", implementation="fast", device=device)
    model.build_from_K_fields(kx_field=np.full(shape, 8.0), ky_field=np.full(shape, 8.0),
        kz_field=np.full(shape, 1.5), active=active, bc_mask=fixed, bc_values=values,
        R_field=3e-4, wel_entries=[((0, 3, 3), -1200.0), ((1, 3, 3), -400.0),
                                   ((0, 2, 4), -600.0), ((0, 2, 4), 250.0)],
        initial_head=np.full(shape, 20.0))
    head, info = model.solve()
    budgets = model.physical_source_budgets()
    print("steady multilayer pumping")
    print("  converged:", info["converged"])
    print("  well-cell heads:", head[0, 3, 3], head[1, 3, 3])
    print("  well budget:", budgets["wel"])
    print("  recharge budget:", budgets["recharge"])
    model.close()


def scheduled_transient_pumping(device: str = "cuda:0") -> None:
    """Run a three-period device-resident history with a well schedule.

    :param device: Warp device identifier.
    :return: None.
    """
    shape = (1, 6, 6)
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    model = WarpDarcySolver3D(nx=6, ny=6, nz=1, dx=50.0, dy=50.0, dz=10.0,
        formulation="mf6_6_6_2_npf", solver="kcycle", implementation="fast", device=device)
    model.build_from_K_fields(kx_field=np.full(shape, 8.0), ky_field=np.full(shape, 8.0),
        kz_field=np.full(shape, 1.5), active=active, bc_mask=fixed,
        bc_values=np.where(fixed != 0, 20.0, 0.0),
        wel_entries=[((0, 3, 3), -1500.0)], initial_head=np.full(shape, 20.0))
    schedule = [[((0, 3, 3), -1500.0)], [((0, 3, 3), -3000.0)], None]
    heads, info = model.solve_transient_npf(dt=[30.0, 30.0, 30.0],
        head_prev=np.full(shape, 20.0), ss=1e-5, sy=0.0, wel_rates=schedule)
    print("scheduled transient pumping")
    print("  converged:", info["converged"], "accepted steps:", info["accepted_steps"])
    for period in info["periods"]:
        print("  step", period["index"], "well outflow", period["source_budgets"]["wel"]["outflow"])
    totals = info["accepted_exchange_volume_totals"]["sources"]["wel"]
    print("  integrated well volume:", totals)
    # Built forcing is restored after the history.
    print("  restored built well:", model.wel_budget())
    model.close()


def pumping_induced_river_reversal(device: str = "cuda:0") -> None:
    """Reverse a gaining river to losing with a nearby sparse well.

    :param device: Warp device identifier.
    :return: None.
    """
    shape = (1, 5, 7)
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    model = WarpDarcySolver3D(nx=7, ny=5, nz=1, dx=40.0, dy=40.0, dz=10.0,
        formulation="mf6_6_6_2_npf", solver="kcycle", implementation="fast", device=device)
    stage_row = 2
    riv_entries = [((0, stage_row, col), 300.0, 19.0, 18.0) for col in range(2, 6)]
    model.build_from_K_fields(kx_field=np.full(shape, 10.0), ky_field=np.full(shape, 10.0),
        kz_field=np.full(shape, 2.0), active=active, bc_mask=fixed,
        bc_values=np.where(fixed != 0, 22.0, 0.0), R_field=5e-4,
        riv_entries=riv_entries, initial_head=np.full(shape, 20.0))
    head, info = model.solve()
    gaining = info["riv_budget"]
    print("pumping-induced river reversal")
    print("  no pumping: river net", gaining["net"], "(negative into aquifer = gaining)")
    model.update_wel([((0, stage_row, 3), -5000.0)])
    head, info = model.solve()
    print("  after pumping: river net", info["riv_budget"]["net"], "head at well", head[0, stage_row, 3])
    rates = info["riv_entry_rates"]
    print("  per-entry river rates:", rates)
    model.close()


if __name__ == "__main__":
    wp.init()
    device = "cuda:0" if wp.is_cuda_available() else "cpu"
    steady_multilayer_pumping(device)
    scheduled_transient_pumping(device)
    pumping_induced_river_reversal(device)
