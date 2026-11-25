"""Manual native NPF examples for 3D DRN, RIV and streamflow reversal."""

from __future__ import annotations

import numpy as np

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def run_steady_drn(*, device: str, conductance: float, elevation: float,
    source_rate: float) -> dict:
    """Solve an isolated DRN cell and return its accepted exchange.

    :param device: Warp device identifier.
    :param conductance: Integrated drain conductance [L²/T].
    :param elevation: Drain activation elevation [L].
    :param source_rate: Recharge/well source into the cell [L³/T].
    :return: Accepted head, DRN budget and per-cell discharge.
    """
    shape = (1, 1, 1)
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
            implementation="fast", formulation="mf6_6_6_2_npf") as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32),
            bc_mask=np.zeros(shape, dtype=np.int32), bc_values=np.zeros(shape),
            rhs=np.full(shape, source_rate), drn_conductance=conductance,
            drn_elevation=elevation)
        head, info = model.solve(max_outer=30, max_cycles=100)
        discharge = model.accepted_drn_discharge(head=head, info=info)
    if not info["converged"]:
        raise RuntimeError(f"DRN example did not converge: {info['stop_reason']}")
    return dict(head=head, budget=info["drn_budget"], discharge=discharge)


def run_steady_riv(*, device: str, conductance: float, stage: float,
    bottom: float, source_rate: float) -> dict:
    """Solve an isolated RIV cell and return its accepted signed exchange.

    :param device: Warp device identifier.
    :param conductance: Integrated riverbed conductance [L²/T].
    :param stage: River stage [L].
    :param bottom: Riverbed bottom [L].
    :param source_rate: Recharge/well source into the cell [L³/T].
    :return: Accepted head, RIV budget and per-cell discharge.
    """
    shape = (1, 1, 1)
    with WarpDarcySolver3D(nx=1, ny=1, nz=1, dx=1.0, device=device,
            implementation="fast", formulation="mf6_6_6_2_npf") as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
            kz_field=np.ones(shape), active=np.ones(shape, dtype=np.int32),
            bc_mask=np.zeros(shape, dtype=np.int32), bc_values=np.zeros(shape),
            rhs=np.full(shape, source_rate), riv_conductance=conductance,
            riv_stage=stage, riv_bottom=bottom)
        head, info = model.solve(max_outer=30, max_cycles=100)
        discharge = model.accepted_riv_discharge(head=head, info=info)
    if not info["converged"]:
        raise RuntimeError(f"RIV example did not converge: {info['stop_reason']}")
    return dict(head=head, budget=info["riv_budget"], discharge=discharge)


def run_multilayer_stream_reversal(*, device: str) -> dict:
    """Demonstrate pumping reversing accepted RIV exchange in layered STO.

    :param device: Warp device identifier.
    :return: Two accepted heads, signed river rates and period metadata.
    :raises RuntimeError: If either transient solve fails to converge.
    """
    shape = (2, 1, 3)
    z_interfaces = np.broadcast_to(np.array([12.0, 6.0, 0.0])[:, None, None],
        (3, 1, 3)).copy()
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    fixed[:, :, -1] = 1
    values = np.full(shape, 11.0)
    values[fixed == 0] = 0.0
    initial = np.full(shape, 11.0)
    source = np.zeros(shape)
    riv_mask = np.zeros(shape, dtype=np.int32)
    riv_mask[0, 0, 1] = 1
    with WarpDarcySolver3D(nx=3, ny=1, nz=2, dx=1.0, dy=1.0,
            device=device, implementation="fast", formulation="mf6_6_6_2_npf") as model:
        model.build_from_layers(z_interfaces=z_interfaces,
            kx_field=np.full(shape, 10.0), ky_field=np.full(shape, 10.0),
            kz_field=np.full(shape, 10.0), active=active, bc_mask=fixed,
            bc_values=values, rhs=source, initial_head=initial,
            riv_conductance=2.0, riv_stage=10.5, riv_bottom=9.5, riv_mask=riv_mask)
        previous = initial
        heads = []
        periods = []
        rates = []
        for pumping_rate in (0.0, -100.0):
            source.fill(0.0)
            source[0, 0, 1] = pumping_rate
            model.update_rhs(rhs=source)
            model.update_initial_head(initial_head=previous)
            head, info = model.solve(transient=True, dt=1.0, head_prev=previous,
                ss=1.0, sy=0.0, max_outer=30, max_cycles=100)
            if not info["converged"]:
                raise RuntimeError(f"Transient example failed: {info['stop_reason']}")
            rates.append(model.accepted_riv_discharge(head=head, info=info)[0, 0, 1])
            heads.append(head)
            periods.append(info)
            previous = head.copy()
    if rates[0] <= 0.0 or rates[1] >= 0.0:
        raise RuntimeError(f"The example did not reverse stream exchange: {rates}")
    return dict(heads=np.asarray(heads), riv_rates=np.asarray(rates), periods=periods)


if __name__ == "__main__":
    device = "cuda:0"
    drn_conductance = 2.0
    drn_elevation = 4.0
    drn_source_rate = 1.0
    riv_conductance = 2.0
    riv_stage = 5.0
    riv_bottom = 2.0
    riv_source_rate = 1.0

    drain_result = run_steady_drn(device=device, conductance=drn_conductance,
        elevation=drn_elevation, source_rate=drn_source_rate)
    river_result = run_steady_riv(device=device, conductance=riv_conductance,
        stage=riv_stage, bottom=riv_bottom, source_rate=riv_source_rate)
    reversal_result = run_multilayer_stream_reversal(device=device)
    print("DRN:", drain_result)
    print("RIV:", river_result)
    print("Transient multilayer RIV rates:", reversal_result["riv_rates"])
