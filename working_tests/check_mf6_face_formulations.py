"""Measure standard and Newton MF6 face laws with all heads prescribed."""
import json
from pathlib import Path

import numpy as np


def measure_native_face(*, executable, workspace, newton, heads, conductivity,
                        top, bottom, dx, dy, ims_controls, smoothing_fraction, flow_tolerance):
    """Measure exchange without an iterative groundwater solution error.

    :param executable: Pinned MF6 6.6.2 binary.
    :param workspace: Native input/output directory.
    :param newton: Enable MF6 NEWTON UNDER_RELAXATION.
    :param heads: Two prescribed groundwater heads.
    :param conductivity: Physical horizontal conductivity in the two cells.
    :param top: Layer top elevation.
    :param bottom: Layer bottom elevation.
    :param dx: Cell width along the connection.
    :param dy: Transverse face width.
    :param ims_controls: Explicit COMPLEX/BICGSTAB controls.
    :param smoothing_fraction: Independently specified MF6 saturation smoothing width.
    :param flow_tolerance: Independent absolute agreement target for this scalar probe.
    :return: Measured native face flow and independently calculated face laws.
    :raises AssertionError: If native convergence or prescribed heads fail.
    """
    import flopy
    from DARCY_WARP_PACKAGE.validation.npf662_mf6 import pinned_native_executable
    binary, version, checksum, unused_build = pinned_native_executable(executable=executable)
    workspace.mkdir(parents=True, exist_ok=True)
    name = 'faceprobe'
    simulation = flopy.mf6.MFSimulation(sim_name=name, sim_ws=str(workspace), exe_name=str(binary))
    simulation.simulation_data.float_precision = 17
    simulation.simulation_data.float_characters = 25
    flopy.mf6.ModflowTdis(simulation, nper=1, perioddata=[(1., 1, 1.)])
    flopy.mf6.ModflowIms(simulation, **ims_controls)
    model = flopy.mf6.ModflowGwf(simulation, modelname=name, save_flows=True,
        newtonoptions='UNDER_RELAXATION' if newton else None)
    flopy.mf6.ModflowGwfdis(model, nlay=1, nrow=1, ncol=len(heads), delr=dx, delc=dy, top=top, botm=bottom)
    flopy.mf6.ModflowGwfic(model, strt=heads.reshape(1, 1, -1))
    flopy.mf6.ModflowGwfnpf(model, icelltype=1, k=conductivity.reshape(1, 1, -1), save_flows=True)
    flopy.mf6.ModflowGwfchd(model, stress_period_data=[((0, 0, index), float(head)) for index, head in enumerate(heads)], save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord=f'{name}.hds', budget_filerecord=f'{name}.cbc',
        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    simulation.write_simulation(silent=True)
    success, lines = simulation.run_simulation(silent=True, report=True)
    (workspace / 'engine.txt').write_text('\n'.join(lines))
    assert success and any('Normal termination' in line for line in lines)
    reader = flopy.utils.HeadFile(str(workspace / f'{name}.hds'), precision='double')
    np.testing.assert_array_equal(reader.get_data().ravel(), heads)
    reader.close()
    budget = flopy.utils.CellBudgetFile(str(workspace / f'{name}.cbc'), precision='double')
    rates = budget.get_data(text='CHD')[0]['q']
    budget.close()
    saturation = np.clip(heads - bottom, 0., top - bottom)
    transmissivity = conductivity * saturation
    gradient = (heads[0] - heads[1]) * dy / dx
    harmonic_transmissivity_flow = 2. * np.prod(transmissivity) / np.sum(transmissivity) * gradient
    upstream_flow = 2. * np.prod(conductivity) / np.sum(conductivity) * saturation[np.argmax(heads)] * gradient
    fraction = saturation / (top - bottom)
    assert np.all((fraction > smoothing_fraction) & (fraction < 1. - smoothing_fraction))
    slope = 1. / (1. - smoothing_fraction)
    smoothed_fraction = slope * fraction + .5 * (1. - slope)
    smoothed_upstream_flow = (2. * np.prod(conductivity) / np.sum(conductivity)
        * (top - bottom) * smoothed_fraction[np.argmax(heads)] * gradient)
    expected = smoothed_upstream_flow if newton else harmonic_transmissivity_flow
    error = abs(float(rates[0]) - expected)
    assert error <= flow_tolerance, (newton, float(rates[0]), expected, error)
    return dict(mf6_version=version, mf6_sha256=checksum, mf6_newton=newton,
        measured_first_chd_inflow_m3_day=float(rates[0]),
        harmonic_transmissivity_flow_m3_day=float(harmonic_transmissivity_flow),
        upstream_saturation_flow_m3_day=float(upstream_flow),
        smoothed_upstream_flow_m3_day=float(smoothed_upstream_flow),
        matched_law_absolute_error_m3_day=error, flow_tolerance_m3_day=flow_tolerance,
        name_file=(workspace / f'{name}.nam').read_text(),
        heads=heads.tolist(), conductivity=conductivity.tolist(), top=top, bottom=bottom)


if __name__ == "__main__":
    from DARCY_WARP_PACKAGE.project_base import require_mf6
    executable = require_mf6()
    workspace = Path('/tmp/darcywarp-mf6-face-formulation-probes')
    heads = np.array([4., 2.])
    conductivity = np.array([1., 4.])
    top = 10.
    bottom = 0.
    dx = 1.
    dy = 1.
    smoothing_fraction = 1.e-6
    flow_tolerance = 1.e-12
    ims_controls = dict(complexity='COMPLEX', linear_acceleration='BICGSTAB',
        outer_maximum=200, outer_dvclose=1.e-12, inner_maximum=200,
        inner_dvclose=1.e-12, rcloserecord=[1.e-12, 'strict'])
    formulations = [('standard', False), ('newton', True)]
    output = Path('benchmark_results/package_audit_repair/mf6_face_formulation_probes.json')
    rows = [measure_native_face(executable=executable, workspace=workspace / name, newton=newton,
        heads=heads, conductivity=conductivity, top=top, bottom=bottom, dx=dx, dy=dy,
        ims_controls=ims_controls, smoothing_fraction=smoothing_fraction,
        flow_tolerance=flow_tolerance) for name, newton in formulations]
    output.write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
