"""Record analytical errors for the repaired physical branches and common mode."""
import json
import os
from pathlib import Path

import numpy as np

def measure_switching_roots(*, devices, shape, cell, boundary_cases, target, cell_dimensions, time_step):
    """Compare high-index isolated boundaries to independent scalar roots.

    :param devices: CPU/CUDA devices to execute.
    :param shape: Groundwater grid shape.
    :param cell: Isolated active groundwater cell.
    :param boundary_cases: Package, inputs, physical source and analytical head.
    :param target: Independent absolute head error gate.
    :param cell_dimensions: Physical dx/dy/dz.
    :param time_step: Timestep with zero physical storage.
    :return: Measured head errors and actual execution diagnostics.
    """
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    rows = []
    active = np.zeros(shape, dtype=np.int32)
    active[cell] = 1
    for device in devices:
        for package, options, source_rate, expected in boundary_cases:
            source = np.zeros(shape)
            source[cell] = source_rate
            with WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0],
                    dx=cell_dimensions[0], dy=cell_dimensions[1], dz=cell_dimensions[2],
                    device=device, implementation='fast', formulation='mf6_6_6_2_npf') as model:
                model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape),
                    kz_field=np.ones(shape), active=active, bc_mask=np.zeros(shape),
                    bc_values=np.zeros(shape), rhs=source, initial_head=np.zeros(shape), **options)
                head, info = model.solve(transient=True, dt=time_step, head_prev=np.zeros(shape), ss=0., sy=0.,
                    npf_device_native=True, npf_preconditioner='kcycle', npf_krylov_method='fgmres')
                error = abs(float(head[cell]) - expected)
                assert info['converged'] and info['preconditioner_cycles'] > 0 and error <= target
                rows.append(dict(device=device, case=package, head_error_m=error, target_m=target,
                    implementation=info['implementation'], preconditioner=info['npf_preconditioner']))
    return rows


def measure_common_mode(*, devices, initial, conductivity, opposing_source, perturbation,
                        ghb_conductance, ghb_head, dx, thickness, dt, ss, target):
    """Derive steady balance and the transient constant mode independently.

    :param devices: CPU/CUDA devices to execute.
    :param initial: Original physical heads.
    :param conductivity: Physical conductivity grid.
    :param opposing_source: Large independently specified opposing source rates.
    :param perturbation: Small independent prescribed source increment.
    :param ghb_conductance: Physical steady component restraint.
    :param ghb_head: External boundary head.
    :param dx: Horizontal cell width.
    :param thickness: Physical cell thickness.
    :param dt: Transient timestep duration.
    :param ss: Physical specific storage.
    :param target: Independent absolute head error gate.
    :return: Measured head errors and actual execution diagnostics.
    """
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    rows = []
    shape = initial.shape
    for device in devices:
        for steady in (True, False):
            source = opposing_source.copy()
            if steady:
                source[0, 0, 1] += perturbation
                boundary = dict(ghb_conductance=np.array([[[ghb_conductance, 0.]]]), ghb_head=ghb_head)
                left = ghb_head + source.sum() / ghb_conductance
                face = conductivity.flat[0] * thickness
                expected = np.array([[[left, left + source[0, 0, 1] / face]]])
                options = {}
            else:
                source += perturbation
                boundary = {}
                expected = initial + perturbation * dt / (ss * dx * dx * thickness)
                options = dict(transient=True, head_prev=initial, dt=dt, ss=ss, sy=0., npf_device_native=True)
            with WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=dx, dy=dx, dz=thickness,
                    device=device, implementation='fast', formulation='mf6_6_6_2_npf') as model:
                model.build_from_K_fields(kx_field=conductivity, ky_field=conductivity, kz_field=conductivity,
                    active=np.ones(shape), bc_mask=np.zeros(shape), bc_values=np.zeros(shape),
                    rhs=source, initial_head=initial, **boundary)
                head, info = model.solve(npf_preconditioner='kcycle', npf_krylov_method='fgmres', **options)
                error = float(np.max(np.abs(head - expected)))
                assert info['converged'] and error <= target
                rows.append(dict(device=device, case='weak_common_mode_steady' if steady else 'weak_common_mode_transient',
                    head_error_m=error, target_m=target, implementation=info['implementation'],
                    preconditioner=info['npf_preconditioner']))
    return rows


if __name__ == "__main__":
    os.environ['DARCY_FLOAT'] = 'float64'
    devices = ['cpu', 'cuda:0']
    shape = (3, 3, 4)
    cell = (2, 2, 3)
    cell_dimensions = (1., 1., 1.)
    time_step = 1.
    boundary_cases = [
        ('drain_activation', dict(drn_entries=[(cell, 1., 2.)]), 1., 3.),
        ('disconnected_river_activation', dict(riv_entries=[(cell, 2., 3., 1.)]), 0., 3.),
    ]
    switching_target = 1.e-12
    common_mode_target = 1.e-6
    initial = np.array([[[11., 10.]]])
    conductivity = np.full_like(initial, 100.)
    opposing_source = np.array([[[1000., -1000.]]])
    perturbation = 8.e-6
    ghb_conductance = 8.e-7
    ghb_head = 11.
    dx = 10.
    thickness = 10.
    dt = 365.
    ss = 1.e-5
    output = Path('benchmark_results/package_audit_repair/final_analytical_errors.json')
    rows = measure_switching_roots(devices=devices, shape=shape, cell=cell,
        boundary_cases=boundary_cases, target=switching_target,
        cell_dimensions=cell_dimensions, time_step=time_step)
    rows += measure_common_mode(devices=devices, initial=initial, conductivity=conductivity,
        opposing_source=opposing_source, perturbation=perturbation, ghb_conductance=ghb_conductance,
        ghb_head=ghb_head, dx=dx, thickness=thickness, dt=dt, ss=ss, target=common_mode_target)
    output.write_text(json.dumps(dict(passed=True, rows=rows), indent=2))
    print(json.dumps(rows, indent=2))
