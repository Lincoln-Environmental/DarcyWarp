# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic layered NPF verification. Developed by Patrick Durney and Jaco
Fourie; software owned by Lincoln Agritech Ltd. Units: metres and days.
Run: DARCY_FLOAT=float64 PYTHONPATH=. python examples/verify_3d_series_flow.py
"""
from pathlib import Path
import json
import numpy as np
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def verify_series_flow(*, interfaces, conductivities, horizontal_shape, spacing,
                       boundary_heads, device, output, head_tolerance, flux_tolerance):
    thickness = -np.diff(interfaces)
    shape = (thickness.size, *horizontal_shape)
    z = np.broadcast_to(interfaces[:, None, None], (thickness.size+1, *horizontal_shape)).copy()
    k = np.broadcast_to(conductivities[:, None, None], shape).copy()
    resistance = (.5*thickness[:-1]/conductivities[:-1]+.5*thickness[1:]/conductivities[1:])/(spacing[0]*spacing[1])
    flux = (boundary_heads[0]-boundary_heads[1])/resistance.sum()
    exact = boundary_heads[0]-flux*np.concatenate(([0.], np.cumsum(resistance)))
    expected = np.broadcast_to(exact[:, None, None], shape)
    fixed = np.zeros(shape, dtype=bool)
    fixed[0] = fixed[-1] = True
    with WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=spacing[0], dy=spacing[1],
            device=device, implementation='fast', formulation='mf6_6_6_2_npf') as model:
        model.build_from_layers(z_interfaces=z, kx_field=k, ky_field=k, kz_field=k,
            active=np.ones(shape, dtype=bool), bc_mask=fixed, bc_values=expected,
            rhs=np.zeros(shape), initial_head=np.full(shape, np.mean(boundary_heads)))
        head, info = model.solve()
    measured_flux = (head[:-1]-head[1:])/resistance[:, None, None]
    head_error = float(np.max(np.abs(head-expected)))
    flux_error = float(np.max(np.abs(measured_flux-flux)))
    if not info['converged'] or head_error > head_tolerance or flux_error > flux_tolerance:
        raise AssertionError(dict(head_error_m=head_error, flux_error_m3_day=flux_error, info=info))
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(file=output/'results.npz', head=head, expected=expected, flux=measured_flux,
                        expected_flux=flux, z_interfaces=z, conductivity=k)
    result = dict(head_error_max_m=head_error, face_flux_error_max_m3_day=flux_error,
                  expected_face_flux_m3_day=float(flux), info=info)
    (output/'results.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == "__main__":
    interfaces = np.array([30., 28., 23., 22., 18., 10.])
    conductivities = np.array([2., .3, 8., 1., 4.])
    horizontal_shape = (3, 4)
    spacing = (2., 3.)
    boundary_heads = (32., 25.)
    device = 'cpu'
    output = Path('release_evidence/example')
    head_tolerance = 1e-7
    flux_tolerance = 1e-8
    result = verify_series_flow(interfaces=interfaces, conductivities=conductivities,
        horizontal_shape=horizontal_shape, spacing=spacing, boundary_heads=boundary_heads, device=device,
        output=output, head_tolerance=head_tolerance, flux_tolerance=flux_tolerance)
    print(json.dumps(dict(head_error_max_m=result['head_error_max_m'],
                          face_flux_error_max_m3_day=result['face_flux_error_max_m3_day'])))
