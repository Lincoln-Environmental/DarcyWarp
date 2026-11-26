"""Matched pinned MF6 references using the existing 2D sanity IMS profiles."""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def native_fields(*, fields, dx, controls, convertible):
    """Map the physical thickness-integrated model to one native groundwater layer.

    :param fields: Actual maintained 2D model fields.
    :param dx: Cell width.
    :param controls: Physical geometry and source controls.
    :param convertible: Use the Picard physical aquifer rather than confined T.
    :return: Equivalent native builder fields.
    """
    shape = (1, *fields['active'].shape)
    thickness = controls['unconfined_top'] if convertible else controls['confined_thickness']
    saturated = controls['initial_head'] if convertible else thickness
    top = thickness if convertible else controls['initial_head']
    conductivity = (fields['T_field'] / saturated)[None]
    return dict(active=fields['active'][None], bc_mask=fields['bc_mask'][None],
        bc_values=fields['bc_values'][None], initial_head=np.full(shape, controls['initial_head']),
        kx_field=conductivity, ky_field=conductivity, kz_field=conductivity,
        R_field=fields['R_field'], wel_entries=[((0, *cell), rate) for cell, rate in fields['wel_entries']],
        drn_entries=None, riv_entries=None, dx=dx, dy=dx, dz=thickness, top_elevation=top)


if __name__ == "__main__":
    os.environ.setdefault('DARCY_FLOAT', 'float64')
    from qualify_package_2d import fields_for_grid
    from qualify_package_mf6 import run_native
    from sanity_3d_controls import read_profiles
    from DARCY_WARP_PACKAGE.project_base import require_mf6
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', choices=('100k', '1M', 'all'), default='all')
    parser.add_argument('--mode', choices=('steady', 'transient', 'picard', 'all'), default='all')
    parser.add_argument('--repeats', type=int, default=3)
    arguments = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = Path('/tmp/darcywarp-package-2d-mf6')
    controls = json.loads((root / 'working_tests/wel_regional_qualification_controls.json').read_text())
    controls.update(confined_thickness=10., unconfined_top=100., unconfined_sy=.15)
    grids = {'100k': (320, 320), '1M': (1000, 1000)}
    modes = ('steady', 'transient', 'picard')
    profiles = read_profiles(root=root)
    executable = require_mf6()
    for label, shape in grids.items():
        if arguments.grid not in ('all', label):
            continue
        for mode in modes:
            if arguments.mode not in ('all', mode):
                continue
            fields, dx = fields_for_grid(shape=shape, controls=controls)
            convertible = mode == 'picard'
            if convertible:
                fields['R_field'] = np.full(shape, np.mean(fields['R_field']))
            native = native_fields(fields=fields, dx=dx, controls=controls, convertible=convertible)
            if mode == 'steady':
                well_factors = [1., *controls['calibration_wel_factors']]
                recharge_factors = [1., *controls['calibration_recharge_factors']]
                durations = [1.] * len(well_factors)
            else:
                well_factors = np.resize(controls['history_wel_pattern'], controls['history_steps'])
                recharge_factors = np.resize(controls['history_recharge_pattern'], controls['history_steps'])
                durations = [controls['history_dt_days']] * controls['history_steps']
            ims = profiles['mf6']['steady_confined' if mode == 'steady' else 'transient']
            run_native(fields=native, durations=durations, well_factors=well_factors, recharge_factors=recharge_factors,
                stage_offsets=[0.] * len(durations), transient=mode != 'steady',
                ss=controls['specific_storage_per_m'], sy=controls['unconfined_sy'] if convertible else 0.,
                cell_type=int(convertible), ims_controls=ims, executable=executable, repeats=arguments.repeats,
                directory=output / label / mode)
