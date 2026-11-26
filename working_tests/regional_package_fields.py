"""Reproducible original regional material and package field fixtures.

Performance qualification lives in ``qualify_package_kcycle.py``.
These original forcing values are also retained for like-for-like diagnostics.
"""
import numpy as np


def heterogeneous_fields(*, shape, seed=11, wells_top=True, drain=True, river=True):
    """Build layered heterogeneous fields with all source/boundary packages.

    :param shape: (nz, ny, nx) grid shape, top layer first.
    :param seed: Reproducible heterogeneity seed.
    :param wells_top: Place abstraction wells in the upper half of each column.
    :param drain: Enable an extensive drainage network on the top layer.
    :param river: Enable a gaining/losing river across the domain.
    :return: Keyword fields for ``build_from_K_fields`` plus the entry lists.
    """
    nz, ny, nx = shape
    rng = np.random.default_rng(seed)
    log_k = rng.normal(loc=np.log(5.0), scale=1.0, size=shape)
    layer_scale = np.linspace(3.0, 0.3, nz)[:, None, None]
    k = np.exp(log_k * 0.6) * layer_scale
    k[:, :: ny // 8, :] *= 0.05  # low-K horizontal inclusions
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    fixed[:, :, -1] = 1
    bc_values = np.where(fixed != 0, 40.0, 0.0)
    dx = dy = 40.0
    dz = 10.0
    recharge = np.zeros((ny, nx))
    recharge[:, :] = 2.0e-4
    recharge[::7, ::5] = 0.0  # dry patches receive none

    entries = []
    if wells_top:
        for row in range(5, ny - 5, 11):
            for col in range(4, nx - 4, 13):
                layer = 0 if (row + col) % 3 else min(1, nz - 1)
                entries.append(((layer, row, col), -3000.0))
                if (row + col) % 5 == 0:
                    # A second, partially screened well shares the cell.
                    entries.append(((layer, row, col), -1500.0))
    drn_entries = None
    if drain:
        drn_entries = [((0, row, col), 25.0, 28.5 + 0.01 * (col % 4))
                       for row in range(2, ny - 2, 3) for col in range(2, nx - 2, 4)]
    riv_entries = None
    if river:
        stage_row = ny // 2
        riv_entries = [((0, stage_row, col), 400.0,
                        39.5 + (0.6 if col % 2 else -0.6), 38.8)
                       for col in range(1, nx - 1, 2)]
    return dict(
        kx_field=k, ky_field=k * 0.6, kz_field=k * 0.1,
        active=active, bc_mask=fixed, bc_values=bc_values,
        R_field=recharge, dx=dx, dy=dy, dz=dz,
        wel_entries=entries, drn_entries=drn_entries, riv_entries=riv_entries,
    )

