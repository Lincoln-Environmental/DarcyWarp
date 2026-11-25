"""Conventional DIS geometry and production faces; no terrain metrics/oracle."""
import numpy as np
from DARCY_WARP_PACKAGE.physics.inputs_3d import binary_mask


def layered_geometry(*, z_interfaces, shape, dx, dy):
    """Construct consistent layer geometry for a conventional structured DIS grid.

    :param z_interfaces: Finite interface elevations, top to bottom,
        shape (nz+1, ny, nx), in length units.
    :param shape: Required nonempty cell-grid shape (nz, ny, nx).
    :param dx: Positive x cell width, length units.
    :param dy: Positive y cell width, length units.
    :returns: Owned top, bottom, thickness, centre elevation and cell-volume grids.
    :raises ValueError: If geometry is invalid, pinched out or not representable.
    """
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError('Layered geometry requires a nonempty (nz,ny,nx) cell shape')
    z = np.asarray(z_interfaces, dtype=np.float64)
    if z.shape != (shape[0] + 1, shape[1], shape[2]) or not np.isfinite(z).all():
        raise ValueError('Finite z_interfaces must have shape (nz+1,ny,nx)')
    if any(np.ndim(value) != 0 or not np.isfinite(value) or value <= 0. for value in (dx, dy)):
        raise ValueError('Positive finite scalar dx and dy required')
    with np.errstate(over='ignore', invalid='ignore'):
        thickness = z[:-1] - z[1:]
        centres = 0.5 * z[:-1] + 0.5 * z[1:]
        volumes = np.asarray(np.longdouble(dx) * np.longdouble(dy) * thickness, dtype=float)
    if not np.isfinite(thickness).all() or np.any(thickness <= 0.):
        raise ValueError('All layered cells require positive finite thickness; pinch-outs are unsupported')
    if not np.isfinite(centres).all() or not np.isfinite(volumes).all() or np.any(volumes <= 0.):
        raise ValueError('Unrepresentable layered centres or volumes')
    return dict(tops=z[:-1].copy(), bottoms=z[1:].copy(), thickness=thickness,
                centre_z=centres, volume=volumes)


def build_layered_dis_faces(*, z_interfaces, kx, ky, kz, active, dx, dy):
    """Build six reciprocal arrays from two owning half-cell resistances.

    Extended exponent range is used only during host-side construction, so
    intermediate K*T products cannot overflow a representable FP64 face.
    These production arrays are independent of the SciPy reference assembly.

    :param z_interfaces: Layer interfaces, top to bottom, shape (nz+1, ny, nx).
    :param kx: Positive active x conductivity grid, length/time.
    :param ky: Positive active y conductivity grid, length/time.
    :param kz: Positive active vertical conductivity grid, length/time.
    :param active: Exact binary cell mask.
    :param dx: Positive uniform x cell width, length units.
    :param dy: Positive uniform y cell width, length units.
    :returns: Six reciprocal FP64 conductance grids in volume/time/length
        and the associated layer geometry.
    :raises ValueError: For invalid geometry, conductivity or conductance.
    """
    active = binary_mask(value=active, name='active')
    shape = active.shape
    geometry = layered_geometry(z_interfaces=z_interfaces, shape=shape, dx=dx, dy=dy)
    thickness = geometry['thickness'].astype(np.longdouble)
    faces = [np.zeros(shape, dtype=float) for unused in range(6)]
    for axis, field, distance, width, p in (
        (2, kx, dx, np.longdouble(dy), 0),
        (1, ky, dy, np.longdouble(dx), 2),
        (0, kz, 1., np.longdouble(dx) * dy, 4),
    ):
        conductivity = np.asarray(field, dtype=np.longdouble)
        if conductivity.shape != shape or not np.isfinite(conductivity).all() or np.any(conductivity[np.asarray(active) != 0] <= 0.):
            raise ValueError('Positive finite active Cartesian conductivity required')
        l, u = [slice(None)] * 3, [slice(None)] * 3
        l[axis], u[axis] = slice(None, -1), slice(1, None)
        l, u = tuple(l), tuple(u)
        connected = (np.asarray(active)[l] != 0) & (np.asarray(active)[u] != 0)
        kl,kr = conductivity[l][connected],conductivity[u][connected]
        tl,tr = thickness[l][connected],thickness[u][connected]
        if axis:
            resistance = np.longdouble(.5)*distance/(width*kl*tl)+np.longdouble(.5)*distance/(width*kr*tr)
        else:
            resistance = np.longdouble(.5)*tl/(np.longdouble(width)*kl)+np.longdouble(.5)*tr/(np.longdouble(width)*kr)
        with np.errstate(over='ignore', under='ignore'):
            coefficient = np.asarray(1./resistance, dtype=float)
        if np.any(~np.isfinite(coefficient)) or np.any(coefficient <= 0.):
            raise ValueError('Unrepresentable layered face conductance')
        faces[p][l][connected] = coefficient
        faces[p + 1][u][connected] = coefficient
    return tuple(faces), geometry
