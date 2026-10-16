"""Terrain coordinates with independently varying geological interfaces.

The computational column coordinate s = k + u runs downward through the
geological layers; inside layer k the physical elevation is the linear
blend z(x,y,k+u) = (1-u)*z_interfaces[k] + u*z_interfaces[k+1]. Each
interface is an independent two-dimensional surface: layers may thicken,
thin and slope differently from the ground surface, with the only
requirements that interfaces stay finite, ordered and strictly positive
in thickness on every active column. The legacy parallel form (a DEM plus
common 1-D depth interfaces) is retained exactly as a special case.
Arrays are owned, read-only snapshots; geometry changes require a new
grid, preventing silent metric cache mutation.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse


def derivative_matrix(*, active, spacing, axis):
    """Centered tangential differences, one-sided at exterior/mask edges."""
    shape = active.shape
    ids = np.arange(active.size).reshape(shape)
    minus = np.zeros(shape, dtype=bool)
    plus = np.zeros(shape, dtype=bool)
    here, there = [slice(None)] * 2, [slice(None)] * 2
    here[axis], there[axis] = slice(1, None), slice(None, -1)
    minus[tuple(here)] = active[tuple(there)]
    plus[tuple(there)] = active[tuple(here)]
    minus &= active
    plus &= active
    cm = np.where(minus, np.where(plus, -0.5, -1.0) / spacing, 0.0)
    cp = np.where(plus, np.where(minus, 0.5, 1.0) / spacing, 0.0)
    offset = 1 if axis == 1 else shape[1]
    rows = np.concatenate((ids.ravel(), ids[minus], ids[plus]))
    columns = np.concatenate((ids.ravel(), ids[minus] - offset, ids[plus] + offset))
    values = np.concatenate((-(cm + cp).ravel(), cm[minus], cp[plus]))
    return sparse.coo_matrix((values, (rows, columns)), shape=(active.size, active.size)).tocsr(), (-(cm + cp), cm, cp)


@dataclass(frozen=True)
class TerrainGrid3D:
    """Independently specified geological interface elevations [m].

    z_interfaces[k] is the elevation surface of the top of layer k; a model
    with nz layers provides nz+1 surfaces of shape (ny,nx). Interfaces may
    curve independently (no shared morphology or fixed vertical offset);
    pinches, disappearing units and internal holes remain unsupported and
    must be rejected explicitly. dx/dy are map spacing.
    """
    surface_elevation: np.ndarray | None = None
    zeta_faces: np.ndarray | None = None
    z_interfaces: np.ndarray | None = None
    dx: float = 0.0
    dy: float = 0.0
    active_columns: np.ndarray | None = None

    def __post_init__(self):
        if self.z_interfaces is not None:
            interfaces = np.array(self.z_interfaces, dtype=np.float64, copy=True)
            zeta = None if self.zeta_faces is None else np.array(self.zeta_faces, dtype=np.float64, copy=True)
            surface = None
        elif self.surface_elevation is not None and self.zeta_faces is not None:
            surface = np.array(self.surface_elevation, dtype=np.float64, copy=True)
            zeta = np.array(self.zeta_faces, dtype=np.float64, copy=True)
            if surface.ndim != 2 or min(surface.shape) < 1 or not np.all(np.isfinite(surface)):
                raise ValueError("surface_elevation must be a finite nonempty (ny,nx) DEM")
            if zeta.ndim != 1 or zeta.size < 2 or zeta[0] != 0 or not np.all(np.isfinite(zeta)) or np.any(np.diff(zeta) <= 0):
                raise ValueError("zeta_faces must be finite, one-dimensional and strictly increasing from zero; variable column interfaces are unsupported")
            interfaces = surface[None] - zeta[:, None, None]
        else:
            raise ValueError("provide either z_interfaces (nz+1,ny,nx) or surface_elevation with zeta_faces")
        if interfaces.ndim != 3 or interfaces.shape[0] < 2 or not np.all(np.isfinite(interfaces)):
            raise ValueError("z_interfaces must be finite with shape (nz+1,ny,nx) and at least one layer")
        if np.any(np.diff(interfaces, axis=0) >= 0):
            raise ValueError("z_interfaces must decrease strictly on every column: crossing, touching or inverted geological surfaces and zero or negative layer thickness are unsupported")
        if not all(np.isfinite(v) and v > 0 for v in (self.dx, self.dy)):
            raise ValueError("terrain dx/dy must be positive and finite")
        active = np.ones(interfaces.shape[1:], dtype=bool) if self.active_columns is None else np.array(self.active_columns, copy=True)
        if active.shape != interfaces.shape[1:] or not np.all((active == 0) | (active == 1)):
            raise ValueError("active_columns must be a binary DEM-shaped mask")
        active = active.astype(bool)
        if active.any():
            # A single terrain layer cannot reconstruct the vertical head
            # gradient that drives along-slope flow, so relief needs >= 2
            # layers (the transformation, not the code, is the limitation).
            relief = np.ptp(interfaces[:, active], axis=1) if interfaces.shape[1] else np.zeros(interfaces.shape[0])
            if interfaces.shape[0] == 2 and np.any(relief > 0):
                raise ValueError("sloping terrain requires at least two layers to reconstruct depth gradients")
        zeta = None if zeta is None else self._readonly(zeta)
        object.__setattr__(self, "z_interfaces", self._readonly(interfaces))
        object.__setattr__(self, "zeta_faces", zeta)
        object.__setattr__(self, "surface_elevation", self._readonly(interfaces[0].copy()))
        object.__setattr__(self, "active_columns", self._readonly(active))

    @staticmethod
    def _readonly(value):
        value.setflags(write=False)
        return value

    @property
    def shape(self):
        return (self.z_interfaces.shape[0] - 1, *self.z_interfaces.shape[1:])

    @property
    def thickness(self):
        # Difference against the per-column surface datum: exactly invariant
        # under a constant elevation shift, unlike direct tops-minus-bottoms.
        interfaces = self.z_interfaces
        return (interfaces[:-1] - interfaces[0]) - (interfaces[1:] - interfaces[0])

    @property
    def active(self):
        return np.broadcast_to(self.active_columns, self.shape).astype(np.int32)

    @property
    def tops(self):
        return self.z_interfaces[:-1]

    @property
    def bottoms(self):
        return self.z_interfaces[1:]

    @property
    def elevations(self):
        return 0.5 * (self.tops + self.bottoms)

    @property
    def volumes(self):
        return self.dx * self.dy * self.thickness


class TerrainMetrics3D:
    """Cached consistent metric and gradient operators, shared geometry only.

    Cell-centred slopes of each layer follow the centred surface
    (z_interfaces[k] + z_interfaces[k+1])/2; horizontal-face slopes average
    the two bounding interfaces at the face. The vertical (s-direction)
    gradient uses one-sided physical centre distances, so it is exact for
    heads affine in elevation regardless of thickness variation.
    """
    def __init__(self, *, grid):
        self.grid = grid
        nz, ny, nx = grid.shape
        ncol = ny * nx
        gx, self.x_coefficients = derivative_matrix(active=grid.active_columns, spacing=grid.dx, axis=1)
        gy, self.y_coefficients = derivative_matrix(active=grid.active_columns, spacing=grid.dy, axis=0)
        self.gx = sparse.kron(sparse.eye(nz), gx, format="csr")
        self.gy = sparse.kron(sparse.eye(nz), gy, format="csr")
        # Subtracting a datum avoids cancellation of large elevations.
        interfaces = grid.z_interfaces - grid.z_interfaces.flat[0]
        self.zx_interfaces = np.stack([(gx @ interfaces[m].ravel()).reshape(ny, nx) for m in range(nz + 1)])
        self.zy_interfaces = np.stack([(gy @ interfaces[m].ravel()).reshape(ny, nx) for m in range(nz + 1)])
        self.zx = 0.5 * (self.zx_interfaces[:-1] + self.zx_interfaces[1:])
        self.zy = 0.5 * (self.zy_interfaces[:-1] + self.zy_interfaces[1:])
        self.zxf = 0.5 * (np.diff(interfaces[:-1], axis=2) + np.diff(interfaces[1:], axis=2)) / grid.dx
        self.zyf = 0.5 * (np.diff(interfaces[:-1], axis=1) + np.diff(interfaces[1:], axis=1)) / grid.dy
        # Surface-level aliases matching the legacy parallel-interface view.
        self.sx = self.zx_interfaces[0]
        self.sy = self.zy_interfaces[0]
        self.sxf = self.zxf[0] if nz > 1 else np.diff(interfaces[0], axis=1) / grid.dx
        self.syf = self.zyf[0] if nz > 1 else np.diff(interfaces[0], axis=0) / grid.dy
        thickness = grid.thickness
        self.depth_distances = 0.5 * (thickness[:-1] + thickness[1:])
        rows, columns, values = [], [], []
        columns_index = np.arange(ncol)
        for layer in range(nz - 1):
            distance = 1.0 / self.depth_distances[layer].ravel()
            rows.append(layer * ncol + columns_index)
            columns.append(layer * ncol + columns_index)
            values.append(-distance)
            rows.append(layer * ncol + columns_index)
            columns.append((layer + 1) * ncol + columns_index)
            values.append(distance)
        if nz > 2:
            distance = 1.0 / self.depth_distances[-1].ravel()
            rows.append((nz - 1) * ncol + columns_index)
            columns.append((nz - 2) * ncol + columns_index)
            values.append(-distance)
            rows.append((nz - 1) * ncol + columns_index)
            columns.append((nz - 1) * ncol + columns_index)
            values.append(distance)
        elif nz == 2:
            distance = 1.0 / self.depth_distances[0].ravel()
            rows.append(ncol + columns_index)
            columns.append(columns_index)
            values.append(-distance)
            rows.append(ncol + columns_index)
            columns.append(ncol + columns_index)
            values.append(distance)
        self.gz = sparse.coo_matrix(
            (np.concatenate(values), (np.concatenate(rows), np.concatenate(columns))),
            shape=(nz * ncol, nz * ncol)).tocsr() if nz > 1 else sparse.csr_matrix((ncol, ncol))


def storage_potential_3d(*, head, bottoms, thickness, sy, ss):
    """Stored volume per horizontal map area [m], without numerical floors."""
    relative = np.asarray(head) - bottoms
    wet = np.clip(relative, 0, thickness)
    return sy * wet + ss * (0.5 * wet**2 + thickness * np.maximum(relative - thickness, 0))
