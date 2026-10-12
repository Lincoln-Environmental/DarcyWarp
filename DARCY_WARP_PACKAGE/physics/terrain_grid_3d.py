"""Fixed-depth terrain coordinates, following RichardsWarp's audited grid.

zeta = S(x,y)-z has unit Jacobian. Arrays are owned, read-only snapshots;
geometry changes require a new grid, preventing silent metric cache mutation.
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
    """Parallel terrain contacts with common depth interfaces [m].

    Supports complete active columns; pinches, internal holes and locally
    varying thickness are outside this transformation. dx/dy are map spacing.
    """
    surface_elevation: np.ndarray
    zeta_faces: np.ndarray
    dx: float
    dy: float
    active_columns: np.ndarray | None = None

    def __post_init__(self):
        surface = np.array(self.surface_elevation, dtype=np.float64, copy=True)
        faces = np.array(self.zeta_faces, dtype=np.float64, copy=True)
        if surface.ndim != 2 or min(surface.shape) < 1 or not np.all(np.isfinite(surface)):
            raise ValueError("surface_elevation must be a finite nonempty (ny,nx) DEM")
        if faces.ndim != 1 or faces.size < 2 or faces[0] != 0 or not np.all(np.isfinite(faces)) or np.any(np.diff(faces) <= 0):
            raise ValueError("zeta_faces must be finite, one-dimensional and strictly increasing from zero; variable column interfaces are unsupported")
        if not all(np.isfinite(v) and v > 0 for v in (self.dx, self.dy)):
            raise ValueError("terrain dx/dy must be positive and finite")
        active = np.ones(surface.shape, dtype=bool) if self.active_columns is None else np.array(self.active_columns, copy=True)
        if active.shape != surface.shape or not np.all((active == 0) | (active == 1)):
            raise ValueError("active_columns must be a binary DEM-shaped mask")
        active = active.astype(bool)
        if faces.size == 2 and np.any(surface[active] != surface[active][0]) if np.any(active) else False:
            raise ValueError("sloping terrain requires at least two layers to reconstruct depth gradients")
        for name, value in (("surface_elevation", surface), ("zeta_faces", faces), ("active_columns", active)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def shape(self):
        return (self.zeta_faces.size - 1, *self.surface_elevation.shape)

    @property
    def thickness(self):
        return np.diff(self.zeta_faces)[:, None, None]

    @property
    def active(self):
        return np.broadcast_to(self.active_columns, self.shape).astype(np.int32)

    @property
    def tops(self):
        return self.surface_elevation[None] - self.zeta_faces[:-1, None, None]

    @property
    def bottoms(self):
        return self.surface_elevation[None] - self.zeta_faces[1:, None, None]

    @property
    def elevations(self):
        return 0.5 * (self.tops + self.bottoms)

    @property
    def volumes(self):
        return np.broadcast_to(self.dx * self.dy * self.thickness, self.shape)


class TerrainMetrics3D:
    """Cached consistent metric and gradient operators, shared geometry only."""
    def __init__(self, *, grid):
        self.grid = grid
        nz, ny, nx = grid.shape
        ncol = ny * nx
        gx, self.x_coefficients = derivative_matrix(active=grid.active_columns, spacing=grid.dx, axis=1)
        gy, self.y_coefficients = derivative_matrix(active=grid.active_columns, spacing=grid.dy, axis=0)
        self.gx = sparse.kron(sparse.eye(nz), gx, format="csr")
        self.gy = sparse.kron(sparse.eye(nz), gy, format="csr")
        # Subtracting a datum avoids cancellation of large elevations.
        surface = grid.surface_elevation - grid.surface_elevation.flat[0]
        self.sx = (gx @ surface.ravel()).reshape(ny, nx)
        self.sy = (gy @ surface.ravel()).reshape(ny, nx)
        self.sxf = np.diff(surface, axis=1) / grid.dx
        self.syf = np.diff(surface, axis=0) / grid.dy
        centres = 0.5 * (grid.zeta_faces[:-1] + grid.zeta_faces[1:])
        self.depth_distances = np.diff(centres)
        gz = sparse.lil_matrix((nz, nz))
        for layer in range(nz - 1):
            gz[layer, layer] = -1 / self.depth_distances[layer]
            gz[layer, layer + 1] = 1 / self.depth_distances[layer]
        if nz > 1:
            gz[-1] = gz[-2]
        self.gz = sparse.kron(gz.tocsr(), sparse.eye(ncol), format="csr")


def storage_potential_3d(*, head, bottoms, thickness, sy, ss):
    """Stored volume per horizontal map area [m], without numerical floors."""
    relative = np.asarray(head) - bottoms
    wet = np.clip(relative, 0, thickness)
    return sy * wet + ss * (0.5 * wet**2 + thickness * np.maximum(relative - thickness, 0))
