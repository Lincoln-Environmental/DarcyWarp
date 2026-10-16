"""Solver-neutral, hash-checked steady terrain inputs (SI length, days).

Arrays use (layer top-to-bottom, y south-to-north, x west-to-east), C order.
MF6 DIS rows run north-to-south, so adapters MUST reverse the y axis.
Conductivity is diagonal in physical Cartesian x,y,z, not geological axes.
"""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainContact3D


def atomic_json(*, path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=float))
    temporary.replace(path)


@dataclass
class TerrainCase:
    case_id: str
    arrays: dict
    physical: dict
    controls: dict
    contacts: tuple

    @property
    def shape(self):
        return self.arrays['initial'].shape

    def grid(self):
        return TerrainGrid3D(z_interfaces=self.arrays['interfaces'], dx=self.physical['dx'],
            dy=self.physical['dy'], active_columns=self.arrays['active'])

    def digest(self):
        digest = hashlib.sha256()
        metadata = dict(schema=1, physical=self.physical, contacts=[asdict(c) for c in self.contacts])
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        for key in sorted(self.arrays):
            array = np.ascontiguousarray(self.arrays[key])
            digest.update(key.encode())
            digest.update(array.dtype.str.encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def validate(self):
        grid = self.grid()
        a = self.arrays
        if any(not np.all(np.isfinite(value)) for value in a.values()):
            raise ValueError('nonfinite case array')
        if any(a[key].shape != self.shape for key in ('kx','ky','kz','fixed','values','source')):
            raise ValueError('cell array shape mismatch')
        if np.any(a['source'][~grid.active.astype(bool)] != 0):
            raise ValueError('inactive source would be lost')
        if any(np.any(a[key][grid.active.astype(bool)] <= 0) for key in ('kx','ky','kz')):
            raise ValueError('nonpositive physical conductivity')
        for c in self.contacts:
            if a['fixed'][c.cell] or not grid.active[c.cell]:
                raise ValueError('contact must belong to active free cell')
        return dict(active=int(grid.active.sum()), free=int(np.sum(grid.active & (a['fixed']==0))),
            source_total=float(a['source'].sum()), source_free=float(a['source'][a['fixed']==0].sum()),
            source_fixed=float(a['source'][a['fixed']!=0].sum()),
            volume=float(grid.volumes.sum()), minimum_thickness=float(grid.thickness.min()))

    def save(self, *, directory: Path):
        directory.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(directory/'inputs.npz', **self.arrays)
        atomic_json(path=directory/'manifest.json', value=dict(schema=1,case_id=self.case_id,
            input_hash=self.digest(), physical=self.physical,controls=self.controls,
            contacts=[asdict(c) for c in self.contacts], checks=self.validate(),
            coordinates='C-order (k top-down,j south-to-north,i west-to-east); centres x=(i+.5)dx,y=(j+.5)dy',
            units=dict(head='m',conductivity='m/day',source='m3/day',flux='m3/day'),
            source_fixed_semantics='Original source retained; fixed cells are reservoirs, excluded from free-cell balance.'))

    @classmethod
    def load(cls, *, directory: Path):
        m = json.loads((directory/'manifest.json').read_text())
        if m['schema'] != 1:
            raise ValueError('unsupported case schema')
        with np.load(directory/'inputs.npz', allow_pickle=False) as data:
            arrays = {key:data[key] for key in data.files}
        contacts = []
        for record in m['contacts']:
            record['cell'] = tuple(record['cell'])
            contacts.append(TerrainContact3D(**record))
        result = cls(case_id=m['case_id'],arrays=arrays,physical=m['physical'],controls=m['controls'],contacts=tuple(contacts))
        if result.digest() != m['input_hash']:
            raise ValueError('canonical input hash mismatch')
        result.validate()
        return result


def canonical_case(*, size: int, basement_k: float, dx: float):
    # Reuse original constructor, never its demonstration main/warm starts.
    from shanz_terrain_case_study import build_case
    raw = build_case(nx=size,ny=size,dx=dx)
    raw['k'][-1] = basement_k
    grid = TerrainGrid3D(z_interfaces=raw['interfaces'],dx=dx,dy=dx,active_columns=raw['active'])
    metrics = TerrainMetrics3D(grid=grid)
    arrays = dict(interfaces=raw['interfaces'],active=raw['active'],fixed=raw['fixed'],
        values=raw['values'],initial=raw['initial'],source=raw['source'],kx=raw['k'],
        ky=.6*raw['k'],kz=.15*raw['k'],centres=grid.elevations,thickness=grid.thickness,
        volumes=grid.volumes,zx=metrics.zx,zy=metrics.zy)
    return TerrainCase(case_id=f'terrain_{size}_k{basement_k:g}',arrays=arrays,
        physical=dict(dx=dx,dy=dx,floor=.01,steady=True,storage=0,conductivity_frame='Cartesian'),
        controls=dict(head_tol=1e-6,flow_tol=1e-12,relative_tol=1e-8,relaxation=.7,
            outer_cap=2000,krylov_budget=2000,restart=24,max_levels=6,min_coarse_n=16), contacts=raw['contacts'])


def mf6_cell(*, cell, shape):
    k,j,i = cell
    if not all(0 <= index < size for index,size in zip(cell,shape)):
        raise ValueError('cell index outside grid')
    return k,shape[1]-1-j,i


def internal_cell(*, cell, shape):
    return mf6_cell(cell=cell,shape=shape)
