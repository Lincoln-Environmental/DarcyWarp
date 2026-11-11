"""NPF transient state with the 2D workspace structure/value refresh split.

Like NewtonOperatorWorkspace2D, this owns timestep snapshots without building
an independent sparse operator. The CPU reference is constructed lazily only
when an independent audit or an uncertified grounding case requires it.
"""
import numpy as np

from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference


class NPF662TransientState:
    device_storage_state = True

    def __init__(self, *, model, case, previous, dt, ss, sy):
        self.device = model.device
        self.shape = case.active.shape
        self.structure_names = tuple(name for name in case.__dataclass_fields__
            if name not in ('source', 'values', 'initial', 'kx', 'ky', 'kz', 'cell_type'))
        self.structure = tuple(np.asarray(getattr(case, name)).copy() for name in self.structure_names)
        self.material = tuple(np.asarray(getattr(case, name)).copy() for name in ('kx', 'ky', 'kz'))
        self.operator_revision = model._session.operator_revision
        self.refresh_count = 0
        self.refresh(model=model, case=case, previous=previous, dt=dt, ss=ss, sy=sy)

    def compatible(self, *, model, case):
        return self.device == model.device and self.shape == case.active.shape and all(
            np.array_equal(saved, np.asarray(getattr(case, name)))
            for name, saved in zip(self.structure_names, self.structure))

    def refresh(self, *, model, case, previous, dt, ss, sy):
        changed = any(not np.array_equal(saved, getattr(case, name))
            for saved, name in zip(self.material, ('kx', 'ky', 'kz')))
        if changed and self.operator_revision == model._session.operator_revision:
            raise ValueError('Conductivity changed without refreshing physical faces; use update_K_fields')
        if not self.refresh_count or changed:
            case.validate()
        else:
            # Geometry, masks and K have already been compared with owned
            # snapshots. Validate every live timestep field, without rebuilding
            # layered centres/volumes or scanning unchanged physical fields.
            for name in ('values', 'source', 'initial'):
                value = np.asarray(getattr(case, name))
                if value.shape != self.shape or not np.isfinite(value).all():
                    raise ValueError(f'Invalid {name}')
            if np.any(case.source[case.active == 0] != 0.):
                raise ValueError('Inactive source would be lost')
            types = np.asarray(case.cell_type)
            if types.ndim != 0 and types.shape != self.shape:
                raise ValueError('cell_type must be scalar or full-grid 0/1 array')
            if np.any((types != 0)&(types != 1)):
                raise ValueError('Only ICELLTYPE 0/1 supported')
            if np.any((case.fixed != 0)&(case.cell_types != 0)&(case.values < case.bottoms)):
                raise ValueError('Convertible CHD head below cell bottom is invalid for MF6 6.6.2')
        if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0.:
            raise ValueError('Positive finite transient dt required')
        self.case = case
        if not self.refresh_count:
            self.free = (case.active != 0)&(case.fixed == 0)
        self.dt = float(dt)
        self.previous = np.asarray(previous, dtype=float).copy()
        if self.previous.shape != self.shape or np.any(
                (~np.isfinite(self.previous)|(np.abs(self.previous) >= 1e20))&(case.active != 0)):
            raise ValueError('Invalid previous transient head')
        ss_owned = np.array(ss, dtype=float, copy=True)
        sy_owned = np.array(sy, dtype=float, copy=True)
        if any(np.any(~np.isfinite(value)) or np.any(value < 0.) for value in (ss_owned, sy_owned)):
            raise ValueError('Finite nonnegative Ss and Sy required')
        if np.any(sy_owned > 1.):
            raise ValueError('Specific yield must not exceed one')
        self.ss = np.broadcast_to(ss_owned, self.shape)
        self.sy = np.broadcast_to(sy_owned, self.shape)
        self.ss_scalar = float(ss_owned) if ss_owned.ndim == 0 else None
        self.sy_scalar = float(sy_owned) if sy_owned.ndim == 0 else None
        # Filled by the persistent device storage kernel during workspace load.
        self.preconditioner_storage_device = None
        self._preconditioner_storage = None
        if changed:
            self.material = tuple(np.asarray(getattr(case, name)).copy() for name in ('kx', 'ky', 'kz'))
        self.operator_revision = model._session.operator_revision
        self._audit = None
        self.refresh_count += 1

    @property
    def preconditioner_storage(self):
        """Owned host snapshot, materialized only for construction or audit."""
        if self._preconditioner_storage is None and self.preconditioner_storage_device is not None:
            self._preconditioner_storage = self.preconditioner_storage_device.numpy().reshape(self.shape).copy()
        return self._preconditioner_storage

    def independent(self):
        if self._audit is None:
            self._audit = NPF662TransientReference(case=self.case, previous=self.previous,
                dt=self.dt, ss=self.ss, sy=self.sy)
        return self._audit

    def check_components(self):
        potential = (self.case.fixed != 0)|((self.ss > 0.)|((self.sy > 0.)&(self.case.cell_types != 0)))
        if np.all(potential[self.case.active != 0]):
            return
        self.independent().check_components()

    def evaluate(self, *, head, **tolerances):
        return self.independent().evaluate(head=head, **tolerances)
