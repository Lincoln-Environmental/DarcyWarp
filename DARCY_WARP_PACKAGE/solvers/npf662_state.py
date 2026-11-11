"""NPF transient state with the 2D workspace structure/value refresh split.

Like NewtonOperatorWorkspace2D, this owns timestep snapshots without building
an independent sparse operator. The CPU reference is constructed lazily only
when an independent audit or an uncertified grounding case requires it.
"""
import numpy as np

from DARCY_WARP_PACKAGE.validation.head_checks import inspect_heads
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
        case.validate()
        changed = any(not np.array_equal(saved, getattr(case, name))
            for saved, name in zip(self.material, ('kx', 'ky', 'kz')))
        if changed and self.operator_revision == model._session.operator_revision:
            raise ValueError('Conductivity changed without refreshing physical faces; use update_K_fields')
        if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0.:
            raise ValueError('Positive finite transient dt required')
        self.case = case
        self.free = (case.active != 0)&(case.fixed == 0)
        self.dt = float(dt)
        self.previous = np.asarray(previous, dtype=float).copy()
        if self.previous.shape != self.shape or not inspect_heads(head=self.previous, active=case.active,
                fixed=np.zeros_like(case.fixed), values=self.previous)['input_valid']:
            raise ValueError('Invalid previous transient head')
        self.ss = np.broadcast_to(np.asarray(ss, dtype=float), self.shape).copy()
        self.sy = np.broadcast_to(np.asarray(sy, dtype=float), self.shape).copy()
        if any(np.any(~np.isfinite(value)) or np.any(value < 0.) for value in (self.ss, self.sy)):
            raise ValueError('Finite nonnegative Ss and Sy required')
        if np.any(self.sy > 1.):
            raise ValueError('Specific yield must not exceed one')
        # Filled by the persistent device storage kernel during workspace load.
        self.preconditioner_storage = None
        self.material = tuple(np.asarray(getattr(case, name)).copy() for name in ('kx', 'ky', 'kz'))
        self.operator_revision = model._session.operator_revision
        self._audit = None
        self.refresh_count += 1

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
