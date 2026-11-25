"""NPF transient state with the 2D workspace structure/value refresh split.

Like NewtonOperatorWorkspace2D, this owns timestep snapshots without building
an independent sparse operator. The CPU reference is constructed lazily only
when an independent audit or an uncertified grounding case requires it.
"""
import numpy as np
from dataclasses import replace
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from DARCY_WARP_PACKAGE.validation.npf662_transient_reference import NPF662TransientReference, storage_previous_heads
from DARCY_WARP_PACKAGE.physics.ghb_3d import validate_ghb_fields


class NPF662TransientState:
    device_storage_state = True

    def __init__(self, *, model, case, previous, dt, ss, sy):
        self.device = model.device
        self.shape = case.active.shape
        self._session = model._session
        self._audit = None
        self._audit_case = None
        self._audit_revisions = None
        self.structure_names = tuple(name for name in case.__dataclass_fields__
            if name not in ('source', 'values', 'initial', 'kx', 'ky', 'kz', 'cell_type',
                'ghb_conductance', 'ghb_head', 'drn_conductance', 'drn_elevation', 'drn_mask',
                'riv_conductance', 'riv_stage', 'riv_bottom', 'riv_mask'))
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
            case.validate_active_heads()
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
        validate_ghb_fields(conductance=case.ghb_conductance,head=case.ghb_head,
            active=case.active,fixed=case.fixed)
        from DARCY_WARP_PACKAGE.physics.drn_riv_3d import validate_drn_fields, validate_riv_fields
        validate_drn_fields(conductance=case.drn_conductance, elevation=case.drn_elevation,
            mask=case.drn_mask, active=case.active, fixed=case.fixed)
        validate_riv_fields(conductance=case.riv_conductance, stage=case.riv_stage,
            bottom=case.riv_bottom, mask=case.riv_mask, active=case.active, fixed=case.fixed)
        if case.ghb_conductance is not None and np.any((np.asarray(case.ghb_conductance)>0.)
                &(case.cell_types!=0)&(np.asarray(case.ghb_head)<case.bottoms)):
            raise ValueError('Convertible GHB head below cell bottom is invalid for MF6 6.6.2')
        self.case = case
        self._session = model._session
        if not self.refresh_count:
            self.free = (case.active != 0)&(case.fixed == 0)
        self.dt = float(dt)
        ss_owned = np.array(ss, dtype=float, copy=True)
        sy_owned = np.array(sy, dtype=float, copy=True)
        if any(np.any(~np.isfinite(value)) or np.any(value < 0.) for value in (ss_owned, sy_owned)):
            raise ValueError('Finite nonnegative Ss and Sy required')
        if np.any(sy_owned > 1.):
            raise ValueError('Specific yield must not exceed one')
        self.ss = np.broadcast_to(ss_owned, self.shape)
        self.sy = np.broadcast_to(sy_owned, self.shape)
        self.previous = storage_previous_heads(case=case, previous=previous, ss=self.ss, sy=self.sy)
        self.ss_scalar = float(ss_owned) if ss_owned.ndim == 0 else None
        self.sy_scalar = float(sy_owned) if sy_owned.ndim == 0 else None
        # Filled by the persistent device storage kernel during workspace load.
        self.preconditioner_storage_device = None
        self._preconditioner_storage = None
        if changed:
            self.material = tuple(np.asarray(getattr(case, name)).copy() for name in ('kx', 'ky', 'kz'))
        self.operator_revision = model._session.operator_revision
        self._audit = None
        self._audit_case = None
        self._audit_revisions = None
        self._activation_watch = None
        self.refresh_count += 1

    @property
    def preconditioner_storage(self):
        """Owned host snapshot, materialized only for construction or audit."""
        if self._preconditioner_storage is None and self.preconditioner_storage_device is not None:
            self._preconditioner_storage = self.preconditioner_storage_device.numpy().reshape(self.shape).copy()
        return self._preconditioner_storage

    def _input_revisions(self):
        """Identify the accepted equation inputs through session revisions."""
        session = self._session
        return (int(session.revision), int(session.operator_revision),
                int(session.topology_revision))

    def _owned_audit_case(self):
        """Snapshot every solver-owned input the independent audit may read.

        ``case_from_model`` aliases live session arrays for the source,
        prescribed values and boundary fields. The audit must stay valid for
        the accepted state even after later model updates, so this snapshot is
        rebuilt whenever a session revision changes and always owns copies.
        """
        revisions = self._input_revisions()
        if self._audit_case is None or revisions != self._audit_revisions:
            case = self.case
            frozen_inputs = ('source', 'values', 'initial', 'kx', 'ky', 'kz',
                'ghb_conductance', 'ghb_head', 'drn_conductance', 'drn_elevation', 'drn_mask',
                'riv_conductance', 'riv_stage', 'riv_bottom', 'riv_mask', 'cell_type')
            owned = {name: (None if getattr(case, name) is None
                            else np.array(getattr(case, name), copy=True))
                     for name in frozen_inputs}
            self._audit_case = replace(case, **owned)
            self._audit_revisions = revisions
        return self._audit_case

    def independent(self):
        """Return the lazily constructed independent transient audit.

        The audit reads an owned snapshot of the accepted inputs, so later
        recharge, well, boundary or prescribed-value updates cannot silently
        re-anchor an already accepted historical state.
        """
        audit_case = self._owned_audit_case()
        if self._audit is None or self._audit.case is not audit_case:
            self._audit = NPF662TransientReference(case=audit_case, previous=self.previous,
                dt=self.dt, ss=self.ss, sy=self.sy)
        return self._audit

    def check_components(self):
        potential = (self.case.fixed != 0)|((self.ss > 0.)|((self.sy > 0.)&(self.case.cell_types != 0)))
        if self.case.ghb_conductance is not None:
            potential |= self.case.ghb_conductance > 0.
        if np.all(potential[self.case.active != 0]):
            return
        self.independent().check_components()

    def _activation_component_labels(self):
        """Build positive-conductance connected-component labels once."""
        if self._activation_watch is None:
            case = self.case
            ids = np.arange(case.active.size).reshape(self.shape)
            left = []
            right = []
            for axis, conductivity in ((2, case.kx), (1, case.ky), (0, case.kz)):
                lower = [slice(None)] * 3
                upper = [slice(None)] * 3
                lower[axis], upper[axis] = slice(None, -1), slice(1, None)
                lower, upper = tuple(lower), tuple(upper)
                mask = (case.active[lower] != 0) & (case.active[upper] != 0)
                mask &= (np.asarray(conductivity)[lower] > 0.) & (np.asarray(conductivity)[upper] > 0.)
                left.append(ids[lower][mask])
                right.append(ids[upper][mask])
            left = np.concatenate(left)
            right = np.concatenate(right)
            graph = coo_matrix((np.ones(left.size), (left, right)),
                shape=(case.active.size, case.active.size)).tocsr()
            unused, labels = connected_components(csgraph=graph, directed=False)
            self._activation_watch = labels
        return self._activation_watch

    def seed_unanchored_boundary_components(self, *, head):
        """Move only unanchored initial guesses to a needed switching threshold.

        A component without prescribed heads, GHB, active package restraint or
        storage potential cannot absorb a positive source until a switching
        boundary opens: no grounded face leaves the component, so its component
        net source equals its residual. A uniform head shift leaves confined
        intercell flows unchanged and opens the nearest available monotone
        boundary branch without pinning any groundwater cell or adding
        fictitious conductance or storage.

        :param head: Initial total-head field for this solve.
        :return: A copied head field and component/threshold seed records.
        """
        candidate = np.array(head, dtype=np.float64, copy=True)
        case = self.case
        drn_active = (case.drn_conductance is not None
                      and np.any(np.broadcast_to(case.drn_conductance, self.shape) > 0.))
        riv_active = (case.riv_conductance is not None
                      and np.any(np.broadcast_to(case.riv_conductance, self.shape) > 0.))
        if not drn_active and not riv_active:
            return candidate, []
        labels = self._activation_component_labels()
        free = self.free.ravel()
        fixed = np.zeros(self.shape, dtype=bool)
        fixed[np.asarray(case.fixed) != 0] = True
        ghb = (np.zeros(self.shape) if case.ghb_conductance is None
               else np.broadcast_to(case.ghb_conductance, self.shape))
        drn = (np.zeros(self.shape) if case.drn_conductance is None
               else np.broadcast_to(case.drn_conductance, self.shape))
        drn_elevation = (np.zeros(self.shape) if case.drn_elevation is None
                         else np.broadcast_to(case.drn_elevation, self.shape))
        riv = (np.zeros(self.shape) if case.riv_conductance is None
               else np.broadcast_to(case.riv_conductance, self.shape))
        riv_stage = (np.zeros(self.shape) if case.riv_stage is None
                     else np.broadcast_to(case.riv_stage, self.shape))
        riv_bottom = (np.zeros(self.shape) if case.riv_bottom is None
                      else np.broadcast_to(case.riv_bottom, self.shape))
        source = np.broadcast_to(case.source, self.shape)
        storage_restraint = (self.ss > 0.) | ((self.sy > 0.) & (np.asarray(case.cell_types) != 0))
        seeds = []
        for label in np.unique(labels[np.asarray(case.active).ravel() != 0]):
            component = (labels == label).reshape(self.shape)
            if np.any(fixed[component]) or np.any(ghb[component] > 0.):
                continue
            if np.any(storage_restraint[component]):
                continue
            if np.any((drn[component] > 0.) & (candidate[component] > drn_elevation[component])):
                continue
            if np.any((riv[component] > 0.) & (candidate[component] > riv_bottom[component])):
                continue
            free_component = component & (free.reshape(self.shape))
            net_need = float(np.sum(np.asarray(source, dtype=np.float64)[free_component]))
            if not np.isfinite(net_need) or net_need <= 0.:
                continue
            threshold_candidates = []
            for cell in zip(*np.nonzero(free_component & (drn > 0.))):
                elevation = drn_elevation[cell]
                if candidate[cell] <= elevation:
                    threshold_candidates.append((cell, elevation, 'DRN'))
            for cell in zip(*np.nonzero(free_component & (riv > 0.))):
                bottom = riv_bottom[cell]
                if candidate[cell] <= bottom:
                    threshold_candidates.append((cell, bottom, 'RIV'))
            if not threshold_candidates:
                continue
            shifts = [(float(np.nextafter(threshold, np.inf) - candidate[cell]),
                       cell, float(threshold), package)
                      for cell, threshold, package in threshold_candidates]
            positive_shifts = [record for record in shifts if record[0] > 0. and np.isfinite(record[0])]
            if not positive_shifts:
                continue
            shift, cell, threshold, package = min(positive_shifts)
            candidate[component] += shift
            # Enforce the intended active branch even when the uniform shift
            # rounds the threshold cell back onto the boundary value.
            guard = 0
            while candidate[cell] <= threshold and guard < 64:
                candidate[cell] = np.nextafter(candidate[cell], np.inf)
                guard += 1
            if candidate[cell] <= threshold:
                raise ValueError('Unanchored boundary seeding failed to cross the threshold')
            seeds.append(dict(component=int(label), package=package,
                cell=[int(value) for value in cell], threshold=float(threshold),
                uniform_shift_m=float(shift), component_net_inflow=net_need))
        return candidate, seeds

    def activation_watch_needed(self):
        """Report whether some component may lose every restraint between steps.

        Only components without prescribed heads, GHB or storage potential can
        become unrestrained; a drain or river is then their only possible
        grounding and the seed guard applies.
        """
        case = self.case
        storage_restraint = (self.ss > 0.) | ((self.sy > 0.) & (np.asarray(case.cell_types) != 0))
        ghb = (np.zeros(self.shape) if case.ghb_conductance is None
               else np.broadcast_to(case.ghb_conductance, self.shape))
        drn = (np.zeros(self.shape) if case.drn_conductance is None
               else np.broadcast_to(case.drn_conductance, self.shape))
        riv = (np.zeros(self.shape) if case.riv_conductance is None
               else np.broadcast_to(case.riv_conductance, self.shape))
        if not np.any(drn > 0.) and not np.any(riv > 0.):
            return False
        labels = self._activation_component_labels().reshape(self.shape)
        free = self.free.reshape(self.shape)
        fixed = np.asarray(case.fixed) != 0
        for label in np.unique(labels[np.asarray(case.active) != 0]):
            component = labels == label
            if np.any(fixed[component]) or np.any(ghb[component] > 0.):
                continue
            if np.any(storage_restraint[component]):
                continue
            if np.any(((drn[component] > 0.) | (riv[component] > 0.)) & free[component]):
                return True
        return False

    def evaluate(self, *, head, **tolerances):
        return self.independent().evaluate(head=head, **tolerances)
