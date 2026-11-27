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
            if name not in ('source', 'source_gross', 'values', 'initial', 'kx', 'ky', 'kz', 'cell_type',
                'ghb_conductance', 'ghb_head', 'drn_conductance', 'drn_elevation', 'drn_mask',
                'riv_conductance', 'riv_stage', 'riv_bottom', 'riv_mask',
                'drn_entries', 'riv_entries'))
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
        self._model = model
        self._case_top = float(getattr(case, 'top', 0.))
        self._case_cell_type = case.cell_type
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
        # Own the equation at the solve boundary, before any later model
        # update can replace or mutate its host fields. Lazy reference assembly
        # must never reconstruct historical inputs from the live model.
        from copy import deepcopy
        self._audit_case = deepcopy(case)
        self._audit_revisions = self._input_revisions()
        self._activation_watch = None
        self._entry_lists_revision = None
        self._drn_entries_list = None
        self._riv_entries_list = None
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

        The snapshot is captured at refresh, including sparse boundary entry
        definitions. Subsequent session revisions belong to a different
        equation and cannot invalidate or rewrite this historical audit.
        """
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
        storage_potential=(self.case.thicknesses>0.)&(
            (self.ss>0.)|((self.sy>0.)&(self.case.cell_types!=0)))
        potential = (self.case.fixed != 0)|storage_potential
        if self.case.ghb_conductance is not None:
            potential |= self.case.ghb_conductance > 0.
        if np.all(potential[self.case.active != 0]):
            return
        self.independent().check_components()

    def apply_nullspace_datum(self, *, head):
        """Use the independent transient audit to retain a physical datum.

        :param head: Converged heads copied from the device history state.
        :return: Datum-adjusted heads and the audit's component records.
        """
        return self.independent().apply_nullspace_datum(head=head)

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
        drn_list, riv_list = self._boundary_entry_lists()
        convertible_storage = (np.asarray(case.cell_types) != 0) & ((self.ss > 0.) | (self.sy > 0.))
        if drn_list is None and riv_list is None and not np.any(convertible_storage):
            return candidate, []
        dry_convertible = ((np.asarray(case.cell_types) != 0)
            & (np.asarray(case.active) != 0) & (candidate <= case.bottoms))
        flat_storage = convertible_storage & ((candidate <= case.bottoms)
            | ((self.ss == 0.) & (candidate >= case.bottoms + case.thicknesses)))
        if np.any(dry_convertible) or np.any(flat_storage):
            # Storage parameters alone cannot certify a dry tangent or
            # hydraulic connectivity. Use the complete owned physical equation
            # for this exceptional activation decision.
            return self.independent().seed_unanchored_boundary_components(head=candidate)
        labels = self._activation_component_labels()
        free = self.free.ravel()
        fixed = np.zeros(self.shape, dtype=bool)
        fixed[np.asarray(case.fixed) != 0] = True
        ghb = (np.zeros(self.shape) if case.ghb_conductance is None
               else np.broadcast_to(case.ghb_conductance, self.shape))
        source = np.broadcast_to(case.source, self.shape)
        storage_restraint = (self.ss > 0.) | ((self.sy > 0.) & (np.asarray(case.cell_types) != 0))
        flat = candidate.ravel()
        seeds = []
        for label in np.unique(labels[np.asarray(case.active).ravel() != 0]):
            component = (labels == label).reshape(self.shape)
            if np.any(fixed[component]) or np.any(ghb[component] > 0.):
                continue
            if np.any(storage_restraint[component]):
                continue
            free_component = component & (free.reshape(self.shape))
            if self._component_restrained(component, candidate, drn_list, riv_list):
                continue
            # For an unrestrained closed component intercell fluxes cancel.
            # Include disconnected river leakage: it is a physical constant
            # source on this branch, even when the numerical source is zero.
            component_source = np.asarray(source, dtype=np.float64)[free_component]
            net_need = float(np.sum(component_source))
            if riv_list is not None:
                cells = riv_list.flat_cells
                included = free_component.ravel()[cells]
                river_out = riv_list.conductance * (
                    np.maximum(flat[cells], riv_list.third_level) - riv_list.second_level)
                net_need -= float(np.sum(river_out[included]))
            if not np.isfinite(net_need) or net_need <= 0.:
                continue
            threshold_candidates = []
            for entries, package, level in ((drn_list, 'DRN', 'first_level'),
                                            (riv_list, 'RIV', 'third_level')):
                if entries is None:
                    continue
                for entry in range(len(entries)):
                    cell = int(entries.flat_cells[entry])
                    if not free_component.ravel()[cell] or entries.conductance[entry] <= 0.:
                        continue
                    threshold = float(getattr(entries, level)[entry])
                    if flat[cell] <= threshold:
                        threshold_candidates.append((cell, threshold, package))
            if not threshold_candidates:
                continue
            shifts = [(float(np.nextafter(threshold, np.inf) - flat[cell]),
                       cell, float(threshold), package)
                      for cell, threshold, package in threshold_candidates]
            positive_shifts = [record for record in shifts if record[0] > 0. and np.isfinite(record[0])]
            if not positive_shifts:
                continue
            shift, cell, threshold, package = min(positive_shifts)
            candidate[component] += shift
            flat = candidate.ravel()
            # Enforce the intended active branch even when the uniform shift
            # rounds the threshold cell back onto the boundary value.
            guard = 0
            while flat[cell] <= threshold and guard < 64:
                flat[cell] = np.nextafter(flat[cell], np.inf)
                guard += 1
            if flat[cell] <= threshold:
                raise ValueError('Unanchored boundary seeding failed to cross the threshold')
            seeds.append(dict(component=int(label), package=package,
                cell=[int(value) for value in np.unravel_index(cell, self.shape)],
                flat_cell=int(cell), threshold=float(threshold),
                uniform_shift_m=float(shift), component_net_inflow=net_need))
        return candidate, seeds

    def _boundary_entry_lists(self):
        """Reduce the case's dense or sparse package inputs to CSR entry lists."""
        if self._entry_lists_revision != (int(self._session.revision),
                                          int(self._session.topology_revision)):
            from DARCY_WARP_PACKAGE.physics.drn_riv_3d import (csr_from_boundary_field,
                validate_drn_fields, validate_riv_fields)
            case = self.case
            drn_c, drn_e = validate_drn_fields(conductance=case.drn_conductance,
                elevation=case.drn_elevation, mask=case.drn_mask,
                active=case.active, fixed=case.fixed)
            riv_c, riv_s, riv_b = validate_riv_fields(conductance=case.riv_conductance,
                stage=case.riv_stage, bottom=case.riv_bottom, mask=case.riv_mask,
                active=case.active, fixed=case.fixed)
            self._drn_entries_list = (case.drn_entries
                if case.drn_entries is not None
                else (csr_from_boundary_field(conductance=drn_c, levels=[drn_e], shape=self.shape)
                    if drn_c is not None else None))
            self._riv_entries_list = (case.riv_entries
                if case.riv_entries is not None
                else (csr_from_boundary_field(conductance=riv_c, levels=[riv_s, riv_b],
                    shape=self.shape) if riv_c is not None else None))
            self._entry_lists_revision = (int(self._session.revision),
                                          int(self._session.topology_revision))
        return self._drn_entries_list, self._riv_entries_list

    def _component_restrained(self, component, candidate, drn_list, riv_list):
        """Report whether any entry of the component is already on its active branch."""
        flat = np.asarray(candidate, dtype=np.float64).ravel()
        for entries, level in ((drn_list, 'first_level'), (riv_list, 'third_level')):
            if entries is None:
                continue
            cells = entries.flat_cells
            active = (entries.conductance > 0.) & (flat[cells] > getattr(entries, level))
            if np.any(active & component.ravel()[cells]):
                return True
        return False

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
        drn_list, riv_list = self._boundary_entry_lists()
        if drn_list is None and riv_list is None:
            return False
        labels = self._activation_component_labels().reshape(self.shape)
        free = self.free.reshape(self.shape)
        fixed = np.asarray(case.fixed) != 0
        for label in np.unique(labels[np.asarray(case.active) != 0]):
            component = labels == label
            if np.any(fixed[component]) or np.any(ghb[component] > 0.):
                continue
            # Positive storage parameters may have a zero dry tangent. Such
            # components must remain watched until the upcoming physical state
            # establishes actual restraint.
            if np.any(storage_restraint[component]) and not np.any(
                    (np.asarray(case.cell_types)[component] != 0)):
                continue
            occupied = np.zeros(self.case.active.size, dtype=bool)
            for entries in (drn_list, riv_list):
                if entries is not None:
                    occupied[entries.flat_cells[entries.conductance > 0.]] = True
            if np.any(occupied.reshape(self.shape) & free & component):
                return True
        return False

    def evaluate(self, *, head, **tolerances):
        return self.independent().evaluate(head=head, **tolerances)
