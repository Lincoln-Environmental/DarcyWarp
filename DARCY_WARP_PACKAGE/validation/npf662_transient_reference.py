"""Independent NumPy NPF/STO residuals for the MF6 6.6.2 default storage law."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference, quadratic_saturation_derivative
from DARCY_WARP_PACKAGE.validation.npf662_metrics import (
    balanced_nullspace_components, exact_zero_equation_rows, stable_rms,
    stable_sum, acceptance_head_rms, local_balance_valid, unbalanced_ungrounded_components,
)
from DARCY_WARP_PACKAGE.validation.head_checks import invalid_head_values


def storage_previous_heads(*, case, previous, ss, sy):
    """Own the previous state and validate only cells used by STO."""
    head = np.array(previous, dtype=float, copy=True)
    if head.shape != case.active.shape:
        raise ValueError("Invalid previous transient head shape")
    free = (case.active != 0) & (case.fixed == 0)
    contributing = free & ((np.asarray(ss) > 0.0) | ((case.cell_types != 0) & (np.asarray(sy) > 0.0)))
    invalid = invalid_head_values(head=head)
    if np.any(contributing & invalid):
        raise ValueError("Invalid previous transient head on an active free storage cell")
    head[free & invalid] = case.initial[free & invalid]
    head[case.fixed != 0] = case.values[case.fixed != 0]
    head[case.active == 0] = 0.0
    return head


class NPF662TransientReference(NPF662Reference):
    def __init__(self, *, case, previous, dt, ss, sy):
        super().__init__(case=case)
        if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0.:
            raise ValueError('Positive finite transient dt required')
        self.dt = float(dt)
        self.ss = np.broadcast_to(np.asarray(ss, dtype=float), self.shape).copy()
        self.sy = np.broadcast_to(np.asarray(sy, dtype=float), self.shape).copy()
        if any(np.any(~np.isfinite(value)) or np.any(value < 0.) for value in (self.ss, self.sy)):
            raise ValueError('Finite nonnegative Ss and Sy required')
        if np.any(self.sy > 1.):
            raise ValueError('Specific yield must not exceed one')
        self.previous = storage_previous_heads(case=case, previous=previous, ss=self.ss, sy=self.sy)
        self.preconditioner_storage = self.storage(head=self.previous)[1]

    def storage(self, *, head):
        """Return accumulation rate and its tangent, positive into storage.

        :param head: Current total head in canonical cell order.
        """
        c = self.case
        sat, derivative = quadratic_saturation_derivative(head=head, bottom=c.bottoms,
            thickness=c.thicknesses, cell_type=c.cell_types)
        old, unused = quadratic_saturation_derivative(head=self.previous, bottom=c.bottoms,
            thickness=c.thicknesses, cell_type=c.cell_types)
        change = sat-old
        yield_coefficient = np.where(c.cell_types != 0, self.sy, 0.)
        # Difference of Ss*T*s*(h-bottom-T*s/2), factored to retain small changes.
        elastic = self.ss*c.thicknesses*(sat*(head-self.previous)
            +change*(self.previous-c.bottoms-.5*c.thicknesses*(sat+old)))
        gravity = yield_coefficient*c.thicknesses*change
        tangent = self.ss*c.thicknesses*(sat+derivative*(head-c.bottoms-c.thicknesses*sat))
        tangent += yield_coefficient*c.thicknesses*derivative
        factor = c.dx*c.dy/self.dt
        return np.where(self.free, (elastic+gravity)*factor, 0.), np.where(self.free, tangent*factor, 0.)

    def check_components(self):
        n = self.case.active.size
        graph = sparse.coo_matrix((np.ones(self.left.size), (self.left, self.right)), shape=(n, n)).tocsr()
        unused, labels = connected_components(csgraph=graph, directed=False)
        drn_occupied, riv_occupied = self.boundary_occupancy()
        storage_potential=(self.case.thicknesses>0.)&(
            (self.ss>0.)|((self.sy>0.)&(self.case.cell_types!=0)))
        grounded=((self.case.fixed!=0)|(self.ghb_conductance>0.)|
                  (drn_occupied|riv_occupied).reshape(self.shape)|storage_potential)
        unbalanced=unbalanced_ungrounded_components(labels=labels,active=self.case.active,
            grounded=grounded,source=self.case.source)
        if unbalanced:
            label,cells,net_source=unbalanced[0]
            raise ValueError(f'Ungrounded transient component {label} with {cells} cells has '
                             f'nonzero net source={net_source:.17g}')

    def jacobian(self, *, head):
        return super().jacobian(head=head)+sparse.diags(self.storage(head=head)[1].ravel())

    def evaluate(self, *, head, roundoff_acceptance=False, **tolerances):
        checks, fields = super().evaluate(head=head, **tolerances)
        storage, tangent = self.storage(head=head)
        residual = np.where(self.free, fields['residual']-storage, 0.)
        diagonal = fields['principal_diagonal']+tangent
        equivalent = np.divide(residual, diagonal, out=np.zeros_like(residual), where=diagonal > 0.)
        equivalent[(diagonal <= 0.)&(residual != 0.)] = np.inf
        throughput = checks['gross_throughput']+stable_sum(values=np.abs(storage[self.free]))
        equation_throughput = checks['equation_throughput']+stable_sum(values=np.abs(storage[self.free]))
        rms = stable_rms(values=residual[self.free])
        hrms = stable_rms(values=equivalent[self.free])
        l1 = stable_sum(values=np.abs(residual[self.free]))
        relative = l1/max(equation_throughput, np.finfo(float).tiny)
        edges = fields['conductance'] > 0.
        graph = sparse.coo_matrix((np.ones(int(edges.sum())), (self.left[edges], self.right[edges])),
            shape=(self.case.active.size, self.case.active.size)).tocsr()
        unused, labels = connected_components(csgraph=graph, directed=False)
        dry_zero = exact_zero_equation_rows(case=self.case, head=head, free=self.free,
            diagonal=diagonal, residual=residual,
            boundary_throughput=fields['boundary_throughput'], storage=storage)
        active_labels = set(labels[(self.case.active.ravel() != 0) & ~dry_zero.ravel()])
        unused_drain_entries, drain, unused_river_entries, river = self.boundary_terms(head=head)
        boundary_tangent = self.ghb_conductance + drain.diagonal + river.diagonal
        anchors = (self.case.fixed != 0)|(boundary_tangent > 0.)|((tangent > 0.)&self.free)
        grounded_mask = anchors.ravel() & (self.case.active.ravel() != 0)
        grounded = set(labels[grounded_mask])
        ungrounded_labels = active_labels-grounded
        boundary_throughput = fields['boundary_throughput']
        limits = checks['final_acceptance_thresholds']
        acceptance_rms = acceptance_head_rms(case=self.case, free=self.free, residual=residual,
            diagonal=diagonal, equivalent=equivalent, left=self.left, right=self.right,
            conductance=fields['conductance'], boundary_tangent=boundary_tangent,
            boundary_throughput=boundary_throughput, physical_tolerance=limits['physical_rms_m3_day'],
            storage=storage, storage_tangent=tangent)
        nullspaces = balanced_nullspace_components(labels=labels, active=self.case.active,
            grounded=grounded_mask, source=self.case.source,
            left=self.left, right=self.right,
            face_left_derivative=fields['dflux_left'],
            face_right_derivative=fields['dflux_right'],
            boundary_tangent=boundary_tangent, boundary_throughput=boundary_throughput,
            storage=storage, storage_tangent=tangent)
        balanced_labels = ungrounded_labels & set(nullspaces)
        ungrounded = len(ungrounded_labels-balanced_labels)
        valid = bool(all(np.isfinite(fields[name]).all() for name in
            ('flux', 'conductance', 'dflux_left', 'dflux_right', 'saturation'))
            and np.isfinite(storage).all() and np.isfinite(tangent).all() and np.all(tangent >= 0.)
            and np.isfinite(residual).all() and np.isfinite(diagonal).all()
            and np.isfinite([rms, acceptance_rms, l1, relative, throughput]).all())
        checks.update(physical_rms=rms, physical_max=float(np.max(np.abs(residual))),
            head_residual_rms=hrms, acceptance_head_residual_rms=acceptance_rms, residual_l1=l1, relative_cell_imbalance=relative,
            gross_throughput=throughput, equation_throughput=equation_throughput, signed_imbalance=stable_sum(values=residual[self.free]),
            ungrounded_current_components=ungrounded,
            balanced_nullspace_components=len(balanced_labels),
            balanced_nullspace_cells=int(sum(nullspaces[label].size for label in balanced_labels)),
            exact_zero_equation_rows=int(dry_zero.sum()),
            exact_dry_zero_rows=int(dry_zero.sum()),
            physical_metrics_valid=valid,
            storage_accumulation_m3_day=stable_sum(values=storage[self.free]))
        checks['independent_accepted'] = bool(checks['input_valid'] and checks['boundary_condition_valid']
            and valid and ungrounded == 0 and acceptance_rms <= limits['head_equivalent_rms_m']
            and (rms <= limits['physical_rms_m3_day'] or relative <= limits['relative_cell_imbalance']))
        checks['local_balance_valid'] = None
        if checks['independent_accepted'] or roundoff_acceptance:
            from DARCY_WARP_PACKAGE.physics.ghb_3d import npf_ghb_component_anchor
            component_anchor = npf_ghb_component_anchor(conductance=boundary_tangent + tangent,
                active=self.case.active, fixed=self.case.fixed, left=self.left, right=self.right,
                face_conductance=fields['conductance'])
            checks['local_balance_valid'] = local_balance_valid(case=self.case, free=self.free, left=self.left,
                right=self.right, head=head, residual=residual, flux=fields['flux'], conductance=fields['conductance'],
                ghb_conductance=boundary_tangent, ghb_exchange=(fields['ghb_outflow']+
                    fields['drn_outflow']+fields['riv_outflow']), component_anchor=component_anchor,
                thresholds=limits, storage=storage, tangent=tangent,
                boundary_throughput=fields['boundary_throughput'])
            checks['independent_accepted'] &= checks['local_balance_valid']
        if roundoff_acceptance:
            from .npf662_metrics import apply_roundoff_acceptance
            apply_roundoff_acceptance(checks=checks,certificate=self.certify_roundoff(
                head=head,residual=residual,conductance=fields['conductance'],storage=storage,tangent=tangent))
        fields.update(residual=residual, principal_diagonal=diagonal, storage_rate=storage, storage_tangent=tangent)
        return checks, fields
