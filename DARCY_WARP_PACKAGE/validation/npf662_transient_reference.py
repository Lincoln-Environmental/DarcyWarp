"""Independent NumPy NPF/STO residuals for the MF6 6.6.2 default storage law."""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference, quadratic_saturation_derivative
from DARCY_WARP_PACKAGE.validation.npf662_metrics import stable_rms, stable_sum
from DARCY_WARP_PACKAGE.validation.head_checks import inspect_heads


class NPF662TransientReference(NPF662Reference):
    def __init__(self, *, case, previous, dt, ss, sy):
        super().__init__(case=case)
        if np.ndim(dt) != 0 or not np.isfinite(dt) or dt <= 0.:
            raise ValueError('Positive finite transient dt required')
        self.dt = float(dt)
        self.previous = np.asarray(previous, dtype=float).copy()
        if not inspect_heads(head=self.previous, active=case.active,
                fixed=np.zeros_like(case.fixed), values=self.previous)['input_valid']:
            raise ValueError('Invalid previous transient head')
        self.ss = np.broadcast_to(np.asarray(ss, dtype=float), self.shape).copy()
        self.sy = np.broadcast_to(np.asarray(sy, dtype=float), self.shape).copy()
        if any(np.any(~np.isfinite(value)) or np.any(value < 0.) for value in (self.ss, self.sy)):
            raise ValueError('Finite nonnegative Ss and Sy required')
        if np.any(self.sy > 1.):
            raise ValueError('Specific yield must not exceed one')
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
        anchors = (self.case.fixed != 0)|((self.ss > 0.)|((self.sy > 0.)&(self.case.cell_types != 0)))
        for label in np.unique(labels[self.case.active.ravel() != 0]):
            if not np.any(anchors.ravel()[labels == label]):
                raise ValueError('Ungrounded transient component without boundary or storage')

    def jacobian(self, *, head):
        return super().jacobian(head=head)+sparse.diags(self.storage(head=head)[1].ravel())

    def evaluate(self, *, head, **tolerances):
        checks, fields = super().evaluate(head=head, **tolerances)
        storage, tangent = self.storage(head=head)
        residual = np.where(self.free, fields['residual']-storage, 0.)
        diagonal = fields['principal_diagonal']+tangent
        equivalent = np.divide(residual, diagonal, out=np.zeros_like(residual), where=diagonal > 0.)
        equivalent[(diagonal <= 0.)&(residual != 0.)] = np.inf
        throughput = checks['gross_throughput']+stable_sum(values=np.abs(storage[self.free]))
        rms = stable_rms(values=residual[self.free])
        hrms = stable_rms(values=equivalent[self.free])
        l1 = stable_sum(values=np.abs(residual[self.free]))
        relative = l1/max(throughput, np.finfo(float).tiny)
        edges = fields['conductance'] > 0.
        graph = sparse.coo_matrix((np.ones(int(edges.sum())), (self.left[edges], self.right[edges])),
            shape=(self.case.active.size, self.case.active.size)).tocsr()
        unused, labels = connected_components(csgraph=graph, directed=False)
        active_labels = set(labels[self.case.active.ravel() != 0])
        anchors = (self.case.fixed != 0)|((tangent > 0.)&self.free)
        grounded = set(labels[anchors.ravel()&(self.case.active.ravel() != 0)])
        ungrounded = len(active_labels-grounded)
        valid = bool(all(np.isfinite(fields[name]).all() for name in
            ('flux', 'conductance', 'dflux_left', 'dflux_right', 'saturation'))
            and np.isfinite(storage).all() and np.isfinite(tangent).all() and np.all(tangent >= 0.)
            and np.isfinite(residual).all() and np.isfinite(diagonal).all()
            and np.isfinite([rms, hrms, l1, relative, throughput]).all())
        checks.update(physical_rms=rms, physical_max=float(np.max(np.abs(residual))),
            head_residual_rms=hrms, residual_l1=l1, relative_cell_imbalance=relative,
            gross_throughput=throughput, signed_imbalance=stable_sum(values=residual[self.free]),
            ungrounded_current_components=ungrounded, physical_metrics_valid=valid,
            storage_accumulation_m3_day=stable_sum(values=storage[self.free]))
        limits = checks['final_acceptance_thresholds']
        checks['independent_accepted'] = bool(checks['input_valid'] and checks['boundary_condition_valid']
            and valid and ungrounded == 0 and hrms <= limits['head_equivalent_rms_m']
            and (rms <= limits['physical_rms_m3_day'] or relative <= limits['relative_cell_imbalance']))
        fields.update(residual=residual, principal_diagonal=diagonal, storage_rate=storage, storage_tangent=tangent)
        return checks, fields
