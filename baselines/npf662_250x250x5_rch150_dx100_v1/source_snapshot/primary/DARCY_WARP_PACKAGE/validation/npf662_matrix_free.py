"""Independent layered physical audit without assembling a flow matrix.

Only an unweighted connectivity graph is materialized to verify grounding.
Physical face incidence and diagonals use NumPy bin counts, not sparse A*h.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from DARCY_WARP_PACKAGE.validation.head_checks import inspect_heads
from DARCY_WARP_PACKAGE.validation.npf662_metrics import final_tolerances, stable_rms, stable_sum
from DARCY_WARP_PACKAGE.validation.npf662_reference import quadratic_saturation_derivative


class NPF662MatrixFreeReference:
    def __init__(self, *, case):
        case.validate()
        if case.geometry != 'structured_dis_layered':
            raise ValueError('Matrix-free layered auditor requires canonical layered inputs')
        self.case = case; self.shape = case.active.shape
        self.free = (case.active != 0)&(case.fixed == 0)
        ids = np.arange(case.active.size).reshape(self.shape)
        left = []; right = []; axes = []; coefficients = []
        for axis, field, distance, width in ((2, case.kx, case.dx, case.dy),
                (1, case.ky, case.dy, case.dx), (0, case.kz, 1., case.dx*case.dy)):
            lower = [slice(None)]*3; upper = [slice(None)]*3
            lower[axis] = slice(None, -1); upper[axis] = slice(1, None)
            lower = tuple(lower); upper = tuple(upper)
            mask = (case.active[lower] != 0)&(case.active[upper] != 0)
            kl = np.asarray(field[lower][mask], dtype=np.longdouble)
            kr = np.asarray(field[upper][mask], dtype=np.longdouble)
            tl = np.asarray(case.thicknesses[lower][mask], dtype=np.longdouble)
            tr = np.asarray(case.thicknesses[upper][mask], dtype=np.longdouble)
            resistance = (.5*distance/kl/tl+.5*distance/kr/tr) if axis else (.5*tl/kl+.5*tr/kr)
            width_extended = np.longdouble(case.dx)*case.dy if axis == 0 else np.longdouble(width)
            coefficient = np.asarray(width_extended/resistance, dtype=float)
            if np.any(~np.isfinite(coefficient)) or np.any(coefficient <= 0.):
                raise ValueError('Unrepresentable physical full-thickness conductance')
            left.append(ids[lower][mask]); right.append(ids[upper][mask])
            coefficients.append(coefficient); axes.append(np.full(shape=coefficient.size, fill_value=axis))
        self.left = np.concatenate(left); self.right = np.concatenate(right)
        self.c0 = np.concatenate(coefficients); self.axes = np.concatenate(axes)

    def divergence(self, *, flux):
        n = self.case.active.size
        return (np.bincount(self.left, weights=flux, minlength=n)-
                np.bincount(self.right, weights=flux, minlength=n)).reshape(self.shape)

    def evaluate(self, *, head, physical_tolerance=1e-12, relative_tolerance=1e-8,
                 head_tolerance_m=1e-6, boundary_tolerance_m=1e-10):
        case = self.case
        thresholds = final_tolerances(physical_tolerance=physical_tolerance, relative_tolerance=relative_tolerance,
            head_tolerance_m=head_tolerance_m, boundary_tolerance_m=boundary_tolerance_m)
        checks = inspect_heads(head=head, active=case.active, fixed=case.fixed,
            values=case.values, boundary_tolerance_m=boundary_tolerance_m)
        h = np.where(case.active, head, 0.).ravel()
        saturation, derivative = quadratic_saturation_derivative(head=head, bottom=case.bottoms,
            thickness=case.thicknesses, cell_type=case.cell_types)
        choose_left = (h[self.left] > h[self.right])|((h[self.left] == h[self.right])&(self.axes == 1))
        donor = np.where(choose_left, self.left, self.right)
        variable = (self.axes != 0)&((case.cell_types.ravel()[self.left] != 0)|(case.cell_types.ravel()[self.right] != 0))
        conductance = self.c0*np.where(variable, saturation.ravel()[donor], 1.)
        flux = conductance*(h[self.left]-h[self.right])
        divergence = self.divergence(flux=flux)
        residual = np.where(self.free, case.source-divergence, 0.)
        n = case.active.size
        diagonal = (np.bincount(self.left, weights=conductance, minlength=n)+
                    np.bincount(self.right, weights=conductance, minlength=n)).reshape(self.shape)
        scaled = np.divide(residual, diagonal, out=np.zeros_like(a=residual), where=diagonal > 0.)
        scaled[(diagonal <= 0.)&(residual != 0.)] = np.inf
        boundary = self.free.ravel()[self.left]&(case.fixed.ravel()[self.right] != 0)
        reverse = self.free.ravel()[self.right]&(case.fixed.ravel()[self.left] != 0)
        exchange = np.concatenate([flux[boundary], -flux[reverse]])
        inflow = stable_sum(values=np.maximum(-exchange, 0.)); outflow = stable_sum(values=np.maximum(exchange, 0.))
        throughput = stable_sum(values=np.concatenate([np.abs(case.source[self.free]), np.abs(exchange)]))
        rms = stable_rms(values=residual[self.free]); hrms = stable_rms(values=scaled[self.free])
        l1 = stable_sum(values=np.abs(residual[self.free])); relative = l1/max(throughput, np.finfo(float).tiny)
        edges = conductance > 0.
        # Graph connectivity is a topology check, not a physical flow matrix.
        graph = sparse.coo_matrix((np.ones(shape=int(edges.sum())), (self.left[edges], self.right[edges])), shape=(n, n)).tocsr()
        unused, labels = connected_components(csgraph=graph, directed=False)
        active_labels = set(labels[case.active.ravel() != 0])
        grounded_labels = set(labels[(case.active.ravel() != 0)&(case.fixed.ravel() != 0)])
        ungrounded = len(active_labels-grounded_labels)
        active = case.active != 0
        valid = all(np.all(np.isfinite(value)) for value in (flux, conductance, saturation[active],
            derivative[active], diagonal[active], divergence[active], residual[self.free], scaled[self.free]))
        valid = bool(valid and np.all(np.isfinite([inflow, outflow, throughput, rms, hrms, l1, relative])))
        result = dict(**checks, head_residual_rms=hrms, physical_rms=rms,
            physical_max=float(np.max(np.abs(residual))), residual_l1=l1, relative_cell_imbalance=relative,
            gross_throughput=throughput, signed_imbalance=stable_sum(values=residual[self.free]),
            boundary_inflow=inflow, boundary_outflow=outflow,
            fixed_cell_source_total=stable_sum(values=case.source[case.fixed != 0]), free_cells=int(self.free.sum()),
            fixed_reservoir_required_inflow=stable_sum(values=(divergence-case.source)[case.fixed != 0]),
            ungrounded_current_components=ungrounded, physical_metrics_valid=valid, final_acceptance_thresholds=thresholds,
            evaluator='matrix-free physical incidence; connectivity graph only')
        result['physical_metrics_valid'] = bool(valid and np.all(np.isfinite([
            result['signed_imbalance'], result['fixed_cell_source_total'], result['fixed_reservoir_required_inflow']])))
        result['independent_accepted'] = bool(checks['input_valid'] and checks['boundary_condition_valid'] and
            result['physical_metrics_valid'] and ungrounded == 0 and hrms <= head_tolerance_m and
            (rms <= physical_tolerance or relative <= relative_tolerance))
        return result, dict(flux=flux, residual=residual, saturation=saturation, conductance=conductance,
            principal_diagonal=diagonal, fixed_reservoir_required_inflow=np.where(case.fixed, divergence-case.source, 0.))
