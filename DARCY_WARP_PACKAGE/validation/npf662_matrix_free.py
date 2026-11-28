"""Independent layered physical audit without assembling a flow matrix.

Only an unweighted connectivity graph is materialized to verify grounding.
Physical face incidence and diagonals use NumPy bin counts, not sparse A*h.
"""
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from DARCY_WARP_PACKAGE.validation.head_checks import inspect_heads
from DARCY_WARP_PACKAGE.validation.npf662_metrics import (
    balanced_nullspace_components, exact_zero_equation_rows, final_tolerances,
    stable_rms, stable_sum, acceptance_head_rms, local_balance_valid,
)
from DARCY_WARP_PACKAGE.physics.ghb_3d import npf_ghb_component_anchor
from DARCY_WARP_PACKAGE.validation.npf662_reference import quadratic_saturation_derivative
from DARCY_WARP_PACKAGE.physics.drn_riv_3d import (csr_entry_terms,
    csr_from_boundary_field, validate_drn_fields, validate_riv_fields)


class NPF662MatrixFreeReference:
    def __init__(self, *, case):
        case.validate()
        if case.geometry != 'structured_dis_layered':
            raise ValueError('Matrix-free layered auditor requires canonical layered inputs')
        self.case = case; self.shape = case.active.shape
        self.ghb_conductance = np.zeros(self.shape) if case.ghb_conductance is None else np.broadcast_to(case.ghb_conductance,self.shape).copy()
        self.ghb_head = np.zeros(self.shape) if case.ghb_head is None else np.broadcast_to(case.ghb_head,self.shape).copy()
        self.free = (case.active != 0)&(case.fixed == 0)
        self.component_anchor = npf_ghb_component_anchor(conductance=case.ghb_conductance,
            active=case.active, fixed=case.fixed)
        drn_c, drn_e = validate_drn_fields(conductance=case.drn_conductance,
            elevation=case.drn_elevation, mask=case.drn_mask, active=case.active, fixed=case.fixed)
        riv_c, riv_s, riv_b = validate_riv_fields(conductance=case.riv_conductance,
            stage=case.riv_stage, bottom=case.riv_bottom, mask=case.riv_mask,
            active=case.active, fixed=case.fixed)
        self.drn = (case.drn_entries if case.drn_entries is not None
            else (csr_from_boundary_field(conductance=drn_c, levels=[drn_e], shape=self.shape)
                if drn_c is not None else None))
        self.riv = (case.riv_entries if case.riv_entries is not None
            else (csr_from_boundary_field(conductance=riv_c, levels=[riv_s, riv_b], shape=self.shape)
                if riv_c is not None else None))
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

    def boundary_terms(self, *, head):
        """Evaluate sparse DRN/RIV entries without assembling a flow matrix."""
        flat = np.asarray(head, dtype=np.float64).ravel()
        shape = self.shape
        if self.drn is None:
            drn_entries = None
            drn = (np.zeros(shape), np.zeros(shape), np.zeros(shape))
        else:
            drn_entries, terms = csr_entry_terms(head_flat=flat, entries=self.drn,
                package='DRN', shape=shape)
            drn = (terms.discharge, terms.diagonal, np.bincount(self.drn.flat_cells,
                weights=np.abs(drn_entries), minlength=flat.size).reshape(shape))
        if self.riv is None:
            riv_entries = None
            riv = (np.zeros(shape), np.zeros(shape), np.zeros(shape))
        else:
            riv_entries, terms = csr_entry_terms(head_flat=flat, entries=self.riv,
                package='RIV', shape=shape)
            riv = (terms.discharge, terms.diagonal, np.bincount(self.riv.flat_cells,
                weights=np.abs(riv_entries), minlength=flat.size).reshape(shape))
        return drn_entries, drn, riv_entries, riv

    def evaluate(self, *, head, physical_tolerance=1e-12, relative_tolerance=1e-8,
                 head_tolerance_m=1e-6, boundary_tolerance_m=1e-10, roundoff_acceptance=False):
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
        extra = np.where(variable, self.c0*derivative.ravel()[donor]*(h[self.left]-h[self.right]), 0.)
        dl = conductance+np.where(choose_left, extra, 0.)
        dr = -conductance+np.where(~choose_left, extra, 0.)
        flux = conductance*(h[self.left]-h[self.right])
        divergence = self.divergence(flux=flux)
        ghb_exchange = np.zeros(self.shape)
        selected = self.ghb_conductance > 0.
        ghb_exchange[selected] = self.ghb_conductance[selected]*(np.asarray(head)[selected]-self.ghb_head[selected])
        drn_entries, drain, riv_entries, river = self.boundary_terms(head=head)
        residual = np.where(self.free, case.source-divergence-ghb_exchange-drain[0]-river[0], 0.)
        n = case.active.size
        diagonal = (np.bincount(self.left, weights=conductance, minlength=n)+
                    np.bincount(self.right, weights=conductance, minlength=n)).reshape(self.shape)+self.ghb_conductance+drain[1]+river[1]
        scaled = np.divide(residual, diagonal, out=np.zeros_like(a=residual), where=diagonal > 0.)
        scaled[(diagonal <= 0.)&(residual != 0.)] = np.inf
        boundary = self.free.ravel()[self.left]&(case.fixed.ravel()[self.right] != 0)
        reverse = self.free.ravel()[self.right]&(case.fixed.ravel()[self.left] != 0)
        drain_rates = drain[0][self.free] if drn_entries is None else drn_entries
        river_rates = river[0][self.free] if riv_entries is None else riv_entries
        exchange = np.concatenate([flux[boundary], -flux[reverse],ghb_exchange[self.free],
            drain_rates, river_rates])
        inflow = stable_sum(values=np.maximum(-exchange, 0.)); outflow = stable_sum(values=np.maximum(exchange, 0.))
        throughput = stable_sum(values=np.concatenate([np.abs(case.source[self.free]), np.abs(exchange)]))
        rms = stable_rms(values=residual[self.free]); hrms = stable_rms(values=scaled[self.free])
        l1 = stable_sum(values=np.abs(residual[self.free])); relative = l1/max(throughput, np.finfo(float).tiny)
        edges = conductance > 0.
        # Graph connectivity is a topology check, not a physical flow matrix.
        graph = sparse.coo_matrix((np.ones(shape=int(edges.sum())), (self.left[edges], self.right[edges])), shape=(n, n)).tocsr()
        unused, labels = connected_components(csgraph=graph, directed=False)
        boundary_gross = np.abs(ghb_exchange) + drain[2] + river[2]
        dry_zero = exact_zero_equation_rows(case=case, head=head, free=self.free,
            diagonal=diagonal, residual=residual, boundary_throughput=boundary_gross)
        active_labels = set(labels[(case.active.ravel() != 0) & ~dry_zero.ravel()])
        boundary_tangent = self.ghb_conductance + drain[1] + river[1]
        grounded_mask = (case.active.ravel() != 0) & ((case.fixed.ravel() != 0) | (boundary_tangent.ravel() > 0.))
        grounded_labels = set(labels[grounded_mask])
        ungrounded_labels = active_labels-grounded_labels
        nullspaces = balanced_nullspace_components(labels=labels, active=case.active,
            grounded=grounded_mask, source=case.source,
            left=self.left, right=self.right, face_left_derivative=dl,
            face_right_derivative=dr,
            boundary_tangent=boundary_tangent,
            boundary_throughput=boundary_gross)
        balanced_labels = ungrounded_labels & set(nullspaces)
        ungrounded = len(ungrounded_labels-balanced_labels)
        acceptance_rms = acceptance_head_rms(case=case, free=self.free, residual=residual,
            diagonal=diagonal, equivalent=scaled, left=self.left, right=self.right,
            conductance=conductance, boundary_tangent=boundary_tangent,
            boundary_throughput=boundary_gross, physical_tolerance=physical_tolerance)
        active = case.active != 0
        valid = all(np.all(np.isfinite(value)) for value in (flux, conductance, dl, dr, saturation[active],
            derivative[active], diagonal[active], divergence[active], residual[self.free]))
        valid = bool(valid and np.all(np.isfinite([inflow, outflow, throughput, rms, acceptance_rms, l1, relative])))
        result = dict(**checks, head_residual_rms=hrms, acceptance_head_residual_rms=acceptance_rms, physical_rms=rms,
            physical_max=float(np.max(np.abs(residual))), residual_l1=l1, relative_cell_imbalance=relative,
            gross_throughput=throughput, signed_imbalance=stable_sum(values=residual[self.free]),
            boundary_inflow=inflow, boundary_outflow=outflow,
            ghb_inflow=stable_sum(values=np.maximum(-ghb_exchange[self.free],0.)),
            ghb_outflow=stable_sum(values=np.maximum(ghb_exchange[self.free],0.)),
            drn_inflow=stable_sum(values=np.maximum(-drain_rates,0.)),
            drn_outflow=stable_sum(values=np.maximum(drain_rates,0.)),
            riv_inflow=stable_sum(values=np.maximum(-river_rates,0.)),
            riv_outflow=stable_sum(values=np.maximum(river_rates,0.)),
            fixed_cell_source_total=stable_sum(values=case.source[case.fixed != 0]), free_cells=int(self.free.sum()),
            fixed_reservoir_required_inflow=stable_sum(values=(divergence-case.source)[case.fixed != 0]),
            ungrounded_current_components=ungrounded,
            balanced_nullspace_components=len(balanced_labels),
            balanced_nullspace_cells=int(sum(nullspaces[label].size for label in balanced_labels)),
            exact_zero_equation_rows=int(dry_zero.sum()),
            exact_dry_zero_rows=int(dry_zero.sum()),
            physical_metrics_valid=valid, final_acceptance_thresholds=thresholds,
            evaluator='matrix-free physical incidence; connectivity graph only')
        result['physical_metrics_valid'] = bool(valid and np.all(np.isfinite([
            result['signed_imbalance'], result['fixed_cell_source_total'], result['fixed_reservoir_required_inflow']])))
        result['independent_accepted'] = bool(checks['input_valid'] and checks['boundary_condition_valid'] and
            result['physical_metrics_valid'] and ungrounded == 0 and acceptance_rms <= head_tolerance_m and
            (rms <= physical_tolerance or relative <= relative_tolerance))
        result['local_balance_valid'] = None
        if result['independent_accepted'] or roundoff_acceptance:
            component_anchor = npf_ghb_component_anchor(conductance=boundary_tangent,
                active=case.active, fixed=case.fixed, left=self.left, right=self.right,
                face_conductance=conductance)
            result['local_balance_valid'] = local_balance_valid(case=case, free=self.free, left=self.left, right=self.right,
                head=head, residual=residual, flux=flux, conductance=conductance, ghb_conductance=boundary_tangent,
                ghb_exchange=ghb_exchange+drain[0]+river[0], thresholds=thresholds,
                component_anchor=component_anchor, boundary_throughput=np.abs(ghb_exchange)+drain[2]+river[2])
            result['independent_accepted'] &= result['local_balance_valid']
        if roundoff_acceptance:
            from .npf662_metrics import componentwise_roundoff, apply_roundoff_acceptance
            apply_roundoff_acceptance(checks=result,certificate=componentwise_roundoff(
                head=head,source=case.source,residual=residual,free=self.free,left=self.left,right=self.right,
                conductance=conductance,ghb_conductance=self.ghb_conductance,ghb_head=self.ghb_head))
        return result, dict(flux=flux, residual=residual, saturation=saturation, conductance=conductance,
            principal_diagonal=diagonal,ghb_outflow=ghb_exchange,
            drn_outflow=drain[0],riv_outflow=river[0],
            fixed_reservoir_required_inflow=np.where(case.fixed, divergence-case.source, 0.))
