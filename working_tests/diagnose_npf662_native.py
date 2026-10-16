"""Read the actual MF6 deck and check the confined problem with matrix-free CG."""
from pathlib import Path
from time import perf_counter
import json

import flopy
import numpy as np
from scipy.sparse.linalg import LinearOperator, cg

from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case
from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
from benchmark_npf662_baseline import write_json


class PhysicalLinearProblem:
    def __init__(self, *, reference, record_every):
        self.reference = reference; self.case = reference.case
        self.record_every = record_every; self.iterations = 0; self.history = []
        self.free = reference.free.ravel(); self.start = perf_counter()
        self.diagonal = reference.evaluate(head=self.case.initial)[1]['principal_diagonal'].ravel()
        self.diagonal[~self.free] = 1.

    def action(self, x):
        free_x = np.where(self.free, x, 0.)
        flux = self.reference.c0*(free_x[self.reference.left]-free_x[self.reference.right])
        result = self.reference.divergence(flux=flux).ravel()
        result[~self.free] = x[~self.free]
        return result

    def precondition(self, x):
        return x/self.diagonal

    def callback(self, x):
        self.iterations += 1
        if self.iterations % self.record_every == 0:
            error = self.rhs-self.action(x=x)
            self.history.append(dict(iteration=self.iterations, seconds=perf_counter()-self.start,
                physical_rms=float(np.sqrt(np.mean(error[self.free]**2)))))


def diagnose(*, baseline, output, max_iterations, rtol, atol, record_every, deck_roundtrip_abs_tolerance):
    output.mkdir(parents=True, exist_ok=False)
    case = load_npf662_case(path=baseline/'confined'/'inputs.npz')
    reference = NPF662MatrixFreeReference(case=case)
    simulation = flopy.mf6.MFSimulation.load(sim_ws=str(baseline/'confined'/'mf6'), verbosity_level=0)
    gwf = simulation.get_model()
    reverse = (slice(None), slice(None, None, -1), slice(None))
    differences = {}
    for name, actual, expected in (
        ('kx', gwf.npf.k.array[reverse], case.kx), ('ky', gwf.npf.k22.array[reverse], case.ky),
        ('kz', gwf.npf.k33.array[reverse], case.kz), ('bottom', gwf.dis.botm.array[reverse], case.bottoms),
        ('top', gwf.dis.top.array[::-1], case.tops[0]), ('cold_head', gwf.ic.strt.array[reverse], case.initial),
        ('cell_type', gwf.npf.icelltype.array[reverse], case.cell_types)):
        differences[name] = float(np.max(np.abs(actual-expected)))
    chd = np.zeros_like(a=case.values); fixed = np.zeros_like(a=case.fixed)
    for record in gwf.chd.stress_period_data.get_data(key=0):
        k, row, column = record['cellid']; j = case.active.shape[1]-1-row
        fixed[k, j, column] = 1; chd[k, j, column] = record['head']
    source = np.zeros_like(a=case.source)
    for record in gwf.get_package('SOURCE').stress_period_data.get_data(key=0):
        k, row, column = record['cellid']; j = case.active.shape[1]-1-row
        source[k, j, column] += record['q']
    differences.update(fixed_mask=float(np.max(np.abs(fixed-case.fixed))),
        chd_values=float(np.max(np.abs((chd-case.values)[case.fixed != 0]))),
        source=float(np.max(np.abs(source-case.source))))
    if any(value > deck_roundtrip_abs_tolerance for value in differences.values()):
        raise AssertionError(f'Actual MF6 deck differs from canonical arrays: {differences}')
    prescribed = np.where(case.fixed, case.values, 0.).ravel()
    flux = reference.c0*(prescribed[reference.left]-prescribed[reference.right])
    rhs = case.source.ravel()-reference.divergence(flux=flux).ravel()
    rhs[~reference.free.ravel()] = 0.
    problem = PhysicalLinearProblem(reference=reference, record_every=record_every); problem.rhs = rhs
    n = case.active.size
    operator = LinearOperator(shape=(n, n), matvec=problem.action, dtype=np.float64)
    preconditioner = LinearOperator(shape=(n, n), matvec=problem.precondition, dtype=np.float64)
    initial = np.where(reference.free, case.initial, 0.).ravel()
    start = perf_counter()
    solution, status = cg(A=operator, b=rhs, x0=initial, M=preconditioner,
        maxiter=max_iterations, rtol=rtol, atol=atol, callback=problem.callback)
    elapsed = perf_counter()-start
    head = (solution+prescribed).reshape(case.active.shape)
    checks, fields = reference.evaluate(head=head)
    np.save(file=output/'independent_confined_head.npy', arr=head)
    result = dict(actual_MF6_deck_max_errors=differences, source_total_m3_day=float(np.sum(source)),
        fixed_cells=int(np.sum(fixed)), source_cells=int(np.count_nonzero(source)),
        source_beyond_top_cells=int(np.count_nonzero(source[1:])), source_on_CHD_cells=int(np.count_nonzero(source[fixed != 0])),
        west_CHD_heads=[float(chd[fixed != 0].min()), float(chd[fixed != 0].max())],
        coast_top_layer_bottom_m=[float(case.bottoms[0, :, 0].min()), float(case.bottoms[0, :, 0].max())],
        top_bottom_m=[float(case.bottoms[0].min()), float(case.bottoms[0].max())],
        cg_status=int(status), cg_iterations=problem.iterations, cg_seconds=elapsed, independent_checks=checks,
        head_range_m=[float(head.min()), float(head.max())], history=problem.history,
        caution='Diagnostic CPU iterative solve, no direct factorization; not an MF6 reference or standalone K-cycle timing.')
    write_json(path=output/'diagnostic.json', value=result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    baseline = Path('benchmark_artifacts/npf662_250x250x5_v1/run_006')
    output = Path('benchmark_artifacts/npf662_250x250x5_v1/native_diagnostic_002')
    max_iterations = 4000
    rtol = 1e-11
    atol = 1e-12
    record_every = 50
    deck_roundtrip_abs_tolerance = 1e-14
    diagnose(baseline=baseline, output=output, max_iterations=max_iterations,
        rtol=rtol, atol=atol, record_every=record_every,
        deck_roundtrip_abs_tolerance=deck_roundtrip_abs_tolerance)
