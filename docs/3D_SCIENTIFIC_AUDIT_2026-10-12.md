# Remaining 3D scientific defects: correction and acceptance report

Branch: `dev`. Reference: previously corrected baseline `9d2a8a4`.
Execution path: `WarpDarcySolver3D.solve` → `ThreeDOperatorSession.solve` →
`solve_fast_confined_kcycle_3d` / `solve_fast_unconfined_kcycle_3d`.
Environment: RTX 4070 Ti SUPER, 16 GiB, NVIDIA driver 595.91.07,
Warp 1.17.0, Python 3.14, NumPy FP64, MODFLOW 6 6.6.2.

## Corrections and evidence

| Defect | Correction and location | Independent evidence |
|---|---|---|
| Conductivity-sum cutoff removed valid connections | `solvers_3d._stable_harmonic_mean` and both face builders: positive means connected; harmonic mean uses the smaller value and a ratio rather than products/sums | Three-axis, both-direction steady tests at K=1, 1e-14, 1e-22, 1e-80 with proportionally scaled sources; extreme harmonic pairs at 1e-200 / 1e200 |
| Small positive diagonals replaced physical equations with identity rows | `kernels_3d` applications/residuals/smoothers/preconditioner; `solvers/face_kernels_3d_f64` applications/residuals/Jacobi/reductions; `fast_3d._sanitise_faces_and_diagonal` | Sparse-reference heads and budgets at small K; default-tolerance K=1e-20 solve cannot accept the initial zero head |
| Only isolated individual cells detected | `solvers_3d._resolve_isolated_free_cells`: connected hydraulic components, cached labels, separate fixed-head/storage constraints and net-source checks | Two connected free cells with Q=(1,0), (1,-1), (0,0); all preserved under positive transient storage |
| Array identity concealed in-place storage changes | `fast_3d._snapshot_operator`, `_same_topology`, confined workspace reuse; session owns copied inputs and revision counters | Sequential modifications of the same storage_coeff / storage_diag array compared to independent sparse assembly; dt and boundary updates; unchanged solves retain buffers and graphs |
| Boundary values could remain stale in an unconfined workspace | `_refresh_levels_in_place` refreshes boundary values on change; the nonlinear cache also checks active/BC/connection topology | Dry→wet boundary updates, face-value refresh and connection-topology rebuild regressions |
| Inferred tops bypassed vertical-contiguity validation | `solvers_3d._validate_flat_layer_geometry`: inferred/supplied tops share all checks | Gaps and overlaps rejected with and without ztop; existing dipping-layer tests retained |
| Public directional face arrays need not be reciprocal | `physics/faces_3d.validate_reciprocal_faces`, wrapper/session build/update, budget and boundary-flux utilities | All three axes, inactive connections and independently rounded float32 faces; accepted roundoff is canonicalized, substantial discrepancies rejected |
| Gross budget throughput underestimated when opposing boundaries touched one cell | `physics/budget_3d.compute_3d_flow_budget` accumulates positive/negative exchanges per face before summing cells | Dry-cell test has simultaneous inflow/outflow; source, net boundary and cell balance remain conserved |
| Recovering a source by subtracting the assembled storage RHS can hide forcing at large storage | `prepare_3d_budget_source` retains original Q; `is_3d_budget_conserved` gates final acceptance using independent cells, not only a head residual or domain sum | Ss=1e16 /m, Q=1 m³/day: exact head increment lies below one head ULP, original Q remains visible and the solve reports nonconvergence. Opposing ±1 cell sources also fail despite exactly zero domain imbalance |
| A nominal 0.1 m floor could exceed the entire thickness of a thin cell | Shared Picard driver caps the effective floor at dz and reports requested/effective values | dz=0.05 m live solve: C cannot exceed fully saturated thickness |
| Actual approximate transfers could destabilize irregular masks | `fast_3d._run_kcycles`: mask prolonged corrections and minimize fine-level energy along them on irregular domains | Previously diverged to ~1e24 m after 1500 cycles; safeguarded solve passes unchanged head/residual/sparse-reference gates |

The independent budget evaluates each validated face once and scatters equal
and opposite fluxes to its adjacent cells. The new diagnostics are evaluated
from the returned final head, independently
of the iteration's GPU residual. The nonlinear operator is rebuilt at that
final head, without transmissivity relaxation, before physical acceptance.

## Governing equations and dimensions

For each active free cell, using total hydraulic head h [m]:

```
sum_j C_ij (h_i - h_j) + (V_i(h_new) - V_i(h_old))/dt = Q_i.
```

Q is positive injection/recharge and negative abstraction [m³/day]. A positive
storage term means uptake, so storage release supplies a sink or outflow.
Each reciprocal internal face has q_ij = C_ij(h_i-h_j) = -q_ji.

Confined C_x = harmonic(Kx_i,Kx_j) dy dz/dx [m²/day], with analogous y,z
faces. K is [m/day]. Confined stored-volume derivative is Ss dx dy dz [m²];
**`storage_coeff` is specific storage [1/m], not dimensionless storativity**
when multiplied by full cell volume in this API. Its diagonal contribution
is Ss dx dy dz/dt [m²/day]. Advanced `storage_diag` supplies an already
assembled diagonal contribution; its caller also owns the corresponding RHS.

Unconfined horizontal C_x = harmonic(Kx_i b_i,Kx_j b_j) dy/dx,
b=clip(h-bottom, min_sat_effective, dz) [m]. Vertical faces retain the
full-thickness harmonic resistance, independent of saturation. Storage uses
the unfloored potential, per unit plan area [m]:

```
W(h) = Sy b_true + Ss [b_true²/2 + dz max(h-top,0)]
b_true = clip(h-bottom,0,dz)
V = dx dy W.
```

Sy is dimensionless and Ss is [1/m]. The secant between the fixed previous
timestep head and the current head linearizes exactly the integrated volume
change, including transitions through layer bottoms/tops. No physical storage
is created by the transmissivity floor.

## Disconnected components and datum policy

Labels depend on active cells and strictly positive internal faces. Each
solve checks current fixed-head constraints, positive storage and forcing,
but labels are rebuilt only if hydraulic connectivity changes. A component
with positive storage is valid even with no fixed head. A steady unconstrained
component with nonzero net source is incompatible and raises, naming its
location, cell count and net source. Source balancing uses a relative FP64
summation tolerance, with no absolute source/conductivity cutoff.

Source-balanced interconnected steady components are **explicitly refused as
undetermined**, requiring the caller to supply a declared head datum. No
hidden pin or automatic physical boundary is added. Source-free singleton
null equations keep their initial heads, carry zero flux and are reported as
excluded. This policy deliberately does not select an arbitrary mean head for
an unconstrained Neumann component.

## Cache and performance architecture

The session owns copied host inputs, stable device buffers, operator/topology
revision counters and workspace records. Array-valued public solve controls
remain mutable: owned content snapshots detect changes even when Python
identities are unchanged. No per-iteration cryptographic hashing is used.
This requires host comparisons of exposed inputs; it is the cost of supporting
arbitrary mutation rather than an immutable/version-only API.

Operator-value changes refresh the existing hierarchy buffers and preserve
the captured CUDA graph. Connection/active/BC topology changes reconstruct the
hierarchy. Unchanged confined operators upload only their changing RHS.
Unconfined Picard retains the existing in-place face/coarse refresh, caches
topology rather than copying full operator snapshots every outer iteration,
and avoids repeated boundary-value uploads. Unfloored faces for floor
diagnostics are rebuilt only where the floor actually changes saturation.

The irregular-mask safeguard uses alpha = (rᵀe)/(eᵀAe) for the prolonged
correction e, with corrections zero on fixed/inactive cells. For the symmetric
positive fine operator this minimizes quadratic energy on that line. It uses
existing device reduction buffers, with no host scalar read and no new device
workspace allocation, and is included in CUDA capture. Rectangular domains
keep the existing full coarse-correction path.

Profiling the first correction version found repeated source/component
reductions on already grounded components, duplicate value/topology checks,
revalidation of every internally constructed coarse face, copied exact faces
in the budget, and repeated terminal GPU residual reductions. The first
large-grid measurements showed up to 24% overhead. The final code caches
component sizes, checks source compatibility only where needed, validates
external fine faces while trusting tested reciprocal coarse construction,
borrows exact budget faces, and reuses terminal checks when h and A have not
changed. The independent budget traverses each undirected face once, improving
both conservation and timing. Intermediate timing JSONs preserve that initial
regression; final tables use fresh serial before/after runs.

## Hierarchy finding: exact Galerkin claim disproved

`test_3d_hierarchy_scientific_audit.py` builds restriction/prolongation matrices
by probing the actual Warp kernels with basis vectors. R is a normalized
free-cell block average. P is bilinear interpolation, with corrections later
masked on fixed/inactive fine cells. They are not adjoints. The actual coarse
operator differs from R A P, including on regular grids.

The previous report's Galerkin check used hypothetical piecewise-constant
injection. Its interpretation as a proof about the live transfers was
incorrect. Crossing-face aggregation and Dirichlet springs do produce a
symmetric, reciprocal, conservative approximate coarse operator. Partial free
blocks are uniformly scaled in that operator while R renormalizes by the
actual free-cell count. The resulting hierarchy is an approximate
preconditioner whose effectiveness is checked through convergence of the
independently assembled **fine** equations. Its architecture was retained;
the demonstrated irregular-mask instability warranted the limited safeguard.

Tests include odd dimensions, inactive holes, mixed Dirichlet blocks,
heterogeneous K, vertical anisotropy, coarse symmetry/positive semidefiniteness,
constant-field conservation and sparse-reference final heads.

## Analytical and MODFLOW validation

The original scientific suite passed 32 tests despite the confirmed defects.
The new tests cover all six Darcy directions, scale invariance, tiny positive
conductances, zero equations, connected singular groups, storage-only groups,
mutable transient operators, reciprocal public inputs, geometry, actual
transfers and approximation diagnostics. Sparse matrices are assembled by
traversing each undirected face once, independently of solver kernels/budgets.

The repaired MF6 workflow uses the same cell geometry, K/Kz, initial heads,
fixed heads, recharge/abstraction schedules, period durations and Ss/Sy fields.
Comparison times are [1,4,11,13] days for the matched histories, with two
drainage periods then restarted recharge and a changing abstraction rate.
There is no extra warmup period or comparison offset. Separate layer-transition
tests use [1,2] days, Sy=[0.2,0.35], and drainage/rewetting through the interface.
The layered case includes K=1e-14 m/day basement cells, rather than treating
them as inactive or identity equations.

Head/water-table comparisons use 2e-5 m absolute tolerance; face flows, global
storage and boundary exchange use 2e-5 m³/day absolute plus 2e-5 relative.
Independently assembled cell balance in the history tests must be below
2e-7 m³/day. The Newton interface case compares its face flow/storage to exact
215 m³/day abstraction then 235 m³/day injection; Warp cell imbalances remain
below 1.4e-8 m³/day and domain relative imbalance below 2.7e-11.

With no Ss in that interface case, exact Warp heads are [9,5] m after drainage
and [13,19] m after rewetting. The actual water table is the **topmost wet cell**,
not the largest pressure head anywhere in the column. MF6 agrees within
8e-6 m in both periods, and intercell fluxes are +40 then -60 m³/day.

### Constitutive comparison and reference safeguards

The retained unsmoothed integrated Ss potential agrees with MF6's revised
unsmoothed storage law, disproving a blanket claim that it is intrinsically
different. MF6 Newton instead uses quadratic saturation smoothing with
epsilon=1e-6. The analytical tests separately invert both potentials:

```
W_MF6(h) = Sy dz s_epsilon(h)
         + Ss dz s_epsilon(h) [h-bottom - dz s_epsilon(h)/2].
```

For bottom=10 m, dz=10 m, Sy=.2, Ss=.001/m, area=100 m², h_old=19.5 m and
Q=20 m³/day for one day, the clipped law predicts 29.5125 m; the smoothed MF6
law predicts 29.5125900001 m. The measured 9e-5 m discrepancy is predicted
by the two constitutive laws, not hidden by widening the 2e-5 m numerical
comparison gate. Each solver is checked against its corresponding independent
potential, with the observed inter-model difference checked against the
predicted difference. Drainage from an initially confined single cell requires
MF6 Newton; the ordinary Picard reference can wrongly dry its first confined
linear iterate before the convertible storage can respond.

Primary references: [MF6 6.6.2 saturation function](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Utilities/SmoothingFunctions.f90),
[storage capacities/terms](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/ModelUtilities/GwfStorageUtils.f90),
[STO smoothing selection](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/GroundWaterFlow/gwf-sto.f90),
[NPF options](https://modflow6.readthedocs.io/en/latest/_mf6io/gwf-npf.html).
The tested constant vertical-conductance cases agree; enabling MF6
VARIABLECV/DEWATERED/PERCHED changes the equations and is not a parity claim
made by this solver.

The confined history test initially requested an absolute residual floor of
zero and a relative tolerance that became 9.41e-14 m³/day in a late period.
An independent direct sparse solve has 1.45e-12 m³/day RMS roundoff, and one
head ULP times the maximum diagonal is 2.98e-12. The new test therefore uses
an attainable 1e-11 m³/day iterative floor, while preserving every independent
head/face/storage/budget gate and the 1e-10 m head-equivalent gate. No existing
physical comparison tolerance was loosened. Solver residuals and independent budgets are
both retained in diagnostics; extremely tight unattainable requested tolerances
continue to report nonconvergence rather than silently declaring success.

MF6 tests have explicit skip reasons when unavailable and print LIVE_MF6
records when run with `-s`. Live MF6 comparisons actually ran in this environment.

Final acceptance also requires independent cell conservation: either the
physical volumetric residual RMS meets the requested linear volumetric
tolerance, or the sum of absolute cell imbalances is at most 0.1% of gross
throughput (`final_budget_relative_tol=1e-3` on the fast public APIs). This
additional gate never relaxes the head/iteration gates. A near-zero-flow
case uses the dimensional criterion rather than an unstable percentage.
Cellwise errors cannot cancel in the L1 gate. The extreme-storage regression
demonstrates why a very small head-equivalent residual alone is insufficient:
the exact increment is smaller than representable head spacing. The solver
reports failure rather than accepting a head that cannot store the declared
volume. Ordinary `storage_coeff` forcing is never inferred by subtracting large
storage terms. Advanced preassembled diagonal/RHS inputs retain their explicit
caller-owned assembly contract.

## Dry-cell approximation and applicability

This remains quasi-3D saturated flow, not Richards/unsaturated flow. Horizontal
transmission below the floor persists and vertical conductance is independent
of saturation. These assumptions cannot be justified by a small solver residual.

`info['dry_cell_approximation']` reports absolute excess horizontal face flow
from the floor at the same final head, flow through nominally dry vertical
interfaces, and ratios to recharge, abstraction and gross water-budget
throughput. Opposing internal face flows cannot cancel in the excess-flow sum.
Absent recharge/abstraction produces an undefined (`None`) ratio, not zero.
This is a local constitutive sensitivity, not a full alternative-solution error
estimate or a Richards reference.

Controlled tests span saturation, approaching a bottom, nominally dry cells,
rewetting, conductivity contrast through 1e-14, and dz=.05 m layers over
low-permeability material. In a live three-cell dry strip with K=1 m/day,
dx=dy=dz=1 m and fixed heads -1/-2 m, the 0.1 m floor carries 0.1 m³/day
summed horizontal excess flow. Adding 0.01 m³/day recharge or abstraction
conserves exactly, yet the artificial flow is **10 times that forcing** and
about **90.9% of the gross budget throughput**. Rewetting to fully saturated
heads removes the floor excess. Thus conservation alone cannot certify the
dry approximation.

For two equally saturated cells with true b<floor, horizontal conductance is
overstated by floor/b. At b=0.05 m with a .1 m floor it doubles; at b=0 it
transmits where the zero-thickness reference cannot. A basement contrast
reduces vertical leakage according to its half-cell resistance, but does not
remove lateral transmission through a dry high-K upper layer. Thin cells for
which floor=dz behave as fully transmissive even when partially wet: capping
the floor fixes excess geometric thickness, not the drying model.

The approximation is exact with respect to the floor wherever all contributing
horizontal saturated thicknesses exceed it. Elsewhere require excess flow to
be small against the **case's physical forcing/throughput tolerance**, and
compare sensitivity to a smaller floor. For a 0.1% budget-error objective,
excess-flow fractions materially above 1e-3 are unacceptable as an accuracy
claim. This is a necessary screening condition, not a proof of global head
accuracy. Dry-interface vertical flow needs a separate physical justification;
there is no saturation-dependent vertical permeability, capillarity or
dry-cell disconnection in this formulation.

## Acceptance and performance results

Full repository suite: **431 passed, 195 warnings in 486.48s (0:08:06)**. No MF6 comparisons were
skipped. After the final performance correction, all 178 then-current 3D tests passed; the final full run includes the additional balanced-cell regressions. Live MF6 run
with printed metrics and strengthened cellwise storage/global budget checks:
**16 passed, 28 deselected, 2 warnings in 53.68s**.

| Fast-path matched-history case | Max head difference (m) | Max face-flow difference (m³/day) | Independent Warp cell imbalance (m³/day) | Global storage difference (m³/day) |
|---|---:|---:|---:|---:|
| confined | 1.12e-11 | 3.35e-10 | 1.28e-10 | 6.97e-11 |
| drain_recharge | 1.23e-09 | 1.09e-07 | 1.24e-09 | 2.39e-07 |
| layered_basement | 4.92e-09 | 2.32e-07 | 5.82e-10 | 9.11e-07 |

Performance uses serial fresh processes, first solve plus median of three
repeats on the same model/initial head. Construction and CUDA context
initialization occur before timing; existing Warp kernel caches are primed.
The before process imports a read-only snapshot of corrected baseline
`9d2a8a4`, not the modified live package. Every measured solve converged,
and all repeat calls reused an already captured graph, with no capture
fallback. Outer/cycle counts and physical residuals are included, rather
than treating a faster nonconverged solve as a speedup.

### 4×64×64: 16,384 cells

| Case | First before/after (s) | Repeat median before/after (s) | Repeat change | Total cycles before/after | Outer before/after | Max physical cell imbalance (m³/day) | Max relative domain imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|
| confined_steady | 0.1158 / 0.1255 | 0.0822 / 0.0805 | -2.1% | 35 / 35 | None / None | 1.91e-08 | 1.44e-09 |
| confined_transient | 0.1050 / 0.0999 | 0.0820 / 0.0828 | +1.0% | 35 / 35 | None / None | 1.88e-08 | 3.79e-10 |
| unconfined_steady | 0.9858 / 0.9954 | 0.9680 / 0.9792 | +1.2% | 405 / 405 | 11 / 11 | 2.36e-06 | 8.24e-08 |
| unconfined_transient | 0.9637 / 0.9706 | 0.9509 / 0.9588 | +0.8% | 395 / 395 | 11 / 11 | 2.35e-06 | 3.78e-09 |
| layered_unconfined_transient | 0.6036 / 0.6148 | 0.5869 / 0.5952 | +1.4% | 240 / 240 | 11 / 11 | 5.26e-07 | 1.75e-08 |

| Case | Maximum live mempool before/after (MiB) | Process high-water before/after (MiB) |
|---|---:|---:|
| confined_steady | 2.79 / 2.79 | 2.79 / 2.79 |
| confined_transient | 2.79 / 2.79 | 5.59 / 4.04 |
| unconfined_steady | 5.59 / 5.59 | 5.59 / 5.59 |
| unconfined_transient | 5.59 / 5.59 | 8.38 / 5.59 |
| layered_unconfined_transient | 8.38 / 5.59 | 8.38 / 8.38 |

### 4×128×128: 65,536 cells

| Case | First before/after (s) | Repeat median before/after (s) | Repeat change | Total cycles before/after | Outer before/after | Max physical cell imbalance (m³/day) | Max relative domain imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|
| confined_steady | 0.1194 / 0.1316 | 0.0798 / 0.0798 | +0.0% | 30 / 30 | None / None | 6.44e-08 | 1.47e-09 |
| confined_transient | 0.1127 / 0.1187 | 0.0915 / 0.0932 | +1.8% | 35 / 35 | None / None | 1.2e-08 | 4.36e-10 |
| unconfined_steady | 1.1485 / 1.1759 | 1.1304 / 1.1537 | +2.1% | 410 / 410 | 11 / 11 | 2.36e-06 | 1.29e-07 |
| unconfined_transient | 1.1154 / 1.1579 | 1.1152 / 1.1392 | +2.2% | 400 / 400 | 11 / 11 | 2.35e-06 | 3.56e-09 |
| layered_unconfined_transient | 0.7249 / 0.7570 | 0.7211 / 0.7472 | +3.6% | 240 / 240 | 11 / 11 | 5.26e-07 | 1.66e-08 |

| Case | Maximum live mempool before/after (MiB) | Process high-water before/after (MiB) |
|---|---:|---:|
| confined_steady | 11.17 / 11.17 | 11.17 / 11.17 |
| confined_transient | 11.17 / 11.17 | 22.34 / 16.17 |
| unconfined_steady | 22.34 / 22.34 | 22.34 / 22.34 |
| unconfined_transient | 22.34 / 22.34 | 33.51 / 22.34 |
| layered_unconfined_transient | 33.51 / 22.34 | 33.51 / 33.51 |

### 6×256×256: 393,216 cells

| Case | First before/after (s) | Repeat median before/after (s) | Repeat change | Total cycles before/after | Outer before/after | Max physical cell imbalance (m³/day) | Max relative domain imbalance |
|---|---:|---:|---:|---:|---:|---:|---:|
| confined_steady | 0.2614 / 0.2759 | 0.1614 / 0.1610 | -0.3% | 35 / 35 | None / None | 4.02e-08 | 1.06e-08 |
| confined_transient | 0.2214 / 0.2473 | 0.1656 / 0.1655 | -0.0% | 35 / 35 | None / None | 9.47e-09 | 6.14e-10 |
| unconfined_steady | 2.4414 / 2.5338 | 2.4221 / 2.5022 | +3.3% | 435 / 435 | 11 / 11 | 2.36e-06 | 1.7e-07 |
| unconfined_transient | 2.4376 / 2.5788 | 2.3867 / 2.5841 | +8.3% | 430 / 430 | 11 / 11 | 2.35e-06 | 7.61e-09 |
| layered_unconfined_transient | 1.7120 / 1.8950 | 1.6831 / 1.8622 | +10.6% | 240 / 240 | 11 / 11 | 6.73e-07 | 1.44e-08 |

| Case | Maximum live mempool before/after (MiB) | Process high-water before/after (MiB) |
|---|---:|---:|
| confined_steady | 67.02 / 67.02 | 67.02 / 67.02 |
| confined_transient | 67.02 / 67.02 | 134.03 / 97.02 |
| unconfined_steady | 134.03 / 134.03 | 134.03 / 134.03 |
| unconfined_transient | 134.03 / 134.03 | 201.05 / 134.03 |
| layered_unconfined_transient | 201.05 / 134.03 | 201.05 / 201.05 |

Mempool values are process-wide Warp allocations, sampled after device
synchronization. High-water values accumulate across cases within a process
and do not represent an individual workspace footprint. Repeated live
memory can vary when Python reclaims a previous graph/workspace; array
identities in the current workspace remain stable (tested explicitly).

The measured overhead comes from component/input comparisons and the
independent physical/dry-flow diagnostics. No defective equation, stale
cache or premature convergence criterion was restored to recover timing.
Obvious redundant per-outer operator snapshots and boundary uploads were
removed; the existing graph/device-buffer architecture is retained.

Raw timing records: `working_tests/baseline_3d_audit_{before,after}_{64,128,6x256}.json`.
Summarized machine-readable results: `docs/3d_scientific_audit_results.json`.

Reproduce with:

```bash
DARCY_FLOAT=float64 python -m pytest tests -q -ra
DARCY_FLOAT=float64 python -m pytest tests/test_3d_physics_validation.py -k mf6 -q -s -ra
DARCY_FLOAT=float64 PYTHONPATH=. python working_tests/benchmark_3d_scientific_audit.py --out /tmp/audit.json --nz 6 --nx 256 --ny 256
```

## Remaining limits

- Only flat, vertically contiguous, scalar-dz uniform Cartesian layers.
- No terrain-following geometry or horizontal K tensor rotation.
- No genuine unsaturated storage/permeability or Richards physics.
- No automatic datum choice for source-balanced disconnected steady components.
- Approximate multigrid hierarchy, not exact Galerkin; the irregular correction
  safeguard is validated on the specified cases, not a universal convergence theorem.
- FP64 finite dynamic range and representable-head roundoff still apply; a
  caller must choose attainable dimensional tolerances.
- Advanced preassembled diagonal inputs require a consistently assembled RHS.
- Timing figures characterize the tested GPU and grids, not a capacity study
  or a new optimization campaign. The production 2D path was not modified.
