# DarcyWarp 3D: Variable Geological Geometry — Final Scientific Report

> 2026-10-16 clarification: general affine physical-head assertions use harmonic-thickness horizontal conductances, which differ from the arithmetic midpoint physical face area when layer thickness varies. The h=z cancellation and corrected vertical metric flux remain exact. New CPU/GPU/MF6 experiments quantify this discretisation error and the MF6 geometry differences in [the independent validation report](TERRAIN_CPU_MF6_VALIDATION_2026-10-16.md). Earlier wording must not be read as exact physical fluxes through every variable-thickness face.

**Date:** 2026-10-13 · **Branch:** `dev` · **Baseline:** `3c203e2`
**Scope:** independently varying geological interfaces, iteration-control
defects, and validation. The production 2D solver and the Cartesian 3D
solver are untouched. All performance comparisons run on the RTX 4070 Ti
SUPER with `DARCY_FLOAT=float64`.

## 1. Governing equations and geometry

Each geological interface is an explicit two-dimensional elevation surface
`z_interfaces[k](x,y)`, `k = 0..nz`. A model with `nz` layers supplies
`nz+1` surfaces. Inside layer `k` the computational coordinate
`s = k + u`, `u ∈ [0,1]`, maps linearly to elevation

```
z(x,y,k+u) = (1-u)·z_interfaces[k](x,y) + u·z_interfaces[k+1](x,y)
```

so layers may thicken, thin and slope independently of the ground surface.
The Jacobian and its inverse are

```
J = [[1,0,0],[0,1,0],[z_x,z_y,z_s]],  z_s = z_bottom − z_top = −T(x,y) < 0,
```

and for diagonal `K = diag(Kx,Ky,Kz)` the conservative transformed tensor
(derived independently before implementation and matching the task
statement) is

```
A = |det J|·J⁻¹·K·J⁻ᵀ =
[[ T·Kx,   0,      Kx·z_x ],
 [ 0,      T·Ky,   Ky·z_y ],
 [ Kx·z_x, Ky·z_y, (Kx·z_x² + Ky·z_y² + Kz)/T ]]
```

with `b = −z_s = T` the local geometric thickness. The old constant-depth
terrain tensor is the special case of parallel interfaces; the parallel
form (DEM plus 1-D `zeta_faces`) is retained exactly as a legacy
constructor path.

**Geometric constraints (enforced, rejecting rather than approximating):**
finite interfaces; strict per-column ordering `z_interfaces[k] >
z_interfaces[k+1]` (crossing, touching and inverted surfaces and
zero/negative-thickness cells raise); one complete stack per active
column; a single terrain layer requires a flat surface because the
transformation itself cannot reconstruct the vertical head gradient that
drives along-slope flow. Pinch-outs, disappearing units, faults and
discontinuous contacts remain unsupported and are rejected explicitly.
Minimum **geometric** thickness is distinct from the 0.01 m numerical
saturation floor.

## 2. Discretization

### Horizontal (x/y) faces

The normal conductance is the harmonic series of half-cell
**transmissivities** `K·f·T` (f = regularised saturated fraction) over the
unit-s map face width, so variable thickness enters exactly where the
physics puts it; for a pure horizontal head gradient across heterogeneous
thickness the scheme returns the exact series resistance. The metric cross
terms use the in-layer average of the two bounding interface slopes at the
face. Affine total-head fields are reproduced exactly, including through
thickness changes (`tests/test_3d_terrain_reference.py::
test_variable_interface_affine_exactness`).

### Vertical (s-direction) faces

Both the normal conductance (`Azz/T` in half-layer series, area = map
cell) and the metric cross terms evaluate the **shared interface's own
slopes**. The resistance split remains the harmonic series of the face,
which keeps low-permeability cells free of couplings scaled to a
transmissive neighbour — a slope-based split was tried and rejected after
it degraded the row-scaled condition number of an aquifer/basement system
to ~1e16 and made restarted GMRES diverge. The residual defect — the
resistance-weighted mean of cell-centre slopes differs from the interface
slope — is cancelled exactly by metric-defect terms `−γ·∂h/∂s` per side,

```
γ_top  =  wl·(ex·zx_if·(zx_c,top − zx_if) + ey·zy_if·(zy_c,top − zy_if))·dx·dy
γ_bot  =  wr·(... zx_c,bot ...),   (wl, wr) = harmonic resistance weights
```

which vanish identically for parallel interfaces and make every affine
total-head field exact for uniform conductivity on **arbitrary**
interfaces. The packed GPU operator carries 13 coefficient columns per
cell (previously 11); the GPU kernels apply the same authoritative flux
equations as the reference, and GPU/CPU face parity is asserted to 3e-13.

Conservation holds by construction (one authoritative face flux, identical
and opposite contributions), the reference assembles independently via
gradient/incidence matrices, and the assembled operator is not assumed
symmetric or positive definite — the Krylov operator is the full
conservative transformed operator and the seven-point principal part is
only a right preconditioner.

### Contacts (coastal/surface)

Each exterior subface retains its own physical elevation, area, distance
and orientation. Membership validation reconstructs the bounding
interfaces at the physical subface point (half a cell toward the face for
lateral faces, including the one-sided slope extrapolation), so a coastal
connection is never inferred from the owning cell's centre elevation.
Top/bottom conductance defaults use their own interface metric. Multiple
subfaces per cell, discharge-only gating and flow signs are validated on
nonparallel geometry (`test_coastal_gating_on_nonparallel_interfaces`).
Recharge continues to use horizontal map area.

## 3. Iteration-control corrections

* **GMRES budget (`solvers/terrain_gmres_3d.py`).** `max_cycles` now means
  the exact permitted Krylov-iteration budget. Each restart block runs
  `min(restart, remaining)` Arnoldi steps; a shortened final block zeroes
  the stale Hessenberg columns so the least-squares solve stays exact.
  CUDA graphs are captured only for full blocks and reused; the eager
  shortened block neither captures nor corrupts the graph. The returned
  count is the actual Krylov iterations. Tests cover budgets below,
  non-divisible by, and equal to the minimum, zero-iteration exit,
  already-converged guesses, nonconvergence at the limit, and graph
  accounting. (Upstream Warp documents overshoot by up to `restart − 1`
  iterations; this subclass no longer overshoots.)
* **Accepted-step diagnostics (`solvers/terrain_fast_3d.py`).** The
  Picard history and last-check info carry proposed **and** accepted head
  changes; `head_change_max` refers to the accepted update, so a small
  damped step is no longer readable as convergence. Final residual and
  independent budget fields describe the accepted state.

## 4. Independent validation

All in `tests/test_3d_terrain_reference.py`, `tests/test_3d_terrain_gpu.py`,
`tests/test_3d_terrain_gmres.py` (192+ tests green, plus the pre-existing
matched flat-terrain MF6 regressions in `tests/test_3d_physics_validation.py`).

**Geometry:** flat Cartesian reduction to the 7-point operator (exact);
constant-thickness terrain-following; laterally thickening/thinning;
opposite-sloping interfaces; curved nonparallel interfaces; strong lateral
thickness changes; very thin strictly positive layers; invalid crossing,
touching, inverted and nonfinite interfaces rejected.

**Analytical:** constant head → exactly zero flux on all faces; pure
vertical physical gradient → exactly zero lateral flux on any interface
shape (discrete cancellation identity `C·dx ≡ wl·t_L + wr·t_R`); independent
horizontal gradients exact through the harmonic series; combined
three-dimensional affine fields exact on arbitrary interfaces with uniform
K (including the vertical faces via the metric-defect terms, tolerance
2e-11); datum-shift invariance of fluxes (legacy form exact, elevation
form within construction rounding, ≤1.4e-14 relative); heterogeneous K and
anisotropy via independent sparse consistency; low-K basement series
resistance recovered to discretisation accuracy; one-flux-per-face
continuity across a 250× aquifer/aquitard contrast.

**Manufactured solutions:** continuous calculus sources (never the
discrete operator) on independently curved interface families converge
under simultaneous x/z refinement with monotone rates (errors dropping
≥3× per refinement level at 16→32→64); the parallel-interface manufactured
regression is retained.

**Transient storage:** per-cell exact integrated Sy/Ss volume on
nonuniform layers (analytic target head as the unique wet-range fixed
point); multi-period drainage/rewetting across nonparallel layers with Sy
contrasts behind a gated coastal contact, where the integrated stored
volume equals the summed forcing minus the reported contact outflow every
period to 1e-6 relative. The 0.01 m floor remains the default, is capped
by local cell thickness, creates no artificial storage
(`storage_potential_3d` exact), and its transmission sensitivity is
reported through the dry diagnostics rather than hidden by mass
conservation.

**Independent reference:** the CPU sparse operator (gradient/incidence
assembly, distinct from the GPU stencil) is validated against the face
divergence, row/column null-space identities, and direct sparse solves;
the frozen-system sparse check inside the benchmark independently solves
the final constitutive system. MODFLOW 6 comparisons remain limited to
matched flat-terrain geometry, where MF6 genuinely represents the same
discrete problem.

## 5. Performance

See `docs/3d_terrain_benchmarks_variable_geometry.json`
(working_tests/benchmark_3d_terrain.py, representative 64³ shallow-aquifer
terrain case — 900 m relief, hard heterogeneous K, transient, GPU repeats
after warm-up; the `3c203e2` acceptance report is the baseline):

| Floor [m] | Baseline first/repeated [s] | Now first/repeated [s] | Krylov baseline→now | Outer baseline→now |
|---|---|---|---|---|
| 0.1   | 6.46 / 6.91 | 7.91 / 7.50 | 4440 → 4452 | 237 → 238 |
| 0.01  | 6.75 / 7.04 | 8.11 / 8.09 | 4704 → 4668 | 248 → 245 |
| 0.001 | 7.04 / 7.08 | 10.38 / 11.06 | 4932 → 5040 | 266 → 275 |

Physics is unchanged on the parallel special case (iteration counts match
to noise); the wall-time delta (~8–15%, up to ~55% at the 0.001 floor) is
per-outer host arithmetic of the generalised operator: per-interface and
per-layer slope fields, the metric-defect terms, and the 13-column packed
coefficient layout. Artificial overhead was removed: coefficients are
packed exactly once per outer iteration, operator builds no longer
re-validate the session's owned K fields (`trusted_k`), and the stale
refresh check compares versioned inputs (K version, geometry revision,
floor, head, storage, boundary, masks) instead of scanning the full packed
array. The previous silent GMRES overshoot of up to `restart−1` Krylov
iterations per linear solve is gone, so reported counts are exact. Per the
task priorities, further host-side optimisation is left for a later pass.

## 6. SHANZ-style demonstration

`working_tests/shanz_terrain_case_study.py` (results:
`docs/shanz_case_study_results.json`): 96×96 columns at 20 m spacing,
137.5 m relief over a ~51 m shallow geological domain; 0.1–0.5 m topsoil,
a ~5 m permeable aquifer with realistic lateral variation, an
independently varying weathered zone, basalt basement (K = 0.1 m/day)
varying independently of the ground surface; water table initially 1 m
below ground; spatially varying recharge ~1.2 mm/day; fixed sea-level
head through the coastal aquifer plus three gated discharge-only springs
on the topsoil coastal subface. Variants warm-start from the best
previous solution; up to 800 Picard outers.

| Quantity | Parallel layers (previous approx.) | Variable interfaces | 0.001 m floor (variable) |
|---|---:|---:|---:|
| Head RMSE vs variable 0.01 [m] | **1.14** (max 2.70) | — | 0.09 (max 0.71) |
| Water-table depth RMSE [m] | 1.16 | — | 0.10 |
| Mean water-table depth [m] | 5.11 | 4.56 | 4.51 |
| Aquifer leakage downward [m³/day] | 1096 | 819 | 819 |
| Weathered→basement leakage [m³/day] | 561 | 344 | 345 |
| Surface discharge-potential cells | 986 | 1264 | 1266 |
| Floor-excess lateral transmission [m³/day] | 606 | 586 | 59 |
| Rel. cell imbalance (achieved, 800 outers) | 1.0e-4 | 1.3e-4 | 1.3e-4 |

Findings: the previous parallel-layer approximation is **materially**
different on this geology — ~1.1 m RMSE / 2.7 m maximum head error,
~35–60% errors in aquifer and basement leakage, and 22% fewer cells
where groundwater could discharge to the surface (the thin, variable
topsoil controls daylighting). The saturation-floor sensitivity matches
the audit experience: refining 0.01→0.001 m moves heads by only ~0.09 m
RMSE but removes ~90% of the gross floor-driven transmission, which is
why the dry diagnostics — not mass conservation — carry the accuracy
statement. Under this recharge scenario the domain drains to the fixed
sea boundary and the springs remain gated (reported outflows ≈ 0);
surface-discharge potential is the meaningful overland indicator. None
of the 800-outer runs reached the strict balance gate: the Picard tail
plateaus near 1e-4 relative on this scale (32² variants with the same
controls converge; two of four reach the strict gate), and the plateaued
results are reported with their achieved residuals rather than declared
converged.

## 7. Remaining limitations

* Nearly-dry cells still transmit through the saturation floor; the dry
  diagnostics quantify the excess transmission relative to recharge and
  abstraction, and floor sensitivity (0.1/0.01/0.001 m) is part of the
  acceptance runs. Physical accuracy is never claimed from mass
  conservation alone.
* The demonstration's Picard tail contracts slowly on extreme
  cases (plateau ~1e-6 relative balance); variants that do not reach the
  gate are reported with `converged: false` and their achieved residuals.
* Interior prescribed-head cells act as implicit reservoirs: the
  integrated domain budget closes only through contact fluxes or
  free-cell bookkeeping (the tests document this explicitly).
* MODFLOW 6 validation remains restricted to matched flat-terrain
  geometry; an unstructured independent discretisation of genuinely
  variable geometry is future work.
* Pinch-outs, disappearing units and discontinuous contacts are rejected,
  not approximated.


## 8. Addendum (same day): GPU-resident Picard port and the 96×96 stagnation isolation

### 8.1 Device-resident Picard (device_picard=True)

The production 2D fast unconfined execution strategy is ported to the 3D
terrain solver (`solvers/terrain_device_3d.py` + `terrain_device_kernels.py`,
commit series from `f2741b5`). Per outer iteration the head, transformed
coefficients, secant storage, RHS, corrections and residual stay on the
device: kernels refresh saturation regularisation, the packed 13-column
face operator and storage diagonals (validated against the host packer to
1e-12 and the host secant terms to 1e-13), freeze the discharge-only gates
at the accepted head (a gate re-evaluated against live Krylov iterates
would make the operator nonlinear inside GMRES), extract the seven-point
principal part into the multigrid level-0 buffers with the row scale, and
rebuild the row-scaled RHS with contact offsets. The inner solve runs in
correction form (A·δ = b − A·h, homogeneous prescribed-head constraints)
under the 2D-style adaptive inexact controller: restart-cycle blocks,
Eisenstat–Walker forcing, divergence rollback via device snapshots,
stagnation detection, and acceptance of usable inexact corrections.
Damped clipped updates and rejection rollback run on device; two scalar
readbacks cross per outer iteration. Final acceptance remains with the
independent host-side physical budget.

### 8.2 Computational result

On the representative SHANZ case the device Picard reaches the same
residual plateau as the host path at a fraction of the cost: 96² steady,
400 outers, 9600 Krylov iterations (24/outer, exact budgets, no
whole-block overshoot) in ~3-4 s versus 576 s / 1.88 M Krylov for 800
host-driven outers — ~140× less work and wall time, with 2 host reads
per outer instead of full-field downloads. On a transient storage-only
analytical case the device and host paths agree to 2e-7 m with the exact
[9, 5] fixed point and the independent integrated volume balance closed
(`test_device_picard_matches_host_and_independent_volume`).

### 8.3 The 96×96 residual stagnation: isolation

The plateau is a NONLINEAR (physical) stagnation, not a solver-machinery
failure, and is isolated by experiment (`docs/terrain_stagnation_isolation.json`):

1. **Not the linear solver.** GMRES corrections on the frozen system are
   usable (10× contraction per block); the TRUE residual recomputed at the
   refreshed coefficients does not follow the linearised prediction.
2. **Not the discharge-only springs.** Removing every spring (fixed-head
   coastal boundary only) reproduces the identical frozen residual
   (0.0385) bit-for-bit in magnitude.
3. **Not the floor magnitude per se.** floor 0.1 m changes the stall level
   (0.034) but not the stall's existence.
4. **Mechanism.** The saturation-floor regularisation makes the conductance
   of clamped (near-dry) cells head-DECOUPLED: the frozen-operator
   Jacobian has no entries to drive the true residual in those regions,
   every damped correction is rejected by the independent backtracking
   evaluation, and the true residual sits at a nonzero local minimum.
   The dry-cell count grows with resolution (735 / 2643 / 8651 at
   16²/32²/64²), which is exactly why finer cases stall worse (16²
   nearly converges; 96² plateaus at ~1e-4 relative). This is the
   documented quasi-3D regularisation limitation (saturation-independent
   vertical K + floor), now quantified as the binding constraint on
   nonlinear convergence.

Per the task's acceptance criteria (§16.8) the remaining failure
mechanism is thus demonstrated with reproducible minimal evidence: any
resolution-ladder run of `working_tests/terrain_convergence_baseline.py
--device-picard` reproduces it in seconds. A smooth (Richards-type or
smoothed-floor) constitutive law in the dry region — a physics change,
explicitly out of scope for this pass — is the identified remedy.


### 8.4 Second-pass audit (post-fix re-assessment)

An implementation audit found and fixed five genuine defects in the first
device port (commit `9a79d6e`): the discharge flag column was lost in the
device boundary pack and subfaces were aggregated per cell before gating
(now per-subface rows, independent gate evaluation, atomic scatter of the
open set — CPU-reference parity for multiple contacts per cell); the
backtracking trial residual reused the outer-top RHS (now the full
nonlinear state — coefficients, storage, gates, aggregation, level-0
extraction, the FULL coarse hierarchy and the candidate's own RHS — is
evaluated at every candidate, with the accepted state preserved and
rolled back in full); the coarse multigrid levels were never refreshed
after construction (now device semi-coarsened kernels replicate
`_coarsen_faces_consistent_axes` exactly, host parity 1e-12, refreshed
every evaluation); Krylov accounting dropped failed-inner iterations and
the discharge-seed statistics were fictional (both fixed); a 2D-style
retry-at-full-effort policy precedes any non-usable-inner failure
declaration.

With the corrected solver the ladder was rerun host-vs-device under
identical physics, initial conditions and tolerances
(`docs/device_host_ladder_fixed.json`): 16² device residual 2.1e-7
(better than host 2.5e-7) at 24 Krylov/outer and 2 s; 64² comparable
residuals (5.0e-2 vs 1.1e-1) at 4.7× less time and 2.7× less Krylov; 32²
device residual is currently worse (4.9e-3) despite lower cost — an
adaptive-tuning matter, reported honestly; 96² declares non-convergence
after 6 outers plus two full-effort retries (0.7 s) where the host
churns 429 s to the same verdict.

**Re-assessed stagnation.** The residual plateau persists on BOTH paths
after every identified implementation defect was fixed and verified by
parity tests. With candidate evaluation, gating, hierarchy refresh and
rollback now demonstrably correct, the earlier isolation stands on a
corrected footing: the plateau is a genuine nonlinear stagnation of the
frozen-operator iteration at the head-decoupled dry-region conductance
kink — not an artifact of the solver machinery. The remedy remains a
smoothed dry-region constitutive law (physics change, future pass).
