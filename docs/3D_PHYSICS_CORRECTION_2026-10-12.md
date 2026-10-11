# 3D Physics Correction — dev branch, 2026-10-12

Comprehensive scientific correction of the DarcyWarp 3D implementation on
`dev`, targeting the **fast** execution path
(`WarpDarcySolver3D(implementation="fast")`,
`DARCY_WARP_PACKAGE/solvers/fast_3d.py` + `session_3d.py` +
`face_kernels_3d_f64.py`) while preserving its architecture (persistent device
buffers, face-array operators, semi-coarsened K-cycle multigrid, CUDA-graph
replay).

## 0. Structural repair: the fast path was missing from `dev`

`warped_darcy_3d.py` (tracked) imported `solvers/session_3d.py`, but
`session_3d.py`, `fast_3d.py`, `face_kernels_3d_f64.py` and five test modules
existed only as **untracked working-tree files**; a clean `dev` checkout could
not run the 3D solver at all.  Commit `36d59fe` adds them verbatim as the
audit baseline; corrections follow in separate commits.

### Actual execution path (post-repair)

- Confined steady/transient: `WarpDarcySolver3D.solve` →
  `ThreeDOperatorSession.solve` → `solve_fast_confined_kcycle_3d`
  (`implementation="fast"`) or `solve_multigrid_kcycle_7point_3d` /
  `solve_chebyshev_7point_3d` (`"classic"`).
- Unconfined: previously the fast path **raised** and only the classic Picard
  path ran.  Commit `b79c0a9` adds `solve_fast_unconfined_kcycle_3d`: the
  shared Picard driver (`_picard_unconfined_7point_3d`) drives inner linear
  solves on a persistent fast workspace whose device buffers are refreshed in
  place each outer iteration, so the captured CUDA graph stays valid
  (verified: repeat-call head diff 5e-13 m with workspace reuse).

## 1. Governing equation and discrete form

Total hydraulic head `h` [m]; positive volumetric source `Q` [m^3/day] =
injection/recharge.  Cell-centered 7-point finite volumes on a uniform
Cartesian grid (dx, dy, dz [m]), layers flat/contiguous (see §7):

```
sum_faces C_f (h_i - h_j)  +  S_i (h_i^{n+1} - h_i^n)/dt  =  Q_i        (1)
```

- Face flux `q_ij = C_ij (h_i - h_j)` [m^3/day]: flow from higher to lower
  head; each internal face appears exactly once in each cell with opposite
  sign (equal and opposite across the shared face; verified by exact
  telescoping in the budget evaluator and the analytical flux tests).
- Confined faces: `C_x = 2 Kx_i Kx_j / (Kx_i + Kx_j) * dy dz / dx`
  (harmonic resistance across the interface), unchanged and independently
  verified by `tests/test_3d_operator_analytical.py`.
- Storage (confined): `S_i = storage_coeff * dx dy dz` (storage_coeff =
  storativity, dimensionless).
- Row assembly: `diag = sum(C) + S/dt`, off-diagonal `-C`; residual
  `r = b - A h`; Dirichlet cells are identity rows pinned to `bc_values`;
  no-flow = absent faces.

## 2. Confirmed defects and corrections

### D1 — Unconfined conductance double-counted saturated thickness (c9db077)

The Picard loop passed `K * sat` (a transmissivity, m^2/day) into the confined
builder, which multiplies by `dy*dz/dx` — horizontal conductances were a
factor `dz` too large and vertical conductances were spuriously scaled by
saturation.  New `build_7point_face_conductance_unconfined_from_sat`
(quasi-3D, MODFLOW 6 NPF convertible semantics):

- horizontal: `C_x = harmonic(T_i, T_j) * dy/dx`, `T = K_h * sat` — the
  saturated thickness enters exactly once through the saturated face area
  `dy * sat`; K keeps units m/day;
- vertical: full-thickness harmonic resistance
  `C_z = dx*dy / (dz/(2 Kz_i) + dz/(2 Kz_j))`, saturation-independent
  (MODFLOW-NWT-style), so columns drain and rewet smoothly;
- `sat = clip(h - zbot, min_sat, dz)`: the pre-existing `min_sat = 0.1 m`
  numerical floor is retained for conductance only (cells never fully
  deactivate; this is not a drying/rewetting package).  Its effect is
  quantified by the drying tests: a dry cell leaks at most
  `T_floor * dh / dx` with `T_floor = K * min_sat`; storage uses the true
  (unfloored) saturated thickness.

### D2 — Transient unconfined storage not volume-conservative (2478c91)

The legacy `phreatic_sy` mode put `Sy*A/dt` on the single current water-table
cell per column; when the water table crosses a layer boundary within a step,
the Sy jump lands on the wrong cell and volumes are wrong.  New default mode
`secant_sy`: backward Euler on the per-cell stored-water potential (per unit
plan area, [m])

```
W_k(h) = Sy_k * b_sat,k(h) + Ss_k * ( b_sat,k(h)^2 / 2 + dz * max(h - ztop_k, 0) )
b_sat,k(h) = clip(h - zbot_k, 0, dz)
```

linearised in the Picard loop with the **secant slope between the fixed
previous-timestep head and the current iterate**, so the converged step
satisfies exactly `A * (W(h_new) - W(h_old)) / dt`.  This handles intra-layer
movement, layer-boundary crossings with different Sy per layer, complete
drying/rewetting, and fully saturated (confined) columns
(`dW/dh = Ss*dz` above the top).  Verified against the analytical inverse of
`W(h)`: within-layer 2e-8 m, crossing into confined storage 5e-10 m,
dry/rewet cycle exact.

Also: a transient unconfined call without usable storage inputs now **raises**
(previously it silently solved a steady inner problem while reporting
`transient=True`).  `phreatic_sy` and `confined_volume` remain as
explicitly-named legacy modes.

### D3 — Coarse-grid operators inconsistent (d0265f1)

Each directional face array was block-averaged independently, so opposite
sides of one coarse interface could disagree; interior faces inflated the
coarse diagonal; the max-rule Dirichlet coarsening turned free fine cells into
fixed coarse cells; block means used edge-replication padding.

New `_coarsen_faces_consistent_axes`: coarse faces aggregate only the fine
faces crossing each interface with both fine cells free, summed and scaled by
`1/(fy*fx)` (the Galerkin scaling for the block-average restriction used by
the K-cycle); both sides of a coarse interface are identical by construction
(reciprocity).  Free→Dirichlet couplings become explicit diagonal **springs**
(`coarse_bc_springs`, carried on the storage-diagonal slot), so mixed
Dirichlet/free coarse blocks stay free yet non-singular — this satisfies
"free cells stay free" without the singular coarse operator a naive all-free
rule produces.  Coarse Dirichlet mask: a coarse cell is fixed only when all
its active fine cells are fixed.  Storage coarsens with the same block-sum
scaling.  Verified against an explicitly assembled Galerkin `R A P` product:
agreement to ~5e-8 on full blocks; symmetric conservative approximation on
Dirichlet-mixed blocks.  Side effect: representative fast confined cases
converge in half the cycles (40 → 20) with ~15x tighter terminal residuals.

### D4 — Identity pinning of singular cells (3a2bfd6)

Cells with zero conductance AND zero storage (equation `0*h = rhs`) were
silently identity-pinned (`diag=1`), manufacturing `h = rhs`.  Now
`_resolve_isolated_free_cells` runs before every solve (and on every
refreshed operator in the unconfined loop): a nonzero source into a
disconnected cell **raises** `ValueError`; source-free singular cells are
excluded from the working set, keep their initial head, carry zero flux and
zero residual, and are reported via `info["n_isolated_cells_excluded"]`.

### D5 — False convergence / missing physics gate (d7ee885)

After every unconfined nonlinear solve the operator is rebuilt at the final
accepted state and the true discrete balance is evaluated:
`physics/budget_3d.compute_3d_flow_budget` reports per-cell imbalance
(max/RMS), source total, Dirichlet boundary inflow/outflow, storage uptake,
and domain imbalance (internal faces telescope exactly).
`info["converged"]` now additionally requires the final head-equivalent
residual RMS `<= final_head_residual_tol` (default `hclose`) — convergence is
no longer declared from small head updates alone (demonstrated: a
head-change-converged case with 2.5e-8 m physical residual above a 1e-8 gate
is correctly refused).  Confined backends attach the same budget to `info`.

## 3. Geometry limitations (task §7)

The 7-point operator with scalar (dx, dy, dz) is exact only for flat,
contiguous, uniform-thickness layers (full face overlap, cell-centered
distances).  `_validate_flat_layer_geometry` enforces this for unconfined
solves: dipping layers, intra-layer elevation variation, non-uniform
thickness, or gaps between layers raise `ValueError` with an explicit message
instead of silently using Cartesian face geometry.  Vertically offset /
dipping-layer support would require a geometric extension (per-face areas and
distances, or a 19-point stencil) — future work.

## 4. Validation summary (tests/test_3d_physics_validation.py, both backends)

| Check | Result |
|---|---|
| 1D horizontal confined flow | exact profile ≤1e-9 m; boundary flux ≤1e-9 rel |
| 1D vertical confined flow | exact profile ≤1e-9 m |
| Two-material series resistance | exact vs physical half-cell resistances ≤1e-9 m |
| Hydrostatic multilayer, heterogeneous K | heads constant ≤1e-8 m; zero boundary flux |
| Recharge/abstraction column | exact head rise/drop ≤1e-9 m; budget closure <1e-9 |
| Confined transient storage, 3 chained steps | exact q*dt/(S*V) ≤1e-10 m |
| Unconfined secant storage (in-layer / crossing confined / dry+rewet) | exact vs W(h) inverse ≤1e-6 m |
| Over-abstraction of disconnected column | convergence refused; budget shows exact unmet sink |
| Isolated regions | excluded, reported, connected region exact; source → ValueError |
| Coarse faces | reciprocal by construction; springs ground mixed blocks; Galerkin ≤5e-8 |
| Dipping geometry | raises |
| MF6 steady unconfined (NPF convertible) | max head diff < 2e-3 m, both backends |
| MF6 transient unconfined (STO, 4 draining periods) | max head diff < 5e-2 m, both backends |

Full repo suite: **326 passed** (75 of them 3D), incl. all pre-existing MF6
parity fixtures.  2D untouched (no 2D file modified).

## 5. Performance (fast path, pre → post correction)

Representative confined cases (`working_tests/benchmark_3d_fast_baseline.py`;
baseline JSONs in `working_tests/baseline_3d_fast_{pre,post}_correction.json`):

| case | solve1 pre | solve1 post | solve2 pre | solve2 post | cycles pre→post | r_rms_end pre→post |
|---|---|---|---|---|---|---|
| steady 4x64x64 | 1.48 s | 0.30 s | 0.174 s | 0.048 s | 40→20 | 7.6e-5→6.3e-6 |
| transient 4x64x64 | 0.18 s | 0.07 s | 0.158 s | 0.048 s | 35→20 | 4.2e-5→3.6e-6 |
| steady 6x256x256 | 0.39 s | 0.19 s | 0.319 s | 0.121 s | 40→20 | 7.8e-5→3.3e-6 |
| transient 6x256x256 | 0.40 s | 0.20 s | 0.330 s | 0.123 s | 40→20 | 4.5e-6→1.5e-6 |

The consistent coarse operator roughly halves the required cycles, so the
correction is *faster* than the baseline everywhere (2–5x on first solves).
GPU memory: 181 → 440 MiB at first solve of the 6x256x256 case, flat across
repeat solves (no per-solve growth).

New unconfined fast path (3x96x96 heterogeneous, CUDA): steady 65.0 s classic
→ 1.7 s fast (~38x); transient step 16.5 s → 0.5 s (~33x); repeat-call head
diff 5e-13 m (workspace + graph reuse).

## 6. Remaining limitations

- **Stiff unconfined cases**: the shared Picard driver's legacy outer
  Chebyshev acceleration can oscillate in the convergence tail on hard
  heterogeneous cases (observed dh ~ 4e-3 m oscillation at outer 300 with
  `chebyshev_enabled=True` on a 3x96x96 lognormal-K case); with it off the
  tail creeps (contraction ~1.0).  Both backends share this machinery and
  report honestly (`converged=False`, physical residuals and budgets reflect
  the achieved iterate).  On validated cases (analytical, MF6) it converges
  cleanly.  The 2D adaptive inner controller / strict-practical acceptance
  machinery is not ported to 3D.
- `min_sat = 0.1 m` conductance floor: dry cells leak
  `K*min_sat*dy/dx * dh`; no full cell deactivation.  Quantified in the drying
  tests; reduce via `unconfined_min_sat` if the leak matters.
- Flat/uniform-thickness/contiguous layers only (see §3).
- Vertical conductance is saturation-independent (NWT-style); perched
  discontinuities with true vadose isolation are not modelled (no Richards).
- rhs units are volumetric per cell [m^3/day]; callers convert rates by the
  cell plan area (now documented in the solve docstrings via the budget
  helper and this note).

## 7. Future performance improvements (not implemented here)

- Device-side face refresh (coarsening + diagonal build as kernels) to remove
  the per-outer host coarsening in unconfined solves.
- Adaptive inner-cycle controller (2D-style) for the 3D unconfined Picard
  loop, and a secant-aware outer accelerator replacing legacy Chebyshev on
  stiff cases.
- Per-layer `dz` support in the face builders (flat layers, variable
  thickness) — small generalisation of the geometry validation.
- FP32 inner-correction mixed precision for the 3D fast path (mirror of the
  2D `mixed_transient_f32`), gated on the corrected FP64 operator.
