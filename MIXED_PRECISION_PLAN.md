# Mixed-Precision Defect-Correction Solver — Audit, Plan, and Results

> Experimental investigation, 2026-07-29/30. Target: FP32 speed/memory with FP64-grade
> absolute-head accuracy for the **steady confined** 2D K-cycle solver.
> Status: COMPLETE — benchmarked; recommendation in §4 (**numerically validated on the
> tested cases but no performance advantage in the current K-cycle implementation**;
> retained as experimental, opt-in, non-production).
>
> **Follow-up:** the optimisation campaign that revisited these conclusions lives in
> the companion document `MIXED_PRECISION_CAMPAIGN.md` (2026-07-30) — it achieved a
> 3.0–4.4× end-to-end win via new fast kernels + CUDA-graphed correction blocks
> (`solvers/mixed_fast.py`), while confirming this document's core diagnosis (the
> original K-cycle's cost profile, not defect correction itself, blocked the win).

## 1. Precision-path audit (deliverable 1)

### 1.1 Global precision switch

- `warped_darcy.py` l. 104–113: `DARCY_FLOAT` env read **at import**; default here is
  `float64`. `config.py` defaults to `float32`. Every model/hierarchy array is
  `WP_FLOAT`; kernel signatures bind `WP_FLOAT` at module import / first compile.
  ⇒ A mixed-precision path cannot reuse FP64-compiled kernels for FP32 work. The
  mixed path therefore runs the solver built under `DARCY_FLOAT=float32` (whole
  K-cycle machinery + hierarchy in FP32) and adds **separately declared
  `wp.float64` master arrays + new FP64 kernels** in a new module.

### 1.2 Array ownership (build_from_truth_inputs, l. 5209–5378; hierarchy l. 4424–4694)

| Array | dtype | Notes |
|---|---|---|
| `T_wp`, `R_wp`, `bc_values_wp`, `gh_head_wp`, `gh_width_wp`, `ghb_factor_wp`, `M_inv_wp` | `WP_FLOAT` | **host fields are cast to `NP_FLOAT` first** (l. 5252–5259), so under `DARCY_FLOAT=float32` the *host* boundary/GHB/T data is already quantized before upload. The mixed path must source its FP64 master data from the caller's original float64 numpy arrays. |
| `active_wp`, `bc_mask_wp`, `gh_mask_wp` | `wp.int32` | exact |
| per-level work: `x, b, r, Ax, e, z, p, Ap, x_prev` | `WP_FLOAT` | level 0 shares model arrays |
| scalar buffers (`rTr_buf`, `rho_buf`, …, `dh_max_buf`) | **`wp.float64` always** | all reductions already FP64 |
| DEM/bottom/top | — | no device arrays on the confined path |

### 1.3 Kernels that store FP32 but compute in FP64

Structural pattern throughout: stencil rows are accumulated in FP64 via in-kernel
`wp.float64(...)` casts, then rounded to `WP_FLOAT` on store:

- `compute_residual_kernel` (l. 1856): row in FP64, `r = WP_FLOAT(rf64)` on store,
  `rTr` atomic in FP64. **Residual state is rounded to FP32 every evaluation.**
- `jacobi_applyA_fused_kernel` (l. 1416): stencil in FP64, but the update
  `x_out = WP_FLOAT(hC) + omega*M_inv*r_ij` rounds, and `r_ij = b − Ah` is formed in
  WP_FLOAT (l. 1500–1501).
- `restrict_blockavg_kernel` (l. 1503) and `prolong_bilinear_any_kernel` (l. 1551):
  all-WP_FLOAT arithmetic.
- `add_correction_kernel` (l. 1599): `x_f += e_f` in WP_FLOAT; Dirichlet cells
  re-pinned to `bc_values` (⇒ correction is zero on Dirichlet — exactly what defect
  correction needs).
- `apply_A_kernel` (l. 2479): neighbor products WP_FLOAT, row sum promoted FP64.
- PCG fused kernels: dot products in FP64 atomics; `x/r/z/p` state WP_FLOAT.

### 1.4 Reductions

All norms/dot products are custom per-thread partials + `wp.atomic_add` into
`wp.float64` scalar buffers — already FP64 everywhere. One host sync per readback.

### 1.5 Boundary/BC semantics of the correction equation

- Dirichlet rows are identity (`A·h = h`, residual forced to 0).
- Coarse levels get homogeneous BCs (`bc_values=0`, `gh_head=0`, l. 4538–4540).
- GHB conductance `C_gh = T·ghb_factor` enters **both** the diagonal and the RHS
  (`rhs += C_gh·gh_head`, `build_rhs_kernel` l. 2267–2272).
- ⇒ The existing K-cycle already *is* an error-equation solver with homogeneous
  Dirichlet corrections and a homogeneous-GHB-source / retained-GHB-diagonal
  operator, provided `bc_values=0` is supplied and `b` is the residual.

### 1.6 Likely ordinary-FP32 accuracy floor

Even though row math is FP64 internally, **all cross-kernel state is FP32**:
heads O(10–100 m) ⇒ FP32 ulp ≈ 1–6e-6 m per update; the Jacobi update's local
residual `b − Ah` is formed in FP32; residuals stored FP32. Expected floor:
head RMS error ~1e-6–1e-5 m vs FP64, with possible stagnation of the
residual-norm convergence criterion near rel_tol ~1e-7. Baselines (§3) will
measure the actual floor.

## 2. Design (deliverable 2)

### 2.1 Algorithm

Single solver invocation, started from the DEM:

```
h64 ← DEM (FP64; exact Dirichlet heads applied)
b64 ← R·dx² (+ C_gh·gh_head) assembled host-side in FP64   (constant)
r64 ← b64 − A(h64)              # FP64 true residual, FP64 norm
while not converged:
    r32 ← float32(r64)                       # zero on Dirichlet/inactive already
    δ32 ← 0
    δ32 ≈ M32⁻¹ r32   # one configurable block of N FP32 K-cycles on the
                      # existing hierarchy, homogeneous BCs, GHB diagonal kept
    h64 ← h64 + float64(δ32)   # FP64 accumulate; Dirichlet re-pinned to bc64
    r64 ← b64 − A(h64)         # FP64 true residual BEFORE any convergence check
converged ⇔ r_rms64 ≤ max(abs_tol_min, rel_tol·r_rms0_64)
            AND dh_max ≤ dh_max_tol AND dh_rms ≤ dh_rms_tol
```

No recursively-updated residuals are used for convergence. The FP32 inner solve
uses `solve_kcycle_device_buffers` (the same machinery used as the Newton
preconditioner) with `bc_values=0`, `x=δ32`, `b=r32`, fixed-work mode
(no inner scalar reads), and the caller's buffers snapshot/restored around the
call — no duplication of the K-cycle implementation.

### 2.2 Retained experimental interface

- **`DARCY_WARP_PACKAGE/solvers/mixed_precision.py`** (experimental, opt-in,
  non-production; not registered in `solver_capabilities.py` and reachable by
  no alias):
  - `_mp_residual_f64_kernel`: FP64 `h/b/r` arrays; coefficients (T,
    ghb_factor) loaded from the model's dtype-generic arrays and promoted to
    FP64 in-kernel; FP64 `rTr` atomic; zero on inactive/Dirichlet.
  - `_mp_cast_r64_to_r32_kernel`: `r32 = float32(r64)`.
  - `_mp_accumulate_kernel`: `h64 += δ32`, Dirichlet re-pin to `bc_values64`,
    `dh_max`/`dh_sq` FP64 atomics.
  - `MixedPrecisionDefectCorrectionSession(model, *, bc_values_f64,
    gh_head_f64, R_f64, max_levels, min_coarse_cells)` + `.solve(
    initial_head_f64, **controls)` — the only public entry point; requires the
    model to be built under `DARCY_FLOAT=float32` (enforced at construction,
    which also emits an experimental-status warning); FP64 master inputs come
    from the caller's original (unquantized) numpy arrays.
- The FP64-transmissivity variant (`use_f64_transmissivity`) was removed after
  benchmarking showed coefficient quantisation was not the limiting error
  source in the tested cases (§3.5); the one-shot forwarding wrapper was
  removed as redundant.
- Explicit opt-in only. No change to FP64/FP32 paths, registry defaults, or
  tolerances.

### 2.3 Benchmark integrity

- Harness `working_tests/mixed_precision_benchmark.py`: modes
  `fp64|fp32|mixed`; every timed solve (cold and CUDA-warm) starts from the same
  DEM host array; one solver invocation per timed run; all K-cycles and all
  defect-correction iterations are inside the timed call.
- MF6 artifacts loaded from the existing cache; MF6 runs only on cache miss.
- FP64 Warp reference heads cached per case for FP32/mixed comparison.
- Accuracy gate unchanged: max |h − h_MF6| ≤ 2e-4 m.

### 2.4 Cases

100×100 (small square), 100×1000 (elongated), 400×400, 1000×1001, 2000×1000
(large); heterogeneous `ugly_t` seed 123 + GHB (production benchmark config),
plus a 400×400 isotropic no-GHB physics-coverage case.

## 3. Results

Benchmarked 2026-07-30 via `working_tests/mixed_precision_benchmark.py` (one mode
per process; every timed solve starts from the original DEM; cold + CUDA-warm
timing; MF6 from cache only). Cases: heterogeneous `ugly_t` seed 123 + GHB
(production config), plus one 400×400 isotropic no-GHB physics-coverage case.
Result JSONs: `working_tests/mixed_precision_results_{fp64,fp32,mixed}_ghb_*`.

**Scope of validity:** steady confined 2D only, on the grids above, on the
benchmark GPU. Nothing here validates transient, unconfined, 3D, or
other-hardware behaviour.

### 3.1 Heterogeneous T + GHB (CUDA-warm solves, all from DEM)

| Case | FP64 warm (cycles) | FP32 warm (cycles, converged) | Mixed warm (outer×inner) |
|---|---|---|---|
| 100x100 | 0.025 s (55) | 0.091 s (200, **no**) | 0.070 s (11×5=55) |
| 100x1000 | 0.093 s (60) | 0.296 s (200, **no**) | 0.160 s (12×5=60) |
| 400x400 | 0.155 s (55) | 0.537 s (200, **no**) | 0.299 s (11×5=55) |
| 1000x1001 | 0.644 s (60) | 2.068 s (200, **no**) | 0.710 s (12×5=60) |
| 2000x1000 | 0.997 s (55) | 3.501 s (200, **no**) | 1.088 s (11×5=55) |

Accuracy (max |Δh|):

| Case | Mixed vs MF6 | Mixed vs FP64 Warp | FP32 vs MF6 | FP32 vs FP64 Warp | Gate 2e-4 (mixed / fp32) |
|---|---|---|---|---|---|
| 100x100 | 3.16e-05 | 1.9e-08 | 6.1e-05 | 6.8e-05 | ✅ / ✅ |
| 100x1000 | 3.77e-05 | 2.5e-08 | 6.1e-05 | 4.9e-05 | ✅ / ✅ |
| 400x400 | 2.91e-05 | 6.3e-09 | 1.2e-04 | 1.1e-04 | ✅ / ✅ |
| 1000x1001 | 2.32e-05 | 2.9e-09 | 1.5e-04 | 1.5e-04 | ✅ / ✅ |
| 2000x1000 | 3.78e-05 | 1.4e-09 | **2.4e-04** | 2.3e-04 | ✅ / **❌** |
| 400x400 iso, no-GHB | 3.71e-05 | 5.8e-10 | 1.7e-04 | 1.5e-04 | ✅ / ✅ |

On the tested cases the mixed solver matches FP64 Warp to ≤ 2.5e-08 m
everywhere and agrees with FP64 against MF6 to the shown precision
(same max diff to 3 significant digits). Mass balance is equivalent to FP64
(percent discrepancy 6.2e-6 % vs 6.7e-6 % at 2000x1000; FP32 degrades ~400×
to 2.4e-3 %).

### 3.2 The decisive measurement: per-K-cycle cost

| Case | FP64 ms/cycle | FP32 ms/cycle | ratio |
|---|---|---|---|
| 100x100 | 0.45 | 0.46 | 1.00 |
| 100x1000 | 1.56 | 1.48 | 0.95 |
| 400x400 | 2.82 | 2.69 | 0.95 |
| 1000x1001 | 10.74 | 10.34 | 0.96 |
| 2000x1000 | 18.12 | 17.51 | 0.97 |

In the **current K-cycle implementation**, FP32 storage makes a cycle only
~3–5 % cheaper. Two reasons, both properties of the current implementation
rather than of defect correction:

1. The multilevel cycle is dominated by kernel-launch and synchronization
   overhead across the six-level hierarchy, not by array bandwidth.
2. Important row calculations already accumulate in FP64 inside the kernels
   (§1.3), so FP32 hierarchy storage currently removes neither arithmetic
   cost nor the dominant fraction of memory traffic.

### 3.3 Ordinary-FP32 accuracy floor (confirms audit §1.6 prediction)

Ordinary FP32 **never satisfies** `rel_tol=5e-7` on any tested case — the
residual norm stagnates at the FP32 storage floor, so it burns all
`max_cycles=200` and returns `converged=False`. Measured head-error floor:
~5e-5–2.3e-4 m vs FP64, worsening as the grid grows; on the largest case
(2000x1000) it drifts past the unchanged 2e-4 m MF6 gate. On these cases
ordinary FP32 is simultaneously ~3.5× slower (wasted cycles) and less
accurate than FP64.

### 3.4 Memory (measured mempool high-water delta per solve)

| Case | FP64 | FP32 | Mixed |
|---|---|---|---|
| 100x100 | 3 MiB | 1 MiB | 2 MiB |
| 1000x1001 | 200 MiB | 100 MiB | 200 MiB |
| 2000x1000 | 300 MiB | 100 MiB | 200 MiB |

What the measurements support: FP32 storage roughly halves the solver-side
allocation, and the mixed configuration lands between FP32 and FP64 because it
carries the FP32 hierarchy *plus* the FP64 master head/RHS/residual arrays.
At ~2M cells the mixed overhead over plain FP32 is ~100 MiB, and mixed saves
~100 MiB versus FP64. Whether that translates into meaningfully better model
concurrency or peak-memory capacity on a given GPU depends on workload and
was not measured; no concurrency benefit is claimed here.

### 3.5 Mixed mode verdict on the acceptance criteria (tested cases only)

- ✅ Every FP64-passing MF6 case still passes the unchanged 2e-4 m gate.
- ✅ No tested grid/GHB/heterogeneous case loses convergence (converges in
  11–12 outer iterations = 55–60 FP32 K-cycles, everywhere).
- ✅ Mass balance within accepted limits (equivalent to FP64).
- ❌ **Runtime does not improve** — in the current implementation mixed is
  0.9×–2.8× slower than plain FP64 (near-identical per-cycle cost × similar
  total cycle count + FP64 residual/accumulate overhead + 2 scalar syncs per
  outer iteration).
- ✅ Method is general within its scope (no DEM/geometry/MF6 tailoring; both
  cold and warm solves independently start from the DEM; cold-vs-warm heads
  identical).
- The FP64-fine-transmissivity variant was **not needed on the tested cases**:
  with FP32 T the mixed heads already matched FP64 to ≤ 2.5e-08 m, i.e.
  coefficient quantisation was not the limiting error source here. The variant
  was removed from the retained implementation.

**Correctness vs value:** as an algorithm, mixed defect correction is
numerically successful on every tested steady confined case. As a performance
optimisation it delivers nothing today, and the reason is the current cycle
execution profile (launch/sync-dominated, FP64-internal kernels) — **not** a
failure of iterative refinement or defect correction.

## 4. Recommendation

Mixed precision is numerically successful but provides no performance
advantage in the current DarcyWarp K-cycle implementation. FP32 hierarchy
cycles cost approximately the same as FP64 cycles because execution is
dominated by multilevel kernel-launch and synchronization overhead, while
important row operations already accumulate in FP64. Mixed defect correction
therefore adds overhead without materially reducing cycle cost. Ordinary FP32
stagnates above the production convergence and MF6 accuracy requirements. The
implementation should remain experimental, opt-in, and non-default. Future
FP32-class optimisation should target cycle execution before revisiting state
precision.

Disposition:

- Keep `DARCY_WARP_PACKAGE/solvers/mixed_precision.py` and the benchmark
  harness as **experimental** (opt-in only; no registry/default changes).
- Ordinary `DARCY_FLOAT=float32` failed the production accuracy requirements
  on the tested large case and never met `rel_tol=5e-7` on any tested case;
  do not use it for production steady confined work.
- Revisit mixed precision **only after** cycle-execution work reduces the FP32
  K-cycle cost materially below the FP64 cycle cost. Future work should first
  target:
  - kernel fusion (fewer kernels per multilevel cycle);
  - CUDA graph capture of repeated cycle sequences;
  - reduced scalar synchronization per cycle/iteration;
  - persistent or device-controlled cycle execution;
  - removal of avoidable FP64 operations only where numerical evidence
    supports it.
