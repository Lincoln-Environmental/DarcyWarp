# Mixed-Precision Multigrid Optimisation Campaign — Experimental Report

> Current status (2026-08-28): the campaign itself is a historical
> experimental report. Its validated `solvers/mixed_fast.py` implementation is
> now the production mixed-precision choice used by the confined steady runner.
> The rejected `mixed_vcycle.py` path and the original slower
> `mixed_precision.py` reference remain non-production.

> Companion to `MIXED_PRECISION_PLAN.md` (which covers the initial audit, the
> validated-but-not-faster defect-correction solver, and the bounded negative
> recommendation).  This document tracks the staged optimisation campaign:
> can a deliberately cheaper low-precision correction cycle, inside an
> authoritative FP64 outer loop, beat the production FP64 K-cycle?
> All work is experimental, opt-in, non-production.  Started 2026-07-30.

## Phase 1 — authoritative baseline

### 1.1 Methodology

- Harness: `working_tests/mixed_precision_profile_baseline.py` (one mode per
  process; `DARCY_FLOAT` pinned; every timed solve starts from the original
  DEM; warm median of 5 reps reported with min/max).
- Launch instrumentation: `wp.launch` wrapped with CUDA-event pairs (eager
  mode: cached graph invalidated + capture disabled) → exact launch counts,
  host sync/readback counts.  Per-launch event windows in the eager
  host-bound regime measure submission cadence, not kernel GPU time, so GPU
  kernel times come from `nsys` traces instead
  (`working_tests/_nsys_solve_driver.py`, `nsys stats -r gpukernsum`).
- Accuracy gates: cached FP64 Warp reference heads + cached MF6 truth
  (unchanged 2e-4 m gate).  Baseline reproduced the confirmed benchmark
  values (fp64 0.997 s, fp32 3.50 s non-converged, mixed 1.09 s at 2000x1000)
  within a few percent.

### 1.2 Baseline facts (2000x1000, heterogeneous ugly_t + GHB, warm CUDA)

| Mode | Warm median | Cycles | Launches/cycle | Host scalar readbacks/solve | Mem high-water Δ |
|---|---|---|---|---|---|
| FP64 production K-cycle | 0.995 s | 55 | **1326** | ~49 | 274 MiB |
| FP32 ordinary K-cycle | 0.363 s (20 cyc capped) | 200 (never converges) | **1326** | ~21 | 146 MiB |
| Mixed (FP32 K-cycle inner) | 1.056 s | 11 outer × 5 | **1326** | ~36 | 196 MiB |

- Production FP64 K-cycle is **already CUDA-graph captured** (whole cycle,
  replayed per cycle; `cuda_graph_reused=True`).  Its 18.1 ms/cycle is
  therefore GPU-bound, not launch-bound.
- The existing mixed inner correction (`solve_kcycle_device_buffers`,
  fixed-work mode) runs **uncaptured**, but at 2M cells it is also GPU-bound
  (GPU busy ≈ 1.03 s of the 1.05 s wall) because its kernels are long.
- nsys 2022.4 does not trace kernels inside CUDA graphs; the full per-kernel
  breakdown below comes from the uncaptured mixed trace (FP32 hierarchy,
  same kernel structure as FP64).

### 1.3 Per-kernel GPU breakdown (mixed trace, 2000x1000, per-solve shares)

| Kernel group | Per-instance (level 0 / level 1) | Share of solve |
|---|---|---|
| `compute_residual` (all levels, per-thread FP64 atomic rTr) | **2.78 ms / 0.72 ms** | **~39 %** |
| `jacobi_applyA_fused` smoother | 0.47 ms / 0.118 ms | ~24 % |
| K-combination (`dot_active`, `apply_A_and_pAp` off coarsest) | 0.70 ms @ L1 (atomic-bound) | ~14 % |
| Coarsest-level PCG (125-cell grid; ~10 tiny kernels/iter × 2^5 fan-in) | 6–10 µs each, ×10⁴ instances | ~10 % |
| `zero_scalar` | 1.5 µs × 136 k | ~3 % |
| Mixed outer: `_mp_residual_f64` 2.79 ms + `_mp_accumulate` 3.45 ms per outer | × 11–12 outer | ~7 % |

### 1.4 The three dominant costs (all structural, none precision-related)

1. **Per-thread FP64 atomic reductions serialize.**  Residual/dot kernels
   run 6–100× below bandwidth: level-0 residual 2.78 ms vs the 0.47 ms
   smoother with identical traffic; `dot_active` at level 1 costs 0.70 ms for
   ~4 MB of reads.  Every reduction (`rTr`, `rho`, `pAp`, `dh_sq`, `dh_max`,
   including the FP64 outer residual and accumulate) pays per-thread atomics
   into a single address.  ≈ 50 % of cycle GPU time.
2. **K-cycle double descent.**  Each level performs two recursive descents
   plus the Krylov combination → 1326 launches/cycle with 2^l growth into the
   coarse levels; roughly doubles all residual/smoother/transfer work versus
   a V-cycle.
3. **Small-kernel floor.**  ~24 % of GPU time is kernels of ≤ 17 µs
   (coarsest PCG, zero_scalar, scalar combine kernels) — kernel-count-bound,
   precision-irrelevant.

FP32-vs-FP64 bandwidth differences are confined to the smoother (~24 % of
cycle time), which is why the measured FP32 K-cycle saving is only 3–5 %.

### 1.5 Implications for the campaign hypotheses

- **H1 (V vs K): strongly supported by structure.**  A V-cycle eliminates the
  second descent and the K-combination — mechanically ~50 % less GPU work and
  ~17× fewer launches (~75 vs 1326).  The open question is contraction per
  cycle; the metric is FP64-residual reduction per millisecond.
- **H2 (true FP32 arithmetic): low expected value as measured.**  Only the
  smoother is bandwidth-sensitive; expected cycle-level saving a few percent.
  Still implemented in Phase 3 for auditability, but not expected to be the
  deciding lever.
- **H3 (launch/sync): production is already graphed.**  The experimental
  correction path must also be graph-captured to compete; scalar readbacks
  (~50/solve) are negligible next to kernel time.
- **New lever (larger than any listed hypothesis): block-reduced
  reductions.**  Warp/block-level partial sums with one atomic per block
  should cut reduction kernels from ~7.3 ms to ~1 ms per K-cycle-equivalent.
  This applies to the experimental path only; production is untouched.
- **Coarsest-level PCG is wildly over-provisioned** (125-cell grid, ~10 %
  of solve).  A cheap fixed Jacobi block at the coarsest level removes most
  of it.

### 1.6 Setup costs

- Hierarchy build (host, once): 0.21–0.23 s at 2M cells (both precisions).
- Mixed session init: ~0.2 ms (buffers only; hierarchy reused).
- Cold vs warm: cold ≈ warm + ~0.03 s once the warp kernel cache is hot.

## Phase 2 — fixed V-cycle correction (Hypothesis 1)

**Verdict: REJECTED (plain V and energy-rescaled V). The K-cycle's per-level
Krylov machinery is load-bearing on this hierarchy.**

### 2.1 What was measured

New experimental code: `DARCY_WARP_PACKAGE/solvers/mixed_vcycle.py`
(`solve_vcycle_device_buffers` — single-descent fixed V-cycle, Jacobi-block
coarsest, no scalar reads — and `MixedPrecisionVcycleSession`, reusing the
validated FP64 outer loop).  Benchmark: `working_tests/mixed_vcycle_benchmark.py`.

Cycle cost (uncaptured, 100x100, FP32): **V = 0.239 ms vs K = 1.164 ms** — the
V-cycle is 4.9× cheaper per cycle, as predicted by structure.

Correction quality on the real defect (100x100, FP64 outer residual):

| Variant | Total warm | Converged | Outer | Contraction per outer |
|---|---|---|---|---|
| mixed k=5 (existing) | 68.3 ms | ✅ | 11 | 0.25 → 0.26 uniform |
| mixed k=2 | 68.9 ms | ✅ | 27 | 0.50 → 0.58 uniform |
| mixed k=1 | 76.0 ms | ✅ | 53 | 0.58 → 0.76 uniform |
| mixed v=1 (V-cycle) | 43.2 ms | ❌ | 60 (cap) | **0.40 → 0.97 stalls** |
| mixed v=2 | 76.9 ms | ❌ | 60 (cap) | 0.27 → 0.94 stalls |

The V-cycle is an excellent corrector while the defect is rough (0.40 first
outer, better than one K-cycle's 0.58), but contraction degrades monotonically
as the FP64 outer loop strips high-frequency content, and it stalls outright
once the defect is smooth (~0.95+ ⇒ never converges; MF6 gate fails).

### 2.2 Why it stalls (diagnostics, 100x100)

- Single-V contraction on the real first-outer defect: **0.40** (K: 0.58).
- Single-V contraction on a smooth constant defect: **3.57× DIVERGENCE** —
  the coarse levels amplify smooth modes.  Root cause: the coarse operators
  are approximate (block-averaged T, non-Galerkin), so the coarse-correction
  *direction* is unreliable for low-frequency error.  This is a hierarchy
  property, not an iteration-count or arithmetic-precision issue (⇒ H4's
  FP64 coarse levels would not fix it).
- Top-level energy line search α = (r·z)/(z·Az): A-norm optimal but the
  Euclidean residual blew up 32× — scaling cannot fix a bad direction.
- Per-level energy rescale of the coarse correction (Krylov safeguard without
  the second descent, `per_level_krylov=True`): smooth-defect divergence
  worsened to 18×.  Rejected.
- The K-cycle holds ~0.76/cycle uniformly at every stage because its second
  descent + per-level 2-term Krylov (z1 + α z2 with α from r1, z2 = B(r1))
  controls the coarse correction at every level.  That structure is not
  removable.

### 2.3 Consequence for the campaign

The correction cycle keeps the K structure.  The cost model (2M cells, outer
FP64 overhead 6.5 ms atomic-bound; inner K 17.5 ms) shows no inner-cycle
count k beats FP64 with current kernels (min ≈ 1.0 s ≈ parity):

| k cycles/outer | outer iters | modelled total |
|---|---|---|
| 1 | ~53 | 1.27 s |
| 2 | ~27 | 1.12 s |
| 5 | ~11 | 1.03 s (measured 1.056 s) |

⇒ The cycle itself must get cheaper.  Phase 1 profiling says the cost is
kernel-level (atomic-serialized reductions ≈ 39 %, smoother ≈ 24 %,
coarsest PCG ≈ 10 %), not structural — that is Phase 3 territory.

## Phase 3 — true-FP32 kernels + cheap reductions

**Verdict: kernels validated and fast per se; the cycle became
host-launch-bound, which Phase 4 resolves.**

New experimental code (production untouched):

- `DARCY_WARP_PACKAGE/solvers/mixed_fast_kernels.py` — face-conductance build
  (one division per face *once*, replacing four FP64 divisions per cell per
  call), face-array Jacobi smoother and residual in **explicitly separate,
  auditable FP32 and FP64 variants**, two-stage (per-block partial +
  single-block combine) reductions, block-reduced FP64 outer residual and
  correction-accumulate kernels.
- `DARCY_WARP_PACKAGE/solvers/mixed_fast.py` — `FastLevel` face arrays,
  `solve_kcycle_fast_device_buffers` (identical two-descent + per-level
  Krylov structure to production; fixed-work, no scalar reads), and
  `MixedPrecisionFastSession` (validated FP64 outer loop; only the inner
  correction and outer-kernel implementations differ).

Kernel validation (48x40 heterogeneous + GHB + inactive cell, FP64 variants
vs production kernels / numpy): residual **bit-identical**, smoother ≤ 5e-15,
dots ≤ 2e-15.  Correction quality identical to production K (3-cycle
contraction 0.417 vs 0.409).

Per-kernel measurements (nsys, 2000x1000, FP32):

| Kernel | Production | Fast | Speedup |
|---|---|---|---|
| Jacobi smoother @ L0 | 472 µs | **134 µs** | 3.5× |
| Residual @ L0 | 2782 µs | **132 µs** | 21× |
| dot/applyA_dot @ L1 | 703/706 µs | **63/58 µs** | 11× |

Root cause of the production costs, confirmed: FP64 row arithmetic (incl.
4 FP64 divisions/cell/call) is compute-bound on consumer GPUs (FP64 ≈ 1/64
rate), and per-thread FP64 atomics serialize every reduction.

**But**: the eager fast cycle measured 33–35 ms at 2M — ~2× slower than
production — because the fast kernels are so short that the ~1400-launch
cycle is host-submission-bound (~22 µs/launch), with ~70 % of launches at
the coarsest level (the 2^l double-descent tree visits the coarsest 32× per
cycle × nu_coarse=30 sweeps).

## Phase 4 — launch/sync reduction (CUDA graphs)

**Verdict: decisive. The fixed correction block is capture-stable; graph
replay removes the host bottleneck entirely.**

- The whole correction block (cast + zero + N fast K-cycles) is captured
  once per session and replayed per outer iteration
  (`MixedPrecisionFastSession._inner_correction_block`; capture-once,
  replay-immediately so the capturing outer still does its work; eager
  fallback when capture is disabled).  Graph-captured and eager solves are
  regression-tested for equivalence (same outer count, heads equal to 1e-9).
- Fairness: production FP64 K-cycle was *already* graph-replayed, so the
  baseline was never inflated by launch overhead.

Cycle-level gate (graph-replayed, like-for-like structure, identical kernels
modulo precision, nu_coarse=10):

| Grid | Fast FP32 | Fast FP64 | FP32 saving | Gate ≥ 20 % |
|---|---|---|---|---|
| 2000x1000 | **3.37 ms** | 5.17 ms | **34.8 %** | ✅ |
| 1000x1001 | **2.25 ms** | 3.61 ms | **37.5 %** | ✅ |

(nu_coarse=30 shrinks the gap to ~6 % because the precision-independent
coarsest-launch floor dominates; nu_coarse=10 keeps contraction at
production parity 0.26 and is the adopted setting.)

## End-to-end results (warm CUDA, median of 5, all solves from the DEM)

| Variant | 100x100 | 400x400 iso no-GHB | 1000x1001 | 2000x1000 |
|---|---|---|---|---|
| FP64 production K-cycle | 24.4 ms | 165 ms | 644 ms | 997 ms |
| FP32 ordinary | 8.7 ms (20 cyc, no conv) | 544 ms (no conv) | 2068 ms (no conv) | 3501 ms (no conv, gate ❌) |
| Mixed K-cycle inner (Phase 0) | 68 ms | 306 ms | 710 ms | 1056 ms |
| Mixed V-cycle inner (Phase 2) | ❌ stalls | — | — | — |
| **Mixed fast k=5 nuc=30** | 21.7 ms | — | 237 ms | 293 ms |
| **Mixed fast k=5 nuc=10** | — | **54.4 ms** | **157 ms** | **227 ms** |
| Mixed fast k=2 nuc=30 | — | — | 254 ms | 338 ms |

**Mixed fast k=5 nuc=10 speedup vs production FP64: 3.0× (iso), 4.1×
(medium), 4.4× (large).**

Numerical acceptance (mixed fast k=5 nuc=10):

| Grid | Converged (outer) | Contraction | Max diff vs FP64 | Max diff vs MF6 | Gate 2e-4 | Mass-balance % |
|---|---|---|---|---|---|---|
| 100x100 | ✅ 11 | 0.27 | 6.0e-06 | 2.7e-05 | ✅ | -1.1e-04 |
| 400x400 iso | ✅ 11 | 0.27 | 1.5e-06 | 3.7e-05 | ✅ | -8.3e-05 |
| 1000x1001 | ✅ 12 | 0.28 | 1.7e-06 | 2.4e-05 | ✅ | -1.5e-05 |
| 2000x1000 | ✅ 11 | 0.26 | 4.5e-07 | 3.8e-05 | ✅ | 1.1e-05 |

Correction contraction is uniform and at production-K parity (0.26–0.28)
on every case.  vs-FP64 agreement degrades from 2.5e-08 m (Phase-0 mixed) to
≤ 6e-06 m because the correction arithmetic is now genuinely FP32 — still
30× below the unchanged 2e-4 m gate.  No fallbacks or escalations occurred
(anywhere in the campaign).

Memory (measured mempool high-water deltas, 2000x1000): FP64 274 MiB,
ordinary FP32 146 MiB, Phase-0 mixed 196 MiB, mixed-fast session ~15 MiB
delta on top of the FP32 hierarchy it shares (face arrays ≈ 53 MiB are
included in that high-water measurement).  No concurrency claim is made:
the campaign did not measure simultaneous-model capacity.

## Phase 5 — scaling (Hypothesis 5): SKIPPED with rationale

Scaling targets the ordinary-FP32 stagnation floor and poor row scaling.
Inside FP64 defect correction there is no stagnation floor to fix (the FP64
outer owns convergence), and the measured correction contraction is uniform
at production-K parity on heterogeneous and isotropic cases — no
scaling-related deficiency exists.  Implementing scaling would add setup,
storage, and conversion cost against zero measured deficit.  Rejected
without implementation; revisitable if a future case shows
contraction degradation.

## Phase 6 — adaptive escalation (Hypothesis 6): SKIPPED with rationale

The escalation policy exists to recover robustness when a cheap correction
is intermittently weak.  The fast K-cycle correction is uniformly strong
(0.26–0.28 on every outer of every case, including the smooth late-stage
defects that killed the V-cycle), and the fixed k=5/nuc=10 configuration
already beats k=1/k=2 end-to-end — no escalation trigger would ever fire.
A controller would add complexity and sync overhead for no measured
benefit.  Rejected without implementation.

## Hypothesis 4 — FP64 coarse levels: REJECTED without implementation

Phase 2 isolated the smooth-mode stall as a coarse-operator *direction*
error (approximate block-averaged coarsening), which arithmetic precision
does not fix; energy rescaling made it worse.  The retained fast K-cycle
holds production-parity contraction with FP32 coarse levels, so the
expected benefit is nil and the cost (parallel FP64 coarse arrays, boundary
casts) is real.

## Final decision: **ADOPT EXPERIMENTALLY** (non-default, opt-in)

The complete mixed solve beats production FP64 by **3.0–4.4×** on the
medium, large, and isotropic cases, with every unchanged numerical gate
passing (convergence, 2e-4 m MF6, mass balance ≈ FP64, no fallback).
The cycle-level gate also passes (34.8–37.5 % FP32-vs-FP64 like-for-like).

### What actually won (honest attribution)

1. **Kernel engineering (~2/3 of the win)**: face-conductance precompute,
   no per-thread-atomic reductions, Jacobi-block coarsest, CUDA-graph
   capture of the fixed correction block.  These are precision-agnostic —
   the same changes applied to the production FP64 K-cycle would speed it
   up comparably (fast-FP64 cycle: 5.17 ms vs fast-FP32 3.37 ms).  Adopting
   them in production is a separate, out-of-scope decision.
2. **The mixed-precision structure (what makes FP32 safe)**: ordinary FP32
   with these same fast kernels would still stagnate at its storage floor
   and fail the 2e-4 m gate.  The authoritative FP64 outer (true residual,
   correction accumulation, convergence) is what allows the FP32 correction
   to be approximate.  The remaining FP32-vs-FP64 cycle-level advantage is
   ~35 %, which compounds with the engineering to the end-to-end 3–4.4×.

### Limits of validity

- Steady confined 2D only; tested grids (100² … 2000x1000), ugly_t seed 123
  + GHB, and one isotropic no-GHB case; one consumer GPU (RTX 4070 Ti
  SUPER, FP64:FP32 ≈ 1:64 — datacenter GPUs with strong FP64 would show a
  smaller precision effect).
- Not validated for transient, unconfined, 3D, or other DEMs/T fields.
- Session is not thread-safe; graph capture assumes a fixed hierarchy for
  the session lifetime.

### Recommended next actions

1. Keep `mixed_fast` experimental; gain broader-case confidence (other T
   fields/seeds, more geometries) before any production discussion.
2. Separately evaluate adopting the precision-agnostic kernel improvements
   (face arrays, block-reduced reductions, graphed fixed-work cycles) into
   the production FP64 K-cycle — likely a similar speedup without any
   mixed-precision machinery.
3. If (2) happens, re-baseline this campaign: the mixed advantage over an
   equivalently-optimised FP64 solver would then be the ~35 % cycle-level
   precision effect alone.

## Addendum (2026-07-30): production adoption + callable experimental settings

Following the campaign, both recommendations were acted on:

### Experimental settings are now callable

- `DARCY_WARP_PACKAGE/solvers/mixed_fast.py` exposes `MixedFastConfig`
  (defaults = the validated k=5/nuc=10 configuration),
  `get_mixed_fast_session(model, ...)` (session cached on the model, reusing
  buffers/faces/graph across solves), and `solve_mixed_fast(model,
  initial_head, ...)`.  Still experimental, opt-in, unregistered.
- Benchmarks: `working_tests/mixed_vcycle_benchmark.py --mode mixed-fast`
  drives the config object (plus `mixed-kcycle`/`mixed-vcycle`/`vcost`
  modes); `working_tests/mixed_precision_profile_baseline.py` profiles all
  Phase-0 modes.
- Tests: `tests/test_mixed_precision_fast.py` (7 tests incl. config-default
  pinning).

### Kernel improvements adopted into straight FP64 (opt-in `implementation="fast"`)

- New production modules: `solvers/face_kernels_f64.py` (FP64 face-array
  kernels + block-reduced check kernel; the experimental
  `mixed_fast_kernels.py` re-exports these as its FP64 single source of
  truth) and `solvers/fast_confined_kcycle.py` (full production convergence
  semantics: initial-residual-relative tolerance, dh safeguards, check
  cadence, PCG divergence fallback, info parity + `implementation: "fast"`,
  graph-captured cycles, Jacobi-block coarsest).
- Opt-in only: `solver.solve_multigrid_kcycle(..., implementation="fast")`
  or `solver.solve(solver="confined_kcycle", implementation="fast")`.
  Default remains `"classic"` everywhere.  Steady confined FP64 only —
  transient/unconfined/FP32 raise explicit errors and keep the classic path.
- Derived-state invalidation: `update_T_in_place*` and
  `update_ghb_factor_in_place` set `model._fast_faces_stale`; the face cache
  refreshes **in place** (arrays stay put, so captured graphs remain valid).
- Benchmark wiring: `DARCY_KCYCLE_IMPL=fast` env switch in
  `model_convergence_and_sanity_tests.py` (default `classic`; separate
  results JSON `..._fast.json`).

Official-benchmark results (`model_convergence_and_sanity_tests.py`, warm
CUDA, MF6 from cache, ugly_t + GHB):

| Case | Classic FP64 | Fast FP64 | Speedup | Cycles (c/f) | vs MF6 (classic) | vs MF6 (fast) | Gate 2e-4 |
|---|---|---|---|---|---|---|---|
| 100x100 | 0.0254 s | 0.0167 s | 1.52× | 55/55 | 3.16e-05 | 2.86e-05 | ✅ |
| 100x1000 | 0.0922 s | 0.0448 s | 2.06× | 60/60 | 3.77e-05 | 7.41e-05 | ✅ |
| 400x400 | 0.1547 s | 0.0736 s | 2.10× | 55/55 | 2.91e-05 | 2.95e-05 | ✅ |
| 1000x1001 | 0.6299 s | 0.2363 s | 2.67× | 60/60 | 2.32e-05 | 2.35e-05 | ✅ |
| 2000x1000 | 1.0146 s | 0.3219 s | 3.15× | 55/55 | 3.78e-05 | 3.80e-05 | ✅ |

Same cycle counts and convergence criteria everywhere; fast-vs-classic head
agreement ≤ 1.5e-06 m; all MF6 gates pass.

### Re-baselined mixed-precision advantage

With production FP64 now equivalently optimised, the residual mixed-precision
advantage (fast FP32 correction vs fast FP64 correction, both graphed) is:

| Case | Fast FP64 | Mixed fast | Mixed advantage |
|---|---|---|---|
| 1000x1001 | 0.236 s | 0.157 s | 1.50× |
| 2000x1000 | 0.322 s | 0.227 s | 1.42× |

Consistent with the ~35 % cycle-level precision effect measured in Phase 4.
This was the campaign verdict on 2026-07-30. It was superseded on 2026-08-28:
`mixed_fast` is now the production mixed-precision choice; the original
defect-correction reference and rejected V-cycle remain non-production.
