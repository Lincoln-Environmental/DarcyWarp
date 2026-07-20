# Plan: Switchable GPU Backend (Warp ⇄ Taichi)

Status: **Proposal / scoping doc** — not yet implemented.
Author intent: the solver should run on more than one GPU platform by making the
backend kernels switchable between NVIDIA **Warp** (current) and **Taichi Lang**.

---

## 1. Context & goal

All GPU work today is hard-wired to NVIDIA Warp: `@wp.kernel` definitions, raw
`wp.launch(...)` calls inline in the solver orchestration, `wp.array/zeros/empty`,
and `wp.atomic_add` into length-1 reduction buffers. There is **no abstraction
layer** — `wp.*` is used directly throughout `warped_darcy.py` and `solvers_3d.py`.

Goal: introduce a backend abstraction so the same solver orchestration can target
either Warp or Taichi at runtime (constructor arg or env var), with numerical
results validated equivalent.

### Quantified Warp surface (measured)
- **44 kernels** — 15 in `kernels_3d.py`, 29 in `warped_darcy.py`.
- **80 `wp.launch` sites** — 23 in `solvers_3d.py`, 57 in `warped_darcy.py`.
- **~446 array-creation calls** — `wp.array` (338), `wp.zeros` (102), `wp.empty` (6); all raw, no wrapper.
- **17 atomics** (15 `atomic_add`, 2 `atomic_max`); ~7 distinct scalar-reduction-buffer patterns (`rho_buf`, `rTr_buf`, `pAp_buf`, `beta_buf`, `dh2_buf`, `dh_max_buf`, …).
- Threading: always per-dimension tuple unpacking (`k,j,i = wp.tid()`); no tile/warp/cooperative primitives.
- `wp.float64` hardcoded for all reduction buffers; field dtype via `WP_FLOAT` (`DARCY_FLOAT` env, float32/float64).
- `wp.init()` is never called explicitly (default init); device is a string (`cuda:0` / `cpu`).

---

## 2. Difficulty assessment

**Rating: medium-hard.** The line count is manageable because patterns are uniform,
but the **parallelism-model mismatch** dominates effort — this is not a thin shim.

### Core idiom gaps
1. **Launch model (the big one).** Warp = *external* grid: `wp.launch(kernel, dim, inputs)`,
   kernel reads `wp.tid()`. Taichi = *internal* loops: a `@ti.kernel` owns its own
   `for` loops and is called like a function; there is no external `dim`/`inputs`.
   ⇒ Every kernel body must be **restructured to own its loop** (e.g. `for k,j,i in ti.ndrange(...)`).
   No 1:1 `launch()` adapter is possible.
2. **Arrays.** `wp.array(data, dtype, ndim, device)` + `.numpy()` vs `ti.ndarray(dtype, shape)` +
   typed arg decls (`ti.types.ndarray(element_dim=3)`) + `.to_numpy()`. Mappable, but the typed
   ndarray declarations in kernel signatures differ.
3. **float64 on Taichi/CUDA.** Historically limited/quirky. The code uses `wp.float64` for *every*
   reduction and optionally runs full float64 solves (`DARCY_FLOAT=float64`). **Must be verified
   on Taichi before committing** — this is the single highest technical risk.
4. **Atomic-sum ordering.** Reductions accumulate in indeterminate order; float sums will differ
   by ULPs across backends, nudging iterative convergence. Needs an equivalence tolerance.
5. **Recurrence kernels.** `vertical_line_relaxation_7point_kernel` does a tridiagonal
   (Thomas) sweep down a column — a sequential recurrence inside a parallel launch. Porting to
   Taichi's structured-for is doable but needs care (1D launch over `(ny*nx)`, internal `k` loop).
6. **Host sync inside iteration loops.** PCG reads `buf.numpy()[0]` each iteration for convergence.
   Portable (Taichi `.to_numpy()` / scalar field read), but keep an eye on per-iteration sync cost.

### Favorable factors
- Kernels are **pure physics** — no solver state embedded (verified).
- One uniform threading idiom, one uniform reduction idiom.
- A strong numerical test suite already exists as equivalence guards
  (`test_3d_solver_matches_scipy_reference`, `test_2d_unconfined`, the PCG comparison tests).

### ⚠️ Strategic caveat — read first
**Taichi is in maintenance/bugfix mode only as of 2026** (active dev halted after Taichi
Computing's 2024 layoffs; [GitHub #8506](https://github.com/taichi-dev/taichi/discussions/8506)).
It still works and ships on PyPI, but expect no major releases and risk around new CUDA/GPU support.
Taichi's value proposition *is* backend breadth (CUDA/CPU/ROCm/Metal/Vulkan) — exactly the part
most exposed to reduced maintenance. **Before investing, decide whether Taichi is the right vehicle**
(see §7 Alternatives).

### Effort estimate
~**3–6 weeks** for one engineer end-to-end (2–3 weeks if already fluent in both frameworks and
scoped 3D-first). Roughly: abstraction + Warp refactor ≈ 1 wk; 3D Taichi port + validation ≈ 1–2 wk;
2D port ≈ 1–2 wk; hardening/CI/docs ≈ 0.5–1 wk.

---

## 3. Recommended architecture

Introduce a **backend interface whose methods are solver operations**, not raw launches.
This abstracts kernels *and* the launch/array/reduction layer in one move.

```
                         ┌──────────────────────────┐
  Solver orchestration   │  Backend (Protocol/ABC)  │
  (warped_darcy.py,      │  .init / .array / .zeros │
   solvers_3d.py)        │  .to_host / .scalar      │
   calls named ops:      │  .apply_A_7point(...)    │
   backend.apply_A_7point│  .dot(...) / .axpy(...)  │
   backend.dot(...)      │  .residual / .relax ...  │
                         └───────┬──────────┬───────┘
                  ┌─────────────┴──┐   ┌─────┴────────────┐
                  │  WarpBackend   │   │  TaichiBackend   │
                  │  wp.launch(...)│   │  ti.kernel(...)  │
                  └────────────────┘   └──────────────────┘
```

**Why operation-level, not `launch(kernel,dim,inputs)`-level:** because Taichi kernels are
called directly (no external grid), a launch-level shim can't be 1:1. Exposing *operations*
(`apply_A_7point`, `dot`, `axpy`, `residual`, `restrict`, `prolong`, `relax_vertical_line`,
`zero_scalar`, …) lets each backend implement them idiomatically.

**Existing kernels to reuse (do not rewrite the Warp side):** the current `@wp.kernel`
functions in `kernels_3d.py` and `warped_darcy.py` become the `WarpBackend` implementation
bodies — `WarpBackend.apply_A_7point(...)` just calls the existing `wp.launch(apply_A_7point_kernel, ...)`.
So Phase 0 is mostly *moving* launch calls behind a method, not rewriting physics.

**Backend selection:** `DARCY_BACKEND=warp|taichi` env var (mirrors the existing `DARCY_FLOAT`
pattern in `config.py`) plus an optional `backend=` constructor arg. Device string
(`cuda:0`/`cpu`) maps to Taichi `arch` internally.

---

## 4. Phased implementation

Each phase ends **green on the existing test suite**; nothing lands broken.

### Phase 0 — Abstraction on Warp only (de-risk the seam)  *[~1 wk]*
1. Define `Backend` Protocol + `WarpBackend` in a new `DARCY_WARP_PACKAGE/backends/` package
   (`base.py`, `warp_backend.py`).
2. Add a `get_backend()` factory reading `DARCY_BACKEND` (default `warp`) in `config.py`.
3. Refactor `solvers_3d.py` (23 launch sites) and `warped_darcy.py` (57 launch sites) to call
   `self.backend.<op>(...)` instead of raw `wp.launch`. Move reduction buffers behind
   `backend.scalar()` / `backend.read_scalar()`.
4. **Validate:** `pytest tests/` must stay at baseline (26 passed; the one pre-existing PCG
   failure is unchanged). Zero Taichi code yet — this proves the abstraction boundary on Warp alone.

### Phase 1 — TaichiBackend for the 3D path only  *[~1–2 wk]*
5. **Kill the float64 risk first:** a 1-day spike — `ti.init(arch=ti.cuda, default_ip=ti.f64)`,
   run one 3D solve in float64, confirm heads match the scipy reference. If f64-on-Taichi is
   broken, stop and reconsider (see §7).
6. Implement `TaichiBackend` operations for the **15 3D kernels** (`kernels_3d.py`), each as a
   `@ti.kernel` with internal `ti.ndrange` loops + typed `ti.types.ndarray` args.
7. Port the 23 3D launch sites in `solvers_3d.py` (they already go through `backend.<op>` after Phase 0).
8. **Validate:** `test_3d_solver_matches_scipy_reference` + `test_vertical_line_one_sweep_matches_numpy_reference`
   pass on the Taichi backend; add a new direct **Warp-vs-Taichi** head-diff test (tolerance ~1e-6 abs,
   looser if float32).

### Phase 2 — TaichiBackend for the 2D path  *[~1–2 wk]*
9. Port the **29 2D kernels** (`warped_darcy.py`) to Taichi, including the Picard/unconfined path
   and GHB. This is the bulk of the work.
10. **Validate:** the `test_2d_unconfined` suite + the PCG/MF6 comparison tests pass on Taichi;
    add Warp-vs-Taichi head-diff tests per solver.

### Phase 3 — Hardening  *[~0.5–1 wk]*
11. Device-string → Taichi `arch` mapping; `ti.init()` lifecycle (once per process, cache dir).
12. Backend selection wired into `create_solver()` (`factory.py`) and CLIs (`bench_and_plot.py`).
13. CI matrix: run the warp-required tests under both backends (skip Taichi if not installed).
14. Docs: note the `DARCY_BACKEND` flag, the float64 caveat, and the equivalence tolerances.

---

## 5. Validation gates (per phase)

| Gate | Test | Meaning |
|---|---|---|
| Phase 0 | `pytest tests/` == baseline | Abstraction didn't change Warp behavior |
| Phase 1 | `test_3d_solver_matches_scipy_reference` on Taichi | 3D correctness vs ground truth |
| Phase 1 | new `test_3d_warp_vs_taichi_heads` | Backends agree (≤ tolerance) |
| Phase 2 | `test_2d_unconfined` suite on Taichi | 2D + unconfined + GHB correctness |
| Phase 2 | new `test_2d_warp_vs_taichi_heads` | Backends agree (≤ tolerance) |
| All | `working_tests/run_*_warp_vs_mf6.py` on GPU | End-to-end vs MODFLOW 6 |

Equivalence tolerance: start at `atol=1e-6, rtol=1e-5` for float64; expect to relax for float32
and for atomic-sum-ordering drift in reductions.

---

## 6. Key risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| **Taichi float64 on CUDA broken/quirky** | Medium-High | Phase-1 step-5 spike *before* any porting; gate the whole effort on it |
| **Taichi stagnation** (no new CUDA/GPU support) | High (2026 reality) | See §7; weight alternatives before committing |
| Atomic-order float drift → convergence differences | Medium | Document tolerance; consider deterministic reduction if divergence |
| `vertical_line_relaxation` recurrence port | Medium | Treat as 1D launch + internal k-loop; unit-test against numpy reference (already exists) |
| 80-site refactor introduces regressions | Medium | Phase 0 keeps Warp as source of truth; full test suite as gate |
| Per-iteration host sync slower on Taichi | Low-Medium | Benchmark; Taichi may need `ti.sync()` discipline |
| First-call compile cost differs (benchmark fairness) | Low | Warm up before timing in benchmarks |

---

## 7. Alternatives (decide before investing)

If "multi-GPU-platform portability" is the real goal, weigh these against a stagnant Taichi:

- **Triton** (OpenAI) — very actively maintained, CUDA + HIP + vendor backends; different model
  (block-based), strong for stencils. Probably the strongest 2026 choice for portability + longevity.
- **CuPy** — mature, CUDA-focused (drop-in numpy-on-GPU); easy port of the numpy-style helpers,
  but kernels would become raw CUDA/`cupyx` rather than a Python-embedded DSL.
- **Numba CUDA** — mature, Python-native `@cuda.jit`, closest *launch model* to Warp
  (`kernel[grid,block](args)` + per-thread indices) → lowest porting friction of the alternatives.
- **CPU/numpy/scipy fallback** — cheapest by far; you already have `CPU_FD.py`. If "runs without a
  GPU" matters more than "runs on a non-NVIDIA GPU," expand this instead.
- **JAX** — if autodiff/AD is ever in scope, JAX gives portability + AD, but a bigger rewrite.

**Recommendation:** if the goal is strictly "not locked to Warp," **Numba CUDA** has the lowest
porting friction (near-identical launch model). If the goal is genuine cross-vendor breadth with
long-term maintenance, **Triton** is the stronger 2026 bet than Taichi.

---

## 8. Decision needed before implementation
1. Is Taichi a hard requirement, or is "multi-platform portability" the real goal? (Drives §7.)
2. If proceeding with Taichi: authorize the **Phase-1 float64 spike first** as a go/no-go gate.
3. Scope: 3D-only first (recommended), or 2D+3D together?
