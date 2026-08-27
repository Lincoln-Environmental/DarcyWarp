# Unconfined Fast-Kernel Plan — Porting the Mixed-Precision Campaign Learnings

> Status (2026-07-30): **Phase A LANDED** (`solvers/face_transient_f64.py`,
> default on via `transient_face_operator_enabled`). Results: 1000x1000 30w
> hard-T 38.3 s vs 92.2 s classic same-session (2.4×), 30/30 strict both,
> RMSE 5.0e-05 both, MB excellent, identical outer counts (441); parity vs
> classic ~1e-12 m (100x100 3w, `working_tests/validate_face_transient_parity.py`).
> One deviation from §2: the coarsest level keeps the classic PCG sweep
> (Jacobi-block there shifted accepted heads ~6e-6 m, above the 1e-6 parity
> target). 500x500 remains launch-bound — Phase B territory.
>
> Status (2026-07-30, later): **Phase B LANDED** (CUDA-graph capture;
> `DARCY_TRANSIENT_FACE_GRAPHS` env / `transient_face_graphs_enabled`
> control, default on, eager fallback on capture failure; disabled while
> `profile_transient_fast_path` is on). Design per §3 with one scope choice:
> the inner capture is exactly ONE K-cycle replayed N times per fixed-work
> block (trivially keyed on buffer-wiring identity + structure; bit-identical
> to N eager cycles) instead of keying a whole-block graph on block_cycles.
> The per-outer refresh segment is one graph keyed on the by-value dt
> (adaptive-dt retries re-capture — rare). Equivalence vs eager face mode:
> identical outer counts, heads ~8e-13 m (100x100 3w,
> `working_tests/validate_face_transient_graphs.py`), 1 kcycle + 1 refresh
> graph reported. Results: 500x500 52w homogeneous **13.8 s graphs vs 21.5 s
> face-eager vs 20.3 s classic baseline** — the small-grid launch-bound floor
> is gone and the face path now beats classic at 500x500; 100x100 3w 0.41 s
> vs 0.72 s eager; 1000x1000 30w hard-T 33.6 s vs 38.3 s eager, all
> acceptance gates PASS everywhere.
>
> Status (2026-07-30, later still): **Phase D LANDED** (GHB on the device
> transient fast path, face-operator mode only — the classic device path
> keeps its coarse-refresh GHB limitation and the gate now says so). The
> face build already folded ``C_gh = T_c*ghb_factor`` into ``diag``; the
> work was the GHB-aware RHS kernel
> (``build_transient_rhs_ghb_f64_kernel``; ``C_gh*gh_head`` from the
> CURRENT outer T(h), bit-identical to the host path's
> ``build_rhs_fd_like`` + ``_prepare_5point_transient_terms`` assembly),
> routed through every RHS site incl. the captured per-outer refresh
> graph (no new graph keys — gh arrays are pointer-stable, gh_head
> fixed).  Validation (`working_tests/validate_face_transient_ghb.py`,
> 100x100 3w, CHD + interior GHB row): device-vs-host parity 5e-9 m with
> ss=0 (with ss=1e-5 a PRE-EXISTING host-path Ss-linearisation drift —
> endpoint ``ss*sat_ref`` in ``picard_unconfined.py`` vs the
> authoritative secant Ss potential — dominates at ~9e-6; kernel-level
> RHS and residual assembly were verified bit-identical separately);
> MF6 truth RMSE 1.9-3.0e-4 m with two-pass conductance matching (MF6
> conductance is fixed, Warp's ``C_gh`` scales with T(h)); strict Picard
> everywhere; cumulative mass budget 0.009 % (excellent).  No-GHB
> regressions: pytest 18/18, 100x100 3w and 500x500 10w replays
> unchanged (identical RMSE, runtime ballpark).
>
> Status (updated 2026-08-28): **Phase C is the production mixed-precision
> path** selected by `run_2d_transient_warp_replay.py`
> (`transient_mixed_precision_enabled=True`). It requires the face operator;
> `DARCY_TRANSIENT_MIXED=1` remains an environment-level selector for
> lower-level calls. The implementation is in
> `solvers/mixed_transient_f32.py`. Design: Option 1 implemented in
> correction form — per Picard outer the FP64 nonlinear residual
> `r = b - A*h^k` is cast to FP32 and fixed-work FP32 K-cycles solve
> `A32*delta32 = r32` (FP32 faces rebuilt every outer inside the captured
> refresh graph; Jacobi-block coarsest; block-reduced FP64 Krylov
> partials; the FP32 K-cycle itself captured once and replayed per block
> cycle), then the correction is cast back to FP64.  Correction form (not
> a plain FP32 head solve) because FP32 representation error then scales
> with `|delta|`, not `|h|` (~6e-8·80 m ≈ 5e-6 m would have floored the
> strict inner residual targets).  The FP64 outer loop, Picard update,
> and acceptance checks are untouched; mixed OFF is bit-identical.
> Results (`working_tests/validate_transient_mixed.py`): 100x100 3w
> identical outer counts, heads 5.8e-6 m vs FP64; 500x500 10w homo heads
> 7.3e-7; 500x500 52w homo **12.5 s vs 13.8 s**, strict 52/52, RMSE
> 3.3e-05; 1000x1000 30w hard-T **29.0 s vs 33.1 s (~12%)**, strict
> 30/30, no practical, 0 retries, outer counts within +1, heads ≤4.5e-5,
> RMSE 5.2e-05, MB excellent.  Memory: 0.608 vs 0.547 GiB mempool
> high-water at 1M (+11% — the FP32 session ADDS arrays on top of the
> retained FP64 state; the win is runtime, not memory).  All Phase C
> gates met (adopt as opt-in experimental; remains non-default).
> Source learnings: `MIXED_PRECISION_CAMPAIGN.md` (face-conductance precompute,
> block-reduced reductions, Jacobi-block coarsest, CUDA-graph capture,
> opt-in FP32 correction inside an authoritative FP64 outer loop).
> Target: the production 2D transient unconfined device fast path
> (`solvers/transient_unconfined.py`, backend `unconfined_picard_kcycle`).

## 1. Why the unconfined path should benefit more than confined did

Per Picard outer iteration, the production device fast path currently runs:

1. `update_unconfined_transmissivity_from_head_kernel` — T(h) rebuild (cheap).
2. `update_secant_sy_storage_kernel` — storage diag + change stats via
   **per-thread FP64 atomics** (`warped_darcy.py:3942`).
3. `build_diag_preconditioner_kernel` — **4 harmonic-mean FP64 divisions per
   cell** (`warped_darcy.py:2278`).
4. `coarsen_transient_operator_level_kernel` per level-pair — **harmonic
   divisions again** (`warped_darcy.py:3971-4014`), plus a per-level M_inv
   rebuild.
5. RHS rebuild, then inner fixed-work K-cycles using the **classic kernels**
   (per-call harmonic divisions, per-thread atomic reductions) with **no
   CUDA-graph capture anywhere on this path**.
6. Outer convergence check with **6 scalar readbacks** (+ 4 per adaptive
   inner-controller block).

Every cost the confined campaign identified is present and amplified: the
harmonic divisions are recomputed **every outer iteration** (not merely per
kernel call), the atomic-serialized reductions also live in the storage and
check kernels, and graph capture is entirely absent.

**Crucial enabler:** the fast kernels in `solvers/face_kernels_f64.py` are
operator-agnostic. `face_jacobi`, `face_residual`, the two-stage reductions,
the check kernel, and the whole K-cycle/graph scaffolding in
`fast_confined_kcycle.py` consume `Te/Tw/Tn/Ts + diag` and masks — they know
nothing about where the coefficients came from. Any operator expressible as
5-point faces + diagonal — including `diag + storage_diag` and Picard-frozen
T(h_k) — drops straight in.

## 2. Phase A — Face-array operator for the device transient path

Precision-agnostic kernel engineering; expected largest win (~2/3 of the
confined speedup came from the equivalent work).

1. **Storage-aware face build**: extend `face_build_f64_kernel` with a
   `storage_diag` input added into `diag` (or a tiny `diag += storage`
   kernel). The face build already returns `diag`, so the separate fine and
   per-level `M_inv` launches (steps 3–4 above) disappear entirely.
2. **Per-outer refresh chain, all in-place** (pointer-stable so future
   graphs survive): T kernel → storage kernel → face/diag build (fusable
   into one launch) → existing device coarsen → per-level coarse face
   builds → RHS build. Coarse faces are built per level from that level's
   coarsened `T_wp` — the same semantics as fast-confined, reusing the
   existing `coarsen_transient_operator_level_kernel`.
3. **New face-kernel variants** (small, mechanical ports):
   - dual-residual kernel (`compute_dual_residual_kernel` → face form;
     head residual = r/diag **with storage in diag**);
   - dual check kernel (`kcycle_check_dh_and_dual_residual_kernel` → mirror
     `face_check_dh_residual_f64_kernel` with block-reduced partials);
   - secant-storage kernel change stats → two-stage block partials.
4. **Inner K-cycle**: swap classic kernels for `face_jacobi` /
   `face_residual` / block-reduced dots in the fixed-work mode of
   `solve_kcycle_device_buffers` (or reuse `solve_kcycle_fast_device_buffers`
   with storage-aware level wiring).

**Gate:** bit-level (or ~1e-12) head agreement vs the current device path on
one period; strict Picard convergence semantics unchanged.

## 3. Phase B — CUDA-graph capture

1. Capture the **inner fixed-work K-cycle block** (currently raw launches).
   The adaptive inner controller's variable block size (2–16 cycles) needs a
   small **graph cache keyed on `block_cycles`** (only a few distinct values
   occur in practice).
2. Capture the **per-outer refresh segment** (Phase A step 2) — launch-
   stable and pointer-stable after Phase A. Key the graph on `dt`/omega,
   which adaptive-dt retries change by value.
3. All convergence checks/readbacks stay **outside** captures (host decision
   points — same discipline as the confined fast path).

**Gate:** graph-vs-eager equivalence test (same outer counts, heads equal to
~1e-9), mirroring `tests/test_mixed_precision_fast.py`.

## 4. Phase C — Optional FP32 correction (mixed precision), opt-in only

Two structures, in increasing ambition:

1. **FP32 inner linear solve per Picard outer** (defect correction per
   linearisation). The strict Picard gate already checks the **true FP64
   dual residual** every outer, so the authoritative-FP64-outer structure
   that makes FP32 safe is naturally present. FP32 face hierarchy + FP64
   level-0 residual/accumulate, following the confined
   `MixedPrecisionFastSession` pattern.
2. **Full FP32 hierarchy with FP64 master head** across the whole period —
   bigger memory win (~halves; matters at 9M+ cells) but the secant-storage
   and adaptive-dt logic must be audited for FP32 sensitivity first.

**Gate:** full production replay acceptance — 500x500 52w and 1000x1000 30w:
strict Picard on all periods, RMSE ≤ ~1e-4 m vs MF6, mass-balance class
`excellent`, adaptive-dt net still a verified no-op. Also re-verify the
confined caveat that ordinary FP32 alone does **not** accidentally pass
gates. Stays out of the solver registry (test-enforced), like its confined
sibling.

## 5. Phase D — GHB on the device transient path

Currently `use_device_transient_fast_path` raises `NotImplementedError` when
GHB is enabled (`transient_unconfined.py:229-230`), so GHB cases fall back
to the much slower host Picard path. The face-build kernel already folds GHB
conductance `C_gh = T_c·ghb_factor` into `diag` (`face_kernels_f64.py:101-107`),
so Phase A makes this mostly a completeness exercise:

1. **RHS contribution**: GHB adds `C_gh·gh_head` to `b`. Fold it into the RHS
   build kernel (`build_transient_rhs_from_storage_kernel` variant) using
   the same per-cell `ghb_factor` used in the face build — must stay
   consistent with the diag term every outer iteration.
2. **Per-level GHB state on device**: `gh_mask`/`ghb_factor` must exist on
   every multigrid level. Confined coarsening already handles GHB arrays
   host-side; port the equivalent to the device coarsen path (or restrict
   GHB to a boundary mask, which coarsens trivially — decide during
   implementation against the host path's exact semantics).
3. **T-dependence**: `C_gh` depends on T(h), so it must be rebuilt every
   outer — already true if it lives in the Phase A face/diag build.
4. **Validation**: device path vs host Picard fallback (GHB already works
   there) on a GHB-enabled transient unconfined case, plus an MF6 truth run
   with GHB (modflow_truth supports `use_ghb`) meeting the same acceptance
   gates as the no-GHB production replays. Only then lift the
   `NotImplementedError` gate.

Depends on Phase A (face/diag build owns GHB in diag); independent of B/C.

## 6. Explicit non-goals / risks

- **Host Picard fallback** (`picard_unconfined.py`) not touched initially —
  it rebuilds the whole hierarchy per outer and would benefit greatly, but
  it is not the production path. A later phase can route it through
  `update_T_in_place` + face refresh.
- **Chebyshev λ bounds** (0.1/2.0, tuned for confined spectra) may need
  retuning once `diag` carries the storage term — expect better
  conditioning, but verify contraction per cycle.
- **Convergence gates are the contract**: strict Picard acceptance, adaptive
  inner-controller behavior, and adaptive-dt retry semantics must be
  observationally identical. Any change in outer-iteration counts vs the
  current path is a red flag to investigate, not accept silently.
- **Registry policy**: Phases A/B/D are internal to the existing
  `unconfined_picard_kcycle` backend (no new backend if head-equal); Phase C
  is opt-in and unregistered.

## 7. Sequencing and success metrics

Order: **A → B → D → C** (A is standalone value; B needs A's pointer
stability; D needs A's face build; C is optional and needs its own
campaign-style validation).

Each phase lands behind the existing replay gates with before/after numbers
on the two production grids. Current baselines (2026-07-19, FP64 classic):

| Case | Baseline | Acceptance |
|---|---|---|
| 500x500 52w homogeneous | 19.7 s (5.5× MF6) | 52/52 strict, RMSE 1.2e-05, MB excellent |
| 1000x1000 30w hard-T | 69.4 s (18.5× MF6) | 30/30 strict, RMSE 5.5e-05, MB excellent |

Confined saw 1.5–3.2× end-to-end from the equivalent kernel work (plus
3.0–4.4× with FP32 correction and graphs). The unconfined path has *more*
per-outer redundancy (harmonic rebuilds every outer, zero graph capture), so
the expected upside is at least that. Success = same acceptance tables at
≥2× the current runtimes, with GHB cases on the device path at parity with
the host-path reference.
