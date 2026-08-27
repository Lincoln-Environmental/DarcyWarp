# 2D Transient Execution-Path Audit

> Phase 1 audit for the transient fast-path propagation campaign (commit
> "Add fast transient, artifact, and sanity-matrix hardening").  Operator
> types: **stencil** = per-level T/scalar arrays + M_inv rebuilds,
> **face** = harmonic face-conductance arrays (Te/Tw/Tn/Ts + diag).

| # | Path | Entry point | Operator | T changes | Storage changes | Recharge changes | dt changes | Face support | Graph support | Block reductions | Remaining host syncs | Production-optimization eligibility |
|---|------|-------------|----------|-----------|-----------------|------------------|------------|--------------|---------------|------------------|----------------------|-------------------------------------|
| 1 | Confined transient K-cycle, classic (default) | `solve_multigrid_kcycle(transient=True)` | stencil + storage diag (+GHB via gh arrays) | no (per solve) | no (per solve) | no (per solve) | no (per solve) | no | no | no (per-thread FP64 atomics) | scalar readbacks at every `check_every_no` cadence; host RHS/storage prep per call | reference implementation; superseded by path 2 for speed |
| 2 | Confined transient K-cycle, fast (opt-in `implementation="fast"`) | `solve_multigrid_kcycle(transient=True, implementation="fast")` | face (`face_transient_f64`), diag = Te+Tw+Tn+Ts + C_gh + storage_diag | no (per solve) | no (per solve) | no (per solve) | no (per solve) | yes (reuses `ensure_/refresh_transient_face_levels`; no duplicated formulas) | yes — per-call CUDA graph of the scalar-info K-cycle (fresh cache per call, exact-once capture/null/fallback semantics, eager fallback) | yes (two-stage partials) | host RHS/storage prep per call; convergence readbacks only at check cadence | **implemented this commit; classic remains default pending ≥1M-cell benchmark promotion gate** |
| 3 | Production unconfined Picard/K-cycle period driver (face + graphs, default) | `solve_transient_2d_unconfined` → `solvers/transient_unconfined.py` | face, refreshed per Picard outer | yes (per outer) | yes (secant-Sy per outer) | yes (per period; visible through captured graphs) | yes (adaptive-dt retry re-keys the refresh graph on dt) | yes (default) | yes (K-cycle + per-outer refresh graphs, keyed on dt and buffer identity) | yes (incl. block-partial storage-change stats, this commit) | one compact set of scalar readbacks per outer decision point; coarsest classic PCG kept for trajectory parity | production baseline — already optimized; only incremental work remains |
| 4 | Classic device transient operator (`transient_face_operator_enabled=False`) | same driver | stencil, per-outer M_inv rebuild | yes | yes | yes | yes | no | no | no (per-thread atomics; scalar stats re-zeroed per refresh) | per-outer scalar readbacks | kept as parity/reference path; not eligible for promotion |
| 5 | CPU/host Picard fallback | same driver (host path) | host stencil | yes | yes | yes | yes | n/a | n/a | n/a | entire solve on host | not eligible (correctness fallback only) |
| 6 | Experimental transient FAS | `solvers/transient_experimental.py` | rediscretized nonlinear FAS hierarchy | yes | yes | yes | yes | no | coarsest sweep block only | partial | per-cycle host checks | **not touched** — fails on hard-T 1M; no promotion |
| 7 | Production transient semismooth Newton alternative | `solvers/transient_experimental.py` | FGMRES + K-cycle preconditioner (cached workspace) | yes | yes | yes | yes | no | no | partial | FGMRES host syncs | production alternative; compatibility driver name retained |
| 8 | GHB variants | paths 2/3 only | C_gh in face diag, `C_gh·gh_head` in RHS | — | — | — | — | yes (required) | yes | yes | — | production on face path; classic device path raises for GHB |
| 9 | Precision variants | paths 2/3 | FP64 master everywhere; FP32 correction (`mixed_transient_f32`, face path only) | — | — | — | — | — | — | — | — | mixed correction is enabled by the production unconfined replay |

## Notes

- Path 2 deliberately routes through `face_transient_f64` so the confined
  transient fast path shares the exact face/storage formulas with the
  production unconfined driver (no second implementation).
- Graph reuse does not cross solver invocations: both transient graph caches
  are per solve call; later-period reuse happens only within one call.
- Steady confined fast (`fast_confined_kcycle.py`) still rejects
  `transient=True`; the transient generalization lives in
  `multigrid_kcycle.solve_multigrid_kcycle_backend` (path 2).
