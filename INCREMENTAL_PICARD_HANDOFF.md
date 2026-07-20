# Handoff: Incremental Picard Form for 1M-Cell Transient Replay

> **Status (2026-07-18): IMPLEMENTED and validated — accuracy-NEUTRAL.**
> The incremental form is complete behind `use_incremental_picard` (default **False**) and
> proven correct (matches the direct path to ~machine precision). It does **not** fix the
> 1M-cell failure: under the existing adaptive inner controller both forms are driven to
> the same inner residual target, so they converge to the same iterate. Default kept off.
> Full A/B numbers and the next-step recommendation are in
> [Implementation Outcome](#implementation-outcome-2026-07-18) at the end of this doc.

## Context

The production transient unconfined replay passes acceptance at 500×500 (52 periods) but fails at 1000×1000 (1M cells, 30 periods):

- **Head accuracy FAIL**: final RMSE 0.00253 m, max abs diff 0.00633 m vs MF6.
- **Runtime FAIL**: 55.5 s vs 30 s production target.
- Mass balance is good (0.00414 % cumulative discrepancy).
- Strict Picard convergence fails in period 1; practical acceptance carries it.

The error grows with grid refinement, suggesting the current Picard/inner-solve loop is not converging tightly enough on fine grids. The user wants to explore an **incremental Picard form** (solve for the correction `δh` and accumulate it) because it can give the inner multigrid solver a zero-Dirichlet correction variable and a residual-driven RHS, which is often more stable and accurate than solving directly for the next head iterate.

## Current production fast-path algorithm

Entry: `WarpDarcySolver.solve_transient_2d_unconfined` → device fast path.

Per Picard outer iteration:

1. Build `T(h^k)` on device (`update_unconfined_transmissivity_from_head_kernel`).
2. Build secant-Sy storage diagonal `storage_diag(h^k, h_prev)` (`update_secant_sy_storage_kernel`).
3. Refresh fine/coarse preconditioner and operators.
4. Build RHS: `rhs = R*dx² + storage_diag * h_prev` (`build_transient_rhs_from_storage_kernel`).
5. Inner linear solve: `A(T, storage_diag) * h_lin = rhs`, initial guess `x = h^k` (`_solve_multigrid_kcycle_device_buffers`).
6. Relaxation/clipping: `h^{k+1} = h^k + ω * clip(h_lin - h^k, max_change)` (`apply_relaxed_clipped_picard_update_kernel`).
7. Physical clamp: `bottom+min_sat ≤ h^{k+1} ≤ top`.
8. Convergence checks on `dh = h^{k+1} - h^k` and nonlinear residual.

The inner solve treats the **full head** as the unknown. BC cells are held at `bc_values` inside the K-cycle via `enforce_constraints_kernel`.

## Proposed incremental Picard form

Keep the same secant-Sy storage coefficient and transmissivity, but reformulate the inner solve:

1. Build `T(h^k)` and `storage_diag(h^k, h_prev)` as before.
2. Build RHS `b = R*dx² + storage_diag * h_prev` as before.
3. **Compute nonlinear residual** at the current iterate:
   `r^k = b - A(T, storage_diag) * h^k`.
4. **Solve the correction equation**:
   `A(T, storage_diag) * δ = r^k`, with `δ = 0` at Dirichlet cells and zero initial guess.
5. **Update**:
   `h^{k+1} = h^k + ω * clip(δ, max_change_per_outer_iteration)`.
6. Apply physical clamp as before.
7. Convergence checks remain based on `dh = h^{k+1} - h^k` and the residual at `h^{k+1}`.

Mathematically, if the inner solve is exact, `h_lin = h^k + δ`, so the update is identical to the current code. Numerically, solving for a correction with zero BCs can change multigrid contraction, residual-based targets, and clipping behavior.

### Why it might fix 1M-cell failure

- The correction `δ` has homogeneous Dirichlet data; coarse-grid enforcement is cleaner.
- The inner solve directly targets the nonlinear residual, not the full head field.
- Starting from `δ = 0` may reduce overshoot/oscillation in early Picard iterations.
- Inner-solver tolerances can be tied more cleanly to the residual being driven to zero.

## Detailed change scope

### 1. New/updated device kernels (`DARCY_WARP_PACKAGE/warped_darcy.py`)

| Kernel | Change |
|--------|--------|
| `apply_relaxed_clipped_picard_update_kernel` | Either add a companion `apply_relaxed_correction_kernel(previous, correction, ...)` or generalize this one. Prefer a new kernel to keep the old path intact for comparison. |
| `build_transient_rhs_from_storage_kernel` | No change (still builds `b`). |
| `compute_residual_kernel` / `compute_dual_residual_kernel` | No signature change. Use `compute_residual_kernel` to form `r^k = b - A h^k` into a new `residual_wp` buffer. |
| `update_secant_sy_storage_kernel` | No change. |
| `update_unconfined_transmissivity_from_head_kernel` | No change. |
| `clamp_unconfined_head_kernel` | No change. |
| K-cycle solver internals (`enforce_constraints_kernel`, etc.) | No change required if we pass a zeroed `bc_values_delta_wp` array when solving for `δ`, so Dirichlet cells are pinned to `δ = 0`. |

### 2. New device buffers

In the device fast-path setup (around line 10330):

- `delta_wp`: correction field, same shape as `h_iter_wp`.
- `residual_wp`: nonlinear residual `b - A h^k`, same shape.
- `zero_bc_values_wp`: a zero array of `bc_values` shape, used only as `bc_values_wp` for the correction solve (so `enforce_constraints_kernel` pins `δ = 0` on Dirichlet cells).

Alternative: add an `is_correction` flag to `_solve_multigrid_kcycle_device_buffers` and zero BCs inside the solver. Passing a separate zeroed `bc_values` array is lower-risk because it avoids touching the K-cycle internals.

### 3. Outer Picard loop changes (`solve_transient_2d_unconfined`, device fast path)

The loop body currently does (simplified):

```python
build T(h_iter)
build storage_diag(h_iter, h_prev)
build rhs_eff = R + storage_diag * h_prev
snapshot h_iter -> h_snapshot
inner_solve(x=h_iter, rhs=rhs_eff)   # solves for h_lin
apply_relaxed_update(h_iter, h_snapshot)  # h_iter = h_snapshot + ω*(h_iter - h_snapshot)
clamp(h_iter)
check dh and residual
```

Replace with:

```python
build T(h_iter)
build storage_diag(h_iter, h_prev)
build rhs_eff = R + storage_diag * h_prev
snapshot h_iter -> h_snapshot
# 1. compute residual r = rhs - A*h_snapshot
compute_residual_kernel(x=h_snapshot, b=rhs_eff, r=residual_wp)
# 2. solve A*delta = residual, with delta=0 on Dirichlet cells
delta_wp.fill_(0.0)
inner_solve(x=delta_wp, rhs=residual_wp, bc_values=zero_bc_values_wp)
# 3. apply relaxed correction
apply_relaxed_correction_kernel(prev=h_snapshot, correction=delta_wp, output=h_iter)
clamp(h_iter)
check dh and residual
```

Important details:

- The K-cycle solver stores `x_prev` and computes convergence based on `x - x_prev`. Since `delta_wp` starts at 0 and `x_prev` is initialized to a copy of `delta_wp` at solver entry, its internal `dh` is just `δ`.
- After the update, `h_iter` holds `h^{k+1}`. The existing `kcycle_check_dh_and_dual_residual_kernel` (which compares `h_iter` to `h_snapshot`) still computes the correct `dh` and the residual at `h^{k+1}`.
- `evaluate_refreshed_nonlinear_candidate` also remains valid; it rebuilds `T`/`storage_diag` at `h^{k+1}` and recomputes the residual.

### 4. Confined pre-solve startup

The startup path (line ~10579) currently solves directly for `h_iter`. Two options:

1. **Keep as is** (direct solve for head). It is only one solve and not on the hot path.
2. **Convert to correction form** for consistency. Lower priority.

Recommended: keep the startup path unchanged initially to reduce risk.

### 5. Adaptive inner controller

The adaptive controller computes targets from the residual before the inner solve (`_fast_path_head_residual_check`). In the incremental form, that residual is `||r^k||`, which is exactly the quantity the correction solve will reduce. After the solve, the post-update residual check (`kcycle_check_dh_and_dual_residual_kernel`) gives `||b - A h^{k+1}|| = ||r^k - A δ||`. Therefore:

- Forcing `η` and target residuals remain meaningful.
- No changes to `_run_adaptive_inner_kcycle_blocks` or `_compute_inner_forcing_eta` are required.
- The initial residual passed to the adaptive controller will be the nonlinear residual at `h^k`, which is appropriate.

### 6. Convergence checks and reporting

- `last_dh_max` / `last_dh_rms` still measure `h^{k+1} - h^k`, i.e., the applied correction.
- `last_head_residual_rms` still measures the nonlinear residual at `h^{k+1}`.
- No reporting keys need to change, although it may be useful to add `incremental_picard_enabled` to the info dict for traceability.

### 7. Host-side helpers

`working_tests/transient_replay_storage.py` provides host reference implementations of the secant-Sy storage term. These do **not** need to change for incremental Picard because the storage coefficient itself is unchanged; only the linear-system solution strategy changes.

If a Newton-style storage Jacobian is attempted later, then `compute_unconfined_storage_components` would need a Jacobian variant. That is out of scope here.

### 8. Settings / controls

Add a solve-control toggle so the change can be A/B tested against the current direct-head path:

```python
"use_incremental_picard": True/False   # default True for production after validation
```

This toggle should be consumed in `solve_transient_2d_unconfined` (pop it from controls before forwarding to the legacy solver). Keep the default `False` until the new path is validated, then flip to `True`.

### 9. Test updates

- `tests/test_2d_transient.py`
- `tests/test_2d_transient_warm_start.py`

Add or update tests to run with `use_incremental_picard=True`. The 500×500 production acceptance should still pass; if it regresses, debug before enabling by default.

## Files expected to change

1. `DARCY_WARP_PACKAGE/warped_darcy.py`
   - New kernel: `apply_relaxed_correction_kernel` (or generalization of existing kernel).
   - Device fast path setup: allocate `delta_wp`, `residual_wp`, `zero_bc_values_wp`.
   - Device fast path Picard loop: compute residual, solve for correction, apply correction.
   - Control ingestion: pop `use_incremental_picard`.
   - Info dict: add `incremental_picard_enabled` flag.

2. `working_tests/transient_replay_settings.py`
   - Add `use_incremental_picard` to `default_solve_controls` (default `False` during validation).

3. `working_tests/transient_replay_support.py`
   - Optionally forward the flag (it will already pass through `solve_controls`).

4. `tests/test_2d_transient.py`, `tests/test_2d_transient_warm_start.py`
   - Add test cases / parametrization for `use_incremental_picard=True`.

5. `AGENTS.md`
   - Document the new production control once it becomes the default.

## Risks and mitigation

| Risk | Mitigation |
|------|------------|
| Incremental form does not improve 1M-cell accuracy. | Keep `use_incremental_picard` toggle; retain direct-head path for A/B comparison. Also consider stricter inner tolerances or smaller `dt` if incremental alone is insufficient. |
| K-cycle solver behaves differently when `bc_values=0` for correction; e.g., coarse-grid restrictions assume physical heads. | The operator `A` is the same and `δ` has zero BCs, which is the standard correction formulation. Verify by comparing 500×500 results. |
| Extra residual computation adds runtime. | One extra kernel launch per outer iteration, but it may be offset by fewer inner cycles. Benchmark both paths. |
| Clipping behavior changes. | In current code the clip is applied to `h_lin - h^k`. In incremental code the clip is applied to `δ`. These are identical mathematically, but numerical differences from the inner solve can lead to different clipping. Monitor `max_head_change_per_outer_iteration` statistics. |
| Breaks host fallback path. | The host path is not the production path. Apply the same simplification only to the device fast path. |

## Validation plan

1. **Unit/smoke**: run `pytest tests/test_2d_transient.py tests/test_2d_transient_warm_start.py` with the new flag enabled.
2. **500×500 full replay**: must still pass all acceptance gates (head accuracy, strict Picard, mass balance, runtime < 30 s).
3. **1000×1000 30-period replay**: the failing case. Goal is to reduce final RMSE and max abs diff to below the acceptance thresholds and bring runtime under 30 s.
4. **Mass balance**: cumulative discrepancy should remain excellent/good.
5. **A/B**: compare `use_incremental_picard=True/False` on both grid sizes to quantify improvement.

## Suggested first implementation steps for glm5.2

1. Add `apply_relaxed_correction_kernel` next to `apply_relaxed_clipped_picard_update_kernel`.
2. Allocate `delta_wp`, `residual_wp`, and `zero_bc_values_wp` in the device fast-path setup.
3. Wrap the new solve sequence in a conditional (`use_incremental_picard`) so the old direct-head path stays untouched.
4. Implement the residual → correction-solve → update sequence inside the main Picard loop.
5. Add `use_incremental_picard=False` to `default_solve_controls`.
6. Run the 500×500 replay with the flag enabled; iterate until it matches the direct path.
7. Run the 1000×1000 replay with the flag enabled; tune inner tolerances / cycle limits if needed.
8. Flip the default to `True` once validated and update tests.

## Notes

- The current production path already uses `storage_reference="current_picard"` and `unconfined_storage_mode="mf6_convertible_secant_sy"`; these remain unchanged.
- The active-set/threshold cleanup was completed before this handoff, so the codebase is in a simplified state with only the validated storage mode.
- Pre-existing issue unrelated to this work: `working_tests/run_device_transient_fast_path_smoke.py` raises `KeyError: 'inner_scalar_synchronization_count'`.

---

## Implementation Outcome (2026-07-18)

The scope above was implemented and validated on warp 1.11.0 / CUDA (RTX 4070 Ti SUPER),
flopy 3.9.5, MF6 `/bin/modflow/mf6`, against the existing truth artifacts.

### What changed (final)

1. `DARCY_WARP_PACKAGE/warped_darcy.py`
   - New kernel `apply_relaxed_correction_kernel` (mirrors `apply_relaxed_clipped_picard_update_kernel`,
     computes `output = previous + omega*clip(correction)`; doubles as the `omega=1`/no-clip
     per-block `h_iter = h^k + delta` sync).
   - Correction buffers allocated in the device fast-path setup: `delta_wp`, `residual_wp`,
     `zero_bc_values_wp`, `delta_snapshot_wp`.
   - `use_incremental_picard` added to the control pop-list and read from `fast_path_controls`.
   - Picard loop: after the snapshot, `r^k = b - A*h^k` is materialised via `compute_residual_kernel`
     and `delta_wp` zeroed; the adaptive block callback and legacy/fallback solve branch to solve
     `A*delta = r^k` (`bc_values = zero_bc_values_wp`); rollback restores `delta_snapshot_wp`;
     the relaxed update branches to `apply_relaxed_correction_kernel`. Confined startup unchanged.
   - `incremental_picard_enabled` added to the per-period info dict.
2. `working_tests/transient_replay_settings.py` — `"use_incremental_picard": False` in `default_solve_controls()`.
3. `tests/test_2d_transient.py` — `test_incremental_picard_matches_direct_head_path`.
4. `tests/test_2d_transient_warm_start.py` — updated the exact-equality default-controls test.

### Deviation from the doc's pseudocode (section 3 / 5)

The doc's section-3 sketch shows a single inner solve and asserts (section 5) the adaptive
controller needs no changes. That holds only when the solve variable *is* the residual-measurement
variable. In correction form they differ, so a one-line addition was required: inside the adaptive
block callback, `h_iter` is synced to `h^k + delta` before the (unchanged) `_fast_path_head_residual_check`
so the controller's contraction logic measures `||b - A*(h^k + delta)|| = ||r^k - A*delta||`. Cost:
one cheap field kernel per adaptive block. The pure-Python `_run_adaptive_inner_kcycle_blocks` itself
is untouched — only its `run_block`/`rollback_block` callbacks branch on the flag.

### Validation — A/B (direct vs incremental)

| Case | Metric | DIRECT (off) | INCREMENTAL (on) | Target |
|------|--------|--------------|-------------------|--------|
| 500×500 / 52p (passing) | final RMSE | 1.1962492e-5 | 1.1962493e-5 | <1e-3 ✓ |
| | production acceptance | PASS | PASS | — |
| | runtime | 19.5 s | 19.5 s | <30 s ✓ |
| 1000×1000 / 30p (failing) | final RMSE | 0.002533 | 0.002541 | <1e-3 ❌ |
| | final max abs diff | 0.006342 | 0.006362 | <5e-3 ❌ |
| | runtime | 55.9 s | 55.8 s | <30 s ❌ |
| | outer iterations | 240 | 240 | — |

Correctness is proven independently: on a near-linear problem the two forms agree to **7.99e-10**
(`test_incremental_picard_matches_direct_head_path`). Mass balance is excellent/good throughout.

### Why it does not improve the 1M-cell case

Under the existing adaptive inner controller, both forms are driven to the **same** inner residual
target (`forcing_eta * ||r^k||`). With an identical stopping criterion the correction form converges
to the **same iterate** as the direct form — the homogeneous-BC correction changes multigrid
*contraction* (potential speed), not the endpoint at a fixed tolerance. This is exactly the fallback
the doc's own risk table anticipated ("stricter inner tolerances / smaller dt if incremental alone
is insufficient"). Full unit suites pass (11/11 in `test_2d_transient.py`; structural warm-start tests).

### Decision and next step

- **Default kept `False`.** Flipping it would not change production output today; it is retained as
  the foundation for the next step and for A/B comparison.
- **To actually move 1M-cell accuracy**, combine the flag with a tighter inner solve (lower
  `inner_head_residual_tol_min`/`_max` and `inner_forcing_eta`), or reduce `dt` — the storage-diagonal
  destabilisation root cause documented in `working_tests/run_transient_unconfined_diagnostics.py`.
  A tolerance sweep on the 1000×1000 case is the suggested follow-up.
- Recorded to project memory (`incremental-picard-form-validated-neutral.md`) and `AGENTS.md`.

