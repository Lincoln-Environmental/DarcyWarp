# Execution Prompt: Implement Adaptive Timestepping for 1M-Cell Transient Replay

> Do not write a plan. Do not summarize the codebase. Do not explore. Implement the exact changes below and run the validation.

## Goal

Add adaptive timestepping to the 2D unconfined transient device fast path in `DARCY_WARP_PACKAGE/warped_darcy.py`. The MF6 artifact still reports heads at fixed 7-day stress periods, but internally each period may be split into smaller substeps when strict Picard convergence is difficult. This improves conditioning (`storage_diag = S·dx²/dt`) on fine grids and should fix the 1000×1000 accuracy failure without the cost of fixed small dt everywhere.

## Files to change

1. `DARCY_WARP_PACKAGE/warped_darcy.py`
2. `working_tests/transient_replay_settings.py`
3. `tests/test_2d_transient.py` or `tests/test_2d_transient_warm_start.py`

Do not modify any other files.

## Algorithm

Inside `WarpDarcySolver.solve_transient_2d_unconfined`, wrap the existing per-period Picard loop in an inner substeps loop. Keep the existing Picard loop body intact; only change how `dt` is applied and how `h_prev` is advanced within a period.

For each stress period `p`:

```text
period_dt    = dt_f_val          # 7.0 days from artifact
recharge_rate = rates[p]
remaining    = period_dt
current_dt   = period_dt         # try full period first
dt_min       = period_dt * adaptive_dt_min_fraction

while remaining > 0:
    actual_dt = min(current_dt, remaining)

    # 1. Set recharge for this sub-step (constant over period)
    self.update_uniform_recharge_in_place(recharge_rate)

    # 2. Solve one transient sub-step of length actual_dt
    #    Use the existing Picard loop, but with:
    #      - h_prev_wp = head at start of sub-step
    #      - dt passed to storage kernel = actual_dt
    #      - strict Picard as the acceptance gate
    converged, accepted_head = run_picard_substep(
        initial_head=h_start_wp,
        h_prev=h_start_wp,
        dt=actual_dt,
        max_outer=adaptive_dt_strict_max_outer,
        acceptance_mode='strict',
    )

    if converged:
        # advance
        h_start_wp = accepted_head
        remaining -= actual_dt
        # try to grow dt, but not above period_dt
        current_dt = min(period_dt, current_dt * adaptive_dt_grow_factor)
    else:
        # shrink and retry from same start time
        if actual_dt <= dt_min + eps:
            # even minimum dt failed strict; accept with practical acceptance at dt_min
            converged, accepted_head = run_picard_substep(
                initial_head=h_start_wp,
                h_prev=h_start_wp,
                dt=actual_dt,
                max_outer=max_outer,
                acceptance_mode='practical',
            )
            if not converged:
                raise RuntimeError(f"adaptive dt failed at dt_min={dt_min}")
            h_start_wp = accepted_head
            remaining -= actual_dt
            # keep dt at dt_min for next sub-step
        else:
            current_dt = max(dt_min, actual_dt * adaptive_dt_shrink_factor)
            # do not advance remaining; retry same start time

# after sub-steps complete, copy final head to heads_per_period[p]
```

### Important details

- The existing Picard loop already updates `h_iter_wp` and uses `h_prev_wp` for storage. Reuse it by making `h_prev_wp` point to `h_start_wp` at the beginning of each sub-step.
- The existing loop already calls `build_transient_rhs_from_storage_kernel` with `storage_diag_wp * h_prev_wp`. Ensure `h_prev_wp` is the sub-step start head, not the period start head.
- At the end of each accepted sub-step, copy `h_iter_wp` into the next sub-step's start head buffer and into `h_prev_wp`.
- Mass-balance and diagnostic reporting are per period, not per sub-step. Accumulate or keep period totals as they are now.
- Do not change the confined startup pre-solve logic.

## New solve controls

Add these to `default_solve_controls()` in `working_tests/transient_replay_settings.py`:

```python
"adaptive_dt_enabled": False,          # default False until validated
"adaptive_dt_min_fraction": 0.0625,    # dt_min = period_dt / 16
"adaptive_dt_shrink_factor": 0.5,
"adaptive_dt_grow_factor": 2.0,
"adaptive_dt_strict_max_outer": 6,     # strict conv must be reached by 6 iters
"adaptive_dt_max_growth_steps": 2,     # optional cap on consecutive growth
```

Pop `"adaptive_dt_enabled"` and the other adaptive-dt keys from `controls` in `solve_transient_2d_unconfined` before forwarding to the legacy solver (same pattern as `use_device_transient_fast_path`).

## Implementation steps

1. In `warped_darcy.py`, locate the device fast path per-period loop (around line 10559). Identify the code that runs from period start through Picard convergence to copying the head into `heads_per_period`.
2. Wrap that code in a `while remaining > 0` substeps loop.
3. Add a sub-step start head buffer `h_substep_start_wp` (same shape as `h_iter_wp`).
4. Replace the fixed `dt_f_val` passed to `update_secant_sy_storage_kernel` and `build_transient_rhs_from_storage_kernel` with a loop variable `actual_dt`.
5. After a successful sub-step, copy `h_iter_wp` to `h_substep_start_wp` and `h_prev_wp`.
6. After a failed strict sub-step, shrink `actual_dt` and retry without advancing.
7. At period end, copy the final accepted head to `heads_per_period[period_index]` as before.
8. Add the new controls to `transient_replay_settings.py`.
9. Add `incremental_picard_enabled` and `adaptive_dt_enabled` to the per-period info dict if not already present.

## Validation

Run in this order and report all numbers:

1. `pytest tests/test_2d_transient.py tests/test_2d_transient_warm_start.py -q --tb=short` — must pass (60 tests).
2. 500×500 full replay with `adaptive_dt_enabled=False` — must still pass all gates (baseline unchanged).
3. 1000×1000 30-period replay with `adaptive_dt_enabled=False` — confirm it still fails as before.
4. 1000×1000 30-period replay with `adaptive_dt_enabled=True`, default fractions — report final RMSE, max abs diff, runtime, and whether production acceptance passes.
5. If (4) improves but not enough, try `adaptive_dt_min_fraction=0.125` (dt_min = period/8) and `adaptive_dt_min_fraction=0.03125` (period/32), and report results.

The artifact path for 1000×1000 is:
`/home/patrickdurney/PycharmProjects/DarcyWarp/DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined_1000x1000_30w/mf6_transient_heads.npz.lzma`

Use the conda env `darcywarp`.

## Acceptance criteria for this task

- 500×500 replay with `adaptive_dt_enabled=False` remains passing.
- 1000×1000 replay with `adaptive_dt_enabled=True` either:
  - passes production acceptance (final RMSE < 1e-3, max abs < 5e-3, runtime < 30 s), or
  - shows a clear monotonic improvement toward those targets as `adaptive_dt_min_fraction` is reduced.

If you cannot reach the targets, report the best configuration and the bottleneck.
