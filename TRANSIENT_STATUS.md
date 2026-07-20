# DarcyWarp Transient / Benchmark Status

Status of the DarcyWarp benchmark and validation matrix across the seven
groundwater-flow regimes. Last updated with the 3D transient confined runner
and the 3D transient unconfined phreatic-storage path.

## Benchmark / validation matrix

| Regime | Status | Coverage |
| --- | --- | --- |
| 2D steady unconfined | Mature | Runner + MF6 comparison + grid/backend/lambda sweeps + fixture replay tests |
| 2D transient confined | Working | Direct Warp tests + mass-balance checks (`tests/test_2d_transient.py`) |
| 2D transient unconfined | Working | Direct Warp tests + mass-balance; MF6 truth generator + Warp-vs-MF6 replay harness |
| 3D steady confined | Working | MF6-vs-Warp layer benchmark scaffolding (`working_tests/run_3d_warp_vs_mf6.py`) |
| 3D steady unconfined | Working | 2D speed controls ported into runner + 3D solver; CPU smoke test |
| 3D transient confined | Working | Solver-level storage/RHS support + transient MF6-vs-Warp runner mode |
| 3D transient unconfined | Working | Phreatic Sy storage path + transient MF6/Warp runner scaffolding; full benchmark grids not run |

## 3D transient confined

`working_tests/run_3d_warp_vs_mf6.py` now has a `transient_confined` mode. MF6
uses `make_mf_model_multilayer_transient` with `iconvert=0` and an STO package;
Warp replays each stress period with `transient=True`, `storage_coeff=Ss`,
`dt=<period length>`, and `head_prev=<previous period head>`.

The runner saves per-period heads, final heads, input metadata, timing, and
per-period/final error metrics. This is a small deterministic validation mode,
not a full layer/grid benchmark. Full benchmark grids have intentionally not
been run.

## 3D transient unconfined

3D transient unconfined now has two explicit storage modes:

- `phreatic_sy`: physical water-table storage. The Picard loop rebuilds storage
  every outer iteration from the current saturated thickness. Sy contributes
  `Sy * dx * dy / dt` only on the per-column water-table cell, and optional Ss
  contributes `Ss * saturated_thickness * dx * dy / dt` on saturated cells.
- `confined_volume`: legacy first-order approximation. This keeps the previous
  behavior where `storage_coeff` is treated as confined-style storage over full
  cell volume (`storage_coeff * dx * dy * dz / dt`).

Previously both `solve_chebyshev_7point_3d` and
`solve_multigrid_kcycle_7point_3d` rejected `transient=True` with
`unconfined=True`; that guard is removed.

### How it works

The shared Picard driver `_picard_unconfined_7point_3d` rebuilds face
conductances from the current saturated thickness and calls the inner solve.
For `phreatic_sy`, the Picard loop also builds the backward-Euler storage
diagonal and RHS contribution:

```
storage_diag = Sy * dx * dy / dt on water-table cells
             + Ss * saturated_thickness * dx * dy / dt on saturated cells
rhs         += storage_diag * head_prev
```

For `confined_volume`, the inner confined-transient solve still applies
`storage_coeff * dx * dy * dz / dt`. This mode is retained for compatibility
and comparison only.

### Validation caveat

The runner now has `transient_unconfined` scaffolding using
`make_mf_model_multilayer_transient(iconvert=1, sy=...)` and Warp replay with
`unconfined_storage_mode="phreatic_sy"`. The full MF6-vs-Warp 3D transient
unconfined validation has not been run in this pass. Wetting/drying transitions
are still limited to the existing `min_sat` floor.

### Reporting

The Picard info dict reports `transient`, `transient_formulation`, `dt`,
`unconfined_storage_mode`, `phreatic_storage_active`, `sy`, and `ss`.

## 2D transient unconfined Warp-vs-MF6 replay

`working_tests/run_2d_transient_warp_replay.py` steps the 2D Warp unconfined
transient solver through every stress period of an MF6 truth artifact
(`working_tests/run_2d_transient_vs_mf6.py`) and compares per-period/final heads.

- **Time units** are carried through unchanged (the flow equation is invariant
  under time-unit rescaling): MF6 day-valued fields are passed to Warp verbatim.
- **Storage**: Warp's unconfined transient term is `storage_coeff * dx**2 / dt`,
  i.e. phreatic storage over the cell area. The replay uses `storage_coeff = Sy`.
  MF6 also carries the much smaller confined `Ss * saturated_thickness` term,
  which Warp neglects (typically <1% of `Sy` for these cases).
- **MF6-free core**: `run_warp_transient_replay` takes plain NumPy spatial
  fields, so it is unit-tested without Flopy/MF6
  (`tests/test_2d_transient.py::test_2d_transient_replay_steps_periods_and_responds_to_recharge`).
- **Warm start**: large-grid transient unconfined replays should start from a
  confined steady-state head, not the raw geometric initial head. The MF6 truth
  workflow now defaults to generating `confined_steady_head` from the same
  spatial fields and mean transient recharge, then uses that head as the MF6
  transient `strt`. The Warp replay defaults to
  `warm_start_mode="confined_steady_mf6"` and requires regenerated artifacts
  containing that field. `warm_start_mode="artifact_initial"` remains available
  for old comparisons; `warm_start_mode="confined_steady_warp"` can generate a
  Warp-side confined steady start from the artifact fields.

On a 100x100, 12-week validation case the end-to-end comparison gave final
rmse ≈ 0.42 m, max-abs-diff ≈ 0.83 m, with a small positive Warp bias
attributable to phreatic-only storage + Picard-vs-Newton linearization +
dry-cell handling differences.

The 500x500 transient artifact must be regenerated before using the new default
warm-start replay path; old artifacts without `confined_steady_head` fail
clearly with regeneration instructions rather than silently starting from the
raw initial head.


## 2D transient unconfined convergence — RESOLVED (2026-07-19)

**The "1M-cell strict-Picard / accuracy failure" was premature practical
acceptance, not a solver deficiency.** On the 1000x1000 case the Picard outer
iteration contracts `dh_max` geometrically (~0.31x per outer iteration, from
~4.8 m on iteration 1 to <1e-4 m by iteration ~11) and the inner K-cycle hits
its target every iteration. The old production gate
(`min_practical_outer_iterations=8`) short-circuited every period ~3
iterations before strict success, shipping an iterate with `dh_max ~ 1.3e-3`.
That leftover head error is exactly the size of the reported accuracy failure
(RMSE 0.0025).

Resolution (defaults in `working_tests/transient_replay_settings.py`):

- `min_practical_outer_iterations`: 8 -> **20**. Practical acceptance reverts
  to a true fallback; strict (dh_max <= hclose AND head-residual RMS <= 1e-6
  AND inner linearisation solved) is the production gate.
- `adaptive_dt_strict_max_outer`: 6 -> **20**. The old value sat below the
  natural full-dt strict iteration count (~12 max) and guaranteed shrink/retry
  storms whenever adaptive dt was enabled.
- `adaptive_dt_enabled`: False -> **True** (MIKE-SHE-style safety net: shrink
  x0.5 on strict failure down to `dt_min = period_dt/16` with practical
  acceptance as the floor; grow x2 after clean strict acceptance, max 2 growth
  steps). With the corrected budgets it is a verified no-op on the production
  case: exactly 1 full-dt sub-step per period, 0 retries, 0 practical
  fallbacks, heads identical to the non-adaptive run.

Validation (2026-07-19, RTX 4070 Ti SUPER):

| Case | Strict periods | Final RMSE vs MF6 | Max abs | Mass balance | Runtime |
| --- | --- | --- | --- | --- | --- |
| 1000x1000 30w | 30/30 (10-12 outer iters) | 5.5e-05 m | 1.3e-4 m | excellent | 69.4 s (18.5x MF6) |
| 500x500 52w | 52/52 (p1: 4 outer iters) | 1.2e-05 m | 4.8e-5 m | excellent | 19.7 s (5.5x MF6) |
| 500x500 10w hard-T (ugly_t s42) | 10/10 (11 outer iters) | 1.4e-05 m | 5.7e-5 m | excellent | 5.9 s (16x MF6) |
| 1000x1000 30w hard-T (ugly_t s42) | 30/30 (9-16 outer iters) | 5.5e-05 m | 1.5e-4 m | excellent | 86.4 s (13.3x MF6) |

The hard-T case adopts the heterogeneous transmissivity field from the
confined steady-state benchmarks (`model_builder.make_ugly_T_field`: lognormal
correlated noise, high-T diagonal channel, low-T lenses; K = T / 100 m sat
thickness per the `export_mf6_truth_npz` convention, K ~ 4-535 m/day). The
case setup is owned by `run_2d_transient_warp_replay.py::build_case_setup()`;
the generator `run_2d_transient_vs_mf6.py` pulls it when run standalone, and
the replay auto-generates the MF6 artifact on first use
(`ensure_case_artifact`). Hard-T artifacts carry a `_ugly_t_s<seed>` suffix so
they never collide with the legacy homogeneous artifacts. On the hard field
strict Picard still converges at full dt (11 outer iterations vs 4 on the
homogeneous 500x500 — real but affordable difficulty) and the adaptive-dt net
remains a no-op.

Both production runs now PASS all gates including strict Picard on every
period; the 500x500 period-1 `startup_warning` mass-balance class is gone (it
was the same premature-acceptance leak). The 1000x1000 runtime exceeds the 30 s
stretch target (~2.3 s/period x 30); inner-cycle tuning is the lever if that
matters.

Cost of strict vs the old short-circuit: +3 outer iterations/period on the
1M-cell case (~0.5 s/period, +25% total). Step 1 (the hardest period)
converges strictly in 2.3 s — no timestep splitting, relaxed criteria, or
sub-stepping required.

### Failure-path economics (2026-07-19)

The production benchmark is simple physics (homogeneous K, uniform recharge,
no wells, no wetting/drying); harder cases will exercise the adaptive-dt
failure path, so it is now priced properly:

- **Early shrink** (`adaptive_dt_early_shrink_*`): once
  `early_shrink_min_outer` (6) iterations give a reliable dh contraction
  estimate, the driver projects iterations-to-hclose from the geometric
  contraction ratio and shrinks dt immediately when the projection cannot make
  the effective budget — instead of paying the full strict budget on a doomed
  sub-step. The projection is compared against `budget +
  extension_max_outer` (when an extension is still available), because an
  extension finishes a near-miss far cheaper than a shrink + retry. The
  projection must persist for `early_shrink_patience` (3) consecutive checks
  before the shrink fires: early-iteration contraction is often pessimistic
  (it accelerates as the Picard iterate settles), and without hysteresis the
  1M hard-T case misfired shrinks on periods that actually converge strictly
  at full dt in 15-16 iterations (within budget 20). Only genuinely hopeless
  sub-steps shrink early.
- **Budget extension** (`adaptive_dt_extension_*`): at budget exhaustion, if
  dh_max is within `extension_factor` (5x) of hclose and still contracting
  (ratio < 0.8), grant one extension of `extension_max_outer` (4) iterations
  instead of shrinking. A sub-step accepted via extension does not qualify
  for dt growth (grow-shrink oscillation guard).
- New per-period reporting: `adaptive_dt_early_shrink_count`,
  `adaptive_dt_extension_count`.

Validation (1000x1000 10w, strict budget forced to 10 so periods 1-2 miss by
one iteration): extension fires on exactly those periods, strict accepted at
iteration 11 at full dt, 102 total outer iterations vs 130 with the
extension-unaware policy and 128 with extension disabled. On a deliberately
broken synthetic (non-converging), early shrink reproduced the shrink
sequence at 144 vs 200 outer iterations (-28%). On the homogeneous and hard-T
production 30w cases all mechanisms are verified no-ops (0 early shrinks, 0
extensions, heads identical). Hard-T lesson: sub-stepping on high-recharge
periods integrates the transient better than MF6's single backward-Euler
step, so an unnecessary shrink shows up as an apparent ~0.1 m accuracy
regression vs the artifact plus the endpoint-flux mass-balance artifact —
not a solver error. The hysteresis fix keeps the net off unless strict
genuinely cannot make it.

### Adaptive dt audit history (2026-07-18)

The adaptive-dt mechanism was implemented (by Codex) to fix the 1M-cell
failure under the premise that full-dt strict convergence was unachievable.
The audit found the driver mechanically correct (fixed-dt/16 reproduction,
no-op on easy problems) and fixed six bugs (retry storm, practical leaking
into strict sub-steps, dt_min slivers, budget validation, reporting gaps,
missing tests). The premise was later shown false by the strict-first
experiment above: sub-stepping was never needed. The mechanism is retained as
the safety net described in `TRANSIENT_STATUS.md` (this section) with
mass-balance caveat: sub-stepped runs report `endpoint_flux_budget_approximation: true`
because the per-period CHD budget evaluates end-of-period rates (metric
artifact, not non-conservation).
