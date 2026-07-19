kimi --model kimi-code/k3
# DarcyWarp — Agent System Memory

> Last updated: 2026-07-18 by ingest of the full repo.
> Purpose: prevent re-ingesting the project from scratch. Read this first, then grep/read the specific files listed below.

---

## 1. What this project is

DarcyWarp is a **GPU-accelerated groundwater-flow solver** built on NVIDIA Warp (`warp-lang`). It solves structured-grid Darcy flow (5-point in 2D, 7-point in 3D) and is benchmarked against **MODFLOW 6** via FloPy, plus an optional CPU finite-difference reference.

- Repo root: `/home/patrickdurney/PycharmProjects/DarcyWarp`
- Conda env: `darcywarp` — the project's dedicated environment (Warp 1.11.0 + CUDA, MF6,
  flopy; created from `environment.yml` via `conda env create -f environment.yml`).
  `quick_flow_env` is a borrowed env from another repo and is **no longer used** here.
- License: AGPL-3.0-or-later
- Precision: set via `DARCY_FLOAT` env (`float32` | `float64`); default is currently `float64` in `warped_darcy.py` and `float32` in `config.py` — **check both**.
- MF6 binary: discovered in `DARCY_WARP_PACKAGE/project_base.py`; default `/bin/modflow/mf6`, fallback `which mf6`. The `darcywarp` env also ships `modflow6`, so `mf6` is on PATH when the env is active.

The code has two eras:

1. **Original hand-written core**: steady 2D K-cycle/PCG, `model_builder.py`, `sparse_operator.py`, benchmark scaffolding.
2. **AI-expanded layers** (GPT/Codex): transient 2D unconfined, adaptive inner controller, 3D transient unconfined, replay infrastructure, mass-budget diagnostics.
3. **Solver-extraction migration** (Codex, 2026-07, `dev` branch): 2D algorithm bodies moved out of the `warped_darcy.py` monolith into `DARCY_WARP_PACKAGE/solvers/` (registry + backends) and `physics/`. `warped_darcy.py` retains the `@wp.kernel` definitions, hierarchy construction, BC/field ownership, and thin compatibility wrappers; backends re-import the model module's globals at call time (`globals().update(...)`) as a deliberate lazy bridge.

---

## 2. Architecture at a glance

```text
DARCY_WARP_PACKAGE/
  config.py              # DARCY_FLOAT -> WP_FLOAT/NP_FLOAT
  project_base.py        # data_store path, MF6 binary discovery
  factory.py             # create_solver(dim=2|3, solver=...)
  model_builder.py       # synthetic domains, DEM, BC masks, T/R fields
  sparse_operator.py     # CPU scipy reference matrix Ah=b mirroring GPU kernels
  modflow_truth.py       # FloPy MF6 truth models + persistent batch workers
  model.py               # public 2D facade re-exporting WarpDarcySolver
  warped_darcy.py        # 2D model: ~7.5k lines — @wp.kernel defs, hierarchy build,
                         # BC/field/array ownership, compatibility wrappers -> solvers/
  solvers/               # extracted 2D backends (post-migration)
    registry.py          # backend selection + legacy aliases (pcg/kcycle/mg/picard)
    multigrid_kcycle.py  # confined K-cycle backend + device-buffers inner solve
    picard_unconfined.py # unconfined Picard/K-cycle backend (host fallback path)
    transient_unconfined.py # transient period driver incl. production device fast path
    pcg.py               # confined PCG backend
    semismooth_newton.py # experimental Newton backend (FGMRES + K-cycle preconditioner)
    fas.py               # experimental FAS V-cycle backend
    fas_hierarchy.py, fas_kernels.py, fas_state.py  # FAS rediscretized hierarchy/state/kernels
    fgmres.py, kcycle_preconditioner.py, newton_kernels.py  # Newton machinery
    capabilities.py      # shim re-exporting solver_capabilities.py
    context.py, base.py, hierarchy.py, convergence.py, resources.py, regression.py
  solver_capabilities.py # CAPABILITIES/ALIASES metadata (experimental,
                         # supports_transient, supports_production_period_driver)
  nonlinear/             # authoritative 2D unconfined nonlinear operator
                         # (context/kernels/operator/reference) — Newton/FAS foundation
  physics/               # extracted operator/storage/budget helpers
  warped_darcy_3d.py     # thin 3D wrapper class
  solvers_3d.py          # 3D algorithms: K-cycle, Chebyshev, Picard
  kernels_3d.py          # all 3D @wp.kernel definitions
  k_cycle_multigrid.py   # standalone reference K-cycle utility

working_tests/
  run_2d_transient_vs_mf6.py        # MF6 truth artifact generator
  run_2d_transient_warp_replay.py   # thin production replay harness
  optimize_2d_transient_kcycle.py   # K-cycle tuning sweep (experiment harness, not production)
  transient_replay_support.py       # core replay harness
  transient_replay_settings.py      # production settings / defaults
  transient_replay_storage.py       # storage-formulation helpers
  transient_replay_mass_balance.py  # water-budget computation
  transient_replay_metrics.py       # head-accuracy metrics
  transient_replay_reporting.py     # acceptance reporting
  run_transient_unconfined_diagnostics.py  # convergence-failure diagnostic ladder
  run_3d_warp_vs_mf6.py             # 3D validation runner

tests/
  test_2d_transient.py
  test_2d_transient_warm_start.py
  test_2d_unconfined.py
  test_3d_solver.py
  test_3d_transient_runner.py
  test_adaptive_inner_controller.py
  test_comparison_results.py
  test_sparse_system_consistency.py
  test_varied_T.py
```

---

## 3. The 2D solver (`warped_darcy.py`)

### Entry points

- `WarpDarcySolver.__init__` (~l. 4100)
- `build_from_truth_inputs` (~l. 5208)
- `build_from_fields` (~l. 5379)
- `solve` (~l. 7250) — public dispatcher (thin delegate to `solvers/registry.py`)
- `solve_transient_2d_unconfined` (~l. 7233) — multi-period API (thin delegate to
  `solvers/transient_unconfined.py::solve_transient_unconfined`, ~l. 2030)
- `solve_multigrid_kcycle` (~l. 7208) — thin delegate; real bodies:
  `solvers/multigrid_kcycle.py::solve_multigrid_kcycle_backend` (~l. 607, host) and
  `solve_kcycle_device_buffers` (~l. 21, device inner solve)
- `update_T_in_place` (~l. 6254) — ensemble T-change fast path

### Solvers

Canonical backends live in `solvers/registry.py` + `solver_capabilities.py`:

- Production: `confined_pcg` (steady only; `transient=True` raises
  `NotImplementedError`), `confined_kcycle`, `unconfined_picard_kcycle`
  (production default; the only backend allowed in the multi-period transient
  driver — gated via `supports_production_period_driver`).
- Experimental (`select_backend` emits a runtime warning; steady or
  single-period transient `solve(...)` only): `unconfined_semismooth_newton_kcycle`,
  `unconfined_fas`.
- Legacy aliases `pcg`, `kcycle`, `multigrid`, `mg`, `picard`, `picard_kcycle`;
  for unconfined solves, `kcycle` still means the Picard/K-cycle backend.

### Unconfined physics

- `T(h) = K * sat(h)` where `sat = clip(h - bottom, min_sat, top - bottom)`.
- `min_sat` (default 0.1 m) is a numerical floor. Cells are **never deactivated** — this is not a full drying/rewetting package.
- Picard outer loop updates T each iteration, then solves the linearised problem.
- Optional: Chebyshev acceleration, adaptive omega, update clipping, transmissivity relaxation.

### Transient physics

- Backward-Euler storage diagonal: `storage_diag = storage_coeff * dx^2 / dt`.
- RHS: `b = R*dx^2 + storage_diag * head_prev`.
- Storage diagonal is carried on every multigrid level (summed in 2×2 coarsening).
- 2D unconfined storage modes (see `transient_replay_storage.py`):
  - `phreatic_only`
  - `integrated_sy_ss`
  - `mf6_convertible`
  - `mf6_convertible_top_switch`
  - **`mf6_convertible_secant_sy`** ← production default
- Production fast path (device-side Picard loop) requires exactly:
  - `unconfined_storage_mode="mf6_convertible_secant_sy"`
  - `storage_reference="current_picard"`
  - GHB disabled.
- Optional `use_incremental_picard` (default **False**): solve the inner system for the
  correction `A·δ = r^k` (δ=0 on Dirichlet cells) instead of the full head
  (`apply_relaxed_correction_kernel`). Validated **accuracy-neutral** vs the direct
  path — it matches direct to ~machine precision but does **not** improve the 1M-cell
  case, because the adaptive inner controller drives both forms to the same residual
  target (hence the same iterate). Kept off by default.
- Optional `adaptive_dt_enabled` (default **True**): MIKE-SHE-style safety net
  with strict-first acceptance, shrink-to-`dt_min` and practical fallback at
  `dt_min` (controls `adaptive_dt_*` in `transient_replay_settings.py`).
  RESOLVED 2026-07-19: the "1M-cell strict-Picard / accuracy failure" was
  premature practical acceptance — the old `min_practical_outer_iterations=8`
  fired ~3 outer iterations before strict success (strict needs 10-12; dh_max
  contracts ~0.31x/outer). With `min_practical_outer_iterations=20` and
  `adaptive_dt_strict_max_outer=20`, strict Picard passes on every period at
  full dt (1000x1000 30w: RMSE 5.5e-05, mass balance excellent, 69 s) and the
  adaptive net is a verified no-op (1 full-dt sub-step/period, 0 retries).
  Sub-stepping engages only when strict genuinely fails within the budget.
  The failure path is priced: **early shrink** projects dh contraction and
  shrinks dt as soon as strict provably cannot make budget+extension (with
  `early_shrink_patience` hysteresis — the projection must persist 3
  consecutive checks, since early-iteration contraction is pessimistic on
  hard-but-convergent periods); **budget extension** grants one 4-iteration
  extension when dh_max is within 5x of hclose and still contracting
  (finishes near-misses without a shrink; extension-assisted accepts don't
  qualify for dt growth). Both are verified no-ops on the homogeneous AND
  hard-T (`ugly_t` seed 42, K ~ 4-535 m/day) 1000x1000 30w production cases.
  Mass-balance reporting for sub-stepped runs carries
  `endpoint_flux_budget_approximation: true` (endpoint-flux budget is a metric
  artifact there, not non-conservation). See `TRANSIENT_STATUS.md` § 2D
  transient unconfined convergence.

### Adaptive inner controller

- Enabled by `adaptive_unconfined_inner_enabled=True` (production default).
- Runs K-cycles in blocks, measures residual contraction, and adjusts block size.
- Uses dual residuals: flow residual `b - Ah` and head-equivalent residual `(b - Ah)/diag(A)`.
- Falls back to legacy head-change-driven cycle cap if residual path fails.
- Heavily configurable; defaults in `transient_replay_settings.py`.

### Convergence / acceptance

- Strict Picard (**production gate**): `max_abs_head_change < hclose` AND
  head-equivalent residual RMS < strict tol AND inner linearisation solved to
  target. Converges in 10-12 outer iterations on the 1000x1000 production case.
- Practical acceptance (**fallback only**): fires no earlier than
  `min_practical_outer_iterations=20`, head residual < practical tol, `dh_rms`
  < practical tol, storage-diag change RMS < practical tol. Only engages when
  strict genuinely stalls.
- Production result is **strict Picard on all periods**.

---

## 4. The 3D solver (`solvers_3d.py` + `kernels_3d.py`)

### Entry points

- `WarpDarcySolver3D` in `warped_darcy_3d.py` (~l. 24)
- `solve_multigrid_kcycle_7point_3d` (~l. 1763)
- `solve_chebyshev_7point_3d` (~l. 1444)
- `_picard_unconfined_7point_3d` (~l. 752) — shared Picard driver

### Key differences from 2D

- 7-point stencil from six face-conductance arrays.
- Horizontal-only semi-coarsening (`1×2×2`) to preserve layers.
- Vertical-line relaxation: Thomas tridiagonal solve per column (launch `dim=(ny*nx,)`); `nz <= 64` default.
- `WarpDarcySolver3D` is intentionally thin; algorithms are in standalone functions.

### Transient 3D

- Confined: volume storage `storage_coeff * dx*dy*dz / dt`.
- Unconfined: two modes
  - `phreatic_sy`: physical — `Sy*dx*dy/dt` on water-table cell only + `Ss*sat*dx*dy/dt` on saturated cells.
  - `confined_volume`: legacy fallback.
- Status: scaffolding exists; full benchmark grids **not run** per `TRANSIENT_STATUS.md`.

---

## 5. Transient replay infrastructure

### Case setup ownership (single source of truth)

`working_tests/run_2d_transient_warp_replay.py::build_case_setup()` owns the
replay case: grid, periods, storage, recharge, warm start, and the T-field
spec. Default T field is the **hard heterogeneous benchmark field**
(`t_field_kind="ugly_t"`, seed 42) adopted from the confined steady-state
benchmarks (`model_builder.make_ugly_T_field`, K = T / 100 m per the
`export_mf6_truth_npz` convention, K ~ 4-535 m/day);
`t_field_kind="homogeneous"` reproduces the legacy K=100 case.

- `ensure_case_artifact(setup)` returns the artifact path, generating the MF6
  truth via a lazy call to `run_2d_transient_vs_mf6.py::main(...)` when the
  artifact is missing.
- `run_2d_transient_vs_mf6.py` run standalone pulls the same setup from the
  replay (`build_case_setup` + `ensure_case_artifact`), so both entry points
  always agree.
- Hard-T artifacts get a `_ugly_t_s<seed>` directory suffix; homogeneous paths
  are unchanged (backwards compatible with existing artifacts).

### Generating truth

```bash
python working_tests/run_2d_transient_vs_mf6.py   # pulls case setup from the replay
```

Produces `DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined_<nx>x<ny>_<n_weeks>w[_ugly_t_s<seed>]/mf6_transient_heads.npz.lzma`.

Artifact contains: per-period heads, final heads, initial/warm-start heads, masks, DEM, bottom, K, Sy, Ss, recharge rates, provenance.

### Running replay

```bash
python working_tests/run_2d_transient_warp_replay.py                          # hard-T default, auto-generates artifact
python working_tests/run_2d_transient_warp_replay.py --t-field-kind homogeneous  # legacy uniform K=100 case
python working_tests/run_2d_transient_warp_replay.py --nx 500 --ny 500 --n-periods 10 --t-field-seed 7
```

The replay is a thin harness: production solve controls come from
`transient_replay_settings.py::default_solve_controls()` (no duplicated
control lists in the script). CLI switches: `--t-field-kind {ugly_t,homogeneous}`,
`--t-field-seed`, `--nx/--ny/--n-periods`, `--artifact`, `--workspace`,
`--device`. The K-cycle tuning sweep is a separate experiment harness:
`python working_tests/optimize_2d_transient_kcycle.py` (same case CLI, plus
`--stop-after-first-accepted`).

- Loads artifact via `transient_artifacts.py`.
- Calls `run_replay_from_artifact()` in `transient_replay_support.py`.
- Defaults to warm start from MF6 unconfined steady head.
- Compares Warp heads to MF6 per-period and final.
- Computes mass balance (volume-Sy preferred for secant-Sy mode).
- Reports strict/practical/production acceptance.

### Production settings lock-down

The production replay only accepts:

- `unconfined_storage_mode="mf6_convertible_secant_sy"`
- `storage_reference="current_picard"`
- `warm_start="unconfined_steady_mf6"`
- `unconfined_startup_mode="confined_pre_solve"`

Deviations must use lower-level solver calls or set `allow_warm_start_mismatch=True`.

---

## 6. Terminology traps

### "Unsaturated" does NOT mean variably-saturated vadose-zone flow

The codebase has **no Richards equation, no van Genuchten, no relative permeability**. When the user says "unsaturated paths", they almost certainly mean **unconfined / water-table / specific-yield** physics:

- Saturated thickness `sat(h)`.
- Specific yield `Sy`.
- Secant-Sy and top-switch logic.

If true unsaturated flow is requested, it does not exist and would be a major new feature.

### "Saturation" in the code

Usually means `sat = h - bottom` (saturated thickness of the aquifer), not soil moisture saturation.

### "Storage coefficient" vs "specific yield"

- `storage_coeff` is the effective storage coefficient used in the discrete equation.
- `sy` / `ss` are physical parameters.
- In production secant-Sy mode, `storage_coeff` is computed from secant Sy + secant Ss each Picard iteration.

---

## 7. Known performance / robustness issues

### 2D transient unconfined

- RESOLVED 2026-07-19: **strict Picard "failures" and the 1M-cell accuracy
  failure were premature practical acceptance**, not solver deficiencies (see
  §3 Convergence / acceptance). 1000x1000 30w: 30/30 strict, RMSE 5.5e-05,
  69.4 s (18.5x MF6). 500x500 52w: 52/52 strict, RMSE 1.2e-05, 19.7 s (5.5x
  MF6), period-1 `startup_warning` mass-balance class eliminated.
- 1000x1000 strict-everywhere runtime (~2.3 s/period) exceeds the 30 s stretch
  target; inner-cycle tuning is the lever if that matters.

### 3D transient unconfined

- Phreatic-Sy path exists but full validation/benchmarks not run.
- Storage rebuild happens on host per Picard iteration (no device fast path).
- Wetting/drying limited to `min_sat` floor.

### General

- Per-iteration host sync for scalar reductions in PCG/K-cycle convergence checks.
- Memory management: explicit `gc.collect()` calls to break Warp array reference cycles during hierarchy rebuilds.
- `warped_darcy.py` is still ~7.5k lines post-migration and the `solvers/*` backends re-import its globals at call time (`globals().update(...)`); small changes can have non-obvious interactions between storage modes, active-set strategies, and adaptive controller fallback.

---

## 8. Test & validation landscape

### Test files

| File | Needs | Validates |
|------|-------|-----------|
| `test_2d_transient.py` | warp | transient storage assembly, kernels, confined/unconfined mass balance |
| `test_2d_transient_warm_start.py` | flopy | replay defaults, production acceptance, mass-balance classes |
| `test_2d_unconfined.py` | warp | host/device parity, dry cells, MF6 truth fixtures |
| `test_3d_solver.py` | warp | 3D transient unconfined smoke, vertical-line reference, scipy reference |
| `test_adaptive_inner_controller.py` | — | pure-Python adaptive block controller |
| `test_solver_registry_2d.py` | — | backend registry, aliases, capability flags |
| `test_nonlinear_operator_2d.py` | warp | Stage-1 nonlinear operator vs host reference, Jv, exact storage |
| `test_semismooth_newton_2d.py` | warp | experimental Newton backend (steady/transient/GHB/fallbacks) |
| `test_fas_2d.py` | warp | experimental FAS backend (hierarchy, cycles, fallbacks, workspace reuse) |
| `test_comparison_results.py` | warp + fixtures | end-to-end Warp vs MF6 truth |

### Important caveats

- Many tests skip if `warp` missing. In the `darcywarp` env `warp` **is** installed (1.11.0, CUDA-enabled).
- Truth fixtures under `tests/fixtures/unconfined_2d/` are git-untracked; clean clone must run `working_tests/regenerate_unconfined_2d_truth.py`.
- 3000×3000 comparison fails on 16 GB GPUs (memory, not correctness).
- `test_2d_transient_warm_start.py` has no explicit warp skip guard; collection can fail if warp missing.

### Recommended smoke commands

```bash
# Check warp availability
python -c "import warp as wp; print(wp.__version__, wp.is_cuda_available())"

# Run 2D unconfined fixture tests (requires pre-generated fixtures)
python -m pytest tests/test_2d_unconfined.py -k "test_unconfined_warp_matches_mf6_truth_all_grids"

# Run transient diagnostic ladder
python working_tests/run_transient_unconfined_diagnostics.py --device cuda:0 --grid small

# Run production replay
python working_tests/run_2d_transient_warp_replay.py
```

---

## 9. Where to find things

| Topic | File(s) | Notes |
|-------|---------|-------|
| Solver factory | `factory.py` | `create_solver(dim=2\|3, ...)` |
| Precision config | `config.py` | `DARCY_FLOAT` env |
| MF6 binary path | `project_base.py` | `MF6`, `require_mf6()` |
| CPU reference matrix | `sparse_operator.py` | `build_sparse_system_fd_like` |
| 2D kernels | `warped_darcy.py` | inlined `@wp.kernel` functions |
| 2D backends/registry | `solvers/registry.py`, `solvers/*.py` | post-migration backend dispatch |
| 3D kernels | `kernels_3d.py` | all 3D kernels |
| 3D algorithms | `solvers_3d.py` | K-cycle, Chebyshev, Picard |
| Adaptive inner controller | `warped_darcy.py` ~l. 631-1363 | `_run_adaptive_inner_kcycle_blocks` ~l. 1178 |
| Production replay | `working_tests/run_2d_transient_warp_replay.py` | |
| Replay core | `working_tests/transient_replay_support.py` | `run_replay_from_artifact` |
| Storage formulations | `working_tests/transient_replay_storage.py` | `compute_unconfined_storage_components` |
| Mass balance | `working_tests/transient_replay_mass_balance.py` | |
| Acceptance reporting | `working_tests/transient_replay_reporting.py` | |
| Diagnostics ladder | `working_tests/run_transient_unconfined_diagnostics.py` | read verdict logic ~l. 1332 |
| Status docs | `TRANSIENT_STATUS.md`, `transient_progress.md`, `working_tests/darcywarp_transient_unconfined_changes.rst` | |
| Benchmark entry | `bench_and_plot.py`, `model_benchmarking_recharge_change.py`, `model_benchmarking_T_change.py` | |

---

## 10. Recommended workflow for common tasks

### "The transient unconfined replay is slow / fails"

1. Check whether strict or practical acceptance fired in the replay summary
   JSON (`strict_picard_convergence_passed` per period). Strict is the
   production gate; practical firing before outer iteration 20 means custom
   controls overrode the corrected defaults.
2. Run `working_tests/run_transient_unconfined_diagnostics.py` for the
   canonical verdict on solver-machinery failures.
3. Inspect `mass_balance_class`, per-period inner-cycle counts, and
   `adaptive_dt_retry_count`/`adaptive_dt_substep_count` — nonzero retries mean
   strict genuinely failed somewhere and the safety net engaged.

### "I need to change a storage formulation"

1. Edit `working_tests/transient_replay_storage.py`.
2. Note the production replay rejects non-default combinations; use lower-level `solver.solve(...)` or `solve_transient_2d_unconfined(...)` directly.
3. Update `transient_replay_settings.py` and tests in `test_2d_transient_warm_start.py`.

### "I need to touch the 3D solver"

1. Kernels go in `kernels_3d.py`.
2. Algorithm logic in `solvers_3d.py`.
3. Wrapper in `warped_darcy_3d.py`.
4. Test in `tests/test_3d_solver.py`.
5. Be aware 3D transient unconfined is not fully validated.

### "I need to change convergence behaviour"

1. 2D: controls in `warped_darcy.py` (search `hclose`, `practical_picard_acceptance_enabled`, adaptive controller config).
2. Replay-level defaults: `transient_replay_settings.py::default_solve_controls`.
3. Acceptance thresholds: `transient_replay_reporting.py`.

---

## 11. Quick facts to avoid re-learning

- **Default solver for production replay**: K-cycle, Chebyshev smoothing.
- **Production storage mode**: `mf6_convertible_secant_sy`.
- **Production warm start**: `unconfined_steady_mf6`.
- **GHB on transient unconfined device fast path**: explicitly `NotImplementedError`.
- **PCG + transient**: explicitly `NotImplementedError`.
- **Steady-state no-storage kernels exist** and are not polluted with storage terms.
- **Coarse operators are approximate preconditioners**, not exact Galerkin representations.
- **Mass balance target**: cumulative < 0.1 %, worst non-startup period < 0.01 %, startup period < 0.2 % — met with class `excellent` on both production grids as of 2026-07-19.
- **Head accuracy target**: final RMSE < 0.001 m, final max abs diff < 0.005 m, worst-period RMSE < 0.005 m — met (1000x1000 30w: RMSE 5.5e-05; 500x500 52w: RMSE 1.2e-05).
- **Production convergence**: strict Picard on all periods; practical acceptance is fallback-only (`min_practical_outer_iterations=20`).
