kimi --model kimi-code/k3
# DarcyWarp — Agent System Memory

> Last updated: 2026-07-18 by ingest of the full repo.
> Purpose: prevent re-ingesting the project from scratch. Read this first, then grep/read the specific files listed below.

---

## 1. What this project is

DarcyWarp is a **GPU-accelerated groundwater-flow solver** built on NVIDIA Warp (`warp-lang`). It solves structured-grid Darcy flow (5-point in 2D, 7-point in 3D) and is benchmarked against **MODFLOW 6** via FloPy, plus an optional CPU finite-difference reference.

- Repo root: `/home/patrickdurney/PycharmProjects/DarcyWarp`
- Conda env: `quick_flow_env` (has Warp + CUDA; `darcywarp` also exists but is not the active dev env)
- License: AGPL-3.0-or-later
- Precision: set via `DARCY_FLOAT` env (`float32` | `float64`); default is currently `float64` in `warped_darcy.py` and `float32` in `config.py` — **check both**.
- MF6 binary: discovered in `DARCY_WARP_PACKAGE/project_base.py`; default `/bin/modflow/mf6`, fallback `which mf6`.

The code has two eras:

1. **Original hand-written core**: steady 2D K-cycle/PCG, `model_builder.py`, `sparse_operator.py`, benchmark scaffolding.
2. **AI-expanded layers** (GPT/Codex): transient 2D unconfined, adaptive inner controller, 3D transient unconfined, replay infrastructure, mass-budget diagnostics.

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
  warped_darcy.py        # 2D solver: ~12k lines, monolith
  warped_darcy_3d.py     # thin 3D wrapper class
  solvers_3d.py          # 3D algorithms: K-cycle, Chebyshev, Picard
  kernels_3d.py          # all 3D @wp.kernel definitions
  k_cycle_multigrid.py   # standalone reference K-cycle utility

working_tests/
  run_2d_transient_vs_mf6.py        # MF6 truth artifact generator
  run_2d_transient_warp_replay.py   # production replay + K-cycle optimizer
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

- `WarpDarcySolver.__init__` (~l. 4624)
- `build_from_truth_inputs` (~l. 5730)
- `build_from_fields` (~l. 5901)
- `solve` (~l. 11959) — public dispatcher
- `solve_transient_2d_unconfined` (~l. 10327) — multi-period API
- `solve_multigrid_kcycle` (~l. 8355) — core nonlinear solve
- `update_T_in_place` (~l. 7001) — ensemble T-change fast path

### Solvers

- `solver="pcg"`: steady confined only. `transient=True` raises `NotImplementedError`.
- `solver="kcycle"`: geometric multigrid, supports confined/unconfined, steady/transient.

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
- Optional `adaptive_dt_enabled` (default **False**): per-period sub-stepping with
  strict-first acceptance, shrink-to-`dt_min` and practical fallback at `dt_min`
  (controls `adaptive_dt_*` in `transient_replay_settings.py`). Audited 2026-07-18:
  the driver is mechanically correct (reproduces fixed-small-dt integration; no-op
  on easy problems) but **cannot fix the 1M-cell accuracy failure** — sub-stepping
  converges to the true transient while the MF6 artifact is one backward-Euler step
  per period, so accuracy-vs-MF6 degrades to ~0.1 m RMSE. Kept off by default.
  Mass-balance reporting for sub-stepped runs carries
  `endpoint_flux_budget_approximation: true` (endpoint-flux budget is a metric
  artifact there, not non-conservation). See `TRANSIENT_STATUS.md` § Adaptive
  timestepping.

### Adaptive inner controller

- Enabled by `adaptive_unconfined_inner_enabled=True` (production default).
- Runs K-cycles in blocks, measures residual contraction, and adjusts block size.
- Uses dual residuals: flow residual `b - Ah` and head-equivalent residual `(b - Ah)/diag(A)`.
- Falls back to legacy head-change-driven cycle cap if residual path fails.
- Heavily configurable; defaults in `transient_replay_settings.py`.

### Convergence / acceptance

- Strict Picard: `max_abs_head_change < hclose` AND head-equivalent residual RMS < strict tol.
- Practical acceptance (production gate): min outer iters, head residual < practical tol, `dh_rms` < practical tol, storage-diag change RMS < practical tol.
- Production result is **practical acceptance**, not strict Picard.

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

### Generating truth

```bash
python working_tests/run_2d_transient_vs_mf6.py
```

Produces `DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined_<nx>x<ny>_<n_weeks>w/mf6_transient_heads.npz.lzma`.

Artifact contains: per-period heads, final heads, initial/warm-start heads, masks, DEM, bottom, K, Sy, Ss, recharge rates, provenance.

### Running replay

```bash
python working_tests/run_2d_transient_warp_replay.py
```

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

- **Strict Picard convergence often fails**; runs pass via practical acceptance.
- Root cause identified in `run_transient_unconfined_diagnostics.py`: transient storage diagonal destabilises the iterative solver when large relative to diffusion diagonal.
- 500×500 10-period replay: Warp ~27 s vs MF6 ~106 s (still ~4× faster), but strict Picard failed and period-1 mass balance is elevated (~0.1 %, classed `startup_warning`).

### 3D transient unconfined

- Phreatic-Sy path exists but full validation/benchmarks not run.
- Storage rebuild happens on host per Picard iteration (no device fast path).
- Wetting/drying limited to `min_sat` floor.

### General

- Per-iteration host sync for scalar reductions in PCG/K-cycle convergence checks.
- Memory management: explicit `gc.collect()` calls to break Warp array reference cycles during hierarchy rebuilds.
- `warped_darcy.py` is ~12k lines and heavily branched; small changes can have non-obvious interactions between storage modes, active-set strategies, and adaptive controller fallback.

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
| `test_comparison_results.py` | warp + fixtures | end-to-end Warp vs MF6 truth |

### Important caveats

- Many tests skip if `warp` missing. In this environment `warp` is **not installed**.
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
| 3D kernels | `kernels_3d.py` | all 3D kernels |
| 3D algorithms | `solvers_3d.py` | K-cycle, Chebyshev, Picard |
| Adaptive inner controller | `warped_darcy.py` ~l. 1088-1500+ | `_run_adaptive_inner_kcycle_blocks` |
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

1. Run `working_tests/run_transient_unconfined_diagnostics.py` first — it gives the canonical verdict.
2. Check whether strict or practical acceptance failed in the replay summary JSON.
3. Inspect `mass_balance_class`, `strict_picard_convergence_passed`, `practical_picard_acceptance_passed`.
4. Look at per-period inner-cycle counts; if huge, the adaptive controller or storage-diagonal scaling is the issue.

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
- **Mass balance target**: cumulative < 0.1 %, worst non-startup period < 0.01 %, startup period < 0.2 %.
- **Head accuracy target**: final RMSE < 0.001 m, final max abs diff < 0.005 m, worst-period RMSE < 0.005 m.
