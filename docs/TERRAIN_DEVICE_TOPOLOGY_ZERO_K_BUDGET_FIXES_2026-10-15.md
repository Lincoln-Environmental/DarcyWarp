# Terrain-device topology, zero-K and host-budget fixes

Starting commit: `ccb985e51aa0e51891209a64ccc8656a20c35aaa`, the current `dev` and `origin/dev` when work began. Work branch: `fix/terrain-device-topology-zero-k-budget`. The checkout already contained uncommitted implementations of these three fixes; they were preserved, reviewed and independently verified. Additional regressions cover active-column reactivation and numerical geometry updates with graph reuse. Integration into `dev` and pushing were subsequently authorized by the user.

Changed files:

- `DARCY_WARP_PACKAGE/solvers/terrain_fast_3d.py`
- `DARCY_WARP_PACKAGE/solvers/terrain_device_kernels.py`
- `tests/test_terrain_device_picard.py`
- `tests/test_3d_terrain_gpu.py`
- `working_tests/run_terrain_cpu_tests.py`
- This report.

The independent reference/budget code, saturation floor, constitutive law, contacts, storage formulation, nonlinear acceptance criteria, final tolerances and Cartesian implementations are unchanged. No packages were installed or upgraded. Unrelated files were preserved.

## Findings and implementation

**1. Topology reuse — confirmed on actual Warp CPU and CUDA.** On the pinned baseline, an isolated positive-K cell retained its 7 m head after adding a valid physical contact whose equilibrium is 4 m. Geometry updates also left stale installed active masks. The device reuse decision checked solver controls but did not ensure that the installed full-operator masks, multigrid hierarchy and captured graph remained compatible.

`workspace_topology_matches()` now shares the host path's definition: the fixed/isolated masks, active masks and positive x/y/vertical face connectivity must match the last installed snapshot. Missing metadata forces rebuilding safely. Incompatible workspaces are closed and replaced before creating the new device state; their captured graph and Krylov operator references are cleared. Unchanged topology allows numeric refresh in existing buffers. Final snapshot masks and connectivity remain synchronized with the installed state. `workspace_reused` reports false after rebuilding.

Regressions:

- `test_device_workspace_rebuilds_when_contact_connects_isolated_cell`
- `test_device_workspace_topology_update_and_numeric_reuse`, with `active`, `reactivate`, `geometry`, `source` and `missing_snapshot` cases.

Each runs on CPU and CUDA. Checks include installed full-operator masks, snapshot connectivity, fine/coarse masks, prescribed/excluded heads, independently reconstructed physical budgets and fresh-session equivalence. CUDA cases require capture/replay, invalidation on replacement, and identical graph identity with increasing replay counts for source and numerical-geometry reuse. No conductivity-update API was added.

**2. Zero-K arithmetic — confirmed on actual Warp CPU and CUDA.** Reciprocal weights generated NaNs for zero/positive and zero/zero endpoints, including vertical faces; the mirrored isolated-cell solve failed. Positive strong contrasts also exposed lost tangential weights from subtracting a nearly unit weight from one.

The explicitly named Warp function `terrain_harmonic_resistance()` matches the independent reference's scaled-ratio conductance and both resistance weights. An explicit branch avoids singular arithmetic. All three directions use it; anisotropy, saturation fraction, slopes, metric defects and active-face handling retain their original equations. No epsilon conductivity, NaN clipping or validation relaxation was introduced.

Regressions:

- `test_device_face_coefficients_zero_safe_reference_parity`: actual kernels on CPU/CUDA, x/y/vertical faces, positive/zero, zero/positive, zero/zero, ordinary positive values, both orientations of a 1e20 contrast, and representable subnormal values. All 13 coefficients must be finite and reference-equivalent. Positive cases include partial saturation, anisotropy and nonparallel interfaces.
- `test_isolated_exclusion_preserves_head_and_normalizes_only_free_equations[True]`: mirrored `kx=[0,1,1]`, with the physical constraint on the connected right portion.
- `test_mixed_zero_material_with_contact_remains_rejected`, alongside existing nonflat mixed-zero rejection coverage.

**3. Host Krylov budget — confirmed on actual CUDA and CPU execution.** A device solve left `_maxiter=8`; the following same-key host invocation requested 13 but received 8. The persistent Krylov object retained the device controller's temporary adaptive-block cap.

The host driver explicitly establishes `_maxiter=int(max_cycles)` beside its tolerance controls before invoking Krylov. Zero-budget calls still skip Krylov. Device per-outer budgets, adaptive blocks and retries are unchanged.

`test_histories_budgets_and_host_device_handoff` now proves the final device cap is shortened, records every following host invocation's cap, requires 13, confirms actual unchanged-topology workspace reuse and compares converged results against a fresh host session. Existing exact-budget, nonmultiple-restart, rollback, retry-accounting and zero-work tests remain passing.

## Reproduction and validation

Environment used: `/home/patrickdurney/miniconda3/envs/darcywarp/bin/python`, Python **3.14.7**, Warp **1.17.0**, FP64. Devices: Warp `cpu` (x86_64) and `cuda:0` (NVIDIA RTX 4070 Ti SUPER, driver 595.91.07).

All commands ran from the repository root unless the isolated baseline directory is specified:

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
```

The clean baseline was exported with `git archive ccb985e51aa0e51891209a64ccc8656a20c35aaa` into `/tmp/darcy-terrain-verify-YsDbXw`. Only the new regression files and CPU test runner were copied into that directory; production solver code stayed pinned. The focused command was run there before accepting the fixes, then on the fixed checkout with identical assertions:

```bash
/home/patrickdurney/miniconda3/envs/darcywarp/bin/python -m pytest -q tests/test_terrain_device_picard.py tests/test_3d_terrain_gpu.py -k 'device_workspace or zero_safe_reference_parity or histories_budgets_and_host_device_handoff or isolated_exclusion or mixed_zero_material_with_contact' --tb=short
```

| Check | Passed | Failed/errors | Skipped | Deselected |
|---|---:|---:|---:|---:|
| Pinned baseline, focused regressions | 18 | 40 failed | 0 | 74 |
| Fixed checkout, same focused regressions | 58 | 0 | 0 | 74 |
| Requested six-file suite, CUDA selected | 278 | 0 | 0 | 0 |
| Requested six-file suite, CPU selected | 274 | 0 | 4 | 0 |
| Additional affected tests | 71 | 0 | 0 | 0 |

A separate pinned-baseline CPU handoff reproduction also failed as expected, with **1 failed, 31 deselected**, showing 8 supplied instead of 13:

```bash
/home/patrickdurney/miniconda3/envs/darcywarp/bin/python working_tests/run_terrain_cpu_tests.py -q tests/test_terrain_device_picard.py -k histories_budgets_and_host_device_handoff --tb=short
```

Exact successful suite commands:

```bash
/home/patrickdurney/miniconda3/envs/darcywarp/bin/python -m pytest -q tests/test_terrain_device_picard.py tests/test_3d_terrain_gmres.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_reference.py tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py --tb=short

/home/patrickdurney/miniconda3/envs/darcywarp/bin/python working_tests/run_terrain_cpu_tests.py -q tests/test_terrain_device_picard.py tests/test_3d_terrain_gmres.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_reference.py tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py --tb=short -rs

/home/patrickdurney/miniconda3/envs/darcywarp/bin/python -m pytest -q tests/test_3d_hierarchy_scientific_audit.py tests/test_3d_solver.py tests/test_3d_physics_validation.py --tb=short
```

The CPU runner selects the shared default test backend before collection. Explicit CPU/CUDA parametrizations still execute their requested backend; it does not globally disable graph caching. Four existing CUDA-only tests skip under CPU selection: graph accounting, device Picard volume parity, nonparallel drying/rewetting parity and coarse refresh. All four execute and pass in the CUDA-selected run. The existing SHANZ 32 regression passes with each default backend; no new convergence ladder or performance campaign was run.

Validation logs for this verification are `/tmp/darcy-terrain-resume-{baseline,targeted,suite-cuda,suite-cpu,additional,baseline-cpu-budget}.log`.

## Checks and remaining limits

These checks pass:

```bash
/home/patrickdurney/miniconda3/envs/darcywarp/bin/ruff check --select E9,F821,F822,F823 DARCY_WARP_PACKAGE/solvers/terrain_fast_3d.py DARCY_WARP_PACKAGE/solvers/terrain_device_kernels.py tests/test_terrain_device_picard.py tests/test_3d_terrain_gpu.py working_tests/run_terrain_cpu_tests.py
/home/patrickdurney/miniconda3/envs/darcywarp/bin/ruff check working_tests/run_terrain_cpu_tests.py
/home/patrickdurney/miniconda3/envs/darcywarp/bin/ruff format --check working_tests/run_terrain_cpu_tests.py
/home/patrickdurney/miniconda3/envs/darcywarp/bin/python -m compileall -q DARCY_WARP_PACKAGE/solvers/terrain_fast_3d.py DARCY_WARP_PACKAGE/solvers/terrain_device_kernels.py tests/test_terrain_device_picard.py tests/test_3d_terrain_gpu.py working_tests/run_terrain_cpu_tests.py
git diff --check
```

Full style checks were also executed:

```bash
/home/patrickdurney/miniconda3/envs/darcywarp/bin/ruff check DARCY_WARP_PACKAGE/solvers/terrain_fast_3d.py DARCY_WARP_PACKAGE/solvers/terrain_device_kernels.py tests/test_terrain_device_picard.py tests/test_3d_terrain_gpu.py
/home/patrickdurney/miniconda3/envs/darcywarp/bin/ruff format --check DARCY_WARP_PACKAGE/solvers/terrain_fast_3d.py DARCY_WARP_PACKAGE/solvers/terrain_device_kernels.py tests/test_terrain_device_picard.py tests/test_3d_terrain_gpu.py
```

They report **59 pre-existing lint findings** and **4 files requiring formatting**, respectively, on both baseline and fixed files. A comparison by rule, message and source line found **zero newly introduced lint findings** (baseline/current counts: 14/14 fast solver, 1/1 kernels, 18/18 Picard tests, 26/26 GPU tests). Broad legacy reformatting was avoided. The new runner passes full lint and formatting checks. The diff contains no new lambdas or unrelated changes.

The documented-version `darcywarp312` environment was located and inspected: Python 3.12.14 / Warp 1.11.0. The same focused command with that interpreter produced **2 collection errors**, because pandas is missing. Direct inspection also shows that Warp 1.11.0 lacks the GMRES API already imported by the pinned baseline. No packages were changed. Validation on that documented toolchain is therefore unavailable; passing results are specifically for the installed Warp 1.17.0 environment. Each working-environment run emitted two existing Warp ctypes deprecation warnings.

No functional failures remain in the executed tests on Warp 1.17.0. Full-repository testing and a new full convergence ladder were not performed.
