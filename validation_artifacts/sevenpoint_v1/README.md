# Local seven-point evidence, version 1

Scientific conclusions and stage gates: [programme report](../../docs/SEVENPOINT_PROGRAMME_VALIDATION_V1.md). This bundle supports bounded numerical parity, not physical certification. All changes and execution remain local. No existing terrain artifact was replaced.

Authoritative final results:

- `stage0/reassessment_v2.json`:360 original saved candidates, no nonlinear updates. `before.log` retains five initial defects; `final_followup.log` records37 passing validation tests. `examples/` includes the shifted-boundary rejection and live masked/contact outputs.
- `stage1/final_experiments/gate.json`:8 confined controls, distinct SciPy/WarpCPU/CUDA fields, eight live MF6 root and eight supplied-state models. The earlier `experiments/` gate predates Stage2 and establishes execution order, but its `cpu.npz` naming collision overwrote the direct CPU field. Use the final repeat for numerical field evidence; original metrics remain preserved.
- `stage2/final_experiments/small/gate.json`:7 matched unconfined controls,7 supplied-state models, derivative sweep, native saturation/source/flux checks.
- `stage2/final_experiments/catchment/gate.json`:distinct flat4x24x32 catchment representation after the small gate. `geometry_mapping.json` and `original_geometry.npz` declare changes from the existing builder; it is not the original terrain problem.
- `final_recheck_v2/returned_head_checks.json`:independent final recomputation of all64 returned fields, source hashes, complete runtime/CPU/BLAS metadata, and verification of2,432 unchanged historical files. No solution advanced. `measured_source/` preserves exact source bytes matching the successful unconfined gate; the subsequent invalid-CHD input guard and final EOF whitespace cleanup are explicitly identified. The guard is independently tested, and all positive fields rechecked against the delivered source. The earlier v1 recheck is retained.

Each case has canonical input arrays and stable input hash, saved head/physical fields, explicit acceptance/BC metrics, native decks/IMS options, full logs/listings/CSV iteration records, HEAD/CBC/GRB binaries and physical budget reconstruction. Methods use separate files. Raw inactive markers are retained; evaluation representations are separately labelled. No result is reused merely because a filename exists.

Large new native iteration listings/CSVs/logs are losslessly packaged in `archives/stage2_native_iterations.tar.xz`. `archives/index.json` records each original relative path and SHA256 and is compatible with existing `validation.evidence_storage.verify_archive/restore_archive`. To restore, set `restore=True` in `working_tests/archive_sevenpoint_logs.py` and run it locally. The writer verifies every archived member before removing the redundant new loose copy; no historical artifact is archived or removed by this script. `delivery_manifest_v2.json` seals the packaged representation and source/syntax/staged-diff checks. The first manifest describes the prepackaged payloads and remains immutable; all of its moved records are recoverable byte-for-byte through the archive index.

Retained negative/intermediate experiments:

- `stage1/layered_*` and initial test logs: native success without sufficiently accurate independent conservation, followed by one-factor reference linear-solver controls.
- `stage2/tests_initial.log`:two derivative fixtures accidentally crossing upstream ties; corrected smooth fixtures retain the same tolerance, with separate selected-branch tie tests.
- `stage2/dry_before_safeguard/`:initial direct CPU/device failure at zero Jacobian row with nonzero source; native MF6 recovery. `dry_after_safeguard/` and final small matrix show recovery with unchanged equations. `dry_initial/` is an earlier3-outer diagnostic, not a completed convergence claim.
- `stage2/experiments/catchment/interrupted.json`:the first native catchment run interrupted after a head-change roundoff stall. Incomplete files are evidence, not success. `catchment_roundoff_control/` tests the input-scaled iteration setting; the final campaign repeats it.
- `stage2/invalid_chd_before_guard/`:convertible CHD below bottom; CPU was initially permissive and MF6 rejects the model. The new common input guard corrects that discrepancy. The original negative deck can be rerun directly with the pinned executable; the delivered adapter now rejects it before writing a new model.

Exact executed command groups from the repository root, unchanged existing environment:

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
export OPENBLAS_NUM_THREADS=1
PY=/home/patrickdurney/miniconda3/envs/darcywarp/bin/python

# Stage 0, before then after repair; logs preserve both outcomes.
"$PY" -m pytest -q tests/test_validation_repairs_3d.py tests/test_terrain_cpu_validation.py tests/test_terrain_mf6_validation.py
"$PY" working_tests/validate_stage0_examples.py
"$PY" working_tests/reassess_terrain_validation_v2.py

# Stage 1 before Stage 2, then a final field-preservation repeat.
"$PY" -m pytest -q tests/test_npf662_confined.py
"$PY" working_tests/validate_npf662_confined.py
"$PY" -m pytest -q tests/test_3d_solver.py tests/test_3d_physics_validation.py tests/test_fast_3d_kcycle.py tests/test_3d_session.py tests/test_3d_operator_analytical.py tests/test_3d_face_kernels.py tests/test_3d_scientific_audit.py tests/test_terrain_cpu_validation.py tests/test_validation_repairs_3d.py tests/test_3d_terrain_gpu.py

# Stage 2 focused tests and complete local matrices.
"$PY" -m pytest -q tests/test_npf662_confined.py tests/test_npf662_newton.py
"$PY" working_tests/validate_npf662_dry_start.py
"$PY" working_tests/validate_npf662_unconfined.py
"$PY" -m pytest -q tests/test_3d*.py tests/test_fast_3d*.py tests/test_terrain*.py tests/test_npf662*.py tests/test_validation_repairs_3d.py
"$PY" -m pytest -q tests/test_npf662_confined.py tests/test_npf662_newton.py tests/test_validation_repairs_3d.py
"$PY" working_tests/recheck_sevenpoint_delivery.py

# Final local syntax/style/evidence checks.
"$PY" working_tests/archive_sevenpoint_logs.py
"$PY" working_tests/seal_sevenpoint_evidence.py
git diff --check
```

Directory configuration lives under each standalone script's execution guard. Existing completed outputs are deliberately immutable: select a **fresh** directory to rerun. Dry before/after experiments used the same initial=-.2m, source=1m³/day, max_outer=200, shared Krylov2000; the earliest dry diagnostic used max_outer3/Krylov24 and is explicitly incomplete. Confined matrix uses shape3x5x7, spacing2/3/4m, K contrast1e-3–1e3 and seed1966. Unconfined uses seed1937, max_outer200, shared Krylov2000, and both local devices. Exact per-run physical/numerical arrays/options are saved, not reconstructed from this prose.

Native-only failures or reference-control repeats can be reproduced by running `/bin/modflow/mf6` from a **copy** of their saved model directory (preserve originals). The executable and every NPF/IMS/package setting are pinned in each summary/deck. There is no hidden storage period, initial-condition conductance fit, terrain warm start or reference-boundary fitting.

Observed test groups overlap:37 Stage0 passes;21 initial confined passes;292 existing affected passes;1,370 broad affected passes;62 final focused passes after the CHD guard. All have zero skips. Two warnings in the latter groups originate in Warp's Python3.14 ctypes deprecation, not failed scientific checks. The full unrelated2D suite was not run. Failed/incomplete native experiments remain labelled separately from model-definition rejection and implementation comparisons.
