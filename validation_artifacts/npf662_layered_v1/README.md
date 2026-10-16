# NPF662 layered DIS campaign V1

Report: [local validation and technical handoff](../../docs/NPF662_LAYERED_DIS_VALIDATION_V2.md). Contract: [layered DIS V2](../../docs/NPF662_LAYERED_DIS_REFERENCE_CONTRACT_V2.md). These are bounded reference-equation checks, not physical-model certification. No old artifact is replaced; all changes remain local on `implementation/npf662-layered-dis`.

Evidence identities:

- `part_a/`:28 failed/1 passed initial assertions, raw before/after NPZ examples,94 passing repaired/NPF tests, unchanged64-field V1 reassessment, actual runtime and source hashes. Before examples use unchanged8352de9 source from the earlier worktree and the same interpreter. Raw infinities remain in NPZ; null in JSON is serialization after validity decisions.
- `contract/`:official tagged6.6.2 `Connections.f90` plus two preimplementation native supplied-head models proving the connection formulas on differing/nonoverlapping lateral elevations. Other tagged constitutive sources remain in the original terrain source bundle, unmodified.
- `confined/`:one initial one-column fixture indexing failure, corrected124-test group,30-test lifecycle closure, and five complete four-method controls in `experiments/`. Exact C0/faces and supplied-head residuals are independently checked. Scalar-dz and explicit uniform interfaces agree; V1 input hashes/loads are preserved.
- `unconfined/`:initial invalid convertible-CHD fixture rejections (13 failures/8 passes), valid separate fixtures passing21 tests,148-test combined group and six complete controls in `experiments/`. No solver repairs invalid prescribed values. The valid convertible controls have declared9/10m end heads; the earlier confined controls have8/9m. They are distinct canonical cases, not one claimed nonlinear-law-only boundary comparison.
- `geology_4x4/`:complete four-method comparison using exact independently varying synthetic builder geology, CHD/WEL and no contacts. All cold methods converge and compare within the original gates.
- `geology/`:additional6x8 geology experiment. Independent CPU cold solve fails at200 outers. Device/MF6 fields pass equation/conservation acceptance, but their max head difference1.091e-7m fails the predeclared1e-7m comparison. This case remains failed overall.
- `final_assessment/`:offline independent checks for52 new fields (51 accepted), exact hashes, same-head native comparisons, all solver-pair head/flux statistics, actual device Jv/flux/residual probe, finite-difference sweep, final unchanged64-field V1 reassessment and runtime. `geology_failure/` contains **verification-only** perturbed CPU restart and stricter independent direct polishing; they do not replace cold inputs/results or change production tolerances.
- `legacy_archive_reassessment_v2/`:210 accessible earlier seven-point fields (171 nonverification/39 constitutive),0 changed recorded statuses,94 unrecorded prior statuses left unknown,0 unreadable fields. The known invalid pre-guard CHD input and interrupted native run without a returned field are explicit. The initial broader-archive helper's missing-json-import error is retained separately and corrected, without advancing any head. Early12-field implicit-confined V1 hashes/roundtrips are preserved.
- `affected_regressions.log`:1,457 affected passes; `final_affected.log/xml`:1,461 passes; `final_focused_v4.log/xml`:153 passes after final schema/connectivity follow-ups. All have0 skips; groups overlap. The two warnings are existing Warp/Python3.14 ctypes deprecations.
- `delivery_manifest.json`:final source hashes, local syntax/no-lambda and diff checks, preserved original-checkout status, and every packaged evidence file checksum.

Each model directory preserves canonical schema/input hash, initial and returned heads, physical fields, budgets, summaries, full NPF/IMS/package inputs, raw HEAD/CBC/GRB outputs and source snapshots. Individual methods have distinct filenames. Large new unconfined/geology native listings/CSVs/stdout are packaged losslessly under `archives/` using the existing verified writer/reader; `archives/index.json` records every original relative path and hash. `validation.evidence_storage.restore_archive(root=...,record=...)` restores exact original files. Condensed scalar summaries are never substituted for those iteration records. Earlier small confined/native records stay loose.

All computations used the existing `/home/patrickdurney/miniconda3/envs/darcywarp/bin/python`, with no environment modifications:

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
export OPENBLAS_NUM_THREADS=1
PY=/home/patrickdurney/miniconda3/envs/darcywarp/bin/python

# Before/after acceptance; same new assertions, no geometry changes yet.
"$PY" -m pytest -q tests/test_npf662_acceptance.py
"$PY" -m pytest -q tests/test_npf662_acceptance.py tests/test_npf662_confined.py tests/test_npf662_newton.py tests/test_validation_repairs_3d.py
"$PY" working_tests/validate_npf662_acceptance.py

# Contract verification before production geometry implementation.
"$PY" working_tests/verify_npf662_layered_contract.py

# Confined milestone: configure the layered driver's guard with phase='confined',
# root='validation_artifacts/npf662_layered_v1/confined/experiments', no geology call.
"$PY" -m pytest -q tests/test_npf662_layers.py tests/test_npf662_acceptance.py tests/test_npf662_confined.py tests/test_npf662_newton.py tests/test_validation_repairs_3d.py
"$PY" working_tests/validate_npf662_layered.py

# After the confined commit, unconfined activation and explicit geology runs.
"$PY" -m pytest -q tests/test_npf662_layered_unconfined.py
"$PY" working_tests/validate_npf662_layered.py
"$PY" -m pytest -q tests/test_3d*.py tests/test_fast_3d*.py tests/test_terrain*.py tests/test_npf662*.py tests/test_validation_repairs_3d.py
"$PY" working_tests/assess_npf662_layered_results.py
"$PY" -m pytest -q tests/test_npf662_acceptance.py tests/test_npf662_confined.py tests/test_npf662_newton.py tests/test_npf662_layers.py tests/test_npf662_layered_unconfined.py tests/test_validation_repairs_3d.py --junitxml=validation_artifacts/npf662_layered_v1/final_focused_v4.xml
"$PY" working_tests/seal_npf662_layered_delivery.py
git diff --check
```

Output roots are configured under execution guards and immutable: select fresh directories to rerun. The initial unconfined driver used geology shape6x8/root`geology`; a separately labelled direct `run_geology_case(root=Path('validation_artifacts/npf662_layered_v1/geology_4x4'),shape=(4,4),dx=20.,devices=['cpu','cuda:0'],executable=Path('/bin/modflow/mf6'),max_outer=200,krylov_budget=2000,seed=2231)` call produced the smaller case. The delivered driver defaults to that successful modest4x4 geometry and raises if its overall gate fails; the original6x8 failure is never hidden by its successful executable exit. Both use the exact original geology constructor, with the explicitly documented contact-free DIS boundary assumptions. This fixed-dx size change is not spatial refinement.

The measured FP64 case equations did not change after the numerical campaigns; final follow-ups add input-shape/precision checks, old-schema preservation, exact native adjacency validation, docstrings, tests and runner output/error handling. Actual source hashes in each gate identify the measured tree, and saved fields are evaluated against the delivered valid-state equations. The broader offline call is `reassess_older_fields(original=Path('validation_artifacts/sevenpoint_v1'),output=Path('validation_artifacts/npf662_layered_v1/legacy_archive_reassessment_v2'))` from `validate_npf662_acceptance`, run with the same interpreter/environment. No cache reuse, warm-start substitution, looser acceptance or altered geometry is presented as a repair to an original experiment.

The default diagonal preconditioner and controller are retained. The 6x8 CPU cold failure and native/device head-comparison miss remain limitations. An independent accurate polish supports near-root numerical error, not a production fix or uniqueness proof. No K-cycle integration, new boundary law, general terrain equivalence, storage/surface validation or regional speedup is claimed.
