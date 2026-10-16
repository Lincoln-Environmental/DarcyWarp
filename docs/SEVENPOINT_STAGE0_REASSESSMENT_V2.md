# Stage 0: independent validation repair and offline reassessment v2

Starting local dev: `296ecd61663613eb1ccf8de7d029f079e9047f88`. Work is isolated on `implementation/mf662-sevenpoint`; no reset, merge, push, PR, workflow change or hosted execution. The original checkout has unrelated untracked files listed in `validation_artifacts/sevenpoint_v1/stage0/starting_worktree_status.txt`. No tracked differences existed initially. Historical terrain equations and evidence are unchanged.

The existing Python3.12.14 environment has Warp1.11.0 and lacks FloPy; it cannot run the reviewed GMRES implementation. The unchanged existing darcywarp interpreter (Python3.14.7, Warp1.17.0, NumPy2.3.5, SciPy1.16.3, FloPy3.9.5) is used, with DARCY_FLOAT=float64 before imports. No environment was modified. Local MF6 `/bin/modflow/mf6` is 6.6.2, SHA256 `21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176`. Exact metadata is saved in stage0/runtime.json.

Five real regression cases failed on the initial source. After repair and two additional guards, the new and existing independent CPU/native MF6 tests pass: **37 passed, zero skipped**. The shifted `12,13,14` field remains exactly balanced but is rejected: eight prescribed cells, maximum absolute error10m, tolerance1e-10m. An elevated datum does not increase this tolerance. Supplied heads are never repaired before checking.

Active invalid heads are rejected. Inactive NaNs/1e30 markers are retained in raw evidence and excluded from statistics. Only an explicitly saved arithmetic evaluation representation substitutes finite geometric centres on inactive cells; this does not assign those cells a physical solution. Empty free layers/subsets report zero counts and null extrema/RMS. Zero residual for an empty equation set is vacuous, not physical validation.

A **live masked DIS model** verifies that 6.6.2 writes full user-grid GRB IAUSR/JAUSR, with empty IA rows on inactive cells; CBC package and DATA-SAT node indices are also user-grid indices. Internal reduced-to-user mapping is saved separately. Empty IA rows must not be treated as diagonal residual slots. Actual source placement beyond the hole, flux signs, exact source allocation and raw inactive markers are tested and preserved. The first native run passed its own loose stopping criteria but failed independent conservation (relative1.43e-7); the reference test explicitly tightened native head/linear residual controls to1e-12. The physical acceptance gate was not loosened.

Native contacts now require explicit identities: `original_convertible_drn`, `confined_contact_control`, `native_head_contacts`, `custom_contact_tpfa`, `custom_contact_xt3d`, or `none`. The fresh reproduction driver executes BOTH native DRN identities in different directories, followed by distinct custom modes. A durable small live run reproduces original below-bottom DRN rejection and the intentionally altered confined-owning-cell control's success. No outlet stage is moved. Implicit contact selection raises a useful error.

The versioned offline reassessment evaluates **360** saved candidate heads, including61 recovered directly from verified archives, with no nonlinear updates. All360 retain valid original prescribed heads;212 pass the corrected DarcyWarp discrete acceptance gate. The other148 include failures, intermediate feedback states and different-formulation MF6 heads, not newly certified roots. No candidate was unavailable/unreadable. All14 original polished GPU Picard/Newton candidates across seven cases remain independently accepted solutions of their respective original boundary-value problems. Thus the prior same-equation branch comparisons survive this repair. Their physical branch selection remains unresolved. Native/coupled MF6 heads remain formulation-different; evaluating the DarcyWarp residual at them does not assert MF6 equation compatibility.

Commands (repository root, existing interpreter):

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
export OPENBLAS_NUM_THREADS=1
PY=/home/patrickdurney/miniconda3/envs/darcywarp/bin/python
"$PY" -m pytest -q tests/test_validation_repairs_3d.py tests/test_terrain_cpu_validation.py tests/test_terrain_mf6_validation.py
"$PY" working_tests/validate_stage0_examples.py
"$PY" working_tests/reassess_terrain_validation_v2.py
```

Evidence: `validation_artifacts/sevenpoint_v1/stage0/`. `before.log` retains the five initial failures; `after.log` the37 passes. Examples have exact canonical arrays, native decks, listing/CSV/head/budget/GRB output and corrected checks. `reassessment_v2.json` lists every candidate, physical-input/head hashes, boundary validity, discrete convergence, conservation, formulation classification and unresolved physical status. Scripts use fresh example directories; change the root to repeat rather than overwriting original evidence.

**Stage0 gate passes on available CPU/MF6 backends.** There is no production formulation change in this milestone. The subsequent confined gate must establish operator and supplied-head parity before any matched unconfined law is added.
