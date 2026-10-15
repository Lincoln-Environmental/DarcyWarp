# Terrain validation evidence, 16 October 2026

Read the [scientific assessment](../../docs/TERRAIN_CPU_MF6_VALIDATION_2026-10-16.md) and [equations/methods](../../docs/TERRAIN_CPU_MF6_METHODS_2026-10-16.md) first. This bundle documents different discrete roots and formulation differences; it is **not model certification**. Failed n32 XT3D and n96 cold CPU experiments remain failures.

The fetched starting dev was `377ada20a92ad5f5a40af1ed9844cd8be31beba4`. Implementation milestones are in git, the staged history in [WORK_LOG.md](WORK_LOG.md), and the sealed source revision/file hashes in `delivery_manifest.json`. Experiments made during development identify their actual source revision where available; do not silently relabel them as final-revision runs. The final cold CPU repetitions were executed at `99b78ed`, before later finite-state rejection/IO/cache guards. No finite benchmark arithmetic changed in those guards. GPU before/after runs straddle the three separately committed production fixes; the ten-run comparison is in `before_after_trajectory_checks.json`. Historical experiments predating the strict request registry are evidence, **not valid cache entries**.

## Runtime and commands

The unchanged existing interpreter was `/home/patrickdurney/miniconda3/envs/darcywarp/bin/python`, Python 3.14.7, NumPy 2.3.5, SciPy 1.16.3, Warp 1.17.0, FloPy 3.9.5. `runtime.json`, `blas_runtime.json`, and `conda_explicit.txt` record the OS/CPU/BLAS/GPU and exact environment. No existing environment was upgraded. The existing MF6 6.6.2 executable was `/bin/modflow/mf6`, SHA256 `21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176`. Its binary build revision is unavailable; official tag source is identified separately in `mf6_source/README.md`.

All commands run from the repository root with:

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
PY=/home/patrickdurney/miniconda3/envs/darcywarp/bin/python
```

These are the final reference CPU BLAS thread controls; earlier attempts have their recorded environment/request metadata. Hardware was shared with other scientific jobs and occasional external GPU activity. Times are work records, not controlled speedup claims. `commands.txt` records the baseline, experiment entry points and test commands. User-adjustable options are directly inside each script's execution guard; there is no new CLI.

Verify the delivered files, source, all archived members, and before/after GPU decisions without running a solver:

```bash
"$PY" working_tests/verify_terrain_delivery.py
```

Run a **fresh**, restart-checked seven-case cold campaign:

```bash
"$PY" working_tests/execute_terrain_validation_programme.py
```

This writes `validation_artifacts/terrain_reproduction`, uses original 2,000-outer GPU caps and separately labelled longer Picard references, CPU reference caps, native MF6 and both contact-coupled MF6 modes. Choose a fresh `root` in its guard when runtime/source/settings change. A matching completed run is checksum-checked; interrupted, unversioned, stale or corrupted runs raise explicit errors. Native DRN rejection is preserved and does not stop the separately labelled custom-contact experiments. Numerical failure is a recorded result, not a cache failure. No CPU solver requires CUDA; the combined campaign includes GPU runs. A CPU-only invocation using serialized inputs is shown in the methods report.

The historical drivers target this immutable dated bundle and some intentionally skip existing directories. Do not use those skips as validated caching or overwrite these outputs. To repeat a diagnostic, select a fresh root in the guard and first copy/save its canonical case inputs. Branch/reduced tests require the saved predecessor heads as specified in their functions. No undocumented `/tmp` data is needed.

## Evidence map

| Location | Meaning |
|---|---|
| `cases/*/{manifest.json,inputs.npz}` | Exact canonical physical arrays, controls, coordinate/unit/source conventions and input hashes |
| `before/` | Original harness capped failure; no changed main routine or warm start |
| `gpu_before/`, `gpu_after/` | Cold capped baselines and separately labelled references; heads and complete compressed histories |
| `cpu_final/` | Final committed-source CPU cold references, settings/source/runtime requests, heads and complete histories |
| `cpu*/`, `development/` | Earlier policies, failures and explicitly interrupted attempts; not promoted to successes |
| `branches/`, `extra_branches/`, `extra_picard_branches/`, `gpu_crossstarts/` | Strict root polishing, deterministic perturbations and verification-only cross-starts |
| `cpu_branch_reductions/`, `reduced_picard/` | Explicitly changed constrained boundary reductions and direct Picard experiments, including failed reductions |
| `scalar_roots/` | Analytically solvable one-free-cell three-root control, CPU/GPU solutions and different-law live MF6 result |
| `mf6/`, `mf6_feedback_v3/`, `mf6_feedback_xt3d/` | Native/TPFA-coupled/XT3D-coupled run summaries and final heads; raw models in archives |
| `mf6_initial_shift/`, `mf6_solver_recovery/`, `mf6_n32_angle_control/` | Labelled initial-guess, effort/globalization and connection-geometry experiments |
| `native_final_fields/` | Independently extracted original native binaries, DATA-SAT, connectivity-derived faces and actual GRB geometry checks; no nonlinear update |
| `analysis/` | Offline returned-head DarcyWarp residuals/budgets, native residual reconstructions, masks, water-table layer selection, faces and integrated comparisons |
| `root_evidence.json`, `validation_matrix.json` | Separate root/conditioning/precision evidence and status dimensions; no single certification flag |
| `run_metrics.{json,csv}`, `native_work.json` | Residuals, work/time and native iteration records; unavailable counters are null |
| `geometry/`, `physics_controls/`, `fixed_boundary_refinement_v2/`, `fully_saturated_affine_xy/` | Analytical/progressive controls, strict fixed-physical-boundary refinement, and nonzero affine discretisation control |
| `formulation_terms.json` | Same-head geometry/floor/contact term sensitivities and interactions, not changed-floor target solutions |
| `mf6_tpfa_and_precision.json` | No-update independent native TPFA residual and extended-precision DarcyWarp flux checks |
| `*.log` | Complete tests/experiments, including retained unsuccessful attempts |
| `archives/index.json` | Whole/part and per-original-file hashes for complete native decks, logs, CSVs, GRB, head and CBC binaries |

The original fixed-dx size ladder changes physical extent. Only the explicitly fixed-domain/boundary experiment is called refinement. Earlier moving-fixed-cell-boundary controls remain labelled controls. The original unconfined model has not been certified through a full physical-domain refinement study.

## Recover raw MF6 models

Raw working files were removed only after checksum-verified lossless archival. Intermediate feedback heads/extractions remain recoverable in the same archives; final fields are directly accessible. XZ files above 40 MB are split into explicitly indexed `.part000`, `.part001`, ... volumes. **Use the index and restoration function**; a lone split part is not an archive. Each part, the reassembled payload, every member and restored file are checked. Source gzip and unsplit duplicate copies were pruned only after successful verification. The split-volume test actually restores without the unsplit source.

To restore one experiment group, run this from the repository root (configuration is inside the execution guard):

```python
import json
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.evidence_storage import restore_archive

if __name__ == "__main__":
    root = Path("validation_artifacts/terrain_2026-10-16")
    original_directory = "mf6_feedback_xt3d/terrain_64_k0.0001"
    records = json.loads((root / "archives/index.json").read_text())
    record = next(item for item in records if item["original_directory"] == original_directory)
    restore_archive(root=root, record=record)
```

For all groups set `mode='restore'` in `working_tests/archive_terrain_validation.py`. Restoration recreates archived bytes at their original relative paths. Copy a restored model into a **new** directory before rerunning `/bin/modflow/mf6`; preserve the original evidence. The model inputs record options and full-precision data, and original listing/inner/outer files retain resolved settings and convergence records.

Offline regeneration without model advancement:

```bash
"$PY" working_tests/extract_terrain_native_evidence.py
"$PY" working_tests/summarize_terrain_cpu_mf6.py
"$PY" working_tests/report_terrain_validation.py
"$PY" working_tests/write_terrain_validation_report.py
```

Extraction reads the archives directly, including split volumes; full restoration is unnecessary. Regeneration can change ZIP container timestamps or environment-dependent diagnostics, so do it in a copy; the sealed delivery checksums describe the original delivered bytes. Figures have saved numerical arrays. The assessment template is versioned in `docs/` and rendered tables come from saved results.

Tests: full repository **1,532 passed** at the production-fix milestone; affected terrain/scientific **1,144 passed**; final focused CPU/MF6/adapter/defect/archive **41 passed**. Counts overlap. These were actual live MF6 tests, not dependency skips. Scope and warnings are retained in the logs and assessment.
