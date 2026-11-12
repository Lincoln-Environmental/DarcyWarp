# Historical Cartesian and NPF/STO GPU qualification: 11 November 2026

> **Historical qualification record.** This report preserves the first
> 3D GPU comparisons following PR #4. Its original 7 to 30-second NPF/STO
> solve-call timings have been superseded by the
> [current GPU history guide](../source/npf_gpu_native.rst):
> 0.972 s versus 104.391 s (convertible) and 1.378 s versus 346.408 s
> (confined) on 500 x 500 x 5 grids for 20 weekly steps. Those are
> specific public history versus saved MF6 engine intervals, not
> end-to-end model-build or independent-audit comparisons.

The report originally qualified `main` revision
`f18912bff7afe5bb696a226cd8095f7a614352bf`
([PR #4](https://github.com/Lincoln-Environmental/DarcyWarp/pull/4))
and was prepared on `docs/3d-gpu-qualification-20261111`.
It is not a new release tag and does not imply all Cartesian cases
converge under the older 2D iteration budgets.

Developers: Patrick Durney and Jaco Fourie. Software ownership: Lincoln
Agritech Ltd. Existing AGPL and adapted NVIDIA Apache-2.0 attribution remain
unchanged. Transformed terrain stays outside the scoped main release.

## Runnable sanity and convergence checks

```bash
conda activate darcywarp
python working_tests/run_3d_sanity.py --device cuda:0 --sizes 16 32
```

The [implementation guide](3D_IMPLEMENTATIONS.rst) documents the portable
runner, cached MF6 references, realistic-size commands and explicit overrides.
Default Cartesian uses `fast`; NPF uses its fast device solver with automatic
preconditioner selection. `--compare` also runs classic Cartesian and both NPF
preconditioners. CPU execution is supported with `--device cpu`.

The runner reads the actual 2D sanity controls and records source hashes.
MF6 residual type is preserved as `RELATIVE_RCLOSE`; it is not replaced by
`STRICT`. NPF's separate fixed physical gates remain unchanged. Reference
validity, strict native NPF audit and GPU convergence are reported separately.
Failed GPU states never advance a physical timestep. Failed reference checks
cannot be mistaken for an early-converged complete GPU history.

Runtime: Python 3.12.14, Warp 1.17.0, FP64, RTX 4070 Ti SUPER 16 GiB;
single-thread OpenBLAS/OMP/MKL. Native executable is MF6 6.6.2, SHA256
`21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176`.


## GPU-native transient follow-up

The subsequent runtime improvements from dev revision `bf59a2a` are also
promoted to main: block residual reductions, device Arnoldi updates and
CUDA graph replay, plus GPU hierarchy value refresh for changing K.
The pre-promotion convertible benchmark measured **5.916 s** per 20-step
history. Paired calibration histories show **28–31% less numerical solver
time**, excluding setup and K updates. See the execution guide for the
measurement scope and saved MF6 parity evidence.

The opt-in `npf_device_native=True` implementation is now on main.
All 320 rerun endpoints passed independent checks; 526 main regressions passed.
The [GPU execution guide](NPF662_GPU_NATIVE.rst) records the updated timings,
controls, conductivity refresh behavior and reproducible commands. The tables
below retain the original implementation baseline.

## Original runtime comparison: solve calls only

Twenty weekly steps at the explicit GPU relative-imbalance limit `1e-7`:

| Model | GPU public solver | MF6 engine | MF6/GPU runtime ratio |
|---|---:|---:|---:|
| 250 × 250 × 5 confined | 7.672 s | 47.589 s | 6.2× |
| 250 × 250 × 5 convertible | 8.109 s | 17.319 s | 2.1× |
| 500 × 500 × 5 confined | 27.636 s | 346.408 s | 12.5× |
| 500 × 500 × 5 convertible | 30.243 s | 104.391 s | 3.5× |

The primary GPU clock sums synchronized public solve calls. It excludes model
construction, `build_from_layers`, reference-cache reads, the extra benchmark
audit and file saving. It retains work performed inside solve, including
internal setup and final acceptance. MF6 engine timing excludes the FloPy
build/write stage. GPU values are three-repeat medians; MF6 values are single
reference-generation engine runs. No repeated MF6 median is claimed.

The extra test audit costs about 4.4 s per 250-grid history and 19.8–20.2 s
per 500-grid history. It is outside this primary timing. The production-facing
solver still contains substantial CPU work: for 500-grid confined, its Newton/
Krylov interval is about 5.3 s within the 27.6 s public solve interval. The
remainder includes host reference assembly, setup, transfers and acceptance;
it has not been attributed entirely to any one of those costs. This motivated the GPU-native performance work later integrated into
`main`, rather than further tolerance relaxation.

All 320 trial endpoints pass the requested GPU limit. Full verified runtime
fell from 13.121/13.473/49.636/51.895 s at `1e-8` to
12.160/12.611/48.241/50.353 s at `1e-7`: reductions of 7.3/6.4/2.8/3.0%.
The strict baseline did not separately record public solve time, so those
reductions are labelled **full verification** improvements, not solve-only
improvements. No production default was changed.

The observed `1e-7` GPU cellwise imbalances range up to `9.7495e-8`; MF6
achieved maxima vary by model. A numerically smaller requested tolerance is
not a universal claim of stricter achieved accuracy under different norms.
Per-model values are retained in
[`NPF_GPU_PERFORMANCE.md`](../release_evidence/NPF_GPU_PERFORMANCE.md) and
[`npf_gpu_comparison.json`](../release_evidence/npf_gpu_comparison.json).

## Supplementary full-verification timings

GPU times below are medians of three cold-state repetitions after an excluded
warmup. Public setup and independent checks are included as recorded by the
driver. They are not kernel-only measurements.

### Large steady NPF: 250 × 250 × 5

| Case | GPU K-cycle FGMRES | GPU diagonal GMRES | Archived MF6 |
|---|---:|---:|---:|
| Confined | 0.625 s | 18.358 s | 10.538 s |
| Convertible upper layer | 0.628 s | 28.734 s | 97.649 s |

All GPU runs passed independent physical acceptance. The immutable saved
baseline's 347 files were verified; heads and all three face-flux arrays were
compared. **The archived MF6 timing uses different stopping controls and is
not a matched contemporary speedup.** K-cycle is the measured faster GPU
selection on these large steady models. Diagonal remains the public default.

### Small transient NPF: 32 × 32 × 5, twenty weekly steps

| Case | GPU K-cycle FGMRES | GPU diagonal GMRES | MF6 single run |
|---|---:|---:|---:|
| Confined | 1.638 s | 0.598 s | 0.233 s |
| Convertible upper layer | 1.135 s | 0.268 s | 0.172 s |
| Phreatic recharge variant | 1.165 s | 0.272 s | 0.166 s |

All these comparisons passed. MF6 uses the actual 2D transient IMS profile.
MF6 is faster at this small size. Its elapsed values are single native
reference-generation runs; GPU values are repeated complete-history times
including construction and independent checks. They are not equivalent
timing intervals or paired process medians.

### Cartesian with the actual 2D sanity work controls

| Case | Fast | Classic | Outcome |
|---|---:|---:|---|
| 64 × 64 × 4 steady | 0.511 s | 2.944 s | Both fail the 200-cycle budget |
| 256 × 256 × 6 steady | 0.955 s | 3.677 s | Both fail the 200-cycle budget |
| 64 × 64 × 4, one confined transient step | 0.030 s | 0.155 s | Both pass |
| 256 × 256 × 6, one confined transient step | 0.264 s | 0.429 s | Both pass |

The steady values are attempted runtimes, not successful solve medians and
not evidence for a successful speedup. Fast is 5.1× / 1.6× faster than classic
on the two successful transient comparisons. No matched MF6 timing was
performed for those Cartesian cases.

Earlier development-checkout and tighter-than-sanity trials remain separate
execution history. Their speedup estimates are excluded from these primary
comparisons. In particular, the earlier 10–35× small transient estimate used
overly strict native settings and is superseded by the table above.

### Why convertible transient can be faster

These cases use `Ss=1e-5 /m`, `Sy=0.15` and a convertible upper layer about
24 m thick. Its specific-yield storage is roughly 600 times the confined
`Ss × thickness` storage, strengthening the timestep diagonal while heads
remain below top. Storage and saturation-dependent conductance both change
the linear systems; a nonlinear case can therefore require less work.
The [MF6 6.6.2 STO specification](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-sto.html)
defines the confined/convertible storage distinction.

The native listings record 2,178 versus 363 linear iterations over twenty
steps at 250², and 3,807 versus 643 at 500², for confined versus convertible.
These counts support the observed speed difference; they do not isolate
nonlinear overhead on an otherwise identical matrix.

### Realistic transient NPF: twenty weekly steps

| Model | GPU K-cycle, relative limit 1e-8 | MF6 single engine run | MF6/GPU runtime ratio |
|---|---:|---:|---:|
| 250 × 250 × 5 confined | 13.121 s | 47.589 s | 3.6× |
| 250 × 250 × 5 convertible | 13.473 s | 17.319 s | 1.3× |
| 500 × 500 × 5 confined | 49.636 s | 346.408 s | 7.0× |
| 500 × 500 × 5 convertible | 51.895 s | 104.391 s | 2.0× |

The domain is 25 km square, resolved at 100 m and 50 m. The models have
312,500 and 1,250,000 cells with five independently varying DIS layers.
The convertible variant changes only the top layer's cell type. Sources,
materials, geometry and cold initial heads are otherwise matched.

All four models completed one warmup and three timed twenty-step GPU histories.
Every one of the **320 GPU endpoints** passed both the device and independent
physical acceptance checks. Worst relative cellwise imbalance was
`9.7013e-9`, worst head-equivalent residual RMS `6.1688e-12 m`, and worst
head difference from MF6 `3.4138e-6 m`.

MF6 completed all endpoints normally with verified source allocation and a
worst native cellwise imbalance of `2.1390e-7`. It meets the 2D sanity
reference criteria, while some endpoints fail the additional stricter
`1e-8` NPF audit; that audit is retained separately. The GPU passes that
stricter limit. These are the same physical inputs with different final
acceptance rules, not a claim of identical stopping criteria.

GPU times are three-repeat medians including setup and independent audits.
MF6 values are single internal elapsed times, excluding the Python adapter's
setup/audits. The ratios compare those stated intervals. All native data were
cached before these serial GPU timing runs.

Evidence: `release_evidence/realistic_transient_verified/results.json` and
`summary.csv`. The first exploratory large run stopped on a reference-audit
reporting issue and is retained separately; its partial runtimes are excluded.

An explicit `1e-7` relative-imbalance experiment uses
`--npf-relative-tolerance 1e-7`. The device and independent gates both receive
the override, while all other tolerances and the default `1e-8` contract stay
unchanged. CPU/CUDA tests use an identical residual between the two limits to
verify the change and verify restoration of the default on a following call.

## Verification and remaining limits

- Final complete scoped regression selection: **516 passed, 43 native tests
  explicitly deselected**, no skipped CUDA cases. The NPF transient analytical,
  Jacobian and state-ownership tests now include CPU and CUDA parameterizations.
- Three additional runner tests pass, including minute/second timing parsing
  and ensuring a reference-quality failure cannot truncate an otherwise
  physically accepted GPU history.
- A source-profile test verifies that editing 2D controls changes the derived
  3D profile, and unresolved native controls raise instead of being omitted.
- The portable runner completed eight small steady/transient Cartesian/NPF
  cases, with fresh references and then a successful cached-only replay.
- Optional native validation-helper settings allow complete source-matched IMS
  profiles. Their existing default behavior and production solver physics,
  defaults and acceptance thresholds are unchanged.
- All four realistic 250² / 500² NPF twenty-step cases passed, as reported
  above. Partial histories are never labelled as complete measurements.

Raw primary evidence is in `release_evidence/gpu_20261111_2d_sanity/`, large
steady evidence in `release_evidence/gpu_20261111/npf_steady/`, and portable
runner evidence in `release_evidence/portable_sanity_pass/` and
`release_evidence/portable_cached_verified/`. Each portable run writes
`results.json`, `summary.csv` and compressed heads. Cache manifests retain
native inputs/results and hashes. These are local results, not a universal
performance or field-accuracy guarantee.
