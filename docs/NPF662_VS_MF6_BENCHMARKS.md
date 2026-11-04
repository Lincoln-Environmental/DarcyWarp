# Repeated 3D NPF versus MODFLOW 6 benchmarks

`working_tests/benchmark_npf662_vs_mf6.py` runs fresh native MF6 solves and
DarcyWarp's explicit `mf6_6_6_2_npf` formulation on identical layered DIS
inputs. The default DarcyWarp path is Newton with K-cycle preconditioning
and flexible GMRES. No solver defaults, equations or physical acceptance
tolerances are changed.

Run from the repository root in the `darcywarp` environment:

```bash
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python working_tests/benchmark_npf662_vs_mf6.py --tier smoke
```

Then run the standard ladder:

```bash
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python working_tests/benchmark_npf662_vs_mf6.py --tier standard
```

The existing reference writer requires its pinned MF6 6.6.2 executable,
including the exact SHA-256 recorded by the NPF contract. `--mf6 /path/to/mf6`
selects it explicitly; the default uses `/bin/modflow/mf6` when present,
then searches PATH.
The benchmark preserves the writer's version and checksum checks.

## Case matrix

| Tier | Horizontal grids | Layers | Total cells per case |
| --- | --- | --- | --- |
| smoke | 16 x 16, 32 x 32 | 5 | 1,280; 5,120 |
| standard | 50 x 50, 100 x 100, 250 x 250, 500 x 500 | 5 | 12,500; 50,000; 312,500; 1,250,000 |
| scale | 1,000 x 1,000 | 5 | 5,000,000 |

Each grid has three variants:

| Variant | Convertible layers | Recharge |
| --- | --- | --- |
| confined | none | 150 mm/year |
| convertible | top layer | 150 mm/year |
| phreatic | top layer | 5 mm/year |

The footprint stays 25 km square, so cell size is `25000 / n` metres.
The 250-cell grid reproduces the frozen baseline's physical settings:
seed 6622505, five varying-thickness layers, heterogeneous anisotropic K,
west-column CHD at 0 m, top-cell recharge and no other external flow.
Recharge uses native RCH6 with FIXED_CELL; DarcyWarp receives the exact
same volumetric source. Only the top layer is convertible. Time units are
days with 365 days/year. These are steady cases, without STO or XT3D.
The existing baseline generator is shared without changing its arithmetic.

The 150 mm/year convertible baseline is predominantly above model top.
The lower-recharge variant exercises partially saturated top cells. Native
head fractions above top and below bottom are recorded rather than assumed.
The fixed seed reproduces each resolution independently; random K noise
is not a restriction of one common fine-grid realization, and this ladder
is not a grid-convergence study.

## Measurements and acceptance

Every case runs one excluded MF6 warmup followed by three fresh MF6
processes. Every requested DarcyWarp preconditioner gets one excluded
warmup and three freshly constructed model solves. All start from the same
saved cold head, including the three subsequent resident-model solves.
Resident runs must demonstrate workspace reuse and, for K-cycle, hierarchy
reuse. They are retained separately and excluded from fresh-model speedups.
The first run includes any lazy compilation; compilation and context startup
are outside the warmed medians. An empty-cache process benchmark remains
available separately in `measure_npf662_cold_process.py`.

Both engines must pass the unchanged independent NPF conservation checks.
MF6 must also pass normal termination and native source-allocation/face-flow
checks. Every DarcyWarp repeat must meet these additional comparison gates:

| Metric | Limit |
| --- | --- |
| Free-cell head RMSE | 0.001 m |
| Maximum free-cell head difference | 0.005 m |
| Relative physical face-flow L2 difference | 0.000001 |

Independent physical acceptance includes prescribed heads, grounded active
components, head-equivalent residual RMS at most 1e-6 m, and physical RMS
at most 1e-12 m3/day OR throughput-normalized cellwise L1 imbalance at most
1e-8. Domain cancellation cannot satisfy that check. The requested
preconditioner and flexible-GMRES path must actually execute. A failed or
missing timed repeat withholds both comparison medians and speedups.
All failures are saved, the remaining cases continue, and the final process
returns a nonzero status if any measurement or case gate failed.

Two timing comparisons are reported:

* `end_to_end_speedup`: MF6 write/run/decode versus DarcyWarp model
  construction/reset/public solve. Both start from canonical arrays in RAM.
  Input generation, NPZ archival, imports, context startup, post-run duplicate
  parity checks and reporting are excluded. MF6 output decoding and the
  public independent audits are included.
* `process_vs_public_speedup`: MF6 subprocess wall time, including native
  I/O and executable startup, versus synchronized DarcyWarp public solve,
  including workspace/graph setup and its public host audit. These are
  explicitly different timing boundaries. The narrower backend timer is
  recorded separately and never substituted into the primary speedup.

This is the serial native MF6 executable comparison. Thread environment,
binary hash, source revision and source hashes, interpreter, CPU/platform,
GPU identity and runtime versions are saved. CUDA pool counters are recorded
with pre-run baselines; the high-water counter is process cumulative, not
a per-run peak. Resident reuse is a fixed-input cold-head test, not ensemble
throughput with changed parameters.

## Selecting cases and outputs

```bash
# Inspect the plan without importing Warp or running either engine.
python working_tests/benchmark_npf662_vs_mf6.py --tier standard --dry-run

# Add the large grid explicitly.
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/benchmark_npf662_vs_mf6.py --tier scale

# Compare the three available preconditioning choices on smaller grids.
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/benchmark_npf662_vs_mf6.py --sizes 50 100 --preconditioners kcycle diagonal diagonal_fgmres

# CPU correctness checks use the same NPF API without CUDA graphs.
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/benchmark_npf662_vs_mf6.py --sizes 4 --device cpu --resident-repeats 1
```

`--variants`, `--repeats`, `--resident-repeats`, `--mf6-timeout`, `--device`
and `--output` expose the remaining run controls. At least three timed
repeats are required. An existing output directory is refused.
The default output is a new timestamped directory under `benchmark_results`.
Each run saves frozen settings, a source snapshot, exact case NPZs, native
decks and binary outputs for every repeat, DarcyWarp head NPZs, detailed
iteration histories and checks in `results.json`, and the compact comparison
table in `summary.csv`. Checkpoints are rewritten atomically after each
measurement. Interrupted runs retain their completed evidence; automatic
resume is not implemented.

Focused checks:

```bash
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python -m pytest -q tests/test_npf662_vs_mf6_benchmark.py
```
