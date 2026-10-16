Observed 3D Newton preconditioner comparison
===========================================

Saved 250x250x5 inputs and MF6 arrays were read unchanged. No model regeneration or native solve is part of this driver.
Fresh model timings include NPZ load, construction, hierarchy/workspaces, solve and public independent acceptance.
Warmup is excluded; successful medians require all three accepted repetitions. Resident runs also reset the cold head.

Multigrid device events measure scaling + graph + output copy; scalar-read wait time overlaps that GPU work.
These costs must not be added as disjoint runtime fractions. Hierarchy time includes graph setup.

confined kcycle: median end-to-end 0.664502s; Newton [5, 5, 5]; Krylov [29, 29, 29]; restarts [0, 0, 0].
  construction_seconds: 0.051892s (7.81% of end-to-end; overlapping categories).
  device_apply_seconds: 0.236356s (35.57% of end-to-end; overlapping categories).
  synchronization_seconds: 0.248831s (37.45% of end-to-end; overlapping categories).
  preconditioner_enqueue_seconds: 0.004686s (0.71% of end-to-end; overlapping categories).
confined diagonal: median end-to-end 18.214375s; Newton [7, 7, 7]; Krylov [21000, 21000, 21000]; restarts [None, None, None].
confined end-to-end speedup: 27.411x.
unconfined kcycle: median end-to-end 0.668938s; Newton [6, 6, 6]; Krylov [28, 28, 28]; restarts [0, 0, 0].
  construction_seconds: 0.053096s (7.94% of end-to-end; overlapping categories).
  device_apply_seconds: 0.229507s (34.31% of end-to-end; overlapping categories).
  synchronization_seconds: 0.240023s (35.88% of end-to-end; overlapping categories).
  preconditioner_enqueue_seconds: 0.004633s (0.69% of end-to-end; overlapping categories).
unconfined diagonal: median end-to-end 28.504101s; Newton [11, 11, 11]; Krylov [33000, 33000, 33000]; restarts [None, None, None].
unconfined end-to-end speedup: 42.611x.

Numerical agreement and work
----------------------------

All 24 warmup, timed and resident runs passed both device and independent
physical acceptance. The six fresh-model K-cycle measurements used exactly
one cycle per application, zero fallback solves and zero FGMRES restarts.
Each confined solve used 29 applications and 34 Jv evaluations; each
unconfined solve used 28 applications and 34 Jv evaluations. True residuals
were recomputed before declaring each linear solve converged.

Against the saved native MF6 arrays, confined head RMSE is 5.480e-8 m,
maximum head error 1.378e-7 m and relative face-flux L2 error 1.870e-9.
Unconfined head RMSE is 1.507e-7 m, maximum head error 2.138e-7 m and
relative face-flux L2 error 2.563e-9. Cellwise relative imbalance is
2.519e-10 / 8.679e-10 (confined / unconfined), below the unchanged 1e-8
acceptance limit. Physical RMS is 7.806e-10 / 2.832e-9 m3/day; acceptance
uses the original conjunction with head-equivalent RMS and the original
physical-RMS OR relative-imbalance test. No tolerance was loosened.

Median warmed solver-only times are 0.297857 / 0.293131 s for K-cycle,
versus 17.909227 / 28.210863 s for diagonal. Median public solve times,
including workspace/hierarchy and independent host acceptance, are
0.563357 / 0.567347 s for K-cycle. Primary end-to-end speedups above include
NPZ load and fresh host model construction as well. Interpreter/context
startup and an empty-cache compile are outside these warmed measurements;
warmup measurements are retained individually, not included in medians.

Reuse, memory and recommendation
--------------------------------

The hierarchy has physical dimensions (5,250,250), (5,125,125), (5,63,63),
(5,32,32), (5,16,16). Every fresh model constructs it once and captures one
graph. All three subsequent resident cold-head solves per variant reused
the same hierarchy, buffers and graph, with zero rebuilds or graph captures.
Median resident end-to-end times (including NPZ load but using an existing
host model) are 0.522748 s confined and 0.540334 s unconfined.

The hierarchy allocates exactly 60,055,520 device bytes (57.27 MiB).
Total incremental live Warp pool allocation during a fresh K-cycle solve
is 530,563,644 bytes (505.98 MiB), versus 219,248,348 bytes (209.09 MiB)
for diagonal. This includes persistent flexible V/Z basis vectors and the
compatibility GMRES workspace. Device-wide 100-ms sampled peaks are
804 / 810 MiB for K-cycle and 522 / 516 MiB for diagonal. These samples
include context/display memory and can miss brief peaks; cumulative pool
high-water counters are retained separately.

Recommend five levels, vertical-line omega=0.8, two pre and two post line
sweeps, 20 additional coarsest line sweeps and one K-cycle per application
for the next experimental stage. Coarsest entry also performs the two
pre-sweeps, matching the existing algorithm. Two cycles or other depths
were unnecessary for these accepted end-to-end solves. The public default
remains diagonal. The preliminary omega=1.0 run is retained locally in
../npf662_kcycle_001; it was stopped after detecting the configuration
mismatch and is excluded from final medians.

Multigrid applications perform zero device-to-host scalar reads. The
reported 241 / 210 scalar reads belong to FGMRES Arnoldi/restart norms.
Their wait time overlaps device application time, so it cannot be treated
as additional independent overhead. Coarse graph/line-smoother work is the
largest measured device cost; host enqueueing is under 0.005 s per solve.

Validation and reproduction
---------------------------

75 explicitly selected CPU/CUDA tests passed; see tests.log. Tests cover
manufactured independent sparse operators, inactive and fixed masking,
nonstationary preconditioners with stored-Z reconstruction, shortened
restart blocks, wet/dry/mixed heterogeneous nonlinear roots, exact zero
rows, nonfinite inputs, zero operators, rejected trials, shared exhausted
budgets, source/boundary reuse, material/interface invalidation and close.
An earlier broad regression selection inadvertently invoked three tiny
existing MF6 checks; the final explicit selection invokes none. The saved
large-model MF6 reference was never rerun or regenerated.

Run working_tests/benchmark_npf662_preconditioner.py from the repository
root with DARCY_FLOAT=float64, PYTHONPATH=.:working_tests and single-thread
BLAS variables as documented in docs/NPF662_KCYCLE_RESULTS.rst. Choose a new
output directory in its execution guard. summary.json contains effective
controls, source hashes, hardware/runtime metadata, every individual run,
complete physical checks, graph/work counters and memory measurements;
histories.json exposes the residual histories without model duplication.
