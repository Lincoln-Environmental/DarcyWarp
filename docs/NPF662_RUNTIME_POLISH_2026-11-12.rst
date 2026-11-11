Further NPF performance search, 12 November 2026
================================================

After requalifying the sub-second release on main, a further search retained
one improvement: ``NPF662KCycleControls(device_hierarchy_build=True)``.
It constructs coarse conductances, fixed-head springs, storage diagonals and
line factors on the device using the existing validated refresh kernels.
Mask topology and finest-input validation remain on the host. This option
is enabled in the three checked-in runtime profiles; its library default is
false. Conductivity can still change between histories.

In a paired 500 x 500 x 5 convertible test, the median complete history
time fell from **0.986 to 0.935 s** across three fresh-model repetitions.
The numerical interval remained **0.641 s**, with exactly 200 Krylov and
49 Newton iterations. This is a saving in internal first-call setup, which
is included in the public timer. It is not a faster nonlinear iteration.
The external model construction and extra benchmark audit remain separate.
The physical case, original initial heads, 20 weekly steps, FP64 physical
acceptance and strict 1e-7 relative imbalance limit are unchanged.

The search also measured 24 multigrid level/sweep/relaxation combinations,
explicit scaled-Jacobian assembly, register-local Thomas work vectors and
a residual-secant line search for the timestep predictor. None gave a clear
additional numerical-runtime improvement for the convertible benchmark.
Jacobian assembly also failed to improve the confined case. The rejected
experimental implementations were removed; their timings and specifications
are retained in ``benchmark_results/gpu_native/polish_search_20261112.json``.
Small differences between separate benchmark sessions should not be treated
as speedups; the construction change is assessed against its paired baseline.

Construction regressions compare every level's six faces, fixed-head springs,
storage, diagonal and Thomas factors against host construction, then compare
the actual K-cycle correction. CPU/CUDA, FP64/FP32, irregular masks, odd
dimensions, semi-coarsening, timestep histories and changed-K reuse are covered.
The portable ``run_3d_sanity.py --npf-history`` commands use the updated
profiles without changing the convergence settings or cache identity.

Development validation: 1,709 tests passed, four CPU-only parameterizations
of CUDA checks skipped, and 52 fresh-MF6 tests deselected. Sphinx builds with
warnings treated as errors. Release qualification uses the saved MF6 histories.
