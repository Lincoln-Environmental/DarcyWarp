Further NPF performance search, 12 November 2026
================================================

Final main qualification
------------------------

The construction change is promoted to main ``871ae34``. The table contains
fresh runs of that revision: one separate warmup and three fresh-model
repetitions. Each row covers the complete 20-week history, with no SS
presolve. Saved MF6 engine times and fingerprinted physical inputs are unchanged.

.. list-table:: Final main timings; medians of three GPU runs
   :header-rows: 1

   * - Profile / grid
     - Entire history (s)
     - Model construction (s)
     - Extra audit (s)
     - Saved MF6 engine (s)
     - Maximum head difference (m)
   * - strict convertible: npf_convertible_250x250x5_transient
     - 0.256
     - 0.050
     - 4.395
     - 17.319
     - 3.69e-08
   * - strict convertible: npf_convertible_500x500x5_transient
     - 0.972
     - 0.227
     - 20.398
     - 104.391
     - 6.84e-08
   * - MF6 accuracy convertible: npf_convertible_500x500x5_transient
     - 0.941
     - 0.221
     - 19.946
     - 104.391
     - 7.88e-08
   * - strict confined: npf_confined_250x250x5_transient
     - 0.635
     - 0.051
     - 4.378
     - 47.589
     - 1.06e-06
   * - strict confined: npf_confined_500x500x5_transient
     - 1.378
     - 0.221
     - 20.002
     - 346.408
     - 3.45e-06

All 400 endpoints pass independent physical audits and MF6 head gates.
Main regression validation: 627 passed, four CPU parameterizations of CUDA-only
checks skipped, 43 fresh-MF6 tests deselected. The prior promotion also reran
the cached Cartesian steady/transient smoke benchmarks; this change touches
only NPF preconditioner construction. Experimental terrain remains on dev.

The public timer still includes internal setup, graph capture, source updates,
production acceptance and output download. Extra benchmark audits and external
model construction are separate. The strict profile remains at 1e-7 relative
cell imbalance; the 1.5e-7 profile matches the saved MF6 physical accuracy
envelope. Both retain the 1e-6 m head-equivalent RMS gate.

Source fingerprints and individual-repeat evidence:
``benchmark_results/gpu_native/polish_main_20261112.json``.

Paired timing and session variability
-------------------------------------

The strict 500-grid qualification repeats are **0.898, 0.976 and 0.972 s**.
Their 0.972 s median is slower than the earlier main session's 0.954 s.
The numerical interval remains 0.645--0.648 s, and internal hierarchy
construction takes less time. Other overhead around the history call rose
from about 0.039 s to 0.090 s in two repetitions. The table retains every
measurement; the slower session is not evidence of a numerical speedup.

A separate five-pair check alternated host/device and device/host construction
order after a warmup for each option. Every call used a fresh model and the
same original initial heads, source history, conductivity and strict gates.
The device option was faster in all five pairs:

.. list-table:: Alternating paired medians, 500 x 500 x 5 convertible
   :header-rows: 1

   * - Measurement
     - Host construction (s)
     - Device construction (s)
   * - Entire public history
     - 0.981
     - 0.928
   * - First-step internal setup
     - 0.195
     - 0.149
   * - Numerical solving
     - 0.640
     - 0.640

The public median saving is 5.4%; all runs take 200 Krylov and 49 Newton
iterations and have the same maximum MF6 head difference, 6.84e-8 m.
Production physical acceptance and all matching-time MF6 head checks were
active. The additional independent CPU audits were performed in the portable
qualification above, outside the timed calls. These paired results support
enabling device construction in the explicit profiles, without claiming a
further improvement to the numerical iteration. Individual measurements and
source fingerprints are in
``benchmark_results/gpu_native/polish_alternating_pairs_20261112.json``.

Search and implementation details
---------------------------------

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
