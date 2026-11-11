NPF runtime search: one-second target
=====================================

This is development work after the ``cd32e71`` main release. The target is
one second for the complete 500 x 500 x 5 convertible NPF/STO history of
20 weekly steps on the RTX 4070 Ti SUPER. The strict target is met: the
complete device history API takes **0.957 s**, median of three fresh-model
runs after a separate warmup. The saved CPU MODFLOW 6.6.2 engine run takes
104.391 s, about **109 times longer**. Maximum head difference is 6.84e-8 m;
all 80 endpoints, including warmup, pass independent physical audits.
The three measured history times are 0.940, 0.957 and 0.998 s; the median
Newton/Krylov interval within those public calls is 0.646 s.

The selected imbalance limit remains 1e-7 and head-equivalent RMS remains
1e-6 m. This result does not depend on loosening convergence. The numerical
work uses FP64 heads, physics, Krylov algebra and acceptance with FP32
multigrid corrections. Only the top of five layers is convertible.

.. list-table:: Qualified strict device history, medians of three runs
   :header-rows: 1

   * - Model
     - Entire history API (s)
     - Model construction (s)
     - Extra independent audit (s)
     - MF6 engine (s)
   * - Convertible 250 x 250 x 5
     - 0.267
     - 0.049
     - 4.348
     - 17.319
   * - Convertible 500 x 500 x 5
     - 0.957
     - 0.220
     - 19.742
     - 104.391
   * - Confined 250 x 250 x 5 (separate profile)
     - 0.645
     - 0.050
     - 4.393
     - 47.589
   * - Confined 500 x 500 x 5 (separate profile)
     - 1.490
     - 0.222
     - 19.709
     - 346.408

The complete 500-case verification takes 20.960 s. Its extra audit is a
benchmark check, separate from the production solver's mandatory physical
acceptance. Public timing includes initial internal setup, graph capture,
every source/timestep update, acceptance and the output history download.
External model construction, cache loading and benchmark file writing are
excluded. MF6's own head/budget output remains inside its engine timer;
the GPU output download remains inside its history timer. MF6 timing is
one saved engine run, not a three-repeat median.

The preceding promoted implementation takes 5.916 s. An intermediate
step-by-step API checkpoint took 2.002 s (2.001, 2.012 and 2.002 s).
The results below retain that intermediate evidence and rejected choices.

These are synchronized public solve times. They include internal workspace
construction, graph capture, coefficient refresh, transfers and production
acceptance. Model construction, saved-reference loading and extra independent
benchmark audits remain outside this timer. Every measured history starts
from the original prescribed heads, on a fresh model, after a separate warmup.
There is no steady-state presolve. Only the upper layer is convertible.

Intermediate qualification and scope
-------------------------------------

The portable-runner repeat median is 1.997 s for the 500 convertible case;
its maximum MF6 head difference is 7.43e-8 m. All 320 endpoints across both
sizes and both cell-type variants pass independent physical checks. The
short hierarchy is tuned for convertible storage: applying it unchanged to
the confined case regresses runtime. The final profiles therefore use a
separately qualified deeper hierarchy and longer restart for confined models.

.. list-table:: Intermediate checkpoint using the same short hierarchy for every model
   :header-rows: 1

   * - Model
     - Public solve (s)
     - Model construction (s)
     - Extra audit (s)
     - MF6 engine (s)
   * - Convertible 250 x 250 x 5
     - 0.563
     - 0.050
     - 4.449
     - 17.319
   * - Convertible 500 x 500 x 5
     - 1.997
     - 0.217
     - 19.565
     - 104.391
   * - Confined 250 x 250 x 5
     - 1.597
     - 0.050
     - 4.386
     - 47.589
   * - Confined 500 x 500 x 5
     - 9.310
     - 0.222
     - 19.887
     - 346.408

Implementation changes
----------------------

Validation at this development checkpoint: 1,667 affected 3D, terrain and
NPF tests pass; 52 tests requiring fresh MF6 executions are deselected.
Saved MF6 histories are used for the realistic runs. Tests cover CPU/CUDA
manufactured solutions, mixed-precision refresh after K changes, storage and
cell-type changes, host/device handoff, predictor rollback/history reset,
dry fallback, invalid inputs and cache verification.

* Retain the previous-state storage tangent on the GPU. An owned host snapshot
  is materialized only for first hierarchy construction or an explicit audit.
* Reuse validated geometry and unchanged material uploads. Owned snapshots
  still detect mutation; source, boundary, initial head, cell type, Ss, Sy,
  timestep and previous head remain live validated inputs.
* Factor the vertical line systems once per coefficient refresh, then reuse
  their factors for smoothing. Refresh all coarse levels when coefficients change.
* Use a shorter hierarchy and fewer coarse sweeps for this transient case.
  These are explicit benchmark controls, not new universal defaults.
* Optionally run the multigrid correction in FP32, following the 2D mixed
  correction design. Physical heads, fluxes, storage, residuals, FGMRES and
  final acceptance remain FP64. Unrepresentable FP32 coefficients raise;
  select the FP64 preconditioner for those inputs.
* Optionally refresh the preconditioner from current Picard conductances and
  storage tangents. The full-saturation fallback still uses full-saturation
  conductances. Physical conductivity is never frozen across calibration runs.
* Optionally extrapolate a timestep starting guess from the preceding accepted
  step. The predictor must reduce both the physical merit and scaled residual.
  It does not alter the fixed previous head in the storage equation. Failed
  solves clear prediction history; unrelated histories cannot reuse it.

The audited setting uses three levels, two pre/post/coarse sweeps, FP32
preconditioning with cached line factors and current coefficients, FGMRES
restart eight, inner forcing 0.01, and first-order timestep prediction.
Head-equivalent RMS remains 1e-6 m and the physical relative imbalance limit
remains 1e-7. The dimensional alternative remains 1e-12 m3/day.

The search also tested one to five levels, lighter smoothing, V-cycles,
fixed inner forcing from 0.003 to 0.3, adaptive forcing and quadratic
prediction. A single level was much slower. V-cycles, adaptive forcing and
quadratic prediction did not beat the selected setting. They remain explicit
experimental choices; numerical acceptance is unchanged.

Reproduce the current qualification
------------------------------------

The portable runner accepts a complete, reviewable control file::

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python working_tests/run_3d_sanity.py \
       --sizes 250 500 --families npf --modes transient \
       --variants convertible --periods 20 --repeats 3 \
       --npf-controls working_tests/npf_runtime_controls.json \
       --npf-history \
       --cache benchmark_results/mf6_3d_cache \
       --output benchmark_results/npf_runtime_qualification

Add ``--cached-only`` to require existing references. They must match the
physical inputs, source schedule, storage and the actual 2D sanity IMS profile.
Use ``--sizes 16 32`` for a smaller convergence ladder. All repeats retain
independent cell-balance checks, MF6 comparison and full timing categories.
The existing Cartesian tests and comparison path remain available without
the NPF control file.

Device history API and final runtime changes
---------------------------------------------

``WarpDarcySolver3D.solve_transient_npf(source_factors=..., dt=...,
head_prev=..., ss=..., sy=..., **solver_options)`` returns the head history
and per-step diagnostics. It requires the fast ``mf6_6_6_2_npf`` formulation
and native K-cycle execution. Timestep lengths and scalar multipliers of
the spatial source field may vary. Geometry, K, storage fields, cell types
and prescribed heads are fixed within a call; physical K may change between
calls using ``update_K_fields``. This is an explicit API, not a replacement
for general transient drivers with arbitrary changing boundary fields.

Intermediate accepted heads and storage state remain on the GPU. Normal
execution downloads the first step head once and the complete output
history once. Every step retains physical acceptance; insufficient storage
grounding invokes the independent CPU fallback, with additional readbacks.
A failed step stops the history, reports the number accepted and includes
the failed candidate in the returned history. Input source/initial fields
are restored after a call, and exception cleanup invalidates solver state.
There is no steady-state presolve or reuse of MF6 solution heads.
The retained output uses eight bytes per cell per requested step on the
GPU (200 MB for this benchmark); split longer histories into shorter calls
when output retention would exceed device memory.

The final speed improvements are:

* Parallel FP64 tile reductions for physical metrics and Krylov dot products,
  including their final partial sums. Nsight found single-thread final sums
  consumed about 18% of kernel time in the preceding implementation.
* Reuse an already evaluated trial state after acceptance. Rejected trials
  have separate derivative, flux and storage buffers. Accepted data copy into
  stable buffers referenced by graphs; the physical criterion is unchanged.
* Capture coefficient refreshes and avoid an intermediate hierarchy refresh
  that current-coefficient preconditioning immediately replaces.
* Remove a multigrid residual norm whose result was never used. Actual
  physical residual checks and FGMRES stopping checks remain.
* Avoid an extra host copy of the output history. Its NumPy base retains
  the dedicated host allocation after the solver closes.

The final strict control file uses inner forcing 0.005, first-order prediction,
tile reductions, accepted-trial reuse, three multigrid levels with two
pre/post/coarse sweeps, cached line factors, FP32 current-coefficient
preconditioning and restart eight. Library defaults are unchanged. Red/black
line smoothing was also tested and remains an explicit experimental choice;
it did not beat the selected Jacobi line smoother on this case.

For confined models use ``working_tests/npf_confined_runtime_controls.json``:
five levels and restart 24. The separately audited step-by-step profile
qualified at 0.990 s (250) and 2.555 s (500), removing the intermediate
short-hierarchy regression shown above. With the final history driver and
parallel reductions, the confined medians are 0.645 s and 1.490 s.
All 160 confined endpoints pass independent checks; maximum MF6 head
differences are 1.06e-6 m and 3.45e-6 m, within the shared sanity head gates.

``working_tests/tune_npf_gpu_runtime.py`` is the lighter experiment harness.
It requires a matching saved reference and verifies its hash and IMS controls.
Use ``--audit --repeats 3`` for a finalist, rather than treating exploratory
single repeats as release evidence. ``--profile`` adds profiler overhead and
must not be used for a reported benchmark time.

Convergence comparison with MF6
-------------------------------

MF6 uses the unchanged 2D transient sanity settings: outer head change 1e-6,
inner head change 1e-8 and ``RELATIVE_RCLOSE 1e-6``. MF6's inner relative L2
reduction is different from the GPU's nonlinear, throughput-normalized L1
cell imbalance. Copying 1e-6 into the GPU setting does not equate these tests.

On the saved history, MF6's independently reconstructed relative cell imbalance
has median 3.862e-8 and maximum 1.422e-7. A separate experiment uses 1.5e-7 as
a rounded envelope of that achieved physical imbalance, alongside the strict
1e-7 track. This is a case-specific accuracy comparison, not an assertion
that both solvers implement identical convergence tests. Both retain the
1e-6 m GPU head-equivalent gate and the independent MF6 head comparison.

.. list-table:: Earlier audited step-by-step convergence sweep, same controls and original history
   :header-rows: 1

   * - Solver / selected relative criterion
     - Solve seconds
     - Achieved median imbalance
     - Achieved maximum imbalance
   * - MF6 / inner L2 reduction 1e-6
     - 104.391
     - 3.862e-8
     - 1.422e-7
   * - GPU / physical imbalance 1e-7
     - 2.002
     - 1.003e-8
     - 8.747e-8
   * - GPU / physical imbalance 1.5e-7
     - 1.925
     - 2.977e-8
     - 1.416e-7
   * - GPU / physical imbalance 1e-6
     - 1.804
     - 7.732e-8
     - 6.715e-7

GPU times are three-repeat medians after a warmup; each of the 80 endpoints
per setting passes an independent audit using its declared criterion. The
1.5e-7 choice achieves similar median and worst physical imbalance to MF6
and saves about 4% against the strict setting. Its maximum MF6 head difference
is 7.88e-8 m. Selecting 1e-6 saves about 10%, but its worst physical imbalance
is about 4.7 times MF6's; matching the numerical parameter alone is therefore
not the matched-accuracy choice. Its maximum head difference is 1.04e-7 m.

The runtime control file keeps the strict 1e-7 setting. The current
MF6-envelope profile is ``working_tests/npf_mf6_accuracy_controls.json``;
select it with ``--npf-controls`` and ``--npf-history``. It retains inner
forcing 0.01. The final complete history takes **0.931 s**, median of three
runs after warmup, with maximum MF6 head difference 7.88e-8 m and all 80
endpoints independently accepted. Compared with the selected strict profile
(0.957 s, inner forcing 0.005), the gain is about 2.6%; the bulk of the
speedup is from execution changes, not convergence relaxation.
The final strict profile achieves median/max relative imbalance
1.53e-8 / 7.56e-8. The MF6-comparable profile achieves 2.98e-8 / 1.416e-7,
against MF6's 3.86e-8 / 1.422e-7 using the same independent physical norm.

Final evidence, source hashes and per-repeat timing:
``benchmark_results/gpu_native/history_qualification.json``. Intermediate
experiments remain in ``benchmark_results/gpu_native/stretch_progress.json``.

Calibration harness correction
------------------------------

The earlier K-change harness passed a Boolean where the case generator expected
layer indices, marking all layers convertible. The lower layers were fully
saturated in those runs, so the active equations coincided, but the metadata
and work differed. It now uses the canonical upper-layer-only selection.
The generator rejects Boolean layer selections. The main MF6 timing case
always used the correct mask and is unaffected. The corrected 500-case
qualification changes all physical K components by factors 0.5, 1 and 2,
then runs 20 weekly steps from the original heads. Every device endpoint
passes an independent physical audit. Host/device maximum head difference
is 1.06e-13 m; the history API matches individual GPU steps exactly on all
three K fields.

.. list-table:: Corrected calibration checks; one run per K, not repeat medians
   :header-rows: 1

   * - K multiplier
     - Host numerical interval (s)
     - Individual GPU steps, numerical interval (s)
     - Prepared-model history API (s)
   * - 0.5
     - 1.434
     - 0.608
     - 0.708
   * - 1
     - 1.579
     - 0.681
     - 0.776
   * - 2
     - 1.723
     - 0.750
     - 0.844

The history measurement follows the individual-step comparison, so the
current K has already been prepared. It is not a fresh-model or first-run
changed-K timing. GPU K updates before the step comparison take 0.231,
0.197 and 0.193 s separately. The numerical intervals exclude setup and
independent audits; full public step timings and all audit times are saved
in the compact evidence. No MF6 timing claim is made for changed-K cases.

Reproduce with ``working_tests/benchmark_npf_calibration_runtime.py
--controls working_tests/npf_runtime_controls.json --output NEW_DIRECTORY``.

Development release details
---------------------------

This qualification is for ``dev`` after the promoted ``cd32e71`` main
release and the ``7598f6c`` mixed-preconditioner development checkpoint.
It adds the explicit NPF/STO history API and three reviewable control files
for strict convertible, MF6-comparable convertible and confined runs.
Select ``--npf-history`` in the portable sanity driver; omit it to exercise
the individual-step API. The Cartesian formulation remains separate and is
covered by the broad CPU/GPU regression runner.

The history path retains host input validation, first hierarchy construction,
small FGMRES rotations/backsolves and stopping decisions. It is not a fully
GPU-controlled nonlinear algorithm. Its measured sub-second result includes
those costs. Dry or ungrounded storage states retain the independent fallback;
the benchmark's fast positive-storage route does not replace that safeguard.

Final validation: **1,697 tests passed**, four CPU parameterizations of
CUDA-only checks skipped, and 52 fresh-MF6 tests deselected. The affected
suite covers Cartesian 3D, terrain and NPF on CPU/CUDA. It includes history
versus explicit-step parity, changed-K refresh, variable dt, independent
fallback, failed-step stopping, exception cleanup, rejected-trial rollback,
cached-state versus recomputed-state equality, graph invalidation and FP32
representability failures. All 400 portable-runner endpoints and 120 external
calibration endpoints pass independent audits. The Sphinx HTML build passes
with warnings treated as errors.
