NPF runtime search: one-second target
=====================================

This is development work after the ``cd32e71`` main release. The target is
one second for the complete 500 x 500 x 5 convertible NPF/STO history of
20 weekly steps on the RTX 4070 Ti SUPER. The target has not yet been met.
The previous promoted implementation takes 5.916 s; the first fully audited
new setting takes 2.002 s (2.001, 2.012 and 2.002 s), about 52 times faster
than the saved 104.391 s CPU MODFLOW 6.6.2 engine run.

These are synchronized public solve times. They include internal workspace
construction, graph capture, coefficient refresh, transfers and production
acceptance. Model construction, saved-reference loading and extra independent
benchmark audits remain outside this timer. Every measured history starts
from the original prescribed heads, on a fresh model, after a separate warmup.
There is no steady-state presolve. Only the upper layer is convertible.

Qualification and scope
-----------------------

The portable-runner repeat median is 1.997 s for the 500 convertible case;
its maximum MF6 head difference is 7.43e-8 m. All 320 endpoints across both
sizes and both cell-type variants pass independent physical checks. The
short hierarchy is tuned for convertible storage: applying it unchanged to
the confined case regresses runtime. Retain the deeper hierarchy and longer
restart for confined models until that separate tuning is qualified.

.. list-table:: Same experimental controls, medians of three complete histories
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

Reproduce the qualification
---------------------------

The portable runner accepts a complete, reviewable control file::

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python working_tests/run_3d_sanity.py \
       --sizes 250 500 --families npf --modes transient \
       --variants convertible --periods 20 --repeats 3 \
       --npf-controls working_tests/npf_runtime_controls.json \
       --cache benchmark_results/mf6_3d_cache \
       --output benchmark_results/npf_runtime_qualification

Add ``--cached-only`` to require existing references. They must match the
physical inputs, source schedule, storage and the actual 2D sanity IMS profile.
Use ``--sizes 16 32`` for a smaller convergence ladder. All repeats retain
independent cell-balance checks, MF6 comparison and full timing categories.
The existing Cartesian tests and comparison path remain available without
the NPF control file.

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

.. list-table:: Audited convergence sweep, same solver controls and original history
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

The runtime control file keeps the strict 1e-7 setting. To reproduce the
MF6-envelope track, add ``--npf-relative-tolerance 1.5e-7`` to the portable
command. Compact evidence, experiment specifications and source hashes are
in ``benchmark_results/gpu_native/stretch_progress.json``.

Calibration harness correction
------------------------------

The earlier K-change harness passed a Boolean where the case generator expected
layer indices, marking all layers convertible. The lower layers were fully
saturated in those runs, so the active equations coincided, but the metadata
and work differed. It now uses the canonical upper-layer-only selection.
The generator rejects Boolean layer selections. The main MF6 timing case
always used the correct mask and is unaffected. New calibration results must
use the corrected harness; do not reuse the earlier timing rows as the new
qualification.
