2D GPU release: runtime and GHB parity
====================================================

12 November 2026

This promotion brings the fast 2D reductions, transient runtime profile and
GHB parity fixes from ``dev`` onto ``main``. It covers steady confined and
unconfined solves and transient unconfined histories, including general-head
boundaries. The production 3D Cartesian and NPF/STO implementations remain
available; this promotion does not change their numerical code.

The numerical changes were developed in ``7b52040`` and ``fe75650`` and
promoted as ``139d85d`` and ``8c53c46``. Main-specific integration uses the
existing replay controls (``ac2972f``) and restores the diagnostic module
already referenced by release tests (``ebdfd06``). Fresh benchmarks compare
``ebdfd06`` against the preceding main release, ``2d448ab``. Source checksums,
reference checksums, every repetition and exact controls are retained in
``benchmark_results/gpu_2d_runtime/main_qualification_20261112.json``.

What changed
------------

* **Parallel residual reductions.** The fast FP64 operators now combine
  partial sums and maxima in a 256-thread CUDA block, replacing a serial
  GPU thread. FP64 arithmetic, scalar overwriting and partial-buffer clearing
  are preserved. CPU execution retains its original serial reduction order.
  Fast confined, Picard inner solves, transient face operators and mixed
  corrections share this improvement.
* **An explicit faster transient profile.** Each timestep starts from its
  accepted previous head, avoiding a separate linear startup solve. Picard
  uses full updates without GHB and relaxation 0.8 with GHB. Coarse smoothing
  remains at omega 1.0. A 100 m per-outer update cap replaces the profile's
  former 10 m cap; final convergence tolerances are unchanged.
* **Correct timestep prediction during capped updates.** The adaptive
  controller estimates contraction using only the unclipped tail of the
  update history. Repeated capped updates no longer masquerade as evidence
  of nonlinear stagnation. Hard iteration limits and genuine failure/retry
  handling remain active.
* **Correct eager mixed-precision refresh.** With CUDA graphs disabled, the
  FP32 face operator is refreshed on every outer iteration, matching graph
  execution as saturated thickness and GHB conductance change.
* **Stronger qualification tools.** The transient benchmark requires strict
  convergence, finite heads, the complete MF6 timestep sequence, existing
  head gates and an optional independent budget audit. The release commands
  enable that audit. GHB comparisons can verify the final native MF6 workspace
  against the saved reference before reading its runtime. Both benchmark
  drivers record source and reference fingerprints and fail unsuccessful
  comparisons instead of reporting them as valid speedups.
* **Runnable tests and diagnostics.** ``working_tests/run_2d_gpu_tests.py``
  gathers the affected CPU/CUDA suites without launching fresh MF6 work.
  ``run_transient_unconfined_replay_analysis.py`` is restored because six
  existing main tests depended on that missing module. It remains a diagnostic
  tool; the new benchmark gates define this release's performance qualification.

The parallel reduction and correctness repairs apply to the existing fast
paths. The runtime profile remains opt-in. Classic solver selection and the
library's startup, damping and 10 m update-cap defaults are preserved.
Main retains its existing public multi-period API; development-only terrain
and other unrelated API changes are outside this promotion.

Measured main performance
-------------------------

.. include:: ../benchmark_results/gpu_2d_runtime/main_timings_20261112.rst

Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, 16 GiB; Python 3.12 and Warp
1.17.0. GPU cases run serially, with a separate warmup followed by three
fresh-model repetitions. The table uses the median of the three timed calls.
The baseline uses the previous **fast** implementation, not classic.

The GPU timer covers the synchronized public solve/history call, including
internal hierarchy and graph setup, all timesteps, production convergence
checks, diagnostic output and head downloads. External model construction,
reference loading, first-use kernel compilation and the extra independent
CPU budget audit are excluded. The evidence retains build/wrapper/audit
timings where available; these are not included in the headline solve time.

MF6 entries are the saved engine timings for the same reference inputs,
including native engine output. They exclude external FloPy construction
and are single reference runs, not freshly repeated CPU measurements. For
GHB, the engine timer belongs to the final independently converged conductance
system: preceding fixed-point runs are reference-generation work and are
excluded. These columns compare the cost of solving the final matched system,
not the full MF6 conductance fixed-point workflow.

The 2D transient cases start from the same saved MF6 steady heads as the
existing sanity checks. No steady solve is included in the timed history.
This differs from the previously released 3D NPF histories, which start from
their prescribed initial heads without an SS presolve.

Convergence and MF6 agreement
-----------------------------

No convergence threshold was loosened for this promotion. The transient
profile inherits the existing production/sanity settings:

.. list-table:: Unchanged transient acceptance and accuracy settings
   :header-rows: 1

   * - Setting
     - Value
   * - Maximum accepted outer head change, ``hclose``
     - 1e-4 m
   * - Head-equivalent RMS, ``strict_head_residual_tol``
     - 1e-6 m
   * - Linear ``rel_tol`` / ``abs_tol_min``
     - 5e-7 / 5e-7
   * - Adaptive inner residual floor
     - 1e-12
   * - Final MF6 head comparison, RMSE / maximum
     - 0.001 m / 0.005 m
   * - Worst-period MF6 head comparison, RMSE / maximum
     - 0.005 m / 0.02 m
   * - Heads within 0.01 m, every period
     - At least 99.9%

The 100 m cap limits an intermediate nonlinear update; it is not an accepted
head error. FP64 physical heads, storage, residuals and convergence checks
surround FP32 inner corrections. Steady qualification uses fast FP64 and its
existing fixture controls, with maximum MF6 head gates of 5e-4 m unconfined
and 2e-4 m confined. MF6 residual settings are not treated as numerically
identical to GPU residual settings; matching heads, timesteps and physical
budgets provide the comparison.

GHB parity is a release requirement
----------------------------------------------------

The small GHB case initially failed by 13.10 m because clipped head updates
triggered early timestep subdivision. Strict convergence of those smaller
steps did not establish agreement with MF6's seven-day backward-Euler steps.
The controller correction and runtime cap address that cause. On the larger
500 x 500 GHB case, full updates additionally produce an oscillatory Picard
tail; GHB relaxation 0.8 is required for the qualified profile.

The old large-case defaults also fail fixed-step parity, by 20.52 m after
subdivision. Consequently its valid old-solver comparator uses the same
100 m cap, retaining the original startup solve and 0.7 relaxation. The table
does not use the failing old-default run to claim a speedup. Current library
defaults also fail that stress case; select the explicit runtime profile
for the demonstrated result.

GHB reference conductance is independently iterated from MF6's own heads,
``C = K * sat(h_MF6) * ghb_factor``. DarcyWarp heads do not define that fixed
point. The 500-grid reference required 16 MF6 runs and reached a terminal
head change of 9.46e-7 m. The benchmark checks the saved final heads and
times before accepting its native engine timer.

Regression coverage includes draining and injecting GHB, CPU and CUDA,
FP64 and mixed corrections, graphs and eager launches, budgets and capped
update histories. Earlier steady GHB checks on uniform, heterogeneous and
rectangular domains are retained in
``benchmark_results/gpu_2d_runtime/steady_ghb_validation_20261112.json``.
These earlier checks establish additional development evidence; their
timings are not part of the fresh main table.

Use the profile and reproduce the checks
----------------------------------------

From the repository root in the DarcyWarp environment:

.. code-block:: bash

   export PYTHONPATH=.:working_tests DARCY_FLOAT=float64 MPLBACKEND=Agg
   export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
   python -m working_tests.run_2d_gpu_tests
   python working_tests/benchmark_2d_runtime.py \
     --artifact DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined_1000x1000_30w_ugly_t_s42/mf6_transient_heads.npz.lzma \
     --controls working_tests/2d_transient_runtime_controls.json \
     --output outputs/2d_runtime_1000 --repeats 3 --audit

The benchmark selects the fast implementation and mixed inner corrections.
It reports all warmup/repetition outcomes and returns a nonzero exit status
if any history fails. CUDA tests skip when CUDA is unavailable, so check the
skip count. Missing large references must first be generated by
``working_tests/run_2d_transient_vs_mf6.py``; the benchmark does not silently
replace a reference. The detailed runtime guide contains the generation,
GHB workspace and steady benchmark commands.

For use through the existing replay API, load
``working_tests/2d_transient_runtime_controls.json`` and pass its
``runtime.solve_controls`` mapping as ``solve_control_overrides``. The exact
overrides are:

.. code-block:: json

   {
     "unconfined_startup_mode": "initial_head",
     "omega": 1.0,
     "omega_max": 1.0,
     "unconfined_ghb_relax": 0.8,
     "max_head_change_per_outer_iteration": 100.0
   }

The JSON alone does not enable mixed precision. The timing driver explicitly
enables ``transient_mixed_precision_enabled=True`` and saves diagnostics.
The production device path requires ``mf6_convertible_secant_sy`` storage
and ``storage_reference="current_picard"``; GHB requires the face operator.
Use the existing replay defaults as the base and then apply the overrides.
Check strict acceptance, head accuracy and water balance for each new model.

Release validation and scope
----------------------------

.. include:: ../benchmark_results/gpu_2d_runtime/main_validation_20261112.rst

The machine-readable test record is
``benchmark_results/gpu_2d_runtime/main_tests_20261112.xml``. Package, test
and driver byte-compilation passes. The documented CPU recharge example
converges with a maximum analytical head error of 2.44e-9 m and relative
water-balance error of 6.26e-10. The Sphinx HTML build passes with nitpicky
checks and warnings treated as errors (``-n -W --keep-going``).

The previous main checkout reproduced all six missing-analysis-tool failures;
restoring that tool resolves them. The unrelated solver-registry assertion
reported in the historical development log does not fail on main. The older
development timings and rejected experiments remain in
``docs/2D_RUNTIME_2026-11-12.rst`` and its linked evidence, separate from the
fresh release qualification.

The results qualify these synthetic reference cases and this GPU. The
largest gains concern the transient iteration strategy; small steady timing
differences should be treated as modest improvements or measurement variation.
The existing saturated-thickness floor and drying limitations are unchanged.
This release does not establish performance for every site model, untested
boundary type or arbitrarily changed convergence controls.

For the already released 3D work, see ``docs/3D_CARTESIAN_NPF.md``,
``docs/3D_IMPLEMENTATIONS.rst`` and
``docs/NPF662_RUNTIME_POLISH_2026-11-12.rst``. The last of these records the
prior main NPF strict-profile result of 0.972 s versus 104.391 s for a
500 x 500 x 5 convertible, 20-step history. That result is preserved
historical 3D qualification, not a new 3D benchmark in this promotion.
