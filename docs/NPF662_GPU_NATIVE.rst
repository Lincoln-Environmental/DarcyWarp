NPF transient GPU execution
===========================

The current development qualification is described in ``NPF662_RUNTIME_SEARCH.rst``
below: **0.957 s** for the strict 20-step convertible history, or **0.931 s**
with a case-specific MF6-comparable physical accuracy envelope. Those changes
are on ``dev``; the following sections preserve the earlier promoted results.

The opt-in transient NPF path reduces the 500 x 500 x 5 convertible model's
20-step solve time from 30.24 s to 9.74 s on an RTX 4070 Ti SUPER. The saved
CPU MODFLOW 6 run of the same problem took 104.39 s: a 10.7-fold speedup.
This opt-in implementation is now available on main. The default remains unchanged.

Main branch rerun
-----------------

Qualification at main revision ``495cdd3``: 526 tests passed, 43 live-MF6
cases deselected. All 320 GPU endpoints independently passed. Saved MF6
references were reused. Timing boundaries and tolerances match the experiment below.

.. list-table:: Main, twenty weekly steps; GPU medians of three repeats
   :header-rows: 1

   * - Model
     - GPU seconds
     - MF6 seconds
     - MF6/GPU
   * - confined_250x250x5
     - 3.250
     - 47.589
     - 14.64x
   * - convertible_250x250x5
     - 3.257
     - 17.319
     - 5.32x
   * - confined_500x500x5
     - 6.864
     - 346.408
     - 50.47x
   * - convertible_500x500x5
     - 9.784
     - 104.391
     - 10.67x

Evidence: ``benchmark_results/gpu_native/main_comparison.json`` and
``main_source_manifest.json``. The development measurements below are retained
as the before-promotion record.

Development comparison
----------------------

All times below exclude model construction. GPU time is the sum of
synchronized public ``model.solve()`` calls across 20 weekly timesteps,
reported as the median of three repeats after one warmup. MF6 time is one
saved engine run, excluding FloPy construction and reference validation.

.. list-table:: Convertible NPF/STO, 1.25 million cells
   :header-rows: 1

   * - Implementation
     - Seconds
   * - CPU MODFLOW 6.6.2
     - 104.391
   * - Previous GPU implementation
     - 30.243
   * - Updated GPU implementation
     - 9.740

The updated timed repeats were 9.710, 9.740 and 9.798 s. Each repeat
starts with a fresh model; workspace and hierarchy setup inside its first
solve are included. Extra independent CPU audits take a median 20.382 s
per repeat and are outside the production solve timer. The median full
verification time is 30.517 s. Model construction takes 0.226 s separately.
The benchmark retains all these timings in JSON and CSV.

The problem spans 25 km square with five variable-thickness DIS layers.
The upper layer is convertible, with Ss=1e-5 and Sy=0.15. Four five-week
source blocks use multipliers 1, 1.5, 0.25 and 1. Conductivity is fixed in
this timing experiment; changing conductivity is covered by CPU/CUDA
regressions and remains supported through ``update_K_fields``.

All 80 endpoints, including warmup, passed independent physical checks.
The maximum head difference from the saved MF6 history is 6.36e-8 m.
Both GPU implementations use FP64, a 1e-6 m head-equivalent RMS gate,
and the explicitly selected 1e-7 relative cellwise imbalance limit.
The imbalance is sum(abs(cell residual)) divided by gross source,
boundary exchange and storage throughput; it is not a maximum per-cell
relative error. The default relative limit remains 1e-8.

MF6 uses the actual 2D transient sanity IMS profile: COMPLEX/BICGSTAB,
outer head close 1e-6, inner head close 1e-8 and RELATIVE_RCLOSE 1e-6.
These are distinct convergence norms; their numerical values do not imply
identical achieved accuracy. Saved references are fingerprinted against
the physical inputs and MF6 controls before reuse.

Reproduce
---------

Use Python 3.12, Warp 1.17.0 and a CUDA device. From the repository root::

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python working_tests/run_3d_sanity.py \
       --sizes 500 --families npf --modes transient --variants convertible \
       --periods 20 --repeats 3 --npf-preconditioner kcycle \
       --npf-relative-tolerance 1e-7 --npf-device-native \
       --cache benchmark_results/mf6_3d_cache \
       --output benchmark_results/npf_native_convertible_500

The first invocation generates the MF6 reference if necessary. Add
``--cached-only`` to require a matching saved reference. Output directories
must be new. Use ``--sizes 16 32`` for a quick convergence ladder; each size
is checked against MF6 and independent physical equations. Omit
``--npf-device-native`` for the previous path. Without that flag, the same
runner also supports steady NPF, confined and phreatic NPF variants, and
classic/fast Cartesian comparisons through ``--compare``.

Release details
---------------

Enable ``npf_device_native=True`` on transient NPF ``model.solve()`` calls,
using ``implementation="fast"`` and ``solver="kcycle"`` on the model.
For the measured implementation also select ``npf_preconditioner="kcycle"``
and ``npf_final_relative_tolerance=1e-7``. The runner supplies the remaining
published benchmark controls; the defaults are not silently changed.

The implementation borrows the 2D solver's structure/value refresh design:

* Timestep state owns snapshots without assembling a sparse CPU reference
  on the positive-storage production route. The existing FP64 device
  storage kernel prepares the previous-state tangent.
* K-cycle levels, device buffers and captured graphs survive storage and
  conductivity value refreshes when geometry and connection topology match.
  Conductivity changes refresh physical faces and hierarchy values; they
  are not frozen or lagged across timesteps.
* Device physical reductions retain the original residual and head gates.
  A final check verifies prescribed heads, finite derivatives and positive
  storage grounding in every free cell. Cases without that grounding
  certificate fall back to the independent CPU acceptance path.

This is a step toward GPU-native execution. Input validation, snapshots,
some coefficient preparation and hierarchy value refresh still use the
host. The benchmark's independent sparse CPU audit remains outside the
timer. ``independent_audit_performed`` reports whether the solver itself
needed that audit; device checks are reported as ``physical_acceptance``
and are not labelled independent.

Validation: 1,608 affected 3D/terrain/NPF tests passed; 52 tests requiring
new native MF6 runs were deselected. The saved realistic MF6 history was
reused without a new engine run. Focused CPU/CUDA tests cover independent
physical agreement, absence of sparse-reference construction on the
positive-storage route, conductivity/dt refresh with stable buffers and
graphs, owned previous-state tangents, and dry ungrounded fallback.

Compact timing evidence and source hashes are stored in
``benchmark_results/gpu_native/comparison.json`` and
``benchmark_results/gpu_native/source_manifest.json``. Raw histories and
heads remain local, outside version control.

Further runtime work promoted to main
-------------------------------------

These changes were developed on dev at ``bf59a2a`` and promoted to main
as ``cd32e71``. Main passed 539 tests with 43 live-MF6 cases deselected.
The following timings were measured on dev before that promotion.
The repeated 500 x 500 x 5 convertible benchmark now takes 5.916 s
per 20-step history (median of three). Its Newton/Krylov interval is 3.835 s;
setup and final acceptance outside that interval remain in the public solve total.
All 80 independently audited endpoints pass; maximum MF6 head difference 6.35e-08 m.
Tolerances, precision and source schedule are unchanged.

Calibration runtime is assessed separately: each of three physical K fields
(0.5, 1 and 2 times the original K) drives a fresh 20-step history on reused
models. K changes between histories. Each row is one paired run, not a
repeat median. Both implementations use the same K, heads, forcing and gates.
Device results pass independent physical audits and agree with the previous
solver. No MF6 speedup is claimed for the changed-K cases.

.. list-table:: Numerical solver seconds, excluding setup and K updates
   :header-rows: 1

   * - K multiplier
     - Previous solver
     - Device reductions
     - Time reduction
   * - 0.5
     - 4.587
     - 3.180
     - 30.7%
   * - 1
     - 5.433
     - 3.853
     - 29.1%
   * - 2
     - 5.794
     - 4.189
     - 27.7%

Iteration findings:

* Device Arnoldi scalar updates and CUDA graph replay cut readbacks but
  saved only about 1% on these cases. That result is retained in the evidence.
* Blocked physical residual reductions remove contention on nine global
  counters during Newton residual evaluations. This produced the main numerical-runtime gain.
* Coarse conductances, fixed-head springs, storage and diagonals now refresh
  on the GPU for K value changes with unchanged connection topology. This
  is a separate setup improvement, checked against fresh hierarchies on
  odd grids, inactive masks, semi-coarsening and anisotropic K updates.

The implementation still uses host input validation and finest physical-face
preparation, small FGMRES rotations/backsolves, and stopping decisions.
Topology changes may rebuild the hierarchy. Uncertified storage grounding
retains the independent CPU fallback. It is not a wholly device-resident API.

Reproduce the calibration comparison from the repository root::

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests \
     OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
     python working_tests/benchmark_npf_calibration_runtime.py \
       --output benchmark_results/calibration_runtime_new

Model size, K factors, source schedule and solver controls are adjustable
in that script's main block. It reports K update, setup, public solve and
numerical times separately. The fixed-K comparison uses the earlier
``run_3d_sanity.py`` command with a new output directory.

Validation: 1,616 broad regression tests passed (52 new native MF6 tests
deselected), followed by 23 focused CPU/CUDA tests including multiple
reduction blocks and an injected CUDA graph capture failure.

Compact results and current source hashes: ``benchmark_results/gpu_native/runtime_progress.json``.
