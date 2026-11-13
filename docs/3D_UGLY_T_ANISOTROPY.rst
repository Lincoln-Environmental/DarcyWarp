3D ugly-T and vertical anisotropy qualification
============================================================

The ``ugly_t`` campaign reuses ``model_builder.make_ugly_T_field`` from
the 2D tests: correlated conductivity variation, a diagonal high-conductivity
channel and two low-conductivity lenses. Seed 42 and the original conversion
``K = T / 100 m`` are retained. Each layer uses the same map, multiplied by
``[1, 2/3, 4/3, 1/2, 5/6]`` to preserve the existing relative stratigraphy.

Within a cell, ``Kx=Ky=Kh`` and ``Kz=Kh/A``, with ``A=1, 10, 100``. Thus
this campaign isolates vertical anisotropy; the earlier smooth-field suite
separately covers unequal horizontal K components. Conductivity axes remain
aligned with the grid. These tests do not qualify rotated tensors, XT3D or
independently generated channel patterns in different layers.

The matrix contains 36 cases at each size: three anisotropy ratios, GHB
off/on, Cartesian confined or NPF confined/convertible, and steady/transient.
There are five layers. The NPF cases retain the existing varying-thickness
DIS geometry and 25 km domain. Cartesian retains its uniform 10 m layers
and 25 m horizontal spacing. Each formulation is compared with its own
matching MF6 model. Source, geometry, initial heads and boundary heads stay
fixed across the anisotropy sweep. Fixed GHB conductance depends on Kh and
is identical across the three ratios; changing Kz does not change it.

Transient cases start directly from the specified initial heads, without
a preliminary steady solve. The full campaign uses twenty seven-day
steps, with source multipliers 1, 1.5, 0.25 and 1 for five steps each,
``Ss=1e-5/m`` and ``Sy=0.15`` in convertible cells. Independent audits use
the matching timestep history and include storage and GHB exchange.

Controls and timing
-------------------

The matrix keeps the original MF6 2D-derived sanity profiles unchanged.
The preferred maximum head difference of ``1e-6 m`` is reported separately
from the existing sanity/conservation gates. A miss is not relabelled as
meeting the preferred target. Native failures, solver failures and missing
timed repetitions suppress speedups and remain in the results.

``sanity_3d_controls.read_profiles`` reads these MF6 controls directly from
the 2D source functions. All three profiles use ``DIAGONAL`` scaling and
``RELATIVE_RCLOSE=1e-6``; other IMS choices retain the selected complexity
profile's defaults.

.. list-table:: Original MF6 controls retained by this matrix
   :header-rows: 1

   * - Cases
     - Complexity / linear solver
     - Outer limit / head change (m)
     - Inner limit / head change (m)
   * - Confined steady
     - SIMPLE / CG
     - 10 / 1e-4
     - 2000 / 1e-4
   * - Convertible steady
     - COMPLEX / BICGSTAB
     - 500 / 1e-6
     - 500 / 1e-8
   * - All transient
     - COMPLEX / BICGSTAB
     - 150 / 1e-6
     - 500 / 1e-8

The inherited broad comparison limits are ``1e-3 m`` head RMSE and
``1e-2 m`` maximum head difference. Native references must also pass their
existing ``1%`` cellwise conservation check; GPU solves must satisfy their
own physical acceptance and GHB boundary-law checks. These broad sanity
limits do not establish ``1e-6 m`` head agreement: that stricter target
has its own result flag and every table reports the measured head error.

The default ``--gpu-profile sanity`` uses the original 2D Cartesian controls:
relative and minimum absolute residual tolerances ``5e-7``, RMS head-change
limit ``1e-4 m`` and maximum head-change limit ``5e-4 m``. NPF uses the
existing runtime profile's relative cellwise imbalance limit ``1e-7`` and
head-equivalent RMS limit ``1e-6 m``. Its dimensional ``1e-12 m3/day`` test
is an alternative to the relative imbalance test, not a mandatory cutoff.
No convergence criterion is tightened to ``1e-12`` for this default matrix.

Cartesian uses FGMRES with the five-level mixed-precision preconditioner. NPF uses
five levels for steady and confined transient cases, and three levels for
convertible transients. The public NPF history API is timed for transients.
Public solve timings include internal setup and acceptance, and exclude
external model construction and duplicate test audits. MF6 timings cover
its native process, excluding input generation and external decoding/audits.
That process includes MF6's own assembly and output writing.

Runs are serial to avoid CPU/GPU contention. The large campaign measures
three repetitions after one warmup. The 100 x 100 x 5 smoke campaign uses
three timesteps and one timed repetition to check all combinations before
the 500 x 500 x 5 campaign. Smoke timings are not production medians.

The benchmark host has an Intel Core Ultra 9 285K and an NVIDIA GeForce
RTX 4070 Ti SUPER with 16 GB of device memory. Runs use FP64 heads and
physical residuals, Warp 1.17.0 and the pinned historical MF6 6.6.2 build.
``OMP_NUM_THREADS``, ``OPENBLAS_NUM_THREADS`` and ``MKL_NUM_THREADS`` are
set to one; the drivers do not set CPU affinity. The reported speedups apply to
this hardware and the recorded solver profiles. The separate MF6 strategy
diagnostics below retain the baseline convergence tolerances.

Practical-control smoke qualification
----------------------------------------

All 36 cases at 100 x 100 x 5 pass with the default practical controls.
Twenty-four meet the preferred ``1e-6 m`` maximum head-difference target.
The largest MF6 differences for Kh/Kv of 1, 10 and 100 are
``6.858e-6``, ``6.300e-6`` and ``5.934e-6 m``, respectively. They are
essentially unchanged from the extra-tight experiment. The largest GPU
head-equivalent RMS residual is ``3.232e-10 m``, well below ``1e-6 m``.
The revised field/matrix and unavailable-reference regression selection
passes 29 tests.

* :download:`Practical-control smoke matrix <../benchmark_results/ugly_t_3d/smoke_100_sanity_qualification.json>`

Large-model baseline results
-----------------------------

All 36 large GPU cases pass the independent physical checks. 32 have valid comparisons with the original MF6 profiles; 4 additional comparisons use the separately recorded strategy below. Together, all 36 comparisons pass the sanity gates. 9 meet the preferred ``1e-06 m`` maximum head target. The largest head difference is ``5.023e-05 m``. The largest GPU head-equivalent RMS is ``2.453e-11 m``.

36 of 36 planned large cases have completed. 36 pass the independent GPU physical checks; 32 pass the full MF6 comparison gates. 6 meet the preferred ``1e-6 m`` maximum head-difference target.

The tables contain three-repeat GPU medians and single native MF6 process times. Build and extra test audits are excluded as described above. Unavailable references have no speedup or head-parity claim.

.. list-table:: Convertible NPF: 500 x 500 x 5, twenty steps for transients
   :header-rows: 1

   * - Kh/Kv
     - GHB
     - Mode
     - GPU (s)
     - MF6 (s)
     - MF6/GPU
     - Maximum head difference (m)
   * - 1
     - Off
     - Steady
     - 1.289
     - 229.647
     - 178.1
     - 5.023e-05
   * - 1
     - Off
     - Transient
     - 2.049
     - 178.780
     - 87.3
     - 5.415e-07
   * - 1
     - On
     - Steady
     - 1.191
     - Unavailable
     - Unavailable
     - Unavailable
   * - 1
     - On
     - Transient
     - 2.266
     - 202.817
     - 89.5
     - 3.562e-07
   * - 10
     - Off
     - Steady
     - 1.585
     - 444.232
     - 280.3
     - 3.473e-06
   * - 10
     - Off
     - Transient
     - 1.803
     - 165.223
     - 91.6
     - 2.093e-07
   * - 10
     - On
     - Steady
     - 1.223
     - Unavailable
     - Unavailable
     - Unavailable
   * - 10
     - On
     - Transient
     - 1.919
     - 199.738
     - 104.1
     - 2.638e-07
   * - 100
     - Off
     - Steady
     - 1.688
     - 393.366
     - 233.0
     - 2.991e-06
   * - 100
     - Off
     - Transient
     - 2.228
     - 158.880
     - 71.3
     - 1.638e-07
   * - 100
     - On
     - Steady
     - 1.240
     - Unavailable
     - Unavailable
     - Unavailable
   * - 100
     - On
     - Transient
     - 2.454
     - 191.525
     - 78.0
     - 2.153e-07

.. list-table:: Confined NPF: 500 x 500 x 5, twenty steps for transients
   :header-rows: 1

   * - Kh/Kv
     - GHB
     - Mode
     - GPU (s)
     - MF6 (s)
     - MF6/GPU
     - Maximum head difference (m)
   * - 1
     - Off
     - Steady
     - 1.224
     - 23.891
     - 19.5
     - 1.317e-05
   * - 1
     - Off
     - Transient
     - 1.809
     - 630.953
     - 348.8
     - 1.412e-05
   * - 1
     - On
     - Steady
     - 1.220
     - 18.905
     - 15.5
     - 1.854e-05
   * - 1
     - On
     - Transient
     - 1.891
     - 602.198
     - 318.5
     - 5.370e-06
   * - 10
     - Off
     - Steady
     - 1.744
     - 18.871
     - 10.8
     - 1.344e-05
   * - 10
     - Off
     - Transient
     - 2.332
     - 464.305
     - 199.1
     - 5.081e-06
   * - 10
     - On
     - Steady
     - 1.230
     - 14.842
     - 12.1
     - 2.128e-05
   * - 10
     - On
     - Transient
     - 1.997
     - 440.574
     - 220.6
     - 2.010e-06
   * - 100
     - Off
     - Steady
     - 1.477
     - 17.849
     - 12.1
     - 1.240e-05
   * - 100
     - Off
     - Transient
     - 2.366
     - 349.138
     - 147.6
     - 1.708e-06
   * - 100
     - On
     - Steady
     - 1.218
     - 14.607
     - 12.0
     - 1.229e-05
   * - 100
     - On
     - Transient
     - 2.223
     - 313.862
     - 141.2
     - 1.574e-06

.. list-table:: Confined Cartesian: 500 x 500 x 5, twenty steps for transients
   :header-rows: 1

   * - Kh/Kv
     - GHB
     - Mode
     - GPU (s)
     - MF6 (s)
     - MF6/GPU
     - Maximum head difference (m)
   * - 1
     - Off
     - Steady
     - 0.749
     - 26.600
     - 35.5
     - 3.401e-05
   * - 1
     - Off
     - Transient
     - 7.624
     - Unavailable
     - Unavailable
     - Unavailable
   * - 1
     - On
     - Steady
     - 0.649
     - 21.375
     - 33.0
     - 3.380e-05
   * - 1
     - On
     - Transient
     - 7.891
     - 788.422
     - 99.9
     - 1.827e-05
   * - 10
     - Off
     - Steady
     - 0.648
     - 19.016
     - 29.3
     - 3.280e-05
   * - 10
     - Off
     - Transient
     - 7.638
     - 748.731
     - 98.0
     - 1.036e-05
   * - 10
     - On
     - Steady
     - 0.636
     - 15.765
     - 24.8
     - 3.550e-05
   * - 10
     - On
     - Transient
     - 8.022
     - 640.897
     - 79.9
     - 7.024e-06
   * - 100
     - Off
     - Steady
     - 1.189
     - 16.785
     - 14.1
     - 3.623e-05
   * - 100
     - Off
     - Transient
     - 10.209
     - 504.471
     - 49.4
     - 3.821e-06
   * - 100
     - On
     - Steady
     - 0.681
     - 14.069
     - 20.7
     - 3.721e-05
   * - 100
     - On
     - Transient
     - 8.466
     - 381.407
     - 45.1
     - 1.589e-06

4 completed case(s) have no valid complete MF6 reference. The retained evidence records these failures independently from GPU convergence.

The large field has Kh from ``2.240`` to ``679.324 m/day``, a contrast of 303.3.

The Kh/Kv=1 GHB-on steady MF6 reference reached its unchanged 1,800-second process limit without convergence. The GPU solve passed with relative cellwise imbalance ``1.230e-8`` and head-equivalent RMS ``1.867e-12 m``.

The Kh/Kv=10 GHB-on steady MF6 reference also reached the 1,800-second limit without convergence. Its GPU solve passed the physical checks at a 1.223 s median.

The Kh/Kv=100 GHB-on steady MF6 reference likewise reached the 1,800-second limit without convergence. Its GPU solve passed the physical checks at a 1.240 s median.

* :download:`Complete large-model results and source hashes <../benchmark_results/ugly_t_3d/large_500_progress.json>`

Initial extra-tight experiment
------------------------------

The initial experiment used ``--gpu-profile roundoff``. All 36 cases at
100 x 100 x 5 passed the recorded sanity and independent
physical checks. Twenty-three met the preferred maximum head difference
of ``1e-6 m``. The largest differences for Kh/Kv of 1, 10 and 100 were
``6.858e-6``, ``6.300e-6`` and ``5.933e-6 m``, respectively. The largest
GPU head-equivalent RMS residual was ``9.405e-16 m``. Measured horizontal
conductivity ranges from ``2.374`` to ``699.364 m/day`` (contrast 294.6).

The field/matrix regression selection passed 27 tests. These check exact
reuse of the 2D field, directional face-conductance scaling, preservation
of other model inputs, invariant GHB across the ratio sweep, distinct cache
identities, invalid inputs and refusal of speedups after failed jobs.

* :download:`Smoke matrix and provenance <../benchmark_results/ugly_t_3d/smoke_100_qualification.json>`

The initial extra-tight Cartesian steady case at 500 x 500 x 5
at Kh/Kv=1 and GHB off reached head-equivalent RMS ``4.855e-16 m`` but
missed the fixed dimensional ``1e-12 m3/day`` cutoff, with residual RMS
``4.352e-12 m3/day`` after 4,000 iterations. It remains a failed strict
qualification; its timing is not advertised as a converged speedup.
The practical campaign uses the original 2D-matched GPU controls above.
The initial experiment
is retained separately and can be reproduced with ``--gpu-profile roundoff``.

The original 500 x 500 x 5 Cartesian transient reference at Kh/Kv=1 with
GHB off reached the external 1,800-second MF6 process limit after saving
six of twenty timesteps. Its IMS settings were unchanged. That incomplete
history is not a valid full-history reference or a qualified timing
comparison; the native files and interrupted strict-profile attempt are
retained. GPU physical checks and MF6 availability are reported separately.

Separate MF6 strategy diagnostics
----------------------------------------

Missing native references were investigated separately, preserving the
original baseline artifacts and MF6 convergence tolerances. The diagnostic
harness checks complete matching time histories, verifies the GHB flux law,
and independently re-audits the saved GPU heads using their original physical
limits. Eight focused diagnostic regressions pass.

The first alternative used no under-relaxation or backtracking, eight ILUT
levels and RCM ordering. The pinned historical MF6 6.6.2 executable exited
with code 174 in ``genrcm`` (``rcm.f90``) before solving the large Kh/Kv=1
GHB-on steady case. This attempt provides no qualifying timing or parity
result. The baseline suite retains its original solver strategy.

Repeating that diagnostic with the original ordering completed normally:
``77.389 s`` MF6 versus the saved ``1.191 s`` GPU median, a ratio of 65.0.
The physical case, initial heads and MF6 convergence tolerances were unchanged.
Maximum head difference is ``2.744e-6 m``; head RMSE is ``1.484e-6 m``.
The reference passes the existing sanity gates, and fresh audits of all four
saved GPU histories pass their original physical limits and the GHB-law
comparison. This establishes an independent comparison for the baseline's
timed-out Kh/Kv=1 GHB-on steady case, while still missing the preferred
``1e-6 m`` maximum head-difference target. This diagnostic timing is separate
from the unchanged baseline table above.

The same original-order strategy recovers the Kh/Kv=10 and Kh/Kv=100
GHB-on convertible steady references. Native MF6 takes ``87.107`` and
``57.678 s``, respectively, versus saved GPU medians of ``1.223`` and
``1.240 s``. Maximum head differences are ``4.038e-7`` and ``6.337e-7 m``:
both meet the preferred ``1e-6 m`` target with unchanged convergence
tolerances. Their saved GPU histories pass fresh physical and GHB-law checks.

For the missing Cartesian transient reference, this strategy reached its
600-second diagnostic limit after saving 15 of 20 timesteps. That attempt
is retained as a failure. A separate confined CG/ILU(0) diagnostic completes
the full history in ``737.054 s``, versus the saved GPU median of ``7.624 s``.
It retains the original convergence tolerances and allows 1,200 seconds
for the native process. All four saved GPU histories pass fresh physical
checks; maximum head difference is ``2.757e-7 m``, meeting the preferred target.

* :download:`MF6 diagnostic experiments and provenance <../benchmark_results/ugly_t_3d/mf6_strategy_experiments.json>`

Final comparisons for missing original references
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Separate MF6 strategy; original convergence tolerances retained
   :header-rows: 1

   * - Formulation / mode
     - Kh/Kv / GHB
     - GPU (s)
     - Diagnostic MF6 (s)
     - MF6/GPU
     - Maximum head difference (m)
   * - cartesian confined / transient / CG
     - 1 / off
     - 7.624
     - 737.054
     - 96.7
     - 2.757e-07
   * - npf convertible / steady / BICGSTAB
     - 1 / on
     - 1.191
     - 77.389
     - 65.0
     - 2.744e-06
   * - npf convertible / steady / BICGSTAB
     - 10 / on
     - 1.223
     - 87.107
     - 71.2
     - 4.038e-07
   * - npf convertible / steady / BICGSTAB
     - 100 / on
     - 1.240
     - 57.678
     - 46.5
     - 6.337e-07

* :download:`Resolved comparison CSV <../benchmark_results/ugly_t_3d/large_500_comparisons.csv>`

Reproduction
------------

From the repository root with CUDA, Warp, FloPy and pinned MF6 6.6.2::

    PYTHONPATH=.:working_tests DARCY_FLOAT=float64 python \
      working_tests/run_3d_ugly_t_matrix.py \
      --sizes 100 --periods 3 --repeats 1 \
      --gpu-profile sanity \
      --output benchmark_results/ugly_t_3d/smoke_100_sanity

    PYTHONPATH=.:working_tests DARCY_FLOAT=float64 python \
      working_tests/run_3d_ugly_t_matrix.py \
      --sizes 500 --periods 20 --repeats 3 \
      --gpu-profile sanity \
      --output benchmark_results/ugly_t_3d/large_500_sanity

The runner sets one CPU thread for BLAS/OpenMP/MKL. ``--mf6-timeout`` is
an external process limit, not an IMS convergence setting. References are
fingerprinted by physical inputs, schedules and unchanged MF6 options.
Job reuse additionally checks the recorded code, controls, executable,
Python, repetition count and results hash. Repeating a command resumes
completed identical jobs; failed attempts are retained. ``--retry-failed``
requests another attempt. Use a new cache directory if a native run left
an incomplete reference cache.

If MF6 cannot supply a valid complete reference, the matrix still measures
the GPU solve and runs the independent physical audit. Such rows explicitly
report unavailable MF6 head/flow comparisons, remain failed comparisons,
and have no qualified speedup. GPU physical acceptance is recorded separately.

For a single case, ``run_3d_sanity.py`` accepts ``--k-field ugly_t``,
``--vertical-anisotropy``, ``--ugly-t-seed`` and
``--ugly-t-reference-thickness``. The default ``layered`` conductivity
keeps previous benchmark inputs unchanged. Every new result records the
actual directional K ranges, contrast and measured Kh/Kv ratio.

The baseline command intentionally exits nonzero when an original MF6 reference is unavailable, even if every GPU physical check passes. To investigate those references separately without changing convergence tolerances::

    PYTHONPATH=.:working_tests DARCY_FLOAT=float64 python \
      working_tests/diagnose_3d_mf6_references.py \
      --matrix benchmark_results/ugly_t_3d/large_500_sanity/matrix.json \
      --expected-cases 36 --timeout 600 \
      --output benchmark_results/ugly_t_3d/mf6_strategy_diagnostics_original_order

The diagnostic verifies baseline hashes and matching time grids and re-audits every saved GPU history. It uses no GPU heads to initialise MF6 and does not rerun or retime the GPU. Original failures remain in the baseline report.

For the separately qualified confined fallback::

    PYTHONPATH=.:working_tests DARCY_FLOAT=float64 python \
      working_tests/diagnose_3d_confined_mf6.py \
      --matrix benchmark_results/ugly_t_3d/large_500_sanity/matrix.json \
      --size 500 --ratio 1 --ghb off --family cartesian --mode transient \
      --timeout 1200 --output benchmark_results/ugly_t_3d/mf6_confined_cg_diagnostic

The confined command selects exactly one missing reference and rejects
convertible cases, failed GPU physical checks and changed source snapshots.
The steady MF6 adapter now preserves distinct timeout, process-failure and
missing-output reasons. This is a reporting correction; the benchmark's
numerical settings and production solver algorithms are unchanged.

The final affected regression selection passes 52 tests on ``dev``, including
live MF6 adapter checks, unchanged-tolerance guards, corrupt-history rejection,
confined-only diagnostic selection and all three native failure causes.
The strict Sphinx documentation build passes.

Main promotion verification
---------------------------

Fresh checks on ``main`` at ``9631ce8`` pass all 52 affected tests
and all 36 small-grid GPU/MF6 cases (100 x 100 x 5, three transient
steps, one timed repeat). The strict Sphinx build passes.

The main large-model rerun covers six cases at 500 x 500 x 5, Kh/Kv=100
and GHB on: Cartesian confined and NPF confined/convertible, each steady
and transient. Transients use twenty weekly steps; each case has one
warmup and three timed GPU repetitions. All six pass independent physical
checks. Five use their saved original MF6 references. Convertible steady
uses a fresh, separately labelled original-order ILUT strategy diagnostic
with unchanged convergence tolerances; the failed original reference is
preserved. This six-case rerun is distinct from the full 36-case large
campaign on dev reported above.

.. list-table:: Fresh main large-model timings, Kh/Kv=100 and GHB on
   :header-rows: 1

   * - Formulation / mode
     - GPU (s)
     - MF6 (s)
     - MF6/GPU
     - Maximum head difference (m)
     - Reference
   * - cartesian confined / steady
     - 0.679
     - 14.069
     - 20.7
     - 3.721e-05
     - Saved original
   * - cartesian confined / transient
     - 8.522
     - 381.407
     - 44.8
     - 1.589e-06
     - Saved original
   * - npf confined / steady
     - 1.238
     - 14.607
     - 11.8
     - 1.229e-05
     - Saved original
   * - npf confined / transient
     - 2.225
     - 313.862
     - 141.1
     - 1.574e-06
     - Saved original
   * - npf convertible / steady
     - 1.262
     - 57.336
     - 45.4
     - 6.337e-07
     - Separate diagnostic
   * - npf convertible / transient
     - 2.442
     - 191.525
     - 78.4
     - 2.153e-07
     - Saved original

* :download:`Main qualification evidence <../benchmark_results/ugly_t_3d/main_qualification.json>`

Reproduce the main rerun with the matrix commands above, using
``--vertical-anisotropy 100 --ghb-modes on`` for the large selection and
a separate output directory. Run ``diagnose_3d_mf6_references.py`` against
that large matrix with ``--expected-cases 6 --timeout 600`` to reproduce
the separate native diagnostic.
