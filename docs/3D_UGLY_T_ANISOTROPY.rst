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

GPU controls use the explicit FP64 roundoff qualification profile. Cartesian
uses FGMRES with the five-level mixed-precision preconditioner. NPF uses
five levels for steady and confined transient cases, and three levels for
convertible transients. The public NPF history API is timed for transients.
Public solve timings include internal setup and acceptance, and exclude
external model construction and duplicate test audits. MF6 timings cover
its native process, excluding input generation and external decoding/audits.

Runs are serial to avoid CPU/GPU contention. The large campaign measures
three repetitions after one warmup. The 100 x 100 x 5 smoke campaign uses
three timesteps and one timed repetition to check all combinations before
the 500 x 500 x 5 campaign. Smoke timings are not production medians.

Completed smoke qualification
-----------------------------

All 36 cases at 100 x 100 x 5 passed the recorded sanity and independent
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

The large campaign is in progress. Its extra-tight Cartesian steady case
at Kh/Kv=1 and GHB off reached head-equivalent RMS ``4.855e-16 m`` but
missed the fixed dimensional ``1e-12 m3/day`` cutoff, with residual RMS
``4.352e-12 m3/day`` after 4,000 iterations. It remains a failed strict
qualification; its timing is not advertised as a converged speedup.

Reproduction
------------

From the repository root with CUDA, Warp, FloPy and pinned MF6 6.6.2::

    PYTHONPATH=.:working_tests DARCY_FLOAT=float64 python \
      working_tests/run_3d_ugly_t_matrix.py \
      --sizes 100 --periods 3 --repeats 1 \
      --output benchmark_results/ugly_t_3d/smoke_100

    PYTHONPATH=.:working_tests DARCY_FLOAT=float64 python \
      working_tests/run_3d_ugly_t_matrix.py \
      --sizes 500 --periods 20 --repeats 3 \
      --output benchmark_results/ugly_t_3d/large_500

The runner sets one CPU thread for BLAS/OpenMP/MKL. ``--mf6-timeout`` is
an external process limit, not an IMS convergence setting. References are
fingerprinted by physical inputs, schedules and unchanged MF6 options.
Job reuse additionally checks the recorded code, controls, executable,
Python, repetition count and results hash. Repeating a command resumes
completed identical jobs; failed attempts are retained. ``--retry-failed``
requests another attempt. Use a new cache directory if a native run left
an incomplete reference cache.

For a single case, ``run_3d_sanity.py`` accepts ``--k-field ugly_t``,
``--vertical-anisotropy``, ``--ugly-t-seed`` and
``--ugly-t-reference-thickness``. The default ``layered`` conductivity
keeps previous benchmark inputs unchanged. Every new result records the
actual directional K ranges, contrast and measured Kh/Kv ratio.
