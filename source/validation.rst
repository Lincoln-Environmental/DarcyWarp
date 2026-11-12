Verification and validation
===========================

DarcyWarp includes analytical checks, independent numerical comparisons and
regression tests. These answer different questions: whether an equation is
implemented correctly, whether two numerical models agree for matched inputs,
and whether a change preserves established behaviour.

This page is an index of the available evidence and how to reproduce it. It
does not report a fresh pass of every test or establish accuracy for an
uncalibrated field model.

Start with the analytical example
---------------------------------

The :doc:`quickstart` runs a confined aquifer with uniform recharge and
prescribed heads. Its simulated head profile is compared with the analytical
solution for the same geometry and boundary locations. This checks a simple
flow problem and demonstrates the API, outputs and units.

Numerical test coverage
-----------------------

The links point to the tests in the repository. Read the relevant test setup
and tolerances when interpreting a result.

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Area
     - Checks represented
     - Source
   * - Linear operator and public API
     - Sparse-system consistency, backend aliases, dispatch equivalence and
       deterministic resource cleanup.
     - `Sparse-system tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_sparse_system_consistency.py>`_;
       `registry tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_solver_registry_2d.py>`_.
   * - Unconfined steady flow
     - Saturated-thickness/transmissivity updates, near-dry cells and comparison
       with saved MODFLOW 6 head fixtures across several grids.
     - `Unconfined tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_2d_unconfined.py>`_.
   * - Transient flow and storage
     - Storage assembly, dense-reference comparison, head response to forcing,
       confined/unconfined budgets and timestep controls.
     - `Transient tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_2d_transient.py>`_.
   * - Nonlinear operator and Newton
     - Residual consistency, exact storage, Jacobian-vector checks, head and
       budget comparisons with Picard, and fallback behaviour.
     - `Nonlinear operator tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_nonlinear_operator_2d.py>`_;
       `Newton tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_semismooth_newton_2d.py>`_.
   * - Fast and mixed-precision paths
     - Fast/classic agreement, graph/eager agreement, FP32 correction
       comparisons with FP64, reductions and operator-cache refresh.
     - `Fast FP64 tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_fast_confined_kcycle.py>`_;
       `fast mixed-precision tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_mixed_precision_fast.py>`_.
   * - Transient replay
     - Warm-start provenance, storage budgets, strict versus practical
       convergence reporting and numerical acceptance rules.
     - `Replay tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_2d_transient_warm_start.py>`_.
   * - Experimental FAS
     - Multiple periods and timesteps, changed forcing/heads, rejected-step
       retries, fallback and comparison of full histories with Picard.
     - `FAS transient tests <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/tests/test_fas_transient_multistep.py>`_.

3D Cartesian and NPF/STO validation
-----------------------------------

The 3D tests cover Cartesian operator accuracy, layered analytical flow,
CPU/MF6 6.6.2 NPF/STO comparisons, transient storage and stepwise physical
acceptance, and device-resident multi-step histories.

* `tests/test_3d_operator_analytical.py` and
  `examples/verify_3d_series_flow.py`: Cartesian and layered analytical checks.
* `tests/test_npf662_confined.py`, `tests/test_npf662_layered_unconfined.py`
  and `tests/test_npf662_transient.py`: NPF/STO head, flux and storage checks.
* `tests/test_npf662_device_native.py` and
  `tests/test_npf662_history.py`: CPU/CUDA parity, device state reuse,
  changing K between histories, rejected timesteps and independent audits.
* `working_tests/run_3d_sanity.py`: portable comparisons with fingerprinted
  MF6 references, independent physical audits, head/flux gates and distinct
  timing intervals.

The :doc:`npf_gpu_native` page summarises the current 500 x 500 x 5
transient GPU qualification and the controls needed to reproduce it.
The checked-in GitHub Actions workflows are CPU smoke tests; they do not
repeat the GPU performance qualification at every commit.

Run the existing CI smoke tests from the repository root:

.. code-block:: bash

   python -m pytest tests/test_sparse_system_consistency.py tests/test_solver_registry_2d.py -v

For wider changes, select tests that exercise the affected formulation, device
and precision. CUDA-dependent tests skip when CUDA is unavailable. Some
comparison workflows also need saved fixtures, FloPy and a MODFLOW 6 executable.
A run containing skipped tests should be described with those exclusions.

The 2D GPU release suites can also be run together using saved references:

.. code-block:: bash

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python -m working_tests.run_2d_gpu_tests

This runner excludes tests that launch native MF6 and reports the exclusions.
The :doc:`release_2d` records the actual release run, GHB regressions and
separate full-history benchmarks. A green unit-test run alone does not
establish the runtime or MF6 parity of a new model.

MODFLOW 6 comparisons
---------------------

The
`unconfined reference fixtures <https://github.com/Lincoln-Environmental/DarcyWarp/tree/main/tests/fixtures/unconfined_2d>`_
support repeatable head comparisons without regenerating every reference run.
The larger workflows are kept under
`working_tests/ <https://github.com/Lincoln-Environmental/DarcyWarp/tree/main/working_tests>`_:

* ``regenerate_unconfined_2d_truth.py`` regenerates the unconfined reference data.
* ``run_2d_unconfined_warp_vs_mf6.py`` runs unconfined comparisons.
* ``run_2d_transient_vs_mf6.py`` and ``run_2d_transient_warp_replay.py`` support
  transient reference and replay workflows.
* ``run_2d_transient_sanity_matrix.py`` organises transient sanity cases.

Retained
`comparison results <https://github.com/Lincoln-Environmental/DarcyWarp/tree/main/data>`_
and
`benchmark data <https://github.com/Lincoln-Environmental/DarcyWarp/tree/main/DARCY_WARP_PACKAGE/data>`_
provide historical evidence. Check the recorded configuration, solver version,
hardware and precision before using those files to describe current behaviour
or performance.

A useful matched comparison records:

* Grid, domain mask, aquifer geometry and hydraulic properties.
* Sources, boundary conditions, initial heads and units.
* Stress periods, timestep lengths and storage formulation.
* Solver, precision, convergence criteria and any fallback or relaxed acceptance.
* Head error, spatial patterns of disagreement, and period/cumulative budgets.
* Commit identifiers, dependency versions and timing scope.

Agreement with MODFLOW 6 is evidence for the cases tested. An analytical
reference and independent mass-balance checks remain valuable because a second
numerical model can also contain errors or use different assumptions.

Performance evidence
--------------------

The :doc:`benchmarking` workflow varies recharge or transmissivity across
repeated confined solves. Those throughput measurements do not automatically
apply to a transient unconfined case, a different convergence target or a
different GPU.

Retain metadata and report whether timings include compilation, model setup,
data transfer, first solve, repeated solves and output. Compare equivalent
numerical accuracy before interpreting a speed ratio.

Original publication
--------------------

The original method is described by
`Durney and Fourie (2026) <https://doi.org/10.1016/j.envsoft.2026.107081>`_
in *Environmental Modelling & Software*, volume 204, article 107081. The paper
covers repeated steady, fixed-transmissivity solves. Subsequent unconfined,
transient and precision developments should be assessed using their relevant
repository tests and comparisons, within the scope of each case.
