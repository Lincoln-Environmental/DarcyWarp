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

The following modules live under ``tests/``. Their cases report which solver,
physical inputs and numerical tolerances were used.

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Module
     - What it checks
   * - ``test_2d_groundwater.py``
     - Confined PCG and K-cycle solutions, transient storage, signed recharge,
       wells, prescribed heads, GHB, DRN and RIV exchanges, Picard, Newton and
       FAS histories, and FP32 forcing through the production entry points.
   * - ``test_3d_groundwater.py``
     - Cartesian and layered flow, package fluxes, resident histories, K-cycle
       execution and source updates against analytical solutions.
   * - ``test_reference_equations.py``
     - Independent face, boundary, Jacobian and storage equations, including
       local conservation and inactive-cell separation.
   * - ``test_native_parity.py``
     - Matched MODFLOW 6.6.2 heads, package exchanges and storage at corresponding
       physical times.
   * - ``test_qualification_evidence.py``
     - Head-field coverage, integration over actual timesteps, gross package
       fluxes and missing evidence in benchmark comparisons.
   * - ``test_model_inputs.py``
     - Georeferenced input orientation, boundary stages, 3D model geometry,
       source units and transient restarts.

The original sparse-system tests compare the linear equations with an independent
finite-difference assembly. The original unconfined tests compare heads against
stored MF6 reference grids.

Running the checks
------------------

From the repository root in the project environment, run the default suite or
select individual test modules.

.. code-block:: bash

   PYTHONPATH=.:working_tests python -m pytest -q -ra
   PYTHONPATH=.:working_tests python -m pytest -q -ra tests/test_native_parity.py

Default collection includes the retained regional parity cases. To run the
regular CI selection, use ``python -m pytest -m "not regional_parity"``.
Use ``python -m pytest -m regional_parity`` on a high-memory runner for the
large saved grids and full transient histories.

The native parity module requires FloPy and the pinned MODFLOW executable.
Report any skips with the run. The solver tests use CUDA when available and
CPU otherwise. Use ``DARCYWARP_TEST_CPU_ONLY=1`` for a CPU-only run.

The dimensional runners select solver and reference modules that can run without
invoking native MF6. The 2D runner also exercises the stored unconfined reference
grids. They write JUnit XML to ``outputs/2d_gpu_tests.xml`` and
``outputs/3d_gpu_tests.xml``, respectively.

.. code-block:: bash

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python -m working_tests.run_2d_gpu_tests
   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python -m working_tests.run_3d_gpu_tests

The :doc:`release_2d` documents the historical device qualification and
full-history benchmarks. The :doc:`npf_gpu_native` records the 500 x 500 x 5
transient GPU qualification and the controls used to reproduce it.

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
