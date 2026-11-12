Current 3D NPF/STO GPU execution
=================================

DarcyWarp ``main`` includes an opt-in, device-resident transient history for
the MF6 6.6.2 NPF/STO formulation. It retains accepted physical states and
intermediate work on the device, reuses multigrid structures, and downloads
the completed head history at the end of the call.

This page describes the current public capability. The dated reports linked
below preserve earlier measurements and experimental optimisation work.

Qualified transient performance
-------------------------------

The recorded qualification used an NVIDIA RTX 4070 Ti SUPER (16 GiB),
Python 3.12, Warp 1.17.0 and a saved MODFLOW 6.6.2 reference.
Each model had 500 x 500 x 5 cells (1.25 million cells), 20 weekly
timesteps and no preliminary steady-state solve.

.. list-table:: Twenty-step histories; GPU median of three and one saved MF6 engine run
   :header-rows: 1
   :widths: 28 17 17 13 25

   * - NPF/STO model, strict profile
     - DarcyWarp GPU (s)
     - MODFLOW 6 (s)
     - MF6 / GPU
     - Maximum head difference (m)
   * - Convertible
     - 0.972
     - 104.391
     - 107x
     - 6.84e-8
   * - Confined
     - 1.378
     - 346.408
     - 251x
     - 3.45e-6

DarcyWarp timing covers the synchronized public ``solve_transient_npf()``
call, including internal setup, all source/timestep updates, physical
acceptance and the output download. External model construction and the
independent benchmark audit are excluded. The saved MF6 timing measures one
engine execution, including its output but excluding FloPy construction.
The ratios therefore compare these specific intervals, not end-to-end
model-building or independently audited workflows.

All 400 endpoints in the recorded final qualification passed independent
physical audits and MF6 head gates at matching timesteps. The strict
runtime profile explicitly selects a ``1e-7`` relative cellwise-imbalance
limit and retains the ``1e-6 m`` head-equivalent RMS gate. The solver's
library defaults are unchanged. The measured GPU path uses FP64 for heads,
storage, fluxes, FGMRES and acceptance, with an explicitly selected FP32
multigrid preconditioner.

The complete profile table, input fingerprints, per-run timings, external
construction, additional audits and qualifications are in the
`final main qualification report <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/docs/NPF662_RUNTIME_POLISH_2026-11-12.rst>`_
and its `machine-readable results <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/benchmark_results/gpu_native/polish_main_20261112.json>`_.

Using the accelerated history
-----------------------------

Build a ``mf6_6_6_2_npf`` model using ``WarpDarcySolver3D`` or
``create_solver(dim=3, ...)`` and call ``solve_transient_npf()`` with source
multipliers and a scalar or vector of timestep lengths. The model's base
source is multiplied by each source factor. The output array has shape
``(number_of_steps, nz, ny, nx)`` when every step is accepted.

Geometry, conductivity, storage and prescribed heads remain fixed within
a single history call. Use ``model.update_K_fields()`` to change K between
history calls. Inspect ``info["converged"]`` and ``info["accepted_steps"]``
before using heads: after a failed solve, the returned history can include
the final rejected trial for diagnosis.

The `structured 3D guide <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/docs/3D_CARTESIAN_NPF.md>`_
contains a complete model-building example and a multi-step history call.

For the measured profile, use the checked-in controls rather than
assuming the library defaults will achieve the same performance:

* ``working_tests/npf_runtime_controls.json``: strict convertible profile.
* ``working_tests/npf_confined_runtime_controls.json``: strict confined profile.
* ``working_tests/npf_mf6_accuracy_controls.json``: explicit MF6-comparable
  physical-accuracy profile, with a looser relative imbalance limit than
  the strict profile.

Reproduce the comparison
------------------------

Run from the repository root with the matching Python/Warp environment and
a CUDA device. The command below requires an existing, validated MF6 cache;
omit ``--cached-only`` and provide ``--mf6`` if you must generate a new
reference using the pinned MODFLOW 6 executable.

.. code-block:: bash

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/run_3d_sanity.py \
     --device cuda:0 --families npf --variants convertible \
     --modes transient --sizes 500 --periods 20 --repeats 3 \
     --npf-history --npf-controls working_tests/npf_runtime_controls.json \
     --cached-only --cache benchmark_results/mf6_3d_cache \
     --output outputs/npf_gpu_current

For the confined result, use `--variants confined` and
``working_tests/npf_confined_runtime_controls.json``. The runner
compares the full head history with the saved MF6 reference, audits
physical conservation and records timing intervals separately.

Scope and archived reports
--------------------------

The supported NPF/STO formulation has confined or convertible cells on
uniform Cartesian or conventional layered DIS grids, default MF6 6.6.2
NPF connections and backward-Euler Ss/Sy storage. XT3D, VARIABLECV,
PERCHED, rewetting, alternative STO options and transformed terrain are
outside this qualified implementation. Results vary with geometry,
forcing, numerical controls, hardware and problem size.

Older experiments are preserved as separate technical evidence, not
combined into this current guide:

* `Initial 3D GPU qualification <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/docs/3D_GPU_RELEASE_NOTES.md>`_
  (early 7 to 30-second solve-call comparisons).
* `NPF GPU-native development <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/docs/NPF662_GPU_NATIVE.rst>`_
  (subsequent 3 to 10-second histories).
* `First main history qualification <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/docs/NPF662_MAIN_2026-11-12.rst>`_
  (earlier main revision).
* `Runtime search and experimental tuning <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/docs/NPF662_RUNTIME_SEARCH.rst>`_.

For related numerical tests, see :doc:`validation`.
