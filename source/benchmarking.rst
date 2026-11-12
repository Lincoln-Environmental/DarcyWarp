Benchmarking
============

The 2D ensemble benchmark workflows compare repeated confined solves while
changing recharge or transmissivity. They run DarcyWarp, MODFLOW 6 and an
optional SciPy-based finite-difference reference, then write JSON summaries
and plots. The current 3D NPF/STO GPU transient comparison, its recorded
timing boundaries and independent MF6 checks are in :doc:`npf_gpu_native`.

For a first model, use :doc:`quickstart`. The commands below are ensemble
benchmarks and can be substantially larger than the introductory example.

For the current fast 2D steady and transient comparison with saved MF6 results,
see :doc:`release_2d` and :doc:`runtime_2d`. Those drivers report synchronized
public solve time separately from external construction and independent audits,
and reject histories that fail convergence, head, budget or timestep parity.

Environment
-----------

Run commands from the repository root in the ``darcywarp`` conda environment.
The full environment includes Warp, NumPy, SciPy, FloPy and Matplotlib.
MODFLOW runs additionally require the ``mf6`` executable; the project searches
``/bin/modflow/mf6`` and then ``PATH``. Installing FloPy alone does not install
that executable.

The examples use ``cuda:0`` for GPU benchmarks. The minimal quick start uses
CPU execution and does not require MODFLOW.

Run benchmarks and plots
------------------------

The wrapper runs both the recharge-change and transmissivity-change suites.
With no component flags, it selects Warp, MODFLOW 6 and the CPU reference:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.bench_and_plot \
     --nx 1000 --ny 1000 --dx 100.0 \
     --n_cases 48 --workers 2,4,8,16,24 \
     --seed 42 --device cuda:0 \
     --write_metadata --force \
     --out_dir paper/tables_figures

Add ``--ghb`` to include the synthetic general-head boundaries.

The following switches select components or suites:

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Behaviour
   * - ``--run_warp``
     - Run DarcyWarp.
   * - ``--run_mf6``
     - Run MODFLOW 6.
   * - ``--run_fd``
     - Run the CPU finite-difference reference.
   * - ``--run_recharge``
     - Run the recharge-change suite.
   * - ``--run_t``
     - Run the transmissivity-change suite.
   * - ``--plots_only``
     - Read existing summaries without running benchmarks.
   * - ``--write_metadata``
     - Save the configuration, settings and output paths for each suite.
   * - ``--force``
     - Rerun selected MODFLOW/CPU reference components even if summaries exist.

If no component flags are supplied, all three components run. If neither suite
flag is supplied, both suites run.

For example, run Warp alone for the recharge suite:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.bench_and_plot \
     --run_warp --run_recharge \
     --nx 1000 --ny 1000 --dx 100.0 --n_cases 48 \
     --seed 42 --device cuda:0 --write_metadata \
     --out_dir paper/tables_figures

Use ``--run_mf6`` or ``--run_fd`` instead of ``--run_warp`` to select either
reference component. Multiple component flags can be combined.

Existing results and plot requirements
--------------------------------------

The wrapper always reruns a selected Warp component. By default, it reuses
existing MODFLOW and CPU-reference summaries when their expected filenames are
present. Those filenames are keyed by total cell count, not the full model
configuration.

Use ``--force`` for a fresh matched comparison after changing grid shape, cell
size, seed, case count, boundaries, parameters or software. Preserve the earlier
results separately if they are needed. Metadata helps establish provenance;
the wrapper does not use it to validate a cached result.

The plotting wrapper requires an MF6 summary and adds a Warp comparison only
when the corresponding Warp summary exists. A Warp-only benchmark can therefore
complete without producing comparison plots. If a retained MF6 file is found,
check that it matches the new Warp run before interpreting the plot.

Direct benchmark commands
-------------------------

Use the suite modules to run without plotting or to select the Warp precision
implementation. These modules expose options that the wrapper does not forward.

Recharge-change suite:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.model_benchmarking_recharge_change \
     --run_warp --run_mf6 \
     --nx 1000 --ny 1000 --dx 100.0 \
     --n_cases 48 --workers 2,4,8,16,24 --seed 42 \
     --device cuda:0 --warp_impl mixed --write_metadata

Transmissivity-change suite:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.model_benchmarking_T_change \
     --run_warp --run_mf6 \
     --nx 1000 --ny 1000 --dx 100.0 \
     --n_cases 48 --workers 2,4,8,16,24 --seed 42 \
     --device cuda:0 --warp_impl mixed --warp_update_mode full \
     --write_metadata

Both direct suites accept:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Option
     - Meaning
   * - ``--nx``, ``--ny``
     - Number of columns and rows.
   * - ``--dx``
     - Square grid-cell size, consistent with the model's length units.
   * - ``--n_cases``
     - Number of ensemble cases.
   * - ``--workers``
     - Comma-separated MODFLOW/CPU process counts. Requested counts are capped
       by the available CPUs and number of cases.
   * - ``--seed``
     - Random seed for repeatable fields.
   * - ``--device``
     - Warp device string.
   * - ``--ghb``
     - Include the synthetic general-head boundaries.
   * - ``--warp_impl``
     - ``mixed`` (default), ``fast`` or ``classic``.
   * - ``--mg_min_coarse_cells``
     - Coarsest-level cell-count control.
   * - ``--write_metadata``
     - Write the additional metadata JSON.

``classic`` uses the FP64 K-cycle path; ``fast`` uses the fast FP64 path;
``mixed`` retains a FP64 master solution with fast FP32 corrections. The mixed
path creates its model under ``DARCY_FLOAT=float32``. Use separate processes
for mixed and FP64 runs so that imported kernel types remain consistent.

The T-change module requires ``--warp_update_mode full`` for the mixed path.
Its ``fast`` and ``ultrafast`` update modes are experimental and require an
additional explicit opt-in. They are not the default benchmark configuration.
Use the module's ``--help`` for advanced controls.

Shared model inputs
-------------------

The recharge suite generates one stack with shape ``(n_cases, ny, nx)`` from
the seed, base recharge and jitter. Every enabled solver receives the same
arrays for each case index. The T-change suite similarly generates a
deterministic transmissivity stack.

This input sharing applies to the components run together. It does not
establish compatibility with a summary retained from an earlier invocation.

Outputs
-------

Summaries are written to ``DARCY_WARP_PACKAGE/data/`` through
``project_base.data_store``. In the names below, ``{cells} = nx * ny``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Suite/component
     - Filename
   * - Recharge / MF6
     - ``mf6_ensemble_benchmark_results_recharge{cells}.json``
   * - Recharge / Warp
     - ``warp_class_ensemble_benchmark_results_recharge_{cells}.json``
   * - Recharge / CPU FD
     - ``fd_ensemble_benchmark_results_recharge{cells}.json``
   * - Transmissivity / MF6
     - ``mf6_T_ensemble_benchmark_results_{cells}.json``
   * - Transmissivity / Warp
     - ``warp_class_T_ensemble_benchmark_results_{cells}.json``
   * - Transmissivity / CPU FD
     - ``fd_T_ensemble_benchmark_results_{cells}.json``
   * - Recharge metadata
     - ``benchmark_metadata_recharge_{cells}.json``
   * - Transmissivity metadata
     - ``benchmark_metadata_T_{cells}.json``

The wrapper writes figures below ``--out_dir`` in ``recharge_change/`` and
``t_change/``. Each suite produces:

* ``throughput_vs_workers.png``: cases per second versus MF6 worker count.
* ``walltime_vs_workers.png``: total ensemble wall time versus worker count.
* ``idealised_completion_curves.png``: completion curves derived from total
  wall time, rather than measured per-case completion timestamps.

Plot existing summaries
-----------------------

Through the wrapper:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.bench_and_plot \
     --plots_only --nx 1000 --ny 1000 \
     --out_dir paper/tables_figures

Or select exact summary files:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.benchmark_plots \
     --mf6_summary DARCY_WARP_PACKAGE/data/mf6_ensemble_benchmark_results_recharge1000000.json \
     --warp_summary DARCY_WARP_PACKAGE/data/warp_class_ensemble_benchmark_results_recharge_1000000.json \
     --out_dir paper/tables_figures \
     --title_prefix "1000x1000, N=48: "

``--mf6_summary`` and ``--out_dir`` are required. Omit ``--warp_summary`` to
plot MF6 results alone. ``--title_prefix`` adds a label to all plot titles.

Troubleshooting
---------------

**No Warp line in the comparison plots.**
Check that the matching Warp summary exists and that the wrapper selected the
intended file. A MODFLOW-only run does not produce a Warp summary.

**No comparison plots after a Warp-only run.**
This is expected when no MF6 summary is available. The numerical benchmark
summary is still written.

**MODFLOW worker cannot unpack a model-builder result.**
The template-writing workflow should call ``make_mf_model(..., run=False)``
without unpacking its return value, load the written simulation using
``flopy.mf6.MFSimulation.load(...)``, then obtain and update its recharge
package. This is relevant when adapting the persistent-worker scripts.

**Results appear to belong to an older configuration.**
Check the metadata and summary paths. Run with ``--force`` for new reference
results and retain configuration-specific copies when comparing multiple cases.

See :doc:`validation` for the evidence needed to interpret numerical agreement
and performance.
