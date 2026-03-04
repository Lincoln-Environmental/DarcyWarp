Benchmarking Workflows
======================

Wrapper CLI (benchmarks + plots)
--------------------------------

The main wrapper CLI runs benchmark suites and then generates plots from the
resulting summary JSON files:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.bench_and_plot \
     --nx 1000 \
     --ny 1000 \
     --dx 100.0 \
     --n_cases 48 \
     --workers 2,4,8,16,24 \
     --seed 42 \
     --device cuda:0 \
     --out_dir paper/tables_figures

Behavior notes (current implementation)
---------------------------------------

* If none of ``--run_warp``, ``--run_mf6``, or ``--run_fd`` are provided, the
  wrapper runs all three.
* If neither ``--run_recharge`` nor ``--run_t`` is provided, the wrapper runs
  both suites.
* ``--plots_only`` skips benchmark execution and only generates plots from
  existing summary JSONs.
* ``--write_metadata`` writes optional metadata JSON files for each suite.

Common examples
---------------

Enable GHB boundaries:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.bench_and_plot --ghb --out_dir paper/tables_figures

Run only Warp:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.bench_and_plot \
     --run_warp --run_recharge --run_t --device cuda:0

Generate plots only from existing summaries:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.bench_and_plot \
     --plots_only \
     --nx 1000 --ny 1000 \
     --out_dir paper/tables_figures

Direct benchmark CLIs
---------------------

Use these when you want to run benchmark suites without invoking plotting:

Recharge-change suite:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.model_benchmarking_recharge_change \
     --run_warp --run_mf6 \
     --nx 1000 --ny 1000 --dx 100.0 \
     --n_cases 48 --workers 2,4,8,16,24 --seed 42 --device cuda:0

Transmissivity-change suite:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.model_benchmarking_T_change \
     --run_warp --run_mf6 \
     --nx 1000 --ny 1000 --dx 100.0 \
     --n_cases 48 --workers 2,4,8,16,24 --seed 42 --device cuda:0

Plotting existing summaries
---------------------------

Generate plots directly from summary JSONs:

.. code-block:: bash

   python -m DARCY_WARP_PACKAGE.benchmark_plots \
     --mf6_summary /path/to/mf6_ensemble_benchmark_results_recharge1000000.json \
     --warp_summary /path/to/warp_class_ensemble_benchmark_results_recharge_1000000.json \
     --out_dir paper/tables_figures \
     --title_prefix "1000x1000, N=48: "

Typical outputs
---------------

Benchmark summaries are written to the package data store (``DARCY_WARP_PACKAGE/data``
via ``project_base.data_store``) and plots are written to the ``--out_dir``
you pass to the CLI.

Common summary filename patterns include:

* ``mf6_ensemble_benchmark_results_recharge{cells}.json``
* ``warp_class_ensemble_benchmark_results_recharge_{cells}.json``
* ``fd_ensemble_benchmark_results_recharge{cells}.json`` (if CPU FD enabled)
* ``mf6_T_ensemble_benchmark_results_{cells}.json``
* ``warp_class_T_ensemble_benchmark_results_{cells}.json``
* ``fd_T_ensemble_benchmark_results_{cells}.json`` (if CPU FD enabled)

Recharge consistency
--------------------

The recharge benchmark workflow generates a single recharge stack and reuses it
across all enabled solvers for the same run, so each solver sees the same input
arrays for each case index.
