DarcyWarp Documentation
=======================

DarcyWarp provides benchmark scripts and plotting utilities for comparing a
Warp-based solver against MODFLOW 6 (and an optional CPU finite-difference
solver) across repeated ensemble runs.

This documentation covers the current benchmark workflow, output files, and the
Sphinx documentation build process used in this repository.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   quickstart
   benchmarking
   docs_build
   license

Key entrypoints
---------------

The most commonly used command-line modules are:

* ``python -m DARCY_WARP_PACKAGE.bench_and_plot``
* ``python -m DARCY_WARP_PACKAGE.model_benchmarking_recharge_change``
* ``python -m DARCY_WARP_PACKAGE.model_benchmarking_T_change``
* ``python -m DARCY_WARP_PACKAGE.benchmark_plots``

Repository layout (high level)
------------------------------

* ``DARCY_WARP_PACKAGE/``: benchmark scripts, solver code, plotting utilities
* ``source/``: Sphinx documentation source
* ``build/``: generated Sphinx outputs (created by ``make docs``)
* ``paper/``: manuscript and generated figures/tables
* ``scripts/``: helper scripts (including Linux docs build wrapper)
