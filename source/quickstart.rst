Quickstart
==========

Environment setup
-----------------

Create and activate the conda environment defined in ``environment.yml``:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate darcywarp

If the environment already exists, update it instead:

.. code-block:: bash

   conda env update -f environment.yml

Core runtime requirements
-------------------------

The benchmark scripts expect the following at runtime:

* Python environment that can import ``DARCY_WARP_PACKAGE``
* ``flopy`` for MODFLOW 6 workflows
* An ``mf6`` executable available on ``PATH`` (or in the configured location)
* Warp/GPU dependencies available for Warp benchmarks

What the benchmark workflow does
--------------------------------

The benchmark tooling can:

* generate synthetic ensemble inputs
* run recharge-change and transmissivity-change suites
* compare Warp, MODFLOW 6, and optional CPU FD implementations
* write JSON summaries for each solver/suite combination
* generate summary plots from existing JSON outputs

Documentation quick check
-------------------------

After installing Sphinx (included in ``environment.yml``), build the docs:

.. code-block:: bash

   make docs

or on Linux:

.. code-block:: bash

   bash scripts/generate_docs.sh
