Quickstart
==========

Start with a small confined aquifer on the CPU. The example creates its own
arrays, solves for hydraulic head, checks the result against an analytical
solution and writes a figure. No external model data or MODFLOW executable
is needed.

Install the environment
-----------------------

Clone the repository, then create and activate the supplied conda environment:

.. code-block:: bash

   git clone https://github.com/Lincoln-Environmental/DarcyWarp.git
   cd DarcyWarp
   conda env create -f environment.yml
   conda activate darcywarp

Run commands from the repository root so Python can import
``DARCY_WARP_PACKAGE``. If the environment already exists, refresh it with
``conda env update -f environment.yml``. NVIDIA Warp is installed as
``warp-lang`` and imported as ``warp``.

The supplied environment also includes tools for the larger benchmark and
geospatial workflows. A supported NVIDIA GPU and driver are needed for
``device="cuda:0"``; the introductory example selects ``device="cpu"``.
MODFLOW comparisons additionally require the ``mf6`` executable; it is
separate from the FloPy Python package.

Run the first model
-------------------

.. code-block:: bash

   python -m examples.steady_recharge

The first run compiles Warp kernels. The script reports convergence, the
maximum head, the maximum analytical head error and the relative water-balance
error. For the default inputs, the expected maximum head is **11.25 m**.

The outputs are:

* ``outputs/steady_recharge/steady_recharge.npz``: model inputs, computed and
  analytical heads, coordinates and numerical checks.
* ``outputs/steady_recharge/steady_recharge.png`` and ``steady_recharge.svg``:
  head map and centreline comparison.

Problem definition
------------------

The example uses a 41 by 17 grid with 25 m square cells. Transmissivity is
0.001 m²/s and uniform recharge is 10⁻⁸ m/s. Fixed heads of 10 m apply at
the west/east boundary cell centres; the north/south outer faces have no flow.
All inputs use metres and seconds.

For constant transmissivity :math:`T`, recharge :math:`R`, equal boundary
heads :math:`h_0` and distance :math:`L` between their cell centres,

.. math::

   h(x) = h_0 + \frac{R}{2T}x(L-x).

Here :math:`L=(41-1)\times25=1000` m, so the maximum head is
:math:`10 + RL^2/(8T)=11.25` m. The source term is zero on prescribed-head
cells. The balance check compares recharge into the remaining cells with
discharge to the fixed-head cells.

The script requires a converged solution, absolute head error below 10⁻⁵ m
and relative water-balance error below 10⁻⁵. This is an introductory check
of a homogeneous confined case, rather than a test of every formulation.

Change explicit inputs
----------------------

Edit the ``main(...)`` call in ``examples/steady_recharge.py``, or call the
example from a fresh Python process:

.. code-block:: python

   from examples.steady_recharge import main

   summary = main(
       nx=41,
       ny=17,
       dx_m=25.0,
       transmissivity_m2_s=1.0e-3,
       recharge_m_s=1.0e-8,
       boundary_head_m=10.0,
       device="cpu",
       output_dir="outputs/my_first_model",
       make_plot=True,
   )

Set ``device="cuda:0"`` for a supported NVIDIA GPU, or ``make_plot=False``
to save only numerical results. The example selects FP64 before importing
the solver and requires ``DARCY_FLOAT=float64``. Precision is chosen at
import time, so restart Python before changing it.

For your own model, provide same-shaped ``(ny, nx)`` arrays to
``model.build_from_fields``: transmissivity, recharge, active-cell mask,
fixed-head mask and fixed-head values. Keep the time unit consistent across
transmissivity, recharge and any transient timestep. The example uses seconds;
other workflows may use days.

Next steps
----------

* :doc:`benchmarking` describes recharge/transmissivity ensembles, MODFLOW
  comparisons and plotting existing results.
* :doc:`docs_build` explains how to build the documentation locally.
* The solver guide in ``DARCY_WARP_PACKAGE/solvers/README.md`` lists the
  confined, unconfined and transient backend choices.
