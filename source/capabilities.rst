Model capabilities
==================

DarcyWarp provides 2D and 3D structured-grid solvers. The 2D solver represents
horizontal groundwater flow in a single thickness-integrated layer. Its
fields have shape ``(ny, nx)`` and require square cells.

Model representation
--------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Capability
     - Representation in the public 2D solver
   * - Confined flow
     - A spatial transmissivity field and a linear, five-point flow operator.
   * - Unconfined flow
     - Hydraulic conductivity and aquifer elevations; transmissivity changes
       with saturated thickness as head changes.
   * - Transient storage
     - Backward-Euler time integration. The unconfined workflows include
       specific yield and specific storage with aquifer bottom/top crossings.
   * - Sources and sinks
     - Spatial recharge/source arrays. Negative sources can represent
       withdrawals after converting volumetric rates to rates per cell area.
   * - Prescribed heads
     - A mask and head values for fixed-head cells.
   * - General-head boundaries
     - Head-dependent exchange through the model's conductance
       parameterisation.
   * - Domain masks
     - Active and inactive cells; unconnected grid faces do not transmit flow.
   * - Repeated scenarios
     - In-place updates of recharge, transmissivity and boundary heads, with
       reusable solver resources.
   * - Execution
     - Warp CPU and NVIDIA CUDA devices. GPU-specific graph and mixed-precision
       paths have additional requirements.

A general-head boundary can represent exchange with an external water body
whose head is prescribed. It does not evolve that water body's flow or stage.
Similarly, a 2D thickness-integrated model does not resolve vertical gradients
between multiple aquifer layers or variably saturated soil-water flow.

Use consistent units
--------------------

The solver accepts numerical arrays; it does not infer or convert their units.
A convenient SI convention is:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Quantity
     - Unit
   * - Grid spacing, head and aquifer elevations
     - m
   * - Hydraulic conductivity
     - m/s
   * - Transmissivity
     - m²/s
   * - Recharge or distributed withdrawal
     - m/s
   * - Time step
     - s
   * - Specific yield and confined storativity
     - Dimensionless
   * - Specific storage
     - m⁻¹

Existing benchmark scripts use their own consistent length/time conventions.
Convert all forcing, parameters and durations together when adapting them.
For a withdrawal of magnitude ``Q`` in m³/s spread over one cell, the
corresponding source rate is ``-Q / dx**2`` in m/s.

Solver choices
--------------

The authoritative capability definitions are in
`solver_capabilities.py <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/DARCY_WARP_PACKAGE/solver_capabilities.py>`_.
The names below can be selected through the public factory and solve API.

.. list-table::
   :header-rows: 1
   :widths: 43 17 17 23

   * - Backend
     - Formulation
     - Time mode
     - Role
   * - ``confined_kcycle``
     - Confined
     - Steady, transient
     - Default confined K-cycle backend.
   * - ``confined_pcg``
     - Confined
     - Steady only
     - Alternative linear solver.
   * - ``unconfined_picard_kcycle``
     - Unconfined
     - Steady, transient
     - Default unconfined backend.
   * - ``unconfined_semismooth_newton_kcycle``
     - Unconfined
     - Steady, transient
     - Explicitly selected nonlinear alternative.
   * - ``unconfined_fas``
     - Unconfined
     - Steady, transient
     - Experimental; selected explicitly.

Legacy names ``kcycle``, ``multigrid`` and ``mg`` select the confined K-cycle
backend for confined solves and the Picard/K-cycle backend for unconfined
solves. ``pcg`` remains an alias for confined PCG. A transient PCG request is
rejected because that backend does not support storage.

The multi-period unconfined driver defaults to Picard. Newton and FAS use the
alternate driver with timestep retries, recorded fallback behaviour and
per-period diagnostics. FAS remains experimental even though it supports
multiple periods. Backend status describes implementation maturity; model
accuracy still requires checks appropriate to the application.

Precision and performance
-------------------------

The standard solver module defaults to ``DARCY_FLOAT=float64``. DarcyWarp also
contains a fast confined FP64 path and a mixed-precision confined steady path
that retains a FP64 master solution while using FP32 corrections.

Mixed precision is an explicit execution choice, separate from the backend
names above. The ensemble command-line runners currently default to their
``mixed`` implementation; this does not make mixed precision the default for
every call to the public solver. The mixed implementation builds its model
under ``DARCY_FLOAT=float32`` and must run in a separate process from an
already-imported FP64 implementation. This process requirement describes the
confined steady mixed implementation.

The transient unconfined device path has a separate mixed correction option,
``transient_mixed_precision_enabled``. It runs under ``DARCY_FLOAT=float64``
and retains FP64 heads, storage and acceptance with FP32 inner corrections.
GHB requires the face operator. The :doc:`release_2d` documents the explicit
runtime profile, unchanged 1e-6 m head-residual gate and GHB-specific damping;
its benchmarks require matching MF6 timesteps as well as heads and budgets.

See :doc:`benchmarking` for the ``--warp_impl`` options and
:doc:`validation` for precision and backend comparison tests. Compare
performance only after checking convergence, head error and water balance
under the same model and forcing.

Inspecting a result
-------------------

Keep the returned head array and solver ``info`` together. For a transient
run, retain the period histories, convergence information, storage terms and
water-budget diagnostics. An available result array alone is not evidence of
convergence.

The public solver owns its device arrays and cached hierarchy. Use it as a
context manager or call ``close()`` in a ``finally`` block, as the quickstart
example does.


Structured 3D flow
------------------

Use ``WarpDarcySolver3D`` or ``create_solver(dim=3, ...)`` from
``DARCY_WARP_PACKAGE``. Fields use ``(nz, ny, nx)`` order. Legacy Cartesian
models use uniform layer thickness. The explicit ``mf6_6_6_2_npf`` formulation
supports both uniform grids and conventional layered DIS interfaces, with
confined or convertible cells and backward Euler Ss/Sy storage. 3D source
arrays contain volumetric rates per cell.

Both formulations also support cell-based general-head boundaries through
``ghb_conductance`` and ``ghb_head`` build inputs and ``update_ghb()``.
The :doc:`native_3d_ghb` page covers steady/transient GHB, budgets, the
Cartesian FGMRES option, explicit FP64 roundoff checks and MF6 benchmarks.

The `3D guide <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/docs/3D_CARTESIAN_NPF.md>`_
describes the public API, the layered example and CPU FP64 validation.
The :doc:`npf_gpu_native` page summarises the current accelerated
multi-step NPF history, strict and confined performance control files,
independent MF6 checks and timing boundaries. The fast history is
opt-in, and its measured runtime is not a library-default guarantee.
Transformed terrain remains experimental on ``dev``.
