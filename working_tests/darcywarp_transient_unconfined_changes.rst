DarcyWarp 2D Transient and Unconfined Solver Extensions
=======================================================

Overview
--------

The original DarcyWarp 2D implementation solved a steady groundwater-flow
system using a fixed transmissivity field and a five-point finite-difference
operator. It supported CUDA execution, multigrid K-cycles, a confined PCG
path, fixed-head boundaries, recharge, and general-head boundary terms.

The extended implementation adds a complete transient and unconfined pathway
while retaining the original steady-state solver. The main additions are:

* backward-Euler storage for confined and unconfined calculations;
* nonlinear unconfined transmissivity updates based on saturated thickness;
* exact old-to-new specific-yield and convertible specific-storage terms;
* a multi-period 2D transient unconfined API;
* a device-side Picard and K-cycle fast path;
* adaptive inexact inner solves;
* final fine-grid nonlinear residual checking;
* transient storage diagnostics and water-budget terms; and
* expanded convergence, timing, and data-transfer diagnostics.

The steady-state no-storage path remains available and uses dedicated kernels
that do not carry transient storage arrays through the original workload.

Baseline steady-state equation
------------------------------

The original 2D solver represents the integrated cell equation as

.. math::

   A(h) h = b,

where the fixed-transmissivity confined operator for an active free cell is

.. math::

   (A h)_C =
   \left(T_E + T_W + T_N + T_S + C_{gh}\right) h_C
   - T_E h_E - T_W h_W - T_N h_N - T_S h_S.

Neighbour transmissivities use harmonic averaging. Recharge enters the
right-hand side as an integrated cell flow:

.. math::

   b_R = R\,\Delta x^2.

Fixed-head and inactive cells retain identity-row treatment. General-head
boundary conductance is included as both a diagonal operator term and an
external-head right-hand-side term.

Transient backward-Euler formulation
-------------------------------------

The transient extension adds a storage diagonal to each active non-Dirichlet
cell:

.. math::

   D_s = S_{eff}\frac{\Delta x^2}{\Delta t}.

The backward-Euler equation is

.. math::

   \left(A + D_s\right)h^{n+1}
   = R^{n+1}\Delta x^2 + D_s h^n.

The device and host paths therefore assemble

.. code-block:: python

   rhs_eff = recharge_rate * dx * dx + storage_diag * head_prev

for active free cells. Inactive cells are zeroed and fixed-head cells retain
identity rows with ``rhs_eff = bc_values``.

The same effective right-hand side is used by the linear solve, fine-grid
residual calculation, and final nonlinear acceptance check. This avoids the
previous possibility of solving one equation and testing convergence against
a differently scaled right-hand side.

Unconfined transmissivity
-------------------------

For the 2D unconfined formulation, transmissivity is updated from the current
Picard head:

.. math::

   T(h) = K\,b_{sat}(h),

with

.. math::

   b_{sat}(h) =
   \operatorname{clip}\left(h-z_b, b_{min}, z_t-z_b\right).

Here, :math:`K` is hydraulic conductivity, :math:`z_b` is aquifer bottom,
:math:`z_t` is aquifer top, and :math:`b_{min}` is the configured minimum
saturated thickness.

The minimum saturated thickness is a numerical regularisation. It prevents a
zero transmissivity row and supports stable nonlinear iteration. It does not
represent a fully dry or deactivated cell. Applications that approach the
cell bottom should test sensitivity to ``min_saturated_thickness``.

Exact convertible storage
-------------------------

Specific yield
~~~~~~~~~~~~~~

The drainable-storage term uses the exact old-to-new secant coefficient. Define
zero-based saturated thickness as

.. math::

   b_0(h) = \operatorname{clip}\left(h-z_b, 0, z_t-z_b\right).

The specific-yield coefficient for a Picard reference head :math:`h^k` is

.. math::

   S_{y,sec} = S_y
   \frac{b_0(h^k)-b_0(h^n)}{h^k-h^n}.

Consequently,

.. math::

   S_{y,sec}(h^k-h^n)
   = S_y\left[b_0(h^k)-b_0(h^n)\right],

which exactly represents the discrete specific-yield volume change. The
formulation handles heads within the aquifer and crossings through the top or
bottom. When the head difference is numerically negligible, the local
derivative is used.

Convertible specific storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Convertible specific storage is represented using a storage potential rather
than the former current-thickness approximation. The potential per unit plan
area is

.. math::

   \Phi_{ss}(h) =
   \begin{cases}
   0, & h \le z_b,\\
   \frac{1}{2}S_s(h-z_b)^2, & z_b < h < z_t,\\
   \frac{1}{2}S_sB^2 + S_sB(h-z_t), & h \ge z_t,
   \end{cases}

where :math:`B=z_t-z_b`.

The exact secant coefficient is

.. math::

   S_{ss,sec} =
   \frac{\Phi_{ss}(h^k)-\Phi_{ss}(h^n)}{h^k-h^n}.

For a negligible head difference, the derivative is used:

.. math::

   \frac{d\Phi_{ss}}{dh} = S_s b_0(h).

The effective storage coefficient is

.. math::

   S_{eff}=S_{y,sec}+S_{ss,sec}.

This formulation is used by both the NumPy host implementation and the Warp
CUDA kernel. It provides exact old-to-new storage-volume consistency for
rising and falling heads, including transitions between partially saturated
and fully saturated conditions.

The compatibility storage-mode name
``"mf6_convertible_secant_sy"`` is retained, although the implementation now
contains both exact secant :math:`S_y` and exact secant :math:`S_s` terms.

Nonlinear Picard solution
-------------------------

Each unconfined transient period is solved through a Picard sequence:

#. Start from the previous-period head or a configured startup solution.
#. Update saturated thickness and transmissivity from the current Picard head.
#. Recompute exact specific-yield and specific-storage secant coefficients.
#. Assemble the fine-grid storage diagonal and transient right-hand side.
#. Refresh the device-side multigrid operator values and diagonal
   preconditioners.
#. Apply an inexact multigrid K-cycle solve.
#. Relax and clip the candidate head update.
#. Measure head change and fine-grid flow and head-equivalent residuals.
#. Repeat until strict convergence or configured practical acceptance is met.

Available startup modes include use of the supplied initial head and a
confined pre-solve. Update relaxation, maximum per-iteration head change,
Chebyshev nonlinear damping, and transmissivity relaxation remain available
as solver controls.

Adaptive inexact inner solves
-----------------------------

The transient device path does not solve every early Picard linearisation to
the maximum K-cycle count. The inner cycle cap is selected from the previous
outer head change. The standard schedule is:

.. list-table:: Adaptive inner K-cycle schedule
   :header-rows: 1

   * - Picard phase
     - Default cap
     - Selection criterion
   * - Early
     - 10
     - First iteration or large previous head change
   * - Middle
     - 25
     - Intermediate previous head change
   * - Late
     - 60
     - Small previous head change

The replay-level ``max_cycles`` value is not passed unchanged into each Picard
iteration. This prevents a transient period from performing hundreds of full
recursive K-cycles before the nonlinear state has stabilised.

When ``return_scalar_info=False`` is used by the fast path, the inner K-cycle
runs without per-cycle host convergence downloads. The outer Picard check
performs the required scalar reductions after the selected inner work is
complete.

Multigrid operator treatment
----------------------------

Transient storage is included in the diagonal on every multigrid level.
Fine-cell storage diagonals are summed into each coarse cell:

.. math::

   D_{s,c}=\sum_{i\in c}D_{s,i}.

Summation preserves integrated storage capacity, including partial edge
blocks. Transmissivity is rediscretised using harmonic aggregation.

The coarse hierarchy is used only as an acceleration mechanism. Production
acceptance is based on the refreshed fine-grid nonlinear operator. The coarse
operators are therefore approximate preconditioners and are not presented as
an exact Galerkin representation of the fine-grid transient equation.

The production CUDA K-cycle uses a guarded recursive correction labelled
``recursive_kcycle_safe_alpha``. Unsafe non-positive or very small curvature
terms are rejected rather than being used to create an unbounded coarse
correction.

The standalone reference K-cycle utility uses restart-free FGMRES for coarse
Krylov acceleration. This replaces ordinary PCG around a recursive variable
preconditioner, for which the fixed linear preconditioner assumptions of PCG
are not guaranteed.

Device-side transient fast path
-------------------------------

The new multi-period fast path keeps the nonlinear transient workflow on the
GPU. Within a period it updates:

* uniform recharge;
* fine-grid transmissivity;
* exact storage coefficients and storage diagonal;
* coarse transmissivity and storage values;
* diagonal preconditioners; and
* the effective transient right-hand side.

The path avoids unconditional full-grid downloads inside the Picard loop. A
full head field is downloaded once per period. Additional coefficient and
storage arrays are downloaded only when full diagnostics are requested.

The fast path tracks data movement, hierarchy refreshes, scalar reductions,
head downloads, K-cycle counts, and phase timings. These counters make it
possible to distinguish numerical work from Python or GPU synchronisation
overhead.

Convergence and production acceptance
-------------------------------------

Dual residual measures
~~~~~~~~~~~~~~~~~~~~~~

The transient solver distinguishes the integrated flow residual

.. math::

   r_f=b-Ah

from the head-equivalent residual

.. math::

   r_h=\frac{b-Ah}{\operatorname{diag}(A)}.

RMS values are calculated over active non-Dirichlet cells only. The diagonal
includes neighbour conductances, supported general-head boundary conductance,
and transient storage.

``final_flow_residual_rms`` retains the physical integrated-flow residual for
diagnostics. ``final_head_residual_rms`` is the head-scale measure used for
nonlinear acceptance.

Final nonlinear residual refresh
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Before a period is accepted, the solver rebuilds transmissivity, exact storage
coefficients, the storage diagonal, and the right-hand side from the final
candidate head. The final residual therefore describes the same nonlinear
state that is saved as the period result.

This check is independent of whether the approximate coarse-grid solve reports
internal convergence.

Strict and practical acceptance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Strict Picard convergence requires both maximum head change and the refreshed
head-equivalent residual to satisfy their strict tolerances.

A practical production criterion is also available for cases where the
fine-grid equation and RMS nonlinear update have converged sufficiently but a
small number of cells prevent the strict maximum-change criterion from being
met. Period diagnostics distinguish:

* ``strict_picard_convergence_passed``;
* ``practical_picard_acceptance_passed``; and
* ``production_acceptance_passed``.

An unaccepted period raises an error by default. The solver does not silently
advance through later stress periods after exhausting the permitted Picard
iterations. A diagnostics-only override can be enabled explicitly through
``allow_unaccepted_transient_period``.

The practical storage-diagonal change threshold is currently dimensional and
should be treated as model-configuration specific. A relative storage-change
criterion is preferable when applying the defaults across different grid
sizes, time units, or timestep lengths.

Public API changes
------------------

Unified single-solve interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The new ``solve()`` method selects formulation, solver, and transient state:

.. code-block:: python

   head, info = solver.solve(
       formulation="unconfined",
       solver="kcycle",
       initial_head=head_old,
       K_field=k_field,
       zbot_field=bottom,
       ztop_field=top,
       transient=True,
       dt=7.0,
       head_prev=head_old,
       sy=0.2,
       ss=1.0e-5,
       unconfined_storage_mode_2d="mf6_convertible_secant_sy",
       storage_reference="current_picard",
   )

The confined PCG pathway remains steady-state only. A transient PCG request
raises ``NotImplementedError`` so storage cannot be silently omitted.

Multi-period transient unconfined interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``solve_transient_2d_unconfined()`` provides the production multi-period API:

.. code-block:: python

   controls = {
       "use_device_transient_fast_path": True,
       "nu_pre": 3,
       "nu_post": 3,
       "nu_coarse": 1,
       "max_levels": 4,
       "unconfined_startup_mode": "confined_pre_solve",
       "unconfined_inner_max_cycles_early": 10,
       "unconfined_inner_max_cycles_middle": 25,
       "unconfined_inner_max_cycles_late": 60,
       "practical_picard_acceptance_enabled": True,
   }

   heads, info = solver.solve_transient_2d_unconfined(
       initial_head=head_initial,
       recharge_rates=recharge_by_period,
       k_field=k_field,
       zbot_field=bottom,
       ztop_field=top,
       sy=0.2,
       ss=1.0e-5,
       dt=7.0,
       active=active,
       bc_mask=constant_head_mask,
       bc_values=constant_head_values,
       storage_reference="current_picard",
       solve_controls=controls,
       min_saturated_thickness=0.1,
       save_diagnostics=False,
       return_info=True,
   )

``recharge_rates`` currently contains one spatially uniform recharge value per
period. Spatially varying recharge can still be staged through the lower-level
field update and single-solve interfaces.

The returned head array has shape ``(n_periods, ny, nx)``. The information
dictionary contains per-period solver summaries, period timings, transfer
counters, and optional full-grid storage diagnostics.

Diagnostics
-----------

Period summaries include the following groups of information.

Convergence
~~~~~~~~~~~

* Picard outer-iteration count;
* strict, practical, and production acceptance states;
* maximum and RMS head change;
* final flow-residual RMS;
* final head-equivalent residual RMS; and
* storage-diagonal change maximum and RMS.

Solver work
~~~~~~~~~~~

* total inner K-cycles;
* maximum inner K-cycles in one Picard iteration;
* selected cycle caps when detailed diagnostics are enabled;
* coarse-operator mode;
* coarse correction method; and
* fine-grid residual-check status.

Performance
~~~~~~~~~~~

* transmissivity update time;
* storage assembly time;
* diagonal-preconditioner refresh time;
* dynamic coarse refresh time;
* right-hand-side assembly time;
* inner solver time;
* outer convergence-check time;
* final nonlinear residual-check time;
* head download time; and
* total period time.

Optional full-grid diagnostics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``save_diagnostics=True``, the solver returns old heads, storage reference
heads, total storage coefficients, separate :math:`S_y` and :math:`S_s`
coefficients, and exact storage-rate terms for each period. These arrays are
not downloaded in the normal production path.

Transient water budget
----------------------

The exact storage diagnostics support a transient water budget based on actual
old-to-new volume change rather than a linearised coefficient evaluated at one
head. Per unit plan area,

.. math::

   \Delta V_y = S_y\left[b_0(h^{n+1})-b_0(h^n)\right]

and

.. math::

   \Delta V_{ss} =
   \Phi_{ss}(h^{n+1})-\Phi_{ss}(h^n).

The storage flow into the groundwater equation is

.. math::

   Q_{STO} =
   -\frac{\left(\Delta V_y+\Delta V_{ss}\right)\Delta x^2}{\Delta t}.

Positive ``STO`` represents water released from storage into groundwater.
Negative ``STO`` represents water entering storage. The replay budget combines
this term with recharge, fixed-head exchange, and supported head-dependent
boundaries.

Validation
----------

The production configuration using ``nu_pre=3``, ``nu_post=3``,
``nu_coarse=1``, and ``max_levels=4`` was compared against the corresponding
MODFLOW 6 transient unconfined replay.

.. list-table:: Validation summary
   :header-rows: 1

   * - Metric
     - Result
   * - Final maximum absolute head difference
     - 0.0002641 m
   * - Final head RMSE
     - 0.00005823 m
   * - Worst-period maximum absolute difference
     - 0.0002890 m, period 8
   * - Worst-period RMSE
     - 0.00006616 m
   * - Cumulative mass-balance discrepancy
     - 0.0002242 percent
   * - Maximum period mass-balance discrepancy
     - 0.0006019 percent
   * - Total runtime
     - 14.42 seconds
   * - Total Picard outer iterations
     - 288
   * - Production result
     - Practical acceptance passed

The target replay did not satisfy the strict maximum-change Picard criterion,
but it passed the refreshed fine-grid head-residual and practical nonlinear
criteria. The sub-millimetre MODFLOW 6 differences and very small transient
budget discrepancy independently support the accepted solution.

Compatibility and retained behaviour
------------------------------------

The following original behaviour is retained:

* steady fixed-transmissivity K-cycle solves;
* steady confined PCG solves;
* harmonic intercell transmissivity;
* fixed-head identity rows;
* recharge and general-head boundary assembly;
* host and device right-hand-side backends;
* hierarchy construction and memory reuse; and
* explicit solver cleanup and context-manager support.

Storage-free specialised CUDA kernels preserve the original steady-state
operator without adding transient storage reads to its inner loops.

Known limitations
-----------------

General-head boundaries
~~~~~~~~~~~~~~~~~~~~~~~

The device transient unconfined fast path currently rejects general-head
boundary right-hand-side assembly with ``NotImplementedError``. This explicit
failure prevents unsupported boundary terms from being silently omitted. The
lower-level steady pathways retain general-head boundary support.

Transient solver selection
~~~~~~~~~~~~~~~~~~~~~~~~~~

Transient storage is implemented for the K-cycle pathway. The PCG interface is
steady confined only.

Drying and rewetting
~~~~~~~~~~~~~~~~~~~~

The implementation uses a minimum saturated-thickness regularisation and does
not deactivate dry cells. It should not be described as a complete drying and
rewetting package.

Coarse-grid representation
~~~~~~~~~~~~~~~~~~~~~~~~~~

Coarse operators are dynamically rediscretised approximations used for
acceleration. Scientific acceptance is based on the fine-grid equation, not on
coarse-grid equivalence.

Practical tolerance portability
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The practical storage-diagonal change tolerance has dimensions and is tied to
grid area, timestep, storage magnitude, and unit system. New applications
should calibrate this control or replace it with a relative measure.

Validation scope
~~~~~~~~~~~~~~~~

The reported validation covers the target multi-period replay. Storage-disabled
regression testing, additional parameter variants, and a reset-controlled real
profiler workload remain useful follow-up checks for broader release
qualification.

Migration guidance
------------------

Existing steady-state users do not need to change their calls. New transient
users should:

#. select ``solver="kcycle"``;
#. supply a previous head and positive ``dt``;
#. use the multi-period unconfined interface when recharge changes by period;
#. explicitly enable the device transient fast path for production replay;
#. inspect ``production_acceptance_passed`` for every period;
#. retain full storage diagnostics during validation runs; and
#. verify transient water-budget closure for each new model configuration.

For scientific reporting, distinguish strict Picard convergence from practical
production acceptance and report the final fine-grid head-equivalent residual,
MODFLOW 6 comparison, and transient budget discrepancy where available.
