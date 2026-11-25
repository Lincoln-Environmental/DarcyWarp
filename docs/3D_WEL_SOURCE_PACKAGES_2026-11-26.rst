WEL and source packages, boundary entries and audit fixes (2026-11-26)
=======================================================================

Scope
-----

This increment integrates latest main into ``feature/sfr-3d-boundaries``,
adds an explicit volumetric WEL package to the maintained 2D and 3D
pathways, supports multiple DRN/RIV entries per cell through one CSR
representation, resolves the DRN/RIV audit findings (activation seeding,
K-cycle restraint, gross exchange metrics, accepted-state ownership,
cached reference refresh) and reports separate gross physical source
budgets with accepted-duration volume integration. SFR routing is not
implemented.

Source ownership conventions
----------------------------

All user inputs stay signed and independently owned; the assembled
numerical source is their sum.

* 2D ``R_field``: signed depth rate [L/T], scalar or ``(ny, nx)``. The
  operator applies the square cell area once.
* 3D ``R_field``: identical dimensional meaning; ``combine_cell_sources``
  converts once with ``dx*dy`` and delivers water to the uppermost active
  non-prescribed cell of each column. Prescribed-head columns receive
  none, and the recharge physical budget counts only receiving cells.
* 3D ``rhs``: signed ``(nz, ny, nx)`` volumetric rates [L3/T], unchanged.
* ``wel_entries``: explicit sparse wells with MF6 sign convention
  (negative abstraction, positive injection, zero inactive). Rates are
  volumetric [L3/T] in both dimensionalities; the 2D package divides by
  the square cell area exactly once so the operator's area application
  is not doubled. Wells on inactive cells are rejected instead of being
  silently discarded; wells on prescribed-head cells are retained in the
  budget while the prescribed head absorbs them.
* ``update_R_in_place``, ``update_rhs``/``update_wel`` (3D) and
  ``update_R_in_place``, ``update_uniform_recharge_in_place``,
  ``update_wel_in_place`` (2D) replace exactly one component, validate
  before mutating and leave the previous state unchanged on failure.
* 2D keeps one device source buffer: ``R_field_host`` carries the
  combined recharge-plus-well depth field, so every operator, host
  backend and CUDA graph path reads a single field without kernel or
  graph changes. Models without wells keep byte-identical behaviour.
* 3D ``solve_transient_npf`` accepts ``wel_rates``: one well-entry value
  per step replacing the built entries. ``source_factors`` scale every
  built component that the call does not explicitly replace (a recharge
  schedule excludes the built recharge; a well schedule excludes the
  built wells). The interaction is tested; scaling never double counts.

Budgets
-------

``WarpDarcySolver3D.physical_source_budgets`` and the 2D
``wel_budget`` report separate gross ``inflow``, ``outflow`` and ``net``
for wells, recharge and raw volumetric sources; opposing entries in one
cell never cancel in the gross ledger. Device physical-row metrics keep
per-package gross inflow and outflow instead of ``abs`` of the combined
exchange, so large opposing exchanges cannot mask local equation errors.
The residual itself keeps the signed sum. Transient histories attach
per-step component budgets integrated over accepted durations, with
totals summed via :py:func:`math.fsum`; rejected steps contribute
nothing.

Boundary entries per cell
-------------------------

Dense single-entry DRN/RIV fields and sparse entry lists reduce to one
owned CSR entry list per package
(``physics/drn_riv_3d.BoundaryEntryList``). One kernel family evaluates
every entry's flux independently before cell contributions sum, so
dense models keep bit-identical physics while multiple drains or river
reaches in one cell keep their own thresholds, stages and accepted
rates. Stages and bed bottoms are never averaged. Per-entry discharge
rates, entry-level budgets, roundoff bounds, grounding checks and
unanchored-component seeding all operate on entries.
``update_drn_entries``/``update_riv_entries`` replace either
representation atomically; supplying one form clears the other. Device
writes use ``assign`` paths (a ``numpy()[:]`` assignment mutates a host
copy on CUDA and silently no-ops), and per-entry output buffers resize
with the entry list.

Audit fixes
-----------

* Activation seeding exists on the device-native transient state with
  explicit active-branch verification (a uniform shift can round back
  onto a threshold; the seed enforces strict crossing and reports the
  verified branch). Resident histories run a static activation watch
  and re-seed and refresh the preconditioner diagonal from the seeded
  guess only for components that can actually lose every restraint.
* The previous-state K-cycle diagonal combines storage, GHB and active
  DRN/RIV restraint evaluated at the previous heads, so systems
  constrained only by a drain or a river keep a nonsingular
  preconditioner. The physical residual and Newton Jacobian are
  unchanged, and genuinely singular components still fail loudly.
* Accepted DRN/RIV discharge is bound to an immutable accepted-state
  token (session revisions plus SHA-256 digests of the accepted head and
  boundary fields), replacing re-derived flux comparisons that could
  not establish ownership for inactive drains or disconnected rivers.
  Returned arrays are owned copies; unrelated or mutated states are
  rejected, and historical states stay valid after model updates because
  the token carries its own digests.
* ``NPF662TransientState.independent`` builds its audit from owned
  snapshots of every solver-aliased input, keyed on session revisions,
  so later recharge, well, boundary or prescribed-value updates cannot
  silently re-anchor an accepted historical state.
* The host reference seed enforces threshold crossings and can re-seed
  up to twice after mid-solve restraint loss.

Known precision envelope
------------------------

``test_common_head_shift_with_opposing_large_wells`` now asserts the
head envelope implied by the independently evaluated accepted residual:
with an 8e-7 GHB conductance, a solution accepted at the documented
1e-12 m3/day absolute gate carries a head uncertainty of order residual
L1 divided by that conductance (~1.25e-6 m). Which device lands inside
or outside a fixed 1e-6 m figure is arithmetic luck. No solver
acceptance threshold changed.

Validation
----------

* Analytical: WEL scipy cell balance, gross budgets for opposing
  entries, independent/atomic updates, schedule volumes, device parity,
  Model3DInputs integration, 2D/3D thickness-integrated equivalence;
  closed-form multi-drain and multi-river roots, dense-CSR equivalence,
  digest sensitivity and replacement updates (``tests/test_3d_wel.py``,
  ``tests/test_2d_wel.py``, ``tests/test_3d_drn_riv_entries.py``).
* MF6 6.6.2: multilayer confined pumping, two-period pumping schedules
  and simultaneous recharge with pumping reproduce matched heads,
  package rates and integrated budgets
  (``tests/test_3d_wel_mf6.py``), alongside the existing seven DRN/RIV
  MF6 comparisons.
* Regression: full suite on CPU and CUDA; the merged baseline reported
  1,475 passed, 21 expected FP32/graph skips and the one pre-existing
  CUDA precision case documented above, since resolved.
* Benchmark: ``working_tests/benchmark_3d_wel_boundaries.py`` measures
  heterogeneous multilayer workloads with substantial abstraction,
  nonzero recharge, gaining/losing rivers, extensive drainage and
  multi-entry cells at ~100k and 1M+ cells, separating construction
  from warm solves and recording update, stage-update and 20-step
  history timings (``benchmark_results/wel_boundaries_3d/``).
* Regional MF6 parity: ``working_tests/mf6_parity_3d_wel_boundaries.py``
  solves the matched MF6 twin of the small benchmark grid (100,800
  heterogeneous cells, 84 well entries including shared cells, 792
  active drains, 44 gaining/losing river entries, receiving-cell
  recharge). With anisotropic horizontal K matched via ``k22``, heads
  agree to RMSE 1.66e-8 m (max 1.17e-6 m), gross WEL rates agree
  exactly (231000 L3/T), and DRN and RIV net rates agree to relative
  4e-10 and 2.3e-9; MF6 reports a 0.0 percent budget discrepancy
  (``benchmark_results/wel_boundaries_3d/mf6_parity_small.json``).

Runnable examples
-----------------

.. code-block:: console

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/example_3d_wel.py

Remaining limitations
---------------------

* MF6 comparisons for WEL cover confined cases and schedules; matched
  convertible/unconfined WEL comparisons with drying and rewetting are
  future work, as are storage-volume budget comparisons over longer
  histories.
* Automatic well index reduction at dry cells is not implemented;
  prescribed-rate wells keep pumping and the budget reports it, as
  specified. An explicit optional flow-reduction mode would need its own
  MF6 validation before shipping.
* The activation watch reads device heads on host once per history step
  only for models whose components could lose all restraint; the common
  regional case (every component grounded by CHD/GHB/storage) pays
  nothing.
* 2D transient histories do not yet expose per-step WEL schedules; wells
  enter 2D transient periods through the combined source field updates.
