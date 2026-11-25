3D DRN/RIV and SFR development
================================

Status
------

Native DRN/RIV is implemented for the fast ``mf6_6_6_2_npf`` 3D solver path.
The public builders accept scalar or ``(nz, ny, nx)`` fields and masks;
``update_drn`` and ``update_riv`` validate before changing owned state. The
NPF Newton residual, trial evaluation, Jacobian action, diagonal/K-cycle
preconditioner, roundoff audit, accepted per-cell rates, separate gross
budgets and transient accepted exchange volumes include both packages.
CPU and CUDA tests cover steady, STO, convertible and layered NPF cases.

The legacy Cartesian classic/fast session explicitly rejects solves with DRN
or RIV. Terrain sessions also reject cell DRN/RIV. Do not describe either
legacy path as package-capable. The current cell-based representation allows
one DRN and one RIV per cell; it is not yet a multi-reach exchange map for SFR.

Verified evidence on this branch:

* ``DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python -m pytest -q``:
  1,442 passed, 21 skipped, 1 failed in 1,278 s. The failure is the existing
  CUDA ``test_common_head_shift_with_opposing_large_wells`` case, exceeding
  its 1e-6 m bound by 1.57e-6 m. The exact same result reproduces at the
  committed pre-integration feature baseline ``ca09497``.
* The broad post-optimization NPF and package regressions report 272 passed,
  4 skipped and the same baseline CUDA failure. A separate layered mixed
  DRN/RIV actual-MF6 case passes. MF6 6.6.2 checks cover confined steady
  DRN/RIV, transient single-cell activation/leakage, and a two-layer steady
  case with both packages. Head error is below 1e-9 m and package-rate error
  below 1e-11 L3/T in these cases.
* The CUDA benchmark ``working_tests/benchmark_3d_drn_riv.py`` covers
  120,000 and 1,012,500 cells. Paired package cases use zero exchange and
  retain the no-package solution (max head difference 7.11e-15 m). Warm
  Newton execution medians are unchanged by the zero-exchange package fields:
  0.112 s at 120k cells and 0.284-0.285 s at 1.0125M cells. Warm public-call
  medians are 0.173 s without packages and 0.214-0.216 s with packages at
  120k cells; at 1.0125M cells they are 0.832 s and 1.235-1.500 s. Reference
  cache refresh takes 3-66 ms and parameter updates take 1-15 ms. Remaining
  public-call overhead includes input validation and package field refresh;
  GPU Newton time itself is unchanged in these equivalent-solution cases.

These tests and measurements do not establish matched MF6 accuracy for
layered or convertible DRN/RIV, nor performance parity on active regional
boundary distributions. This branch is a verified NPF implementation
increment, not completion of the wider qualification matrix.

Boundary contract
-----------------

Rates are positive out of the aquifer. Conductance is integrated [L2/T];
it must not be multiplied by cell area, saturated thickness or conductivity.
For head h, drain elevation e, stage s, riverbed bottom b, and conductance C:

* DRN: q = C max(h - e, 0).
* RIV: q = C (max(h, b) - s).
* The diagonal is C on the active head-dependent branch and zero otherwise.
  At an activation threshold choose the zero derivative, consistently with 2D.
* For RIV below the bed bottom, the RHS is C(s - b), not C s. That branch
  has no head restraint.
* Physical discharge is evaluated from head differences directly, rather
  than recovered from diagonal times head minus RHS.
* DRN, RIV, GHB, recharge, prescribed-head exchange and physical storage
  retain distinct budget terms. Opposite signs may cancel in a net total,
  but not in the gross flow ledger.
* The first representation allows one drain and one river entry per
  structured cell. Multiple reaches sharing a groundwater cell at different
  stages/bottoms require an entry-based exchange representation; do not
  average away thresholds.

Public API and runnable examples
--------------------------------

The public entry points are ``WarpDarcySolver3D.build_from_K_fields``,
``build_from_layers``, ``update_drn``, ``update_riv``,
``accepted_drn_discharge`` and ``accepted_riv_discharge``. Conductance is
integrated [L2/T]; accepted rates are [L3/T], positive out of the aquifer.
Budget ``net`` is inflow minus outflow. Storage remains a separate term.

Run manual CUDA examples from the repository root with:

.. code-block:: console

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/example_3d_drn_riv.py

The script solves isolated DRN and RIV cells, then a two-layer transient case
where accepted RIV exchange reverses after pumping in the stream-connected
cell. Reproduce the GPU timing artifact with:

.. code-block:: console

   DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python working_tests/benchmark_3d_drn_riv.py

Remaining implementation and validation
----------------------------------------

1. Expand actual MF6 comparison to multilayer and convertible DRN/RIV,
   gross package and storage budgets, drying/rewetting and flow reversal.
   Current MF6 regressions are confined steady and single-cell transient.
2. Add independent multicell and multilayer analytical roots and refinement
   studies for vertical series flow, heterogeneous leakage, simultaneous
   gains/losses, and transient threshold crossings.
3. Profile package kernels and reduce measured 1M-cell overhead without
   relaxing acceptance tests. Compare active sparse/distributed cases against
   equally accurate MF6 results before making regional performance claims.
4. Extend native DRN/RIV to legacy Cartesian confined/unconfined solvers only
   with matching residual, Jacobian, budgets and acceptance. Their explicit
   rejection remains intentional until those pieces are integrated.
5. Preserve the accepted-state safeguards:

   * Inactive drains and disconnected rivers do not restrain groundwater head.
   * Check compatible source balances and non-unique components; do not pin
     cells or add artificial conductance to disguise singular systems.
   * Only accepted per-cell rates enter transient exchange volumes. Rejected
     trials do not update previous heads or accepted budgets.
   * Keep storage separate from boundary conductance and report gross flows.

SFR follow-on
-------------

Do not implement SFR as part of this task. Extract only the needed reach
network, Manning rating and conservative routing logic from Overlander. Keep
its hillslope and 2D hydraulic code outside DarcyWarp. Standardise on seconds
and integrated SI conductance. Start with equilibrium routing for MF6
comparison, then add implicit channel storage and explicit kinematic routing
as justified. Hold accepted previous-time states fixed during coupling
iterations and require identical accepted exchange volumes in channel and
aquifer budgets. Losing-stream water availability is an SFR constraint, not
a change to ordinary prescribed-stage RIV.
