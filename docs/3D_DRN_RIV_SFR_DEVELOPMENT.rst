3D DRN/RIV and SFR development
================================

Status
------

This branch starts with the structured-grid DRN/RIV reference physics in
``DARCY_WARP_PACKAGE.physics.drn_riv_3d`` and its isolated analytical tests.
The public 3D solve API does not yet accept DRN/RIV. Do not advertise these
packages as supported until the residual, linearisation, convergence
criteria, and accepted budgets are all integrated and tested.

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
* The physical discharge must be evaluated from head differences directly,
  rather than recovering it from diagonal times head minus RHS.
* DRN, RIV, GHB, recharge, prescribed-head exchange and physical storage
  retain distinct budget terms. Opposite signs may cancel in a net total,
  but not in the gross flow ledger.
* The first representation allows one drain and one river entry per
  structured cell, with independently validated masks. Multiple reaches
  sharing a groundwater cell at different stages/bottoms will require
  an entry-based exchange representation; do not average away thresholds.

Integration order
-----------------

1. Add owned DRN/RIV fields to ``ThreeDOperatorSession`` and the public
   build/update API. Validate before mutation; unsupported formulations
   must raise rather than silently ignoring boundaries. Reuse the existing
   3D input helpers. A boundary update may change both forcing and
   operator/topological restraint; invalidate or refresh affected caches.
2. Extend the pinned NPF case and independent steady/transient reference
   evaluation. Use the physical flux laws in cell residuals, local and
   domain budgets, and analytical reference acceptance. Do not introduce
   a boundary term into physical storage. Include the package state in
   the appropriate cache compatibility/revision contracts.
3. Extend ``npf662_kernels.py``, ``npf662_newton.py``,
   ``npf662_state.py`` and their preconditioner assembly as required.
   Evaluate the same branches for accepted/trial heads, residual,
   Jacobian action, diagonal and line-search acceptance. Support both
   host-audited and device-native NPF paths. Ensure graph replay uses live
   boundary values and does not retain stale active branches.
4. Treat grounding separately from the existence of a boundary entry.
   An inactive drain or disconnected RIV is not a linear head anchor.
   Check compatible source balances and non-unique components; do not
   pin a cell or add an artificial conductance to disguise a singular
   system. Allow legitimate threshold crossings during nonlinear solves.
5. Add accepted per-cell DRN/RIV rates and gross inflow/outflow budgets.
   Match the accepted head, fields and timestep; reject stale or failed
   nonlinear states. Preserve previous-time heads across trial solves.
   Extend the Cartesian classic/fast backends only when the same
   convergence and conservation requirements are satisfied.
6. Run independent analytical and pinned MF6 6.6.2 checks before
   promoting the capability. Analytical correctness is the primary
   acceptance target; MF6 comparisons are the production interoperability
   target. Run actual CUDA tests rather than extrapolating CPU results.

Minimum validation matrix
-------------------------

* Single-cell steady and backward-Euler roots, with and without GHB.
* Connected multilayer flow, convertible cells, drying/rewetting and
  thresholds crossed in both directions; no-source closed components,
  inactive drains and disconnected rivers with/without an independent anchor.
* Simultaneous gains and losses (including different layers and co-located
  DRN/RIV/GHB), local residuals and separate gross budgets.
* Changing conductance, stage, bed bottom and masks between timesteps;
  no-boundary to boundary and boundary to no-boundary changes after a
  cached solve; invalid update atomicity and input ownership.
* CPU/CUDA, eager/device-native and graph replay where supported,
  independently checked heads and exchange/physical storage volumes.
* Existing no-package GHB, NPF/STO, Cartesian and 2D regressions.

SFR follow-on
-------------

Extract only the needed reach-network, Manning rating and conservative
routing logic from Overlander. Keep its hillslope and 2D hydraulic code
outside DarcyWarp. Standardise on seconds and integrated SI conductance.
Start with equilibrium streamflow routing for MF6 comparison, then add
implicit channel storage and explicit kinematic routing as justified.
For coupled solves, hold the accepted previous-time states fixed during
iterations; require identical accepted exchange volumes in both budgets.
Losing-stream water availability is a constraint on SFR exchange, not a
change to the ordinary prescribed-stage RIV law.
