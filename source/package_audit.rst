WEL, DRN and RIV release
========================

This page summarises the production package release audited at
``72cb1ee535c3db5f5b8e21121cf70169cdb08cc8``. It supersedes the interim
package-status pages as current user documentation. Those dated reports remain
in ``docs/`` as development provenance; they describe intermediate revisions,
open work and test counts and must not be read as the final release contract.

New capabilities
----------------

The fast ``mf6_6_6_2_npf`` formulation now supports native 3D DRN and RIV on
uniform and layered DIS grids for confined or convertible, steady or transient
NPF/STO simulations. Inputs may be dense cell fields or sparse entries, and
multiple drains or river reaches may occupy one groundwater cell without
averaging their conductance, elevation, stage or bed-bottom properties.
Conductance is an integrated coefficient [L2/T]. With exchange positive out of
the aquifer,

.. math::

   Q_{DRN}=C_{DRN}\max(h-e,0),
   \qquad
   Q_{RIV}=C_{RIV}[\max(h,b)-s].

``update_drn()``, ``update_riv()``, ``update_drn_entries()`` and
``update_riv_entries()`` replace or remove package definitions between solves.
``accepted_drn_discharge()``, ``accepted_riv_discharge()`` and
``accepted_boundary_entry_rates()`` return rates bound to an accepted state.
The packages are accepted by ``build_from_K_fields()`` and
``build_from_layers()``.

Explicit sparse WEL entries are supported in 2D and 3D. Signed rates are
volumetric [L3/T]: negative is abstraction and positive is injection. The 2D
equation divides each rate by cell area; the 3D equation receives it directly.
Multiple wells may occupy one cell. Public interfaces include ``wel_entries``,
``update_wel_in_place()`` in 2D, ``update_wel()`` in 3D, ``wel_budget()``,
``accepted_wel_rates()`` and 3D ``physical_source_budgets()``.
``Model3DInputs`` accepts explicit wells, and supported transient histories
accept ``wel_rates`` schedules.

Recharge remains a signed depth rate [L/T], with cell area applied once. In 3D
it enters the uppermost active cell unless that cell has prescribed head, in
which case the column receives no recharge. The raw 3D ``rhs`` remains a signed
volumetric source field. Recharge, WEL and raw volumetric sources are owned and
updated independently; an explicit transient schedule replaces only its
corresponding built component.

Accepted source and boundary states bind physical inputs, numerical heads,
package rates and physical timestep duration. They retain per-entry and
per-cell rates, gross inflow, gross outflow, net exchange and integrated
transient volumes. Rejected numerical trials do not contribute. A configured
WEL rate is the requested physical source. Prescribed-head compensation is the
balancing exchange required to hold that head, while realised MF6 cell-budget
exchange is MF6's reported exchange for the matched cell and formulation; these
quantities are related but are not interchangeable.

Bug fixes inherited from the previous main release
--------------------------------------------------

* Weakly restrained confined components now receive a component-wide physical
  balance correction for the common-head mode. This improves absolute heads
  when large opposing sources coexist with weak GHB conductance without adding
  artificial conductance or relaxing acceptance tolerances.
* Component restraint and acceptance now follow actual conducting groundwater
  connections rather than geometric adjacency. Dry faces may split components;
  static confined connectivity is cached and convertible changes are detected
  on the device. Production and independent reference calculations use the
  same hydraulic-component rule.
* Canterbury prescribed-head overlaps now apply explicit sea, south and north
  precedence instead of dictionary insertion order. A valid zero-metre sea
  head remains distinct from a missing stage.

Solver default
--------------

K-cycle-preconditioned FGMRES is the default for nonlinear 3D NPF solves.
Explicit diagonal preconditioning remains available as a reference option.
Eligible uniform confined models retain the direct fast Cartesian path. The
qualified mixed-precision path keeps FP64 heads, residuals and outer FGMRES
with an FP32 multigrid preconditioner. Saturated- and current-coefficient
K-cycle paths both include the package restraint terms; the existing qualified
control files and CUDA graph execution are retained.

Numerical qualification
-----------------------

The consolidated maintained suite contains 100 cases:

.. list-table::
   :header-rows: 1

   * - Group
     - Cases
   * - 2D groundwater equations and solver pathways
     - 23
   * - 3D groundwater equations and solver pathways
     - 24
   * - Model inputs and spatial preparation
     - 8
   * - Live MF6 parity
     - 14
   * - Independent reference equations
     - 10
   * - Qualification evidence
     - 6
   * - Original comparisons and retained regional references
     - 15
   * - Total
     - 100

It replaces the generated regression matrix while preserving the original
parity tests and their reference data. Coverage emphasises analytical
solutions, conservation, matched MF6 heads and package rates, physical
timesteps, integrated volumes, mathematical invariance and execution of the
production solver. Independent NPF, transient and matrix-free references, the
generic MF6 writer and regional comparison tools include WEL/DRN/RIV; canonical
archives retain individual sparse entries and distinguish MF6 standard and
Newton formulations.

The live-tree million-cell package evidence records approximately 1.36 s for
steady NPF with current-coefficient FP32 K-cycle preconditioning and 2.35 s for
a 20-step resident NPF/STO history. Eight archived strict regional 3D
comparisons passed their recorded head and individual DRN/RIV exchange gates;
the largest recorded head difference was about ``9.55e-8 m`` and the largest
individual boundary-rate difference about ``6.99e-7 m3/day``. These are
historical measurements of their recorded source revisions, not fresh results
for this documentation revision. The earlier 100--250x MF6 transient speedups
remain evidence only for their original matched no-package configurations.

Remaining limitations
---------------------

* SFR, UZF and transport are not included.
* Legacy Cartesian and experimental terrain DRN/RIV are unsupported.
* Automatic dry-cell well-rate reduction is not implemented.
* Comprehensive drying/rewetting parity is outside the established
  qualification.
* Historical 2D regional comparisons did not all meet the preferred ``1e-6 m``
  maximum head-error target.
* Particular Chebyshev configurations have known convergence limitations on
  heterogeneous regional models.

The historical
`WEL/DRN/RIV audit-repair report <https://github.com/Lincoln-Environmental/DarcyWarp/blob/main/docs/WEL_DRN_RIV_AUDIT_REPAIR_2026-11-26.rst>`_
and related dated reports are retained for traceability. They are not current
release notes and their internal development defect inventory is not reproduced
here.
