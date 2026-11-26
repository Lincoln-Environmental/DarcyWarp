# DarcyWarp changelog

Version numbers identify immutable source milestones. The original paper
describes the steady, fixed-transmissivity two-dimensional solver; subsequent
versions extend that implementation. Versions 0.1.0–0.3.0 are retrospective
release milestones selected from the recorded main history, not releases
published on those original dates. Historical snapshots retain their original
source, documentation, licences and validation evidence.

## v0.4.0

### Added and improved

- Explicit volumetric WEL in 2D and 3D, independent recharge and well updates,
  and supported transient source schedules.
- Native DRN and RIV for the fast 3D MF6 6.6.2 NPF/STO formulation, including
  confined and convertible cells, uniform and layered DIS grids, and steady
  and transient solves.
- Dense or sparse DRN/RIV inputs; multiple independent boundary entries per
  cell; accepted per-entry and per-cell rates, gross/net budgets and integrated
  exchange volumes.
- K-cycle-preconditioned FGMRES is the nonlinear NPF default; the direct fast
  Cartesian pathway remains available for eligible confined models.
- Consolidated 100-case numerical suite, retaining original reference data and
  emphasising analytical accuracy, conservation, matched MF6 results and actual
  production solver execution.

### Fixed (relative to v0.3.0)

- Improved the weakly restrained confined groundwater head mode through a
  component-wide physical balance correction.
- Corrected component restraint and acceptance to use conducting faces rather
  than geometric adjacency, with static connectivity caching and device-side
  change detection.
- Made Canterbury overlapping prescribed-head boundary precedence explicit
  (sea, south, north), independent of dictionary insertion order.

### Scope and evidence

The production implementation was audited at
[72cb1ee](https://github.com/Lincoln-Environmental/DarcyWarp/commit/72cb1ee535c3db5f5b8e21121cf70169cdb08cc8).
See the [package release guide](https://github.com/Lincoln-Environmental/DarcyWarp/blob/v0.4.0/source/package_audit.rst)
for the public API, documented numerical qualification and limitations.
Archived benchmark results apply to their recorded revisions and configurations;
they are not new measurements for this tag. Native 3D DRN/RIV are restricted to
the supported NPF formulation. SFR, UZF and transport are not included;
automatic dry-cell well-rate reduction and comprehensive drying/rewetting
parity remain outside the established qualification.

## v0.3.0

Retrospective source milestone:
[65cba8d](https://github.com/Lincoln-Environmental/DarcyWarp/commit/65cba8df6154bcdfd8eb73c1b20a2623a744d73c),
the last main revision before the explicit WEL/DRN/RIV increment.

- Introduced structured 3D Cartesian and layered DIS groundwater modelling.
- Added the matched MF6 6.6.2 NPF/STO formulation and native 3D GHB.
- Added GPU-accelerated steady and transient histories, K-cycle/FGMRES,
  qualified mixed-precision preconditioning and CUDA graph pathways.
- Retained and improved the existing 2D solvers and their numerical validation.

See the [historical 3D guide](https://github.com/Lincoln-Environmental/DarcyWarp/blob/v0.3.0/docs/3D_CARTESIAN_NPF.md)
and [native GHB guide](https://github.com/Lincoln-Environmental/DarcyWarp/blob/v0.3.0/docs/3D_NATIVE_GHB.rst)
for the supported options and qualification scope at that revision.

## v0.2.0

Retrospective source milestone:
[ca67fdc](https://github.com/Lincoln-Environmental/DarcyWarp/commit/ca67fdcf5302fb3cdd5ae00383f87b77db34a8dc),
the last 2D-only main revision before the 3D promotion.

- Extended 2D horizontal flow to unconfined, saturation-dependent
  transmissivity and transient groundwater flow with storage.
- Supported confined and unconfined steady/transient solver pathways,
  changing forcing and general-head boundaries.
- Included reproducible examples, MODFLOW comparisons and ensemble benchmarks.

## v0.1.0

Retrospective source milestone:
[ba94383](https://github.com/Lincoln-Environmental/DarcyWarp/commit/ba94383f7b95932e397454ae5960c66132a8a507),
the original tracked fixed-transmissivity implementation.

- Original 2D steady-state groundwater GPU multigrid framework.
- Repeated solves with changing recharge and transmissivity.
- Reference comparisons and ensemble benchmarks forming the basis of the
  published methods work.

Paper: Durney and Fourie (2026), *Environmental Modelling & Software*,
204, 107081. [doi:10.1016/j.envsoft.2026.107081](https://doi.org/10.1016/j.envsoft.2026.107081).
