# 3D capability guide — integration candidate

This guide describes the feature branch, not a public release. DarcyWarp is
developed by **Patrick Durney and Jaco Fourie** and owned by **Lincoln Agritech
Ltd**. Project-owned code remains AGPL-3.0-or-later. The adapted NVIDIA Warp
GMRES implementation retains its Apache-2.0 attribution.

## Equations, units and inputs

Darcy flux is **q = −K ∇h**, with total hydraulic head h. Each free cell solves
`sum(outward face flows) + stored-water change / dt = cell source`.
Steady flow omits storage. Arrays use `(nz, ny, nx)`, with layer index increasing
downwards. Use a consistent unit system; examples and pinned NPF validation use
metres and days: K [m/day], h/elevations [m], face conductance [m²/day], and
**rhs [m³/day per cell]**. Multiply recharge [m/day] by `dx*dy` once when adding
it to the intended receiving layer. Do not apply recharge to every layer.
Specific yield Sy is dimensionless; specific storage Ss and confined
`storage_coeff` have units [1/m]. `dt` is in days in this unit system.

`active`, `bc_mask` and NPF cell types accept Boolean or exact numeric 0/1
values. Fractional, nonfinite and other mask values raise before integer
conversion. Fixed-head cells must be active. K-built models own their physical
conductivity: use `update_K_fields()` or the appropriate builder to change it.
Independent face models use `update_face_conductance()` for confined flow.
Unconfined legacy low-level solvers additionally require physical K and layer
bottoms; supplying different K through a solve call on a K-owned public model
is rejected. Terrain has `update_terrain_K_fields()` and explicit geometry and
contact updates. Failed builders retain the previous usable model and session.

## Formulations remain separate

| Formulation / selection | Geometry and operator | Steady / transient | Boundaries and status |
| --- | --- | --- | --- |
| `legacy` + `build_from_K_fields()` or reciprocal face builder | Flat uniform-dz Cartesian cells; conservative seven-point harmonic conductances | Confined; quasi-3D unconfined Picard; backward-Euler confined and secant Sy/Ss transient modes | Prescribed cell heads, volumetric sources, impermeable exterior/mask faces. Analytical and MF6 regressions; drying is approximate. |
| `mf6_6_6_2_npf` + K or layered builder | Conventional Cartesian/DIS layers; half-cell resistance connections, no transformed metric cross terms; ICELLTYPE 0/1 and MF6 6.6.2 upstream horizontal saturation law | **Steady only**, confined or convertible/unconfined | CHD cell heads and volumetric sources. Pinned-law independent NumPy/SciPy and native MODFLOW tests. No XT3D, GHB, DRN/RIV, rewet, perched, variablecv or NPF transient storage. |
| `experimental_transformed_terrain` + `build_from_terrain()` | Independent ordered elevation interfaces or DEM plus parallel depths; conservative transformed tensor `A = abs(det J) J^-1 K J^-T`, geometric cross terms and physical subfaces | Confined and approximate unconfined; confined-volume and supported secant Sy/Ss transient modes | Prescribed cell heads, physical head/exchange contacts and discharge-only subfaces. **Experimental**; multiple independently accepted discrete roots are retained. |

Layered NPF permits independent layer elevations/thickness and inactive cells
without vertical pass-through. Terrain requires complete active columns and
strictly ordered positive layer thickness. Pinch-outs, faults, crossing layers
and arbitrary internal terrain holes are unsupported. Nonflat flowing terrain
requires positive K in all directions; unsupported zero-K closures raise.
Anisotropic diagonal K is supported; rotated full physical conductivity tensors
are outside these public APIs.

Legacy unconfined flow floors saturated thickness (default 0.01 m, capped by
cell thickness), scales horizontal transmission, and retains saturated vertical
K. Terrain has its own 0.01 m floor, metric couplings and vertical transmission
approximation. Storage remains physical and unfloored. Neither is a Richards
solver or a complete dry-cell deactivation/rewetting package. Matched NPF has
its own quadratic saturation smoothing (`epsilon=1e-6`) and no terrain floor.
These laws and boundary representations cannot be exchanged to force agreement.

## Runtime, solvers and acceptance

The tested runtime is **Python 3.12 and warp-lang 1.17.0**. The adapted GMRES
uses private Warp interfaces and raises a version-specific ImportError under
unsupported versions. Set `DARCY_FLOAT=float64` **before importing** solver
modules for fast, NPF and terrain paths. Warp CPU is supported. CUDA paths
exist, but the final CUDA release campaign is deferred because the user
requested CPU-only work during device contention. Classic legacy precision
follows `DARCY_FLOAT`; the fast and matched paths require FP64.

Legacy defaults retain classic K-cycle (`solver='kcycle'`); Chebyshev and
`implementation='fast'` are explicit alternatives. Uniform confined NPF retains
its historical direct session path when no Krylov method is requested. Its
`rel_tol`, `max_levels` and `min_coarse_n` affect that backend. Explicit
`npf_krylov_method='gmres'` or `npf_preconditioner='diagonal'` selects diagonal GMRES. Layered/unconfined NPF uses
diagonal GMRES by default. **K-cycle FGMRES is opt-in** through
`npf_preconditioner='kcycle'`; it is not promoted to the default.
Use `NPF662NewtonControls` for forcing, line search and nonlinear iteration
controls, and `NPF662KCycleControls` for hierarchy/smoother/cycle settings.
Legacy tolerance/hierarchy overrides on the NPF Newton dispatch raise with the
supported control name, including explicitly supplied former default values.
Omitted controls retain their historical effective defaults. Returned `effective_solver_settings`, `multigrid` and
final-threshold diagnostics identify the settings actually used.

NPF default final acceptance requires finite fields/aggregates, grounded active
components, prescribed-head error ≤1e-10 m, head-equivalent RMS ≤1e-6 m,
and **either** cellwise physical RMS ≤1e-12 m³/day **or** cellwise L1 imbalance
normalised by gross throughput ≤1e-8. Confined tolerance overrides retain their
existing contract; convertible final thresholds remain fixed. Terrain and
legacy retain their separate existing thresholds. A signed whole-domain sum,
a finite head field or a small update alone cannot establish acceptance.
Overflowed throughput, residuals, diagonals or fluxes cannot pass the common
physical budget gate. Opposing boundary flows enter the normalisation before
cancellation. Very small/large residual RMS uses scaled reductions.

Device work counters execute inside captured graphs. Terrain reports actual
preconditioner/V-cycle applications; restart blocks are separate. NPF reports
Krylov work, Jv products, accepted/rejected updates and preconditioner work.
`seconds` is the synchronized device iteration interval for NPF Newton;
`setup_seconds` is workspace/Krylov/hierarchy setup. Public total time includes
independent reference setup and final evaluation. CPU/GPU event, enqueue and
synchronisation subintervals can overlap and must not be added as independent
costs. Older archived cycle labels must be interpreted using their source
version. Legacy cycle counts describe top-level solver cycles, not every
recursive/coarsest operation.

## Runnable verification and evidence

From the repository root in the declared environment:

```bash
CUDA_VISIBLE_DEVICES=-1 DARCY_FLOAT=float64 PYTHONPATH=. python examples/verify_3d_series_flow.py
```

This runs an actual 5 × 3 × 4 layered public NPF model. A separate series
resistance calculation supplies exact heads and vertical flux. It asserts
physical convergence and saves inputs, numerical heads/fluxes and diagnostics
under `example_results/3d_series_flow/`. The completed audit run is retained in
`release_evidence/example/`. No site data, GPU or MODFLOW is required.

The [readiness assessment](3D_RELEASE_READINESS.md) distinguishes fresh execution
from archived evidence. Detailed formulation contracts remain in the
[Cartesian NPF contract](NPF662_CARTESIAN_REFERENCE_CONTRACT_V1.md),
[layered NPF contract](NPF662_LAYERED_DIS_REFERENCE_CONTRACT_V2.md),
[terrain independent validation](TERRAIN_CPU_MF6_VALIDATION_2026-10-16.md) and
[variable-interface report](3D_VARIABLE_GEOMETRY_SCIENTIFIC_REPORT_2026-10-13.md).
The [multiple-root fixture](../tests/fixtures/terrain_multiple_roots/README.md)
retains distinct accepted terrain branches. Agreement with one MF6 root does
not establish uniqueness or equivalence of terrain and NPF equations.

This guide makes no general field-scale accuracy or performance guarantee.
The strict historical `4×6×8` NPF CUDA/MF6 head comparison miss remains in the
record even though independent polishing reduces it to about 8.7e-12 m.
