# Structured 3D groundwater flow

Use `WarpDarcySolver3D` or `create_solver(dim=3, ...)` from
`DARCY_WARP_PACKAGE`. The factory takes `nx`, `ny`, `nz`, `dx` and `dz`.
`dy` defaults to `dx`. Cell fields have shape `(nz, ny, nx)`, ordered from the
top layer down. Use the active and fixed-head masks to set the model domain
and prescribed heads. Exterior and inactive faces are no-flow.

Supply surface recharge in the same way as in 2D. Pass `R_field` as a scalar
or a horizontal `(ny, nx)` array in length per unit time, positive for recharge
and negative for withdrawal. With metres and days, 1 mm/day is `0.001`.
DarcyWarp multiplies by `dx * dy` and applies the rate to the uppermost active
cell in each column. A completely inactive column receives zero. If the
uppermost active cell has prescribed head, the column receives zero recharge.
For a MODFLOW 6 RCHA comparison with inactive cells in the top layer, set
the RCHA receiving-layer array (`IRCH`) to the same active layers.

For explicit screened wells, supply sparse `wel_entries` with signed volumetric
rates [L³/T]: negative rates abstract and positive rates inject. Multiple wells
may occupy one cell. Use `update_wel()` to replace or remove them,
`wel_budget()` for gross/net package totals and `accepted_wel_rates()` for
rates bound to an accepted state. `Model3DInputs` also accepts wells. A raw
`(nz, ny, nx)` `rhs` remains available for other signed volumetric sources.
Recharge, WEL and raw `rhs` are independently owned, so updating one does not
overwrite the others.

| Formulation | Geometry | Supported flow |
| --- | --- | --- |
| `legacy` | Flat Cartesian, uniform layer thickness | Confined and unconfined, steady and backward Euler transient |
| `mf6_6_6_2_npf` | Uniform Cartesian or conventional layered DIS | Confined and convertible, steady and backward Euler NPF/STO |
| Transformed terrain | Experimental development on `dev` | Separate formulation outside the Cartesian/NPF release |

Build legacy and uniform NPF models with `build_from_K_fields`. Legacy
confined models can also use six reciprocal face arrays through
`build_from_face_conductance`. `Model3DInputs` is a reusable container for
uniform grids. Use `build_from_layers` for NPF with explicit elevation
interfaces of shape `(nz + 1, ny, nx)`. Each column's interfaces must
decrease strictly with depth. Layered DIS uses conventional orthogonal
connections.

Both Cartesian and NPF/STO support native cell-based GHB in steady and
transient solves. Supply `ghb_conductance` and `ghb_head` when building,
then use `update_ghb()` for boundary changes. Conductance is integrated,
independent of K and saturation, and contributes both a diagonal term and
external-head forcing. See the [native 3D GHB release](3D_NATIVE_GHB.rst)
for budgets, CPU/CUDA and MF6 checks, precision controls and runtime profiles.
Exterior face arrays remain no-flow; they cannot encode a GHB.

The fast `mf6_6_6_2_npf` formulation additionally supports native DRN and RIV
for confined or convertible, steady or transient models. Dense cell fields and
sparse entries are accepted by `build_from_K_fields()` and
`build_from_layers()`. Multiple entries in one cell retain independent
conductance and elevations. Use `update_drn()`, `update_riv()`,
`update_drn_entries()` and `update_riv_entries()` to replace or remove package
definitions. `accepted_drn_discharge()`, `accepted_riv_discharge()` and
`accepted_boundary_entry_rates()` expose accepted cell and entry rates,
positive out of the aquifer. Legacy Cartesian and experimental terrain DRN/RIV
are not supported. See the [current package release](../source/package_audit.rst).

NPF requires `DARCY_FLOAT=float64` before import. Use Python 3.12 and Warp
1.17.0 for the validated configuration. Convertible, layered and transient
NPF use `implementation="fast"` and `solver="kcycle"` on the model. The
default Krylov choice for those NPF solves is K-cycle-preconditioned FGMRES.
Explicit `npf_preconditioner="kcycle", npf_krylov_method="fgmres"` selects
the same path. Diagonal GMRES remains an explicit reference selected with
`npf_preconditioner="diagonal"`; pair it with `npf_krylov_method="fgmres"`
only when diagonal FGMRES is specifically intended.
`NPF662NewtonControls` and `NPF662KCycleControls` are package exports.
See the [GPU implementation guide](3D_IMPLEMENTATIONS.rst) for the measured
K-cycle configuration, Cartesian fast-path selection and timing conventions.
Physical acceptance thresholds are unchanged.

### NPF numerical robustness (v0.4.1)

Current-coefficient K-cycle preconditioning includes the upstream saturation
derivative on its Newton diagonal by default. The physical Jacobian is
unchanged. Set newton_diagonal=False on NPF662KCycleControls only for
diagnostic comparisons with the earlier approximation. Saturated-coefficient
preconditioning and Picard directions are unaffected.

FGMRES detects true-residual stagnation and the nonlinear solver can retry
an ineffective Newton direction with a saturated-coefficient or diagonal
preconditioner before trying its existing nonlinear safeguards. The
independent physical auditors recognise a fully dry isolated zero equation
only if its physical diagonal, residual, gross forcing and package/storage
exchange are zero. No head or conductance floor is added.

For difficult steady convertible models, an optional numerical continuation
shift can be supplied on NPF662NewtonControls. For example, use the
qualified current-coefficient mixed-preconditioner configuration, choosing
a shift in integrated conductance units from the actual model:

    from DARCY_WARP_PACKAGE import NPF662NewtonControls, NPF662KCycleControls

    controls = NPF662NewtonControls(
        steady_initial_shift=initial_conductance_shift,
        reuse_trial_state=True,
        tiled_reductions=True,
    )
    multigrid = NPF662KCycleControls(
        levels=5, coarse_sweeps=2, cached_line_factors=True,
        coefficient_state="current", preconditioner_precision="float32",
        device_hierarchy_build=True, use_cuda_graph=True,
    )
    head, info = model.solve(
        unconfined=True, npf_cell_type=cell_types,
        npf_newton_controls=controls, npf_kcycle_controls=multigrid,
        krylov_restart=16, max_outer=80, max_cycles=500,
    )

Here the initial_conductance_shift is a caller-supplied nonnegative
coefficient with units L2/T. In the regional development experiments an
initial shift of 0.08 times the median full-thickness horizontal conductance
scale was effective. This is an empirical starting value, not a universal
default. The public continuation default is zero (disabled); explicit
steady continuation is available for any number of convertible layers.
It is not a transient storage term. When enabled, the drying safeguard
does not clip a vertically connected head or a pumped cell, and an exact
dry endpoint projection must pass the independent physical audit before
being accepted. Full drying/rewetting parity remains outside the release
qualification. Existing precision selection, solver profiles and tolerances
remain unchanged. MF6 workspace import and automatic regional tuning are
not part of v0.4.1.

Uniform confined steady NPF retains the direct Cartesian solver when Krylov
controls are omitted.

Pass `unconfined=True` and a binary `npf_cell_type` array for mixed
confined and convertible cells. For a uniform convertible grid, pass a flat
`top_elevation` at solve time. Layered DIS uses the supplied interfaces.
A seven-day NPF/STO step uses the previous physical head and Ss/Sy storage.

```python
head, info = model.solve(
    unconfined=True,
    npf_cell_type=cell_types,
    transient=True,
    dt=7.0,
    head_prev=previous_head,
    ss=1e-5,
    sy=0.15,
)
```

`head_prev` is the previous physical state. Use `update_initial_head`
to choose a separate iterative starting point, `update_R_in_place(rate)`
to change the recharge depth rate, `update_wel()` to change explicit wells,
`update_rhs` to change additional volumetric sources,
`update_boundary_values` to change prescribed heads,
and `update_K_fields` to change physical conductivity and its derived
faces. Check `info["converged"]`, the residual and the water balance before
advancing the timestep. Returned heads own their storage. A later steady
solve clears transient storage terms.

The MF6 6.6.2 contract covers default NPF connections and default Ss/Sy
storage with Newton saturation smoothing. XT3D, variable CV, rewetting,
perched flow and alternative STO options are outside this contract. SFR, UZF,
transport, automatic dry-cell well-rate reduction and comprehensive
drying/rewetting parity are not included. The original 3D promotion was qualified on CPU FP64;
subsequent GPU qualification and device-native history optimisations are
now on `main`. The [current NPF GPU performance guide](../source/npf_gpu_native.rst)
summarises final timings and acceptance. The
[initial GPU qualification](3D_GPU_RELEASE_NOTES.md) remains available as
historical evidence. GPU convergence still depends on the model and its
explicit iteration budget.

Transient NPF also offers `npf_device_native=True` for device acceptance
and persistent hierarchy refresh, and `solve_transient_npf()` for
multi-step GPU-resident histories. See the
[current GPU guide](../source/npf_gpu_native.rst) for timings, validation
and the reproducible comparison command.

## Device-resident multi-step NPF/STO history

For repeated transient steps, the explicitly selected `solve_transient_npf()`
API keeps intermediate heads on the device and returns the full history
after the final step. The following self-contained example uses the same
two-layer DIS configuration and physical units as the small NPF tests.
Set `DARCY_FLOAT=float64` before importing Warp.

```python
import numpy as np

from DARCY_WARP_PACKAGE import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls
from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662NewtonControls

nz, ny, nx = 2, 4, 6
shape = (nz, ny, nx)
z_interfaces = np.broadcast_to(
    np.array([6.0, 3.0, 0.0])[:, None, None],
    (nz + 1, ny, nx),
).copy()
active = np.ones(shape, dtype=int)
fixed = np.zeros(shape, dtype=int)
fixed[:, :, 0] = 1
values = np.full(shape, 4.0)
initial = np.full(shape, 4.0)
recharge = np.full((ny, nx), 0.001)  # 1 mm/day
cell_types = np.zeros(shape, dtype=int)
cell_types[0] = 1  # upper layer convertible

with WarpDarcySolver3D(
    nx=nx, ny=ny, nz=nz, dx=2.0, dy=3.0,
    device="cuda:0", implementation="fast",
    formulation="mf6_6_6_2_npf",
) as model:
    model.build_from_layers(
        z_interfaces=z_interfaces,
        kx_field=np.ones(shape),
        ky_field=np.full(shape, 0.7),
        kz_field=np.full(shape, 0.2),
        active=active, bc_mask=fixed,
        bc_values=values, R_field=recharge, initial_head=initial,
    )
    heads, info = model.solve_transient_npf(
        recharge_rates=[0.001, 0.0015, 0.00025],  # m/day in successive periods
        dt=7.0, head_prev=initial,
        ss=1e-3, sy=0.15,
        unconfined=True, npf_cell_type=cell_types,
        npf_device_native=True,
        npf_preconditioner="kcycle",
        npf_newton_controls=NPF662NewtonControls(
            timestep_predictor=True, reuse_trial_state=True,
        ),
        npf_kcycle_controls=NPF662KCycleControls(
            levels=3, coarse_sweeps=2,
            cached_line_factors=True,
            preconditioner_precision="float32",
            device_hierarchy_build=True,
        ),
        max_outer=50, max_cycles=500, krylov_restart=12,
    )
    if not info["converged"] or info["accepted_steps"] != 3:
        raise RuntimeError(f"NPF/STO history not accepted: {info}")
    assert heads.shape == (3, nz, ny, nx)
```

`recharge_rates` supplies absolute depth rates for the history, replacing the
built-in `R_field` for those steps. Supply a scalar, one rate per step, a
static `(ny, nx)` map or a `(nstep, ny, nx)` stack. `wel_rates` independently
supplies the well schedule and replaces the built WEL component for those
steps. With explicit recharge and WEL schedules, optional `source_factors`
scale only the additional volumetric `rhs`.
Timestep lengths may also vary, but conductivity, storage, cell types, geometry and
prescribed heads are fixed *within* a single history call. Call
`update_K_fields()` between histories when hydraulic conductivity changes.

Always check `info["converged"]` and `info["accepted_steps"]` before
using the result. If a timestep fails, the output can include a final
unaccepted trial for diagnosis, which must not be advanced as a valid head.
Accepted package/source objects bind the physical inputs, heads, rates and
timestep duration. Their gross/net rates and integrated volumes exclude
rejected trials. Configured WEL rates, prescribed-head compensation and the
realised MF6 cell-budget exchange are distinct quantities and should be
reported separately.

The [current NPF GPU performance guide](../source/npf_gpu_native.rst)
gives the qualified 500 x 500 x 5 runtime, strict control files, matching
MF6 comparison and timing boundaries. This small example demonstrates the
API, not the large benchmark profile.

## MODFLOW 6 formulation attribution

The `mf6_6_6_2_npf` option is an independent implementation of selected
**USGS MODFLOW 6** Node Property Flow (NPF) and Storage (STO) equations and
numerical conventions, using Python and NVIDIA Warp. It is not an official
USGS distribution or a claim to implement every MODFLOW 6 option.

Reference: Langevin, C. D., Hughes, J. D., Provost, A. M., and contributors
(2025), *MODFLOW 6 Modular Hydrologic Model*, version 6.6.2,
USGS software release, [doi:10.5066/F76Q1VQV](https://doi.org/10.5066/F76Q1VQV).
The [official software citation](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/CITATION.cff)
gives the complete contributor list. MODFLOW 6 is released under
[CC0-1.0](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/LICENSE.md).

See [third-party acknowledgements](../THIRD_PARTY_NOTICES.md) for the
complete attribution and the separate NVIDIA Warp adapted-code licence.

## Verification

The layered example checks head and flow against an independent series-resistance solution.

```bash
DARCY_FLOAT=float64 python -m examples.verify_3d_series_flow
```

The November 5 CPU validation covered 20 physical cases over 20 weekly
steps, with 400 successful DarcyWarp/MF6 endpoint comparisons and 800
independently audited engine endpoints. The maximum head difference was
`7.74e-8 m` and the maximum relative face-flow L2 difference was `1.65e-7`.
The historical protocol, archived metrics and replay command are recorded
at the validated revision.

- [Steady NPF comparison](https://github.com/Lincoln-Environmental/DarcyWarp/blob/bbb0e4b594730a8e4e8359b911b342124e2083e2/docs/NPF662_CPU_PARITY_2026-11-05.md)
- [Twenty-step NPF/STO comparison](https://github.com/Lincoln-Environmental/DarcyWarp/blob/bbb0e4b594730a8e4e8359b911b342124e2083e2/docs/NPF662_TRANSIENT_CPU_PARITY_2026-11-05.md)
