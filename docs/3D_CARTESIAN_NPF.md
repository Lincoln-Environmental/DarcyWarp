# Structured 3D groundwater flow

Use `WarpDarcySolver3D` or `create_solver(dim=3, ...)` from
`DARCY_WARP_PACKAGE`. The factory takes `nx`, `ny`, `nz`, `dx` and `dz`.
`dy` defaults to `dx`. Cell fields have shape `(nz, ny, nx)`, ordered from the
top layer down. With metres and days as units, conductivity is in m/day and
`rhs` is volumetric injection in m³/day. Multiply an areal recharge rate by
`dx * dy` once to form the source. Use the active and fixed-head masks to set
the model domain and prescribed heads. Exterior and inactive faces are no-flow.

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

NPF requires `DARCY_FLOAT=float64` before import. Use Python 3.12 and Warp
1.17.0 for the validated configuration. Convertible, layered and transient
NPF use `implementation="fast"` and `solver="kcycle"` on the model. The
default Krylov choice for those NPF solves is diagonal GMRES. Select
`npf_preconditioner="kcycle"` for flexible GMRES with a K-cycle
preconditioner, or `npf_krylov_method="fgmres"` for diagonal flexible GMRES.
`NPF662NewtonControls` and `NPF662KCycleControls` are package exports.
See the [GPU implementation guide](3D_IMPLEMENTATIONS.rst) for the measured
K-cycle configuration, Cartesian fast-path selection and timing conventions.
Existing defaults remain unchanged.
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
to choose a separate iterative starting point, `update_rhs` to change
the volumetric source, `update_boundary_values` to change prescribed heads,
and `update_K_fields` to change physical conductivity and its derived
faces. Check `info["converged"]`, the residual and the water balance before
advancing the timestep. Returned heads own their storage. A later steady
solve clears transient storage terms.

The MF6 6.6.2 contract covers default NPF connections and default Ss/Sy
storage with Newton saturation smoothing. XT3D, variable CV, rewetting,
perched flow and alternative STO options are outside this contract. Fresh
release qualification was CPU FP64. The subsequent local GPU campaign,
portable sanity runner, timings and remaining convergence limits are recorded
in the [GPU release notes](3D_GPU_RELEASE_NOTES.md). CPU qualification does
not imply that every GPU case meets the same iteration budget.

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
