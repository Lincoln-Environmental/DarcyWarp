# warped_darcy_v1

Recovered experimental DarcyWarp code from the pre-release history.

Provenance:

- `35e07df`: transient draft for the confined case, steady-state unconfined path, Chebyshev smoother, and AGM/K-cycle work.
- `de1b0f8`: draft 7-point stencil work.
- `f4f6b07`: flattened the unconfined Picard path and extracted `_solve_chebyshev_7point_3d_linear`.
- `351e3b1` / `defa3d7`: split the 3D kernels and solvers into `kernels_3d.py`, `solvers_3d.py`, and `sparse_operator.py`.

What is restored here:

- `solvers_3d.py`: 7-point Chebyshev and K-cycle solvers, plus the 7-point face-conductance, diagonal-preconditioner, and transient-term helpers.
- `kernels_3d.py`: explicit 3D kernel export surface.
- `sparse_operator.py`: sparse 5-point operator helper used by consistency tests.

Reusable 3D case inputs and the river-loss example
---------------------------------------------------

`model_3d_inputs.Model3DInputs` is a validated, solver-neutral container for
structured `(nz, ny, nx)` conductivity, activity, boundary, source, and
initial-head fields. It can build a `WarpDarcySolver3D` through the existing
public API without adding case-specific concepts to the solver.

The historical river-loss case-study dependencies are absent from the tracked
checkout, so its compatibility runner is not a self-contained verification
example. The runnable synthetic verification is
[`examples/verify_3d_series_flow.py`](../examples/verify_3d_series_flow.py):

```bash
DARCY_FLOAT=float64 PYTHONPATH=. python examples/verify_3d_series_flow.py
```

It checks independently known series-resistance heads and fluxes and saves
the completed numerical results. See the feature branch's
[3D capability guide](../docs/3D_CAPABILITIES.md) for formulation-specific limits.

3D execution notes
-------------------

These recovered execution notes are historical. The feature branch's
[capability guide](../docs/3D_CAPABILITIES.md) records the audited API and limits.

The 3D wrapper owns validated host-side operator inputs with in-place RHS,
initial-head, and face-conductance updates. Classic K-cycle/Chebyshev remains
the default. Thin-grid hierarchy construction is axis-aware: a length-one
horizontal axis is never coarsened. The opt-in
`WarpDarcySolver3D(..., implementation="fast")` backend is FP64,
steady-confined, and uses six face arrays, block-partial reductions,
device-side K-cycle alpha updates, and CUDA-graph replay when available.
Unconfined and transient solves continue to use the classic implementation.

Known state:

- The unconfined implementation is Picard-based and exposed through `unconfined=True`.
- The 7-point stencil is exposed through `solve_chebyshev_7point_3d` and `solve_multigrid_kcycle_7point_3d` in `solvers_3d.py`.
- 2D K-cycle transient support is exposed through `transient=True`, `storage_coeff`, `dt`, and `head_prev`.
- 2D unconfined transient Warp smoke and mass-balance tests live in `tests/test_2d_transient.py`.
- Steady-state 2D unconfined MF6 replay fixtures live in `tests/fixtures/unconfined_2d` and are exercised by `tests/test_2d_unconfined.py`.
- The 2D transient unconfined MF6 truth generator is `working_tests/run_2d_transient_vs_mf6.py`; a completed run should create `DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined/mf6_transient_heads.npz.lzma`.
- 3D confined and Picard unconfined transient support is present in
  `solvers_3d.py`; unconfined callers continue to use the classic path until
  a face-refresh implementation is promoted.
