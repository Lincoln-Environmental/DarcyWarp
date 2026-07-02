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

Known state:

- The unconfined implementation is Picard-based and exposed through `unconfined=True`.
- The 7-point stencil is exposed through `solve_chebyshev_7point_3d` and `solve_multigrid_kcycle_7point_3d` in `solvers_3d.py`.
- Confined transient support is present through `transient=True`, `storage_coeff`, `dt`, and `head_prev`.
- Transient unconfined support is scaffolded but explicitly raises as not implemented.
