# 2D solver backends

`WarpDarcySolver` owns model construction, host fields, Warp arrays, the
multigrid hierarchy, CUDA graph cache, and `close()` cleanup.  It creates a
`SolverContext` containing references to those resources and dispatches through
`registry.py`.

| Backend | Formulation | Time mode |
| --- | --- | --- |
| `confined_pcg` | confined | steady only |
| `confined_kcycle` | confined | steady or transient |
| `unconfined_picard_kcycle` | unconfined Picard | steady or transient |

The backend adapters invoke the extracted implementations directly through
the typed, model-owned context. PCG, K-cycle orchestration/device buffers,
Picard, and the transient period driver live in this package; the model keeps
only construction, resource ownership, public compatibility wrappers, and
dispatch preparation. Execution order, hierarchy reuse, diagnostics, and
resource lifetime remain unchanged.

Future nonlinear backends should consume `SolverContext`, use the K-cycle
callback for linear work, and return the existing `(head, info)` contract.  They
must not take ownership of Warp arrays or call `close()`; only the model may do
that.

Legacy aliases are resolved in `registry.py`: `pcg`, `kcycle`, `multigrid`, and
`mg`.  For unconfined solves, `kcycle` continues to mean the Picard/K-cycle
backend.
