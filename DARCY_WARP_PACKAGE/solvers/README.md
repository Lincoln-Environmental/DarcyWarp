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

The backend adapters call the established numerical kernels through explicit
context hooks. This intentionally keeps execution order, hierarchy reuse,
diagnostics, and resource lifetime unchanged while the large K-cycle and
Picard bodies are moved. PCG is already a standalone implementation. The
remaining K-cycle and Picard bodies are still private backend hooks on the
model; they are not duplicated in the new package.

Future nonlinear backends should consume `SolverContext`, use the K-cycle
callback for linear work, and return the existing `(head, info)` contract.  They
must not take ownership of Warp arrays or call `close()`; only the model may do
that.

Legacy aliases are resolved in `registry.py`: `pcg`, `kcycle`, `multigrid`, and
`mg`.  For unconfined solves, `kcycle` continues to mean the Picard/K-cycle
backend.
