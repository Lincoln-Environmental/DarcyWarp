# 2D solver backends

`WarpDarcySolver` owns model construction, host fields, Warp arrays, the
multigrid hierarchy, CUDA graph cache, and `close()` cleanup.  It creates a
`SolverContext` containing references to those resources and dispatches through
`registry.py`.

| Backend | Formulation | Time mode | Status |
| --- | --- | --- | --- |
| `confined_pcg` | confined | steady only | production |
| `confined_kcycle` | confined | steady or transient | production |
| `unconfined_picard_kcycle` | unconfined Picard | steady or transient | production default |
| `unconfined_semismooth_newton_kcycle` | unconfined Newton (FGMRES + K-cycle preconditioner) | steady or single-period transient | experimental |
| `unconfined_fas` | unconfined FAS V-cycle | steady or single-period transient | experimental |

Capability metadata (`formulations`, `supports_transient`, `experimental`,
`production_default`, `supports_production_period_driver`) lives in
`DARCY_WARP_PACKAGE/solver_capabilities.py` (re-exported here as
`solvers/capabilities.py`). `select_backend` emits a runtime warning when an
experimental backend is selected, and the multi-period transient production
driver (`transient_unconfined.py`) gates on
`supports_production_period_driver`, so the experimental backends are
single-period `solve(...)` propositions today. Their shared nonlinear
foundations (authoritative residual/Jacobian-vector operator, kernels, host
reference) live in `DARCY_WARP_PACKAGE/nonlinear/`; Newton machinery in
`fgmres.py`, `kcycle_preconditioner.py`, and `newton_kernels.py`; the FAS
hierarchy/state/kernels in `fas_hierarchy.py`, `fas_state.py`, and
`fas_kernels.py`.

The backend adapters invoke the extracted implementations directly through
the typed, model-owned context. PCG, K-cycle orchestration/device buffers,
Picard, and the transient period driver live in this package; the model keeps
only construction, resource ownership, public compatibility wrappers, and
dispatch preparation. Execution order, hierarchy reuse, diagnostics, and
resource lifetime remain unchanged.

Future nonlinear backends should consume `SolverContext`, use the K-cycle
callback for linear work, and return the existing `(head, info)` contract.  They
must not take ownership of Warp arrays or call `close()`; only the model may do
that.  The semismooth-Newton and FAS backends above are the reference
implementations of this pattern.

Legacy aliases are resolved in `registry.py`: `pcg`, `kcycle`, `multigrid`, and
`mg`.  For unconfined solves, `kcycle` continues to mean the Picard/K-cycle
backend.
