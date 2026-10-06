# Persistent transient-unconfined period API

`TransientUnconfinedSession` in `transient_unconfined.py` exposes the existing
production Picard/device/adaptive period engine. The multi-period replay and
the public single-period `model.solve` path share this engine. The extraction
retains its equations, storage law, convergence controls and inner/adaptive
algorithm.

Use `production_secant_sy_settings()` from `transient_config.py`. The working
replay settings module imports the same factory. It configures convertible
secant Sy storage, current-Picard storage reference, confined startup,
Chebyshev K-cycles, four hierarchy levels and 500 minimum coarse cells.
No numerical constants changed during this move.

```python
from DARCY_WARP_PACKAGE.solvers.transient_config import production_secant_sy_settings
from DARCY_WARP_PACKAGE.solvers.transient_unconfined import TransientUnconfinedSession

settings = production_secant_sy_settings()
session = TransientUnconfinedSession(
    model=model, k_field=conductivity, zbot_field=bottom, ztop_field=top,
    sy=specific_yield, ss=specific_storage,
    active=active, bc_mask=prescribed_head_mask, bc_values=prescribed_heads,
    storage_mode=settings["unconfined_storage_mode"],
    storage_reference=settings["storage_reference"],
    solve_controls=settings["solve_controls"],
)
head, diagnostics = session.advance(
    head_prev=accepted_previous_head,
    recharge_rate_m_per_day=spatial_recharge,
    dt=duration_days,
)
```

Recharge must be finite, match the grid and be zero on inactive and prescribed
head cells. It may have either sign. Accepted hydrological state belongs to
the caller. Every period explicitly seeds head work from `head_prev`, updates
spatial recharge in place and resets adaptive period controls. A rejected
trial can be followed by another call from the restored accepted head without
restoring opaque solver state.

The suspended shared engine retains the hierarchy, face operator, multigrid
arrays, transient head/storage arrays and CUDA graphs. Neither a new
hierarchy nor a new set of full-grid transient arrays is built per ordinary
period. Static grid/material/boundary changes require a new compatible
session. `invalidate()` releases derived scratch and graphs. `model.close()`
releases the cached public interval session.

The public path is `model.solve(formulation="unconfined", transient=True,
solver="unconfined_picard_kcycle", use_device_transient_fast_path=True, ...)`.
Its interval bridge caches a session on the model, checks static compatibility,
and integrates native storage and boundary budgets over accepted adaptive
substeps. It never calls the whole multi-period driver per interval.

DRN/RIV gated boundaries and native surface complementarity require the
existing semismooth Newton solver. A coupled caller invalidates Picard work
when entering contact and seeds a subsequent normal period from the explicit
accepted post-contact head. No contact equation is implemented here.

The canonical factory defaults to FP64. The production replay entrypoint
optionally sets `transient_mixed_precision_enabled=True`; that enables its
existing FP64 outer / FP32 inner implementation, not the unrelated steady
`MixedFastConfig` solver. Report the actual precision and solver path.

Diagnostics include strict/practical acceptance, outer and inner work,
adaptive accepted substeps/retries, residuals, hierarchy depth, face/device
path, session setup/period counts and graph reuse. A returned head is not
evidence of convergence on its own.

## Validation

From the DarcyWarp root:

```bash
MPLBACKEND=Agg python -m pytest -q tests/test_transient_session.py
MPLBACKEND=Agg python -m working_tests.validate_transient_session \
  --artifact DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined_100x100_3w_ugly_t_s42/mf6_transient_heads.npz.lzma \
  --oracle-revision 5163cb1 --output /tmp/transient-session-small.json
MPLBACKEND=Agg python -m working_tests.validate_transient_session \
  --artifact DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined_1000x1000_30w_ugly_t_s42/mf6_transient_heads.npz.lzma \
  --oracle-revision 5163cb1 --mixed --production-acceptance \
  --output /tmp/transient-session-production.json
```

The oracle loads the original Git source without resetting the checkout.
Small cases compare head/storage fields at roundoff and work/status exactly.
The million-cell mixed replay has nondeterministic reduction/iteration work:
original-versus-original maximum head variation was 2.80e-5 m. Its gate uses
unchanged native stopping and MF6 head/mass acceptance, reports actual
head/work differences, and checks uniform raster forcing, alternate-trial
rollback and persistent buffer/hierarchy/graph identities. It does not claim
bitwise equality or identical work on that large mixed case.
