# Public 2D surface/groundwater coupling contract

The current `dev` exposes the genuine coupling interfaces previously developed
on the separate SHANZ coupling branch. Their absence from dev was an integration
gap, not removal of the underlying transient or Newton solvers.

`transient_config.production_secant_sy_settings()` returns fresh controls from
the current production configuration. The replay harness imports the same
settings, retaining its existing names and values. Coupling supplies its own
accepted previous head; the profile does not initialize a historical warm start.

Ordinary `solve(..., transient=True, use_device_transient_fast_path=True)` uses
the shared production period engine through `transient_interval`. It accepts
spatial recharge, preserves the actual strict/adaptive solve controls, and
integrates budgets over accepted substeps. `initial_head` must equal the explicit
accepted `head_prev`. Cached sessions own workspace, not accepted physical state;
contact invalidates that cache and rollback reissues the accepted state.

`accepted_drn_discharge_2d` and `accepted_riv_discharge_2d` validate a converged
Newton state and return its physical boundary rates. Positive rates leave the
aquifer. Dynamic river fields remain explicit and unsupported backends reject.
`surface_contact_2d.solve_surface_contact_2d` imposes storage-preserving head
contact with finite supply and atmospheric release. Its reaction comes from the
same unconstrained FV residual, including exact storage; these cells are not CHD
cells and keep their physical storage. Transient contact requires `head_prev`.

The only correction to an existing physical operator removes the steady
isolated-cell identity pin from transient residual/Jacobian rows. A transient
isolated cell has zero hydraulic flow and physical storage, not a fictitious
head-dependent sink. The steady pin and all other established constitutive
laws/defaults are unchanged; an independent storage/conservation regression
covers the correction. Accepted RIV and DRN budget reporting was completed.
Coastal closures, steady startup changes and unrelated historical work were not
integrated.

Validation uses the existing CPU/CUDA nonlinear, production transient, accepted
flux, contact, session and exact rollback tests, plus Overlander's 84 gated
controls on ordinary current Overlander/DarcyWarp/RichardsWarp/SHANZ checkouts.
No historical source export is required.
