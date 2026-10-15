# Terrain CPU/MF6 validation work log

- Source fetched 2026-10-16: origin/dev 377ada20a92ad5f5a40af1ed9844cd8be31beba4; separate validation/terrain-cpu-mf6 worktree. Original untracked work untouched.
- Existing darcywarp environment retained: Python3.14.7, NumPy2.3.5, SciPy1.16.3, Warp1.17.0, FloPy3.9.5, MF6 6.6.2. MF6 SHA256 21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176. No upgrades.
- Reviewed reports from physics correction through Newton validation, underlying comparison JSONs and host audit. Located prior diagnosis in /tmp/darcy-low-k-diagnosis/docs/LOW_K_64_DIAGNOSIS_2026-10-15.md. TERRAIN_LOW_K_64_DIAGNOSIS report absent, not substituted.
- Canonical input serialization retains exact original build_case arrays; no demo main, warm starts, or source redistribution. Coordinate conventions and source-on-fixed semantics explicit.
- Original GPU capped baseline and complete unchanged GPU matrix executed before production edits.
- Independent CPU residual uses existing NumPy flux/gradient-incidence reference. New sparse Jacobian derives coefficient sensitivities directly from host face equations. Smooth n32 directional FD reaches relative error 6.03e-10; no GPU derivative used.
- MF6 native adapter is explicitly different formulation, not certified parity: DIS/XT3D, Cartesian tensor, WEL exact per-cell sources, native DRN with independently prescribed full-natural conductance. Rejected adapter attempts retained; no successful claim for invalid input/output.

## Predeclared comparison criteria

Same-equation residual/flux parity: existing FP64 scientific tolerances, relative 3e-12 plus absolute 2e-11 m3/day; derivative relative 2e-7 at smooth states before roundoff. Final nonlinear acceptance unchanged (head RMS1e-6; physical RMS1e-12 OR L1/throughput1e-8). Head differences >1e-4 m flag investigation, not automatic wrong-model status. Physical-output differences are dimensional and normalized to source/gross exchange; no retrospective percentage pass threshold. MF6 native success AND independently reconstructed cell conservation (RMS1e-12 OR L1/gross1e-8), exact WEL allocation, and finite non-sentinel heads required. Different formulations assessed by controlled geometry/saturation/contact experiments and fixed-domain refinement, never a same-equation parity flag.
