# Development archive and branch policy

The stable public release is `main` / `v0.4.0` at
`2b6c452f982e12c76128109f167da88ffe8c6133`. All cleanup is confined to `dev`.

The previous dev and all pre-cleanup branch tips are preserved under the
[development archive tag](https://github.com/Lincoln-Environmental/DarcyWarp/tree/667c80388ea1b1e67d3e7060a7916b8d1dafedf8).
The archive commit retains the original branch heads as parents. The
[branch-tip manifest](https://github.com/Lincoln-Environmental/DarcyWarp/blob/667c80388ea1b1e67d3e7060a7916b8d1dafedf8/archive/branch-tips.json)
gives each original name and full commit SHA. This archive is not a software
release and its merge parents do not imply that old solver source was merged.

## Recover historical work

Fetch tags before inspecting or recovering the archive:

```bash
git fetch origin --tags
git show archive/pre-dev-reset-2026-11-27:archive/branch-tips.json
git bundle create darcywarp-history.bundle --all
git bundle verify darcywarp-history.bundle
```

Use an original tip from the manifest to inspect its source with
`git show <commit>:<path>`, or create a local research branch if necessary.
Do not merge the historical dev tree into the new dev.

The [retired file manifest](retired_files_20261127.json) records exact paths,
blob hashes and sizes for files removed or relocated from the v0.4.0
development baseline. All remain in the immutable v0.4.0 release. The
published paper's reference datasets, maintained test fixtures and current
qualification receipts remain in the working tree.

## Development workflow

- Main is the stable release branch. No automatic promotion from dev.
- Dev starts from the released main and contains current work.
- Use short-lived branches based on dev when isolation is needed; remove them
  after integration or verified archival.
- Keep generated profiler outputs, scratch runs, duplicate source snapshots
  and large intermediate experiment arrays outside the tracked source tree.
- Preserve meaningful reference data and compact qualification evidence.
- Keep the 100-case maintained numerical suite; do not restore historical
  generated regression matrices.

## SHANZ compatibility

The old dev at `3fe45a5a70655c2936896448fbb075e9f2c324b7` retained a separate
coupling implementation, described in its
[SHANZ contract](https://github.com/Lincoln-Environmental/DarcyWarp/blob/3fe45a5a70655c2936896448fbb075e9f2c324b7/DARCY_WARP_PACKAGE/solvers/SHANZ_COUPLING.md).
The current SHANZ Orton Bradley branch at
`606da961523f049cd5f20f044e5083ee15839704` still imports the public
`production_secant_sy_settings` function and `solve_surface_contact_2d`.
Its Darcy hot-start path also imports `nonlinear.coastal.CoastalSubfaces2D`
from older SHANZ-specific branches. These are actual downstream dependencies.

Recover the configuration function from the current solver settings without
changing their numerical values. Port surface contact separately: the
current semismooth Newton solver needs an explicit contact constraint and
reaction output while retaining accepted previous heads, physical storage,
WEL/recharge ownership and exact exchange budgets. The old persistent
transient interval also requires reconciliation with the current period
driver; it cannot simply be copied because it depends on
`TransientUnconfinedSession`, absent from the released implementation.
Reconcile coastal contact and hot-start APIs only against the active SHANZ
callers. Keep the transformed-terrain research solver, experimental steady
preconditioners, domain QA and cuDSS work in the archive until needed.

No full SHANZ integration is claimed by repository housekeeping. Preserve
the archived original tips until these interfaces have been deliberately
ported and checked. Do not reintroduce the obsolete solver trees.
