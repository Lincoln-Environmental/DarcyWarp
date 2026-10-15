# Local active-stencil validation for terrain contacts

Starting local commit: `48dd46df6626176526473864e49e3e4ddf67c6a5`
(`fix/terrain-contact-law-runtime`). Starting working tree: no tracked edits;
58 unrelated untracked entries, preserved. The original status is recorded in
`/tmp/darcy-masked-contact-start.status`. Local `dev` remains at `e77c8fff`;
it does not contain the starting contact-law commit. Work is on the separate
local branch `fix/terrain-masked-contact-stencils`, based on 48dd46d.

Changed files:

- `DARCY_WARP_PACKAGE/physics/terrain_reference_3d.py`
- `tests/test_terrain_contact_stencils.py`
- This report.

No workflows were created, changed, deleted or triggered. No GitHub Actions,
hosted validation, remote writes, push, merge, pull request, cloud compute or
package installation was used. Existing environments and unrelated work were
preserved. The existing workflow blob remains
`6abbeae7ebe78ae4aae769a81bf53d3c0ae79ee6`.

## Baseline reproduction

Actual production grids, metrics, contacts and reference operators reproduced
the supplied two-layer, unit-spacing case with horizontal mask:

```
0 0 0
1 1 1
0 0 0
```

For the west contact at `(0, 1, 0)`, `h=10+y`, owning head 11,
`y_offset=-0.25`, boundary head 10.75, C=1, area=1, distance=0.5 and
elevation=1.5, the physical west-plane flux is zero. The baseline accepted
the contact and returned **+0.25**, in both the independent reference and
the actual production Warp CPU/CUDA refresh, gate and flux kernels.
The discharge-only contact also returned +0.25 and opened its gate.

The missing derivative row has **nnz=1**, but its sole stored entry is zero:
both neighbour coefficients are zero. Thus checking sparse `nnz` would retain
the bug. Restoring just the minus-y neighbour gives coefficients (-1,0);
restoring just the plus-y neighbour gives (0,1). Both give exactly zero flux
in the reference and both Warp backends, with closed discharge-only gates.
The executed numerical probe is recorded in
`/tmp/darcy-masked-contact-reproduction.log`.

All 321 new regressions were added before production changes. On the starting
code they produced **130 failed, 191 passed**. With the validation fix, the
same assertions produced **321 passed**, including after final error-message
formatting. Logs:
`/tmp/darcy-masked-contact-{baseline,fixed,targeted-final}.log`.

## Validation rule and scope

`validate_contact_tangential_stencils()` reads the existing minus/plus
coefficients produced by `derivative_matrix()` at the owning column. A required
axis is available if either neighbour coefficient is nonzero. This is constant
work per contact, with no sparse copies, global scans or additional cache.

Lateral contacts require their tangential derivative when its offset is
nonzero. Top/bottom contacts additionally require any horizontal derivative
whose bounding-interface slope can contribute a metric coupling. Positive
physical K is already enforced by the existing contact/material validation.
These requirements reflect couplings possible over the constitutive states,
not their value at the current saturation, head or discharge gate.

Only explicit C=0 waives unnecessary horizontal reconstruction: the preserved
physical contact law makes all its exchange coefficients and RHS exactly
zero. A natural lateral contact with C=0 only at its dry endpoint still requires
the stencil because its conductance becomes positive on wetting. Unused axes
are not required, and either one-sided orientation remains sufficient.

The old dimension-only horizontal checks are replaced. All other validation
is retained, including finite values, offsets within the owning face, positive
area/distance, active ownership, face-area limits, positive material constraints
and the separate one-layer vertical-reconstruction rule. No contact equations,
constitutive law, floor, storage, stopping tolerances or solver controls changed.
The earlier completed topology/zero-K/Krylov/runtime fixes are untouched.

Errors identify the cell, face, required axis, missing active-neighbour stencil
and inability to reconstruct the specified offset/metric coupling on the local
active mask. `update_contacts()` rejects invalid contacts at its existing
validation boundary. `update_geometry()` still installs geometry as before;
the next solve rejects invalid contact reconstruction before using it.
Restoring a valid neighbourhood creates new metrics and works correctly.
No session transaction behavior was redesigned.

## Regression coverage

All names below are in `tests/test_terrain_contact_stencils.py`:

- `test_masked_offset_requires_local_stencil`: all six faces, x/y top/bottom
  offsets, ordinary/discharge-only, explicitly stored zero derivative rows.
- `test_missing_stencil_rejected_independent_of_state`: dry/partial/saturated,
  explicit positive/natural conductance, constant current horizontal heads and
  matching boundary heads (zero current flux / closed discharge gates).
- `test_one_sided_contact_reconstructs_physical_affine_flux`: both neighbour
  orientations, ordinary/discharge-only, explicit/natural C, independent zero
  normal-flux expectations and actual Warp CPU/CUDA coefficients/flux/gates.
- `test_centered_contact_does_not_require_unused_horizontal_stencil` and
  `test_sloping_contact_requires_only_used_horizontal_axis`: flat centred and
  sloping top/bottom controls with independent nonzero physical affine flux.
- `test_zero_exchange_needs_no_offset_stencil`: exact absent-contact matrix,
  RHS and flux equivalence, plus actual Warp zero coefficients/flux.
- `test_zero_exchange_offset_allowed_with_single_column_dimension` and
  `test_zero_exchange_keeps_other_geometry_and_material_checks`.
- `test_vertical_reconstruction_requirement_remains_separate`.
- `test_session_contact_update_validates_mask_and_zero_exchange`: constrained
  absent/zero-contact solve equivalence and independent physical budgets.
- `test_geometry_update_revalidates_contact_stencil`: both one-sided layouts,
  both drivers and devices; valid numerical contact updates reuse workspace
  and captured CUDA graph, invalid masks raise, restored masks rebuild safely.

## Locally executed validation

Reused existing `/tmp/darcy-terrain-contact-py312`, with Python **3.12.14**,
Warp **1.17.0**, actual `WP_FLOAT` and `NP_FLOAT` both **float64**. Devices:
Warp CPU x86_64 and local cuda:0, NVIDIA GeForce RTX 4070 Ti SUPER, driver
595.91.07 (Warp toolkit 12.9). No packages or shared environments were changed.

Set before imports:

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
PYTHON=/tmp/darcy-terrain-contact-py312/bin/python
RUFF=/home/patrickdurney/miniconda3/envs/darcywarp/bin/ruff
```

Exact test commands (first command run on both baseline and fixed code):

```bash
$PYTHON -m pytest -q tests/test_terrain_contact_stencils.py --tb=short

$PYTHON -m pytest -q tests/test_terrain_contact_exchange.py tests/test_terrain_lateral_contacts.py tests/test_3d_terrain_reference.py tests/test_terrain_device_picard.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_gmres.py tests/test_terrain_runtime.py tests/test_terrain_contact_stencils.py --tb=short

$PYTHON working_tests/run_terrain_cpu_tests.py -q tests/test_terrain_contact_exchange.py tests/test_terrain_lateral_contacts.py tests/test_3d_terrain_reference.py tests/test_terrain_device_picard.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_gmres.py tests/test_terrain_runtime.py tests/test_terrain_contact_stencils.py --tb=short -rs

$PYTHON -m pytest -q tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py --tb=short

$PYTHON working_tests/run_terrain_cpu_tests.py -q tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py --tb=short -rs
```

| Validation | Passed | Failed | Skipped |
|---|---:|---:|---:|
| New regressions, baseline | 191 | 130 | 0 |
| New regressions, fixed | 321 | 0 | 0 |
| Requested suites + new file, CUDA selected | 958 | 0 | 0 |
| Same suites, CPU selected | 954 | 0 | 4 |
| Additional scientific/SHANZ diagnostics, CUDA selected | 77 | 0 | 0 |
| Same additional tests, CPU selected | 77 | 0 | 0 |

CPU selection changes the shared default backend; explicitly parametrized
CPU/CUDA tests still execute their requested backend. The four CPU-default
skips are existing CUDA-only volume parity, drying/rewetting parity, coarse
refresh and GMRES graph accounting tests. All four execute and pass with CUDA
selected. Capture and reuse remain enabled. Existing SHANZ 32, full rollback,
candidate acceptance, constitutive refresh, contact gates, workspace topology,
zero-K coefficients and host/device budget handoff tests remain passing.

Full-suite logs are `/tmp/darcy-masked-contact-suite-{cpu,cuda}.log`;
additional logs are `/tmp/darcy-masked-contact-additional-{cpu,cuda}.log`.

Local checks:

```bash
$RUFF check DARCY_WARP_PACKAGE/physics/terrain_reference_3d.py tests/test_terrain_contact_stencils.py
$RUFF format --check tests/test_terrain_contact_stencils.py
$PYTHON -m compileall -q DARCY_WARP_PACKAGE/physics/terrain_reference_3d.py tests/test_terrain_contact_stencils.py
git diff --check
```

All pass. Full `ruff format --check` on the modified existing reference file
also ran; it reports one file requiring legacy formatting on both baseline
and current code. Broad formatting was not applied. Logs are
`/tmp/darcy-masked-contact-format-{baseline,current}.log`.

No functional failures remain in the executed fixed tests. Full-repository
testing and the separate 64x64 low-K convergence problem were not run. No GPU
performance conclusions are drawn. All changes remain local; workflow files
and remote branches are untouched.
