# Terrain contact exchange and supported runtime

Starting commit: `e77c8fff83d22c0f1bafd21ad0c68c36015d859c` (`dev`).
Work branch: `fix/terrain-contact-law-runtime`. No merge or push is authorized.
Unrelated untracked work is preserved.

## Contact law derivation (before implementation)

`TerrainContact3D` documents an override of the normal `K*area/distance`
conductance, retaining metric correction. Existing lateral contacts already
scale their entire reconstructed flux with the override. No call site defines
an alternative numerical-splitting contract. The old top/bottom override test
instead expected natural-K flux for different overrides; that expectation
conflicts with the physical exchange contract and must change intentionally.

Let `s=-1` at the top and `s=+1` at the bottom, `P` be physical subface area
projected onto the map plane, `d` its logical distance, `(u,v)` its offsets,
`(sx,sy)` the bounding interface slopes and `(cx,cy)` the cell-centre slopes.
With wet fraction `f`, put `Ex=Kx*f`, `Ey=Ky*f`,
`N=Kz+Ex*sx**2+Ey*sy**2`, and
`A=Kz+Ex*sx*cx+Ey*sy*cy`. Natural conductance is `Cn=P*N/d`.
For a physical exchange conductance `C`, scale the complete natural flux by
`C/Cn`, giving `S=C*d/N`; without an override, `C=Cn` and `S=P`.
The existing reconstructed contact row is

```
q = C*(h-hb) - bz*gz - bx*gx - by*gy
bx = s*S*Ex*sx - C*u
by = s*S*Ey*sy - C*v
bz = s*(S*A-C*d)
   = s*S*(Ex*sx*(cx-sx)+Ey*sy*(cy-sy))
```

The final form of `bz` avoids subtracting equal terms on flat/parallel faces.
Here `gz` differentiates in positive depth; for physical affine head
`h=a*x+b*y+c*z`, `gz=-c`, `gx=a+c*cx`, `gy=b+c*cy` and
`h-hb=-u*gx-v*gy-s*d*gz`. Substitution gives
`q=s*S*(Kz*c-Ex*sx*a-Ey*sy*b)`, exactly `C/Cn` times the physical natural
Darcy flux. Thus flat centred contacts give `C*(h-hb)` for arbitrary discrete
interior heads, `C=0` annihilates the complete contact row and RHS, and
`C=Cn` recovers the natural law. Apply discharge gating only after computing
this physical flux. No interior equation, floor or storage law changes.

For overridden top/bottom contacts, all three metric coefficients have form
`constant + f*B/(N0+f*D)`, where `N0=Kz>0` under the existing contact
validation. Given dry/wet endpoints, the exact interpolation weight is
`w=f*N1/((1-f)*N0+f*N1)`. This same weight is required for `bz`, `bx`, AND
`by`. Other contacts and natural top/bottom rows remain affine in `f`.
No epsilon conductivity is needed. Candidate evaluation and rollback must
continue refreshing these rows before gates, physical residuals and MG state.

The 64x64 low-K stress case uses lateral contacts without overrides; this
boundary fix provides no evidence that its separate convergence failure is
resolved.

### Authorized extension: lateral tangential offsets (derived before coding)

After the additional defect was reproduced on both e77c8fff and the fixed tree,
the user explicitly requested fixing lateral tangential offsets too. For an
x-normal physical plane, let its centroid be `(s*d,v,dz)` relative to the
cell centre. The physical affine head difference is
`h-hb=-s*d*a-v*b-dz*c`, while the stored tangential gradient is
`gy=b+c*cy` and `gz=-c`. With `by=-C*v`, the coefficient required to recover
`q=-s*C*d*a` is

```
bz = C*(dz-v*cy) = -C*(depth_offset+v*cy)
```

For a y-normal plane the corresponding coefficient is
`bz=-C*(depth_offset+u*cx)`, with `bx=-C*u`. The slope correction includes
only the tangential offset, never the normal distance. This removes the
spurious `C*c*(u*cx+v*cy)` flux without changing zero-offset lateral contacts
or the normal physical conductance. These coefficients remain affine in wet
fraction, so the existing lateral device endpoint interpolation is exact.

## Confirmed reproductions and focused implementation

Actual reference objects and production Warp CPU/CUDA refresh, gate and flux
kernels reproduced the flat unit contact defect in both orientations:

| Override C | Physical outward flux | Baseline reference and Warp flux |
|---:|---:|---:|
| 0 | 0 | -1 |
| 0.1 | 0.1 | -0.85 |
| 1 | 1 | 0.5 |
| 2 | 2 | 2 |

Reversed head gradients also opened zero-conductance discharge-only contacts.
The original 224 new contact cases gave **138 failed, 86 passed** before the
production changes. The same assertions then gave **224 passed**. This includes
flat independent `C*delta_head` expectations, not just CPU/device agreement.

The reference now uses the derived metric-defect form of `bz`; the other
overridden coefficients already used the correct scale. Device endpoint
refresh applies the normal-conductivity weight to all three metric terms.
Natural contacts remain affine in wet fraction. The old override test now
expects the complete natural flux scaled by `C/Cnatural`, an intentional
correction to its former natural-flux-for-every-override expectation.
No budget code change was needed: it already evaluates the physical contact
flux, so correcting the law also corrects its conservation accounting.

New regressions:

- `test_flat_contact_physical_exchange`: both orientations, ordinary and
  discharge-only, both head differences, C=0/0.1/1/2/omitted, on CPU/CUDA.
- `test_sloping_override_refresh_and_affine_flux`: independent affine physical
  flux; nonparallel slopes, anisotropy and both centroid offsets; f=0/0.2/0.7/1;
  C=0/0.1/3/omitted; exact reference/device coefficients and fluxes; overrides
  equal to each state's natural conductance are checked through actual kernels.
- `test_zero_contact_equals_absent_operator_solve_budget`: sparse matrix and
  RHS equality, zero packed couplings/RHS, independently recomputed budgets,
  final fresh solves with both drivers and both devices, CUDA capture/replay.

The six existing terrain files preserve natural-K affine and h=z tests,
discharge gating, candidate acceptance, full rollback, coarse refresh,
topology updates/reuse, zero-K orientations, handoff budgets and SHANZ 32.

## Supported environment and clean installation

Warp 1.11.0 actually raises `ImportError: cannot import name 'GMRES'` on the
baseline; its uppercase GMRES class and required private helpers are absent.
The pre-fix environment/runtime regressions gave **4 failed, 1 passed,
4 deselected**. Dependency pins and terrain diagnostics were the failures.

The canonical policy is **Python 3.12 with exactly warp-lang 1.17.0**.
The private API is guarded at terrain-module use. Unsupported versions and
missing private symbols produce an actionable installation message. Lightweight
package imports stay lazy; no guard was added to Cartesian solvers. Both
dependency files and README agree. A CPU smoke GitHub Actions workflow uses
Python 3.12 and the canonical pip constraint; it has been parsed locally but
has not been run on GitHub (nothing was pushed).

Runtime regressions are `test_canonical_terrain_runtime_pins`,
`test_package_import_stays_lazy_without_site_packages`,
`test_terrain_entry_reports_unsupported_runtime` and
`test_real_terrain_runtime_smoke` (9 parametrized cases in total).

Installation validation used a new venv, not a modified developer environment:

```bash
/home/patrickdurney/miniconda3/envs/darcywarp312/bin/python -m venv /tmp/darcy-terrain-contact-py312
/tmp/darcy-terrain-contact-py312/bin/python -m pip install warp-lang==1.17.0 numpy scipy pandas pytest flopy
# Confirm the installation also satisfies the final canonical constraint:
/tmp/darcy-terrain-contact-py312/bin/python -m pip install -c requirements-pip.txt warp-lang numpy scipy pandas pytest flopy
/tmp/darcy-terrain-contact-py312/bin/python -m pip check
/tmp/darcy-terrain-contact-py312/bin/python -m pip freeze
```

Observed Python **3.12.14**, Warp **1.17.0**, NumPy **2.5.4**, SciPy **1.18.1**,
pandas **3.0.6**, pytest **9.1.1**, FloPy **3.11.0**. `pip check` reports no broken
requirements. The solver-only installation does not validate the optional
GIS/torch/MODFLOW installation from the full conda specification; those
dependencies were not changed. Initial baseline/contact checks also used the
existing Python 3.14.7 / Warp 1.17.0 environment, without changing it.

A second clean venv verified actual unsupported-version diagnostics:

```bash
/home/patrickdurney/miniconda3/envs/darcywarp312/bin/python -m venv /tmp/darcy-terrain-unsupported-py312
/tmp/darcy-terrain-unsupported-py312/bin/python -m pip install warp-lang==1.11.0 numpy scipy pandas flopy
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests /tmp/darcy-terrain-unsupported-py312/bin/python working_tests/smoke_terrain_runtime.py --device cpu
```

The last command fails intentionally at real `build_from_terrain()` with
`Terrain GMRES requires warp-lang==1.17.0 ... found 1.11.0` and the pip command
to repair it. Existing environments were preserved. The old developer
`darcywarp312` environment additionally lacks pandas, so its full public-entry
probe stops at that missing dependency; the second clean venv removes that
ambiguity. No unsupported Warp solve is reported as passed.

## Validation commands and observed results

Commands below use the new venv and these variables set before imports:

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
PYTHON=/tmp/darcy-terrain-contact-py312/bin/python
```

| Check | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Final six-file suite + all new contact/runtime tests, CUDA default | 714 | 0 | 0 |
| Final same suite, CPU default | 710 | 0 | 4 |
| Additional hierarchy, Cartesian/terrain physics and live MF6 suite | 74 | 0 | 0 |
| New top/bottom contact/runtime regressions, targeted | 233 | 0 | 0 |
| New lateral regressions, targeted | 196 | 0 | 0 |

```bash
$PYTHON -m pytest -q tests/test_terrain_device_picard.py tests/test_3d_terrain_gmres.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_reference.py tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py --tb=short

# Final combined CUDA-selected validation, including the authorized extension:
$PYTHON -m pytest -q tests/test_terrain_device_picard.py tests/test_3d_terrain_gmres.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_reference.py tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py tests/test_terrain_contact_exchange.py tests/test_terrain_lateral_contacts.py tests/test_terrain_runtime.py --tb=short

$PYTHON -m pytest -q tests/test_terrain_contact_exchange.py tests/test_terrain_runtime.py --tb=short

$PYTHON -m pytest -q tests/test_terrain_lateral_contacts.py --tb=short

$PYTHON working_tests/run_terrain_cpu_tests.py -q tests/test_terrain_device_picard.py tests/test_3d_terrain_gmres.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_reference.py tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py tests/test_terrain_contact_exchange.py tests/test_terrain_lateral_contacts.py tests/test_terrain_runtime.py --tb=short -rs

$PYTHON -m pytest -q tests/test_3d_hierarchy_scientific_audit.py tests/test_3d_hierarchy_axes.py tests/test_3d_solver.py tests/test_3d_physics_validation.py --tb=short

$PYTHON working_tests/smoke_terrain_runtime.py --device cpu
$PYTHON working_tests/smoke_terrain_runtime.py --device cuda:0
```

The standalone requested six-file command also ran before the lateral extension,
giving **278 passed**. The final combined commands rerun all six files after the
extension, including seven additional centred/offset fixture cases; none of
the earlier topology, zero-K or budget assertions were changed.

The full additional four-file command completed all **74 tests**. A redundant
confirmation with `-k 'not identical_transient_histories_vs_mf6 or terrain'`
was stopped once that superset passed: it recorded **54 passed, 6 deselected**
before KeyboardInterrupt, and is not claimed as a completed passing suite.

CPU default changes only the shared test backend; explicit CPU/CUDA parameters
still run as requested. The four skips are CUDA-only graph accounting,
device Picard volume parity, nonparallel drying/rewetting parity and coarse
refresh. All four run and pass with CUDA default. No graph cache was disabled.

Devices: Warp CPU x86_64 and cuda:0, NVIDIA GeForce RTX 4070 Ti SUPER, driver
595.91.07 (Warp reports CUDA toolkit 12.9). Both standalone smoke commands pass
both host and device Picard, validate the real GMRES subclass and independently
recomputed budgets, and retain the same workspace on the second solve.
Each yields max head error **8.1358e-11 m**, head residual RMS **3.54395e-11 m**,
volumetric cell residual RMS **7.79669e-11**, and relative cell imbalance
**1.04604e-10**. Each CUDA driver retains one captured graph with 40 total
replays. CPU reports zero CUDA captures/replays as expected.
Live MF6 comparisons used the existing `/bin/modflow/mf6`, version **6.6.2
(05/12/2025)**; it was not installed or modified by the pip smoke setup.

Final suite logs: `/tmp/darcy-contact-expanded-suite-{cpu,cuda}.log`.
Other logs: `/tmp/darcy-contact-{baseline,fixed,suite-cuda,final-requested-cuda,final-regressions,final-suite-cpu,additional,additional-affected}.log`,
`/tmp/darcy-runtime-{baseline,fixed}.log`,
`/tmp/darcy-clean-smoke-{cpu,cuda}.log`,
`/tmp/darcy-unsupported-clean-entry.log`.

## Lateral-offset extension: reproduced and corrected

The supplemental affine probe gave **-0.0289266666666674** instead of the
physical **-0.03** (east, C=0.2). The initial two-direction probe gave **30
failed, 42 passed** on both untouched e77c8fff and the first top/bottom fix.
The user then explicitly authorized correcting this additional finding.
Before the lateral implementation, the expanded four-direction regression
reproduced **96 failed, 96 passed**. The assertions were retained unchanged.

The reference's lateral vertical-gradient coefficient now includes the
physical-to-logical tangent conversion derived above. Existing device endpoint
refresh is already exact for these affine-in-wet-fraction rows. Zero-offset
lateral contacts retain their previous coefficients.

`test_lateral_offset_affine_physical_exchange` checks all four orientations,
ordinary/discharge-only contacts, dry/partial/saturated states, C=0/0.2/3/omitted,
CPU/CUDA coefficients, independent plane flux and sparse conservation.
`test_lateral_offset_session_update_reuses_graph` checks ordinary and
single-sided contacts on CPU/CUDA: affine physical flux remains 0.03 before
and after the centroid update; both final heads and budgets match a fresh
session; workspace and captured graph identities survive the public
`update_contacts()` numerical refresh, with increasing replay counts.
Together these final lateral regressions give **196 passed**.

The existing `physical_driver` fixture now tests centred and offset lateral
contacts. Candidate residuals, full rollback, handoff budgets and actual restart
accounting pass for both geometries. The lateral/device-Picard targeted run
produced **231 passed**.

Original diagnostic logs are `/tmp/darcy-contact-lateral-{baseline,audit}.log`;
the expanded pre-fix log is `/tmp/darcy-lateral-expanded-baseline.log`, and
final lateral tests are `/tmp/darcy-lateral-final-regressions.log`.
No failed assertion was weakened, skipped or reclassified to pass.

The separate 64x64 low-K stress case was not rerun. Its contacts have neither
these tangential offsets nor top/bottom overrides, so these corrections do not
establish a fix for that convergence failure. No new convergence ladder,
full-repository run or performance conclusion is claimed.

## Files and checks

Source: `physics/terrain_reference_3d.py`, `solvers/terrain_device_kernels.py`,
`solvers/terrain_fast_3d.py`, `solvers/terrain_gmres_3d.py` under
`DARCY_WARP_PACKAGE/`. Existing test changes are
`tests/test_3d_terrain_reference.py` (the derived override expectation) and
`tests/test_terrain_device_picard.py` (centred/offset fixture coverage).
New scoped tests are `tests/test_terrain_contact_exchange.py` and
`tests/test_terrain_runtime.py`. The added lateral regression file is
`tests/test_terrain_lateral_contacts.py`. Setup/smoke changes are
`environment.yml`, `requirements-pip.txt`, `README.md`, the environment facts
in `AGENTS.md`, `working_tests/smoke_terrain_runtime.py` and
`.github/workflows/terrain-runtime-smoke.yml`, plus this report.

Executed checks:

```bash
RUFF=/home/patrickdurney/miniconda3/envs/darcywarp/bin/ruff
$RUFF check tests/test_terrain_contact_exchange.py tests/test_terrain_runtime.py tests/test_terrain_lateral_contacts.py working_tests/smoke_terrain_runtime.py DARCY_WARP_PACKAGE/solvers/terrain_gmres_3d.py
$RUFF format --check tests/test_terrain_contact_exchange.py tests/test_terrain_runtime.py tests/test_terrain_lateral_contacts.py working_tests/smoke_terrain_runtime.py
$RUFF check --select E9,F821,F822,F823 DARCY_WARP_PACKAGE/physics/terrain_reference_3d.py DARCY_WARP_PACKAGE/solvers/terrain_fast_3d.py DARCY_WARP_PACKAGE/solvers/terrain_device_kernels.py DARCY_WARP_PACKAGE/solvers/terrain_gmres_3d.py tests/test_3d_terrain_reference.py tests/test_terrain_device_picard.py tests/test_terrain_contact_exchange.py tests/test_terrain_runtime.py tests/test_terrain_lateral_contacts.py working_tests/smoke_terrain_runtime.py
$PYTHON -m compileall -q DARCY_WARP_PACKAGE/physics/terrain_reference_3d.py DARCY_WARP_PACKAGE/solvers/terrain_fast_3d.py DARCY_WARP_PACKAGE/solvers/terrain_device_kernels.py DARCY_WARP_PACKAGE/solvers/terrain_gmres_3d.py tests/test_3d_terrain_reference.py tests/test_terrain_device_picard.py tests/test_terrain_contact_exchange.py tests/test_terrain_runtime.py tests/test_terrain_lateral_contacts.py working_tests/smoke_terrain_runtime.py
git diff --check
```

These pass. The CI YAML was also parsed with the existing base conda
`ruamel.yaml` installation; its Python 3.12 selection and constrained install
were verified. All new files pass full configured Ruff lint/format checks.
No new lambdas were introduced.

Full configured checks on the modified existing Python files were also
run with `ruff check --output-format=json` and `ruff format --check`, on the
baseline and current tree. They show **43 baseline / 42 current lint findings**,
with **no introduced rule/message findings**, and **6 files requiring
formatting** on both trees. One old import-order finding was resolved by the
focused GMRES import change. Broad legacy formatting was not applied. Logs:
`/tmp/darcy-contact-lint-{baseline,current}.json` and
`/tmp/darcy-contact-format-{baseline,current}.log`.
