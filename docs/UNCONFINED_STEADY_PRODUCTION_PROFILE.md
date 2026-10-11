# Fast unconfined steady production profile

## Purpose and scope

Use `DARCY_WARP_PACKAGE.solvers.steady_config.production_unconfined_steady_controls()`
for opt-in, FP64 GPU unconfined steady initial-condition generation with
DRN/RIV/coastal boundaries. It names the existing accelerated numerical method;
it does not replace public defaults or change the governing equation.

**Validation status (11 October 2026):** CPU/CUDA gated-flow components pass;
full Purau steady solves converge and close independent groundwater budgets.
The original channel-only solution leaves upward hydraulic potential outside
its outlets; this is not intrinsically invalid groundwater pressure or proof
of ponding. Adding finite Richards-profile soil seepage DRNs produces a
budget-accepted full-catchment solution and an independently inventoried
storm-zero pressure/retention checkpoint. See the
[soil-profile outlet extension](#soil-profile-seepage-extension-interpretation-correction).
The controlled channel-only benchmarks below retain their historical workflow
labels. A Richards transient or routed storm has not yet been validated from
the new state, and the regional model is not an exact 3D Richards equilibrium.

A steady solve has no storage term, timestep controller or secant-Sy update.
The regional Dupuit groundwater problem is an approximation used to prepare
heads, not the three-dimensional Richards equation or a transient simulation.

## Solver selection

Dispatch lives in `DARCY_WARP_PACKAGE/solvers/registry.py` and capability/default
selection in `DARCY_WARP_PACKAGE/solver_capabilities.py`.

* Ordinary unconfined flow defaults to `unconfined_picard_kcycle`; steady
  Picard supports `inner_implementation="fast"`.
* Gated DRN/RIV/coastal flow must explicitly request
  `unconfined_semismooth_newton_kcycle`. It evaluates the authoritative nonlinear
  residual and analytical Jv, uses restarted FGMRES, freezes a face-based
  K-cycle preconditioner and can solve a corrective piecewise-affine boundary
  model. Every accepted direction still passes the actual nonlinear line search.
* The profile disables Picard fallback. Native code also disables it for
  DRN/RIV/coastal models because the plain Picard fallback cannot retain these
  gated terms. Failure remains failure; do not accept an ungated solution.

The reference steady benchmark has an additional **ungated Picard warm solve
on a separate model** for DRN/RIV cases. This is only its initial iterate, not
its final equation. Do not copy that pre-solve into a flux-only Purau domain:
it removes the available outlets. The named profile instead uses native
fixed-transmissivity **gated active-set** preparation, retaining every real
source and boundary.

## Authoritative configurations

Read these sources rather than copying a transient control dictionary:

| Configuration | Exact source / function |
|---|---|
| Tuned steady benchmark | `working_tests/run_2d_unconfined_warp_vs_mf6.py`: `run_warp_unconfined()`, `solve1_kwargs`, executable main block |
| Named steady profile | `DARCY_WARP_PACKAGE/solvers/steady_config.py`: `production_unconfined_steady_controls()` |
| Transient public controls | `DARCY_WARP_PACKAGE/solvers/transient_config.py`: `default_solve_controls()`, `production_secant_sy_settings()` |
| Transient executable choices | `working_tests/run_2d_transient_warp_replay.py`; `working_tests/transient_replay_settings.py` |
| Transient validation history | `UNCONFINED_FAST_PLAN.md` |
| Effective Newton controls | `DARCY_WARP_PACKAGE/solvers/semismooth_newton.py`: `solve_semismooth_newton()` |
| Fixed preconditioner | `DARCY_WARP_PACKAGE/solvers/kcycle_preconditioner.py` |
| True linear acceptance | `DARCY_WARP_PACKAGE/solvers/fgmres.py`: `RestartedFGMRES2D.solve()` |
| Face kernels and graph execution | `DARCY_WARP_PACKAGE/solvers/face_transient_f64.py` |

### Supplied versus consumed settings

“Native” means inherited by the reference Newton runner; the named profile
makes the same controls explicit unless indicated below.

| Control | Steady reference supply | Effective steady Newton / named profile | Production transient Picard |
|---|---|---|---|
| Backend | gated cases explicitly select Newton | `unconfined_semismooth_newton_kcycle` | `unconfined_picard_kcycle` |
| `max_cycles` | 80 | **ignored**; fixed preconditioner uses 2 cycles | 200 ceiling, adaptive inner blocks |
| `max_levels` / `min_coarse_cells` | 5 / 500 | consumed; named default 5 / 500 | 4 / 500 |
| `rel_tol`, `abs_tol_min` | 5e-7 / 5e-7 | consumed by **confined startup**, not outer Newton FGMRES | inner/Picard controls |
| `max_outer_iterations` | 60 | confined active-set startup budget, not Newton budget | 100, with adaptive strict substep budget 20 |
| `newton_max_iterations` | 100 for gated cases | 100, shared between outer Newton and corrective gated work | not a Picard control |
| Nonlinear residual RMS | native | 1e-6 volume/time per free cell | distinct strict/practical Picard checks |
| Head-equivalent RMS | `hclose=1e-4` | 1e-4 m (explicit profile) | `hclose=1e-4`, strict head residual 1e-6 |
| FGMRES restart / maximum | native | 20 / 60 per linear solve | not the production Picard controller |
| FGMRES relative tolerance | native adaptive | eta initial .1, min 1e-7, max .5, power 1 | adaptive K-cycle block targets, not Newton forcing |
| FGMRES absolute tolerance | native | 1e-10 | separate inner convergence |
| Forcing safeguards | native | previous backtracks >3 tightens; final residual/tolerance ratio <=100 uses eta floor | separate adaptive block controller |
| K-cycles per preconditioner | native | 2, no internal acceptance or scalar read | adaptive 5–20-cycle blocks |
| Pre/post/coarse work | native | 2 / 2 / 30 | 1 / 1 / 1 baseline |
| Smoother | Chebyshev | Chebyshev | Chebyshev |
| Spectral bounds | .1 / 2.0 | .1 / 2.0 | .1 / 2.0 |
| `omega` | .7 | recorded .7; weighted-Jacobi option, Chebyshev uses its spectral sequence | .7 |
| Coarse restriction | native | `average`; fast path rejects `conservative_sum` | production face-path convention |
| Startup | confined pre-solve | native gated fixed-T active-set solve | transient-specific startup/accepted prior state |
| Maximum head change | 10 m legacy name | 10 m explicit Newton guard | transient outer update guard |
| Line search | native | alpha=1, halves; max 10 backtracks; minimum 2^-10; Armijo 1e-4 | Picard/controller-specific |
| Corrective gated model | native enabled | enabled; used only when native safeguard requests it | cannot substitute an ungated Picard solve |
| Flow thickness floor | .1 m | .1 m, actual physical dryness diagnosed separately | .1 m plus exact transient storage |
| Precision | FP64 environment | FP64 operator, vectors, FGMRES, face correction; no FP32 correction | FP64 outer, optional FP32 inner |
| CUDA graphs | fast implementation | keyed fixed-cycle captures, actual launch counters | refresh and inner-block graphs |
| Picard fallback | native request default true | explicitly false; also forcibly disabled for gated boundaries | own validated period driver |

Other steady-reference controls ignored by this Newton backend include
`dh_rms_tol`, `residual_floor_tol`, `inner_forcing_eta`,
`inner_head_residual_tol_min/max`, `chebyshev_reset_factor`,
`transmissivity_relaxation_enabled`, `omega_min/max`, `chebyshev_enabled/order`,
`chebyshev_rejection_factor` and `check_every_no`. They are Picard controls.
`initial_saturated_thickness` only constructs an iterate when `initial_head`
is absent; Purau supplies H=DEM. `hclose` and the legacy maximum-head-change
name provide Newton defaults only if explicit Newton counterparts are absent.

Native FGMRES accepts only when the recomputed full-operator L2 residual is
`<= max(atol, requested_rtol * initial_rhs_norm)`; the Arnoldi estimate alone
is insufficient. Inspect per-solve requests in `linear_history` and
`confined_startup.confined_history`. Adaptive early tolerances are not a
claim of fixed 1e-6 convergence. Final groundwater acceptance independently
requires **both** true nonlinear and diagonal-scaled head-equivalent RMS.

### Reusable accelerated components

Steady Newton reuses the face-array kernels, in-place hierarchy coefficient
refresh, fixed-work K-cycle, keyed CUDA graph capture and model-owned buffers.
Its “storage diagonal” buffer holds frozen gated-boundary derivatives in a
steady solve; there is no invented transient storage. Transient-specific
secant-Sy, exact storage/time terms, adaptive timesteps, practical Picard
fallback and FP32 corrections are **not** imported into this profile.
The historical 1000×1000 heterogeneous-T, 30-period transient result of 29.0 s
is documented in `UNCONFINED_FAST_PLAN.md`; it is not a measured Purau steady
time and was not rerun for this task.

## Recommended fast steady configuration

The API is opt-in and returns a fresh dictionary:

```python
from DARCY_WARP_PACKAGE.solvers.steady_config import production_unconfined_steady_controls


def solve_existing_model(*, model, initial_head, reference_t, bottom, top):
    controls = production_unconfined_steady_controls(max_levels=5)
    head, info = model.solve(initial_head=initial_head,
        reference_transmissivity_field=reference_t, zbot_field=bottom,
        ztop_field=top, **controls)
    assert info["solver_backend"] == "unconfined_semismooth_newton_kcycle"
    assert info["preconditioner_implementation"] == "fast"
    if not info["converged"] or info["newton_fallback_used"]:
        raise RuntimeError(info["newton_failure_reason"])
    return head, info
```

Alternatively supply `K_field` instead of `reference_transmissivity_field`;
never both. Reference T is converted to K=T/(top-bottom); actual transmissivity
remains head-dependent, capped at full physical column thickness and floored
at the explicit numerical minimum. Build the model with the same physical
DRN/RIV/coast arrays before solving.

This **complete executable example** constructs the real Purau model from a
prepared steady-array archive, checks execution, independently sums recharge,
DRN and coast, requires represented surface outlets and rejects unreviewed flow-floor dependence, and exports only an
accepted head. The called implementation is committed at
`working_tests/run_steady_production_profile.py` (not pseudocode). Run from the
DarcyWarp repository:

```python
from pathlib import Path
from working_tests.run_steady_production_profile import solve_and_export


if __name__ == "__main__":
    input_archive = Path("/home/patrickdurney/PycharmProjects/SHANZ/results/darcy_hot_start/fast_confined_connected_300_20261011/recharge_300_conductance_1/steady_state.npz")
    output_archive = Path("accepted_regional_head.npz")
    recharge_mm_year = 300.0
    days_per_year = 365.25
    cell_size_m = 5.0
    max_levels = 5
    device = "cuda:0"
    budget_relative = 1e-6
    budget_absolute = 1e-6
    head_tolerance_m = 1e-6
    metadata = solve_and_export(input_archive=input_archive,
        output_archive=output_archive, recharge_mm_year=recharge_mm_year,
        days_per_year=days_per_year, cell_size_m=cell_size_m,
        max_levels=max_levels, device=device, budget_relative=budget_relative,
        budget_absolute=budget_absolute, head_tolerance_m=head_tolerance_m)
    print(metadata)
```

Use `DARCY_FLOAT=float64` before importing the model. On the current channel-only
Purau inputs, this example deliberately **raises at the physical gate** rather
than exporting a head without its intended soil-surface discharge mechanism. The array archive supplies geometry
and physical terms, not a trusted restart head. For other models including
CHD/GHB/RIV, extend the independent external budget accordingly; the example
is specifically the Purau DRN/coastal case.

Runtime `info["effective_configuration"]` reports local effective variables,
actual hierarchy, preconditioner controls, forcing, startup and acceptance
thresholds; `info["source_identity"]` records resolved imported paths, Git SHA,
tracked dirty state and exact source-file hashes. These are obtained once
after the solve, not in GPU iterations. `kcycle_graph_execution` counts real
captures/launches/eager cycles. Cached graph entries alone are not proof of
execution. Optional `newton_profile_timing=True` resolves CUDA events after
the solve; CPU submission time differs from GPU elapsed time.

## Performance validation

Hardware: NVIDIA GeForce RTX 4070 Ti SUPER, 16,376 MiB reported by NVIDIA;
Conda interpreter `/home/patrickdurney/miniconda3/envs/shanz/bin/python`.
Native source began at `5ec3252a2593223e78bafe3f36ce8f3d22ff17a2`, branch
`feature/shanz-surface-coupling`; SHANZ began at
`2d68cfbdc998c4306dee12e0af8bf8d2e644be10`, branch
`feature/surface-groundwater-coupling`. New-profile runs additionally record
exact working source hashes in `native.source_identity.file_sha256`.
Unrelated research files were preserved.

### Equivalent frozen Purau linear system

Historical same-system evidence on committed DarcyWarp `7929945` / SHANZ
`1f6e5e6`:
`SHANZ/results/darcy_hot_start/fast_kcycle_frozen_20261011_v2/report.json`.
Physical H=DEM, T, gated diagonal, RHS and hierarchy are fingerprinted once.
The initial top-clipped derivative vanishes; the benchmark used the frozen
Picard action. Subsequent analytical-Jacobian evidence is recorded below.

| Measurement | Classic | Fast |
|---|---:|---:|
| Warm two-cycle application | 131.69 ms | 31.24 ms |
| Residual ratio after application | .4442188484 | .4442188484 |
| FGMRES iterations | 38 | 38 |
| Warm complete solve including refresh | 5.50 s | 1.60 s |
| True relative residual | 4.41175e-7 | 4.41175e-7 |
| GPU process snapshot | 1330 MiB | 2614 MiB |

All six independent true-residual checks passed at requested relative/absolute
5e-7. Fast converged-solve time is about 3.43× better, not merely faster kernels.
First fast application .133 s; first FGMRES solve 3.515 s; repeats 1.603/1.605 s.
Modules were already disk-cached: these are first-call/graph warm-up costs,
**not uncached CUDA compilation benchmarks**. Sequential memory snapshots
include allocator pools and prior buffers; they are not isolated peak allocations.

### Complete named-profile hierarchy comparison

Output: `SHANZ/results/darcy_hot_start/production_profile_levels_20261011/`.
Every repeat starts at the identical original H=DEM. No steady hot-head reuse
is presented as startup speedup. Five levels are the reference; seven requested
levels produce only six under the unchanged minimum coarse size, so testing
both caps six and seven would duplicate the same topology.

| Requested cap | Actual levels | First native call | Warm repeat | Newton | FGMRES | K-cycles | Native / budget / physical |
|---|---:|---:|---:|---:|---:|---:|---|
| 5 | 5 | 30.610 s | 29.674 s | 15 | 904 | 1808 | pass / pass / fail |
| 6 | 6 | 40.535 s | 38.719 s | 15 | 910 | 1820 | pass / pass / fail |

Five-level hierarchy: `1310×766 → 655×383 → 328×192 → 164×96 → 82×48`.
Six adds `41×24`. The reference five-level configuration is 23.4% faster on
warm repeat; maximum head difference versus six levels is 4.54e-9 m. Six-level
startup took 16.764 / 14.962 s with eight active-set iterations /351 FGMRES;
five used eight /337. The seven-level cap reaches the same six-level hierarchy
at minimum coarse size500 and therefore was not redundantly benchmarked.
Input-array loading .119 s; model build .372 s for five levels, .128 s for six
in the already warmed process; head export .120–.122 s. Process memory snapshots
1944 /2524 MiB include allocator pools. CUDA modules were disk-cached and no
uncached compilation claim is made. Five-level confined startup took 12.020 s first, 11.045 s
warm. Both five-level runs used five corrective gated iterations and 43 line
search backtracks. True nonlinear RMS was 8.646e-10 volume/day; independent
signed budget residuals -3.2613e-8 / -3.0776e-8 m³/day (allowance .013556).
The first call captured 20 graphs and launched 1808 cycles, with zero eager
fallbacks; the warm repeat captured zero and launched the same 1808 cycles.
Measured GPU K-cycle work was 21.213 / 20.331 s: linear work dominates.

Original direct H=DEM startup (without confined preparation) exhausted the
unchanged 100 combined Newton/gated work budget in 42.66 s: 83 accepted Newton
updates +17 gated work, 1005 FGMRES, nonlinear RMS3.55745 and 4.05% budget error.
The prepared solve succeeds without increased limits or relaxed tolerances.
Continuation was not needed and was not implemented.

### Full analytical Jacobian at an intermediate Purau state

Output: `SHANZ/results/darcy_hot_start/production_profile_analytical_linear_20261011/report.json`.
Both variants apply the same native analytical Jv at
H=.5*DEM+.5*saved converged diagnostic head, with the same sources, masks,
gated boundaries, RHS, six-level topology, FP64 device and relative/absolute
5e-7. This interpolation is a linear diagnostic state, never an accepted restart.

| Measurement | Classic | Fast |
|---|---:|---:|
| First coefficient setup (cached modules) | .000634 s | .008181 s |
| First two-cycle application | .140093 s | .130277 s |
| Warm two-cycle application | .137817 s | .031332 s |
| Residual ratio after application | .3906461186 | .3906461186 |
| FGMRES / K-cycles per solve | 32 /64 | 32 /64 |
| First complete solve including refresh | 4.7151 s | 3.2537 s |
| Warm complete solve including refresh | 4.6955 s | 1.3337 s |
| True relative residual | 3.63094e-7 | 3.63094e-7 |

All four independent true-residual checks passed. Correction fields differ
by at most8.81e-13 m. Warm converged linear solve is3.52× faster; warm application
throughput is4.40× faster. Fast first-call capture/warm-up overhead is measured
by the first-vs-warm difference, not mislabelled as cold compilation. New-profile
nonlinear measurements confirm real graph execution. The hierarchy comparison
isolates the steady numerical setting; neither test changes physical parameters.

## Scientific limitations

The preserved Purau physical experiment has 660,178 active cells, 5×5 m
resolution, area16,504,450 m², reference T50 m²/day, regional base -50 m AMSL,
net recharge300 mm/year on a365.25-day year, 2566 connected DRN cells and89
selected coastal faces (445 m). Original finite DRN conductance is2.5 m²/day:
.1 m/day bed K ×1 m width ×5 m bed length /.2 m resistance. No conductance,
mask, rainfall, material, physical budget or coastline was changed in the
hierarchy comparison. Sea head is absolute0 m; submerged subfaces exchange
both ways and exposed subfaces are outward-only, using the prepared regional
hydrostatic boundary bands.

Recharge is groundwater input, not atmospheric storm rainfall. Independent
steady acceptance is `R + sea_in = DRN_out + sea_out`, using native physical
areas and outflow gates, without invented correction water.

The five-level solution has R13556.0164271, DRN12050.2371774, coastalout1505.77924975,
coastalin0 m³/day. It has18241 non-drain heads above DEM, maximum9.504745 m;
these indicate upward gradients whose soil-surface discharge is not represented by the
selected channel-only outlet assumptions. Positive head above DEM is not an
intrinsically invalid unconfined/artesian state and is not surface-water depth. They must not be clipped into an accepted
hot start or silently replaced by all-cell drains. Boundary/geometry assumptions
require explicit physical review separately from numerical-profile validation.

629956 columns (95.422%) have regional heads below Richards' shallow5 m bottom.
This is **not** dryness below Darcy's -50 m regional bottom. There are zero
regional dry/floor-dependent columns in this case. The .1 m artificial flow
floor can carry nonphysical transmissivity in actually dry columns; report
its occurrence and flux dependence and reject unreviewed floor-dependent
hot states. Do not remove this validity gate to gain convergence.

DRNs obey C*max(H-zdrn,0), not prescribed heads. Their active-side derivative
at equality permits recharged startup; coastal exposed subfaces similarly
retain outward-only complementarity. Storm channels are not made permanently
fixed-head cells by initialisation.

A regional Dupuit head field maps to Richards pressure as H-actual cell elevation,
without clipping to the shallow bottom. Constitutive saturation/storage must
then be evaluated with native materials and coordinates. It is not an exact
steady 3D Richards solution and does not justify overwriting the full profile
with60% saturation. No Richards validation was launched from physically
rejected Purau heads in this task.

## Troubleshooting

* **Fast unexpectedly absent:** verify `solver_backend`,
  `preconditioner_implementation`, imported paths/source SHA and
  `newton_fallback_used`. `inner_implementation` is the Newton option;
  `implementation="fast"` alone is a different confined interface.
* **FGMRES stagnation:** inspect true residuals and per-linear requested
  tolerances, actual hierarchy and preconditioner cycles. Distinguish adaptive
  forcing from fixed tolerance. Compare the same frozen analytical Jacobian,
  state, RHS and gates; do not increase limits blindly or switch to CPU sparse
  methods as the default response.
* **Excessive Newton updates:** separate linear iterations from gated work and
  accepted head corrections/backtracking. H=DEM direct startup can be slow;
  retain gated confined preparation. A fast preconditioner does not guarantee
  fast nonlinear contraction.
* **Gated line search failure:** inspect `outer_history.gated_model`, branch
  changes and corrective-model true residuals. Candidate acceptance must use
  the full residual; no ungated Picard fallback or boundary omission.
* **Confined preparation failure:** inspect `confined_startup.confined_history`,
  true linear convergence, active-set budget and actual unconfined residual
  reduction. It only commits a converged fixed-T candidate which improves the
  actual nonlinear residual. A recharged closed component without any outlet
  has no steady state.
* **Budget failure:** independently sum plan-area recharge and signed external
  boundary fluxes, including DRN/RIV/GHB/CHD where present. Native residual RMS
  alone cannot replace the integrated physical budget.
* **Dry cells:** compare H against regional bottom, physical saturated thickness
  and numerical floor separately; a head below Richards bottom is permitted.
  Never artificially wet the shallow domain for a hot start.
* **Checkpoint incompatibility:** compare active/grid/vertical/material/coastal
  fingerprints. These diagnostic head archives are not accepted storm restart
  checkpoints and contain no accepted Richards momentum/controller state.
* **Hierarchy cap seemingly ignored:** cached topology is model-owned. Use a
  fresh model for comparisons changing max_levels; inspect actual shapes.

## Reproducibility

From `/home/patrickdurney/PycharmProjects/DarcyWarp`:

```bash
MPLBACKEND=Agg DARCY_FLOAT=float64 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/patrickdurney/miniconda3/envs/shanz/bin/python -m pytest -q \
 tests/test_unconfined_steady_profile.py tests/test_drain_threshold_startup.py \
 tests/test_transient_session.py tests/test_semismooth_newton_2d.py \
 tests/test_nonlinear_operator_2d.py tests/test_dynamic_riv_flux_2d.py \
 tests/test_shanz_drn_flux_2d.py tests/test_fast_confined_kcycle.py \
 tests/test_2d_transient.py tests/test_2d_transient_warm_start.py

DARCY_FLOAT=float64 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/patrickdurney/miniconda3/envs/shanz/bin/python \
 working_tests/benchmark_purau_steady_profile.py \
 --run-directory /home/patrickdurney/PycharmProjects/SHANZ/results/darcy_hot_start/fast_confined_connected_300_20261011/recharge_300_conductance_1 \
 --output-directory /home/patrickdurney/PycharmProjects/SHANZ/results/darcy_hot_start/production_profile_levels_20261011 \
 --max-levels 5 6 --repetitions 2
```

Output directories must be new; existing evidence is never overwritten. Source
array SHA256, effective settings, actual import paths, work, budgets and
individual diagnostic head paths are saved per repeat. Log:
`/tmp/purau_production_profile_levels.log`. CPU/CUDA targeted suite log:
`/tmp/darcy_steady_profile_final_tests.log`.
Final expanded suite:158 passed, six pre-existing small-grid coarsening warnings; no skips in that
selection. An initial new-test assertion used a fixed1e-6 domain budget rather
than the specified external-input-scaled allowance and failed two DRN cases;
it was corrected to the documented independent criterion without changing
solver tolerances or production acceptance. No broad scientific success claim
is inferred from the component suite.

## SHANZ integration guidance

`shanz/initialization/darcy_hot_start.py` can call the named profile as the
optional steady API, then retain its existing independent budget/physical
checks, material conversion, inventory and checkpoint provenance gates.
`scripts/run_purau_darcy_hot_start.py` currently obtains parity settings via
`shanz/initialization/darcy_parity.py`, which parses the reference runner's AST.

The minimal future integration is to request
`production_unconfined_steady_controls(max_levels=5)` explicitly, remove duplicate
`solver/formulation/minimum thickness/fallback` keywords at the call site, and
record the returned profile and native effective report. Retain AST parsing as
a reference-regression mode, not as the long-term production API. Numerical
profile tests and catchment convergence support this API; the unresolved
physical Purau hot-start gate prevents promoting the current scenario as a
validated production antecedent state. SHANZ production coupling and its
acceptance policy were not modified for this task.

Analytical linear benchmark, from the SHANZ repository:

```bash
SHANZ_SOLVER_ROOT=/tmp/hot-start-solvers DARCY_FLOAT=float64 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/home/patrickdurney/miniconda3/envs/shanz/bin/python \
 scripts/benchmark_purau_darcy_kcycle.py \
 --run-directory results/darcy_hot_start/fast_confined_connected_300_20261011/recharge_300_conductance_1 \
 --output-directory results/darcy_hot_start/production_profile_analytical_linear_20261011 \
 --multigrid-levels 6 --repeats 2 --applications 3 \
 --operator-mode analytical_jacobian --state-fraction 0.5
```

The input archive SHA256 is
`dbed03f599d373c3c7791e3c390c026b240dc3811cf84dd95e5e7f7231dfec45`.
Logs: `/tmp/purau_production_analytical_linear.log`. The native source path resolves
to this DarcyWarp checkout, not the older isolated worktree. Tested working
source fingerprints are saved even when Git HEAD predates the profile commit;
the final implementation/tests/documentation commit is recoverable with
`git log --all -- docs/UNCONFINED_STEADY_PRODUCTION_PROFILE.md`.

## Soil-profile seepage extension (interpretation correction)

The channel-only numerical solution above is a valid converged solution of its
stated groundwater equations. The earlier workflow's "physical failure" label
was too categorical: H>DEM represents an upward hydraulic potential. To model
the equivalent Richards soil discharge, SHANZ now offers
`--surface-seepage soil_profile --numerical-profile production`.

`shanz/initialization/darcy_seepage.py` adds potential non-channel surface DRNs
using full cell area and the actual Richards vertical saturated resistance:

`C = A / (0.2/Kv_FSL + 0.8/0.1 + 4.0/10)` in m²/day for the5 m profile.

The implementation obtains horizon depths/Kv from the saved Richards
configuration, not from an unrelated drain tuning parameter. Existing selected
channel-bed conductances and coastal subfaces remain distinct and unchanged.
Native DRN gating activates surface discharge only where H>DEM; below DEM
there is no withdrawal or atmospheric inflow. Potential contacts at every
non-channel cell avoid a second host-side activation solver and allow newly
emerging discharge within the existing native semismooth active set.

These are finite-conductance outlets, not fixed heads or inferred standing
surface water. Regional head is interpreted as the hydraulic potential at
the base of the shallow Richards-equivalent vertical resistance. This remains
a reduced-order approximation, not an exact terrain-metric Richards steady
solution. Channel and soil surface discharge are separately budgeted; no
antecedent discharge is added again as storm forcing. The resulting saturated
positive-pressure initial columns must retain valid constitutive inventories.

SHANZ distinguishes valid upward pressure from **unrepresented surface
seepage contacts** in its diagnostics. Its hot-start export still requires
native convergence, an independent external budget, supported geometry and
explicit discharge ownership. Historical evidence files preserve the old
label verbatim; this section corrects its scientific interpretation.

The soil-profile extension subsequently converged on full Purau in39.719 s,
with16 Newton,1161 FGMRES and2322 executed fast K-cycles. Independent groundwater
budget residual is6.04e-9 m³/day; channel DRNs8144.97168, soil seepage4529.94141,
coastout881.10334 m³/day sum to recharge13556.01643.5606 soil contacts discharge;
zero regional flow-floor columns. A storm-zero Richards pressure/retention
checkpoint was generated and independently inventoried, with no Richards
advance. Sources: SHANZ7d90f2e, DarcyWarpd7ee792 (native code unchanged).
Outputs and exact reproduction are preserved in sibling SHANZ
`docs/DARCY_SOIL_SEEPAGE_2026-10-11.md` and
`results/darcy_hot_start/soil_profile_seepage_300_20261011/recharge_300_conductance_1`.
This changes outlet physics; do not claim an equivalent-physics speedup from
its comparison with the channel-only numerical benchmark above.

## Grid resolution, conservative input aggregation, groundwater parameter upscaling and head-field transfer to finer models

**Status (11 October 2026):** validated on the Purau steady channel-only case
at 20 m, 25 m and 50 m against the 5 m reference. All coarsening lives in
SHANZ scripts (`scripts/coarsen_purau_groundwater.py`,
`scripts/benchmark_coarse_darcy_steady.py`,
`scripts/map_coarse_head_to_richards_ic.py`); no solver code changes — the
solver only requires square cells and aligned `(ny, nx)` fields, so a
coarsened field archive with a coarser `dx` is a valid model.

### Upscaling rules (never plain averaging of everything)

- **Active mask:** any-coverage — a coarse cell is active iff it contains at
  least one fine active cell (preserves extent; active-area inflation at
  fringes is recorded: 1.0%/1.4%/3.2% at 20/25/50 m, and recharge is
  volume-corrected below).
- **DEM:** active-weighted mean per block (preserves mean terrain, not
  extrema).
- **Reference transmissivity:** active-weighted arithmetic mean per block —
  the correct rule for horizontal flow through parallel fine columns (exact
  for the spatially uniform Purau reference T).
- **Regional bottom:** active-weighted mean (uniform −50 m); validated below
  every coarse DEM cell.
- **Recharge:** per-block volume-preserving rate,
  `R_c = R_fine · n_fine · A_fine / A_coarse`, so every block's integrated
  recharge volume is exactly the fine volume (recharge is a per-area rate;
  a uniform rate over an any-coverage mask would inflate volume by the
  fringe area ratio).
- **DRN:** any-coverage drain mask; conductance is the **sum** of the fine
  per-cell conductances in the block (parallel bed conductances add),
  preserving total drain conductance exactly and total channel length to
  block-path accuracy; drain elevation is the conductance-weighted mean bed.
- **Coastal boundary:** the fine `CoastalSubfaces2D` records are aggregated
  by coarse owner cell (`owner//f`): submerged conductance is summed
  exactly; exposed conductance is summed exactly and re-banded over the
  coarse owner's exposed height with the same band rule. A face-mask
  reduction cannot represent a coastline shorter than a coarse fringe row
  (the 445 m Purau coast loses all but 7 of 89 faces at 20 m that way);
  owner remapping preserves the total conductance exactly at every
  resolution.

### Measured accuracy and performance (Purau, recharge 300 mm/yr, connected-channel DRN, conductance ×100, confined pre-solve startup, fast K-cycle, FP64)

| Resolution | Cells | Cold solve | Warm (head restart) | Newton | Head vs 5 m (bias, p50, p99, max) | DRN block rel-L2 | DRN:coastal partition |
|---|---|---|---|---|---|---|---|
| 5 m (reference, re-solved) | 660,178 | 80.2 s | 0.32 s | 19/0 | 0 | 0 | 0.912 |
| 20 m | 41,661 | 6.4 s (12.5×) | 0.05 s | 19/0 | −0.005, 0.089, 0.34, 1.13 m | 0.050 | 0.908 |
| 25 m | 26,781 | 5.9 s (13.7×) | 0.05 s | 19/0 | −0.087, 0.148, 0.53, 2.03 m | 0.059 | 0.906 |
| 50 m | 6,810 | 2.7 s (30×) | 0.04 s | 26/0 | −0.305, 0.300, 1.20, 2.41 m | 0.083 | 0.905 |

All resolutions: converged, no Picard fallback, executed fast K-cycle,
independent budget residual ≤ 3e-8 m³/day against a 1e-6 allowance, and the
same documented physical-gate caveat as the 5 m reference (non-drain heads
above DEM; the count falls with coarsening as extrema average out). A warm
head restart converges in zero Newton iterations at every resolution. The
head-mapping and checkpoint-write costs are ≤ 0.1 s and ≤ 0.5 s
respectively, i.e. negligible against solve time.

### Head-field transfer to finer models

`scripts/map_coarse_head_to_richards_ic.py`: inactive coarse cells are
nearest-active filled (any-coverage guarantees every fine active cell lies
in an active coarse cell), the coarse head is bilinearly sampled at fine
cell centres, and the mapped head is independently gated on the fine grid
(non-drain heads above DEM counted exactly as for the fine reference, none
below the regional bottom, Richards-domain water-table fraction reported)
before building the Richards initial pressure with the production soil
profile. The constitutive storage inventory is computed independently and
reported against the fine-reference inventory; the difference is physical
state, never a mass correction. At 20 m the mapped Richards IC differs from
the fine-reference IC by a −4.8 mm bias and 0.34 m p99 on head.

### Guidance and limits

- Use 20 m for steady initial-condition generation when ≤0.35 m p99 head
  error and ~5% DRN partition error are acceptable; treat 50 m as a
  screening option (bias −0.30 m).
- Keep the same `max_levels` cap across resolutions; inspect
  `effective_configuration.actual_level_shapes` — coarse grids bottom out
  earlier by `min_coarse_cells`, which is intended.
- Multigrid/CUDA-graph caches are model-instance local: cross-resolution
  reuse is impossible; only head warm starts transfer (and they are
  essentially free convergence, see the warm column).
- Do not compare coarse-run wall time against the 5 m reference as an
  equivalent-physics speedup: the coarse model solves an upscaled
  equation set. The accuracy rows above are the equivalence statement;
  budget closure alone is not.
