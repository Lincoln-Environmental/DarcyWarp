# DarcyWarp 3D integration-readiness assessment

**Status: ready for scoped review as a CPU-validated 3D integration candidate
against `dev`; not a release approval or a clean wholesale merge to `main`.**
The remaining known test failure is an existing experimental 2D FAS/Picard
history issue, reproduced on the unchanged baseline. CUDA and GPU benchmark
work are deferred at the user's request because of device contention.
CPU correctness is assessed without added wall-clock cutoffs; existing solver
iteration budgets and physical tolerances are unchanged.

DarcyWarp developers: **Patrick Durney and Jaco Fourie**. Software ownership:
**Lincoln Agritech Ltd**. Project-owned code remains AGPL-3.0-or-later; the
adapted NVIDIA GMRES retains its Apache-2.0 notice and attribution.

## 1. Repository state and scope

- Starting checkout: `dev`, HEAD `f02f4d91ba70e681628c7ea43dbf004a5ccf997c`.
- Fetched `origin/dev`: `f02f4d91ba70e681628c7ea43dbf004a5ccf997c`, unchanged from
  the audited baseline. Findings were reproduced against that exact source.
- Fetched `origin/main`: `ca67fdcf5302fb3cdd5ae00383f87b77db34a8dc`; it was not
  integrated, checked out, modified or pushed.
- Feature branch: `fix/3d-release-preparation`, isolated in `.release-worktree`.
- Finishing audited code/test/evidence snapshot: `40ef932046ac0946772703c553f5154a6645b5c0`.
  This report is a subsequent documentation-only commit; the PR also contains it.

The original checkout had no tracked modifications and contained untracked
experiments/data. These were preserved, together with all existing branches.
Exact initial inventories are retained in
[status](../release_evidence/starting_status.txt),
[refs](../release_evidence/starting_refs.txt), and
[fetched remote refs](../release_evidence/fetched_refs.txt). A detached baseline
worktree was used for reproduction, without changing the development checkout.
No merge, rebase, cherry-pick, force-push, branch deletion or destructive cleanup
was performed. The root public `README.md` is unchanged.

## 2. Findings and dispositions

| Severity | Finding | Disposition |
| --- | --- | --- |
| P1 | Uniform confined NPF Newton/K-cycle dispatch silently ignored legacy tolerance and hierarchy controls. | Corrected. Unsupported controls raise with the supported `npf_newton_controls`/`npf_kcycle_controls` alternative. Explicit former default values are distinguished from omission, so contradictory settings cannot bypass validation. |
| P1 | Explicit diagonal/GMRES selection could fall through to the historical direct backend. | Corrected. Explicit diagonal selects GMRES; explicit K-cycle selects FGMRES. Omission preserves historical uniform-confined dispatch. Layered/unconfined default remains diagonal GMRES; K-cycle is opt-in. |
| P1 | Raw face updates left stale conductivity available for unconfined reconstruction. | Corrected. K-owned models reject independent face updates. `update_K_fields()` derives all faces from validated physical properties. Independent-face models remain confined; NPF requires K/layered construction. |
| P1 | Failed builders changed metadata or closed the existing session before replacement validation. | Corrected. Replacement validation/construction precedes installation; the previous session closes after installation. Tests inject constructor failures and reject malformed fields, masks, geometry and contacts while retaining the previous usable model. |
| P1 | Invalid NPF replacements could install unsupported independent faces, zero active K or inactive sources. | Corrected. NPF physical inputs are validated before replacement. Inactive-source updates preserve the prior forcing when rejected. |
| P1 | Fractional/nonfinite masks were narrowed or treated as Boolean before validation. | Corrected for public builders, sessions, low-level Cartesian/layered face construction and independent budgets. Boolean/exact numeric 0/1 remain supported. NPF cell types already had raw binary validation; that contract is retained. |
| P1 | Overflowed throughput could produce a zero relative imbalance and false acceptance. | Corrected. Physical fields, diagonals and aggregates must be finite before either cellwise criterion can pass. Invalid ratios are not reported as zero. |
| P1 | Residual squaring underflowed/overflowed at representable physical scales. | Corrected in independent budgets and final head-equivalent diagnostics using scaled RMS/sums. Dimensional RMS and gross-throughput-normalised cellwise L1 criteria remain unchanged. |
| P1 | Positive physical face conductance could underflow to an artificial impermeable connection. | Corrected. A positive connection must survive conversion to the selected precision; rejected replacement leaves the old model intact. Exactly zero physical K retains its separate supported legacy meaning. |
| P1 | CPU returned heads aliased reusable fast/NPF device buffers. | Corrected with owned result snapshots. Later solves cannot overwrite retained heads or corrupt caller-side histories/storage calculations. |
| P2 | Independent SciPy Newton line-searched finite corrections without checking achieved linear accuracy. | Corrected. Row/column equilibration, extended-precision residual evaluation and bounded iterative refinement enforce both forcing and componentwise backward-error criteria before line search. |
| P2 | Terrain multigrid work was inferred from Krylov/restart counts. | Corrected. CPU/CUDA-executed counters measure actual preconditioner/V-cycle calls, including graph replay. Restart blocks, accepted/rejected updates and Jv work have separate diagnostics. |
| P2 | Repeated public NPF solves reconstructed the independent sparse reference. | Corrected with exact dependency snapshots. K, geometry, masks, cell types and spacing define identity; source, boundary values and initial heads remain current. Independent final acceptance remains mandatory. |
| P2 | Empty-domain comparison formatted its numeric maximum as `None` before the strengthened physical gate. | The uncapped full run exposed this regression during the audit. Acceptance now precedes display formatting; the existing empty-domain test and the complete 3D rerun pass. No finite check was weakened. |
| P2 | The existing runtime smoke was the only practical 3D CI gate. | Added CPU contract, NPF, terrain and 2D groups, plus separate manually invoked CUDA and pinned-MODFLOW infrastructure suites. No added CPU wall-clock cutoffs remain. |
| Scientific limitation | Terrain supports distinct independently accepted discrete roots. | Preserved and freshly reproduced. No change to floors, constitutive averaging, boundaries, initial-head policy or physical tolerances was used to force a single branch or MF6 agreement. |
| Existing 2D defect | FAS/Picard multistep history comparison raises `wp.copy` on a non-array storage destination. | Reproduced on unchanged baseline; outside the 3D repair. It remains explicitly recorded rather than converted into a pass. |

The inspected dispatch contract is:

| Selection | Actual supported backend / controls |
| --- | --- |
| Legacy Cartesian | Classic K-cycle/Chebyshev; opt-in fast K-cycle. Physical K owns unconfined reconstruction. Independent faces are confined. |
| Uniform confined NPF, no explicit NPF method/preconditioner/control object | Historical direct session backend; `rel_tol`, `max_levels`, `min_coarse_n` are applied. Explicit Newton-only controls require selecting the NPF method. |
| Uniform confined NPF with explicit diagonal/GMRES or Newton controls | Fast/kcycle model required; diagonal GMRES with Newton iteration controls. Legacy hierarchy/tolerance aliases are rejected. |
| Layered or convertible NPF | Fast/kcycle model required; diagonal GMRES by default, explicit K-cycle FGMRES alternative. Layer interfaces are authoritative; convertible physical thresholds remain fixed. |
| NPF K-cycle selection | FGMRES required; hierarchy/sweeps/cycles are `NPF662KCycleControls`, with actual dimensions/counts reported. Contradictory GMRES and legacy aliases raise. |
| Transformed terrain | Explicit terrain builder and fast/kcycle model; full conservative GMRES operator with its separate principal-part hierarchy and contact/dry-cell law. NPF options do not change its formulation. |

`validation/npf662_metrics.py` was inspected and its existing stable reductions
were reused. The primary overflow defect was in missing caller-side aggregate
validation, not a reason to change physical acceptance thresholds.

## 3. Critical before/after evidence

| Correction | Baseline / historical result | Corrected result |
| --- | --- | --- |
| Public state/API regression set | Final baseline test run retained separately; many parameter cases fail. | Focused contract and NPF suites pass; see execution table below. |
| Mask conversion | `0.9`, `1.9`, `-0.1`, NaN or infinity could become usable masks. | Raw values raise before narrowing/thresholding. |
| Failed rebuild | Prior session closed or geometry/K metadata changed. | Same prior session and properties survive; subsequent numerical solve remains usable. |
| Property ownership | Updated faces and retained K could describe different models. | Incompatible updates raise; reused updated models agree with fresh models and repeated solves. |
| Extreme RMS | Squaring `1e-200`/`1e200` residuals lost scale or overflowed. | Scaled nonzero, finite RMS; opposing cell errors do not cancel into acceptance. |
| Terrain overflow | The public solver reported `converged=True` with finite heads, head residual <1e-6 m and infinite throughput. | The same state reports `converged=False` and invalid physical metrics. A direct public-API regression fails on baseline at the old `True` result. |
| Independent cold geology solve | Fresh baseline fails at 200 outers; physical RMS `0.520290069628705 m³/day`. | Same original cold inputs/200-outer limit: converged in 120 outers; physical RMS `3.8964562733450605e-13 m³/day`. |
| Retained result arrays | A later CPU solve overwrote earlier returned heads. | Owned snapshots remain unchanged after boundary/property changes and workspace reuse. |
| Terrain work accounting | Cycles inferred from restart arithmetic. | Reported cycles equal independently counted preconditioner calls in the CPU regression. |

The [public overflow reproduction](../release_evidence/public_terrain_overflow_before.log)
uses unchanged physical tolerances and a zero Krylov budget to inspect the
actual initial physical state. Its small head residual cannot override invalid
physical aggregates.

The independent linear controller rejects a deliberately finite, inaccurate
correction that the baseline actually accepted for line search. That behavioural
reproduction is retained in
[the baseline test](../release_evidence/reference_controller_before.log).
The original cold geology failure was also rerun on the unchanged baseline in
the same Python 3.12 environment, not inferred solely from archived prose.

## 4. Independent numerical comparisons

Eight fresh **SciPy / Warp CPU / pinned MODFLOW 6.6.2** cases cover constant and
x/y/z flow, strong vertical anisotropy/layer contrasts, heterogeneous K,
irregular active masks and disconnected grounded components.

| Quantity | Maximum across the eight fresh cases |
| --- | ---: |
| Warp–SciPy head difference | `4.231281991451397e-12 m` |
| MODFLOW–SciPy head difference | `4.462208380573429e-12 m` |
| Same-head native/reference face-flow error | `2.842170943040401e-14 m³/day` |
| Warp relative cellwise imbalance | `4.19537171884228e-10` |
| Warp physical cell RMS | `1.6950661438563779e-12 m³/day` |

The largest dimensional RMS exceeds `1e-12`; that case passes the unchanged
alternative cellwise relative criterion, together with its head/boundary gates.
The comparison does not replace cellwise conservation with signed domain balance.
Inputs, native input/output files, result arrays, errors and source hashes are
retained in [the comparison gate](../release_evidence/independent_confined_verified/gate.json).
Existing analytical, storage, transient conservation and dry-start tests provide
additional formulation-specific coverage; their actual execution is recorded
in JUnit, rather than relabelled as a new large validation campaign.

For the retained varying-geology `4×6×8` case:

- Original cold reference input SHA256:
  `d86f4c8ba1870c7fba46d20c9df92153ada4d6141b108eea289f85f95867ec09`.
- The improved cold solve uses the same initial field and 200-outer budget.
  It accepts 14 Newton and 106 existing full-saturated safeguard updates.
  The forcing criterion rejects 106 inaccurate Newton directions; a small
  backward error alone does not guarantee an adequate achieved residual on
  an ill-conditioned linearisation.
- Fresh Warp CPU is physically accepted, with physical RMS
  `2.70131755714104e-9 m³/day`, relative cell imbalance
  `5.1054279250620265e-9` and head-equivalent RMS
  `1.971956690049444e-10 m`. Its maximum difference from the archived native
  solution is `1.0925405291573043e-7 m`: the strict `1e-7 m` comparison still
  misses. That miss is retained.
- The archived original CUDA difference is `1.0910235204164564e-7 m`.
  Archived independent polishing reduced it to `8.711253940418828e-12 m`.
  Those are verification/polishing results, not a new cold start or changed
  production acceptance. The small original miss is not evidence of changed
  physics. New CUDA release validation is deferred at the user's request.

See [cold/reference/device evidence](../release_evidence/geology/results.json),
[fresh baseline failure](../release_evidence/geology/baseline_checks.json), and
[historical polishing provenance](../benchmark_results/npf662_polishing_evidence.json).

The two terrain branches were reproduced from separately perturbed archived
seeds using the existing independent CPU reference. Both converge with relative
cell imbalances `3.793231931861109e-13` and `3.3235085820226876e-13`, while their
maximum head separation remains **0.3771228345055704 m**. This is branch
verification, not a global cold-start or uniqueness claim. Both heads and
budgets are retained in [terrain-root evidence](../release_evidence/terrain_roots_verified/results.json).

## 5. Executed tests and runtime

Baseline runtime: Python **3.12.14**, Warp **1.17.0**, NumPy **2.5.2**, SciPy
**1.18.1**, pytest **9.1.1**, FloPy **3.12.0**, Linux, Warp CPU/FP64. The original
`darcywarp` environment had Python 3.14.7 despite the declared 3.12 pin; the
existing 3.12 environment was aligned to Warp 1.17.0 for this campaign. The
Warp pin and public main runtime were not upgraded. Solver imports and the
completed synthetic public-API example verify basic operation. A completely
new conda installation including optional GIS/Torch dependencies was not run.

Native executable: `/bin/modflow/mf6`, version `6.6.2 05/12/2025`, SHA256
`21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176`.
CUDA was available on an RTX 4070 Ti SUPER, but the user requested CPU-only
execution because of contention. The final campaign uses
`CUDA_VISIBLE_DEVICES=-1`; CUDA skips are not passes.

The uncapped full repository run completed. Both 2000²/3000² truth checks
and all three 500×500 replay checks passed naturally on CPU. Earlier cutoffs
are superseded execution history, not numerical failures or release criteria.

| Execution | Passed | Failed | Setup errors | Skipped | Warnings |
| --- | ---: | ---: | ---: | ---: | ---: |
| Uncapped full repository | 1395 | 2 | 0 | 465 | 188 |
| Final complete 3D selection after the empty-domain fix | 1157 | 0 | 0 | 426 | 0 |
| Final NPF/control-type recheck | 228 | 0 | 0 | 50 | 0 |
| Final expanded focused contracts/reference/validation | 100 | 0 | 0 | 0 | 0 |
| Empty-domain/terrain/reference follow-up | 126 | 0 | 0 | 0 | 0 |
| Local CI contract group, no added cutoff | 145 | 0 | 0 | 4 | 0 |
| Local CI NPF group, no added cutoff | 137 | 0 | 0 | 15 | 0 |
| Local CI terrain group, no added cutoff | 295 | 0 | 0 | 94 | 0 |
| Local CI 2D group, no added cutoff | 39 | 0 | 0 | 6 | 0 |

The full-run failures were:

1. `test_18_fas_vs_picard_complete_transient_histories`: existing 2D storage
   destination error; independently reproduced on baseline (one failure,
   four experimental-backend warnings).
2. `test_empty_free_layers_and_comparison_subsets`: an audit-introduced
   presentation-order regression. The strengthened gate encountered a display
   `None` instead of the numeric empty-domain maximum. The physical check now
   runs before formatting. The test and the entire 3D selection pass afterward.

The original full-run XML remains unchanged; it is not relabelled as green.
The complete 3D rerun follows that correction, and final integer-control/API
checks follow the last validation refinements. These runs overlap and must not
be added as unique tests. The differential 71-case baseline snapshot retained
39 failures and 32 passing controls; the final expanded focused set passes.

The broad initial 3D CPU run had 1,129 passes, 425 skips and one test-harness
failure from a function that unconditionally included CUDA after its CPU check.
That function is now parametrised by device, preserving the CPU check while
reporting unavailable CUDA separately. Its recheck passed. Later focused tests
cover the final API/builders and physical budget mask/range guards. Full-suite
and focused counts overlap and must not be added as unique tests.

The earlier capped full run recorded 1,382 passes, 465 skips, 3 failures,
3 setup errors and 135 warnings. Five red outcomes were wall-clock cutoffs;
those cutoffs were removed at the user's request. The remaining FAS/Picard
`wp.copy` defect was separately reproduced on baseline `f02f4d91`:
[baseline FAS result](../release_evidence/fas_baseline.log). No 2D solver body
was changed in this 3D maintenance work.

The [machine-readable index](../release_evidence/validation_summary.json) and
[raw execution records](../release_evidence/) retain pass/fail/error/skip/warning
counts separately. Experimental FAS and coarse-geometry warnings are not counted
as test passes or silently suppressed. The final source hashes and
[artifact checksums](../release_evidence/SHA256SUMS.json) make the evidence
independently checkable. Late validation-only/API refinements receive focused
rechecks; each JUnit file describes the tests actually executed at that stage.

## 6. Architecture, ownership and accounting

Repeated RHS, initial-head and boundary-value solves reuse compatible buffers.
Physical K/geometry/topology changes refresh or invalidate their dependent
operators, hierarchies and reference snapshots. Owned K inputs and returned CPU
heads prevent contradictory provenance and history aliasing. Closing a session
releases its holders and breaks Krylov/operator ownership cycles.

The independent NPF sparse operator cache uses owned exact-value dependency
snapshots, not pointer identity or an unchecked revision counter. Current
forcing, boundary values and initial heads are supplied to every solve;
independent final physical acceptance is always evaluated. The retained
`250×250×5` inputs were used to check exact matrix identity between fresh and
cached construction. Earlier CPU setup observations are retained in
[the scoped setup record](../release_evidence/reference_cache_profile.json).
They are not a complete-solve/GPU speed claim or a CPU readiness condition.
No further speed testing or CPU execution deadline is part of this campaign.
The original full GPU benchmark remains archived and was not repeated under
contention or replaced by an unequal CPU timing comparison.

Terrain preconditioner/V-cycle counts execute on CPU/CUDA inside the actual
operator calls and captured graph replay. Restart blocks are counted separately;
K-cycle FGMRES has its existing application/cycle counters. NPF Jv products,
nonlinear updates and scalar readbacks are separately reported. Setup, iteration
and public-call times have explicit boundaries; the public total includes final
independent checks. GPU event, enqueue and synchronisation intervals may overlap
and must not be summed as independent costs. CPU timings are diagnostic only.
Legacy top-level cycle counts are not claims about every recursive/coarsest
operation. CUDA graph accounting and resource-release behaviour still require
the deferred dedicated CUDA integration run.

## 7. File-level changes

| Files | Reason |
| --- | --- |
| `physics/inputs_3d.py` | Shared raw binary-mask, floating-field and integer-control validation. |
| `warped_darcy_3d.py` | Transactional builders, physical-property ownership/update API, NPF replacement guards, independent-face mode contract, public timing and source-unit documentation. |
| `solvers/session_3d.py` | Owned floating state, raw mask validation, validated atomic field/face updates and existing revision semantics. |
| `physics/faces_3d.py`, `physics/npf662_layers.py` | Validate original active masks in reciprocal and layered face construction. |
| `physics/terrain_grid_3d.py` | Reject ambiguous geometry specifications and empty terrain dimensions. |
| `solvers_3d.py` | Raw masks, stable final head RMS and rejection of unrepresentable positive physical connections. |
| `solvers/fast_3d.py` | Owned returned heads, stable final head RMS and actual hierarchy metadata at the all-prescribed exit. |
| `physics/budget_3d.py`, `physics/terrain_budget_3d.py` | Scaled reductions, finite physical-state/aggregate gates, raw budget masks and rejection of invalid storage. |
| `solvers/npf662_3d.py` | Explicit control dispatch, effective settings, exact-input independent-reference reuse and consistently resolved initial guesses. |
| `solvers/npf662_newton.py`, `solvers/npf662_kcycle.py` | Owned heads, upfront integer-control validation, actual work counters, update accounting and setup/iteration diagnostics; K-cycle mathematics/defaults retained. |
| `solvers/terrain_fast_3d.py` | Replacement/contact/geometry validation, common final acceptance and measured preconditioner/V-cycle work. |
| `solvers/terrain_gmres_3d.py`, `solvers/work_counters_3d.py` | Actual restart blocks and graph-executed counters; NVIDIA attribution retained. |
| `validation/npf662_reference.py` | Independent equilibrated linear solves, achieved forcing/backward error, refinement and rejection before line search. |
| `validation/terrain_cpu.py`, `validation/terrain_compare.py` | Independent terrain acceptance honours invalid aggregates and is evaluated before empty-domain display formatting. |
| `tests/test_3d_release_contract.py`, `tests/test_npf662_reference_controller.py` | Defect reproductions, supported/contradictory controls, property equivalence, rollback, masks, overflow/underflow, reference forcing and counting regressions. |
| `tests/test_fast_3d_kcycle.py`, `tests/test_npf662_layers.py`, `tests/conftest.py` | Exercise physical updates, separate CPU/CUDA cases, and explicitly hide/skip CUDA in CPU-only execution. |
| `.github/workflows/three-d-release-*.yml`, `working_tests/run_3d_release_cpu.py` | Practical CPU regressions and separate CUDA/MODFLOW infrastructure suites without added wall-clock cutoffs. |
| `working_tests/reproduce_3d_release_geology.py`, `working_tests/verify_3d_release_cpu.py` | Reproducible archived-input cold/device comparisons, native checks and both terrain branches. |
| `working_tests/profile_npf662_reference_cache.py`, `working_tests/summarize_3d_release_evidence.py` | Scoped historical setup observation and machine-readable evidence/source index. |
| `working_tests/validate_npf662_confined.py` | Report only the devices actually executed, rather than always printing CUDA. |
| `examples/verify_3d_series_flow.py` | Genuinely completed public-API series-resistance verification with saved numerical outputs. |
| `docs/3D_CAPABILITIES.md`, `docs/README_3D_DRAFT.md`, this report | Formulation-specific capability/limitation guide, future README draft and readiness assessment. |
| `DARCY_WARP_PACKAGE/README.md` | Replace the missing self-contained river-study example reference with the runnable synthetic example; label recovered execution notes as historical. |
| `.gitignore`, `release_evidence/` | Preserve local scratch/earlier runs without mixing them with the retained numerical/JUnit/native evidence. Raw native output whitespace is preserved for checksum integrity. |

## 8. Reproduction and infrastructure

Set the declared environment and run from the repository root:

```bash
export CUDA_VISIBLE_DEVICES=-1
export DARCYWARP_TEST_CPU_ONLY=1
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
python -m pytest -q tests --tb=short
python examples/verify_3d_series_flow.py
python working_tests/reproduce_3d_release_geology.py
python working_tests/verify_3d_release_cpu.py
```

The numerical drivers write fresh outputs under `validation_runs/3d_release/`;
the synthetic example writes `example_results/3d_series_flow/`. They were actually
executed with those defaults, so rerunning them does not overwrite the retained
report evidence. Select a new output directory in the execution guard for an
additional independent-comparison campaign; existing comparison directories are
not destructively cleaned. All adjustable script inputs are in the exact
`if __name__ == "__main__":` guard, without argparse or lambda expressions.

CPU CI groups run through `working_tests/run_3d_release_cpu.py` with
`DARCYWARP_RELEASE_GROUP=contract|npf|terrain|2d`. The dedicated integration
workflow is manual: `suite=cuda` requires a prepared Python 3.12/Warp 1.17.0
self-hosted runner labelled `gpu`; `suite=modflow` requires the pinned executable
and label `modflow662`. Do not unset CUDA hiding on a contended device merely
to turn skipped coverage green. No dedicated CUDA run has been claimed here.

A genuine completed `5×3×4` layered public NPF example has maximum head error
`7.105427357601002e-15 m` and face-flow error `1.554312234475219e-14 m³/day`
against an independent series-resistance calculation. Its actual inputs,
heads/fluxes and physical acceptance are retained in
[example results](../release_evidence/example/results.json). No comparison
figure or site model was invented.

## 9. Scientific limits and integration candidate

The governing equations remain separate: conventional matched NPF, legacy
Cartesian/quasi-3D, and experimental conservative transformed terrain. The
maintenance changes do not alter saturation floors, MF6 smoothing, K averaging,
vertical transmission, storage, contact laws, prescribed heads or physical
acceptance thresholds. NPF remains steady only. Legacy/terrain transient support
must not be advertised as NPF transient support. The legacy and terrain dry-cell
approximations are not Richards physics or complete deactivation/rewetting.

The candidate scope is the reviewed 3D API/state/acceptance corrections,
independent reference controller, CPU regressions and accurate documentation.
Terrain remains experimental, including its discrete branch nonuniqueness.
A useful future experiment is basin/continuation and mesh-refinement comparison
with fixed physics and independently checked budgets for every accepted branch;
changing constitutive laws or selecting initial heads to favour MF6 is outside
this maintenance task.

## 10. Remaining work before any main integration

1. Resolve or explicitly scope the existing experimental 2D FAS/Picard
   history failure in a separate 2D change before any wholesale branch integration.
   The CPU wall-clock cutoff outcomes have been cleared by the uncapped run.
2. Run the dedicated CUDA correctness, graph replay, buffer/property refresh and
   memory-release suite on uncontended infrastructure. Current CPU validation
   cannot certify CUDA execution or GPU memory behaviour.
3. Revisit the retained GPU benchmark only when the appropriate infrastructure
   is available and comparisons can retain identical inputs, physical criteria
   and timing boundaries. CPU speed is not an acceptance condition.
4. Make an explicit future integration decision about main's runtime pin and
   scope; private GMRES requires the tested Warp 1.17.0 baseline. Do not silently
   upgrade it or promote experimental terrain claims.
5. Review the feature branch/PR against `dev`. This work does not merge, publish,
   release or make a public 3D capability claim on `main`.
