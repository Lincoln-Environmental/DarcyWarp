# Local seven-point programme: bounded reference parity

Stages 0, 1 and 2 pass their declared local gates. Validation now rejects a balanced field with incorrect prescribed heads, handles inactive MF6 markers and empty subsets, and distinguishes native contact experiments. The confined and selected Newton unconfined Cartesian equations agree with an independent NumPy/SciPy implementation, Warp CPU, CUDA, and local MF6 6.6.2 on the executed controls. All **64 final returned fields** pass a fresh independent evaluation without advancing a solution. This establishes implementation and reference parity for the contracts below; it does not certify the groundwater model, establish general uniqueness, or resolve the historical terrain branches.

The legacy Cartesian default and experimental transformed terrain equations remain intact. No terrain cross terms were deleted to create the new mode. Historical evidence verification confirms **2,432 files unchanged**. There were no remote writes, pushes, merges, PRs, workflow changes, CI or hosted computations. Changes are on the separate local branch `implementation/mf662-sevenpoint` in a dedicated worktree. Main dev remains at the starting revision.

## Provenance and environment

Actual starting local branch was `dev`, HEAD `296ecd61663613eb1ccf8de7d029f079e9047f88`. There were no tracked differences; unrelated untracked work is recorded in `validation_artifacts/sevenpoint_v1/stage0/starting_worktree_status.txt` and preserved. Applicable root instructions and existing Cartesian, validation, MF6 and terrain implementations/reports were read before implementation. Stage0 and confined parity were committed before adding the unconfined law; a subsequent validation-only commit covers empty/inactive diagnostic presentation and immutable reassessments.

No environment was modified. The existing `darcywarp` interpreter is Python **3.14.7**, NumPy **2.3.5**, SciPy **1.16.3**, Warp **1.17.0**, FloPy **3.9.5**, pytest **9.1.1**. `DARCY_FLOAT=float64` is set before imports, and BLAS threads are set to one. The available Python3.12.14 environment instead has Warp1.11.0 and lacks FloPy; it was unsuitable for the existing GMRES implementation. Thus this is not a claim to have repeated the historical Python3.12/Warp1.17 combination.

Local hardware is an RTX4070TiSUPER (16GiB), NVIDIA driver595.91.07; Warp reports CUDA toolkit12.9/driver13.2. Full OS, `lscpu`, BLAS configuration and GPU activity snapshot are in `final_recheck_v2/returned_head_checks.json`. GPU timing was not the objective, and no performance claim is made.

Native executable invoked: `/bin/modflow/mf6` (resolved `/usr/bin/modflow/mf6`), version **6.6.2 05/12/2025**, SHA256 **`21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176`**. The adapter refuses a different binary. Its exact local build revision is unavailable. Official version-tagged source, rather than an inferred executable build SHA, supports the constitutive derivation. Version-pinned documentation is [NPF](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-npf.html) and [DIS](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-dis.html). Exact conductance, smoothing and tie behavior is source-confirmed and measured, not merely assumed from the option names.

## Stage gates and scope

| Gate | Independent SciPy CPU | Warp CPU | Local CUDA | Live MF6 6.6.2 | Outcome |
|---|---|---|---|---|---|
| 0: acceptance, markers, empty subsets, contact identities | pass | unchanged equations | unaffected regressions | live masks and both contact identities | complete |
| 1: confined supplied-head/operator/flux parity and roots | pass | pass | pass | 8 roots + 8 supplied-state models | complete for contract v1 |
| 2: matched unconfined supplied-head/derivative/flux parity and roots | pass | pass | pass | 7 small roots + 7 supplied-state models + flat catchment | complete for addendum v1 |
| General terrain/MF6 physical equivalence or branch selection | unresolved historical evidence retained | unchanged | unchanged | formulation differences remain | not claimed |
| 3: variable layer elevations, local GHB/DRN | deferred | deferred | deferred | not run | not implemented |
| 4: transient storage, surface coupling, SHANZ | deferred | deferred | deferred | not run | not implemented |

The two reference contracts are [confined Cartesian v1](NPF662_CARTESIAN_REFERENCE_CONTRACT_V1.md) and [selected Newton/NPF addendum v1](NPF662_UNCONFINED_ADDENDUM_V1.md). They define orthogonal, flat contiguous uniform-dz cells, scalar dx/dy (unequal allowed), unrotated physical diagonal positive K, CHD cell constraints, no-flow and unreduced volumetric WEL sources. Source on inactive cells is rejected. Original fixed-cell source remains recorded reservoir forcing; it is excluded from free equations/WEL and included in explicitly reported reservoir requirements. No source redistribution occurs.

Confined conductance is face area divided by the sum of half-cell resistances. Unconfined horizontal conductance is full-thickness harmonic conductance times the **higher-head donor's MF6 quadratic saturation**, epsilon1e-6; mixed confined/convertible cell types and MF6 user-index tie selection are explicit. Vertical conductance remains full-thickness with VARIABLECV/PERCHED off. There is no terrain floor. The addendum explains the apparent documentation/initial-head ambiguity through the selected NEWTON initialization source path and measures initial-guess invariance in the controls.

All final root comparisons require finite valid active heads, absolute CHD error <=1e-10m, head-equivalent RMS <=1e-6m AND (physical RMS <=1e-12m³/day OR cellwise residual L1/gross external throughput <=1e-8). Same-head conductance tolerance is 3e-13 relative without an absolute floor; face/residual tolerance is 3e-12 relative +2e-11m³/day; complete head tolerance is 1e-7m absolute. These were declared before the full control results. The tiny-throughput denominator remains FP64.tiny. Very small throughputs can have a large relative residual even for an acceptable absolute residual: the constant-head CPU case illustrates why the physical alternatives are OR, not AND. Signed domain balance is reported separately and never substitutes for cellwise conservation.

## Validation repair and historical reassessment

The [Stage0 report](SEVENPOINT_STAGE0_REASSESSMENT_V2.md) records five genuine initial test failures and the repairs. The exact balanced shifted field `12,13,14` is rejected with eight prescribed cells checked, 10m maximum boundary error, tolerance1e-10m, and zero free-cell residual. No boundary repair is performed before checking.

Raw inactive NaN/1e30 markers remain in evidence. A separately labelled inactive arithmetic placeholder prevents NaN contamination without implying a physical inactive solution; active invalid values still fail. Empty layers/diagnostic subsets have counts and null statistics. A live masked model verifies full-user-grid binary IA/JA/package numbering with empty inactive rows rather than presuming reduced indices are full indices.

Native contact identities are explicit. The original convertible-cell DRN rejection remains an expected invalid-model outcome; the deliberately confined contact control is separately named. The full historical terrain campaign was not rerun. Offline reassessment of360 accessible/recovered archived candidates finds all360 have valid prescribed heads,212 meet the DarcyWarp discrete gate,148 do not, and none are unreadable. All14 polished original GPU root candidates across seven historical cases remain valid solutions of their respective original boundary-value problems. Their branch differences and physical interpretation remain unresolved; native/coupled MF6 heads are not relabelled as same-equation solutions.

## Focused implementation and safeguards

`validation/npf662_reference.py` independently assembles physical faces, incidence, sparse residuals and endpoint Jacobians without importing Warp or using GPU coefficient arrays. An import-blocking regression actually performs a CPU nonlinear solve. Sparse direct Newton has its own globalization and records linear residuals/factorization failures. Each face contributes opposite incidence exactly once.

`solvers/npf662_kernels.py` implements device physical faces, principal head scaling, endpoint derivatives and matrix-free Jv. `solvers/npf662_newton.py` owns a persistent device workspace and a nonsymmetric restarted GMRES solve; it reuses the existing generic Krylov machinery, not terrain conductances. S=1/full-saturated principal diagonal is fixed and applied consistently to J and R. For R=Q-div(q), the implemented linear solve is D(div(q))*delta=R. Jacobian tests check coefficient and gradient derivatives, actual <=7-column support, fixed corrections, anisotropy, strong contrasts, active sets and MF6 tie conventions.

Newton, frozen Picard and the documented full-saturated safeguard share one per-outer Krylov allowance. Target/budget status, actual work, Jv, candidate/residual counts, relaxation, merits, fallback and acceptance are recorded. Candidate head/flux/RHS buffers do not overwrite accepted linearizations. Rejected and nonfinite trials, shared-budget exhaustion, cold reset, numerical reuse, topology rebuild and workspace close are tested. Only scalar reductions are downloaded in the nonlinear/Krylov loop; the head is downloaded for final independent evaluation.

The dry-source test exposed a genuine unusable zero-Jacobian-row/nonzero-RHS initial state. Before the safeguard both independent CPU and device methods failed, while native MF6 recovered. The implementation diagnoses that state and, after Newton/frozen Picard cannot produce acceptance, uses a full-saturated C0 solve **only as a direction**. All candidate laws retain zero floor and are fully recomputed. The after experiment converges to the same root on all backends. Balanced but ungrounded dry components remain rejected; no dry unknowns are removed or arbitrarily pinned.

Another real negative model had convertible CHD below its cell bottom: MF6 rejects it, whereas the initially permissive CPU validator accepted a balanced head. A common preflight now rejects that invalid model without modifying prescribed values. The rejected deck, original CPU result and classification are preserved. This is an input compatibility defect, not evidence of two accepted formulations disagreeing.

## Numerical evidence

All rows below separately satisfy native success (where applicable), independent conservation, declared formulation compatibility and the predeclared same-root head gate. Physical model validation remains unestablished. Complete checks and arrays are in the immutable per-case directories and final offline recheck.

Confined controls (m for head differences; m³/day for physical RMS):

| Case | CUDA–SciPy max head | MF6–SciPy max head | CUDA physical RMS | MF6 physical RMS |
|---|---:|---:|---:|---:|
| Constant | 5.33e-15 | 5.33e-15 | 0 | 0 |
| x gradient | 1.08e-13 | 1.24e-14 | 5.59e-13 | 1.58e-14 |
| y gradient | 1.42e-14 | 1.07e-14 | 3.92e-14 | 1.24e-14 |
| z gradient | 2.01e-13 | 3.55e-14 | 7.90e-14 | 1.84e-14 |
| Layered series resistance | 5.97e-10 | 1.10e-12 | 4.24e-13 | 8.27e-15 |
| Six-order heterogeneous K | 5.41e-12 | 1.23e-13 | 4.52e-11 | 4.06e-13 |
| Masked hole | 1.04e-11 | 1.24e-13 | 4.87e-11 | 5.05e-13 |
| Two grounded components | 2.13e-14 | 7.11e-15 | 9.70e-14 | 1.50e-14 |

All confined native supplied-head flux errors are <=2.85e-14m³/day. Source allocation error is zero. Arbitrary-head residual/operator comparisons, analytic complete solutions, anisotropy and unequal dx/dy pass. Ungrounded components with zero/positive/negative net source are rejected. A native success flag at insufficient linear accuracy initially failed independent acceptance in the low-K series case. One-factor reference numerical controls show that tolerances/CG/scaling alone were insufficient; fill100/drop0 resolves it. Failed listings/CSVs remain archived. No physical law or acceptance threshold was changed.

Matched unconfined controls:

| Case | CUDA outer / Krylov | CUDA–SciPy max head (m) | MF6–SciPy max head (m) | CUDA head RMS (m) | CUDA physical RMS (m³/day) | CUDA L1/gross |
|---|---:|---:|---:|---:|---:|---:|
| Scalar upstream | 1 / 24 | 0 | 0 | 7.40e-17 | 2.22e-16 | 4.90e-17 |
| Exactly dry horizontal source | 6 / 144 | 0 | 1.41e-11 | 2.42e-14 | 2.42e-14 | 1.21e-14 |
| Partial vertical column | 1 / 24 | 0 | 3.75e-12 | 0 | 0 | 0 |
| Heterogeneous wet start | 5 / 120 | 1.06e-9 | 1.08e-9 | 5.83e-12 | 1.24e-10 | 2.53e-9 |
| Heterogeneous dry start | 6 / 144 | 1.16e-11 | 4.09e-11 | 8.30e-13 | 2.96e-11 | 5.41e-10 |
| Heterogeneous mixed start | 6 / 144 | 5.05e-11 | 9.24e-12 | 5.81e-13 | 3.28e-11 | 4.75e-10 |
| Masked mixed start | 6 / 144 | 1.18e-11 | 2.05e-11 | 5.01e-14 | 3.00e-12 | 4.69e-11 |
| Flat catchment representation | 6 / 408 | 3.34e-9 | 2.16e-11 | 6.08e-12 | 7.09e-8 | 6.16e-11 |

Unconfined supplied-head native face errors are <=7.11e-15m³/day. Stored DATA-SAT agrees with the independent saturation law within FP64 rounding. The saved directional derivative sweep gives relative L2 error1.01e-12 at perturbation1e-4, followed by the expected roundoff growth at smaller steps. The1e-3 trial crosses upstream selections and is not treated as a smooth central-difference test. Separate threshold/tie one-sided tests pass.

The new scalar solution is **1.266666666667m**, derived piecewise in the addendum. The original harmonic-law scalar test and its three roots were not changed. Passing the new scalar control proves nothing about general root uniqueness. Wet/dry/mixed starts here agree within the declared root tolerance; no preferred solution was selected by majority vote.

The larger4x24x32 representation has2,728 free cells. It preserves the existing catchment builder's K, source, masks, CHD values and cold initial heads but explicitly replaces geometry with flat uniform-dz layers; it is **not** the original terrain case. Geometry mapping and original arrays are saved. CPU sparse Newton takes5 outers, device methods6. MF6 takes36 outer/70,002 inner iterations. Gross throughput is1.767768e6m³/day; CUDA signed residual is separately recorded. Its physical RMS7.09e-8 exceeds the absolute alternative but its relative cellwise imbalance6.16e-11 passes the unchanged alternative, together with head RMS. CPU and MF6 relative residuals are1.24e-13 and5.35e-13. The large gross coastal counterflow is retained, not cancelled before normalization.

The first catchment MF6 run stalled at a2.84e-14m head update under1e-14m DVCLOSE and was explicitly interrupted. It remains incomplete, not passed. The completed comparison uses an input-scaled numerical head tolerance2.13e-12m (32FP64 eps times active initial/prescribed head magnitude); STRICT RCLOSE remains1e-14m³/day and independent gates are unchanged. This is recorded reference iteration precision, not an operator change, warm start or tolerance relaxation of the acceptance criteria.

## Public API and reproducibility

Existing calls retain their legacy behavior. The explicit pinned formulation is opt-in:

```python
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

solver = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
    device="cuda:0", implementation="fast", formulation="mf6_6_6_2_npf")
solver.build_from_K_fields(kx_field=kx, ky_field=ky, kz_field=kz, active=active,
    bc_mask=fixed, bc_values=values, rhs=volumetric_source, initial_head=cold_head)
head, info = solver.solve(max_cycles=2000)  # confined
head, info = solver.solve(unconfined=True, top_elevation=flat_top,
    npf_cell_type=cell_types, max_outer=2000, max_cycles=2000, krylov_restart=24)
solver.close()
```

Each call starts from the supplied session initial field; reuse does not imply warm starting. Results identify geometry/formulation/contract/effective options and independent final checks. Unsupported terrain builds, arbitrary-face overrides, floor/storage flags, variable elevations, optional NPF laws or unsupported backend combinations raise useful errors. CHD/no-flow/WEL are the only supported packages in this mode. No general MF6 compatibility name or changed default is used.

Exact commands, logs, inputs, hashes, fields, failures, source snapshots and native files are indexed in [the evidence README](../validation_artifacts/sevenpoint_v1/README.md). All runs were local. Runner directories are immutable (`exist_ok=False`); choose a fresh output root inside the script's execution guard to repeat. Source hashes, rather than an uncommitted-tree HEAD alone, identify the measured implementation. The final CHD input guard was added after the successful unconfined campaign: measured bytes are preserved by their recorded hashes, final positive fields rechecked, and the guard tested in the final62-test run. Final staged diff checking also removed extra EOF blank lines; the measured source copy preserves those original bytes. Neither change alters any valid model's equations. Native iteration records are stored losslessly using the repository's verified archive/restore utilities, with individual byte hashes and no discarded records.

Observed tests (overlapping groups, not additive): Stage0 **37 passed,0 skipped**; confined **21 passed,0 skipped**; existing affected subset **292 passed,0 skipped**; broad affected 3D/fast/terrain/new-validation suite **1,370 passed,0 skipped,2 dependency deprecation warnings**; final focused suite after CHD guard **62 passed,0 skipped,2 warnings**. CPU, Warp CPU, CUDA and live MF6 were all exercised. The full unrelated 2D repository suite and historical terrain campaign were not run. No unavailable backend was counted as a pass. Syntax/AST checks and `git diff --check` are also recorded.

## Remaining limits and next stages

This is a steady, flat orthogonal Cartesian contract with bounded validated settings, not terrain physical-model validation or a performance promotion. No MF2005/NWT run was needed or claimed. The exact MF6 executable build revision and historical Python3.12/Warp1.17 runtime reproduction remain unavailable. Ill-conditioned or ungrounded nonlinear states may still fail honestly; the implementation does not claim global uniqueness or universal recovery from every dry guess.

Stage3 should derive the selected MODFLOW-style variable-thickness connection approximation, then add local GHB/DRN laws as separate changes. It must preserve boundary locations/source ownership and distinguish that layered approximation from transformed terrain geometry. Stage4 should add version-matched storage, surface exchange and SHANZ integration with timestep-by-timestep accounting before any regional performance campaign. Multipoint terrain, rotated tensors, new transmissivity laws, branch/stability theory and Richards physics remain separate research. The historical terrain evidence is preserved for that work, not silently superseded by a simpler formulation.
