# Layered DIS implementation and local validation

Implemented the two acceptance repairs, confined layered geometry, and its existing matched unconfined law in that order. Starting local dev was **8352de9c423471619c5914ce1d028120935b7f30**, tracked clean with unrelated untracked work preserved. The separate local branch is `implementation/npf662-layered-dis`; main dev is unchanged. Acceptance milestone9f3ddea precedes confined milestone01f4121. No remote writes, merges, CI, workflow changes, environment upgrades or K-cycle development occurred.

The [acceptance report](NPF662_ACCEPTANCE_REPAIR_V1.md) and [versioned layered contract](NPF662_LAYERED_DIS_REFERENCE_CONTRACT_V2.md) give the exact policies and equations. Defaults remain CHD absolute error1e-10m, head-equivalent RMS1e-6m, AND(physical RMS1e-12m³/day OR L1/gross1e-8), with FP64.tiny throughput semantics. Confined caller physical/relative overrides are now honoured independently and reported; Krylov tolerances remain separate. Nonfinite operands/aggregates reject before serialization. Unconfined V1 final thresholds remain fixed; its iteration controls do not change acceptance.

Independent CPU assembly and production faces use separate implementations of conventional same-k harmonic **owning-thickness** resistance. Per-cell saturation and derivatives use the owning bottom and thickness. The unchanged controller now receives device thickness arrays rather than a captured scalar dz. Six reciprocal face arrays, endpoint Jacobian, fixed corrections, trial isolation and shared Newton/Picard/saturated-safeguard budgets are preserved. The layered path uses the existing diagonal-preconditioned GMRES controller for both confined and convertible cases; the uniform confined legacy backend is unchanged. No terrain metrics, overlap clipping, rotations, extra spatial couplings, new boundary packages or storage were added.

New public builder: `build_from_layers`. Authoritative interfaces have shape(nz+1,ny,nx). Tops/bottoms/thickness/centres/volumes are owned by the session; session.dz=None. Solve-time scalar top overrides are errors. Recalling the builder closes all geometry-dependent buffers even for the same shape, truthfully returning workspace_reused=False. RHS/boundary updates refresh compatible buffers from the original cold state. Tests cover masks/topology, same-shape interface changes, uniform/layered reconstruction, close/rebuild and fresh-versus-reused equivalence. Existing V1 dataclass fields and input hashes remain unchanged; separate `NPF662LayeredCase`/schema2 and `load_npf662_case` support new inputs without reinterpreting old NPZ files.

Changed implementation files: `warped_darcy_3d.py`, `solvers/session_3d.py`, `solvers/npf662_{3d,newton,kernels}.py`, independent `validation/npf662_{reference,mf6,metrics}.py`, and new production `physics/npf662_layers.py`. Focused acceptance/layered regressions and local evidence drivers accompany them. Historical Cartesian/terrain evidence and production terrain arithmetic are intact.

Existing runtime: Python3.14.7, NumPy2.3.5, SciPy1.16.3, Warp1.17.0, FloPy3.9.5, FP64 set before imports. CPU and RTX4070TiSUPER CUDA were available; Warp reports toolkit12.9/driver13.2, NVIDIA595.91.07. No Python3.12/Warp1.17 reproduction is claimed. MF6 `/bin/modflow/mf6` resolves to `/usr/bin/modflow/mf6`, version6.6.2 05/12/2025, SHA256 `21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176`. Official source tag6.6.2 is commit7cb7958940186ecfdcd3ea9e6a563cc4609581c9, not asserted as the binary's exact build revision. Full runtime snapshots and source hashes are archived. No timing or regional performance claim is made.

The raw pre-fix acceptance examples reproduce the passing overflow and strict-caller false positive; initial assertions give28 failures/1 pass. Repaired Part A gives94 passes, zero skips. Stable large/subnormal and zero/empty controls pass; invalid active results still fail and inactive markers remain raw. Offline reassessment before geometry and again at closure preserves all64 authoritative V1 final classifications without advancing a head or modifying their evidence. A broader archive reassessment additionally evaluates210 accessible fields (171 nonverification,39 constitutive checks), with0 changed recorded classifications;94 have no recorded prior independent status, which is not inferred. One already-known invalid convertible-CHD definition remains rejected, one interrupted native run has no returned field, and no discovered NPZ is unreadable. Early confined V1 files omitting ICELLTYPE retain their12-field hashes and roundtrips.

| Local check | Outcome |
|---|---|
| Preimplementation confined/Newton MF6 supplied-head connection checks | 2 native models pass; max flux error2.85e-14m³/day |
| Confined layered focused group | 124 passed,0 skipped; geometry lifecycle follow-up30 passed |
| Unconfined derivative/active-set/safeguard group | 21 passed after constructing valid distinct convertible fixtures |
| Broad affected 3D/fast/terrain/NPF/validation regressions | Initial1,457; follow-up **1,461 passed,0 skipped**,2 Warp Python3.14 deprecation warnings |
| Final focused group, including schema/input-precision/connectivity/CPU-independence/subnormal regressions | **153 passed,0 skipped**, same2 warnings |
| Saved final-field reassessment | 52 new fields:51 accepted,1 cold CPU failure;64 V1 fields unchanged |

These test groups overlap and must not be added. The latest broad run preceded the last connectivity/schema follow-ups; the final focused run covers them. The final reference geometry views promote raw lower-precision interfaces to FP64 before differencing, matching production while preserving raw-input hashes; a cross-sign float32 test exercises the otherwise lost thickness increments. Native extraction now checks the complete expected adjacency set, including an interior inactive cell with no vertical pass-through. Independent layered CPU solving is exercised with Warp imports prohibited. Both local Warp devices and live native models are exercised. The unrelated full2D suite and historical terrain campaign were not run.

Backend counts below describe the **13 numerical cold cases**, separately from pytest: independent SciPy CPU12 accepted/1 failed; Warp CPU13 accepted; CUDA13 accepted; MF6 native13 successful and independently conserved. Another13 supplied-state native models and2 preimplementation constitutive models preserve their heads and pass equation checks. A constitutive all-CHD model is never counted as a free-equation solve. Invalid below-bottom convertible supplied states are rejected, not forced through MF6 by changing cell type.

| Cases | CPU cold | Warp CPU/CUDA | MF6 | Maximum CUDA–MF6 head difference |
|---|---|---|---|---:|
| 5 confined controls: uniform,x,y,independent,masked | pass | pass | pass | 3.44e-10m |
| 6 unconfined controls: wet/dry/mixed,masked,unequal-T dry-source starts | pass | pass | pass | 2.84e-9m |
| Actual varying geology4x4x4,52 free cells | pass(10 outers) | pass(11) | pass | 5.66e-10m |
| Additional varying geology4x6x8,174 free cells | **fail at200 outers** | accepted(15) | accepted | **1.091e-7m; fails declared1e-7m comparison** |

Across the12 fully matched cases, maximum CUDA–CPU head difference is2.35e-9m, CUDA–MF6 difference2.84e-9m, face-flow difference at independently converged heads1.56e-8m³/day, and boundary/gross-throughput difference1.22e-8m³/day. Those solved-head flux differences are distinct from same-head arithmetic parity. Across all supplied-state checks, maximum native face difference2.28e-13m³/day and cell-residual difference3.98e-13m³/day pass the original scale-aware tolerances. Realised native source allocation error is0; CHD errors are0. Complete norms, gross flows, signed imbalance and source/reservoir budgets remain in per-case evidence.

The saved independent derivative sweep has relative L2 error4.92e-13 at step1e-3m, increasing with roundoff at smaller steps. The same-state device probe gives flux error0, residual error1.12e-16m³/day, and Jv error2.85e-14 (for its saved direction) on CPU/CUDA. Actual Jacobians have at most7 spatial columns. Tests separately cover full/thin/dry and smoothing-buffer states, mixed ICELLTYPE, endpoint one-sided derivatives, and equal-head/different-bottom/thickness donor ties in x/y. Dry-source recovery uses the existing saturated direction while evaluating the unchanged zero-floor law.

Both geology runs use the exact existing synthetic builder interfaces,K,masks,coastal CHD,WEL sources and cold initial heads, without running its demonstration main. They are **separately identified contact-free DIS cases**: the three historical west terrain discharge contacts are omitted and those exterior faces are no-flow; transformed terrain conductance/floor laws are replaced by the stated pinned NPF law. Original contact records and the changes are archived. No claim of the identical original terrain problem or of fixed-domain refinement is made.

The additional6x8 failure is retained, not used to loosen any gate. Its CPU head is not a second accepted root: head RMS.256m, physical RMS.520m³/day and relative imbalance.601. Late CPU direct corrections have linear residual maxima around.94–.99m³/day and tiny relaxation, despite finite factorization output; no preconditioner change can be inferred from this direct-solve failure. Exact conditioning/globalization causality remains open. A verification-only CPU restart from a deterministic1e-3m perturbation of the device result converges in2 steps. One independently factored Newton polish changes the accepted CUDA field by1.091e-7m and agrees with MF6 within8.72e-12m, with relative imbalance7.06e-13. This supports ordinary residual-controlled near-root error in the CUDA/native difference, not evidence of distinct accepted branches. Neither diagnostic converts the original cold failure or failed head-comparison gate into a pass. The CPU reference controller is intentionally unchanged.

Public API example using the exact validated varying interfaces (set `DARCY_FLOAT=float64` in the environment first):

```python
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

if __name__ == "__main__":
    input_path = Path("validation_artifacts/npf662_layered_v1/geology_4x4/varying_geology/inputs.npz")
    device = "cuda:0"
    max_outer = 200
    krylov_budget = 2000
    case = load_npf662_case(path=input_path)
    nz, ny, nx = case.active.shape
    with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy,
            device=device, implementation="fast", formulation="mf6_6_6_2_npf") as solver:
        solver.build_from_layers(z_interfaces=case.z_interfaces,
            kx_field=case.kx, ky_field=case.ky, kz_field=case.kz,
            active=case.active, bc_mask=case.fixed, bc_values=case.values,
            rhs=case.source, initial_head=case.initial)
        head, info = solver.solve(unconfined=True, npf_cell_type=case.cell_types,
            max_outer=max_outer, max_cycles=krylov_budget)
```

Exact commands, failed assertions, immutable inputs/heads/budgets, iteration records, offline analysis and checksums are indexed in [the evidence README](../validation_artifacts/npf662_layered_v1/README.md). Native iteration tolerances are stored separately: head-change `max(1e-14,32*eps*max(active initial/prescribed magnitude))`, STRICT flow1e-14m³/day; independent gates remain unchanged. Large native logs use the existing lossless archive/restore mechanism. Syntax/no-lambda and staged/unstaged diff checks are recorded. All changes remain local.

Unsupported: XT3D/terrain metrics, variable horizontal spacing, rotated K, pass-through/pinch-outs, new contacts/packages, optional NPF laws, floors and storage. Only bounded conventional DIS reference compatibility is established. The larger CPU cold failure and unpolished head-comparison miss remain explicit limitations; no general physical-model validation, uniqueness or regional performance claim follows.

Technical handoff for the **next, separate K-cycle milestone**:

- Physical C0 is built by `physics.npf662_layers.build_layered_dis_faces`, independently checked by `validation.npf662_reference`. Session faces are FP64 `(xp,xm,yp,ym,zp,zm)`; workspace.c0 stores positive x/y/z slots and left/right own connectivity. Layered session.dz is deliberatelyNone.
- Workspace bottom/thickness/types determine saturation; accepted q/c and dl/dr encode the physical endpoint flux/Jacobian. `jacobian_action` applies D(div q) with zero fixed corrections, consistently row-scaled. It may be nonsymmetric; do not assume CG suitability.
- `scale` stores full-saturated principal P0=sum(C0); RHS is R/P0 and matvec is J/P0. A future approximate physical solve must account for this S on both sides. Current `diagonal_action` applies the inverse absolute scaled Jacobian diagonal and stays fixed within an ordinary GMRES solve.
- The session's `_npf662_workspace_holder` owns the device workspace; builder reconstruction closes it, ordinary value updates reload original cold heads,source,C0,bottom,T and scales. Trial buffers cannot alter accepted dl/dr or RHS. Preserve that ownership and rollback.
- Newton/Picard/saturated safeguards share exact per-outer Krylov budgets. Replacing the preconditioner must preserve counters, fixed-within-solve semantics or use a compatible flexible method. The constructor's legacy solver selector does not mean layered K-cycle integration exists: this implementation remains `device_newton_gmres` with a diagonal preconditioner.
