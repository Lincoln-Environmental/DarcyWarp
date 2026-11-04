# NPF/STO CPU parity over twenty weekly steps

The same 20 saved physical cases used by the steady CPU/MF6 comparison are
advanced for **20 steps of seven days**, ending at day 140. Every endpoint is
compared with live MODFLOW 6.6.2. The final numerical summary and individual
case results accompany this document.

**PASS: all 400 endpoint comparisons passed.**

| Worst metric across all cases and weeks | Measured |
| --- | ---: |
| Head RMSE (m) | 3.06533e-08 |
| Maximum head difference (m) | 7.73754e-08 |
| Relative face-flow L2 difference | 1.65460e-07 |
| Storage-flow L1 difference / throughput | 1.55336e-08 |
| DarcyWarp cellwise relative imbalance | 9.82234e-09 |
| MF6 native cellwise relative imbalance | 5.30753e-10 |

The focused CPU regression selection passed 109 tests, with five environment-dependent checks skipped and CUDA cases excluded. The new transient tests all passed, including the storage-anchored isolated cell. A final independent re-audit of both saved histories passed all 800 individual engine endpoints. All 60 case-input/native-history/Warp-history NPZ files passed ZIP CRC validation.

## Inputs and comparison

All original geometry, K, active/fixed masks, boundary values, source spatial
patterns, convertible types and cold initial heads are preserved. Their
canonical case hashes must match the original steady evidence. The nine
scaling cases use native RCH; the eleven layered controls use native WEL,
retaining the original volumetric source allocation.

Storage parameters, absent from the steady cases, are explicitly assigned:

| Parameter | Value |
| --- | --- |
| Specific storage Ss | 1e-5 per metre |
| Specific yield Sy | 0.15 in convertible cells |
| Timestep | 7 days |
| Timesteps per case | 20 |
| Final time | 140 days |

The original source is multiplied by 1.0 in weeks 1 to 5, 1.5 in weeks 6 to 10,
0.25 in weeks 11 to 15, and 1.0 in weeks 16 to 20. Boundary heads remain fixed.
Each engine advances its own previous head; MF6 heads are never supplied as
DarcyWarp's previous state or nonlinear starting guess. Previous states are
fixed throughout each nonlinear solve and checked for mutation.

Acceptance at every endpoint requires both engines' independent physical
cellwise balance, head-equivalent residual and boundary checks to pass.
Additional comparison gates are head RMSE at most 0.001 m, maximum head
difference at most 0.005 m, relative face-flow L2 difference at most 1e-6,
and storage-flow L1 difference divided by gross throughput at most 1e-6.
The physical cellwise L1 imbalance gate remains 1e-8. Signed domain budget
cancellation cannot substitute for the per-cell gate.

Native HEAD/CBC/GRB outputs are decoded at all 20 weekly endpoints. Source
allocation and native cellwise balance are checked, and actual STO-SS and
STO-SY output is compared with the independently evaluated storage change.
Both complete head/face-flow/storage histories are retained.

## NPF transient extension

The previously tested explicit `mf6_6_6_2_npf` formulation was steady-only.
It now accepts an opt-in implicit transient step using:

```python
head, info = model.solve(
    transient=True,
    dt=7.0,
    head_prev=previous_head,
    ss=1e-5,
    sy=0.15,
    unconfined=True,
    npf_cell_type=cell_types,
    npf_preconditioner='kcycle',
    npf_kcycle_controls=NPF662KCycleControls(use_cuda_graph=False),
)
```

The current step's source is set with `update_rhs`; its initial iterate can
be set with `update_initial_head`. `head_prev` denotes the immutable physical
previous timestep, independently of the current initial iterate. Steady
calls clear storage buffers, preventing state leakage after transient use.
The existing non-NPF 2D/3D/terrain backends retain their original dispatch.

Storage matches default MF6 6.6.2 `SsTerms` and `SyTerms`, including Newton
quadratic saturation smoothing at convertible-cell tops/bottoms. With T the
cell thickness, s(h) its smoothed saturated fraction and A the plan area,
the positive accumulation rate is the difference over dt of
`A * [Ss*T*s*(h-bottom-T*s/2) + Sy*T*s]`. Confined cells use s=1 and Sy=0.
Stable factored differences retain small elastic increments alongside larger
unchanged yield storage. The production Warp kernel and independent NumPy
reference evaluate separate implementations of this law and its derivative.

The storage tangent enters the physical residual, Newton Jacobian, diagonal
preconditioner, zero-row safeguards and independent final acceptance. The
K-cycle preconditioner includes a frozen previous-state storage tangent.
Its cache identity includes storage values; changing dt or capacity refreshes
the hierarchy. Storage can anchor a transient component without a CHD.
These additions do not introduce a saturation conductance floor.

This scope uses the same NPF and STO convertible types, fixed Ss/Sy within
each step, backward Euler and the default STO law. Time-varying storage
parameters with integrated TVS changes, STORAGECOEFFICIENT, SS_CONFINED_ONLY,
legacy specific-storage options, XT3D and variable-CV are outside this contract.
The new path is numerically validated on CPU here; CUDA execution/performance
has not been measured in this environment.

Primary formulation references are MF6 6.6.2
[GwfStorageUtils.f90](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/ModelUtilities/GwfStorageUtils.f90),
[gwf-sto.f90](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/GroundWaterFlow/gwf-sto.f90)
and [SmoothingFunctions.f90](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Utilities/SmoothingFunctions.f90).

## Reproduction

Extract the preceding steady evidence archive so that `--steady-evidence`
contains `validated_16_32`, `validated_50`, `layered_confined` and
`layered_unconfined`. From the repository root, in the darcywarp environment:

```bash
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 python working_tests/validate_npf662_transient_cpu.py --steady-evidence /path/to/steady_evidence --output /path/to/new_transient_evidence --mf6 /path/to/mf6 --mf6-sha256 9acdb07cae589555355b5d25b5ec1a7af7a0127ff95339ee14971ec7df6641ab
```

The hash above identifies the official Linux MF6 6.6.2 release used for this
test. The original workstation build pin remains the default. Both the exact
version and the explicit hash must match. A new output directory is required.
`--case` selects named cases; defaults run all 20. The runner checkpoints all
period metrics, retains failures, saves source snapshots and exits nonzero
if any case fails. `--ss` and `--sy` are explicit optional storage overrides.

The tests include an analytic confined transient root, repeated-step ownership,
dt-dependent hierarchy invalidation, transient-to-steady buffer reset, storage
anchoring of an isolated cell, and finite-difference checks of the complete
device Jacobian in dry, partially saturated, saturated and smoothing regimes.
