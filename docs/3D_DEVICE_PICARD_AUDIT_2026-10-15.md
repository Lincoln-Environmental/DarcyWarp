# Device terrain Picard/GMRES numerical audit — 2026-10-15

Audited current `dev` at `947d20d` after fetching `origin/dev`. Validation used FP64, the `darcywarp` environment, Warp 1.17.0, and CUDA on an RTX 4070 Ti SUPER. This report concerns numerical convergence. The largest case has 36,864 active cells; these runs support no GPU performance conclusion.

The original variable-geology SHANZ ladder converges at every requested size with the original 0.01 m saturation floor and unchanged governing equations. The 400-outer budget is insufficient at 64 and 96; rerunning the same initial-value solves with a 2,000-outer cap reaches strict physical acceptance. Each size starts independently from its own initial head. `dx=dy=20 m`; the case family changes its horizontal extent with grid size, so this is a convergence ladder rather than a fixed-domain spatial-order study.

| Grid n x n x 4 | Outer iterations | Head residual RMS [m] | Physical residual RMS [m3/day] | Relative cell imbalance | Result |
|---|---:|---:|---:|---:|---|
| 16 | 152 | 2.343669e-08 | 7.798838e-09 | 8.845655e-09 | pass |
| 32 | 139 | 6.519509e-10 | 6.165028e-09 | 9.777210e-09 | pass |
| 64 | 859 | 7.320908e-11 | 9.789518e-09 | 9.355561e-09 | pass |
| 96 | 985 | 7.552065e-11 | 1.155845e-08 | 9.852489e-09 | pass |

Final acceptance is unchanged: finite heads, head-equivalent residual RMS `<=1e-6 m`, and either physical residual RMS `<=1e-12 m3/day` or cellwise residual L1 / gross physical throughput `<=1e-8`. The passing ladder uses the relative physical gate. Small accepted head changes never replace these residual checks. Inner adaptive targets are in head-equivalent RMS units; GMRES alone receives L2 tolerances after multiplication by `sqrt(N_free)`.

The audit corrected mixed RMS/L2 contraction ratios, restart counts mistaken for Krylov iterations, budget multiplication by the restart length, zero-work stopping loops, stale final inner norms, rejection of finite useful inexact corrections, and retry accounting. `max_cycles` is an exact Krylov budget shared by all blocks and retries in one outer iteration. A shortened final restart solves and updates only its active columns. One history row records each proposed outer update, including all linear work, retry events and rejected candidates.

Nonlinear acceptance now reconstructs the current and candidate physical balances from conservative face fluxes, individual contact fluxes, original forcing and exact stored-volume change. It avoids subtracting rounded storage-weighted RHS/head terms. It refreshes saturation-dependent contact conductances and metric couplings before per-subface gating. Candidate rejection restores coefficients, storage, boundary aggregation, row scaling, RHS and every multigrid level. Fixed/isolated rows have unit scale, isolated heads remain fixed at their accepted value, and RMS reductions count the actual free equations.

The previous infinite/stale best-residual threshold is removed. At accepted state `h_k`, define `M(h)=(R_head(h)/max(R_head(h_k),h_tol))^2 + (R_flow(h)/max(R_flow(h_k),q_tol))^2`. Normalizing scalars stay fixed during each search. The driver first seeks a finite candidate with `M(h)<M(h_k)`. If that search fails, it tries a finite, nonzero Picard recovery with `M(h)<=1.10^2*M(h_k)`. Recovery changes globalization only; final convergence gates stay unchanged. Both phases re-evaluate the full candidate state. If neither works, the remaining linear budget refines the best correction before another search.

This recovery matters: requiring separate descent of physical and head-equivalent norms stopped the 32 case at outer 20 with head RMS `4.882921e-3 m`, physical RMS `4.127076e-2 m3/day` and relative cell imbalance `6.679096e-2`. The corrected case passes at outer 139. The converged unchanged-physics ladder supersedes the earlier inference that saturation-floor physics necessarily prevents this case from converging.

The parallel-layer control uses the same tolerances and effort caps:

| Grid n x n x 4 | Outer iterations | Head residual RMS [m] | Physical residual RMS [m3/day] | Relative cell imbalance | Result |
|---|---:|---:|---:|---:|---|
| 16 | 160 | 8.239453e-09 | 5.302482e-09 | 9.810874e-09 | pass |
| 32 | 134 | 7.543832e-10 | 5.293188e-09 | 9.318861e-09 | pass |
| 64 | 1502 | 7.799640e-11 | 9.220922e-09 | 9.933955e-09 | pass |
| 96 | 1116 | 5.768252e-11 | 8.404867e-09 | 9.902676e-09 | pass |

An additional stress ladder sets basement Kx to `1e-4 m/day` (aquifer Kx remains `8 m/day`, an 80,000:1 contrast; Ky/Kx=0.6, Kz/Kx=0.15). Its final 2,000-outer results are:

| Grid n x n x 4 | Outer iterations | Head residual RMS [m] | Physical residual RMS [m3/day] | Relative cell imbalance | Result |
|---|---:|---:|---:|---:|---|
| 16 | 1311 | 4.962046e-10 | 6.684065e-09 | 9.774311e-09 | pass |
| 32 | 1955 | 3.784346e-10 | 1.542731e-08 | 9.804904e-09 | pass |
| 64 | 2000 | 3.944583e-07 | 6.202167e-05 | 6.008032e-05 | outer_budget |
| 96 | 1863 | 7.656447e-11 | 1.204910e-08 | 9.974421e-09 | pass |

64: stop `outer_budget`, last linear RMS `9.422770e-09 m` against target `3.969617e-08 m`, accepted dh RMS `8.076473e-07 m`, relaxation `1.093750e-02`.

Failures at an explicit outer cap remain failures; the report makes no claim that another constitutive law is required. Histories retain the evidence needed to investigate forcing, damping and conditioning.

Validation: **222 tests passed** across `test_terrain_device_picard.py`, `test_3d_terrain_gmres.py`, `test_3d_terrain_gpu.py`, `test_3d_terrain_reference.py`, `test_3d_scientific_audit.py` and `test_terrain_shanz_diagnostics.py`. Two dependency deprecation warnings came from Warp ctypes layout declarations. Regressions cover short/partial/reused restart graphs and independently recomputed L2 norms, exact budgets and retries, current-state first-step backtracking, bounded recovery, nonfinite/zero-work behavior, actual free-cell RMS, raw/clipped/accepted updates, isolated cells, host/device cache handoff, candidate contact refresh at dry/partial/full saturation, exact transient volume, full rollback and independent coarse-grid coarsening. The mandatory nonparallel exterior `h=z` and affine flux checks pass. A source increment below representable head precision correctly reports failure.

A host-controlled comparison at 16/32 reaches head RMS `2.515557e-7` / `9.165794e-7 m` after 400 outers but misses the physical gate with relative cell imbalance `4.174401e-8` / `1.359096e-6`. Passing the head gate alone is insufficient. Candidate device physical balances are checked directly against the independent CPU reference at identical states; final artifact summaries also record device-minus-host residual differences.

Reproduce with `DARCY_FLOAT=float64`, `PYTHONPATH=.:working_tests` and the `darcywarp` Python environment. The ladder harness keeps its adjustable run controls in its execution guard. Its defaults retain the original floor, Krylov restart, anisotropy, relaxation and final tolerances.

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
python working_tests/terrain_convergence_baseline.py --device-picard --sizes 16 32 64 96 --max-cycles 2000 --max-outer 400 --out docs/terrain_picard_audit_variable.json
python working_tests/terrain_convergence_baseline.py --device-picard --sizes 64 96 --max-cycles 2000 --max-outer 2000 --out docs/terrain_picard_audit_extended.json
python working_tests/terrain_convergence_baseline.py --device-picard --variant parallel --sizes 16 32 64 96 --max-cycles 2000 --max-outer 400 --out docs/terrain_picard_audit_parallel.json
python working_tests/terrain_convergence_baseline.py --device-picard --variant parallel --sizes 64 96 --max-cycles 2000 --max-outer 2000 --out docs/terrain_picard_audit_parallel_extended.json
python working_tests/terrain_convergence_baseline.py --device-picard --basement-k .0001 --sizes 16 32 64 96 --max-cycles 2000 --max-outer 2000 --out docs/terrain_picard_audit_low_k_extended.json
python -m pytest -q tests/test_terrain_device_picard.py tests/test_3d_terrain_gmres.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_reference.py tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py
```

Summaries are in `terrain_picard_audit_{variable,extended,parallel,parallel_extended,low_k,low_k_extended,host}.json`. Each links its `.history.json.gz` archive, which contains every outer history row and trial residuals for recovery, retry and final updates. Seconds are recorded only as run metadata. The physics, saturation-floor formulation and final tolerances are unchanged. No Newton or Richards method was introduced.
