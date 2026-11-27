# USGS regional workspace expansion, November 2026

## Scope and execution

The 15 workspaces not covered by the Zell–Sanford qualification are listed in [`usgs_regional_results.csv`](../benchmark_results/usgs_remaining_20261127/usgs_regional_results.csv). Each was run with native MF6 6.6.2 from a copied workspace, retaining its archived physical inputs, initial heads and IMS settings. DarcyWarp imported the archived workspace and started from its original initial heads using the `dev` mixed-fast CUDA defaults. Native heads were used only for comparison.

Independent FloPy inputs, native `FLOW-JA-FACE`, cell budget records, package entries, CPU reference fields and production CUDA fields were audited. The accepted DarcyWarp state was compared with the original native result using the existing limits: 0.005 m maximum head error, 1e-4 m head RMSE, and the existing face/package-flow tolerances. These limits were not changed.

The 15 workspaces span 399,117–2,448,601 active cells. Every workspace in `USGS_examples/models.*/` has a one-layer DIS grid; this campaign therefore provides no evidence about genuinely multilayer generalized-Newton cases. The importer and solver were not changed for performance in this campaign.

## Results

Ten of the 15 DarcyWarp solves passed the original-head, native-equation and head/package comparison gates. Three additional DarcyWarp solves passed physical acceptance, but their heads narrowly missed the comparison limits against the original MF6 outputs. The remaining two did not pass DarcyWarp physical acceptance. Their status and timings are retained in the consolidated CSV and per-model JSON receipts.

### Original MF6 closure and tighter diagnostic references

The 0308, 0405 and 0406 original MF6 runs terminated normally under their archived IMS configuration. Those files specify the `COMPLEX` preset and iteration limits but leave inner and outer closure tolerances at MF6 defaults. At the saved native heads, the independent DarcyWarp physical evaluator found the following native cell residuals:

| Model | Original MF6 time (s) | Native residual RMS (m³/day) | Native maximum residual (m³/day) | DarcyWarp head RMSE vs original (m) | DarcyWarp maximum head error (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0308 | 42.97 | 7.25e-3 | 6.67 | 1.172e-4 | 5.454e-3 |
| 0405 | 29.93 | 1.41e-3 | 0.793 | 3.454e-4 | 9.107e-3 |
| 0406 | 35.41 | 8.30e-3 | 3.18 | 4.365e-4 | 9.425e-3 |

Separate copies of those three workspaces were run with `OUTER_DVCLOSE` and `INNER_DVCLOSE` set to 1e-10. No physical input changed, and the original native outputs were preserved. DarcyWarp passed the same head limits against each tighter MF6 result:

| Model | Tight MF6 time (s) | Head RMSE vs tight MF6 (m) | Maximum head error (m) |
| --- | ---: | ---: | ---: |
| 0308 | 266.85 | 1.19e-11 | 2.39e-9 |
| 0405 | 227.42 | 1.31e-6 | 4.81e-5 |
| 0406 | 262.82 | 1.45e-7 | 4.52e-6 |

This explains the three comparison misses: native MF6 accepted its archived stopping point, which was too loose for the requested cross-solver head limits. DarcyWarp's final physical residual RMS values were 1.30e-9, 1.74e-5 and 3.35e-6 m³/day respectively; CPU/GPU physical evaluations agreed. The tight-reference audits are in `tight_references/<model>/`.

### Zero equations and balanced nullspaces: 0403 and 0414

The original DarcyWarp runs for `0403_0404_0707_0709_0712` and `0414_0415` were rejected as `ungrounded_saturation_components`. The rejection was too broad: a free cell with an exactly zero equation was rejected based on its saturation state, and valid unforced floating components were treated as failed equations. Both workspaces now converge through the mixed-fast CUDA solver from their archived initial heads.

The exact-zero-row check now accepts a free row only when its diagonal, residual, source and boundary throughput are exactly zero (and, for transient problems, its storage contribution is zero). Saturation does not determine whether a zero equation is valid. For a multi-cell component, the solver confirms exact local balance and an actual constant-head null vector before accepting the component. It then retains the original initial-head mean as a datum. This shifts only an unconstrained nullspace; it adds no boundary, conductance or source. A transient storage term anchors head and prevents this treatment. The final physical residual is reevaluated after datum selection.

| Model | Accepted floating cells | CUDA solve | Independent physical acceptance | Tight MF6 head comparison |
| --- | ---: | --- | --- | --- |
| 0403 | 1 saturated isolated cell `(0,935,962)` | 38 Newton, 187 Krylov; 2.90 s | Passed; relative cell imbalance 8.19e-10 | Passed raw: 1.06e-7 m RMSE, 1.81e-5 m maximum |
| 0414 | 24 components / 28 cells | 27 Newton, 152 Krylov; 3.39 s | Passed; relative cell imbalance 3.50e-11 | Passed after datum alignment; raw MF6 heads drift in nullspace components |

For 0414, the approximately 0.024 m head-equivalent residual at the rejected state came from three near-dry cells at `(0,579,2083)`, `(0,578,2087)` and `(0,578,2088)`. Their saturation was about 4e-17 and their diagonal derivatives were about 7e-15 m³/day/m. Dry projection snaps these cells to the bottom; it was previously rejected because the independent checker also rejected the other balanced floating components. After those components were handled, the existing dry projection was accepted, reducing the head-equivalent residual to 9.63e-9 m without changing the physical model.

The 0414 tight MF6 copy used `OUTER_DVCLOSE=INNER_DVCLOSE=1e-6` m and completed in 792.69 s. Independent input translation, native-head face-law, package and residual audits passed. The raw head comparison is 2.16e-4 m RMSE and 0.12135 m maximum because native MF6 drifted the datum in 24 unforced components (28 cells); the original starting-head datum was retained by DarcyWarp. Aligning MF6 by a constant per-component shift back to each original initial-head mean gives 2.50e-6 m RMSE and 0.00171 m maximum over all active cells, within the unchanged 1e-4 m / 0.005 m limits. On the 956,350 physically determined active cells, RMSE is 1.01e-8 m and maximum difference is 1.12e-6 m. The raw metric remains in the evidence and is not overwritten by the datum-aware metric.

At the tight 0414 reference, maximum face-flow difference is 0.00127 m³/day (RMS 1.52e-6); gross DRN, GHB and recharge budget comparisons pass. Absolute aggregate budget deltas are 7.72e-4 m³/day DRN outflow, 2.65e-4 m³/day GHB outflow and zero recharge. DarcyWarp physical acceptance reports 6.66e-7 m³/day residual RMS and 3.50e-11 relative cell imbalance. The detailed raw and datum-aligned fields are retained in `tight_references_1e6/0414_0415_MF6_SS_Unconfined_250/solution/`.

The component audit includes zero-based coordinates and original/native/rejected states under each model's `component_diagnostic.json`. The change is covered by the existing reference-equation tests, including a vertically connected multilayer nullspace and transient storage anchoring.

## Audit and benchmark changes

The benchmark audit now accepts the group 04 workspaces that have no CHD package, and it reads multiple packages of the same type. This mattered for workspaces with two GHB packages: native CBC writes separate `GHB` records whose `node2` indices restart for each package. The audit retains package order and remaps those local indices before comparing individual exchanges. Original archived files are not edited.

The combined Matplotlib panel stalled while rendering the large 0307 grid. The audit now writes one raster per spatial field and layer; the compressed cellwise map arrays remain available locally alongside the PNGs. `benchmark_usgs_remaining.py` resumes from saved case receipts. `benchmark_usgs_tight_references.py` runs the separate tight diagnostic references. `diagnose_usgs_components.py` locates the cells behind the topology rejection.

The complete local native workspaces, HDS/CBC/GRB outputs, rejected arrays and full compressed map arrays are retained under `benchmark_results/usgs_remaining_20261127/` but are excluded from this commit because they total about 13 GiB. The commit includes the consolidated CSV, per-case JSON receipts, spatial PNGs and component summaries. Use the benchmark scripts to recreate native binary evidence from the archived inputs.

## Current limitations

The 0308/0405/0406 rows remain marked as failed against their original native outputs in the baseline results table, even though each passes against a tighter MF6 diagnostic reference. For 0414, raw native heads in mathematically undetermined cells do not pass the head gate; the datum-aligned comparison does. The 74 currently available USGS workspaces are all one layer, so this regional campaign does not qualify generalized multilayer behavior. The user-provided `USGS_examples` model inputs are local and are not included in the commit.
