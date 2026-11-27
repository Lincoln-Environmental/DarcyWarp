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

### Rejected ungrounded components

`0403_0404_0707_0709_0712` and `0414_0415` remain unresolved. DarcyWarp returned rejected states and did not write them as accepted heads. The default solve stopped with `ungrounded_saturation_components`; tolerances were not relaxed.

Here, a component is a set of active cells joined by positive current face conductances. It is called ungrounded when none of its cells has a fixed-head condition or a positive head-dependent boundary tangent from GHB, DRN or RIV. Such a component has no external head anchor. This check is distinct from global mass balance: an unanchored group can have zero net source and zero local residual while its absolute head remains undetermined.

The saved native and rejected states identify the specific groups:

| Model | Native-head groups | DarcyWarp rejected-state groups | Component structure |
| --- | ---: | ---: | --- |
| 0403 | 1 | 1 | One isolated active cell at zero-based `(layer=0, row=935, column=962)` |
| 0414 | 24 | 25 | 21 native / 22 rejected singletons, two pairs and one three-cell group |

All listed groups have zero recharge/source, no GHB or active drain tangent, and zero physical residual within the group. The 0403 singleton is saturated, so it is not covered by the existing exact-dry-zero-row exception. In 0414 the rejected state has physical residual RMS 1.14e-10 m³/day and relative imbalance 1.07e-13, but its head-equivalent residual RMS is about 0.024 m. The topology rejection is therefore not the only unresolved acceptance issue for 0414. Detailed coordinates, heads and component fields are in each `component_diagnostic.json` and `ungrounded_components.npz`; rejected head and residual arrays remain in `darcywarp_rejected.npz`.

These findings show that the affected regions are tiny, zero-forcing floating pieces of the active grid, rather than a broad region with incorrect boundary exchange. They do **not** yet establish whether to handle them as mathematically irrelevant zero equations or whether additional model/topology semantics are needed. No production acceptance change has been made.

## Audit and benchmark changes

The benchmark audit now accepts the group 04 workspaces that have no CHD package, and it reads multiple packages of the same type. This mattered for workspaces with two GHB packages: native CBC writes separate `GHB` records whose `node2` indices restart for each package. The audit retains package order and remaps those local indices before comparing individual exchanges. Original archived files are not edited.

The combined Matplotlib panel stalled while rendering the large 0307 grid. The audit now writes one raster per spatial field and layer; the compressed cellwise map arrays remain available locally alongside the PNGs. `benchmark_usgs_remaining.py` resumes from saved case receipts. `benchmark_usgs_tight_references.py` runs the separate tight diagnostic references. `diagnose_usgs_components.py` locates the cells behind the topology rejection.

The complete local native workspaces, HDS/CBC/GRB outputs, rejected arrays and full compressed map arrays are retained under `benchmark_results/usgs_remaining_20261127/` but are excluded from this commit because they total about 13 GiB. The commit includes the consolidated CSV, per-case JSON receipts, spatial PNGs and component summaries. Use the benchmark scripts to recreate native binary evidence from the archived inputs.

## Current limitations

The 0308/0405/0406 rows remain marked as failed against their original native outputs in the results table, even though each passes against a tighter MF6 diagnostic reference. 0403 and 0414 remain rejected by DarcyWarp and need a justified treatment of their zero-forcing ungrounded regions. All cases in this campaign are one layer, so generalized multilayer Newton behavior remains unqualified. The user-provided `USGS_examples` model inputs are local and were not added to the commit.
