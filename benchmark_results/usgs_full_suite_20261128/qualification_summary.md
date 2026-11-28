# USGS 74-model CUDA mixed-fast requalification

## Run conditions

All 74 discovered workspaces were run natively with MODFLOW 6.6.2 from copied inputs using archived IMS settings, original package inputs, and original initial heads. The run used no more than 10 CPU cores. DarcyWarp used `dev` mixed-fast CUDA defaults on the RTX 4070 Ti SUPER, started from the original heads, and used the shared steady Newton limit of 240. No acceptance tolerance or physical input was changed. Original-settings native references were reused for this DarcyWarp requalification.

All 74 workspaces are single-layer; active cells range from 261,841 to 4,471,511, and the largest grid has 8,306,826 total cells. This suite does not qualify multilayer behavior.

The per-model CSV contains runtimes, setup times, sampled GPU memory, iteration counts, raw and datum-aligned head differences, face-flow differences, package budgets and physical balance. Per-model spatial maps and JSON receipts remain beside each case in `darcywarp_qualification/`.

![Runtime and head comparison](darcywarp_qualification/qualification_summary_plots.png)

## Results

**28/74 passed every unchanged physical and parity gate.** DarcyWarp reached and physically accepted a solution in 60 cases. The head-parity failures converged physically but miss comparison against the archived, comparatively loose native endpoint; they remain unqualified until checked against tighter MF6 endpoints.

| Result | Cases |
|---|---:|
| passed all gates | 28 |
| converged; head parity miss | 32 |
| native-head audit rejected | 2 |
| line search failed | 5 |
| 240-step limit | 5 |
| ungrounded component rejection | 2 |

Among fully accepted cases, median native runtime was 91.1 s, median DarcyWarp wall runtime 39.6 s, median numerical solve time 1.68 s, and median paired DarcyWarp/native wall-time ratio 0.422. Maximum sampled GPU use over all cases was 11634 MiB. Runtime comparisons are reported only for the fully accepted subset.

## Remaining findings

The five line-search / Newton-limit / ungrounded-component groups are genuine solver failures; no gate was relaxed. The two native-head audits reject individual DRN exchange differences at high-conductance entries by about one saved-head ULP, just above the unchanged per-entry absolute gate. The detailed native-head receipts retain the offending entries.

Prior tight-reference evidence records 23 head-comparison cases passing all existing gates against tighter MF6 endpoints; controlled 240-step comparisons also validated 1015 and 1308. Separate `dvclose=1e-6` runs are now in progress for 1025, 1103, 1109, 1205 and 1602. Case 1805 also reached a new endpoint and still needs tight-reference confirmation. Earlier tight attempts for 1109 and 1501 were manually stopped after poor MF6 contraction, so those head comparisons remain unresolved. No claim that every archived-reference mismatch is resolved is made here.

### DarcyWarp solver failures

- `1018_1019_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: line_search_failed
- `1102_1108_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: line_search_failed
- `1210_1211_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: outer_budget
- `1304_1307_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: ungrounded_saturation_components
- `1404_1405_1406_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: line_search_failed
- `1502_1506_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: outer_budget
- `1503_1507_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: line_search_failed
- `1504_1505_1508_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: outer_budget
- `1606_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: ungrounded_saturation_components
- `1801_1802_1808_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: line_search_failed
- `1807_1810_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: outer_budget
- `1809_MF6_SS_Unconfined_250`: RuntimeError: DarcyWarp failed at period 1, step 1: outer_budget

### Converged states missing archived MF6 head gates

- `0308_0309_0310_MF6_SS_Unconfined_250`: RMSE 0.0001172 m; maximum 0.005454 m.
- `0403_0404_0707_0709_0712_MF6_SS_Unconfined_250`: RMSE 0.0004362 m; maximum 0.01877 m.
- `0405_0409_0410_MF6_SS_Unconfined_250`: RMSE 0.0003454 m; maximum 0.009107 m.
- `0406_0407_0408_0416_MF6_SS_Unconfined_250`: RMSE 0.0004365 m; maximum 0.009425 m.
- `0414_0415_MF6_SS_Unconfined_250`: RMSE 0.0001028 m; maximum 0.008364 m.
- `0501_0502_0503_MF6_SS_Unconfined_250`: RMSE 0.0001178 m; maximum 0.005456 m.
- `0504_0506_0508_0509_MF6_SS_Unconfined_250`: RMSE 0.0001046 m; maximum 0.01052 m.
- `0511_0513_0514_MF6_SS_Unconfined_250`: RMSE 5.301e-05 m; maximum 0.007339 m.
- `0512_MF6_SS_Unconfined_250`: RMSE 0.0001604 m; maximum 0.008999 m.
- `0711_0713_0714_MF6_SS_Unconfined_250`: RMSE 1.938e-05 m; maximum 0.01625 m.
- `0804_0805_0808_MF6_SS_Unconfined_250`: RMSE 6.782e-05 m; maximum 0.009845 m.
- `0901_0902_0903_MF6_SS_Unconfined_250`: RMSE 0.0007013 m; maximum 0.03364 m.
- `0904_1002_1003_1004_1005_1007_MF6_SS_Unconfined_250`: RMSE 4.14e-05 m; maximum 0.008638 m.
- `1006_1010_1011_MF6_SS_Unconfined_250`: RMSE 0.001412 m; maximum 0.05233 m.
- `1008_1009_1012_MF6_SS_Unconfined_250`: RMSE 0.0007286 m; maximum 0.01894 m.
- `1015_1020_1021_1022_1023_MF6_SS_Unconfined_250`: RMSE 0.0002907 m; maximum 0.008269 m.
- `1016_1017_MF6_SS_Unconfined_250`: RMSE 0.0005343 m; maximum 0.01779 m.
- `1025_1026_1027_MF6_SS_Unconfined_250`: RMSE 0.001367 m; maximum 0.01915 m.
- `1103_1104_1105_1106_MF6_SS_Unconfined_250`: RMSE 0.0006844 m; maximum 0.01154 m.
- `1109_1110_1112_1113_MF6_SS_Unconfined_250`: RMSE 0.000109 m; maximum 0.002839 m.
- `1201_1202_1203_1204_MF6_SS_Unconfined_250`: RMSE 0.0005265 m; maximum 0.08299 m.
- `1205_1208_MF6_SS_Unconfined_250`: RMSE 0.0004397 m; maximum 0.004487 m.
- `1308_1309_MF6_SS_Unconfined_250`: RMSE 0.01665 m; maximum 0.162 m.
- `1403_MF6_SS_Unconfined_250`: RMSE 0.0008909 m; maximum 0.03783 m.
- `1408_MF6_SS_Unconfined_250`: RMSE 0.001836 m; maximum 0.0295 m.
- `1501_MF6_SS_Unconfined_250`: RMSE 0.0002924 m; maximum 0.006378 m.
- `1602_1603_MF6_SS_Unconfined_250`: RMSE 0.000171 m; maximum 0.006213 m.
- `1701_1702_1703_MF6_SS_Unconfined_250`: RMSE 0.0001003 m; maximum 0.00433 m.
- `1708_1711_1713_MF6_SS_Unconfined_250`: RMSE 0.0001197 m; maximum 0.0115 m.
- `1712_MF6_SS_Unconfined_250`: RMSE 0.000321 m; maximum 0.008816 m.
- `1803_1804_MF6_SS_Unconfined_250`: RMSE 0.0001117 m; maximum 0.03327 m.
- `1805_1806_MF6_SS_Unconfined_250`: RMSE 0.0007424 m; maximum 0.04827 m.

### Native-head audit rejections

- `1301_1302_MF6_SS_Unconfined_250`: RuntimeError: native_heads parity failed: DRN: native entry exchange; see /home/patrickdurney/PycharmProjects/DarcyWarp/benchmark_results/usgs_full_suite_20261128/darcywarp_qualification/1301_1302_MF6_SS_Unconfined_250/native_heads.json
- `1303_1305_1306_MF6_SS_Unconfined_250`: RuntimeError: native_heads parity failed: DRN: native entry exchange; see /home/patrickdurney/PycharmProjects/DarcyWarp/benchmark_results/usgs_full_suite_20261128/darcywarp_qualification/1303_1305_1306_MF6_SS_Unconfined_250/native_heads.json

Acceptance thresholds remain head maximum 0.005 m, head RMSE 1e-4 m, saturation thickness 0.1 m, flow absolute 1e-6 and relative 1e-10, and gross package-budget relative 1e-3.

See [`darcywarp_qualification/usgs_regional_results.csv`](darcywarp_qualification/usgs_regional_results.csv) for all 74 records.

## Targeted follow-up to `20d4381`

Saved receipts establish 55 distinct fully qualified workspaces, including
1801 from this follow-up. The 28 archived-reference passes and 27
tighter-reference passes are disjoint. Six additional DarcyWarp solutions
passed the physical checks and still need sufficiently accurate MF6
comparisons.

The two native DRN audits now pass with the cached MF6 outputs. The saved
rates match the expanded expression `C*z - C*h` exactly. The independent
factored expression differed by at most 1.4833e-6 m³/day in 1301 and
1.2065e-6 m³/day in 1303. The audit uses a per-entry bound on the rounding
of these operations. The newly unblocked DarcyWarp runs from original
heads took 70.8 and 78.7 s, respectively, and both ended with three
ungrounded saturation components.

The initial Newton directions for 1304 and 1606 reduced the physical
residual at small steps. Maximum corrections of 67,382 m and 1,363,259 m
restricted the step taken across each grid. In 1606 the worst row was
nearly dry, with a current Newton diagonal of 53.9 and a full-saturated
scaling diagonal of 47,803 m²/day. The drying guard changed none of the
heads in the small-step comparisons. In 1304, increasing the correction
shift in high-conductance rows cut the first maximum correction to
1,566 m and the 240-step physical RMS residual from 2,670 to 11.9 m³/day.
Two ungrounded components remained after 184.4 s. Each was a single
recharged cell with zero flow paths. The cached MF6 heads put the two
cells 0.43 and 0.23 mm above their bottoms, where they can pass the
recharge onward. A fully saturated preliminary solve stopped at its
line search after 13 steps and yielded no starting guess.

At the 1102 rejected state, the scaled Newton solve met its linear
target with a 72% defect in the physical norm. Physical weighting cut
that defect to 0.98%, but the tested trial steps still increased the
physical residual. At the 1801 rejected state, physical weighting
changed the initial slope of the squared physical residual from +0.520
to −1.987, normalized by its starting value.

A subsequent 1801 run reached a relative flow imbalance of 1.6e-11
but stopped after 285.6 s with a head-equivalent RMS error of 0.022 m.
Projecting three unforced, nearly dry cells to their bottoms cleared the
head-error check. The existing scaled solve still stalled on the
cell-by-cell flow check. From that saved projected state, physical
weighting cleared the unchanged physical gates in two steps with a
head-equivalent RMS error of 9.7e-11 m.

With the corrected solver, 1801 passed the physical checks from its
original heads in 142 Newton steps and 132.6 s, including 78.4 s in
Newton and 7,933 CUDA graph replays. Its final head-equivalent RMS
error was 9.8e-11 m and its relative cell imbalance was 1.2e-12.
The archived MF6 comparison missed only the two head gates. A new
MF6 reference at `dvclose=1e-6` took 135.4 s. The same DarcyWarp heads
passed every comparison gate against that reference, with datum-aligned
RMSE 4.02e-8 m and maximum difference 9.00e-6 m.

## Targeted MF6 Newton-guard qualification

The documented follow-up tally was 55 fully qualified, six awaiting a tight
MF6 comparison, and 13 solver failures. Four of those solver failures now
pass all gates from the archived initial heads. The tally is **59 fully
qualified, six awaiting tighter comparisons, and nine solver failures**.
The remaining failures are 1301, 1303, 1404, 1502, 1503, 1504, 1606, 1807,
and 1809. These are targeted results, with no full-suite rerun.

| Model | MF6 reference | DarcyWarp wall s | Newton / Krylov | Relative cell imbalance | Head RMSE m | Head maximum m |
|---|---|---:|---:|---:|---:|---:|
| 1018 | `dvclose=1e-6` | 91.7 | 112 / 2131 | 1.23e-10 | 1.09e-7 | 4.83e-6 |
| 1102 | archived | 63.8 | 168 / 3854 | 9.20e-12 | 7.16e-5 | 4.77e-3 |
| 1210 | `dvclose=1e-6` | 56.0 | 79 / 1224 | 9.79e-13 | 2.33e-7 | 1.58e-6 |
| 1304 | archived | 185.3 | 239 / 16181 | 8.78e-10 | 4.83e-5 | 9.86e-4 |
| 1801 regression | cached `dvclose=1e-6` | 106.7 | 44 / 5450 | 1.35e-12 | 4.02e-8 | 9.00e-6 |

The 1018 and 1210 tight references took 1233.5 and 736.2 seconds. Their
DarcyWarp solutions were reused for comparison. Their reports are in
`remaining_20d4381/regional_tight_comparison/`. The original DarcyWarp
reports for 1018 and 1102 are in `remaining_20d4381/refinement_krylov_budget/`.
The 1210 DarcyWarp report is in `remaining_20d4381/spatial_refinement_pending/`.
The 1304 and 1801 CUDA reruns, audits, and 1304 accepted-iterate trace are in
`remaining_20d4381/mf6_guard_c8553b7ba7fa/`. The 1304 wall time includes
recording the two-cell trace. The 1801 run used the dry projection and
replayed its mixed-fast K-cycle graph 5450 times.

Both 1304 and 1801 explicitly request `NEWTON UNDER_RELAXATION`. The importer
now enables the column-bottom rule from that option. A NEWTON-only
single-cell test with GHB inflow balancing a 2 m³/day well reaches −1 m
from a 1 m wet start. An exact or rounded-to-bottom trial still preserves
an isolated previously wet recharge outlet. A trial one representable value
above bottom retains that value.

Neither 1304 cell changed its four positive lateral connections on any
accepted iterate. The minimum wet thicknesses were 9.91e-9 m and
1.14e-12 m, respectively. Both drains stayed inactive at the endpoint.
Cell indices are zero-based in layer, row, column order.

| Cell | Starting wet thickness m | First accepted connectivity change | Final wet thickness mm | Recharge m³/day | Net face outflow m³/day | Local residual m³/day |
|---|---:|---|---:|---:|---:|---:|
| (0, 580, 758) | 29.54957 | none | 0.428511 | 3.61271395625 | 3.61271395712 | −8.71e-10 |
| (0, 504, 714) | 48.29021 | none | 0.228234 | 3.22035316250 | 3.22035316041 | 2.09e-9 |

### Saved-state and MF6 safeguard diagnostics

The archived IMS files for the remaining steady convertible cases use
`COMPLEX`. MF6 6.6.2 applies delta-bar-delta to the raw correction, adjusts
below-bottom heads when requested, and backtracks along the resulting step.
The preset reduces the weight by 0.8 on a sign reversal, restores it by
0.0001 otherwise, and backtracks by a factor of 0.1 when the flow norm grows
by more than 5%. It repairs a near-zero linear diagonal before adding the
steady pseudo-transient shift. GWF starts that shift from the maximum
absolute cell residual divided by cell volume. These are numerical
correction rules. [MF6 numerical solution](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Solution/NumericalSolution.f90) · [MF6 GWF model](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/GroundWaterFlow/gwf.f90)

On the saved 1102 failure, the spatially shifted Newton solve with physical
Krylov weighting used 39 iterations and left 461 for recovery. The full
trial clipped 57 cells and raised flow RMS from 5.246 to 60.649 m³/day.
The accepted one-eighth trial clipped four cells and reduced RMS to
4.977 m³/day. This replay did not exhaust the fallback budget.

At the previously identified 1606 row `(0, 1599, 775)`, the spatial shift
was 956.05 m²/day against a current Newton diagonal of 53.89 m²/day. Its
full-saturated principal diagonal was 47,802.57 m²/day and the grid median
was 277.17 m²/day. The MF6 residual-based initial shift computed from the
archived heads was 0.105324. Testing the MF6 relaxation sequence with that
shift and physical Krylov weighting still ended short of acceptance at
0.293 relative imbalance. A saved-state replay isolated the weighting
issue by reducing imbalance from 0.743 to 0.012 in six steps.

1504 and 1807 remain unqualified. The from-original runs with `cbe5d5d`
reached relative imbalance of 8.07e-5 and 1.25e-4, respectively, before
line-search rejection. The saved 1807 state produced a descending Newton
trial when Krylov used the physical residual. The accepted one-quarter
trial reduced flow RMS from 0.04492 to 0.04289 m³/day and head-equivalent
RMS from 16.10 to 0.02392 m. The replay does not change the qualification
count.

### 1807 original-head convergence with IMS backtracking

The highest-residual cell in the stalled 1807 checkpoint was 27.62 m
above bottom with four conducting faces. Ten neighbours contributed 40%
of the squared flow residual. An independent direct solve of the shifted
Newton matrix reduced the relative linear defect to 5.5e-12, yet the full
trial increased the physical flow norm by a factor of 449,000. Taking
1/128 of the direction reduced the flow norm by only 0.73%.

MF6 6.6.2 stops backtracking when another reduction would take the maximum
head displacement below DVCLOSE. The archived 1807 listing used this exit
at outer 88. Applying it after ordinary descent failed, with physical
Krylov weighting, allowed two Picard recovery steps in the original-head
DarcyWarp run. Independent physical acceptance passed in 204 outers and
43,989 Krylov iterations. Mixed-fast CUDA took 211.13 s including import,
with 177.71 s in the solve. Head-equivalent RMS was 1.90e-10 m and relative
cell imbalance was 1.92e-10.

Only head parity failed against the archived MF6 result, with 0.0014470 m
RMS and 0.0295536 m maximum difference. The last native head change was
0.09367 m. The original-start receipt is under
`remaining_20d4381/ims_head_limit_original_01/1807_1810_MF6_SS_Unconfined_250/`.
The tight MF6 6.6.2 reference at `dvclose=1e-6` finished in 800.37 s.
Both the native-head audit and the saved DarcyWarp solution passed every
gate. Head RMSE was 1.69e-7 m and the maximum difference was 5.30e-6 m.
The comparison receipts are under `remaining_20d4381/regional_tight_comparison/`.
The tally is 60 fully qualified, six more physically accepted awaiting
reference comparison, and eight solver failures.

### 1301 and 1303 after IMS backtracking

Both mixed-fast CUDA runs started from the archived initial heads and
reached the 240-outer limit. The model-specific results below use the
saved head at that limit for each follow-up. Directory names refer to
`remaining_20d4381/`.

| Model | Original run | Follow-up from the saved head |
| --- | --- | --- |
| 1301 | 340.4 s, relative cell imbalance 7.53e-5, head RMSE against cached MF6 0.01563 m | 96 accepted updates reduced flow RMS from 0.03431 to 0.0001722 m³/day and relative imbalance to 3.47e-7. |
| 1303 | 564.1 s, relative cell imbalance 4.26e-8, head-equivalent RMS 0.01709 m | Holding the correction at zero for exactly disconnected, unforced dry rows gave independent physical acceptance in 35 further updates, 8,432 Krylov iterations and 148.0 s including setup. Final head-equivalent RMS was 1.06e-8 m and relative imbalance was 1.48e-9. |

In 1303, cell `(0, 1926, 3)` ended one representable head above bottom in
both solvers. Its flow residual was −2.99e-20 m³/day, but its diagonal of
1.19e-21 m²/day translated that into a 25.15 m head-equivalent error.
After projection to bottom, its residual and all incident flow derivatives
were exactly zero. The linear solver could move this unconstrained head
by roundoff and reopen the cell. The replay held it still only while the
source, residual, package/storage tangent and incident face derivatives
were all zero. The original-start receipts are in
`ims_head_limit_original_01`. Replay receipts are in
`ims_head_limit_continuation_01` and `dry_zero_hold_saved_01`.

### 1606 nonlinear flow and checkpoint results

At a saved 1606 state, a direct diagnostic solve reduced the relative
linear defect to 2.82e-11. The resulting Newton step proposed a 471.6 m
rise at a cell where the cached MF6 head was only 42.64 m higher. Even
1/512 of the step increased the flow residual norm by a factor of 3.15.

A separate check interpolated from a later saved state towards the
cached MF6 heads. Moving 1% of the way increased flow RMS by a factor
of 8.59. Halfway it was 214 times the starting value. At the dominant
cell, the water table rose before the 15.33 m head difference with its
eastern neighbour had flattened. The midpoint combined 22.27 m of wet
thickness with a 7.67 m head difference. This produced about 27,100 m³/day
of eastward flow, compared with 12.6 m³/day at the native endpoint.
Neither the donor cell nor the drain branch changed at that row.

| Saved-state trial | Measured outcome |
| --- | --- |
| Physical Krylov weighting | Two accepted updates reduced relative imbalance from 1.29% to 0.575%. The third line search stalled. |
| Add delta-bar-delta relaxation alone | The second maximum correction fell from 67.90 to 54.32 m. Flow RMS reached 1.5854 m³/day and the third step still stalled. |
| Add adaptive pseudo-transient continuation alone | The shift fell from 5.67e-4 to 1.74e-4. Flow RMS reached 1.5880 m³/day and the third step still stalled. |
| Change the fine and coarse preconditioner | Including the receiving-cell derivative improved the first linear reduction from 0.23% to 0.99%. Using a conservative coarse operator then accepted seven steps, but raw corrections reached 41,340 m and relative imbalance increased from 0.575% to 0.590%. |
| Permit IMS head-change backtracking after the stall | Over 80 updates, head RMSE against cached MF6 fell from 3.154 to 1.923 m. Relative imbalance rose from 0.575% to 0.636%. |
| Continue with adaptive pseudo-transient continuation and IMS backtracking | Over the next 80 updates, flow RMS fell from 1.374 to 0.8524 m³/day, relative imbalance fell to 0.386%, and the final maximum proposed correction was 9.01 m. |

The isolated runs are in `weight_checkpoint_01`, `weight_dbd_only_01`,
`weight_ptc_only_01`, `directed_fine_saved_01` and
`conservative_coarse_saved_01`. The two successive 80-update runs are in
`mf6_backtrack_dvclose_01` and `adaptive_backtrack_continuation_01`.
All three models remain in the eight outstanding original-start solver
failures alongside 1404, 1502, 1503, 1504 and 1809.

A further 1606 continuation (`adaptive_backtrack_continuation_02`) used
the previous endpoint and its saved shift of 0.0003515478833. Through
66 accepted updates in 181.6 s it reduced flow RMS from 0.8524 to
0.3637 m³/day and relative imbalance from 0.386% to 0.137%. The process
terminated with SIGSEGV during the linear solve at update 67. The last
accepted metrics come from the progress log. The most recent saved heads
are still those at the end of `adaptive_backtrack_continuation_01`.

### Dry-cell correction from the original 1303 heads

The steady solver now holds an exactly dry convertible head in place when
its unforced equation and every incident face derivative are zero. The
original-head CUDA run reached 240 outers in 611.6 s with 48,503 Krylov
iterations. Head-equivalent RMS fell from 0.01709 m in the previous run to
4.57e-8 m. Relative imbalance fell from 4.26e-8 to 4.81e-9. Both device
and independent calculations still rejected local water balance. Maximum
cell flow error was 0.001906 m³/day, and 100 of 2,164,867 cells accounted
for 98.54% of the squared flow error. The receipt is under
`remaining_20d4381/hold_dry_zero_original_01`.

For 1301, 48 further updates from the previous checkpoint used square-root
diagonal scaling for Newton and measured the true linear defect in flow
units. Flow RMS fell from 1.72e-4 to 2.64e-5 m³/day, and relative
imbalance fell from 3.47e-7 to 5.85e-8. The first scaled Newton solve
reduced its weighted defect by 13.7% but increased its physical defect by
2.0%. The existing physical check rejected that direction. The receipt is
under `remaining_20d4381/equilibrated_refinement_01`. These two runs leave
the qualification tally at 60 full passes, six awaiting comparison and eight
solver failures.

### 1404 passes the physical solve from original heads

With `97b95e4`, 1404 passed independent physical acceptance in 98 Newton
updates and 10,477 Krylov iterations. Every accepted update reduced the
physical flow norm. Solve time was 97.32 s and total time including import
was 148.48 s. Head-equivalent RMS was 1.05e-8 m and relative cell
imbalance was 4.94e-11. Only the two head gates failed against the archived
MF6 output, with RMS difference 0.0005307 m and maximum 0.02089 m. The
archived MF6 final change was 0.09531 m. The saved DarcyWarp endpoint and
comparison are in `remaining_20d4381/97b95e4_original/1404_1405_1406_MF6_SS_Unconfined_250/`.
Pending the tighter comparison, the tally is 60 fully qualified, seven
physically accepted awaiting comparison, and seven solver failures.

### Linear and nonlinear limits at the saved heads

- In 1303, the 16-by-16 block containing the largest flow residual has an
  aggregate Newton diagonal of 167,059 m²/day. The symmetric approximation
  gives 1,032,549 m²/day because 865,491 m²/day of internal saturation
  derivatives are counted as restraint. Removing this term only on the
  coarse levels gave independent physical acceptance after 25 checkpoint
  steps (6,147 Krylov iterations, 126.18 s including import). Including
  directed transfers on the fine grid as well ended at 4.05% relative
  imbalance after 32 steps. A sub-millimetre Picard recovery in the first
  replay increased the physical norm from 0.0110 to 129.0 m³/day.
- The unshifted, accurately solved 1303 Newton direction at the original
  run's rejected endpoint had relative linear defect 6.02e-12 and maximum
  displacement 0.640 mm. Taking the full step reduced the nonlinear flow
  norm by 97.45%. In 1301 the coarse diagonal at the largest residual
  was only 6.29% above the aggregate Newton diagonal.
- On 1606, cellwise DBD with adaptive continuation reduced the largest
  individual recovery spike. At update 64 it had used 49 backtracking
  recoveries and reached 0.373% relative imbalance. Continuation alone
  from the same initial heads and shift used three recoveries and reached
  0.144% at the same update. The DBD receipt is under
  `remaining_20d4381/adaptive_backtrack_dbd_saved_01`.

### Local water balance and fallback steps

At the original 1303 endpoint under `97b95e4`, 65 cells failed local
water balance. They accounted for 94.4% of the squared flow residual. The
worst error was 0.0019064 m³/day against its 0.0006449 m³/day allowance.
Continuing from those heads passed the independent check after 14 full
Newton steps and 3,376 Krylov iterations (36.82 s solve, 101.02 s total).
Every step reduced flow error, so this replay provides a baseline for
testing changes to the preconditioner. Including directed Newton face
transfers on every multigrid level reproduced the independently calculated
coarse diagonal of 167,059 m²/day. On the same heads the resulting
preconditioner missed acceptance after 32 updates and 10,698 Krylov
iterations. The receipts are in `deferred_backtrack_01` and
`conservative_hierarchy_01` under `remaining_20d4381`.

On 1301, retaining the previous restart correction within the existing
Krylov budget reduced flow RMS from 2.64e-5 to 9.61e-6 m³/day over 48
additional checkpoint updates. Eight cells still failed local balance.
The worst had an error of 0.001022 m³/day against 0.0008974 m³/day
allowed. The receipt is in `restart_refinement_02`.

On 1606, starting from the same heads and continuation shift,
trying the remaining fallback directions before accepting an uphill
head-limited step reduced the relative imbalance at update 64 from
0.001622 to 3.95e-6 (146.98 s solve, 198.37 s total for the latter).
Neither endpoint passed. At the improved endpoint, an independent solve
of the unshifted Newton equation had relative linear defect 3.36e-12.
The full step increased the flow norm by a factor of 7.74. The cell with
the largest resulting flow error started 0.2354 m above bottom and its
head fell 0.2068 m. The original heads were used for the 1502 run under
`97b95e4`, which reached 240 outers and 44,791 Krylov iterations in
315.67 s with relative imbalance 9.84e-5. The subsequent 1503 run
stopped during a Warp CUDA readback at outer 205. The last accepted
update had relative imbalance 0.04053. These results leave seven
original-start solver failures.

### 1404 qualifies against tight MF6

Both the native audit and the original-head DarcyWarp solution passed
all gates against MF6 6.6.2 at `dvclose=1e-6`. The tight reference took
1,887.52 s. Head RMSE was 1.79e-7 m and maximum difference was 1.17e-5 m.
The comparison reused the 148.48 s DarcyWarp run. The receipts are in
`remaining_20d4381/regional_tight_comparison/1404_1405_1406_MF6_SS_Unconfined_250`.
This brings the tally to **61 fully qualified, six physically accepted
awaiting comparison, and seven original-head solver failures**.

1606 passed independent physical acceptance from the saved head in
`deferred_backtrack_01` after 45 further updates and 12,026 Krylov
iterations (119.54 s solve, 171.50 s including import). Head-equivalent
RMS was 9.95e-7 m and relative imbalance was 4.23e-9. Only the head
comparisons failed against the archived MF6 output, with RMS difference
0.0001970 m and maximum 0.01565 m. The last archived MF6 head change was
0.09557 m. The saved-head receipt is in `deferred_backtrack_continuation_01`.
An original-head run is still required for qualification.

On 1503, an accurate unshifted Newton solve at the saved
`physical_mf6_drain_damping` endpoint had relative linear defect 3.30e-10.
Cell (0, 568, 931) was 8.882 m below a drain of conductance 4.875e6 m²/day.
Its proposed 347.72 m rise generated 1.652e9 m³/day of drain outflow.
Archived MF6 had the water table just 2.022 mm above that drain.
A diagnostic using only the archived open/closed drain flags changed 83
branches and cut the maximum correction from 347.8 m to 44.15 m. The full
step still increased the flow norm by a factor of 2,848. Neither the
diagnostic flags nor MF6 heads were used to initialise a DarcyWarp run.

On 1303, the trial that deferred uphill fallback steps stopped during a
Warp CUDA-to-host Arnoldi column readback at outer 213. The last accepted
update had relative imbalance 3.53e-5. The same readback stopped 1503 at
outer 205. No new endpoint was recorded for either interrupted run.

### 1606 passes physical acceptance from original heads

Using the normal importer with deferred fallback ordering, 1606 passed
the independent physical checks from archived initial heads in 164 outers
and 35,403 Krylov iterations. The mixed-fast CUDA solve took 351.24 s,
or 585.87 s including import and setup. Head-equivalent RMS was 3.79e-7 m
and relative imbalance was 4.11e-9. Only the two head gates failed against
archived MF6, with 0.0001976 m RMS and 0.01567 m maximum difference.
The receipt is in `remaining_20d4381/deferred_backtrack_original_01`.
This run used the importer's spatial shift. The earlier 45-step checkpoint
used a uniform adaptive shift. The tally is 61 fully qualified, seven
physically accepted awaiting comparison and six original-head solver
failures (1301, 1303, 1502, 1503, 1504 and 1809).

The 1502 history under `97b95e4` records its first uphill Picard step at
outer 124. From outer 123's flow RMS of 0.38045246 m³/day it increased
flow error by a factor of 137.25, with 114 Krylov iterations left. Across
the six uphill steps, five increases were between 130 and 157 times.
Outer 157 had a smaller increase of 18.83 times.

### Original-head continuation and local closure

The uniform adaptive shift used for the successful 45-update 1606
continuation was also tested from the archived initial heads. Both models
used physical Krylov weighting, mixed-fast CUDA, deferred backtracking and
the existing 240-update limit.

| Model | Krylov iterations | Solve time | Final relative water imbalance |
| --- | ---: | ---: | ---: |
| 1303 | 25,206 | 291.86 s | 4.35e-4 |
| 1606 | 9,642 | 100.44 s | 0.708 |

Both runs missed physical acceptance. Their accepted head fields and
continuation states are under `remaining_20d4381/uniform_adaptive_original_01`.
The spatial shift produced better original-head results for both cases.

An independent CPU audit applied a proposed 3 µm local head-equivalent
allowance to hydraulically anchored cells. With all other gates held at
their qualification settings, the saved 1303 original-head endpoint and
the saved 1301 continuation both passed physical acceptance. At the
existing 1 µm allowance they missed in 65 and eight cells, respectively.
The 1303 original-head field differed from its strictly accepted continuation
by 3.19e-6 m RMS and 4.62e-4 m maximum over 2,164,867 active cells.
Relative water imbalance was 4.81e-9 for 1303 and 1.32e-8 for 1301,
compared with 2.12e-6 and 2.03e-6 at the archived MF6 heads. Adoption of
the new regional allowance awaits tight MF6 parity. The 1301 continuation
is diagnostic. The qualification count is unchanged.
