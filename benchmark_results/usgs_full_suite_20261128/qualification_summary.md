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

Saved receipts establish 54 distinct fully qualified workspaces. The 28
archived-reference passes and 26 tighter-reference passes are disjoint. Six
additional DarcyWarp solutions passed the physical checks and still need
sufficiently accurate MF6 comparisons.

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
Two ungrounded components remained after 184.4 s. A fully saturated
preliminary solve stopped at its line search after 13 steps and yielded
no starting guess.

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
head-equivalent RMS error of 9.7e-11 m. This restart provides numerical
evidence for the correction. Full qualification requires a run from the
original heads and a comparison with the cached MF6 endpoint.
