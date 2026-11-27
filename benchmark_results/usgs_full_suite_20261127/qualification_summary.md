# USGS MODFLOW 6 regional qualification

## Run conditions

All 74 archived workspaces were run natively with MODFLOW 6.6.2 from copied inputs, retaining original initial heads, original IMS settings, and original package inputs. DarcyWarp used the current `dev` mixed-fast CUDA defaults on the RTX 4070 Ti SUPER. No converged native heads were used to initialize DarcyWarp. The CPU worker cap remained at 10 or fewer.

All 74 workspaces are single-layer, with active cell counts from 261,841 to 4,471,511. This suite contains no multilayer case, so it does not qualify multilayer behavior.

The per-model CSV records native runtime, DarcyWarp import/operator/setup/solve/wall time, sampled GPU memory, iterations, head statistics, face-flow errors, budget differences, physical residuals, and the original failure text. Per-model native-head and solution maps remain beside each receipt. See [the results table](darcywarp_qualification/usgs_regional_results.csv) and [selected committed spatial plots](qualification_evidence/spatial_maps_index.md); the full per-case map set is retained under `darcywarp_qualification/`.

## Outcome

**28/74 cases passed every existing parity and physical acceptance gate.** The other 46 remain recorded as failures; thresholds were not changed.

For the 28 passing cases, median native runtime was 166.3 s, median DarcyWarp wall runtime was 44.6 s, and the median DarcyWarp/native wall-time ratio was 0.429. These comparisons use the same archived physical inputs and initial conditions. Both runtime medians and the ratio are calculated over the 28 passing cases.

Failure groups:

**head comparison (25):** `0308_0309_0310_MF6_SS_Unconfined_250`, `0403_0404_0707_0709_0712_MF6_SS_Unconfined_250`, `0405_0409_0410_MF6_SS_Unconfined_250`, `0406_0407_0408_0416_MF6_SS_Unconfined_250`, `0414_0415_MF6_SS_Unconfined_250`, `0501_0502_0503_MF6_SS_Unconfined_250`, `0504_0506_0508_0509_MF6_SS_Unconfined_250`, `0511_0513_0514_MF6_SS_Unconfined_250`, `0512_MF6_SS_Unconfined_250`, `0711_0713_0714_MF6_SS_Unconfined_250`, `0804_0805_0808_MF6_SS_Unconfined_250`, `0901_0902_0903_MF6_SS_Unconfined_250`, `0904_1002_1003_1004_1005_1007_MF6_SS_Unconfined_250`, `1006_1010_1011_MF6_SS_Unconfined_250`, `1008_1009_1012_MF6_SS_Unconfined_250`, `1016_1017_MF6_SS_Unconfined_250`, `1109_1110_1112_1113_MF6_SS_Unconfined_250`, `1201_1202_1203_1204_MF6_SS_Unconfined_250`, `1403_MF6_SS_Unconfined_250`, `1408_MF6_SS_Unconfined_250`, `1501_MF6_SS_Unconfined_250`, `1701_1702_1703_MF6_SS_Unconfined_250`, `1708_1711_1713_MF6_SS_Unconfined_250`, `1712_MF6_SS_Unconfined_250`, `1803_1804_MF6_SS_Unconfined_250`
**Newton outer budget (14):** `1015_1020_1021_1022_1023_MF6_SS_Unconfined_250`, `1025_1026_1027_MF6_SS_Unconfined_250`, `1103_1104_1105_1106_MF6_SS_Unconfined_250`, `1205_1208_MF6_SS_Unconfined_250`, `1210_1211_MF6_SS_Unconfined_250`, `1308_1309_MF6_SS_Unconfined_250`, `1404_1405_1406_MF6_SS_Unconfined_250`, `1502_1506_MF6_SS_Unconfined_250`, `1503_1507_MF6_SS_Unconfined_250`, `1504_1505_1508_MF6_SS_Unconfined_250`, `1602_1603_MF6_SS_Unconfined_250`, `1801_1802_1808_MF6_SS_Unconfined_250`, `1805_1806_MF6_SS_Unconfined_250`, `1807_1810_MF6_SS_Unconfined_250`
**ungrounded saturation component (4):** `1018_1019_MF6_SS_Unconfined_250`, `1304_1307_MF6_SS_Unconfined_250`, `1606_MF6_SS_Unconfined_250`, `1809_MF6_SS_Unconfined_250`
**line search failed (1):** `1102_1108_MF6_SS_Unconfined_250`
**native-head physical audit (2):** `1301_1302_MF6_SS_Unconfined_250`, `1303_1305_1306_MF6_SS_Unconfined_250`

## Native tolerance check

Seven saved tighter MF6 references were re-audited on CUDA against the DarcyWarp solutions. All passed the existing head, face-flow, and package gates. This establishes that the original head-only misses in these seven cases came from the looser original MF6 solution, not a relaxed DarcyWarp criterion:

| Model | Tight MF6 seconds | Datum-aligned head RMSE (m) | Maximum absolute difference (m) | Tight comparison |
|---|---:|---:|---:|---|
| 0308_0309_0310_MF6_SS_Unconfined_250 | 495.2 | 1.19e-11 | 2.39e-09 | passed |
| 0403_0404_0707_0709_0712_MF6_SS_Unconfined_250 | 243.4 | 1.06e-07 | 1.81e-05 | passed |
| 0405_0409_0410_MF6_SS_Unconfined_250 | 227.4 | 1.31e-06 | 4.81e-05 | passed |
| 0406_0407_0408_0416_MF6_SS_Unconfined_250 | 262.8 | 1.45e-07 | 4.52e-06 | passed |
| 0414_0415_MF6_SS_Unconfined_250 | 792.7 | 3.96e-08 | 2.65e-05 | passed |
| 1201_1202_1203_1204_MF6_SS_Unconfined_250 | 109.7 | 1.90e-10 | 1.39e-08 | passed |
| 1408_MF6_SS_Unconfined_250 | 228.5 | 1.03e-06 | 1.96e-05 | passed |

The other 18 head-parity misses have not yet been checked against a tighter native reference. Their saved native-head equation audits passed, but the head mismatches remain unresolved until tighter references are evaluated.

## Physical audit failures and solver failures

Two native-head audits stop before a DarcyWarp solve because an individual DRN entry differs from the native budget by 1.4833e-6 and 1.2065e-6 m³/day, respectively, just above the unchanged 1e-6 absolute gate. Independent FloPy reconstruction shows the discrepancy is one HDS double-precision ULP at the worst high-conductance entries: cell (0, 1146, 828), conductance 3.3125e6, in 1301/1302; and cell (0, 1099, 1378), conductance 5.4612e6, in 1303/1305/1306. The native CBC flux implies a head 4.55e-13 m and 2.27e-13 m from the saved head, respectively. CPU and GPU DRN calculations agree exactly at the saved heads, the imported arrays match, and aggregate DRN budgets pass. The per-entry gate failures remain open; no tolerance was raised.

The remaining 19 failures are solver failures: 14 reach the Newton outer limit, four end with ungrounded saturation components, and one fails line search. A few are close but still fail physical balance (1015/1020/1021/1022/1023 ends at 0.0448 m head-equivalent RMS and 0.00582 relative cell imbalance; 1103/1104/1105/1106 at 0.0188 m and 0.00317; 1308/1309 at 0.000692 m and 0.00234). Several others diverge severely, with head-equivalent residuals from thousands to 1e14 m. Rejected states and solver histories are retained under each model directory.

The 1602/1603 IndexError from the first batch did not reproduce. A fresh run under the same defaults ended at the Newton outer limit with a very large head residual; its official receipt now records that reproducible outcome.

## Setup cost correction

Dry-state active-set seeding previously rebuilt full-grid component masks and rescanned all boundary entries for every component. On model 1606 this stage exceeded 17 minutes without entering Newton. The revised CPU reference groups cells and package entries by component once; the same case reached Newton in about 114 seconds. Its solver then failed on physical convergence, which remains a separate issue. A regression now exercises three disconnected components with separate drain activation thresholds.

## Retained diagnostics

Each case directory retains the native-head audit JSON, raw spatial arrays and maps, solver report, and rejected head/residual output when a solve failed. Seven tighter-reference audits and their solution maps are under `darcywarp_qualification/tight_reference_audit/`. The summary plot is `darcywarp_qualification/qualification_summary_plots.png`. Selected maps and the seven tight-reference audit JSON reports are committed under `qualification_evidence/`.
