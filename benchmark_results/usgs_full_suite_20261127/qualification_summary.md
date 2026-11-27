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

23 of the 25 original head-only gate misses were re-audited against separate MF6 references at tighter IMS head closure (1e-6; 0308 used 1e-10). All 23 passed the existing head, face-flow, package and physical acceptance gates. DarcyWarp used its original-input initial heads; native converged heads were used only in these comparison audits. The original 74-case status and gates remain unchanged.

| Model | Tight MF6 seconds | Datum-aligned head RMSE (m) | Maximum absolute difference (m) | Tight comparison |
|---|---:|---:|---:|---|
| 0308_0309_0310_MF6_SS_Unconfined_250 | 495.2 | 1.19e-11 | 2.39e-09 | passed |
| 0403_0404_0707_0709_0712_MF6_SS_Unconfined_250 | 243.4 | 1.06e-07 | 1.81e-05 | passed |
| 0405_0409_0410_MF6_SS_Unconfined_250 | 227.4 | 1.31e-06 | 4.81e-05 | passed |
| 0406_0407_0408_0416_MF6_SS_Unconfined_250 | 262.8 | 1.45e-07 | 4.52e-06 | passed |
| 0414_0415_MF6_SS_Unconfined_250 | 792.7 | 3.96e-08 | 2.65e-05 | passed |
| 0501_0502_0503_MF6_SS_Unconfined_250 | 96.7 | 8.66e-10 | 1.76e-07 | passed |
| 0504_0506_0508_0509_MF6_SS_Unconfined_250 | 96.8 | 2.83e-08 | 2.3e-06 | passed |
| 0511_0513_0514_MF6_SS_Unconfined_250 | 131.6 | 2.66e-09 | 4.16e-07 | passed |
| 0512_MF6_SS_Unconfined_250 | 121.4 | 9.97e-08 | 4.77e-06 | passed |
| 0711_0713_0714_MF6_SS_Unconfined_250 | 282.7 | 9.5e-09 | 6.25e-07 | passed |
| 0804_0805_0808_MF6_SS_Unconfined_250 | 118.8 | 2.62e-10 | 1.83e-08 | passed |
| 0901_0902_0903_MF6_SS_Unconfined_250 | 215.0 | 6.88e-08 | 1.5e-05 | passed |
| 0904_1002_1003_1004_1005_1007_MF6_SS_Unconfined_250 | 210.4 | 5.5e-09 | 1.18e-06 | passed |
| 1006_1010_1011_MF6_SS_Unconfined_250 | 343.0 | 2.11e-07 | 9.85e-06 | passed |
| 1008_1009_1012_MF6_SS_Unconfined_250 | 802.8 | 5.48e-09 | 2.05e-06 | passed |
| 1016_1017_MF6_SS_Unconfined_250 | 125.5 | 5.55e-08 | 2.72e-06 | passed |
| 1201_1202_1203_1204_MF6_SS_Unconfined_250 | 109.7 | 1.9e-10 | 1.39e-08 | passed |
| 1403_MF6_SS_Unconfined_250 | 54.9 | 1.51e-08 | 1.51e-06 | passed |
| 1408_MF6_SS_Unconfined_250 | 228.5 | 1.03e-06 | 1.96e-05 | passed |
| 1701_1702_1703_MF6_SS_Unconfined_250 | 718.2 | 3.55e-06 | 0.00433 | passed |
| 1708_1711_1713_MF6_SS_Unconfined_250 | 59.8 | 3.31e-08 | 3.17e-06 | passed |
| 1712_MF6_SS_Unconfined_250 | 223.4 | 8.02e-08 | 3.46e-06 | passed |
| 1803_1804_MF6_SS_Unconfined_250 | 592.9 | 6.29e-07 | 0.000317 | passed |

Two supplemental tight runs remain unresolved because they showed poor contraction under the archived IMS iteration limits. These are diagnostic-only runs; they do not invalidate the original-settings native references or change the qualification result:

- `1109_1110_1112_1113_MF6_SS_Unconfined_250`: manual stop requested after observing model iteration 9; the flushed log records 9 model iterations and a final Backtracking 10 residual of 4.401e6, down from 4.755e6. Linear solves used up to 200 iterations.
- `1501_MF6_SS_Unconfined_250`: manual stop requested after observing iteration 29 due to poor contraction; the flushed log records model iteration 30 and a final Backtracking 31 residual of 1.263e6, down from 2.261e6.

Both were manually interrupted diagnostic copies, not natural MF6 failures. Their final logs and explicit status are retained under `darcywarp_qualification/tight_references/`; no maximum iteration limit or physical tolerance was changed.
## Physical audit failures and solver failures

Two native-head audits stop before a DarcyWarp solve because an individual DRN entry differs from the native budget by 1.4833e-6 and 1.2065e-6 m³/day, respectively, just above the unchanged 1e-6 absolute gate. Independent FloPy reconstruction shows the discrepancy is one HDS double-precision ULP at the worst high-conductance entries: cell (0, 1146, 828), conductance 3.3125e6, in 1301/1302; and cell (0, 1099, 1378), conductance 5.4612e6, in 1303/1305/1306. The native CBC flux implies a head 4.55e-13 m and 2.27e-13 m from the saved head, respectively. CPU and GPU DRN calculations agree exactly at the saved heads, the imported arrays match, and aggregate DRN budgets pass. The per-entry gate failures remain open; no tolerance was raised.

The remaining 19 failures are solver failures: 14 reach the Newton outer limit, four end with ungrounded saturation components, and one fails line search. The four component labels occur on rejected nonlinear states with relative cell imbalances of 0.00350, 2.009, 1.859, and 0.445; these are not balanced floating components and accepting them by datum would suppress real residual. A few are close but still fail physical balance (1015/1020/1021/1022/1023 ends at 0.0448 m head-equivalent RMS and 0.00582 relative cell imbalance; 1103/1104/1105/1106 at 0.0188 m and 0.00317; 1308/1309 at 0.000692 m and 0.00234). Several others diverge severely, with head-equivalent residuals from thousands to 1e14 m. Rejected states and solver histories are retained under each model directory.

The 1602/1603 IndexError from the first batch did not reproduce. A fresh run under the same defaults ended at the Newton outer limit with a very large head residual; its official receipt now records that reproducible outcome.

### Controlled 1304 solver diagnostics

The original-input 1304 case was rerun with the same initial heads and physical acceptance gates to isolate precision and multigrid depth. These are diagnostics only; qualification remains on mixed-fast defaults.

| Precision | K-cycle levels | DarcyWarp wall seconds | Newton steps | Krylov iterations | Physical RMS (m³/day) | Relative cell imbalance | Result |
|---|---:|---:|---:|---:|---:|---:|---|
| mixed-fast | 5 | 89.7 | 80 | 695 | 3454.2 | 2.0094 | failed |
| FP64 | 5 | 47.0 | 80 | 722 | 3032.8 | 2.0541 | failed |
| mixed-fast | 7 | 47.3 | 80 | 733 | 2750.4 | 1.7951 | failed |
| mixed-fast | 9 | 56.2 | 80 | 711 | 2915.3 | 1.8632 | failed |

FP64 preconditioning did not restore balance. Seven levels improved the endpoint, while nine levels regressed slightly; neither approached acceptance. The five-level hierarchy contributes to convergence quality, but depth alone does not explain the failure. These runs preserve their full iteration histories under `solver_diagnostics/`.

The largest rejected residual rows inspected in 1018 and 1304 lie on listed high-conductance drains, but the rejected heads there are respectively 96.8 m and 68.2 m below drain elevation. Those entries are inactive at those heads, so their drain discharge does not explain the large local residuals. The residual is concentrated in the regional flow field; its nonlinear cause remains unresolved.

## Setup cost correction

Dry-state active-set seeding previously rebuilt full-grid component masks and rescanned all boundary entries for every component. On model 1606 this stage exceeded 17 minutes without entering Newton. The revised CPU reference groups cells and package entries by component once; the same case reached Newton in about 114 seconds. Its solver then failed on physical convergence, which remains a separate issue. A regression now exercises three disconnected components with separate drain activation thresholds.

## Retained diagnostics

Each case directory retains the native-head audit JSON, raw spatial arrays and maps, solver report, and rejected head/residual output when a solve failed. Twenty-three tighter-reference audits and their solution maps are retained under `darcywarp_qualification/tight_references/`. The summary plot is `darcywarp_qualification/qualification_summary_plots.png`. Selected head and saturated-thickness maps, all 23 compact passing tight-reference audit records and the two interrupted-run receipts are committed under `qualification_evidence/`.
