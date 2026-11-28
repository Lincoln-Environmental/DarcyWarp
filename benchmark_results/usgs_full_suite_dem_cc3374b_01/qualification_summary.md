# USGS regional qualification from DEM starting heads

DarcyWarp qualified **71 of 74** models from DEM starting heads. It passed independent physical balance on **72 of 74**. MF6 6.6.2 terminated normally and passed its native output audit on all 74. Of the qualified pairs, 38 passed against the timed native run, 28 used cached tight native references, and five passed against a tighter solve started from the timed native endpoint.

Both solvers used identical archived DIS top elevations for every free active cell. Fixed and inactive starting heads retained their archived values. All 74 cases are single-layer. The timed MF6 runs used archived IMS settings with up to ten single-core jobs in parallel. DarcyWarp ran one solve at a time on an RTX 4070 Ti SUPER with CPU cores 10–23 available, five K-cycle levels, mixed-fast precision, CUDA graphs, deferred fallback ordering and a 240-update limit. The DarcyWarp source was `cc3374b`. The local anchored-cell allowance was 10 µm, the acceptance head RMS gate was 1 µm, and the relative water-balance gate was 1e-7.

Among the 71 qualified pairs, median elapsed time was 139.36 s for native MF6 and 80.41 s for DarcyWarp end to end. Median DarcyWarp import and numerical solve times were 33.54 s and 2.10 s. Tight reference timings are separate from the timed cold runs. The six new tight solves used DVCLOSE 1e-6 m and took 10.11–45.42 s from their saved native endpoints.

| Case | Result | MF6 time s | DarcyWarp wall s | DarcyWarp solve s | Newton updates | FGMRES iterations | Head RMS mm | Head max mm |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1303 | Qualified | 3438.47 | 410.13 | 313.87 | 128 | 25915 | 0.02032 | 3.396 |
| 1304 | Qualified | 4523.35 | 117.01 | 43.92 | 53 | 5136 | 0.03256 | 4.389 |
| 1502 | Qualified | 1646.82 | 73.78 | 11.84 | 31 | 1763 | 0.000392 | 0.04477 |
| 1504 | Qualified | 3858.71 | 92.33 | 46.66 | 44 | 7624 | 0.001810 | 0.9490 |
| 1606 | Qualified | 1103.94 | 226.04 | 160.07 | 69 | 15944 | 0.007961 | 1.131 |
| 1807 | Qualified | 473.05 | 127.84 | 92.36 | 84 | 23287 | 0.000150 | 0.003082 |
| 1809 | Head maximum failed | 697.40 | 109.92 | 74.82 | 76 | 14709 | 0.02428 | 7.918 |

## Outstanding cases

- **1301** reached update 240 after 304.11 solver seconds and 56,022 FGMRES iterations. At update 238 it accepted a deferred saturated fallback that raised flow RMS from 0.00703 to 5.72 m³/day. The final flow RMS was 0.10745 m³/day, relative imbalance was 1.4263e-5, and 10,110 cells exceeded their local allowance. Three rows accounted for 95.6% of the squared flow error.
- **1503** reached update 240 after 289.85 solver seconds and 49,897 FGMRES iterations. A deferred Picard fallback at the last update raised flow RMS from 0.03063 to 0.26453 m³/day. Final relative imbalance was 1.3763e-4, and 223,677 cells exceeded their local allowance. The largest errors were in face exchanges. The worst row had an outflow imbalance of about 152 m³/day and no drain exchange.
- **1809** passed independent physical balance. Thirteen neighbouring cells exceeded the 5 mm head maximum against the tight native refinement. The cached tight MF6 endpoint and the independently refined DEM endpoint agreed within 0.022 mm, confirming the discrepancy persisted across both references.

[Final qualification table](dem_qualification.csv) · [Run manifest](manifest.json)
