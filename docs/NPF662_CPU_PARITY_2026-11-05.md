# DarcyWarp NPF CPU numerical parity

5 November 2026

**PASS: 20 steady 3D cases compared with live MODFLOW 6.6.2.**

DarcyWarp dev source: `7f2c411a9a32cef8920248c0eccfd5fa17fc9625`, with the attached recharge bookkeeping correction. Warp 1.17.0, FP64, FloPy 3.9.5, Python 3.12, CPU execution. These results measure numerical agreement; they establish no GPU timing claim.

| Case group | Worst head RMSE (m) | Worst head difference (m) | Worst relative face-flow L2 difference |
| --- | ---: | ---: | ---: |
| confined | 4.8064e-09 | 1.0977e-08 | 1.3256e-10 |
| convertible | 4.0409e-08 | 9.8241e-08 | 1.3387e-09 |
| phreatic | 6.9043e-09 | 1.2443e-08 | 4.0934e-09 |

Each group includes 16 x 16 x 5, 32 x 32 x 5 and 50 x 50 x 5 grids over the same 25 km footprint. Each case had four live MF6 runs (one warmup plus three repeats), four fresh-model DarcyWarp CPU runs, and two resident-model cold-head runs. All 90 measurements passed the final gates. The phreatic top layer remained below model top in all tested free columns.

Five additional confined layered controls covered uniform, x-varying, y-varying, independently varying and masked geometry. Six nonlinear layered controls covered wet, dry and mixed starts, a masked mixed domain, and dry-source recovery from two initial heads. They compare DarcyWarp CPU, an independent NumPy/SciPy root, and live MF6, with separate supplied-head constitutive checks. All 11 controls passed.

Across all 20 cases, maximum head difference was 9.82412587e-08 m; maximum relative face-flow difference was 4.09336422e-09. Worst normalized cellwise imbalance was 4.35240166e-09 for DarcyWarp and 5.14999446e-09 for MF6, both below 1e-8.

## Recharge bookkeeping correction

Initial live native runs were rejected solely by a fixed 1e-14 m3/day source-allocation check. RCH converts recharge to volume by multiplying by cell area; the observed mismatch reached 2.27e-13 m3/day per cell for sources around 1,000 m3/day. The corrected check uses a per-cell FP64 arithmetic allowance, retaining the existing 1e-14 absolute floor. The domain and cellwise physical conservation gates are unchanged. An added live regression verifies RCH against the equivalent WEL allocation and DarcyWarp CPU. The focused suite passes 15 tests, including live native checks.

## Native executable and scope

The official Linux MF6 6.6.2 release was downloaded from https://github.com/MODFLOW-ORG/modflow6/releases/download/6.6.2/mf6.6.2_linux.zip . Executable SHA-256: `9acdb07cae589555355b5d25b5ec1a7af7a0127ff95339ee14971ec7df6641ab`.

The workstation pin is a different binary build of the same version. Only the detached validation writer uses the official-build checksum; the repository binary pin remains unchanged. The original failed evidence, corrected runs, native decks/HEAD/CBC/GRB outputs, exact inputs, source snapshots, comparison metrics and reproduction scripts are retained in the evidence archive. The native executable is not included.

Confirmation applies to the tested steady structured DIS/NPF cases with axis-aligned conductivity, CHD and source forcing, including the selected convertible-cell and dry-start behavior. Transient storage, XT3D, other packages, large-grid GPU parity and GPU performance were outside these runs.
