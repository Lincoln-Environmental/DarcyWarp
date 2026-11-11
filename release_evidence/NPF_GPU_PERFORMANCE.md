# NPF GPU performance: solve-only versus MF6

Twenty weekly steps; same physical inputs and cached native results. GPU medians use three repetitions after warmup.
The 1e-7 override changes only GPU relative cellwise acceptance; production defaults remain 1e-8.

| Model | GPU solve | MF6 engine | MF6/GPU | Extra test audit | Model build |
|---|---:|---:|---:|---:|---:|
| npf_confined_250x250x5_transient | 7.672 s | 47.589 s | 6.20x | 4.402 s | 0.049 s |
| npf_convertible_250x250x5_transient | 8.109 s | 17.319 s | 2.14x | 4.424 s | 0.052 s |
| npf_confined_500x500x5_transient | 27.636 s | 346.408 s | 12.53x | 20.166 s | 0.228 s |
| npf_convertible_500x500x5_transient | 30.243 s | 104.391 s | 3.45x | 19.776 s | 0.219 s |

Public solver time excludes external model construction, cached-reference reads, the extra test audit and file saving.
It retains setup and acceptance performed internally by solve(). MF6 engine timing excludes the FloPy build/write stage.
MF6 values are single reference-generation runs, not repeated process medians.

All 320 GPU endpoints passed independent physical acceptance at the requested 1e-7 relative limit.
The tighter 1e-8 baseline also passed all 320 endpoints. The full-verification runtime reduction was 2.8–7.3%;
the baseline did not separately record public solve time, so that reduction is not labelled a solve-only improvement.

## Achieved metrics

| Model | GPU cellwise imbalance | MF6 cellwise imbalance | Maximum head difference |
|---|---:|---:|---:|
| npf_confined_250x250x5_transient | 5.93719e-08 | 9.89242e-08 | 1.06753e-06 m |
| npf_convertible_250x250x5_transient | 9.2969e-08 | 3.45501e-08 | 2.34963e-08 m |
| npf_confined_500x500x5_transient | 7.13876e-08 | 2.13899e-07 | 3.49349e-06 m |
| npf_convertible_500x500x5_transient | 9.74949e-08 | 1.42258e-07 | 6.35474e-08 m |

These are achieved metrics, not equivalent input stopping criteria. MF6 uses the actual 2D sanity IMS profile;
its stricter NPF audit remains separately recorded, and GPU head/absolute/boundary limits are unchanged.

Detailed records: `realistic_transient_verified/results.json`, `realistic_transient_rel1e7/results.json`, and native `realistic_reference_cache/*/native/`.
Large raw outputs remain local and are excluded from Git. Reproduce them with `working_tests/run_3d_sanity.py`.
