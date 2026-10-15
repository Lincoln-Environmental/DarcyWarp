# Diagnosis of the 64 by 64 low conductivity terrain failure

Current dev **reproduces the recorded failure**. The original cold-start experiment reaches its 2,000-outer limit with head residual RMS `3.944583e-7 m`, physical residual RMS `6.202167e-5 m3/day`, and relative cell imbalance `6.008032e-5`. Its head gate passes and both physical alternatives fail. This is a slowly contracting, heavily damped Picard trajectory. An independently labelled 4,000-outer-cap experiment converges at **outer 3,804**, with unchanged physics, initial conditions and final tolerances. The larger budget demonstrates convergence; it does not fix or pass the original experiment.

**Baseline and runtime.** Local dev and `git ls-remote origin refs/heads/dev` both returned `d056eaf666268e3c5f3e0017c91d51c3160162de`. This is two commits after the last reviewed `e77c8fff83d22c0f1bafd21ad0c68c36015d859c`. Diagnosis work is isolated on `diagnosis/low-k-64` in `/tmp/darcy-low-k-diagnosis`. Historical commit `ccb985e51aa0e51891209a64ccc8656a20c35aaa` was checked out separately in `/tmp/darcy-low-k-historical`. Neither dev nor unrelated files were modified; nothing was pushed or merged.

All numerical runs used the existing `/home/patrickdurney/miniconda3/envs/darcywarp/bin/python`: Python **3.14.7**, Warp **1.17.0**, `DARCY_FLOAT=float64`, and `cuda:0`, an NVIDIA GeForce RTX 4070 Ti SUPER with 16 GiB. Warp reported CUDA toolkit 12.9 and driver API 13.2; NVIDIA driver version is 595.91.07. No environment was installed or upgraded. The reviewed e77c8ff environment files did pin Warp 1.11.0; current dev pins 1.17.0. The installed interpreter differs from the canonical Python 3.12 pin. These are measurements under the recorded runtime, not a validation of the canonical environment or Warp 1.11.0. Other GPU jobs were active; recorded seconds support no performance comparison.

The requested AGENTS.md and all five specified source/evidence files were read. The only repository AGENTS.md applies throughout these worktrees; no nested instructions were found. The exact original `terrain_convergence_baseline.py` command ran before instrumented experiments. `shanz_terrain_case_study.py` supplied only `build_case`, never its main routine or warm starts.

**Experiment controls.** Every low-K run retains four independently varying layers, 16,384 active cells, 16,192 free cells, `dx=dy=20 m`, Kx `[2,8,0.5,0.0001] m/day`, Ky/Kx 0.6, Kz/Kx 0.15, recharge, coastal fixed heads, three natural-conductance west discharge-only contacts, and the 0.01 m floor. These are steady solves with zero storage. Required acceptance remains head RMS <=1e-6 m AND (physical RMS <=1e-12 m3/day OR residual L1/gross throughput <=1e-8). Initial relaxation is 0.7; restart 24; max_levels 6; min_coarse_n 16; the per-outer Krylov budget is 2,000, shared across blocks/retries. Only the explicitly labelled factor changes below.

| Experiment | Outers | Krylov iterations | Head RMS m | Physical RMS m3/day | Relative cell imbalance | Result |
|---|---:|---:|---:|---:|---:|---|
| Original current dev | 2000 | 48264 | 3.944583e-7 | 6.202167e-5 | 6.008032e-5 | outer_budget |
| Historical revision, same runtime | 2000 | 48264 | 3.944583e-7 | 6.202167e-5 | 6.008032e-5 | outer_budget |
| Current dev with read-only snapshots | 2000 | 48264 | 3.944583e-7 | 6.202167e-5 | 6.008032e-5 | outer_budget |
| Longer cap only, cold start | 3804 | 91560 | 7.004846e-11 | 1.029375e-8 | 9.971978e-9 | pass via relative physical gate |
| Inner targets multiplied by 0.01 only | 2000 | 343320 | 2.972507e-6 | 4.215272e-4 | 6.033976e-4 | outer_budget |
| Ordinary Kx basement 0.1 control | 859 | 20616 | 7.320904e-11 | 9.789511e-9 | 9.355530e-9 | pass; altered physics |

The historical artifact, historical rerun and current baseline have identical per-outer Krylov counts, relaxations and acceptance modes for all 2,000 rows. Their largest physical-RMS differences from current baseline are `9.33e-14` and `6.95e-14 m3/day`. Thus there is no material historical/current difference here to attribute to e77c8ff arithmetic or workspace changes. The later contact changes also leave this trajectory unchanged at the measured precision. This case has lateral contacts without overrides; the separate top/bottom override issue has no relevant call path.

**Observed convergence regimes.** All 2,000 baseline inner solves reach their target; none exhausts its Krylov budget or retries. Nineteen early updates use bounded recovery. The head gate first passes at outer 1,824. Physical progress continues through that event. The tighter-target experiment reaches 1,991 inner targets and exhausts the shared 2,000-Krylov budget on nine outers; it uses 75 bounded recoveries and never passes the head gate.

| Outer interval | Physical RMS at interval end | Geometric contraction per outer | Physical-growth steps |
|---|---:|---:|---:|
| 1–20 | 3.685611 | 0.88545 | 9 |
| 21–100 | 1.385303 | 0.98784 | 35 |
| 101–400 | 0.163486 | 0.99290 | 58 |
| 401–800 | 0.0217691 | 0.99497 | 31 |
| 801–1200 | 0.00314335 | 0.99517 | 35 |
| 1201–1600 | 0.000453850 | 0.99517 | 33 |
| 1601–2000 | 0.0000620217 | 0.99504 | 18 |
| 2001–2400, longer cap | 0.00000885827 | 0.99515 | 3 |
| 2401–3200, longer cap | 0.000000185526 | approximately 0.99518 | 0 |

This is early regime switching followed by approximately 0.5% residual reduction per outer, with oscillatory excursions that diminish late. It is neither a nonzero residual plateau nor evidence of an unattainable physical root. From roughly outer 800 onward, relaxation is 0.0109375 or 0.021875. The different-size historical results change domain extent at fixed dx and are not a spatial refinement study; they were not substituted for the current 64-cell experiment.

![Convergence histories and accepted relaxation](../audit_outputs/convergence.png)

**Residual hotspots and cancellation.** Indices below are zero-based `(layer, y, x)` at the actual accepted 2,000-outer head. Positive residual means original source minus outward flux is positive. Locations in metres use a cell-centre convention with the map origin at the southwest grid edge; numerical indices are authoritative.

| Cell | Residual m3/day | Head minus layer bottom m | Floor active | Kx m/day |
|---|---:|---:|---|---:|
| (1,15,16), x=330, y=310 m | +7.906695e-4 | +0.109096 | no | 8 |
| (1,16,16), x=330, y=330 m | +7.773326e-4 | +0.135726 | no | 8 |
| (1,16,17), x=350, y=330 m | +7.725795e-4 | +0.156747 | no | 8 |
| (0,16,15), x=310, y=330 m | -7.144172e-4 | -4.715300 | yes | 2 |

The topsoil, aquifer, weathered layer and basement contribute respectively **44.6068%, 55.0990%, 0.2942%, and 1.46e-10%** of the squared physical residual. Their residual L1 contributions are `0.107391`, `0.120115`, `0.00873951`, and `1.94e-7 m3/day`. The worst residual is in thinly saturated high-K aquifer cells, beside dry topsoil; it is not in the basement itself. There are 3,792 floor-active free topsoil cells and 1,368 floor-active aquifer cells. This floor-active count includes cells with positive saturation below 0.01 m; the separate strictly dry-cell count is 5,154.

Residual L1 is **0.2362456 m3/day**, gross throughput **3932.162682 m3/day**, but signed domain imbalance is only **-0.002681851 m3/day**. Its normalized magnitude `6.8203e-7` understates the cellwise relative imbalance by about 88 times. All three contact gates are open at the final state, with outflows `[0.000260748, 0.000780208, 0.000729641] m3/day`. The complete top-ten hotspot list, layer contributions, diagonals, fluxes and residual fields are in the numerical artifacts.

**Smallest discriminating frozen-state experiment.** At the accepted state immediately before outer 2,000, the nonlinear physical RMS is `6.227535e-5 m3/day`. Freeze the original operator and source residual there, keep fixed-head corrections zero, and compare three corrections:

| Frozen solve | Requested inner RMS m | Achieved inner RMS m | Krylov work | Nonlinear physical ratio at relaxation 0.0109375 |
|---|---:|---:|---:|---:|
| Baseline GPU | 3.969617e-8 | 9.422770e-9 | 24 | 0.9959265 |
| Fresh GPU workspace, target 100 times tighter | 3.969617e-10 | 3.720723e-10 | 264 | 0.9953062 |
| Independent sparse direct solve | direct factorization | 2.438040e-16 | not Krylov | 0.9934865 |

The sparse solve uses independently assembled flux/gradient/incidence equations, not the host nonlinear Picard algorithm. Its residual is checked against the frozen equations. The baseline correction differs from the direct correction by 76.9% in relative L2 and `2.28078e-4 m` in maximum magnitude. A small row-scaled residual is therefore not a certificate of an accurate correction. The estimated one-norm condition number of the row-scaled free-cell matrix is **6.924e5**; this is a condition estimate, not an exact spectral calculation. Principal diagonals range from `0.00681083` to `295.90162 m2/day`.

Nevertheless, more accurate corrections do not remove the nonlinear damping. At relaxation 0.021875, physical RMS ratios are 1.003604 (baseline), 1.006787 (tighter GPU), and 1.004071 (direct): all worsen physical balance. At 0.7 the ratios are 6.915, 8.101 and 8.305. Each correction first satisfies the actual descending merit test at **0.0109375** among the solver's halving candidates. The baseline raw and clipped correction RMS are both `7.384203e-5 m`, raw maximum `6.643239e-4 m`; the 50 m clip is inactive. The accepted RMS is `8.076473e-7 m` and maximum `7.266043e-6 m`.

Small directional finite differences along the direct correction (amplification is ||J delta|| / ||r||), using steps 1e-4 and 1e-5, give nonlinear directional amplification approximately **11.9433**, cosine with the physical residual **0.1150**, and a local physical-norm optimal relaxation approximately **0.00963**. The two step sizes agree closely and switch no floor cells. Thus this late direction is locally descending but strongly oblique: large steps amplify the physical residual. This quantifies the damping mechanism without claiming a full nonlinear Jacobian spectrum.

**Hypotheses tested.**

| Leading cause or hypothesis | Smallest supporting or rejecting experiment | Assessment |
|---|---|---|
| Slow nonlinear contraction exhausts the cap | Same cold-start physics with cap 4000; passes at 3804 | Supported directly. Original 2000-outer result remains a failure. |
| Krylov budget is insufficient | Every baseline inner solve hits target; no budget exhaustion or retry | Rejected as the immediate stopping mechanism. Raising max_cycles alone would not change target-limited exits. |
| Head-scaled forcing hides inaccurate corrections | Same-state direct and tighter GPU solves; correction changes substantially despite a small baseline residual | Supported as a conditioning/accuracy issue. It is not sufficient to explain or cure the long tail. |
| Globally tighter inner solves cure the failure | Factor 0.01 targets use 7.11 times the Krylov work and finish with 6.80 times the physical RMS | Rejected for the tested change. The tested tighter policy follows a different, slower nonlinear trajectory. |
| Late damping is caused by floor/gate switching | Baseline scans before outers 1600, 1800 and 2000; all tested relaxations have zero floor and gate switches | Rejected for these late baseline states. Early scans do cross floor thresholds, so switching can contribute early. |
| Head-tolerance normalization causes the late restriction | At outer 2000, remove the head-tolerance floor only in offline merit comparison | Relaxation 0.021875 still increases the equally relative-scaled merit to 2.00606 from 2. The physical norm itself increases. Not the principal cause at this state. |
| Frozen operator or residual disagrees between host and device | Identical accepted heads, independent fluxes/diagonals, fresh GPU frozen solves | No meaningful discrepancy found. Errors are near FP64 arithmetic noise. |
| Low-K stress is irrelevant | Change only basement Kx to the ordinary 0.1 control | Control passes at 859 outers. Contrast contributes to difficulty, but this alone does not isolate a particular nonlinear eigenmode. |
| A smoothed constitutive law is necessary | Unchanged-law longer-cap experiment passes | Not supported; necessity is contradicted for this case. |

**Corrective options ranked by evidence.**

1. **Operational budget allowance, if acceptable:** a 4,000-outer cap is demonstrated sufficient for this case with honest unchanged acceptance. Treat it as a separately authorized operating choice, not a fix to the original 2,000-outer experiment. Preserve failure reporting at the original cap.
2. **Investigate nonlinear acceleration or improved direction selection in a separate solver task:** the same-state scans and directional probes show why accurate Picard directions still need small relaxation. A useful future change must improve contraction under the original physics and final gates, and be checked on the ordinary-K and size controls. No such change is implemented or claimed validated here.
3. **Investigate selective forcing tied to physical progress:** head-only forcing can leave large correction errors in the conditioned frozen system. Tightening selectively after the head gate passes is an open experiment; the modest same-state gain and failed global-tightening experiment argue against adopting blanket tighter targets. Better linear preconditioning may reduce the cost of tighter solves but would not by itself remove the demonstrated nonlinear damping.

The exact nonlinear mode responsible for the oblique directions, its dependence on geological contrast versus geometry, and the benefit of alternative acceleration remain open. No nonlinear Jacobian spectrum, new host nonlinear solution, or independent MF6 truth solve was obtained for this stress case. Those validations are unavailable here, not passing tests. No physics redesign is justified by the iteration limit alone.

**Instrumentation and validation.** The wrapper changes no production source. Process-local wrappers around `run_inner`, `backtrack` and `run` only download selected fields and retain full scalar histories; all original methods are restored in `finally`. The separate target-factor experiment explicitly wraps `inner_target`. Default arithmetic, stopping tests and accepted-state updates remain in the original solver. Offline reconstruction starts only after the solve closes. Additional GPU frozen solves use disposable workspaces; they cannot mutate the accepted operator, RHS, gates or multigrid hierarchy.

The baseline with snapshots makes identical Krylov-count, relaxation and acceptance decisions on all 2,000 rows. Its largest physical/head RMS differences from the uninstrumented run are `8.48e-14 m3/day` / `3.85e-16 m`, respectively; this is trajectory agreement to floating-point noise, not a claim of bitwise identity. The longer-cap prefix also preserves every discrete decision. Across baseline checkpoints, maximum cell residual difference is `3.13e-13 m3/day`, principal diagonal difference `1.14e-13 m2/day`, face flux difference `3.13e-13 m3/day`, and contact flux difference `1.74e-18 m3/day`. At the difficult state, a fresh host-assembled GPU workspace reproduces the original frozen correction within `3.53e-15 m`.

Fifteen selected existing regressions passed, with 24 deselected and two Warp ctypes deprecation warnings. They cover RMS/budget accounting, latest inner norms, remaining-budget retries, physical acceptance, candidate contacts, rollback, and raw/clipped/accepted corrections. Script compilation, independent numerical reconstruction and all 17 [artifact validation checks](../audit_outputs/validation.json) also passed. The full repository suite was not run.

**Reproduction and artifacts.** From the diagnosis repository root:

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
/home/patrickdurney/miniconda3/envs/darcywarp/bin/python \
  working_tests/terrain_convergence_baseline.py \
  --device-picard --device cuda:0 --variant variable \
  --sizes 64 --dx 20 --floor 0.01 --basement-k 0.0001 \
  --max-cycles 2000 --max-outer 2000 \
  --out audit_outputs/low_k_64_baseline.json
```

All experiment, analysis and selected-test commands are recorded in [commands.sh](../audit_outputs/commands.sh). Revision/runtime metadata are in [revision_runtime.json](../audit_outputs/revision_runtime.json). The [diagnostic script](../working_tests/diagnose_terrain_low_k_64.py) provides run, `--analyze`, `--summarize` and `--validate` modes. Its controls and tuning values are defined in the exact execution guard; there are no lambdas.

- [Original result](../audit_outputs/low_k_64_baseline.json) and [same-runtime historical result](../audit_outputs/historical_same_runtime.json), each with a compressed outer history.
- [History comparisons and contraction regimes](../audit_outputs/history_analysis.json), plus the convergence figure above.
- [Baseline field analysis](../audit_outputs/current_instrumented/analysis.json), [longer-cap analysis](../audit_outputs/current_long/analysis.json), and [tighter-target analysis](../audit_outputs/current_tight/analysis.json).
- `current_instrumented/final.npz` contains the actual final head and case fields. `checkpoint_NNNN.npz` stores the accepted head immediately **before** outer N, physical residual, principal diagonal, correction RHS, coefficients, gate states, face/contact fluxes and raw correction. `accepted_NNNN.npz` stores the actual accepted state **after** that outer. Early, transitional and late snapshots include outers 1, 20, 100, 400, 800, 1200, 1600, 1800 and 2000.
- Corresponding `_reference.npz` and `_direct.npz` files hold independently reconstructed residual/diagonal/flux fields and sparse direct corrections. `final_reference.npz` holds final physical fields. Full trial histories, including merit values in the baseline instrumentation, are gzip JSON. Longer-cap and tighter-target directories retain their own final heads, selected checkpoints, runtime metadata and histories.
- [Regression output](../audit_outputs/regressions.log). [SHA256SUMS](../audit_outputs/SHA256SUMS) identifies the delivered numerical and command artifacts.
