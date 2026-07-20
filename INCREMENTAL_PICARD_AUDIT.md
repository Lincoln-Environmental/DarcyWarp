# Audit: Incremental Picard Implementation and 1M-Cell Failure

## Executive summary

The incremental Picard form was implemented correctly and is **accuracy-neutral** versus the direct-head path. It is **not the fix** for the 1000×1000 failure. The real problem is that the production Picard loop is stopping too early on fine grids: practical acceptance is satisfied after the minimum 8 outer iterations, but the head field is still changing by several millimetres per period and the error accumulates through the 30 stress periods.

Tightening the inner-solve and practical tolerances improves accuracy (RMSE drops from 2.55e-3 to 1.07e-3) but does **not** pass the acceptance gate and makes runtime slightly worse. A different lever — likely time-step sub-division, stronger multigrid preconditioning, or stricter outer convergence with compensating speedups — is needed.

## What was audited

| Run | Grid | `use_incremental_picard` | Tolerances | Final RMSE | Final max abs | Runtime | Acceptance |
|-----|------|--------------------------|------------|------------|---------------|---------|------------|
| Direct baseline | 500×500, 52p | False | default | 1.20e-5 | 4.75e-5 | 19.6 s | PASS |
| Incremental | 500×500, 52p | True | default | 1.20e-5 | 4.75e-5 | 19.6 s | PASS |
| Direct baseline | 1000×1000, 30p | False | default | 2.55e-3 | 6.38e-3 | 55.6 s | FAIL |
| Incremental | 1000×1000, 30p | True | default | 2.53e-3 | 6.33e-3 | 56.4 s | FAIL |
| Tight tol | 1000×1000, 30p | False | η=0.01, tol_min=1e-7, tol_max=1e-5, prac_head=1e-6, prac_dh=1e-3 | 1.07e-3 | 2.62e-3 | 57.2 s | FAIL |

Acceptance criteria: final RMSE < 1e-3, final max abs < 5e-3, runtime < 30 s.

## Key observations from the 1000×1000 production run

From `transient_replay_summary.json` (direct baseline):

- **Every period stops at the minimum** `min_practical_outer_iterations = 8`.
- **Strict Picard convergence never passes** in any period.
- **Head residual is tiny**: ~1e-8 RMS (well below `practical_head_residual_tol = 1e-5`).
- **Flow residual grows** from ~3e-3 in period 1 to ~8e-3 in period 30.
- **Head change between outer iterations is not small**: final max abs head change grows from 1.3 mm (period 1) to 3.6 mm (period 30); RMS head change grows from 5.3e-4 to 1.4e-3.
- **Inner K-cycles per period**: ~150–170 total, ~20–25 per outer iteration.
- **`storage_diag_change_rms` is reported as 0.0 every period** — this is suspicious and may indicate a reporting bug or that the storage-diagonal change is being zeroed before measurement.

### What this means

The inner linear solve is converging tightly (head residual ~1e-8). The problem is **outer Picard convergence**: the solver accepts the iterate because the residual is small, but the head field would keep changing if more outer iterations were allowed. At 1M cells the linear system is more ill-conditioned (smaller cell area → smaller storage diagonal → weaker diagonal dominance), so a small residual corresponds to a larger head change. The practical acceptance criteria do not detect this and stop too early.

Because the error grows period-to-period, the warm-start head for each subsequent period is slightly wrong, and the drift compounds.

## Why incremental Picard did not help

Mathematically, the incremental and direct forms converge to the same fixed point if the inner solve reaches the same residual target. The implementation confirms this: the two forms match to ~1e-9 on a near-linear test case and give effectively identical results on the production replay. The incremental form changes the multigrid **contraction factor** (how fast the inner solve converges), but the production gate is the **final residual**, not the number of inner cycles. Since both forms are driven to the same target, they arrive at the same (slightly unconverged) iterate.

## Root-cause hypotheses (ranked)

1. **Ill-conditioned linear system on fine grids (most likely)**
   - Storage diagonal scales as `S * dx² / dt`. At 1000×1000, `dx` is half the 500×500 value, so the storage diagonal is ~4× smaller relative to the transmissivity terms. This weakens diagonal dominance and makes the head change large even when the residual is small.
   - Evidence: tiny head residual but millimetre-scale head changes; error grows with grid refinement; 500×500 passes, 1000×1000 fails.

2. **Practical acceptance criteria are too loose for fine grids**
   - `practical_dh_rms_tol = 3e-3` allows the solver to stop while head changes are still 1.4e-3 RMS and 3.6 mm peak.
   - `min_practical_outer_iterations = 8` is a hard floor that is binding every period.
   - Evidence: every period stops at 8 iterations; strict Picard never passes; tighter tol helps but not enough.

3. **Accumulating period-to-period drift**
   - Each period’s accepted head is used as the initial head for the next period. A small uncorrected error in one period propagates and grows.
   - Evidence: final max abs head change grows monotonically from period 1 to period 30.

4. **Possible storage-diagonal reporting bug**
   - `storage_diag_change_rms = 0.0` every period is inconsistent with the code path that copies `storage_diag_prev_wp` and updates `storage_diag_wp`. This should be investigated because `practical_storage_diag_change_rms_tol = 30.0` is part of the acceptance gate.

5. **Discretization differences vs MF6 at fine grids (less likely but worth ruling out)**
   - If Warp and MF6 differ in how they handle wet/dry transitions or inter-cell transmissivity at fine resolution, the error could be a modelling difference rather than solver non-convergence.
   - Evidence against: the error pattern is monotonic growth consistent with drift, not localized oscillation.

## Next-step options

### A. Accuracy-focused changes

| Option | Expected effect | Risk/cost |
|--------|----------------|-----------|
| **A1. Stricter practical acceptance** — lower `practical_dh_rms_tol` to ~5e-4, raise `min_practical_outer_iterations` to 12–16, or require strict Picard | Forces more outer iterations, should reduce drift | Directly increases runtime; may not be enough if inner system is ill-conditioned |
| **A2. Time-step sub-division** — split each 7-day period into N=2–4 substeps with `dt/N` | Larger effective storage diagonal per sub-step → better conditioning; also higher temporal accuracy | N× more linear solves unless outer iterations drop enough to compensate; must compare against same MF6 truth (dt=7) |
| **A3. Stronger multigrid preconditioner** — increase `nu_pre`/`nu_post` to 2, raise `max_levels` for 1M cells, or increase Chebyshev order | Better contraction per inner cycle, may let Picard converge in fewer outer iterations | More work per cycle; needs benchmarking |
| **A4. Better coarse-grid solver** — direct solve on coarsest level if small enough, or more `nu_coarse` | Removes coarse-grid bottleneck | Implementation cost; may already be sufficient |
| **A5. Anderson acceleration / nonlinear acceleration** | Reduces outer Picard iterations | More complex; may destabilize near wet/dry front |
| **A6. Investigate `storage_diag_change_rms = 0.0`** | Fixes a possible acceptance-gate bug | May not move accuracy much, but acceptance logic is currently blind to this metric |

### B. Speed-focused changes

| Option | Expected effect | Risk/cost |
|--------|----------------|-----------|
| **B1. Skip hierarchy refresh on some outer iterations** | Biggest per-iteration cost outside the inner solve; refreshing every 2–4 iterations could save 20–40% | Operator lag can slow convergence; needs safety checks |
| **B2. Reduce `min_practical_outer_iterations`** | Directly reduces runtime | Already at 8 and binding; lowering would worsen accuracy |
| **B3. Adaptive hierarchy refresh** — refresh only when `dh_max` or storage change exceeds threshold | Balances A1 and B1 | Needs tuning |
| **B4. Profile kernel launch overhead** — merge small kernels, reduce scalar synchronizations | Currently many tiny kernels per outer iteration; scalar reductions synchronize the GPU | Implementation cost; gain likely modest unless overhead is dominant |
| **B5. Mixed precision / float32 for the inner solve** | Faster arithmetic and less memory traffic on GPU | May reduce accuracy; the production default is float64 |

### C. Diagnostic experiments

Before committing to a big change, run these targeted experiments on the 1000×1000 case:

1. **Strict-Picard-only run**: disable practical acceptance, set `max_outer_iterations = 50`, `hclose = 1e-4`, and measure how many outer iterations are actually needed to reach strict convergence and what the final accuracy/runtime are. This establishes a lower bound on the accuracy achievable with the current discretization.
2. **Sub-stepping sweep**: run with N=2, 4, 8 sub-steps per period (constant recharge per sub-step) and compare accuracy/runtime to the N=1 baseline. This tests the ill-conditioning hypothesis directly.
3. **Multigrid parameter sweep**: vary `max_levels` (4→5→6), `nu_pre`/`nu_post` (1→2), and `cheby_lambda_max` on the 1000×1000 case with tight tolerances to find the fastest setting that reaches strict convergence.
4. **Storage-diagonal change bug check**: verify why `storage_diag_change_rms` is reported as 0.0; fix if it is a measurement bug and re-run.
5. **Spatial error map**: compare Warp vs MF6 heads at period 30 to see whether error is diffuse (solver drift) or localized (discretization/wet-dry issue).

## Recommended plan

1. **Immediate (1–2 hours)**
   - Confirm the strict-Picard lower bound: run 1000×1000 with `practical_picard_acceptance_enabled=False`, `max_outer_iterations=50`, default hclose/strict tol. If strict convergence still fails or accuracy remains poor, the issue is solver conditioning/discretization, not just loose acceptance.
   - Fix/verify the `storage_diag_change_rms = 0.0` reporting; this acceptance gate should be functional.

2. **Short term (1 day)**
   - Run a sub-stepping sweep (N=2, 4) on 1000×1000. This is the highest-leverage accuracy fix and directly addresses the ill-conditioning hypothesis.
   - Run a multigrid parameter sweep to see if stronger smoothing reduces total runtime at fixed accuracy.

3. **Medium term (2–3 days)**
   - Combine the best sub-stepping factor with adaptive hierarchy refresh (B3) to recover runtime.
   - Re-evaluate whether `use_incremental_picard` can then be enabled by default; the incremental form may become useful if it reduces inner cycles once the conditioning is improved.

4. **Decision gate**
   - If sub-stepping at N=2 reaches RMSE < 1e-3 with runtime < 35 s, optimize from there (B3/B4) to hit < 30 s.
   - If even N=4 does not reach RMSE < 1e-3, the problem is likely a discretization or wet/dry modelling difference, and the focus should shift to matching MF6’s handling of the unconfined transition.

## Notes on the incremental implementation

The incremental code should be kept: it is a clean, verified alternative formulation and a useful A/B tool. Default should remain `False` until it is shown to improve a validated production configuration. If sub-stepping + stronger preconditioning makes the inner solve the bottleneck again, the incremental form may finally pay off by giving the multigrid solver a better-conditioned correction problem.
