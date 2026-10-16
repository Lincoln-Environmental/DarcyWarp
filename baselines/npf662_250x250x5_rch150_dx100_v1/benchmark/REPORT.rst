Final canonical case
=====================

Shape (nz,ny,nx)=(5,250,250); 100 m square cells; 25 km square domain.
Recharge is 150 mm/year (365 days), supplied by native RCH6, with IRCH=1
and FIXED_CELL. No WEL package occurs in either published native deck.
The only CHD boundary is the west column at head 0 m. Layer 0 is
convertible in the unconfined variant; all other layers are confined.
No conductance saturation floor or governing equation was changed.

At the final recharge and cell size, the convertible variant reaches
about 82.904 m maximum head; 59,054/62,500 columns are above model top.
It is mostly in the fully saturated regime. Surface discharge/runoff
is outside the specified single-coast/no-flow model.

summary.csv is the consolidated performance table. complete_results.json
contains all settings, measurements, counters and comparisons. Native
HEAD/CBC/GRB/decks/iteration CSVs are in ../<variant>/native.tar.gz.
Exact inputs are in ../<variant>/inputs.npz and ../<variant>/inputs/.
Decoded native arrays are in ../<variant>/mf6/.
The source/runtime/hardware metadata and all failures remain archived.

Validation: 6 focused regressions and 16 existing targeted regressions passed.
The backend metadata retains a legacy WEL_SOURCE name for its volumetric
RHS interface; the actual native package is RCH6, verified from each NAM
file and the returned recharge budget.

Measured local results
======================

Baseline source HEAD: 867b79e14dea83cfed7b4898264e366a1d05c971
Device: NVIDIA GeForce RTX 4070 Ti SUPER
Runtime: 3.14.7 | packaged by conda-forge | (main, Aug 30 2026, 16:54:41) [GCC 15.3.0]; Warp 1.17.0; NumPy 2.3.5; SciPy 1.16.3; FloPy 3.9.5.

Every successful median below uses three independently accepted runs from the exact cold initial state.
Public wall includes final independent host acceptance; the backend timer excludes it.

Variant       MF6 process   warmed solver   warmed public   warmed end-to-end   cold end-to-end + explicit compile
------------- ------------- --------------- --------------- ------------------- -----------------------------------
confined            10.5376         17.8940         18.1025             18.2010                             19.0409

confined: native accepted=True; native physical RMS=5.54541e-09; relative cell imbalance=2.02311e-09.
Recharge 255821.918 m3/day in 62250 top-layer RCH cells; native CHD exchange -255821.918 m3/day (injection sign).
Native full-cell maximum budget residual 9.12207e-08 m3/day; constitutive/native face-flow mismatch 1.7053e-13 m3/day.
Repeat 0: accepted=True, Newton=7, Krylov=21000, Jv=21448, candidates=7, fallback Krylov=0; head RMSE/max=2.5408832416121115e-07/4.982358490224215e-07 m; face relative L2=4.567474342934181e-09; device sample peak=520.0 MiB.
Repeat 1: accepted=True, Newton=7, Krylov=21000, Jv=21448, candidates=7, fallback Krylov=0; head RMSE/max=2.5408832416121115e-07/4.982358490224215e-07 m; face relative L2=4.567474342934181e-09; device sample peak=520.0 MiB.
Repeat 2: accepted=True, Newton=7, Krylov=21000, Jv=21448, candidates=7, fallback Krylov=0; head RMSE/max=2.5408832416121115e-07/4.982358490224215e-07 m; face relative L2=4.567474342934181e-09; device sample peak=520.0 MiB.
unconfined          97.6488         28.1123         28.3423             28.4457                             28.7706

unconfined: native accepted=True; native physical RMS=4.13522e-09; relative cell imbalance=1.68256e-09.
Recharge 255821.918 m3/day in 62250 top-layer RCH cells; native CHD exchange -255821.918 m3/day (injection sign).
Native full-cell maximum budget residual 1.01839e-07 m3/day; constitutive/native face-flow mismatch 9.92685e-08 m3/day.
Repeat 0: accepted=True, Newton=11, Krylov=33000, Jv=33704, candidates=11, fallback Krylov=0; head RMSE/max=4.6283177632838425e-07/6.597462913759955e-07 m; face relative L2=7.776657594667928e-09; device sample peak=520.0 MiB.
Repeat 1: accepted=True, Newton=11, Krylov=33000, Jv=33704, candidates=11, fallback Krylov=0; head RMSE/max=4.6283177632838425e-07/6.597462913759955e-07 m; face relative L2=7.776657594667928e-09; device sample peak=520.0 MiB.
Repeat 2: accepted=True, Newton=11, Krylov=33000, Jv=33704, candidates=11, fallback Krylov=0; head RMSE/max=4.6283177632838425e-07/6.597462913759955e-07 m; face relative L2=7.776657594667928e-09; device sample peak=520.0 MiB.

Standalone full-saturated K-cycle
----------------------------------

These are linear A0 timings, not preconditioned Newton timings.

Levels Smoother                    Cycles Accepted Wall_s       Cycle_s      Hierarchy_s
------ --------------------------- ------ -------- ------------ ------------ ------------
     3 chebyshev                      100 False         0.59906      0.38954      0.04000
     4 chebyshev                      100 False         0.89364      0.68675      0.03887
     5 chebyshev                      100 False         1.49060      1.28093      0.03992
     6 chebyshev                      100 False         2.70136      2.48807      0.03990
     7 chebyshev                      100 False         5.07106      4.86051      0.04099
     8 chebyshev                      100 False         9.87116      9.65833      0.04067
     5 vertical_line                   32 True          0.57923      0.43635      0.04122
     5 chebyshev_vertical_line         33 True          0.75635      0.61133      0.03916

Candidates for the NEXT integration task:
* L5 vertical_line: 0.57923 s total, 32 cycles; single observation, not a repeated performance ranking.
* L5 chebyshev_vertical_line: 0.75635 s total, 33 cycles; single observation, not a repeated performance ranking.

Resident workspace/graph reuse (each run still starts from the exact cold head):
* confined: 3 accepted repeats; median backend solver 17.8896 s; median public wall 18.0724 s. Shared construction is outside these per-call times.
* unconfined: 3 accepted repeats; median backend solver 28.1932 s; median public wall 28.3781 s. Shared construction is outside these per-call times.

Launch and readback profiling is a separate one-cycle run and excluded from timing medians:
* L3: 153 explicit wp.launch calls, 16 scalar readbacks; launch API wall 0.00284014 s, readback/wait wall 0.00324342 s.
* L5: 669 explicit wp.launch calls, 76 scalar readbacks; launch API wall 0.0105798 s, readback/wait wall 0.00271243 s.
* L8: 5485 explicit wp.launch calls, 636 scalar readbacks; launch API wall 0.0831557 s, readback/wait wall 0.0136648 s.

One-cycle L5 additivity error: 0.240888; homogeneity error: 0. A homogeneous map can still be nonlinear: this is evidence against treating this cycle as one fixed linear inverse.

Evidence and interpretation
----------------------------

The initial CG attempt is retained under benchmark_artifacts/npf662_250x250x5_v1/run_003.
Earlier namespaces preserve setup failures, CG/BICGSTAB diagnostics, excessive ILUT fill setup and superseded source mappings.
The final namespace uses the user-selected 100 m cells and 150 mm/year native RCH recharge.
No failed or superseded run contributes to a successful timing median.
The cold compilation sum above includes the NPF and Warp linear modules; K-cycle compilation is separate. First solve also captures specialized kernels/graphs.
Cold totals exclude interpreter/import/context startup and report-writing. Warm totals still reconstruct the model/workspace.
Memory samples include display/context; pool high-water values are process cumulative. Exact hierarchy allocation bytes are in JSON.
One native solve and one K-cycle observation per configuration limit timing precision; these are first-pass rankings.
See docs/NPF662_250_BASELINE.rst for array signs, equations, acceptance criteria and scaling/reuse/FGMRES integration requirements.

True fresh-process cold end-to-end
----------------------------------

confined: 20.216863 seconds.
unconfined: 30.397708 seconds.

Includes interpreter/imports, CUDA context startup, private empty-cache JIT, model/workspace/graphs, solve and independent physical acceptance, through process exit. One cold process per variant; these are not timing medians. The earlier cold column is the separately measured component total.
