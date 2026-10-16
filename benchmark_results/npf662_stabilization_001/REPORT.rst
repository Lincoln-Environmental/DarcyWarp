Observed 3D Newton preconditioner comparison
===========================================

Saved 250x250x5 inputs and MF6 arrays were read unchanged. No model regeneration or native solve is part of this driver.
Fresh model timings include NPZ load, construction, hierarchy/workspaces, solve and public independent acceptance.
Warmup is excluded; successful medians require all three accepted repetitions. Resident runs also reset the cold head.

Multigrid device events measure scaling + graph + output copy; scalar-read wait time overlaps that GPU work.
These costs must not be added as disjoint runtime fractions. Hierarchy time includes graph setup.

confined kcycle: median end-to-end 0.683851s; Newton [5, 5, 5]; Krylov [29, 29, 29]; restarts [0, 0, 0].
  construction_seconds: 0.055639s (8.14% of end-to-end; overlapping categories).
  device_apply_seconds: 0.236155s (34.53% of end-to-end; overlapping categories).
  synchronization_seconds: 0.247919s (36.25% of end-to-end; overlapping categories).
  preconditioner_enqueue_seconds: 0.004539s (0.66% of end-to-end; overlapping categories).
confined diagonal: median end-to-end 18.248002s; Newton [7, 7, 7]; Krylov [21000, 21000, 21000]; restarts [None, None, None].
confined diagonal_fgmres: median end-to-end 139.960021s; Newton [8, 8, 7]; Krylov [24000, 24000, 19912]; restarts [496, 496, 411].
confined end-to-end speedup: 26.684x.
confined K-cycle versus diagonal FGMRES speedup: 204.665x.
unconfined kcycle: median end-to-end 0.652275s; Newton [6, 6, 6]; Krylov [28, 28, 28]; restarts [0, 0, 0].
  construction_seconds: 0.051887s (7.95% of end-to-end; overlapping categories).
  device_apply_seconds: 0.229181s (35.14% of end-to-end; overlapping categories).
  synchronization_seconds: 0.239584s (36.73% of end-to-end; overlapping categories).
  preconditioner_enqueue_seconds: 0.004298s (0.66% of end-to-end; overlapping categories).
unconfined diagonal: median end-to-end 28.522205s; Newton [11, 11, 11]; Krylov [33000, 33000, 33000]; restarts [None, None, None].
unconfined diagonal_fgmres: median end-to-end 173.740760s; Newton [10, 10, 11]; Krylov [30000, 27085, 33000]; restarts [620, 559, 682].
unconfined end-to-end speedup: 43.727x.
unconfined K-cycle versus diagonal FGMRES speedup: 266.361x.

Stabilization implementation and complete measurements
-------------------------------------------------------

Parent revision: dba5db283f22deb50329c4dbb518ef8acd8423d2. Shared hydraulic buffers remain owned by NPF662Workspace.
Only the selected Krylov basis is allocated. FGMRES-to-GMRES closes V/Z; GMRES-to-FGMRES releases the legacy solver/graphs.
Switching between diagonal FGMRES and K-cycle FGMRES retains V/Z, and compatible multigrid remains cached across method switches.
Public default remains diagonal GMRES. Diagnostic selection is npf_preconditioner="diagonal", npf_krylov_method="fgmres".

All 18 timed runs and all 32 total warmup/timed/resident runs passed unchanged device and independent acceptance.
The nonlinear controller, physical operator, tolerances, input hashes, restart=48, per-outer Krylov budget=4000,
outer budget=80 and five-level/one-cycle multigrid controls match the previous campaign.

confined kcycle:
  Median solver-only 0.297016s; public solve 0.578281s; host construction 0.055998s.
  Incremental live Warp allocation 402565388 bytes (383.916 MiB); sampled device-wide peak 1308 MiB.
  Jv [34, 34, 34]; fallback solves [0, 0, 0].
  Actual preconditioner applications [29, 29, 29]; K-cycles [29, 29, 29].
  FGMRES scalar-read wait median 0.247919s; readbacks [241, 241, 241]. CPU wait overlaps device work.
  Maximum across repetitions head_rmse_m: 5.47992024962e-08.
  Maximum across repetitions head_max_m: 1.3783019881e-07.
  Maximum across repetitions flux_relative_l2: 1.87026659558e-09.
  Maximum across repetitions flux_max_m3_day: 9.33396336222e-07.
  Maximum independent physical_rms: 7.8060803773e-10.
  Maximum independent head_residual_rms: 3.74017265257e-13.
  Maximum independent relative_cell_imbalance: 2.51899566661e-10.
confined diagonal:
  Median solver-only 17.905076s; public solve 18.143330s; host construction 0.054122s.
  Incremental live Warp allocation 219248348 bytes (209.092 MiB); sampled device-wide peak 1148 MiB.
  Jv [21448, 21448, 21448]; fallback solves [0, 0, 0].
  K-cycles 0. Legacy per-preconditioner application counts and scalar wait times are not instrumented (CSV null); no inferred counts are presented.
  Maximum across repetitions head_rmse_m: 2.54598030439e-07.
  Maximum across repetitions head_max_m: 4.98235849022e-07.
  Maximum across repetitions flux_relative_l2: 4.56747434293e-09.
  Maximum across repetitions flux_max_m3_day: 2.45591297698e-06.
  Maximum independent physical_rms: 3.45740807995e-09.
  Maximum independent head_residual_rms: 1.75269736289e-12.
  Maximum independent relative_cell_imbalance: 1.80326591087e-09.
confined diagonal_fgmres:
  Median solver-only 139.642533s; public solve 139.857596s; host construction 0.052341s.
  Incremental live Warp allocation 342509868 bytes (326.643 MiB); sampled device-wide peak 1284 MiB.
  Jv [24504, 24504, 20330]; fallback solves [0, 0, 0].
  Actual preconditioner applications [24000, 24000, 19912]; K-cycles [0, 0, 0].
  FGMRES scalar-read wait median 80.876624s; readbacks [1195904, 1195904, 992249]. CPU wait overlaps device work.
  Maximum across repetitions head_rmse_m: 2.43110828478e-07.
  Maximum across repetitions head_max_m: 4.81619991888e-07.
  Maximum across repetitions flux_relative_l2: 4.38817237004e-09.
  Maximum across repetitions flux_max_m3_day: 2.37020151417e-06.
  Maximum independent physical_rms: 3.29766597472e-09.
  Maximum independent head_residual_rms: 1.67361552829e-12.
  Maximum independent relative_cell_imbalance: 1.72205791615e-09.
confined K-cycle allocation savings: 127998256 bytes (122.069 MiB; 24.12%).
  Previous-control end_to_end_seconds: 0.664502s -> 0.683851s (+2.91%).
  Previous-control solve_only_seconds: 0.297857s -> 0.297016s (-0.28%).
  Three resident cold-head solves reused hierarchy, V/Z and graph with zero captures/rebuilds.

unconfined kcycle:
  Median solver-only 0.291634s; public solve 0.553871s; host construction 0.050763s.
  Incremental live Warp allocation 402565388 bytes (383.916 MiB); sampled device-wide peak 995 MiB.
  Jv [34, 34, 34]; fallback solves [0, 0, 0].
  Actual preconditioner applications [28, 28, 28]; K-cycles [28, 28, 28].
  FGMRES scalar-read wait median 0.239584s; readbacks [210, 210, 210]. CPU wait overlaps device work.
  Maximum across repetitions head_rmse_m: 1.50741355398e-07.
  Maximum across repetitions head_max_m: 2.13807538785e-07.
  Maximum across repetitions flux_relative_l2: 2.5633411556e-09.
  Maximum across repetitions flux_max_m3_day: 1.05760904034e-06.
  Maximum independent physical_rms: 2.83162892169e-09.
  Maximum independent head_residual_rms: 1.43733271409e-12.
  Maximum independent relative_cell_imbalance: 8.67857087695e-10.
unconfined diagonal:
  Median solver-only 28.216151s; public solve 28.423710s; host construction 0.050303s.
  Incremental live Warp allocation 219248348 bytes (209.092 MiB); sampled device-wide peak 843 MiB.
  Jv [33704, 33704, 33704]; fallback solves [0, 0, 0].
  K-cycles 0. Legacy per-preconditioner application counts and scalar wait times are not instrumented (CSV null); no inferred counts are presented.
  Maximum across repetitions head_rmse_m: 4.63760226161e-07.
  Maximum across repetitions head_max_m: 6.59746291376e-07.
  Maximum across repetitions flux_relative_l2: 7.77665759467e-09.
  Maximum across repetitions flux_max_m3_day: 3.53526331764e-06.
  Maximum independent physical_rms: 4.28927750444e-09.
  Maximum independent head_residual_rms: 2.17705433807e-12.
  Maximum independent relative_cell_imbalance: 2.24083846478e-09.
unconfined diagonal_fgmres:
  Median solver-only 173.454524s; public solve 173.647758s; host construction 0.047628s.
  Incremental live Warp allocation 342509868 bytes (326.643 MiB); sampled device-wide peak 650 MiB.
  Jv [30630, 27654, 33693]; fallback solves [0, 0, 0].
  Actual preconditioner applications [30000, 27085, 33000]; K-cycles [0, 0, 0].
  FGMRES scalar-read wait median 101.208933s; readbacks [1494880, 1349238, 1644368]. CPU wait overlaps device work.
  Maximum across repetitions head_rmse_m: 1.58096420399e-06.
  Maximum across repetitions head_max_m: 2.24667445536e-06.
  Maximum across repetitions flux_relative_l2: 2.64847147006e-08.
  Maximum across repetitions flux_max_m3_day: 1.1960668644e-05.
  Maximum independent physical_rms: 1.9074121244e-08.
  Maximum independent head_residual_rms: 9.6811123067e-12.
  Maximum independent relative_cell_imbalance: 9.96126616494e-09.
unconfined K-cycle allocation savings: 127998256 bytes (122.069 MiB; 24.12%).
  Previous-control end_to_end_seconds: 0.668938s -> 0.652275s (-2.49%).
  Previous-control solve_only_seconds: 0.293131s -> 0.291634s (-0.51%).
  Three resident cold-head solves reused hierarchy, V/Z and graph with zero captures/rebuilds.

Interpretation and limitations
-------------------------------

K-cycle solver time and Newton/Krylov work remain essentially unchanged; there is no material solve regression.
Fresh end-to-end time drifts by a few percent in either direction; no numerical settings were adjusted.
The diagonal FGMRES control is numerically accepted but much slower than original GMRES. Its per-dot host synchronization
and orthogonalization work dominate at tens of thousands of Krylov steps. The observed FGMRES-only K-cycle ratios include
this implementation cost; they are not hardware-intrinsic multigrid-only speedups. Individual iteration counts vary across
accepted diagonal-FGMRES repetitions; those values and all timings are retained, not replaced by a deterministic claim.
Device-wide peaks include other GPU processes (gpu_processes.txt recorded two idle 308 MiB allocations). Use the exact live
Warp-pool deltas for solver-owned allocation comparisons. High-water counters are process cumulative, not per-run peaks.
GPU event time and CPU scalar wait overlap; do not sum them as separate independent costs.

Recommended next experimental selection: K-cycle FGMRES, five levels, vertical line omega=0.8, two pre/post sweeps,
20 additional coarsest line sweeps and one cycle/application. Keep the original public diagonal-GMRES default unchanged.

Preserved independent polishing evidence
-----------------------------------------

Scope: existing 4x6x8 layered parity case, separate from the large performance case. No polishing or native run was repeated.
Original returned max/RMS difference from MF6: 1.09102352041646e-07 / 2.96430726717624e-08 m.
Polished max/RMS difference from MF6: 8.71125394041883e-12 / 3.05272514533801e-12 m.
Physical RMS before/after: 2.69962889383359e-09 / 3.9074358993494e-13 m3/day.
Head-equivalent RMS before/after: 1.96854168635946e-10 / 6.81919900770795e-15 m.
Reference head scale 78.9724618714878 m; FP64 epsilon*scale 1.75354090962116e-14 m; spacing 1.4210854715202e-14 m.
Original and polished arrays remain distinct original_cuda/head entries in the unchanged verification_polished.npz archive.
See ../npf662_polishing_evidence.json for source hashes, the original stricter verification criteria and production-criteria evaluations.
Formulation parity is resolved; verification polishing is not substituted for any primary benchmark stopping point.

Changed code: solvers/npf662_newton.py (lazy ownership/selection), npf662_fgmres.py (zero-copy diagonal adapter),
npf662_3d.py (explicit diagnostic option), tests/test_npf662_kcycle.py (switching, linear/nonlinear control and refresh),
working_tests/benchmark_npf662_preconditioner.py (three-way comparison), run_npf662_stabilization_tests.py (broad no-MF6 suite),
record_npf662_polishing.py (saved-data recording). Shared 2D numerical code is unchanged.

Final validation
-----------------

The complete affected-suite run passed 1,444 CPU/CUDA tests, with 50 native-MF6 cases explicitly deselected.
There were zero failures or skipped cases. Two existing Warp/Python structure-layout deprecation warnings remain.
tests.log retains the complete run and every native exclusion. No environment upgrade was performed.
All 347 immutable baseline files passed SHA256 verification after the tests/benchmarks. All source hashes and
saved polishing source hashes match their records. No native MF6 solve or model regeneration was performed.
