3D K-cycle stabilization
========================

Starting revision: dba5db283f22deb50329c4dbb518ef8acd8423d2.

Krylov ownership and diagnostic control
---------------------------------------

NPF662Workspace allocates the shared hydraulic/operator buffers without
allocating a Krylov solver. Its select_krylov method constructs only the
requested implementation. Switching to FGMRES releases TerrainGMRES3D and
its graph/basis arrays. Switching to GMRES closes and releases the flexible
V/Z arrays. Repeated compatible FGMRES solves keep the same V/Z arrays.
The compatible A0 hierarchy remains cached across method switches; a change
in geometry, topology, material conductances, precision or device retains
the existing invalidation/close behavior. No duplicate FP64 operator fields
are introduced by the diagonal-FGMRES adapter.

The three explicit public selections are::

  # Original default, unchanged:
  head, info = model.solve(npf_preconditioner="diagonal")

  # Diagnostic control using the new Krylov method with the old diagonal:
  head, info = model.solve(
      npf_preconditioner="diagonal", npf_krylov_method="fgmres",
  )

  # Existing experimental multigrid implementation:
  head, info = model.solve(npf_preconditioner="kcycle")

All cases use the same nonlinear controller and frozen Jacobian/diagonal
coefficients during each linear solve. K-cycle with ordinary GMRES is
explicitly rejected. Result dictionaries identify the actual method and
preconditioner, expose flexible restarts/application counts, and distinguish
Krylov-buffer reuse from hierarchy reuse. Cycles are zero for diagonal
FGMRES; its preconditioner owns no independent arrays.

Saved polishing evidence
-------------------------

working_tests/record_npf662_polishing.py reads existing arrays only and
records their hashes, comparison errors and residuals in
benchmark_results/npf662_polishing_evidence.json. It performs no solve.
This saved parity evidence concerns the 4x6x8 independently varying layered
geology case, separately from the 250x250x5 performance reference.

The original returned head and polished head remain distinct NPZ entries
original_cuda and head in the existing verification_polished.npz archive.
The maximum/RMS differences from saved MF6 are 1.0910235204164564e-7 /
2.9643072671762388e-8 m before polishing and 8.711253940418828e-12 /
3.052725145338007e-12 m afterward. Physical residual RMS decreases from
2.6996288938335863e-9 to 3.9074358993494044e-13 m3/day; head-equivalent
residual RMS decreases from 1.9685416863594553e-10 to
6.819199007707952e-15 m. Relative cell imbalance decreases from
5.100894678234298e-9 to 7.052542384965323e-13.

The reference head scale is 78.9724618714878 m. FP64 epsilon is
2.220446049250313e-16; epsilon times this head scale is
1.7535409096211606e-14 m and spacing at this magnitude is
1.4210854715202004e-14 m. The saved comparison is near the numerical
roundoff/conditioning scale, rather than an assertion of bitwise equality.
Formulation parity is resolved; the verification-only correction does not
replace any primary returned solution or alter production stopping rules.

Tests and performance evidence
--------------------------------

working_tests/run_npf662_stabilization_tests.py runs the affected 3D, NPF,
terrain, fast-solver, multigrid and validation modules. It explicitly
deselects functions invoking native MF6, lists them in the log, and adds a
process-launch guard. This honors the prohibition on rerunning MF6 while
retaining analytical, independent sparse, saved-data, CPU and CUDA checks.
Shared 2D numerical code is unchanged.

The final full run passed 1,444 CPU/CUDA cases, with 50 native-MF6 cases
explicitly deselected. There were no failures or skipped cases. The two
warnings are existing Warp/Python structure-layout deprecations. The final
log is benchmark_results/npf662_stabilization_001/tests.log.

The three-way saved-model comparison is in
benchmark_results/npf662_stabilization_001/. Its summary.csv and summary.json
retain individual accepted timings, iteration/work counts, reference
comparisons and memory data. REPORT.rst gives the observed medians and
limitations. The driver reads canonical NPZs and MF6 arrays unchanged; no
model generation, native run, source snapshot or new baseline archive is
performed. Warmup is excluded, and each successful configuration has three
warmed repetitions from identical cold heads.

GPU event time and scalar-read wait time overlap and must not be summed.
Device-wide peaks include background processes; the live Warp-pool delta
is the attributable solver allocation measurement. gpu_processes.txt records
the observed background allocations. Existing primary benchmark and
polishing archives are unchanged.

Median end-to-end confined/unconfined times are 18.248/28.522 s for
original diagonal GMRES, 139.960/173.741 s for diagonal FGMRES, and
0.683851/0.652275 s for K-cycle FGMRES. The latter is 26.68x/43.73x faster
than the original method. Its live Warp allocation falls from 505.985 to
383.916 MiB, saving 122.069 MiB (24.13%). K-cycle Newton/Krylov counts remain
5/29 and 6/28. All 18 timed runs and all 32 total runs passed independent
physical acceptance; all 347 immutable baseline files passed final hashes.

Diagonal FGMRES is a diagnostic control, not a recommended performance
default: its host Arnoldi readbacks are expensive and accepted iteration
counts vary across repetitions. Keep diagonal GMRES as the public default;
recommend five-level vertical-line K-cycle FGMRES as the experimental
selection on these saved models. No unresolved numerical regression was
observed.
