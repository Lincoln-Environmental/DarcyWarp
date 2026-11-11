.. list-table:: GHB follow-up: entire GPU call seconds; medians of three repetitions
   :header-rows: 1

   * - Case
     - Original fast solver
     - Current runtime profile
     - CPU MF6 engine
     - Maximum MF6 head error (m)
   * - Heterogeneous GHB, 100 x 100, 3 weekly steps
     - 0.1339
     - 0.0777
     - 0.386
     - 1.84e-05
   * - Heterogeneous GHB, 500 x 500, 10 weekly steps
     - 1.6656 (*)
     - 0.8171
     - 80.396
     - 4.21e-05
   * - Heterogeneous, 1000 x 1000, 30 weekly steps
     - previous table
     - 6.7306
     - 1276.151
     - 3.09e-05
   * - Uniform, 500 x 500, 52 weekly steps
     - previous table
     - 2.5129
     - 153.291
     - 6.39e-06
   * - Heterogeneous, 100 x 250, 3 weekly steps
     - not measured
     - 0.0474
     - 0.572
     - 6.29e-06

All entries pass every warmup and repeat with strict convergence, matching
timesteps and the independent budget gate. GHB timings use the final MF6
engine run at independently converged conductance; its preceding fixed-point
iterations are reference-generation cost and are excluded from this column.
The no-GHB production heads are identical to the earlier qualified profile.

(*) The large-case original solver uses a 100 m update cap, keeping its
original startup solve and 0.7 relaxation. Its unmodified 10 m cap failed
fixed-step parity and is excluded from the timing comparison.
