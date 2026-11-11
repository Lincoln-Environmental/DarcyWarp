.. list-table:: Fresh main timings: complete solve/history seconds
   :header-rows: 1

   * - Case
     - Previous fast main
     - Promoted fast main
     - Saved CPU MF6
     - Maximum MF6 head error (m)
   * - Unconfined heterogeneous, 1000 x 1000, 30 weekly steps
     - 25.209
     - 6.055
     - 1276.151
     - 3.09e-05
   * - Unconfined uniform, 500 x 500, 52 weekly steps
     - 5.898
     - 2.028
     - 153.291
     - 6.39e-06
   * - Unconfined heterogeneous GHB, 100 x 100, 3 weekly steps
     - 0.131
     - 0.079
     - 0.386
     - 1.84e-05
   * - Unconfined heterogeneous GHB, 500 x 500, 10 weekly steps
     - 1.652 (*)
     - 0.784
     - 80.396
     - 4.21e-05
   * - Unconfined heterogeneous, 100 x 250, 3 weekly steps
     - Not rerun
     - 0.045
     - 0.572
     - 6.29e-06
   * - Steady unconfined, 1000 x 1000
     - 2.334
     - 2.319
     - 56.438
     - 4.46e-06
   * - Steady confined heterogeneous GHB, 1000 x 1000
     - 0.343
     - 0.332
     - 17.370
     - 2.61e-05

Every warmup and repeat passes its convergence and MF6 head gates.
All transient histories additionally pass matching-timestep and independent
water-budget gates; worst absolute period discrepancy is 4.53e-05%.
All promoted transient cases use the explicit runtime profile, with strict
acceptance, zero timestep retries and zero startup-solve cycles.

(*) The original large GHB solver uses the parity-qualified 100 m update
cap, with its original startup solve and 0.7 relaxation. Its unmodified
defaults fail fixed-step parity and are excluded as a speed comparator.
