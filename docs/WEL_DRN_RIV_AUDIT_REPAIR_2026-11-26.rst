WEL/DRN/RIV audit repair: implementation and qualification status
===============================================================

This is an intermediate status report, not an SFR readiness certificate.
All work remains on ``feature/sfr-3d-boundaries``. Main at
``65cba8df6154bcdfd8eb73c1b20a2623a744d73c`` was already integrated when the
campaign began. No SFR implementation or main merge is included.

Implemented corrections
----------------------

* Raw volumetric sources retain independent ownership when WEL is introduced,
  replaced or removed. Recharge keeps the established main convention.
* Previous-state K-cycle restraint assembles authoritative CSR DRN/RIV entry
  derivatives together with GHB and storage; individual switching thresholds
  and duplicate entries are preserved. Dense boundary device copies were removed.
* Activation uses physical component balance, including disconnected river
  leakage. Cell and entry indexing are distinct; recovery can run during Newton.
  Zero current storage tangents are distinguished from physical storage parameters.
* Resident histories freeze previous accepted physical heads before upcoming
  forcing, storage and activation guesses are prepared.
* Owned accepted equation and source objects bind model/session identity,
  period-specific inputs, physical previous heads, timestep and accepted outputs.
  Mutated or unrelated heads, entry rates and cell rates are rejected. Historical
  audits retain their actual equation after later model updates or forcing restore.
* Gross source and boundary throughput preserves opposing package entries.
  Prescribed-head exchange absorbs prescribed source contributions.
* Maintained 2D Picard, Newton and mixed paths retain independent WEL/recharge
  schedules and accepted source/volume budgets. Scheduled well updates now avoid
  a redundant recharge upload before the optimized uniform device update.
* Raw well coordinates, scalar rates, destination representability, accumulated
  overflow and recharge updates are validated before state mutation.
* The weak common-head regression again uses its independent fixed head target.
  A confined external component-balance correction resolves the constant mode;
  the final CUDA analytical head error is zero at stored precision in that case.
* NPF Newton defaults to K-cycle-preconditioned FGMRES. Explicit diagonal remains
  available. Prescribed sources alone retain direct confined fast execution.

Validation boundaries
---------------------

The targeted initial audit selection reproduced 16 failures out of 17 tests.
On ``191d3de`` the complete maintained suite finished with **1607 passed,
21 skipped**, including actual CUDA and pinned live MF6 execution, in 1020.62 s.
Precision-specific follow-up: **17 passed** under FP32. After the final scheduled
source-upload optimization, affected CPU/CUDA tests: **21 passed**. A new complete
suite against the final implementation commit remains required. The strict HTML
Sphinx build passed before this status document was added.

The previous default-change runs are archived separately and contain 18
backend-contract failures. They are not final passing results. Explicit diagonal
reference tests now request diagonal, and default tests assert FGMRES/K-cycle
and executed multigrid cycles. No physical tolerances were relaxed for that change.

Measured production results so far
---------------------------------

All figures below are synchronized public medians of three warm measurements,
with identical original initial heads for steady runs. Calibration uses the
previous accepted head. Construction, first execution, updates and extraction
are recorded separately in the JSON evidence. Numerical runtime is separate.

.. list-table:: Regional CUDA measurements, seconds
   :header-rows: 1

   * - Execution
     - Approximately 100k cells
     - Approximately 1M cells
   * - NPF saturated FP64 K-cycle / FGMRES, packages
     - 0.4055
     - 2.0332
   * - NPF current FP64 K-cycle / FGMRES, packages
     - 0.1955
     - 1.8735
   * - NPF current FP32 K-cycle / FP64 FGMRES, packages
     - 0.2120
     - 1.8874
   * - Direct Cartesian linear FGMRES / K-cycle, WEL
     - 0.0856
     - 0.3807
   * - Resident NPF/STO, 20 weekly steps
     - 1.1953
     - 2.4541
   * - Device-native 2D Picard, 20 weekly steps
     - 0.8314
     - 6.7977

Direct linear FGMRES is the existing confined fast pathway, not Newton.
Actual preconditioner applications and CUDA graph execution are asserted.
All resident periods pass physical acceptance; one output-history transfer is
reported. Raw-source controls and WEL equivalents retain the same Krylov work.

Pinned MF6 6.6.2, 100k-cell package models: the steady sequence plus three forcing
updates has a native engine median of 55.1605 s (range 54.4937--58.6625). The
20-step transient median is 322.5929 s (309.2861--332.4043). The maximum accepted
GPU/MF6 head difference is 3.383e-7 m across the steady and transient endpoints;
maximum DRN/RIV entry-rate difference is 3.254e-6 m3/day. Independent confined
storage, calculated directly from consecutive accepted heads, differs from MF6
by at most 6.953e-9 m3/day per cell; the total history accumulation difference is
1.029e-6 m3. Native convergence and time grids are checked. Large native reference
runs remain in progress; their missing comparisons are not described as passes.

Unresolved qualification and preserved failures
----------------------------------------------

The original direct Cartesian Chebyshev controls reached their 200-cycle limit
on the new heterogeneous regional field. Existing direct linear FGMRES/K-cycle
passes the same physical models. The raw failure records remain in the matrix.

The 2D confined stress field exposed sensitivity to Chebyshev spectrum and
coarse controls. Using the existing mixed profile's [0.1, 2.0] spectrum resolves
the 100k-cell FP64 steady/transient diagnostic without changing acceptance gates.
The million-cell mixed Chebyshev default diverges on this field, with or without
wells. Deeper coarsening alone does not repair it. The existing Jacobi alternative
converges in 13 refinements; full repeated qualification remains to be completed.
These are measured configuration limitations, not changed test assertions.

The short-restart (8) diagonal reference completed the 100k cases, but its
million-cell raw-source cold solve returned no accepted result after 611.77 s
and was interrupted. It has no valid million-cell timing. The interruption is
archived explicitly. At the user's request, no further diagonal solves will run.
The previous 179-second result used different forcing/controls and cannot support
a direct speedup ratio against the new regional timing table.

Remaining work includes final confined/mixed configuration qualification,
additional smoothing/coarsening/graph comparisons, completion of native regional
references and per-package/volume comparisons, a final full regression run,
and the completed audit report. SFR and a main merge remain outside this task.

Reproduction and evidence
-------------------------

Run in the project environment with ``PYTHONPATH=.:working_tests``:

* ``working_tests/qualify_package_kcycle.py --configuration fast``
* ``working_tests/qualify_package_2d.py --mode steady|mixed|transient|picard``
* ``working_tests/qualify_package_mf6.py``
* ``working_tests/qualify_package_2d_mf6.py``
* ``working_tests/compare_package_qualification.py --help``

Use ``DARCY_FLOAT=float32`` for the mixed process and FP64 otherwise. Benchmark
physical controls are in ``wel_regional_qualification_controls.json``; NPF
controls originate in the previously qualified ``npf_ghb_five_level_controls.json``.
The separately labelled ILUT native diagnostic retains all original tolerances.
Compact evidence is under ``benchmark_results/package_audit_repair``. Full raw
head arrays and native workspaces stay local rather than entering Git history.
