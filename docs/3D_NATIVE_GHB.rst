Native 3D general-head boundaries
==========================================

The Cartesian and ``mf6_6_6_2_npf`` formulations accept native cell-based
general-head boundaries in steady and transient solves. Conductance is an
integrated coefficient in square metres per day, independent of conductivity
and saturation. Positive outward exchange is

.. math::

   q_{GHB} = C(h-h_b), \qquad Q = \operatorname{div}(q)+q_{GHB}+q_{STO}.

MF6 reports the opposite sign for package flow into the aquifer; see the
`USGS MODFLOW 6 groundwater-flow formulation <https://pubs.usgs.gov/tm/06/a55/tm6a55.pdf>`_.

The linear operator receives ``+C`` on its diagonal and ``+C*h_b`` on its
right-hand side. NPF includes the same terms in its physical residual,
Jacobian, safeguards, multigrid hierarchy and grounding checks. Independent
budgets evaluate ``C*(h-h_b)`` directly and keep GHB separate from storage.

Inputs and updates
------------------

``build_from_K_fields``, ``build_from_face_conductance`` and
``build_from_layers`` accept ``ghb_conductance`` and ``ghb_head``. Both may
be full ``(nz, ny, nx)`` arrays or scalars. An optional binary ``ghb_mask``
selects cells when supplying a scalar conductance. Without a mask, positive
conductance selects cells. Zero conductance disables exchange.

GHB cells must be active free cells; overlap with CHD is rejected. Interior
cells are allowed. Values must be finite and conductances nonnegative. NPF
also rejects a GHB head below a convertible cell's bottom at solve time,
matching MF6 6.6.2. Arrays are copied. Field-validation failures during an
update leave the existing model intact.

After construction, use::

    model.update_ghb(ghb_conductance=new_conductance, ghb_head=new_head)

Supply both fields on each update. Conductivity updates do not implicitly
change GHB conductance. Native transient histories hold GHB values fixed
within one call; use separate calls or individual timesteps for changing
boundary schedules. Device diagonals and cached solver graphs are refreshed
when GHB is introduced or changed.

GHB is supplied through the build/update API, not ``solve(use_ghb=True)``.
Exterior face-conductance slots remain no-flow: changing a discarded
exterior slot cannot create a boundary condition.

Budgets and formulation scope
------------------------------------------

``ghb_inflow`` and ``ghb_outflow`` are positive magnitudes; ``ghb_net`` is
positive into the model. Cartesian exposes them in ``info['budget']``;
NPF Newton exposes ``info['ghb_budget']`` and its physical acceptance
metrics. Cartesian's existing aggregate ``boundary_outflow`` remains
negative for compatibility. NPF aggregate outflow is a positive magnitude.
Gross throughput counts inflow and outflow before cancellation.

NPF's donor-saturation and MF6 STO laws remain distinct from legacy
Cartesian Picard's harmonic saturated-transmissivity and secant-storage
laws. Native GHB does not make these different unconfined formulations
identical. Use NPF for the pinned MF6 convertible contract. Confined
Cartesian steady/transient equations are directly comparable to MF6.

Experimental terrain contacts are a separate API on ``dev``. Flat centred
head contacts with conductance obey the scalar GHB law. Sloping or offset
contacts include gradient terms, and contacts are restricted to exterior
subfaces. They are not substitutes for native Cartesian/NPF GHB.

Qualification
-------------

``tests/test_3d_ghb.py`` covers analytical steady/transient exchange,
GHB-only grounding, boundary updates, introduction after a cached no-GHB
solve, input ownership and legacy Picard. ``tests/test_3d_ghb_mf6.py`` runs
live pinned MF6 6.6.2 comparisons of heads, GHB flows and storage, including
convertible NPF histories and confined Cartesian histories. CPU and CUDA
are exercised. Small analytical cases target FP64 roundoff; live comparisons
currently assert an absolute head/flow bound of ``2e-11``. This test bound
is not a claim of bitwise equality between solvers.

The final solver regression selection passed 961 tests on ``dev`` and 798
on ``main`` (which excludes experimental terrain and includes the restored
2D standalone-validator checks). Each run has four expected CPU entries
skipped for CUDA-only checks; their CUDA counterparts ran. The live MF6
tests cover both the original acceptance mode and explicit roundoff mode.
Analytical tests cover dry-start GHB grounding, tiny positive conductance,
gross inflow/outflow cancellation, storage datum rounding and changing
boundary values after graph capture.

Cartesian linear acceleration
-----------------------------

Fast confined Cartesian solves additionally accept
``linear_acceleration='fgmres'``. This retains the Cartesian face operator and
independent physical budget, and shares the tested NPF flexible-GMRES and
vertical-line K-cycle machinery. FP64 defect correction recomputes the true
residual between linear solves. ``max_cycles`` is an exact total Krylov
iteration budget on this path, rather than a count of standalone K-cycles.
``linear_krylov_restart`` controls the basis size; ``linear_kcycle_controls``
accepts ``NPF662KCycleControls``. Defaults use an FP64 preconditioner. An
explicit FP32 preconditioner still keeps heads, residuals and acceptance in
FP64. The existing standalone K-cycle remains the default.

This path fixes the observed tight-accuracy stall of standalone K-cycles on
the heterogeneous 100 x 100 x 5 GHB benchmark. The stalled run exhausted
4,000 cycles at approximately ``3.3e-7`` volumetric RMS. FGMRES reached
``1.8e-13`` in 26 Krylov iterations, with a three-repeat public-solve median
of 0.262 seconds versus the original tight MF6 run's 1.053 seconds. A trial
of coarse-correction energy projection failed to resolve the stall and was
discarded. These development measurements are retained separately from the
larger release qualification below.

The acceleration tests include changing K, boundary conductance/head, zero
conductance, FP64/FP32 preconditioners, exact iteration budgets, invalid
controls and resource release. The focused accelerator/GHB/session selection
passes 97 tests on CPU/CUDA.

Reproducible sanity and runtime checks
----------------------------------------------

The existing portable runner now accepts ``--ghb``. It supplies identical
fixed conductance/head arrays to DarcyWarp and native MF6, decodes native GHB
package flows, and checks the scalar law separately from intercell flows and
storage. Saved references include these arrays in their fingerprints. Changed
GHB values cannot reuse a reference for another case. Every measured repeat
must converge and pass the independent head, boundary and budget checks.

For the faster Cartesian iteration and existing NPF runtime controls::

    DARCY_FLOAT=float64 PYTHONPATH=.:working_tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python working_tests/run_3d_sanity.py --sizes 16 32 --ghb --gpu-tight \
      --families cartesian npf --variants confined convertible \
      --modes steady transient --cartesian-fgmres --npf-history \
      --npf-controls working_tests/npf_ghb_five_level_controls.json --repeats 3

``--gpu-tight`` retains the original 2D MF6 sanity settings while tightening
the GPU: NPF requests relative cell imbalance ``1e-12`` (physical RMS floor
``1e-12``) and explicitly enables the roundoff mode described below;
Cartesian relative residual ``1e-14``,
absolute RMS ``1e-12`` and head-equivalent residual ``1e-14``. These settings
are explicit overrides, not changed production defaults. The runner's
Cartesian FGMRES profile uses a five-level FP32 preconditioner, restart 8,
and FP64 heads/residuals/refinement. The API default preconditioner is FP64.

``--tight`` also replaces MF6's IMS controls and uses stricter head-comparison
gates. It is a numerical investigation mode, not the default timing profile.
The development 100 x 100 x 5 campaign failed two of those tighter comparison
gates despite both engines converging: the largest head difference was
``1.56e-9 m``. Independently evaluated Cartesian GPU head-equivalent residuals
were ``7e-17`` to ``4e-16 m`` versus MF6's ``4e-14`` to ``3e-13 m``. The failed
campaign is retained; it is not presented as machine-precision MF6 parity.
Likewise, an NPF relative imbalance request of ``1e-14`` on the coarse
16 x 16 x 5 layered grid exhausted its outer budget at FP64 roundoff.

Runtime reports distinguish synchronized public solve/history calls, model
construction and external verification. Transient calls start from the
declared initial heads, without a hidden steady-state pre-solve. Timed GPU
results use three repetitions after warmup; the cached MF6 engine time is a
single separately identified native reference run. Source hashes, controls,
all endpoints and failed outcomes are retained with the results.

For the 500 x 500 x 5 release matrix, use five levels for steady and
confined transient solves, and the existing three-level profile for
convertible transients. These commands keep the original MF6 sanity controls::

    export DARCY_FLOAT=float64 PYTHONPATH=.:working_tests
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

    python working_tests/run_3d_sanity.py --sizes 500 --ghb --gpu-tight \
      --families cartesian npf --variants confined convertible --modes steady \
      --cartesian-fgmres --npf-controls working_tests/npf_ghb_five_level_controls.json \
      --periods 20 --repeats 3 --cache benchmark_results/ghb_references \
      --output benchmark_results/ghb_steady

    python working_tests/run_3d_sanity.py --sizes 500 --ghb --gpu-tight \
      --families cartesian npf --variants confined --modes transient \
      --cartesian-fgmres --npf-history \
      --npf-controls working_tests/npf_ghb_five_level_controls.json \
      --periods 20 --repeats 3 --cache benchmark_results/ghb_references \
      --output benchmark_results/ghb_confined_transient

    python working_tests/run_3d_sanity.py --sizes 500 --ghb --gpu-tight \
      --families npf --variants convertible --modes transient --npf-history \
      --npf-controls working_tests/npf_runtime_controls.json \
      --periods 20 --repeats 3 --cache benchmark_results/ghb_references \
      --output benchmark_results/ghb_convertible_transient

Use a new output directory for each invocation. Add ``--cached-only`` to
require an existing validated MF6 reference. A first run needs pinned MF6
6.6.2 and includes reference generation, native-flow decoding and audits;
the console's reference-processing time includes all of that work. The
reported MF6 engine time excludes reference decoding and audit work.
For a small CPU sanity check, select ``--sizes 16 --device cpu --periods 3``.

Five levels reduced large confined steady NPF from 765 to 42 Krylov
iterations. The sixth-level trial did not improve its measured public time.
This is a preconditioning change; the precision and conservation checks
are unchanged. The stronger Sy storage in convertible transients can make
those systems easier to solve than weak-storage confined systems. Compare
each case with its own matching MF6 model, rather than comparing different
formulations' runtimes as if their equations were identical.

Explicit NPF roundoff acceptance
------------------------------------------

``npf_roundoff_acceptance=True`` selects a componentwise backward-residual
test. It is off by default. The portable runner enables it for ``--gpu-tight``
and ``--tight``. Every free cell must satisfy

.. math::

   |r_i| \leq 32\epsilon_{64}\left(|Q_i|
   +\sum_j C_{ij}(|h_i|+|h_j|)+C_i^{GHB}(|h_i|+|h_b|)
   +|q_i^{STO}|+(|S_i|+|S_i^{previous}|)a_i\right),

   a_i=|h_i|+|h_i^{previous}|+
       2\,\mathbf{1}_{convertible}(|b_i|+T_i).

``S`` is the physical storage tangent; it excludes GHB. ``b`` and ``T`` are
cell bottom and thickness. Convertible storage subtracts saturations formed
relative to those elevations, so small absolute heads do not imply small
arithmetic operands. The 32-epsilon allowance scales the residual by the
equation and constitutive-evaluation operands. It is a backward-residual
indicator, not a rigorous forward head-error bound or an exact-arithmetic
nonlinear certificate. Conditioning can amplify head differences.

Acceptance also requires head-equivalent RMS at most ``1e-12 m``, the original
conservation gate (physical RMS at most ``1e-12`` or relative cell imbalance
at most ``1e-8``), finite quantities, prescribed heads and grounded hydraulic
components. A small head-equivalent residual cannot bypass bad mass balance.
Zero rows require exactly zero residual. CUDA evaluates every row and reduces
the maximum allowance ratio; an independent NumPy implementation checks the
reported endpoints outside the benchmark solve timer.

``requested_thresholds_met`` records the original numerical threshold result;
``roundoff_accepted`` records this explicit mode's result. ``precision_limited``
is true when the latter passes while the former does not. The maximum row
ratio is reported as ``roundoff_bound_ratio_max`` and must be at most one.
This distinction matters on the 1.25-million-cell GHB case: a physical RMS
of ``1.15e-12`` narrowly misses a literal ``1e-12`` request despite a
head-equivalent RMS of ``8.1e-16 m``. Its original fixed-threshold run remains
recorded as failed. The roundoff-mode run is a separate experiment.

An initial roundoff-mode trial also failed the convertible transient case:
its storage scale used only absolute current/previous heads. The worst cell
had heads 0.01013/0.00401 m, bottom -15.98994 m and thickness 24.00802 m.
Its residual ``1.31e-13`` exceeded that incomplete allowance by 13.3%.
Including the actual layer-datum operands corrects the scale; a dedicated
CPU/CUDA analytical storage regression protects this case. The original
failed experiment is retained in the evidence.

The runtime profile's ``tiled_reductions=True`` also enables tiled physical
reductions and device Arnoldi updates for steady NPF solves. Steady solves
retain their independent final audit. Defaults remain unchanged; toggling
the option on a cached GHB model is checked against the original endpoint.

The terrain mask repair is separate: ``04a2bca`` validates original values
before integer conversion and passed 851 terrain regressions on CPU/CUDA.
Fractional masks such as 0.9, 1.9 and -0.1 now raise instead of silently
changing boundary ownership.

2026-11-12 release qualification
================================

Fresh main runs at ``d679da3`` used an RTX 4070 Ti SUPER (16 GiB),
Python 3.12 and Warp 1.17.0. Each model has 500 x 500 x 5 cells.
Transient histories contain twenty seven-day steps from the declared
initial heads, with no preliminary steady solve. MF6 6.6.2 results
are saved, validated references for these exact cases and original
2D sanity controls. GPU times are medians of three runs after warmup;
each MF6 time is one native engine run on this host.

.. list-table:: Solver time in seconds; model build and external audits excluded
   :header-rows: 1

   * - Case
     - GPU
     - MF6
     - MF6 / GPU
     - Max head difference (m)
   * - Cartesian confined, steady
     - 0.734
     - 13.915
     - 19.0x
     - 5.187e-05
   * - NPF confined, steady
     - 1.355
     - 13.090
     - 9.7x
     - 6.467e-05
   * - NPF convertible, steady
     - 1.355
     - 291.607
     - 215.3x
     - 7.076e-06
   * - Cartesian confined, 20-step transient
     - 8.680
     - 458.459
     - 52.8x
     - 5.184e-06
   * - NPF confined, 20-step transient
     - 2.633
     - 293.947
     - 111.6x
     - 1.846e-06
   * - NPF convertible, 20-step transient
     - 1.819
     - 142.638
     - 78.4x
     - 5.338e-08

All 252 endpoints, including warmups, passed independent checks.
The portable 16/32-grid runner also passed all twelve steady/transient
cases and 48 additional endpoint audits against fresh MF6 references.
Its CPU run passed all six 16-grid cases and 24 endpoint audits against
the same validated references.
The largest GPU head-equivalent RMS residual was ``8.431e-16 m``.
All 168 NPF endpoint audits passed the componentwise roundoff test;
the largest allowance ratio was ``0.821326``. Of these, 12
used the explicit roundoff alternative while missing the requested
fixed flow thresholds; they are labelled ``precision_limited``.
These are FP64 residual results. MF6 head agreement is tolerance-based,
with the measured differences shown above; bitwise head identity is
not a qualification claim. Small live MF6 regressions also pass their
``2e-11`` absolute head, GHB-flow and storage bounds.

.. list-table:: Separately reported costs in seconds
   :header-rows: 1

   * - Case
     - Model construction
     - Extra test audit
     - Full verified run
   * - Cartesian confined, steady
     - 0.162
     - 0.593
     - 1.502
   * - NPF confined, steady
     - 0.233
     - 0.635
     - 2.234
   * - NPF convertible, steady
     - 0.232
     - 0.617
     - 2.216
   * - Cartesian confined, 20-step transient
     - 0.158
     - 22.915
     - 31.895
   * - NPF confined, 20-step transient
     - 0.237
     - 23.501
     - 26.421
   * - NPF convertible, 20-step transient
     - 0.237
     - 24.568
     - 26.701

Internal solver setup and production acceptance remain inside the
public solve timer. Cartesian time sums the individual public solves;
source-update preparation is outside those calls. NPF transient time
covers the entire public history call, including source updates and
the final history download. The external audit is test work and is
excluded from both primary runtime comparisons. Full verified time
also includes orchestration; medians of components need not sum.

Compact evidence retains every endpoint audit, the exact source hashes,
controls, MF6 build identity and input/output fingerprints. Large raw
head and flow files can be regenerated using the commands above.

* :download:`Main qualification <../benchmark_results/ghb_3d/main_qualification.json>`
* :download:`Complete timing CSV <../benchmark_results/ghb_3d/main_summary.csv>`
* :download:`Development qualification and tuning <../benchmark_results/ghb_3d/dev_qualification.json>`
* :download:`Retained failed experiments <../benchmark_results/ghb_3d/development_experiments.json>`

Release changes were selectively promoted: native GHB ``3496a50`` to
``561e92d``; Cartesian FGMRES ``74ddc6a`` to ``2ac1c32``; final
roundoff/storage and device-reduction changes ``2ab046d`` to
``bc93bd4``; qualified profiles ``5097ead`` to ``d679da3``. Main
also restores the missing 2D standalone GHB validator in ``057f439``.
The experimental terrain mask repair remains on dev as ``04a2bca``.

Direct MF6 parity audit, 13 November 2026
------------------------------------------------------------

The preferred solution comparison target is a maximum head difference of
``1e-6 m``. The measured large-model envelope of ``6.467e-5 m`` is accepted
for the current qualification, but is not labelled as meeting that preferred
target. The six 500 x 500 x 5 cases give:

.. list-table:: Maximum difference over every saved repetition and timestep
   :header-rows: 1

   * - Case
     - Maximum head difference (m)
     - Meets 1e-6 m
   * - Cartesian confined steady
     - 5.187e-5
     - No
   * - Cartesian confined transient
     - 5.184e-6
     - No
   * - NPF confined steady
     - 6.467e-5
     - No
   * - NPF confined transient
     - 1.846e-6
     - No
   * - NPF convertible steady
     - 7.076e-6
     - No
   * - NPF convertible transient
     - 5.338e-8
     - Yes

The original MF6 profiles are unchanged. Confined steady uses the 2D
sanity profile's ``outer_dvclose=inner_dvclose=1e-4``; convertible steady
and all transient cases use ``outer_dvclose=1e-6`` and
``inner_dvclose=1e-8``. All retain ``RELATIVE_RCLOSE=1e-6``. A stopping
criterion is not a bound on the maximum difference between two independently
computed solutions. No production solver controls changed in this audit;
the runtime table above continues to describe the qualified implementation.

The new ``verify_3d_ghb_mf6_parity.py`` auditor evaluates the production
operator at each saved MF6 head, using the saved MF6 previous head for
storage. Zero iteration budgets prevent the evaluation from advancing the
state. Reference arrays, input fingerprints, timestep grids and saved
benchmark metadata are checked before comparison. It reads all 63 distinct
large-model endpoints and compares solution heads from all four histories
per case, including the warmup.

Every confined Cartesian and NPF face, GHB, storage and free-cell residual
comparison passes the explicit FP64 arithmetic allowance. GHB and STO also
pass at every convertible endpoint. The allowance is ``32*eps*scale`` per
entry with no absolute floor; ``scale`` includes the evaluated equation
operands and, for convertible saturation/storage, the layer datum. This
measures arithmetic consistency, not maximum forward error in heads.
The Cartesian face comparison reconstructs flux from its actual device
coefficients; its full row comparison reads the production device residual.
NPF face, storage and residual arrays come directly from production kernels.

Convertible MF6 output requires an additional distinction. The saved
``DATA-SAT`` values differ from saturation recalculated at final heads by
up to ``2.136e-8`` in steady flow and ``3.889e-8`` in the transient history.
The pinned source's ``npf_cf`` updates saturation during coefficient
formation, while ``npf_cq``/``sgwf_npf_qcalc`` use that stored saturation
when reporting flow. Consequently, comparing its reported budget directly
with a fully refreshed final-head operator fails a roundoff-level test:
the largest face differences are ``2.548e-6`` and ``7.979e-7 m3/day``.

Using MF6's reported saturation reproduces every native face flow within
the arithmetic allowance, with maximum differences of ``1.137e-13`` and
``1.421e-14 m3/day``. This diagnostic is a host reconstruction, separately
labelled in the report; it does not replace the failed final-head check.
Three additional native all-CHD probes prescribe MF6's own saved steady,
first-transient and last-transient heads. They retain the original IMS
settings and compare the production NPF flow law at exactly those states.
All three pass, with maximum face differences of ``1.137e-13``,
``1.421e-14`` and ``7.105e-15 m3/day``. These are constitutive probes,
not new solutions or benchmark timings. MF6 disables GHB/STO at CHD cells,
so those packages are checked on the original free-cell histories instead.

An already-saved, separately labelled 100 x 100 x 5 experiment with tighter
MF6 controls provides an additional cross-check: all six steady/transient
cases meet the ``1e-6 m`` target, with maximum differences from ``9.082e-13``
to ``1.561e-9 m``. Its original tighter settings are recorded in its report.
No MF6 settings were changed or rerun for that comparison, and its results
do not replace the original large-model timing or accuracy evidence.

Reproduce the cached audit with::

    DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python \
      working_tests/verify_3d_ghb_mf6_parity.py \
      --qualification benchmark_results/ghb_3d/main_qualification.json \
      --cache benchmark_results/ghb_3d/mf6_sanity_cache \
      --head-tolerance 1e-6 --device cuda:0

For portable runs, first generate a local ``results.json`` with
``run_3d_sanity.py --ghb`` and pass it as ``--qualification``. Use
``--raw-root`` if relocating the campaign directories containing saved
DarcyWarp heads. Add ``--prescribed-flow-probes NEW_DIRECTORY`` to run the
separate first/last convertible flow-law probes, with ``--mf6`` pointing
to the pinned executable. Without that option the command only reads MF6
artifacts. The JSON retains separate ``operator_parity`` and
``solution_parity`` flags: the former compares the reported native budgets
with the refreshed final-head operator; the latter tests the requested
head target. The strict command exits with status 1 on the current large
baseline, retaining both the preferred-target misses and saturation-state
differences rather than treating them as passes.

The affected regression selection passes 107 tests on CPU/CUDA, including
the new native-state tests and deliberate face-order, GHB-sign, storage-sign,
nonfinite-data, cache-hash and timestep corruptions. The qualification
establishes the tested flow/storage laws at matched states and records the
remaining solution tolerance differences; it does not claim bitwise MF6
head identity.

* :download:`Large-model direct parity audit <../benchmark_results/ghb_3d/mf6_parity_large_final.json>`
* :download:`Existing tighter-reference diagnostic <../benchmark_results/ghb_3d/mf6_parity_existing_tighter_reference.json>`

Pinned source: ``validation_artifacts/terrain_2026-10-16/mf6_source/gwf-npf.f90``
(``6.6.2``, official source commit ``7cb7958940186ecfdcd3ea9e6a563cc4609581c9``).
