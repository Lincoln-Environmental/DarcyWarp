DarcyWarp main numerical audit
==============================

The repair branch starts at ``f8e7683d13b94f03f1c0f830d807527e4255ce04``.
The original ``origin/dev`` snapshot was
``e5535ddb523c9783089fab7bdd1cd1143046e0cc``. The supplied audit test
reproduced nine failures on main and eight on dev. The existing 2D Newton
DRN/RIV package code and its regressions supplied the main port. Dev retains
its persistent spatial-recharge engine and terrain and cuDSS work.

Numerical acceptance
--------------------

Classic 3D K-cycle and Chebyshev require a head-equivalent row residual at
the initial state and before declaring convergence. The independent final
flow check uses the returned head and equation diagonal. It retains the
existing volumetric tolerance as a cap on the conductance-scaled absolute
allowance and retains the throughput-relative alternative. The head check
applies to both branches. A zero-throughflow equilibrium can therefore
pass at iteration zero. The default 3D NPF solver uses the pinned acceptance tolerances.

Confined 2D recovery applies preconditioned CG to the fine equation with
its assembled storage diagonal and previous-head forcing. High-accuracy
linear convergence cannot compensate for a rejected nonlinear timestep.
The Picard acceptance check recomputes conductances, boundary exchanges
and convertible storage at the candidate head. The domain flow allowance
is the greater of ``0.001 * (IN + OUT) / 2`` and a 32-epsilon bound on
external-flow and storage operands. Established head and update checks
still apply. In the explicitly frozen ``previous_period`` mode, storage
uses the tangent computed at the previous accepted head.

The MODFLOW discrepancy is ``200 * (IN - OUT) / (IN + OUT)`` percent.
For zero inflow and zero outflow it is zero. The fresh replay budgets use
this convention. Dated reports retain their original classification and
provenance. Rejected diagnostic trials have their own period state and
leave the accepted endpoint and clock at the previous accepted period.

The 3D matrix requires a relative L2 intercell-flow difference of at most
0.01, with a 32-epsilon operand allowance for negligible flow. Its matched
state reconstruction uses native DATA-SAT for convertible faces and
requires agreement within the arithmetic allowance at the saved MF6
heads. It requires native source allocation, CHD and GHB package checks
and STO agreement within the existing 1% budget scale. Head agreement and the native IMS use the established tolerances. The large steady convertible
comparison repeats the separately identified original-order ILUT diagnostic
from the previous main qualification.

Canterbury
----------

The archived export has 1,826 observations, RMSE 3.995985 m, mean
simulated-minus-observed residual +1.364346 m and weighted SSE 90,043.49.
The recorded stage-two observational weighted SSE was 91,807.18.
An FP64 replay with the archived boundary masks and fitted
``gh_alpha=2.0`` gives 91,806.98 weighted SSE, 4.018469 m RMSE and
+1.341032 m mean residual. It converges with 32,485 free GHB cells.

The repaired boundary assembly uses source masks when supplied and the
archive's zero-background convention otherwise. The prepared
``inputs/fixed_bcs_100.npz`` supplies separate ``fixed_nth``,
``fixed_sth``, ``fixed_wst`` and ``fixed_sea`` rasters. It has no
separate CHD source masks. The archive's GHB stage and mask are
independent of those rasters. West remains no-flow.
The northern cells ``(58, 834)``, ``(59, 841)``, ``(59, 842)``,
``(59, 843)`` and ``(60, 852)`` hold missing stages in the input
raster. They receive no CHD. The last cell retains its independently
specified zero-stage GHB. The inputs give both southern 1.3 m and sea
0.1 m at ``(625, 455)``. The replay applies sea precedence as an explicit
choice pending confirmation of the intended boundary. Indices are
zero-based, with rows increasing southward.

With the cleaned boundary assembly, the same parameters give observational
weighted SSE 91,805.41, RMSE 4.018468 m and mean residual +1.341037 m.
The weighted regularisation term is 2,732.21 and the water-budget
discrepancy is ``-5.47e-6%``. The near-identical fit provides no basis
for recalibration from this replay. The five northern stages and ownership
of the overlap remain field-data questions.

See the :download:`archived-parameter replay JSON
<../benchmark_results/main_audit_20261125/canterbury_archived_parameter_replay.json>`
for the controls, source coordinates, water budget and archive hashes.

Stochastic calibration exports
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Subsequent fixes retain the winning realised pilot transmissivities for
both Canterbury optimisers. Replaying a new fit or plotting its parameter
summary uses that saved grid, including clipping at the transmissivity
bounds. The archived Canterbury case above has zero perturbation and
can be replayed from its original pilots. An older fit with nonzero
perturbation needs the actual winning realisation for an exact replay.

Qualification
-------------

The pinned suite ran with Warp 1.17.0 and MF6 6.6.2. The workstation has
an Intel Core Ultra 9 285K and a GeForce RTX 4070 Ti SUPER (16 GiB,
NVIDIA driver 595.99.02). The CPU suite ran with CUDA disabled. The CUDA
runs used FP64, with the named mixed Picard profile using FP32 inner
corrections and FP64 outer state and acceptance.

The full main CPU suite passed 976 tests across three selections with
230 CPU-mode skips. The dev integration selection passed 1,700 tests
with 611 skips and ran separately from the large 2D replays.
The main CPU tests covered the 2,000² and 3,000² steady unconfined fixtures and
all 52 periods of each 500² Picard, semismooth Newton and FAS replay.
The CUDA qualification also covered both steady grids, all three
transient backends, the three Picard runtime profiles, all 36 cases at
100 × 100 × 5, and all six 500 × 500 × 5 cases at Kh/Kv=100 with GHB.

The closed analytical 8 × 8 recharge test gave these cumulative balances
in millimetres.

.. list-table::
   :header-rows: 1

   * - Recharge (m/day)
     - Accepted periods
     - Applied
     - Stored
     - Signed error
   * - 0.0000864
     - 30
     - 2.592
     - 2.589468750
     - -0.002531250
   * - 0.0000100
     - 30
     - 0.300
     - 0.299707031
     - -0.000292969
   * - 0.0000864, clipped updates
     - 1
     - 0.0864
     - 0.086362500
     - -0.000037500

The qualification report records measured source revisions, per-case
head and flow errors, package budgets and timing scopes. The 2D steady
references and the three Picard profile reference timings come from
archived fixtures. The 3D native references were regenerated with pinned
MF6. The six large native reference runs shared the CPU. Their engine wall
times include that load. The GPU timings are serial synchronized public
solves with model construction and independent audits timed separately.
For the 500² transient backends, the independent face comparison
reconstructs the harmonic Warp fluxes at the accepted heads and
compares those rates with MF6's saved FLOW-JA-FACE records at each
stress-period endpoint. The archived CUDA qualification lists these endpoints.
Its JSON omits the accepted internal timestep durations. The current replay
also verifies that each named solver accepted exactly one seven-day step
per period and records the durations in its face-flow reports. It compares
active faces with at least one free cell and records the maximum RMS, relative
L2 and individual face errors.

A CPU replay with the historical Newton controls accepted two half-week steps
after one rejection. Starting week 28 from the native day-189 head reproduced
the seven-day FGMRES nonconvergence and both accepted 3.5-day steps.
The fixed-week test disables subdivision and backend
fallback for Newton and FAS. It gives Newton up to 240 Krylov iterations and
sets its residual RMS and head-equivalent RMS tolerances to ``1e-7`` and
``1e-8``, respectively. The FP64 CPU reproduction verified 497,503 faces at
all 52 seven-day steps for each backend. The worst per-period relative L2
face errors were ``1.85726e-6`` for Picard, ``3.24424e-8`` for semismooth
Newton and ``3.24178e-8`` for FAS. Reconstructing flows directly from the
native heads gave ``6.73139e-10`` on the same records. The
:download:`fixed-step CPU reports
<../benchmark_results/main_audit_20261125/followup_fixed_step_cpu.json>`
record the accepted steps and face-flow errors for each backend and week.

The follow-up 4 × 4 test starts every cell at 0.25 m, with a 0.5 m GHB
stage, a 1 m minimum saturated thickness, and a one-day timestep. Both
the Newton and FAS period drivers reproduce the direct solve at 0.375 m
and the analytical GHB inflow of 0.400 m³/day. In a separate 4 × 4 test,
raising the prescribed edge from 5 to 6 m gives 2.076623 m³/day of
free-cell storage. The published storage maps now sum to that budget even
when the boundary and bottom were built from the same input array.

See the :download:`qualification JSON
<../benchmark_results/main_audit_20261125/qualification.json>`
for the numerical results and native reference fingerprints.

Reproduction
------------

Set ``DARCY_FLOAT=float64`` and ``PYTHONPATH=.:working_tests`` before
importing Warp or running the commands. Place MF6 6.6.2 on PATH.
Run the CPU suite with ``DARCYWARP_TEST_CPU_ONLY=1``. Clear that setting
for the CUDA replays.

.. code-block:: bash

   DARCYWARP_TEST_CPU_ONLY=1 python -m pytest tests
   python -m pytest -s tests/test_unconfined_solvers_500x500.py
   python working_tests/benchmark_2d_steady_runtime.py --formulation unconfined \
     --artifact tests/fixtures/unconfined_2d/truth_2000x2000.npz.lzma \
     --output /tmp/darcywarp-audit-steady-2000 --device cuda:0 --repeats 3
   python working_tests/benchmark_2d_steady_runtime.py --formulation unconfined \
     --artifact tests/fixtures/unconfined_2d/truth_3000x3000.npz.lzma \
     --output /tmp/darcywarp-audit-steady-3000 --device cuda:0 --repeats 3
   python working_tests/run_3d_ugly_t_matrix.py --sizes 100 \
     --vertical-anisotropy 1 10 100 --ghb-modes off on --repeats 1 \
     --device cuda:0 --mf6 "$(command -v mf6)" --cache /tmp/darcywarp-audit-mf6-cache \
     --output /tmp/darcywarp-audit-3d-small
   python working_tests/run_3d_ugly_t_matrix.py --sizes 500 \
     --vertical-anisotropy 100 --ghb-modes on --repeats 3 --mf6-convertible-ilut \
     --device cuda:0 --mf6 "$(command -v mf6)" --cache /tmp/darcywarp-audit-mf6-cache \
     --output /tmp/darcywarp-audit-3d-large
   python -m DARCY_WARP_PACKAGE.canterbury_case_study.replay_archived_parameters \
     --output /tmp/darcywarp-canterbury-replay --device cuda:0

The 500 × 500 replay test writes one ``<backend>_face_flows.json`` next to the
replay fixture for each backend. Each file records the signed-flow comparison
over all active faces touching a free cell at all 52 endpoints. The test checks
the native head, grid and budget time records and uses a 1% relative L2 ceiling
at each endpoint. The archived results above report the measured errors. When
the compressed head fixture is present alone, the test generates a fresh matching
native reference in its temporary directory and prints the output paths.

The JSON includes the native adapter and executable fingerprints to
distinguish current references. Use the 3D runner's ``--reference-only``
switch to prepare references before timing the GPU.
