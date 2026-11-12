Native 3D general-head boundaries
================================

The Cartesian and ``mf6_6_6_2_npf`` formulations accept native cell-based
general-head boundaries in steady and transient solves. Conductance is an
integrated coefficient in square metres per day, independent of conductivity
and saturation. Positive outward exchange is

.. math::

   q_{GHB} = C(h-h_b), \qquad Q = \operatorname{div}(q)+q_{GHB}+q_{STO}.

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
also rejects a GHB head below a convertible cell's bottom, matching MF6
6.6.2. Arrays are copied. Invalid updates leave the existing model intact.

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
----------------------------

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

The native GHB implementation milestone passed 892 broad 3D/NPF regressions
(four existing skips), followed by 113 focused tests including all new GHB
checks. Realistic GPU/MF6 timing qualification is in progress on ``dev``;
this milestone is not yet the completed ``main`` release.

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
discarded. These are development measurements; full release timings follow
the larger qualification campaign.

The acceleration tests include changing K, boundary conductance/head, zero
conductance, FP64/FP32 preconditioners, exact iteration budgets, invalid
controls and resource release. The focused accelerator/GHB/session selection
passes 97 tests on CPU/CUDA.

Reproducible sanity and runtime checks
-------------------------------------

The existing portable runner now accepts ``--ghb``. It supplies identical
fixed conductance/head arrays to DarcyWarp and native MF6, decodes native GHB
package flows, and checks the scalar law separately from intercell flows and
storage. Saved references include these arrays in their fingerprints. Changed
GHB values cannot reuse a reference for another case. Every measured repeat
must converge and pass the independent head, boundary and budget checks.

For the faster Cartesian iteration and existing NPF runtime controls::

    DARCY_FLOAT=float64 PYTHONPATH=.:working_tests OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      python working_tests/run_3d_sanity.py --sizes 16 32 --ghb \
      --families cartesian npf --variants confined convertible \
      --modes steady transient --cartesian-fgmres --npf-history \
      --npf-controls working_tests/npf_runtime_controls.json --repeats 3

``--gpu-tight`` retains the original 2D MF6 sanity settings while tightening
the GPU: NPF relative cell imbalance ``1e-12`` (physical RMS floor ``1e-12``
and the existing head-equivalent gate); Cartesian relative residual ``1e-14``,
absolute RMS ``1e-12`` and head-equivalent residual ``1e-14``. These settings
are explicit overrides, not changed production defaults. The runner's
Cartesian FGMRES profile uses a three-level FP32 preconditioner, restart 8,
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

The terrain mask repair is separate: ``04a2bca`` validates original values
before integer conversion and passed 851 terrain regressions on CPU/CUDA.
Fractional masks such as 0.9, 1.9 and -0.1 now raise instead of silently
changing boundary ownership.
