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

The terrain mask repair is separate: ``04a2bca`` validates original values
before integer conversion and passed 851 terrain regressions on CPU/CUDA.
Fractional masks such as 0.9, 1.9 and -0.1 now raise instead of silently
changing boundary ownership.
