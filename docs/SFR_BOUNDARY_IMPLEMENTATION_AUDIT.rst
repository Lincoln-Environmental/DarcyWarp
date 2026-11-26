Boundary implementation audit and test rewrite
==============================================

Audit target
------------

The implementation reviewed was commit
``a2fa91bb3519b4571062a350573730407fd224b7`` on
``feature/sfr-3d-boundaries``. Its main baseline was ``65cba8d``.
The results below use Warp 1.17.0 on CPU and, for native comparisons,
MODFLOW 6.6.2. The test change leaves production equations and tolerances
as they were at the audited revision.

Open findings and repairs
-------------------------

1. **Hydraulic disconnection can cause false convergence.** At the initial
   heads of 10, 9 and 30 m, one of the two faces has zero current conductance.
   Only a weak GHB grounds the first two cells. Their residual and the GHB
   conductance imply heads of approximately 20 and 19 m. The production NPF
   solver accepts the initial heads at zero iterations. The reference shares
   the geometric connectivity assumption. In ``solvers/npf662_newton.py``
   and ``physics/ghb_3d.py``, construct the component restraints from the
   current conducting faces. Refresh them when the wet connections change.
   Apply the same physical criterion independently in the reference auditor.

2. **Building from a single-pass boundary iterable drops the package.**
   A unit drain on the free cell of a two-cell aquifer should lower the head
   from 10 m to 5 m. Passing its records as an iterator leaves the computed
   head at 10 m. In ``warped_darcy_3d.py``, canonicalise the records once
   and pass the owned values through validation and session construction.

3. **Copying a public diagnostic token invalidates the original result.**
   Dropping a shallow copy of a drain token releases the original snapshot,
   so accepted discharge can no longer be retrieved from the live result.
   In ``solvers/accepted_npf_state.py``, bind cleanup to a shared owner
   with an immutable private identifier. Keep the snapshot alive until the
   last owner releases it. A public dictionary edit must never select a
   different snapshot for deletion.

4. **The 2D source ledger overstates fixed-head compensation.** Opposing
   wells of 3.5 units of volume per time in the same prescribed cell give
   gross WEL inflow and outflow of 3.5 each. That cell requires zero net
   compensating CHD exchange. The source ledger reports CHD inflow and
   outflow of 3.5 each. In ``physics/wel.py``, sum wells per fixed cell
   before splitting compensation by sign. Retain the individual entries
   when reporting gross WEL exchange. The distinct-cell control should
   continue to report real compensation in both directions.

5. **Historical WEL verification accepts an altered period duration.**
   Changing a two-day Newton period to 200 days in its public metadata
   still authenticates the WEL result. Compare the public duration of the
   relevant period or substep with its owned duration in
   ``physics/wel.py``, using the same time interval for its volume ledger.

6. **Sparse RIV validation admits a reserved dry marker.** A conducting
   reach accepts a bed elevation of −1e30 in an update. Apply the dense
   finite-head and sentinel rule to active sparse levels in
   ``physics/drn_riv_3d.py``, before replacing the previous package.

7. **A typed boundary entry can silently move to another cell.** A layer
   index of 1.5 becomes layer 1 when a reconstructed ``BoundaryEntryList``
   is validated. Validate the original index type, bounds, and lengths of
   all parallel arrays before conversion or iteration in
   ``physics/drn_riv_3d.py``.

8. **Saving an empty drain package raises an exception.** An explicitly
   empty entry list passes case validation. The archive writer dereferences
   the absent canonical package. In ``validation/npf662_reference.py``,
   handle an empty validated package the same way in the writer, reader and
   digest, so a period with its last drain disabled can be replayed.

9. **A head field of the wrong shape passes qualification.** One returned
   head column can broadcast across three reference columns and earn a
   passing receipt. In ``working_tests/compare_package_qualification.py``,
   verify exact field shapes against the physical domain before subtraction.
   Enforce the corresponding one-dimensional layout on per-entry rates.

10. **The time-integrated exchange gate uses the first entry count for every
    period.** Three entries for one day and one entry for nine days permit
    0.12 m³ at a gate of 0.01 m³/day per entry. The comparator permits
    0.30 m³ and certifies a 0.20 m³ discrepancy. Accumulate the bound as
    the sum of each period's entry count times its duration times the rate
    gate in ``working_tests/compare_package_qualification.py``.

11. **Canterbury boundary precedence depends on dictionary insertion order.**
    Reordering the same inputs changes two prescribed heads from 0 and 4 m
    to 8 and 9 m. In ``canterbury_case_study/canterbury_data_prep.py``,
    iterate the documented source priority explicitly and use it to record
    the selected source in the overlap audit. Preserve the source masks
    that distinguish a valid zero sea stage from a missing value.

What the passing checks establish
---------------------------------

* The 2D mixed solver applies WEL to both C- and Fortran-ordered recharge
  grids. Built and scheduled wells retain the tested single-precision
  response. Nonunit areas distinguish a volumetric WEL input from an
  areal recharge rate. Picard, Newton and FAS histories have separate
  analytical checks on changes in head and accepted volume.
* The 3D Cartesian and NPF fast paths produce analytical strip heads with
  actual multilevel preconditioner applications. A diagonal solve cannot
  satisfy the execution counters. CUDA graph replay has a device-specific
  assertion that requires a CUDA runner. With the checked-in five-level
  profile, the CPU case recorded 21 FGMRES iterations, 21 K-cycle applications
  and six hierarchy refreshes. Maximum absolute head error was 1.57e−10 m.
* Fourteen live parity assertions share five independently built native
  models. They compare heads, per-entry and gross exchanges, flow direction,
  and storage over unequal timesteps. Numerical acceptance, preserved
  historical results, model input handling and the qualification comparator
  have separate tests.

Suite and provenance
--------------------

Repo-root pytest collection at the audit target produced 1,703 cases.
The replacement collects 100 in total. The three modules present in the
initial commit ``ba94383`` retain all four original test methods. The
seven frozen steady unconfined MODFLOW comparisons share their original
test implementation and all seven grids. The three 500 by 500 full-history
comparisons and the 56 original MF6 archives are retained. A fixture
inventory check fails if any of the seven steady grids disappears.

The ``regional_parity`` marker selects the large archived cases for a
high-memory run. Default pytest discovery includes them. The numerical CI
job runs the complementary selection and installs the pinned native
executable. No runtime or speedup threshold determines a correctness
verdict.
