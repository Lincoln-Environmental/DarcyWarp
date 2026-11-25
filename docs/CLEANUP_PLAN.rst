DarcyWarp cleanup plan
======================

The starting point is main commit ``49d437e``. The objective is one maintained
implementation of each supported operation, with a readable Python entry point,
an executable verification case and an API contract generated from its source.
The branch is ``refactor/readability-and-sphinx``.

Audit at the starting commit
----------------------------

* The 120 Python files under ``DARCY_WARP_PACKAGE`` contain 52,087 lines and
  1,069 functions and methods, counting nested definitions and Warp kernels.
  Of 916 definitions with parameters other than ``self`` or ``cls``, 40 have a
  matching Sphinx parameter field for every argument. This counts field names,
  not the quality of their descriptions. The baseline HTML manual builds from
  narrative sources. Its configuration does not extract Python docstrings.
* ``warped_darcy.py`` contains 7,706 lines. The separate
  ``solve_transient_unconfined_backend`` in ``solvers/transient_unconfined.py``
  spans 2,506 lines including nested helpers. It interprets controls, manages
  device state, retries timesteps, checks water balance and builds the returned
  histories. The 3D seven-point K-cycle entry point spans 1,126 lines.
* The recharge and transmissivity benchmark drivers repeat worker selection
  and metadata output. The latter also contains a second implementation of
  its MF6 single-case runner behind a broad import exception.
* The stand-alone ``DarcyMultigridKCycle`` prototype has no tracked callers and
  calls an undefined method on its first solve. The public factory dispatches
  to other implementations. Conversely, ``formulation="legacy"`` is the
  current default 3D Cartesian formulation. ``legacy_archive`` is also used
  to reproduce an archived Canterbury boundary policy.
* Most repository bytes are reference data. The package contains separate
  sets of 56 MF6 and 56 finite-difference truth arrays, about 455 MB each
  in raw Git blob sizes. Separate producers make these independent controls.
  Four profiler products account for about 66 MB. Treat code structure and
  data distribution as separate decisions.

Implementation order
--------------------

1. **Remove proven dead paths and repair executable tools.** The first branch
   changes remove the incomplete K-cycle prototype, share the benchmark support
   routines, repair the second-machine table generator and publish the factory
   API in Sphinx. They also remove a 45 kB distribution ZIP after matching
   all 24 members byte for byte with individual files in the same revision. Next, trace the Quickflow mass-balance script's inputs and
   outputs: it writes results and then raises unconditionally. Identify which
   results its consumers still need, port those to the maintained budget
   runner, compare them, and remove the superseded entry point. Inventory
   imports, factory strings, CLI invocations, workflows, examples and saved
   run metadata before removing other candidates.

2. **Separate runtime, experiments and evidence.** Keep model construction,
   physics and solver execution in the importable package. Give retained
   publication, plotting and replay tools explicit inputs and outputs.
   Consolidate duplicated Canterbury map helpers after comparing the rendered
   plots. Eliminate import-time hostname output and filesystem changes in
   ``project_base.py`` after checking its callers and executable discovery.
   For generated profiler files, identify a retrievable release artifact,
   checksum and generating command before removing the in-tree copy. Keep
   the native MF6 and independent FD controls reproducible from a clean
   checkout. No history rewrite is needed.

3. **Untangle one live solver at a time.** Start with 2D transient Picard.
   Extract control validation and result assembly first. Then isolate a single
   timestep attempt, the decision to accept or retry it, and the update of
   accepted head and storage. Preserve numerical operation order, public
   signatures, defaults, units, dtypes, return layouts and ownership of arrays.
   A rejected trial must leave the previous accepted state usable. Apply the
   same process to the 3D seven-point driver. Consolidate host or device
   routines only after comparing coefficients, boundary signs, storage,
   precision and buffer lifetimes. Keep independent numerical reference
   implementations independent.

4. **Make parameter documentation part of each change.** Each retained
   function, including private helpers and kernels, gets a short purpose
   and a ``:param name:`` field for each real parameter. Describe grid shape
   and axis order, physical units, sign, admissible values, optional defaults,
   and in-place writes where they affect the caller. Document meaningful
   return values and raised exceptions. For a device kernel, identify output
   buffers and the launch domain. For a test with fixtures, explain their role.
   Give argument-free functions a purpose without artificial parameter fields.
   Use explicit names and control flow. Replace anonymous lambdas with named
   functions or direct expressions when editing their callers. Record the remaining coverage by
   module and check signature/field agreement as each module is converted.
   Link public API pages from the manual. Use the docstrings as the source of
   the API reference and keep examples and modelling guidance in the manual.

5. **Qualify each removal and move.** Compare the changed entry point on
   the base and branch with identical inputs. Start with analytical head and
   flux solutions and local and whole-domain budgets, then compare matched
   formulations against pinned native MODFLOW 6.6.2. Exercise FP32 and FP64
   in separate processes, including failed updates, state reuse, retries and
   fallback. For GPU path changes, compare eager and captured execution on
   real CUDA hardware, check graph invalidation, peak memory and warmed-up
   runtime on the same device. Use the Canterbury replay and the 500 by 500
   weekly benchmark for changes touching their production paths. Keep
   refactoring and numerical-method changes in separate commits.

Completion
----------

A supported model can be traced from the public factory to its equations and
acceptance checks without passing through obsolete implementations. Every
retired module has been checked against current callers and its supported
use cases have a working route. Compatibility spellings needed by public
inputs or saved results resolve in one place. All retained parameterised
functions have accurate field lists. The tests for each affected backend
pass on its supported devices and precisions, with known baseline failures
and unexecuted cases reported separately. A fresh documentation build succeeds
with the following command and the generated API pages show the parameters.

.. code-block:: bash

   python -m sphinx -b html -n -W --keep-going source build/html
