Selecting a 3D GPU implementation
=================================

This guide applies to the Cartesian and MF6 6.6.2 NPF/STO implementation
promoted by PR #4, at main revision
``f18912bff7afe5bb696a226cd8095f7a614352bf``. Transformed terrain remains
experimental development on ``dev`` and is outside this release.

Set ``DARCY_FLOAT=float64`` before importing the package. Qualification uses
Python 3.12, warp-lang 1.17.0 and an NVIDIA CUDA device. K, head, source and
time must use consistent units; the tests use m/day, m, m3/day and days.
Cell arrays are ``(nz, ny, nx)``. Convert recharge to a volumetric source by
multiplying by ``dx*dy`` once, in the intended receiving layer.

NPF steady and transient flow
-----------------------------

Construct ``WarpDarcySolver3D`` with
``formulation="mf6_6_6_2_npf", implementation="fast", solver="kcycle"``.
Use ``build_from_K_fields()`` for uniform Cartesian layers or
``build_from_layers()`` for conventional DIS interfaces. The latter takes
strictly decreasing ``z_interfaces`` of shape ``(nz+1, ny, nx)``.

The fast selection measured on the retained layered models is K-cycle
preconditioned flexible GMRES. After building ``model``::

  from DARCY_WARP_PACKAGE import NPF662KCycleControls

  if __name__ == "__main__":
      controls = NPF662KCycleControls(
          levels=5, pre_sweeps=2, post_sweeps=2, coarse_sweeps=20,
          cycles=1, omega=0.8, use_cuda_graph=True,
      )
      solve_options = dict(
          npf_preconditioner="kcycle", npf_kcycle_controls=controls,
          max_outer=80, max_cycles=4000, krylov_restart=48,
      )
      head, info = model.solve(**solve_options)
      if not info["converged"]:
          raise RuntimeError("Physical acceptance failed")

For convertible cells pass ``unconfined=True`` and the intended binary
``npf_cell_type`` array. Uniform convertible models also need a flat
``top_elevation``. Layered models use their supplied interfaces.
For an NPF/STO step add the previous physical state and storage parameters::

  if __name__ == "__main__":
      dt_days = 7.0
      specific_storage = 1.0e-5
      specific_yield = 0.15
      head, info = model.solve(
          transient=True, dt=dt_days, head_prev=previous_head,
          ss=specific_storage, sy=specific_yield,
          unconfined=True, npf_cell_type=cell_types,
          **solve_options,
      )
      if not info["converged"]:
          raise RuntimeError("Timestep failed physical acceptance")
      previous_head = head.copy()

``ss`` has units 1/m and ``sy`` is dimensionless. NPF/STO implements the
default MF6 6.6.2 storage/saturation law with backward Euler time integration.
It does not use the legacy saturated-thickness floor. XT3D, VARIABLECV,
PERCHED, rewetting and alternative STO options remain unsupported.

Diagonal GMRES remains the public default for layered, convertible and
transient NPF. Select it with ``npf_preconditioner="diagonal"``. Diagonal
FGMRES is a diagnostic control selected with
``npf_preconditioner="diagonal", npf_krylov_method="fgmres"``; it is not the
recommended performance choice on the retained large models. Ordinary
GMRES with K-cycle preconditioning is rejected. Uniform confined steady
NPF retains its direct Cartesian dispatch when Krylov controls are omitted.
No default or physical acceptance threshold is changed by this qualification.

For an explicit tolerance experiment, pass
``npf_final_relative_tolerance=1e-7`` to ``model.solve()`` or use the runner's
``--npf-relative-tolerance 1e-7`` option. This changes the relative cellwise
imbalance limit for that call, on both device and independent acceptance.
The physical RMS, head-equivalent and prescribed-head criteria remain
unchanged. Defaults stay at 1e-8, and ``acceptance_override`` records the
requested departure from the default contract. Use a separate output directory
to retain the strict baseline; the same MF6 reference cache remains reusable.

Cartesian steady and transient flow
-----------------------------------

Use ``formulation="legacy", implementation="fast", solver="kcycle"`` for
the FP64 face-array K-cycle. ``implementation="classic"`` remains the
public default and reference alternative. Compare cold starts with cold
starts: a second solve starting from an already converged head measures a
different workload.

Confined transient Cartesian solves use ``transient=True``, ``dt``,
``head_prev`` and ``storage_coeff``. Here ``storage_coeff`` is specific
storage in 1/m; the solver applies volume and timestep. Unconfined legacy
solves retain their existing Picard and secant Sy/Ss controls. Approximate
drying retains the 0.01 m saturated-thickness floor, capped by actual layer
thickness, while storage remains physical and unfloored. Vertical K remains
saturated. These laws differ from matched NPF and are not interchangeable.

Keep ``head_prev`` fixed within a step. Advance it only after convergence.
Use ``update_rhs(rhs=...)``, ``update_initial_head(initial_head=...)`` and
``update_boundary_values(bc_values=...)`` for repeated scenarios. Change
physical conductivity with ``update_K_fields()``. Returned heads own their
storage. Compatible workspaces and graphs are reused; changing the timestep
or storage must refresh storage-dependent preconditioners.

Interpret timing and acceptance
-------------------------------

Check ``info["converged"]`` and physical diagnostics before accepting a
result. By default NPF requires finite fields/aggregates, valid grounded components,
prescribed-head error at most 1e-10 m, head-equivalent RMS at most 1e-6 m,
and either free-cell physical RMS at most 1e-12 m3/day or cellwise L1
imbalance divided by gross throughput at most 1e-8. A signed domain balance
or a small update alone is insufficient.

Report synchronized public solve and end-to-end costs separately from
device iteration time. Public solve includes independent acceptance;
end-to-end also includes input/model setup as identified by each driver.
Warmups are excluded from medians. Every timed cold solve resets the same
physical initial state. GPU event and scalar-read wait times overlap and
cannot be added as disjoint costs. Small models may favour CPU execution.

The retained evidence identifies source revision, runtime, controls, input
hashes and every accepted/failed repeat. Historical MF6 timings are not
fresh CPU benchmarks. See the GPU release notes for measured results and
the distinction between reused and newly generated native references.

Reproduce the qualification
---------------------------

The portable entry point generates deterministic cases and validates cached
MF6 references. It requires no machine-specific baseline directory. From the
repository root in the declared environment::

  python working_tests/run_3d_sanity.py --device cuda:0 --sizes 16 32

The default selects fast Cartesian and the fast NPF device implementation.
NPF automatically selects diagonal GMRES below 50,000 cells and K-cycle
FGMRES at larger sizes. This is a case-size heuristic, not a guarantee of the
fastest method for every geology. Compare the alternatives explicitly::

  python working_tests/run_3d_sanity.py --device cuda:0 --sizes 32 --compare

For realistic NPF transient models, run 20 weekly steps on 312,500 and
1,250,000 cells::

  python working_tests/run_3d_sanity.py --device cuda:0 --families npf \
    --modes transient --sizes 250 500 --variants confined convertible \
    --periods 20 --repeats 3

Use ``--mf6 /path/to/mf6`` for the pinned 6.6.2 executable. MF6 runs only on
cache misses. ``--cached-only`` prohibits regeneration; missing or damaged
references fail explicitly. Each cache fingerprint includes canonical
physical inputs, native solver controls, timestep, source schedule and
storage. Input/result arrays are SHA256 checked on reuse. Cache generation
retains the native deck, head/flux/storage outputs and diagnostics.

``--device auto`` uses CUDA when available and otherwise CPU; results record
the actual device. Each invocation creates a new output directory by default.
``results.json`` retains convergence histories and failed cases;
``summary.csv`` reports timings and errors, and NPZ files retain GPU heads.
Warmups are excluded from medians and every measured history starts from the
same cold physical state. Primary runtime is the sum of synchronized public
solve calls, excluding model build, cache loading, the extra test audit and
file saving. CSV columns separately report model construction, extra audit
and full verification runtime. A nonzero exit code means at least one case failed.

``sanity_3d_controls.py`` reads the actual 2D steady and transient source
functions without executing them, and saves their hashes. Residual controls
are ``rel_tol=5e-7`` / ``abs_tol_min=5e-7``; the head-change RMS is ``1e-4 m``.
Steady checks occur every five cycles and transient checks every cycle.
Steady classic/fast use 2/10 coarse sweeps, respectively. The transient
profile uses one pre/post/coarse sweep. MF6 transient uses the source
``RELATIVE_RCLOSE=1e-6``, outer head ``1e-6`` and inner head ``1e-8`` profile.
The steady confined/unconfined MF6 profiles are read separately.

The 2D head-comparison gates are RMSE at most 1e-3 m and maximum difference
at most 1e-2 m. MF6 references must have finite physical metrics, correct
source allocation and fixed heads, and cellwise imbalance at most 1%.
This cellwise check is stronger than accepting a cancelling domain sum.
The stricter native NPF audit is retained separately: failing that extra audit
does not silently change the native solver controls. GPU NPF still requires
its unchanged, stricter physical acceptance. Its final head-equivalent RMS
is 1e-6 m; Newton/Krylov work units and its 1e-12 / 1e-8 conservation gates
cannot be relabelled as the 2D K-cycle stopping rules. The 2D coarse-cell
limit also has no identical 3D minimum-width parameter.

The 200-cycle 2D budget does not converge every tested Cartesian steady case.
These attempts remain failures with their measured residuals. An explicit
``--legacy-max-cycles`` override is recorded; the runner never silently raises
the budget or loosens tolerances to produce a passing timing.

Run the wider numerical regression suites separately::

  DARCY_FLOAT=float64 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=.:working_tests python working_tests/run_3d_gpu_tests.py

Earlier campaign drivers are retained under
``release_evidence/source_snapshot/working_tests/`` for provenance. Use
``run_3d_sanity.py`` for new portable runs. The regression test runner
explicitly excludes tests that launch MF6 and reports those exclusions.
