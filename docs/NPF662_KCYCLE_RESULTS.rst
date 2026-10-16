Persistent 3D K-cycle Newton implementation
==========================================

Implemented locally from dev 5a335e91a3252cd4b916d70135e1e4036730b6d7.
The MF6-compatible governing residual, Jv, nonlinear line search, fallback
order, shared Krylov budgets and final acceptance criteria are unchanged.
The default preconditioner remains diagonal.

Select the alternative explicitly on an existing NPF model::

  from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls

  controls = NPF662KCycleControls(
      levels=5, pre_sweeps=2, post_sweeps=2,
      coarse_sweeps=20, cycles=1, omega=0.8,
  )
  head, info = model.solve(
      npf_preconditioner="kcycle",
      npf_kcycle_controls=controls,
      unconfined=unconfined,
      npf_cell_type=cell_types,
      max_outer=80, max_cycles=4000, krylov_restart=48,
  )

npf662_kcycle.py owns the frozen full-saturated hierarchy and zero-boundary
correction buffers. It reuses fast_3d.py hierarchy construction and device
energy-correction kernels, together with the validated vertical-line
relaxation kernel. The fixed-work graph contains the complete recursive
K-cycle; no convergence checks or host scalar reads occur inside it.
The scaled Krylov vector v is converted to physical RHS D0*v exactly once.
Its A0 correction is already in head units.

npf662_fgmres.py owns persistent V and Z vectors and reconstructs the
correction from Z. It uses masked FP64 products, reorthogonalized Arnoldi,
true restart/final residuals, explicit breakdown checks and exact shortened
budgets. Small Hessenberg arrays and scalar products use the host; grid
vectors stay on the device. CUDA events measure multigrid application
without adding another synchronization. CPU execution uses the same
numerics with eager kernel launches.

The NPF model's existing workspace owns both objects. Value snapshots
detect conductance/topology/fixed-mask changes. Source, initial-head and
prescribed-value refreshes preserve the hierarchy. Rebuilding layered
geometry or material fields closes the previous session and resources.
Device/restart/work-control incompatibilities recreate required state.
Repeated close is idempotent.

Observed local CUDA comparison
------------------------------

NVIDIA RTX 4070 Ti SUPER, Warp 1.17.0, FP64. Each median includes three
successful fresh-model runs with warmed kernels and identical cold heads.
End-to-end includes saved NPZ load, model construction, hierarchy/workspace,
solve and the public independent physical acceptance check.

========== ============= ============ ======== ============== ==============
Model      Diagonal time K-cycle time Speedup  Newton D / K   Krylov D / K
========== ============= ============ ======== ============== ==============
Confined   18.2144 s     0.6645 s     27.41x   7 / 5          21000 / 29
Unconfined 28.5041 s     0.6689 s     42.61x   11 / 6         33000 / 28
========== ============= ============ ======== ============== ==============

Both solutions pass unchanged independent physical acceptance and agree
with the saved MF6 heads and face flows. The new solves each use 34 Jv
evaluations, zero fallback work and zero Krylov restarts. The five-level
hierarchy remains nz=5 and coarsens 250,125,63,32,16 horizontally.

Hierarchy construction takes about 0.052/0.053 s (7.8/7.9% of end-to-end).
Device multigrid application takes 0.236/0.230 s (35.6/34.3%). FGMRES scalar
read waits take 0.249/0.240 s (37.5/35.9%), overlapping the GPU work; these
fractions must not be summed. Three resident cold-head repeats per variant
reuse the hierarchy and graph without rebuilding. Five levels, omega=0.8,
2/2 pre/post sweeps, 20 coarsest sweeps and one cycle are recommended for
the next experimental stage; no depth sweep was needed.

Complete compact results, individual measurements and limitations are in
benchmark_results/npf662_kcycle_002/REPORT.rst, summary.csv, summary.json
and histories.json. The immutable baseline remains in baselines/ and is
not copied into the results. The saved large MF6 model was not rerun.

Reproduce from the repository root after choosing a NEW output directory
in the driver's exact execution guard::

  DARCY_FLOAT=float64 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=.:working_tests \
    /home/patrickdurney/miniconda3/envs/darcywarp/bin/python \
    working_tests/benchmark_npf662_preconditioner.py

75 explicit CPU/CUDA regressions pass. An earlier broad regression
selection accidentally ran three existing tiny MF6 checks; the final
explicit selection invokes none. See the compact results report for details.
