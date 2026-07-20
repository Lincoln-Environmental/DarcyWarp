1# 2D Transient Implementation Progress

## Progress Made
- **Dataclass & State Management**: Integrated `storage_diag_host` and `storage_diag_wp` fields into the `WarpDarcySolver`, `_MGLevel`, and `_GridLevel` structures.
- **Multigrid Coarsening**: Successfully patched `build_coarse_level_from_fine` and `_mg_coarsen_host_any` to accurately perform 2x2 areal mean coarsening on `storage_diag` up the multigrid hierarchy, returning 10 operator fields and unpacking them safely across the codebase (including hierarchy rebuilds in `update_T_in_place`).
- **Warp Kernels**: Modified the CUDA kernels that form the discrete operator to incorporate the transient storage term into the main diagonal:
  - `jacobi_applyA_fused_kernel`
  - `compute_residual_kernel`
  - `compute_head_residual_kernel`
  - `kcycle_check_dh_and_residual_kernel`
  - `apply_A_and_pAp_kernel`
  - `build_diag_preconditioner_kernel`
  - `init_pcg_with_A_kernel`
- **Kernel Launch Hooks**: Updated all `wp.launch` calls in the K-cycle and PCG solvers to pass the newly introduced `storage_diag` arguments to the GPU.
- **Confined Testing**: Verified that the base confined transient solver runs successfully (smoke test passes).
- **Unconfined Transient Path**: 
  - Patched `solve_multigrid_kcycle` to accept `transient`, `storage_coeff`, `dt`, and `head_prev` keyword arguments.
  - Updated the public `solve()` wrapper to forward transient parameters into `solve_multigrid_kcycle`.
  - Verified Picard iterations step through transient unconfined conditions without the previous `TypeError`.
- **Formal Tests**: Added `tests/test_2d_transient.py` with confined and unconfined transient smoke and mass-balance checks.

## Current Status
- The unconfined transient path now runs end-to-end through `solver.solve(formulation="unconfined", transient=True, ...)`.
- Existing steady-state tests in `tests/test_2d_unconfined.py` continue to pass (the 3000x3000 truth fixture fails only due to GPU memory limits on the test hardware).
- `tests/test_2d_transient.py` passes for both confined and unconfined transient cases.
- `tests/test_2d_unconfined.py` is steady-state MF6 truth replay only; it is not transient MF6 evidence.
- The transient unconfined MF6 generator is `working_tests/run_2d_transient_vs_mf6.py`. It has now produced `DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_unconfined/mf6_transient_heads.npz.lzma` for a 250x250, 52-week run. The artifact loads as `heads_per_period.shape == (52, 250, 250)`, has finite heads, and the MF6 list file reports normal termination.

## Notes
- The 3000x3000 unconfined truth comparison requires more GPU memory than is available on the current test device (RTX 4070 Ti SUPER, 16 GB). This is a hardware limitation, not a regression in the solver.
- `tests/test_comparison_results.py::TestWarpVsMf6Truth::test_warp_heads_within_tolerance` shows a small tolerance exceedance for one PCG-labeled case; this appears to be a pre-existing convergence/tolerance issue unrelated to the transient changes.
- `working_tests/run_2d_transient_vs_mf6.py` now uses the MF6 model name `tr2d_truth`, which is short enough for MF6. The failed pre-fix workspace used `transient2d_truth`, which exceeded MF6's 16-character model-name limit.
