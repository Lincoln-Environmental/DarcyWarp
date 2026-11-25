import time
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import gc

from DARCY_WARP_PACKAGE.model_builder import (
    _build_domain,
    _build_dem,
    compare_head_fields
)

from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model
from DARCY_WARP_PACKAGE.validation.steady_truth_2d import (
    load_cached_mf6_truth, load_or_run_mf6_truth, save_mf6_truth, steady_transmissivity,
)

from DARCY_WARP_PACKAGE.CPU_FD import run_fd_truth_forward
from DARCY_WARP_PACKAGE.project_base import data_store

# DARCY_FLOAT must be pinned before warped_darcy/config are imported.  The
# production mixed-precision path needs a float32-built hierarchy, so the
# __main__ switch below relaunches this script once with DARCY_MIXED_FP32=1.
os.environ["DARCY_FLOAT"] = (
    "float32" if os.environ.get("DARCY_MIXED_FP32") == "1" else "float64"
)
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver as wds
from DARCY_WARP_PACKAGE.warped_darcy import compute_mass_balance_budget
import warp as wp
from DARCY_WARP_PACKAGE.sanity_case_config import (
    SPATIAL_GRID_CASES,
    DEFAULT_DX,
    DEFAULT_R_TRUTH,
    DEFAULT_THICKNESS,
    DEFAULT_GHB,
    DEFAULT_ISOTROPIC,
    DEFAULT_T_SEED,
    DEFAULT_ISOTROPIC_T,
)

# Optional GHB helper: use if available
try:
    from DARCY_WARP_PACKAGE.model_builder import _build_ghb_boundary_mask
except ImportError:
    _build_ghb_boundary_mask = None


def run_solve(solver, check_every_no: int = 5):
    print(
        "Both timed solves start from the same DEM input; "
        "the second solve only reuses CUDA runtime state."
    )
    kcycle_impl = os.environ.get("DARCY_KCYCLE_IMPL", "classic").strip().lower()
    if kcycle_impl == "mixed":
        return run_solve_mixed(solver)
    nu_coarse = 10 if kcycle_impl == "fast" else 2
    solve_kwargs = dict(
        max_cycles=200,
        nu_pre=2,
        nu_post=2,
        nu_coarse=nu_coarse,
        omega=0.7,
        rel_tol=5.0e-7,
        abs_tol_min=5.0e-7,
        initial_head=dem,
        return_info=True,
        max_levels=6,
        check_every_no=int(check_every_no),
    )
    if kcycle_impl != "classic":
        solve_kwargs["implementation"] = kcycle_impl
    t0 = time.perf_counter()
    with wp.ScopedTimer("kcycle_solve", use_nvtx=False):
        head1, info1 = solver.solve_multigrid_kcycle(**solve_kwargs)
    t1 = time.perf_counter()
    print("CUDA cold-runtime solve time:", t1 - t0)

    t2 = time.perf_counter()
    with wp.ScopedTimer("kcycle_solve", use_nvtx=False):
        heads, info = solver.solve_multigrid_kcycle(**solve_kwargs)
    t3 = time.perf_counter()
    print("CUDA warm-runtime solve time:", t3 - t2)
    print(info)
    cold_runtime_seconds = t1-t0
    warm_runtime_seconds = t3-t2
    return heads, info, cold_runtime_seconds, warm_runtime_seconds


def run_solve_mixed(solver):
    """Production mixed-precision solve (FP64 master + FP32 fast correction).

    Requires the model to have been built under DARCY_FLOAT=float32 (the
    __main__ switch handles the relaunch).  Uses the validated
    MixedFastConfig defaults (k=5 inner cycles, nu_coarse=10).
    """
    from DARCY_WARP_PACKAGE.solvers.mixed_fast import (
        MixedFastConfig,
        solve_mixed_fast,
    )

    bc_values = np.asarray(solver.bc_values_host, dtype=np.float64)
    gh_head = (
        np.asarray(solver.gh_head_host, dtype=np.float64)
        if getattr(solver, "use_ghb", False)
        else None
    )
    R_field = np.asarray(solver.R_field_host, dtype=np.float64)
    config = MixedFastConfig()

    t0 = time.perf_counter()
    with wp.ScopedTimer("mixed_fast_solve", use_nvtx=False):
        head1, info1 = solve_mixed_fast(
            solver,
            dem,
            bc_values_f64=bc_values,
            gh_head_f64=gh_head,
            R_f64=R_field,
            config=config,
        )
    t1 = time.perf_counter()
    print("CUDA cold-runtime solve time:", t1 - t0)

    t2 = time.perf_counter()
    with wp.ScopedTimer("mixed_fast_solve", use_nvtx=False):
        heads, info = solve_mixed_fast(
            solver,
            dem,
            bc_values_f64=bc_values,
            gh_head_f64=gh_head,
            R_f64=R_field,
            config=config,
        )
    t3 = time.perf_counter()
    print("CUDA warm-runtime solve time:", t3 - t2)
    print(info)
    cold_runtime_seconds = t1 - t0
    warm_runtime_seconds = t3 - t2
    return heads, info, cold_runtime_seconds, warm_runtime_seconds

def solve_callable():
    return run_solve(solver=single_solver)

def _mb(nbytes: int) -> float:
    return float(nbytes) / (1024.0 * 1024.0)

def log_pool(tag: str, device: str = "cuda:0") -> None:
    used_cur = wp.get_mempool_used_mem_current(device)
    used_hi = wp.get_mempool_used_mem_high(device)
    thr = wp.get_mempool_release_threshold(device)
    print(
        f"[{tag}] mempool used_cur={_mb(used_cur):.1f} MiB, "
        f"used_hi={_mb(used_hi):.1f} MiB, "
        f"release_threshold={thr}"
    )


def build_mf6_truth_path(
    truth_dir: Path,
    label: str,
    ghb: bool,
    isotropic: bool,
) -> Path:
    """Return the canonical saved MF6 truth path for one grid/configuration."""
    filename = (
        f"mf6_truth_{label}_ghb_{bool(ghb)}_"
        f"t_isotropic_{bool(isotropic)}.npz"
    )
    return truth_dir.joinpath(filename)


if __name__ == "__main__":
    ghb = bool(DEFAULT_GHB)
    isotropic = bool(DEFAULT_ISOTROPIC)

    # --- production solver implementation switches ---
    # Fast FP64 K-cycle: face arrays + block-reduced reductions + graphed
    # cycles (implementation="fast").  See MIXED_PRECISION_CAMPAIGN.md.
    use_fast_fp64 = True
    # Production mixed precision: FP64 master head + FP32 fast correction
    # (solvers/mixed_fast.py).  Overrides use_fast_fp64.  Needs a float32
    # model hierarchy, so this script relaunches itself once with
    # DARCY_FLOAT=float32 pinned. Select it by flipping this switch locally.
    use_mixed_precision_fp32 = True
    # Run the CPU FD (numpy) reference solve and its comparisons.  Disable
    # to skip the slow host solve on large grids.
    run_fd_reference = False

    if use_mixed_precision_fp32:
        if os.environ.get("DARCY_MIXED_FP32") != "1":
            env = dict(os.environ, DARCY_MIXED_FP32="1")
            raise SystemExit(
                subprocess.call(
                    [sys.executable, os.path.abspath(__file__)], env=env
                )
            )
        os.environ["DARCY_KCYCLE_IMPL"] = "mixed"
    elif use_fast_fp64:
        os.environ["DARCY_KCYCLE_IMPL"] = "fast"

    # Grid cases to benchmark: the complete shared spatial registry, in
    # catalog order (comment an entry out of SPATIAL_GRID_CASES to skip it).
    grid_cases = SPATIAL_GRID_CASES

    dx_truth = float(DEFAULT_DX)
    R_truth = float(DEFAULT_R_TRUTH)
    thickness = float(DEFAULT_THICKNESS)
    width = dx_truth
    t_isotropic_value = float(DEFAULT_ISOTROPIC_T)
    truth_dir = data_store.joinpath("mf6_truth_npz")
    truth_output_dtype = np.dtype(np.float32)
    convergence_check_interval = 5

    all_results = {}

    for label, cfg in grid_cases.items():
        print("\n" + "=" * 60)
        print(f"Running benchmark case {label}")
        print("=" * 60)

        nx_truth = int(cfg["nx"])
        ny_truth = int(cfg["ny"])

        # 1. Grid and domain
        domain = _build_domain(nx=nx_truth, ny=ny_truth)
        dem = _build_dem(domain)
        active_mask = (domain == 1)

        # 2. Physics common to FD, Warp, and MF6
        T_field = steady_transmissivity(
            domain=domain, isotropic=isotropic, isotropic_value=t_isotropic_value, seed=int(DEFAULT_T_SEED),
        )
        R_field = np.full_like(domain, R_truth, dtype=np.float64)

        log_pool("before build")

        with wds(
                nx=nx_truth,
                ny=ny_truth,
                dx=dx_truth,
                device="cuda:0",
                use_ghb=ghb,
                solver_type="pcg",
                aq_thickness=thickness,
        ) as single_solver:

            single_solver.build_from_truth_inputs(
                T_truth=T_field,
                R_truth=R_field,
                width=width,
            )

            print("Running single case Warp-Kcycles solver...")
            head_k_warp, info_k, cold_runtime, warm_runtime = run_solve(
                solver=single_solver,
                check_every_no=convergence_check_interval,
            )

            print("\nK-cycle info:")
            for k in sorted(info_k.keys()):
                print(f"  {k}: {info_k[k]}")

            # If your kcycle returns lvl0.x_wp.numpy(), head_k_warp is already a numpy array.
            # If it is a Warp array, convert to numpy and drop the Warp reference.
            if hasattr(head_k_warp, "numpy"):
                head_cpu_warp = head_k_warp.numpy()
            else:
                head_cpu_warp = np.asarray(head_k_warp)

            wp.synchronize_device("cuda:0")
            log_pool("after solve")

        # Exiting the with-block calls single_solver.close() via __exit__()
        # Now do a final sync and GC to ensure Python releases references promptly.
        wp.synchronize_device("cuda:0")
        gc.collect()
        wp.synchronize_device("cuda:0")
        log_pool("after cleanup")


        # 4. FD reference solve (same T, R, no GHB) — optional
        if run_fd_reference:
            head_fd, t_fd = run_fd_truth_forward(
                nx=nx_truth,
                ny=ny_truth,
                dx=dx_truth,
                T_truth=T_field,
                R_truth=R_field,
                use_ghb=ghb,
                aq_thickness=thickness,
                width=width
            )
        else:
            head_fd, t_fd = None, None

        # 5. MF6 truth: load an exact cached artifact, or run and cache once.
        ws = data_store.joinpath(
            f"Paper_mf6_truth_{label}_ghb_{ghb}_t_isotropic_{isotropic}"
        )
        hk_field = T_field / thickness
        mf6_truth_path = build_mf6_truth_path(
            truth_dir=truth_dir,
            label=label,
            ghb=ghb,
            isotropic=isotropic,
        )
        reference_provenance = {}
        mf_head, t_mf, mf6_source = load_or_run_mf6_truth(
            provenance=reference_provenance,
            truth_path=mf6_truth_path,
            workspace=ws,
            label=label,
            nx=nx_truth,
            ny=ny_truth,
            dx=dx_truth,
            ghb=ghb,
            isotropic=isotropic,
            t_isotropic_value=t_isotropic_value,
            thickness=thickness,
            width=width,
            recharge_rate=R_truth,
            recharge_field=R_field,
            seed=int(DEFAULT_T_SEED),
            hk_field=hk_field,
            output_dtype=truth_output_dtype,
            mf6_runner=make_mf_model,
            domain=domain,
        )

        if run_fd_reference:
            print("\nFD vs Warp_k_cycle (should be almost machine identical):")
            FD_vs_warp_k_cycle = compare_head_fields(
                head_ref=head_fd,
                head_warp=head_k_warp,
                active_mask=active_mask,
            )

            print("\nMF6 vs FD:")
            mf_vs_fd = compare_head_fields(
                head_ref=mf_head,
                head_warp=head_fd,
                active_mask=active_mask,
            )
        else:
            FD_vs_warp_k_cycle = None
            mf_vs_fd = None

        print("\nK-cycle Warp vs MF6:")
        k_cycle_vs_mf = compare_head_fields(
            head_ref=mf_head,
            head_warp=head_k_warp,
            active_mask=active_mask,
        )

        bud_k = compute_mass_balance_budget(
            T_field=single_solver.T_field_host,
            R_field=single_solver.R_field_host,
            head=head_k_warp,
            active=single_solver.active_host,
            bc_mask=single_solver.bc_mask_host,
            bc_values=single_solver.bc_values_host,
            dx=float(single_solver.dx),
            gh_mask=single_solver.gh_mask_host,
            gh_head=single_solver.gh_head_host,
            gh_width=single_solver.gh_width_host,
            gh_alpha=float(single_solver.gh_alpha),
            aq_thickness=float(single_solver.aq_thickness),
        )

        print("K-cycle mass balance")
        for k, v in bud_k.items():
            if isinstance(v, (float, int, np.floating, np.integer)):
                print(f"  {k}: {v:.3f}")
            else:
                try:
                    v_rounded = np.round(v, 3)
                    print(f"  {k}: {v_rounded}")
                except Exception:
                    print(f"  {k}: {v}")

        n_active = int(np.count_nonzero(active_mask))
        n_total = int(nx_truth * ny_truth)

        # Acceptance gates: a reported result must prove the K-cycle converged
        # and agrees with MF6 and closes its mass balance.  Tolerances carry
        # wide margin over the observed classic agreement (rmse <= ~4e-5 m,
        # max_abs <= ~7e-5 m on the hard-T catalog) while still catching a
        # stalled or non-conservative solve.
        budget_percent_discrepancy = float(bud_k["percent_discrepancy"].iloc[0])
        acceptance_gates = {
            "kcycle_converged": bool(info_k.get("converged", False)),
            "head_agreement_vs_mf6_rmse_le_1e-3": float(k_cycle_vs_mf["rmse"]) <= 1.0e-3,
            "head_agreement_vs_mf6_max_abs_le_1e-2": float(k_cycle_vs_mf["max_abs_diff"]) <= 1.0e-2,
            "mass_balance_percent_discrepancy_le_1pct": abs(budget_percent_discrepancy) <= 1.0,
        }
        sanity_passed = bool(all(acceptance_gates.values()))
        print("Acceptance gates:")
        for gate_name, gate_passed in acceptance_gates.items():
            print(f"  {gate_name}: {'PASS' if gate_passed else 'FAIL'}")

        all_results[label] = {
            "nx": nx_truth,
            "ny": ny_truth,
            "n_cells_total": n_total,
            "n_cells_active": n_active,
            "kcycle_implementation": os.environ.get("DARCY_KCYCLE_IMPL", "classic"),
            "diagnostics": info_k,
            "mf_vs_fd": mf_vs_fd,
            "k_cycle_vs_mf": k_cycle_vs_mf,
            "fd_vs_k_cycle": FD_vs_warp_k_cycle,
            "acceptance_gates": acceptance_gates,
            "sanity_passed": sanity_passed,
            "timings": {
                "fd_seconds": None if t_fd is None else float(t_fd),
                "warp_seconds_cuda_cold_runtime": float(cold_runtime),
                "warp_seconds_cuda_warm_runtime": float(warm_runtime),
                "mf6_seconds": None if t_mf is None else float(t_mf),
            },
            "warp_initial_head": "DEM for both timed solves",
            "mf6_result_source": str(mf6_source),
            "mf6_truth_path": reference_provenance["path"],
            "mf6_truth_sha256": reference_provenance["sha256"],
            "mf6_reference_fingerprint": reference_provenance["fingerprint"],
        }

    # 7. Save or update JSON log
    kcycle_impl = os.environ.get("DARCY_KCYCLE_IMPL", "classic").strip().lower()
    suffix = "hard_vat_t" if kcycle_impl == "classic" else f"hard_vat_t_{kcycle_impl}"
    results_path = data_store.joinpath(f"comparison_results_ghb_{ghb}{suffix}.json")

    try:
        if results_path.exists():
            with results_path.open("r") as f:
                existing = json.load(f)
        else:
            existing = {}
    except Exception:
        existing = {}

    existing.update(all_results)

    with results_path.open("w") as f:
        json.dump(existing, f, indent=4)

    print(f"\nSaved comparison results to {results_path}")

    failed_cases = [label for label, row in all_results.items() if not row.get("sanity_passed", False)]
    if failed_cases:
        print(f"\nSANITY GATES FAILED for cases: {failed_cases}")
        raise SystemExit(1)
    print("\nAll sanity gates passed.")
