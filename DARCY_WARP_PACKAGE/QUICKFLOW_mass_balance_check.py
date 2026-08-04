import time
import json
import os
import numpy as np
import gc

from DARCY_WARP_PACKAGE.model_builder import (
    _build_domain,
    _build_dem,
    make_ugly_T_field, compare_head_fields
)

from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model
from DARCY_WARP_PACKAGE.CPU_FD import run_fd_truth_forward
from DARCY_WARP_PACKAGE.project_base import data_store

os.environ["DARCY_FLOAT"] = "float64"
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver as wds
from DARCY_WARP_PACKAGE.warped_darcy import compute_mass_balance_budget
import warp as wp

# Optional GHB helper: use if available
try:
    from DARCY_WARP_PACKAGE.model_builder import _build_ghb_boundary_mask
except ImportError:
    _build_ghb_boundary_mask = None


def run_solve(solver, check_every_no: int = 1):
    t0 = time.perf_counter()
    with wp.ScopedTimer("kcycle_solve", use_nvtx=True):
        head1, info1 = solver.solve_multigrid_kcycle(
            max_cycles=200,
            nu_pre=2,
            nu_post=2,
            nu_coarse=2,
            omega=0.7,
            rel_tol=5.0e-7,
            abs_tol_min=5.0e-7,
            initial_head=dem,
            return_info=True,
            max_levels=6,
            check_every_no=int(check_every_no),
        )
    t1 = time.perf_counter()
    print("call1 solving time:", t1 - t0)

    t2 = time.perf_counter()
    with wp.ScopedTimer("kcycle_solve", use_nvtx=True):
        heads, info = solver.solve_multigrid_kcycle(
            max_cycles=200,
            nu_pre=2,
            nu_post=2,
            nu_coarse=2,
            omega=0.7,
            rel_tol=5.0e-7,
            abs_tol_min=5.0e-7,
            initial_head=dem,
            return_info=True,
            max_levels=6,
            check_every_no=int(check_every_no),
        )
    t3 = time.perf_counter()
    print("call2 solving time:", t3 - t2)
    print(info)
    cold = t1-t0
    warm = t3-t2
    return heads, info, cold, warm

def solve_callable():
    return run_solve(single_solver)

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


if __name__ == "__main__":
    ghb = True
    isotropic_T = False
    # Grid cases to benchmark: label -> (nx, ny), drawn from the shared
    # spatial catalog.  The capacity cases are selected explicitly here
    # (they are manual-only in the catalog and never auto-selected).
    from DARCY_WARP_PACKAGE.sanity_case_config import SPATIAL_GRID_CASES

    grid_cases = {
        label: {
            "nx": int(SPATIAL_GRID_CASES[label]["nx"]),
            "ny": int(SPATIAL_GRID_CASES[label]["ny"]),
        }
        for label in (
            "100x100",
            "100x250",
            "400x400",
            "100x1000",
            "250x1000",
            "1000x1001",
            "2000x1000",
            "3000x111",
            "3000x223",
            "3000x333",
            "3000x999",
            "3000x1999",
            "3000x2999",
            "3000x3000",
            # "2600x10000",  # nz at 100m resolution
        )
    }

    dx_truth = 100
    R_truth = 1.0e-4
    thickness = 300.0
    width = dx_truth

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


        if isotropic_T:
            T_field = 3000.0 * np.ones_like(domain)
        else:
            # 2. Physics common to FD, Warp, and MF6
            T_field = make_ugly_T_field(
                nx=nx_truth,
                ny=ny_truth,
                domain=domain,
                seed=123,
            )


        R_field_ugly = np.full_like(domain, R_truth, dtype=np.float64)

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
                R_truth=R_field_ugly,
                width=width,
            )

            print("Running single case Warp-Kcycles solver...")
            head_k_warp, info_k, cold, warm = run_solve(single_solver)

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
            case=label,
        )

        print("K-cycle mass balance")
        for k in sorted(bud_k.keys()):
            print(f"  {k}: {bud_k[k]}")

        mass_balance_budget_kcycle = bud_k.iloc[0].to_dict()


        n_active = int(np.count_nonzero(active_mask))
        n_total = int(nx_truth * ny_truth)

        all_results[label] = {
            "nx": nx_truth,
            "ny": ny_truth,
            "n_cells_total": n_total,
            "n_cells_active": n_active,
            "diagnostics": info_k,
            "timings": {
                "warp_seconds_cold_start": float(cold),
                "warp_seconds_warm_start": float(warm),
            },
            "mass_balance_budget_kcycle": mass_balance_budget_kcycle,
        }

    # 7. Save or update JSON log
    results_path = data_store.joinpath(f"mass_balance_results_ghb_{ghb}_t_isotropic_{isotropic_T}.json")

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

    raise NotImplementedError("timing results profiling has bug")
    timing_results = profile_one_solve(
        solve_callable=solve_callable,
        out_csv=Path("profiles").joinpath("darcy_warp_timing.csv"),
        warmup_runs=0,
)
