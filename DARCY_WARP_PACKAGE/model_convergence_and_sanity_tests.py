import time
import json
import os
from pathlib import Path
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
from DARCY_WARP_PACKAGE.sanity_case_config import (
    GRID_CASES,
    DEFAULT_DX,
    DEFAULT_R_TRUTH,
    DEFAULT_THICKNESS,
    DEFAULT_GHB,
    DEFAULT_T_SEED,
    DEFAULT_ISOTROPIC_T,
)

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


def load_cached_mf6_truth(
    truth_path: Path,
    label: str,
    nx: int,
    ny: int,
    dx: float,
    ghb: bool,
    isotropic: bool,
    t_isotropic_value: float,
    thickness: float,
    width: float,
    recharge: float,
    seed: int,
) -> tuple[np.ndarray, float | None] | None:
    """Load a matching MF6 artifact, returning ``None`` for a cache miss."""
    if not truth_path.exists():
        return None

    try:
        with np.load(truth_path, allow_pickle=False) as truth:
            expected_scalars = {
                "nx": int(nx),
                "ny": int(ny),
                "ghb": int(bool(ghb)),
                "t_isotropic": int(bool(isotropic)),
                "seed": int(seed),
            }
            for key, expected in expected_scalars.items():
                if key not in truth.files or int(truth[key]) != expected:
                    return None

            expected_floats = {
                "dx": float(dx),
                "t_isotropic_value": float(t_isotropic_value),
                "thickness": float(thickness),
                "width": float(width),
                "r_truth": float(recharge),
            }
            for key, expected in expected_floats.items():
                if key not in truth.files or not np.isclose(
                    float(truth[key]),
                    expected,
                    rtol=1.0e-6,
                    atol=1.0e-12,
                ):
                    return None

            if "label" not in truth.files or str(truth["label"]) != str(label):
                return None
            if "heads" not in truth.files:
                return None

            heads = np.asarray(truth["heads"], dtype=np.float64)
            if heads.shape != (int(ny), int(nx)) or not np.all(np.isfinite(heads)):
                return None

            mf6_seconds = None
            if "mf6_seconds" in truth.files:
                candidate_seconds = float(truth["mf6_seconds"])
                if np.isfinite(candidate_seconds) and candidate_seconds >= 0.0:
                    mf6_seconds = candidate_seconds
    except (OSError, ValueError, KeyError, TypeError):
        return None

    return heads, mf6_seconds


def save_mf6_truth(
    truth_path: Path,
    heads: np.ndarray,
    label: str,
    nx: int,
    ny: int,
    dx: float,
    ghb: bool,
    isotropic: bool,
    t_isotropic_value: float,
    thickness: float,
    width: float,
    recharge: float,
    seed: int,
    mf6_seconds: float,
    output_dtype: np.dtype,
) -> None:
    """Atomically save one reusable MF6 truth artifact."""
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = truth_path.with_name(f".{truth_path.name}.staging.npz")
    float_dtype = np.dtype(output_dtype)
    np.savez_compressed(
        temporary_path,
        heads=np.asarray(heads, dtype=float_dtype),
        nx=np.int32(nx),
        ny=np.int32(ny),
        dx=np.asarray(dx, dtype=float_dtype),
        ghb=np.int32(1 if ghb else 0),
        t_isotropic=np.int32(1 if isotropic else 0),
        t_isotropic_value=np.asarray(t_isotropic_value, dtype=float_dtype),
        thickness=np.asarray(thickness, dtype=float_dtype),
        width=np.asarray(width, dtype=float_dtype),
        r_truth=np.asarray(recharge, dtype=float_dtype),
        seed=np.int32(seed),
        label=np.array(label),
        mf6_seconds=np.asarray(mf6_seconds, dtype=np.float64),
    )
    temporary_path.replace(truth_path)


def load_or_run_mf6_truth(
    truth_path: Path,
    workspace: Path,
    label: str,
    nx: int,
    ny: int,
    dx: float,
    ghb: bool,
    isotropic: bool,
    t_isotropic_value: float,
    thickness: float,
    width: float,
    recharge_rate: float,
    recharge_field: np.ndarray,
    seed: int,
    hk_field: np.ndarray,
    output_dtype: np.dtype,
    mf6_runner,
) -> tuple[np.ndarray, float | None, str]:
    """Load matching MF6 heads or run MF6 once and populate the cache."""
    cached = load_cached_mf6_truth(
        truth_path=truth_path,
        label=label,
        nx=nx,
        ny=ny,
        dx=dx,
        ghb=ghb,
        isotropic=isotropic,
        t_isotropic_value=t_isotropic_value,
        thickness=thickness,
        width=width,
        recharge=recharge_rate,
        seed=seed,
    )
    if cached is not None:
        heads, mf6_seconds = cached
        print(f"Loading cached MF6 truth: {truth_path}")
        return heads, mf6_seconds, "cache"

    print(f"MF6 truth cache miss; running model for {label}: {truth_path}")
    heads, mf6_seconds = mf6_runner(
        nx=nx,
        ny=ny,
        grid_size=dx,
        nper=1,
        workspace=workspace,
        hk=hk_field,
        recharge=recharge_field,
        run=True,
        use_ghb=ghb,
    )
    save_mf6_truth(
        truth_path=truth_path,
        heads=heads,
        label=label,
        nx=nx,
        ny=ny,
        dx=dx,
        ghb=ghb,
        isotropic=isotropic,
        t_isotropic_value=t_isotropic_value,
        thickness=thickness,
        width=width,
        recharge=recharge_rate,
        seed=seed,
        mf6_seconds=float(mf6_seconds),
        output_dtype=output_dtype,
    )
    return np.asarray(heads, dtype=np.float64), float(mf6_seconds), "generated"


if __name__ == "__main__":
    ghb = bool(DEFAULT_GHB)
    isotropic = False
    # Grid cases to benchmark: label -> (nx, ny)
    grid_cases = GRID_CASES

    dx_truth = float(DEFAULT_DX)
    R_truth = float(DEFAULT_R_TRUTH)
    thickness = float(DEFAULT_THICKNESS)
    width = dx_truth
    t_isotropic_value = float(DEFAULT_ISOTROPIC_T)
    truth_dir = data_store.joinpath("mf6_truth_npz")
    truth_output_dtype = np.dtype(np.float32)

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
        T_field_ugly = make_ugly_T_field(
            nx=nx_truth,
            ny=ny_truth,
            domain=domain,
            seed=int(DEFAULT_T_SEED),
        )

        # testing overwrite of T_field
        # T_field = 3000.0 * np.ones_like(T_field)

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
                T_truth=T_field_ugly,
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


        # 4. FD reference solve (same T, R, no GHB)
        head_fd, t_fd = run_fd_truth_forward(
            nx=nx_truth,
            ny=ny_truth,
            dx=dx_truth,
            T_truth=T_field_ugly,
            R_truth=R_field_ugly,
            use_ghb=ghb,
            aq_thickness=thickness,
            width=width
        )

        # 5. MF6 truth: load an exact cached artifact, or run and cache once.
        ws = data_store.joinpath(
            f"Paper_mf6_truth_{label}_ghb_{ghb}_t_isotropic_{isotropic}"
        )
        hk_field = T_field_ugly / thickness
        mf6_truth_path = build_mf6_truth_path(
            truth_dir=truth_dir,
            label=label,
            ghb=ghb,
            isotropic=isotropic,
        )
        mf_head, t_mf, mf6_source = load_or_run_mf6_truth(
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
            recharge_field=R_field_ugly,
            seed=int(DEFAULT_T_SEED),
            hk_field=hk_field,
            output_dtype=truth_output_dtype,
            mf6_runner=make_mf_model,
        )

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

        all_results[label] = {
            "nx": nx_truth,
            "ny": ny_truth,
            "n_cells_total": n_total,
            "n_cells_active": n_active,
            "diagnostics": info_k,
            "mf_vs_fd": mf_vs_fd,
            "k_cycle_vs_mf": k_cycle_vs_mf,
            "fd_vs_k_cycle": FD_vs_warp_k_cycle,
            "timings": {
                "fd_seconds": float(t_fd),
                "warp_seconds_cold_start": float(cold),
                "warp_seconds_warm_start": float(warm),
                "mf6_seconds": None if t_mf is None else float(t_mf),
            },
            "mf6_result_source": str(mf6_source),
            "mf6_truth_path": str(mf6_truth_path),
        }

    # 7. Save or update JSON log
    results_path = data_store.joinpath(f"comparison_results_ghb_{ghb}hard_vat_t.json")

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
