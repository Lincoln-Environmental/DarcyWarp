import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import numpy as np
import argparse
from datetime import datetime, timezone
from importlib import import_module

from DARCY_WARP_PACKAGE.project_base import data_store
from DARCY_WARP_PACKAGE.model_builder import build_base_fields
from multiprocessing.shared_memory import SharedMemory


def _write_metadata_if_enabled(out_path: Path, payload: dict, enabled: bool) -> None:
    if not bool(enabled):
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"Saved optional benchmark metadata to {out_path}")


def _default_workers_arg() -> str:
    cpus = os.cpu_count()
    if not cpus or cpus < 1:
        cpus = 1
    candidates = [2, 4, 8, 16, 24]
    limited = [w for w in candidates if w <= cpus]
    if not limited:
        return str(cpus)
    if cpus < candidates[-1] and limited[-1] != cpus:
        limited.append(cpus)
    return ",".join(str(w) for w in limited)


def _effective_worker_count(requested_workers: int, n_cases: int) -> int:
    if int(requested_workers) < 1:
        raise ValueError("n_workers must be >= 1")
    cpus = os.cpu_count()
    if not cpus or cpus < 1:
        cpus = 1
    capped = min(int(requested_workers), int(n_cases), int(cpus))
    return max(1, int(capped))


def run_single_mf6_case(
    case_id: int,
    nx: int,
    ny: int,
    dx: float,
    T_field: np.ndarray,
    R_field: np.ndarray,
    base_workspace,
    ghb: bool = False,
) -> float:
    """
    Run a single MF6 case in its own workspace and return the measured wall time.

    :param case_id: Case index, used for workspace naming.
    :param nx: Number of columns.
    :param ny: Number of rows.
    :param dx: Cell size.
    :param T_field: Transmissivity field, shape (ny, nx).
    :param R_field: Recharge field, shape (ny, nx).
    :param base_workspace: Base directory for case workspaces.
    :param ghb: If True, include GHB boundaries.
    :return: Wall clock seconds for this run (write + run + optional extract depending on make_mf_model).
    """
    from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model

    ws_case = Path(base_workspace).joinpath(f"mf6_case_{int(case_id):03d}")
    ws_case.mkdir(parents=True, exist_ok=True)

    thickness = 300.0
    hk_field = np.asarray(T_field, dtype=float) / thickness

    _heads, total_time = make_mf_model(
        nx=nx,
        ny=ny,
        grid_size=dx,
        nper=1,
        workspace=ws_case,
        hk=hk_field,
        recharge=np.asarray(R_field, dtype=float),
        run=True,
        use_ghb=ghb,
        record_full_time=True,
    )

    return float(total_time)


def run_single_fd_case(case_id, nx, ny, dx, T_field, R_field, base_workspace, ghb=False):
    """
    Run a single CPU finite difference reference solve and return the internal elapsed time.

    :param case_id: Case index, used for workspace naming (optional).
    :param nx: Number of columns.
    :param ny: Number of rows.
    :param dx: Cell size.
    :param T_field: Transmissivity field, shape (ny, nx).
    :param R_field: Recharge field, shape (ny, nx).
    :param base_workspace: Base directory for case workspaces.
    :param ghb: If True, include GHB boundaries.
    :return: Elapsed seconds reported by the FD solver.
    """
    from DARCY_WARP_PACKAGE.CPU_FD import run_fd_truth_forward

    ws_case = Path(base_workspace).joinpath(f"fd_case_{case_id:03d}")
    ws_case.mkdir(parents=True, exist_ok=True)

    head_fd, elapsed_internal = run_fd_truth_forward(
        nx=nx,
        ny=ny,
        dx=dx,
        T_truth=T_field,
        R_truth=R_field,
        use_ghb=ghb,
    )
    return float(elapsed_internal)


def benchmark_mf6_ensemble_persistent(
    nx: int,
    ny: int,
    dx: float,
    n_cases: int,
    n_workers_list: list[int],
    base_workspace,
    ghb: bool = False,
    recharge_stack: np.ndarray | None = None,
    extract_heads: bool = True,
):
    """
       Benchmark MF6 using persistent workers.

       Each worker builds a template simulation once, then loops over an assigned
       contiguous range of cases, updating only the recharge package each time.

       :param nx: Number of columns.
       :param ny: Number of rows.
       :param dx: Cell size.
       :param n_cases: Total number of cases.
       :param n_workers_list: Worker counts to test.
       :param base_workspace: Base directory for benchmark folders.
       :param ghb: If True, include GHB boundaries.
       :param recharge_stack: Optional recharge array stack, shape (n_cases, ny, nx).
       :param extract_heads: If True, read heads after each run (cost included in timing).
       :return: Results dictionary keyed by "mf6_persistent_workers_{W}".
       """
    from DARCY_WARP_PACKAGE.modflow_truth import run_mf6_persistent_worker_batch_shm

    base_workspace = Path(base_workspace)
    base_workspace.mkdir(parents=True, exist_ok=True)

    domain, dem, T_field_ugly, R_default = build_base_fields(nx, ny, dx)
    thickness = 300.0
    hk_field = (np.asarray(T_field_ugly, dtype=np.float64) / float(thickness)).astype(np.float64, copy=False)

    if recharge_stack is None:
        recharge_stack = np.empty((int(n_cases), int(ny), int(nx)), dtype=np.float64)
        recharge_stack[:, :, :] = np.asarray(R_default, dtype=np.float64)[None, :, :]
    else:
        recharge_stack = np.asarray(recharge_stack, dtype=np.float64)
        if recharge_stack.shape != (int(n_cases), int(ny), int(nx)):
            raise ValueError(f"recharge_stack shape {recharge_stack.shape} expected {(int(n_cases), int(ny), int(nx))}")

    hk_shm = SharedMemory(create=True, size=int(hk_field.nbytes))
    rch_shm = SharedMemory(create=True, size=int(recharge_stack.nbytes))

    try:
        hk_view = np.ndarray(hk_field.shape, dtype=hk_field.dtype, buffer=hk_shm.buf)
        hk_view[:, :] = hk_field[:, :]

        rch_view = np.ndarray(recharge_stack.shape, dtype=recharge_stack.dtype, buffer=rch_shm.buf)
        rch_view[:, :, :] = recharge_stack[:, :, :]

        results = {}

        seen_workers: set[int] = set()
        for n_workers in n_workers_list:
            requested_workers = int(n_workers)
            n_workers_i = _effective_worker_count(requested_workers, int(n_cases))
            if n_workers_i != requested_workers:
                print(
                    f"Adjusted requested MF6 workers from {requested_workers} to {n_workers_i} "
                    f"(cpu_count/n_cases cap)."
                )
            if n_workers_i in seen_workers:
                print(f"Skipping duplicate MF6 worker configuration W={n_workers_i}.")
                continue
            seen_workers.add(int(n_workers_i))

            run_root = base_workspace.joinpath(f"mf6_persistent_workers_{n_workers_i}")
            run_root.mkdir(parents=True, exist_ok=True)

            # contiguous case id ranges per worker
            batches: list[tuple[int, int, int]] = []
            start = 0
            for w in range(int(n_workers_i)):
                end = (w + 1) * int(n_cases) // int(n_workers_i)
                if end > start:
                    batches.append((w, start, end))
                start = end

            t_wall0 = time.perf_counter()
            worker_payloads = []

            with ProcessPoolExecutor(max_workers=n_workers_i) as pool:
                futures = []
                for w, i0, i1 in batches:
                    futures.append(
                        pool.submit(
                            run_mf6_persistent_worker_batch_shm,
                            w,
                            int(nx),
                            int(ny),
                            float(dx),
                            hk_shm.name,
                            hk_field.shape,
                            str(hk_field.dtype),
                            rch_shm.name,
                            recharge_stack.shape,
                            str(recharge_stack.dtype),
                            int(i0),
                            int(i1),
                            run_root,
                            bool(ghb),
                            bool(extract_heads),
                        )
                    )

                for fut in as_completed(futures):
                    worker_payloads.append(fut.result())

            t_wall1 = time.perf_counter()
            total_wall = float(t_wall1 - t_wall0)

            # aggregate the same way you already do
            template_build_times = []
            update_times = []
            write_times = []
            run_times = []
            extract_times = []
            total_times = []
            total_cases_done = 0

            for payload in worker_payloads:
                template_build_times.append(float(payload["template_build_time"]))
                update_times.extend([float(x) for x in payload["update_times"]])
                write_times.extend([float(x) for x in payload["write_times"]])
                run_times.extend([float(x) for x in payload["run_times"]])
                extract_times.extend([float(x) for x in payload["extract_times"]])
                total_times.extend([float(x) for x in payload["total_times"]])
                total_cases_done += int(payload["n_cases"])

            if total_cases_done != int(n_cases):
                raise RuntimeError(f"Expected {int(n_cases)} cases, got {total_cases_done}")

            update_arr = np.asarray(update_times, dtype=float)
            write_arr = np.asarray(write_times, dtype=float)
            run_arr = np.asarray(run_times, dtype=float)
            extract_arr = np.asarray(extract_times, dtype=float)
            total_arr = np.asarray(total_times, dtype=float)

            entry = {
                "nx": int(nx),
                "ny": int(ny),
                "n_cells_total": int(nx * ny),
                "n_cases": int(n_cases),
                "n_workers": int(n_workers_i),
                "total_wall_seconds": float(total_wall),
                "template_build_total_seconds": float(np.sum(np.asarray(template_build_times, dtype=float))),
                "mean_update_seconds": float(update_arr.mean()),
                "mean_write_seconds": float(write_arr.mean()),
                "mean_run_seconds": float(run_arr.mean()),
                "mean_extract_seconds": float(extract_arr.mean()),
                "mean_total_seconds": float(total_arr.mean()),
                "throughput_cases_per_second": float(n_cases / total_wall) if total_wall > 0.0 else float("nan"),
            }

            results[f"mf6_persistent_workers_{n_workers_i}"] = entry

            out_path = base_workspace.joinpath(f"mf6_persistent_benchmark_{nx}x{ny}_N{n_cases}_W{n_workers_i}.json")
            with out_path.open("w") as f:
                json.dump(entry, f, indent=2)

        return results

    finally:
        hk_shm.close()
        hk_shm.unlink()
        rch_shm.close()
        rch_shm.unlink()


def benchmark_fd_ensemble(
    nx: int,
    ny: int,
    dx: float,
    n_cases: int,
    n_workers_list: list[int],
    base_workspace,
    recharge_stack: np.ndarray | None = None,
    ghb: bool = False,
) -> dict:
    """
    Benchmark CPU finite difference solves across a set of worker counts.

    :param nx: Number of columns.
    :param ny: Number of rows.
    :param dx: Cell size.
    :param n_cases: Total number of cases.
    :param n_workers_list: Worker counts to test.
    :param base_workspace: Base directory for benchmark folders.
    :param recharge_stack: Optional recharge array stack, shape (n_cases, ny, nx).
    :param ghb: If True, include GHB boundaries.
    :return: Results dictionary keyed by "fd_workers_{W}".
    """
    base_workspace = Path(base_workspace)
    base_workspace.mkdir(parents=True, exist_ok=True)

    domain, dem, T_field_ugly, R_default = build_base_fields(nx, ny, dx)

    if recharge_stack is not None:
        recharge_stack = np.asarray(recharge_stack, dtype=np.float64)
        expected = (int(n_cases), int(ny), int(nx))
        if recharge_stack.shape != expected:
            raise ValueError(f"recharge_stack shape {recharge_stack.shape} expected {expected}")

    results: dict[str, dict] = {}

    seen_workers: set[int] = set()
    for n_workers in n_workers_list:
        requested_workers = int(n_workers)
        n_workers_i = _effective_worker_count(requested_workers, int(n_cases))
        if n_workers_i != requested_workers:
            print(
                f"Adjusted requested FD workers from {requested_workers} to {n_workers_i} "
                f"(cpu_count/n_cases cap)."
            )
        if n_workers_i in seen_workers:
            print(f"Skipping duplicate FD worker configuration W={n_workers_i}.")
            continue
        seen_workers.add(int(n_workers_i))

        print(f"\n=== FD benchmark: {nx}x{ny}, {n_cases} cases, {int(n_workers_i)} workers ===")

        t_start = time.perf_counter()
        per_case_times: list[float] = []

        with ProcessPoolExecutor(max_workers=int(n_workers_i)) as pool:
            futures = []
            for case_id in range(int(n_cases)):
                if recharge_stack is None:
                    R_case = R_default
                else:
                    R_case = recharge_stack[int(case_id), :, :]

                futures.append(
                    pool.submit(
                        run_single_fd_case,
                        int(case_id),
                        int(nx),
                        int(ny),
                        float(dx),
                        T_field_ugly,
                        R_case,
                        base_workspace,
                        bool(ghb),
                    )
                )

            for fut in as_completed(futures):
                per_case_times.append(float(fut.result()))

        total_wall = float(time.perf_counter() - t_start)
        per_case_array = np.asarray(per_case_times, dtype=float)

        result_entry = {
            "nx": int(nx),
            "ny": int(ny),
            "n_cells_total": int(nx * ny),
            "n_cases": int(n_cases),
            "n_workers": int(n_workers_i),
            "total_wall_seconds": float(total_wall),
            "mean_case_seconds": float(per_case_array.mean()),
            "min_case_seconds": float(per_case_array.min()),
            "max_case_seconds": float(per_case_array.max()),
            "throughput_cases_per_second": float(n_cases / total_wall) if total_wall > 0.0 else float("nan"),
        }
        results[f"fd_workers_{int(n_workers_i)}"] = result_entry

    return results


def _benchmark_warp_multigrid_class_solvers_core(
    nx: int,
    ny: int,
    dx: float,
    n_cases: int,
    base_workspace,
    recharge_stack: np.ndarray | None = None,
    warp_device: str = "cuda:0",
    n_iter: int = 200,
    nu_pre: int = 2,
    nu_post: int = 2,
    nu_coarse: int = 2,
    rel_tol: float = 1.0e-5,
    abs_tol_min: float = 1.0e-5,
    use_ghb: bool = False,
    gh_alpha: float = 1.0,
    warmup: int = 1,
    solver_module: str = "DARCY_WARP_PACKAGE.warped_darcy",
    solver_variant: str = "warp_kcycle_default",
    solve_kwargs: dict | None = None,
    mg_min_coarse_cells: int | None = 500,
) -> dict:
    """
    Core benchmark for Warp-class recharge-change runs.

    Reuses one solver instance and benchmarks two update paths:
    host-only recharge copy and `update_R_in_place`.
    """
    os.environ["DARCY_FLOAT"] = "float64"
    WarpDarcySolver = import_module(str(solver_module)).WarpDarcySolver

    base_workspace = Path(base_workspace)
    base_workspace.mkdir(parents=True, exist_ok=True)

    domain, dem, T_field_ugly, R_field_ugly = build_base_fields(nx, ny, dx)

    if recharge_stack is not None:
        recharge_stack = np.asarray(recharge_stack, dtype=np.float64)
        expected = (int(n_cases), int(ny), int(nx))
        if recharge_stack.shape != expected:
            raise ValueError(f"recharge_stack shape {recharge_stack.shape} expected {expected}")

    print(
        f"\n=== Warp class benchmark ({solver_variant}): {nx}x{ny}, {n_cases} cases, "
        f"device={warp_device} ==="
    )

    single_solver = WarpDarcySolver(
        nx=int(nx),
        ny=int(ny),
        dx=float(dx),
        device=str(warp_device),
        use_ghb=bool(use_ghb),
        solver_type="pcg",
        aq_thickness= 300
    )

    single_solver.build_from_truth_inputs(
        T_truth=T_field_ugly,
        R_truth=R_field_ugly,
        gh_alpha=float(gh_alpha),
    )

    if single_solver.R_field_host is None:
        raise RuntimeError("Expected single_solver.R_field_host to be initialized after build_from_truth_inputs().")

    if tuple(single_solver.R_field_host.shape) != (int(ny), int(nx)):
        raise RuntimeError(
            f"Unexpected R_field_host shape {single_solver.R_field_host.shape}, expected ({int(ny)}, {int(nx)})."
        )

    target_dtype = single_solver.R_field_host.dtype

    def _get_recharge_case(k: int) -> np.ndarray:
        if recharge_stack is None:
            return R_field_ugly
        return recharge_stack[int(k), :, :]

    mg_min_coarse_cells_norm = None if mg_min_coarse_cells is None else int(mg_min_coarse_cells)
    if mg_min_coarse_cells_norm is not None and mg_min_coarse_cells_norm < 1:
        raise ValueError("mg_min_coarse_cells must be >= 1 when provided")

    solve_call_kwargs = {
        "max_cycles": int(n_iter),
        "nu_pre": int(nu_pre),
        "nu_post": int(nu_post),
        "nu_coarse": int(nu_coarse),
        "rel_tol": float(rel_tol),
        "abs_tol_min": float(abs_tol_min),
        "initial_head": dem,
        "return_info": False,
        "max_levels": 6,
        "check_every_no": 1,
    }
    if mg_min_coarse_cells_norm is not None:
        solve_call_kwargs["min_coarse_cells"] = int(mg_min_coarse_cells_norm)
    if solve_kwargs:
        solve_call_kwargs.update(dict(solve_kwargs))

    # Warmup (not timed) to trigger compilation/caching
    warmup_kwargs = dict(solve_call_kwargs)
    warmup_kwargs.update(
        {
            "max_cycles": 2,
            "nu_pre": 1,
            "nu_post": 1,
            "nu_coarse": 1,
            "rel_tol": 1.0e-3,
            "abs_tol_min": 1.0e-3,
            "initial_head": dem,
            "return_info": False,
            "max_levels": 6,
            "check_every_no": 1,
        }
    )

    for k in range(int(max(0, warmup))):
        R_case = _get_recharge_case(k)
        R_arr = np.asarray(R_case, dtype=target_dtype, order="C")
        single_solver.R_field_host[:, :] = R_arr[:, :]
        _ = single_solver.solve_multigrid_kcycle(**warmup_kwargs)

    def _run_variant(update_mode: str) -> dict:
        per_case_times: list[float] = []
        per_case_update_times: list[float] = []
        per_case_solve_times: list[float] = []

        t_start = time.perf_counter()

        for k in range(int(n_cases)):
            R_case = _get_recharge_case(k)
            R_arr = np.asarray(R_case, dtype=target_dtype, order="C")

            t0 = time.perf_counter()
            t_up0 = time.perf_counter()

            if update_mode == "host_only":
                single_solver.R_field_host[:, :] = R_arr[:, :]
            elif update_mode == "in_place":
                single_solver.update_R_in_place(R_arr)
            else:
                raise ValueError(f"Unknown update_mode '{update_mode}'")

            t_up1 = time.perf_counter()

            _ = single_solver.solve_multigrid_kcycle(**solve_call_kwargs)
            t1 = time.perf_counter()

            per_case_update_times.append(float(t_up1 - t_up0))
            per_case_solve_times.append(float(t1 - t_up1))
            per_case_times.append(float(t1 - t0))

        total_wall = float(time.perf_counter() - t_start)
        per_case_arr = np.asarray(per_case_times, dtype=float)
        update_arr = np.asarray(per_case_update_times, dtype=float)
        solve_arr = np.asarray(per_case_solve_times, dtype=float)

        return {
            "nx": int(nx),
            "ny": int(ny),
            "n_cells_total": int(nx * ny),
            "n_cases": int(n_cases),
            "device": str(warp_device),
            "update_mode": str(update_mode),
            "solver_module": str(solver_module),
            "solver_variant": str(solver_variant),
            "warmup": int(max(0, warmup)),
            "total_wall_seconds": float(total_wall),
            "mean_case_seconds": float(per_case_arr.mean()),
            "mean_update_seconds": float(update_arr.mean()),
            "mean_solve_seconds": float(solve_arr.mean()),
            "min_case_seconds": float(per_case_arr.min()),
            "max_case_seconds": float(per_case_arr.max()),
            "throughput_cases_per_second": float(n_cases / total_wall) if total_wall > 0.0 else float("nan"),
        }

    host_only_entry = _run_variant("host_only")
    in_place_entry = _run_variant("in_place")

    results = {
        "warp_class_single_host_only": host_only_entry,
        "warp_class_single_in_place": in_place_entry,
    }

    results_path = base_workspace.joinpath(f"warp_class_benchmark_{nx}x{ny}_N{n_cases}.json")
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    return results


def benchmark_warp_multigrid_class_solvers(
    nx: int,
    ny: int,
    dx: float,
    n_cases: int,
    base_workspace,
    recharge_stack: np.ndarray | None = None,
    warp_device: str = "cuda:0",
    n_iter: int = 200,
    nu_pre: int = 2,
    nu_post: int = 2,
    nu_coarse: int = 2,
    rel_tol: float = 1.0e-5,
    abs_tol_min: float = 1.0e-5,
    use_ghb: bool = False,
    gh_alpha: float = 1.0,
    warmup: int = 1,
    mg_min_coarse_cells: int | None = 500,
) -> dict:
    """
    Benchmark the existing Warp K-cycle path (`warped_darcy`) for recharge-change ensembles.
    """
    return _benchmark_warp_multigrid_class_solvers_core(
        nx=nx,
        ny=ny,
        dx=dx,
        n_cases=n_cases,
        base_workspace=base_workspace,
        recharge_stack=recharge_stack,
        warp_device=warp_device,
        n_iter=n_iter,
        nu_pre=nu_pre,
        nu_post=nu_post,
        nu_coarse=nu_coarse,
        rel_tol=rel_tol,
        abs_tol_min=abs_tol_min,
        use_ghb=use_ghb,
        gh_alpha=gh_alpha,
        warmup=warmup,
        solver_module="DARCY_WARP_PACKAGE.warped_darcy",
        solver_variant="warp_kcycle_default",
        solve_kwargs=None,
        mg_min_coarse_cells=mg_min_coarse_cells,
    )


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for ensemble benchmarks.

    Builds a deterministic recharge_stack from seed, then runs any selected
    benchmark components and writes JSON summaries to data_store.

    :param argv: Optional argument list for testing, defaults to sys.argv.
    :return: Process exit code, 0 on success.
    """
    parser = argparse.ArgumentParser(description="Run ensemble benchmarks (Warp, MF6, FD).")
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=1000)
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--n_cases", type=int, default=48)
    parser.add_argument("--workers", type=str, default=_default_workers_arg())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ghb", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--mg_min_coarse_cells",
        type=int,
        default=500,
        help="Minimum total cells on the coarsest MG level (<=0 disables this limit).",
    )

    parser.add_argument("--run_warp", action="store_true")
    parser.add_argument("--run_mf6", action="store_true")
    parser.add_argument("--run_fd", action="store_true")
    parser.add_argument(
        "--write_metadata",
        action="store_true",
        help="Write optional metadata JSON containing run configuration and output file paths.",
    )

    args = parser.parse_args(argv)

    nx_truth = int(args.nx)
    ny_truth = int(args.ny)
    dx_truth = float(args.dx)
    cells = int(nx_truth * ny_truth)
    n_cases = int(args.n_cases)
    ghb = bool(args.ghb)
    mg_min_coarse_cells = None if int(args.mg_min_coarse_cells) <= 0 else int(args.mg_min_coarse_cells)

    requested_workers_list: list[int] = []
    for part in str(args.workers).split(","):
        s = part.strip()
        if len(s) > 0:
            requested_workers_list.append(int(s))

    n_workers_list: list[int] = []
    for requested in requested_workers_list:
        effective = _effective_worker_count(int(requested), int(n_cases))
        if effective != int(requested):
            print(
                f"Adjusted requested workers from {int(requested)} to {int(effective)} "
                f"(cpu_count/n_cases cap)."
            )
        if int(effective) in n_workers_list:
            print(f"Skipping duplicate worker configuration W={int(effective)} after capping.")
            continue
        n_workers_list.append(int(effective))

    # If no flags set, run everything
    if (not args.run_warp) and (not args.run_mf6) and (not args.run_fd):
        run_warp = True
        run_mf6 = True
        run_fd = True
    else:
        run_warp = bool(args.run_warp)
        run_mf6 = bool(args.run_mf6)
        run_fd = bool(args.run_fd)

    cpus = os.cpu_count()
    print(f"Detected {cpus} CPU cores available for parallel processing.")

    rng = np.random.default_rng(int(args.seed))

    base_recharge = 1.0e-4
    jitter = 0.5e-4

    recharge_stack = np.empty((int(n_cases), int(ny_truth), int(nx_truth)), dtype=np.float64)
    for case_id in range(int(n_cases)):
        variation = rng.uniform(-jitter, jitter, size=(int(ny_truth), int(nx_truth)))
        recharge_stack[int(case_id), :, :] = base_recharge + variation

    written_outputs: dict[str, str] = {}

    if run_warp:
        warp_class_ws = data_store.joinpath(f"warp_class_ensemble_benchmark_{cells}")
        warp_class_ws.mkdir(exist_ok=True)

        warp_class_results = benchmark_warp_multigrid_class_solvers(
            nx=nx_truth,
            ny=ny_truth,
            dx=dx_truth,
            n_cases=n_cases,
            base_workspace=warp_class_ws,
            warp_device=str(args.device),
            n_iter=200,
            nu_pre=2,
            nu_post=2,
            nu_coarse=2,
            rel_tol=1.0e-5,
            abs_tol_min=1.0e-5,
            use_ghb=ghb,
            gh_alpha=1.0,
            recharge_stack=recharge_stack,
            warmup=1,
            mg_min_coarse_cells=mg_min_coarse_cells,
        )

        warp_class_results_path = data_store.joinpath(
            f"warp_class_ensemble_benchmark_results_recharge_{cells}.json"
        )
        with warp_class_results_path.open("w") as f:
            json.dump(warp_class_results, f, indent=4)
        written_outputs["warp_summary_json"] = str(warp_class_results_path)

        print(f"Saved Warp class benchmark results to {warp_class_results_path}")

    if run_mf6:
        mp.set_start_method("spawn", force=True)

        mf6_bench_ws = data_store.joinpath("mf6_ensemble_benchmark")
        mf6_bench_ws.mkdir(exist_ok=True)

        mf6_results = benchmark_mf6_ensemble_persistent(
            nx=nx_truth,
            ny=ny_truth,
            dx=dx_truth,
            n_cases=n_cases,
            n_workers_list=n_workers_list,
            base_workspace=mf6_bench_ws,
            ghb=ghb,
            recharge_stack=recharge_stack,
            extract_heads=True,
        )

        mf6_results_path = data_store.joinpath(
            f"mf6_ensemble_benchmark_results_recharge{cells}.json"
        )
        with mf6_results_path.open("w") as f:
            json.dump(mf6_results, f, indent=4)
        written_outputs["mf6_summary_json"] = str(mf6_results_path)

        print(f"Saved MF6 benchmark results to {mf6_results_path}")

    if run_fd:
        fd_bench_ws = data_store.joinpath("fd_ensemble_benchmark")
        fd_bench_ws.mkdir(exist_ok=True)

        fd_results = benchmark_fd_ensemble(
            nx=nx_truth,
            ny=ny_truth,
            dx=dx_truth,
            n_cases=n_cases,
            n_workers_list=n_workers_list,
            base_workspace=fd_bench_ws,
            recharge_stack=recharge_stack,
            ghb=ghb,
        )

        fd_results_path = data_store.joinpath(
            f"fd_ensemble_benchmark_results_recharge{cells}.json"
        )
        with fd_results_path.open("w") as f:
            json.dump(fd_results, f, indent=4)
        written_outputs["fd_summary_json"] = str(fd_results_path)

        print(f"Saved FD benchmark results to {fd_results_path}")

    metadata_payload = {
        "suite": "recharge_change",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "grid": {
            "nx": int(nx_truth),
            "ny": int(ny_truth),
            "dx": float(dx_truth),
            "n_cells_total": int(cells),
        },
        "ensemble": {
            "n_cases": int(n_cases),
            "seed": int(args.seed),
            "base_recharge": float(base_recharge),
            "jitter": float(jitter),
            "stack_shape": [int(n_cases), int(ny_truth), int(nx_truth)],
        },
        "solver_parameters": {
            "warp": {
                "device": str(args.device),
                "max_cycles": 200,
                "nu_pre": 2,
                "nu_post": 2,
                "nu_coarse": 2,
                "rel_tol": 1.0e-5,
                "abs_tol_min": 1.0e-5,
                "warmup": 1,
                "check_every_no": 1,
                "min_coarse_cells": mg_min_coarse_cells,
            },
            "mf6": {
                "persistent_workers": True,
                "extract_heads": True,
                "workers": [int(w) for w in n_workers_list],
            },
            "fd": {
                "workers": [int(w) for w in n_workers_list],
            },
        },
        "runtime_mode_flags": {
            "ghb": bool(ghb),
            "run_warp": bool(run_warp),
            "run_mf6": bool(run_mf6),
            "run_fd": bool(run_fd),
        },
        "outputs": written_outputs,
    }
    metadata_path = data_store.joinpath(f"benchmark_metadata_recharge_{cells}.json")
    _write_metadata_if_enabled(metadata_path, metadata_payload, bool(args.write_metadata))

    return 0


if __name__ == "__main__":

    raise SystemExit(main())
