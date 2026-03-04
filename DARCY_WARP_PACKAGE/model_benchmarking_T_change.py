import time
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
import numpy as np
import argparse
from multiprocessing.shared_memory import SharedMemory
from datetime import datetime, timezone
from importlib import import_module

from DARCY_WARP_PACKAGE.project_base import data_store
from DARCY_WARP_PACKAGE.model_builder import build_base_fields


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


# Reuse the single-case runners from the recharge benchmarking module if present.
# Note: this import may fail in some environments; we still want this module to run.
_have_recharge_module = False
run_single_fd_case = None
try:
    from DARCY_WARP_PACKAGE.model_benchmarking_recharge_change import run_single_mf6_case as run_single_mf6_case
    from DARCY_WARP_PACKAGE.model_benchmarking_recharge_change import run_single_fd_case as _run_single_fd_case
    run_single_fd_case = _run_single_fd_case
    _have_recharge_module = True
except Exception:
    # Fallback definitions (minimally compatible) if import fails
    def run_single_mf6_case(
            case_id: int,
            nx: int = 100,
            ny: int = 100,
            dx: float = 100.0,
            T_field: np.ndarray | float = 3000.0,
            R_field: np.ndarray | float = 1.0e-4,
            base_workspace=None,
            ghb: bool = False,
    ) -> float:
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


def benchmark_fd_ensemble_persistent(
        nx: int,
        ny: int,
        dx: float,
        n_cases: int,
        n_workers_list: list[int],
        base_workspace,
        T_stack: np.ndarray,
        R_default: np.ndarray,
        ghb: bool = False,
) -> dict:
    base_workspace = Path(base_workspace)
    base_workspace.mkdir(parents=True, exist_ok=True)

    T_stack = np.asarray(T_stack, dtype=np.float64, order="C")
    R_default = np.asarray(R_default, dtype=np.float64, order="C")

    expected_T = (int(n_cases), int(ny), int(nx))
    if T_stack.shape != expected_T:
        raise ValueError(f"T_stack shape {T_stack.shape} expected {expected_T}")

    expected_R = (int(ny), int(nx))
    if R_default.shape != expected_R:
        raise ValueError(f"R_default shape {R_default.shape} expected {expected_R}")

    # Shared memory buffers
    T_shm = SharedMemory(create=True, size=int(T_stack.nbytes))
    R_shm = SharedMemory(create=True, size=int(R_default.nbytes))

    try:
        # Populate shared memory
        T_view = np.ndarray(T_stack.shape, dtype=T_stack.dtype, buffer=T_shm.buf)
        T_view[:, :, :] = T_stack[:, :, :]

        R_view = np.ndarray(R_default.shape, dtype=R_default.dtype, buffer=R_shm.buf)
        R_view[:, :] = R_default[:, :]

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

            batches: list[tuple[int, int, int]] = []
            start = 0
            for w in range(int(n_workers_i)):
                end = (w + 1) * int(n_cases) // int(n_workers_i)
                if end > start:
                    batches.append((w, start, end))
                start = end

            t0 = time.perf_counter()
            payloads = []

            with ProcessPoolExecutor(max_workers=int(n_workers_i)) as pool:
                futures = []
                for w, i0, i1 in batches:
                    futures.append(
                        pool.submit(
                            run_fd_persistent_worker_batch_shm,
                            int(w),
                            int(nx),
                            int(ny),
                            float(dx),
                            str(T_shm.name),
                            T_stack.shape,
                            str(T_stack.dtype),
                            str(R_shm.name),
                            R_default.shape,
                            str(R_default.dtype),
                            int(i0),
                            int(i1),
                            bool(ghb),
                        )
                    )

                for fut in as_completed(futures):
                    payloads.append(fut.result())

            total_wall = float(time.perf_counter() - t0)

            per_case = []
            total_cases_done = 0
            for p in payloads:
                per_case.extend([float(x) for x in p["per_case_wall_seconds"]])
                total_cases_done += int(p["n_cases"])

            if total_cases_done != int(n_cases):
                raise RuntimeError(f"Expected {int(n_cases)} cases, got {total_cases_done}")

            per_case_arr = np.asarray(per_case, dtype=float)
            per_case_arr.sort()

            p50 = float(per_case_arr[int(0.50 * (len(per_case_arr) - 1))])
            p95 = float(per_case_arr[int(0.95 * (len(per_case_arr) - 1))])

            entry = {
                "component": "fd_persistent_T",
                "nx": int(nx),
                "ny": int(ny),
                "n_cells_total": int(nx * ny),
                "n_cases": int(n_cases),
                "n_workers": int(n_workers_i),
                "total_wall_seconds": float(total_wall),
                "throughput_cases_per_second": float(n_cases / total_wall) if total_wall > 0.0 else float("nan"),
                "mean_case_seconds": float(per_case_arr.mean()),
                "p50_case_seconds": float(p50),
                "p95_case_seconds": float(p95),
                "max_case_seconds": float(per_case_arr.max()),
            }

            results[f"fd_persistent_T_workers_{n_workers_i}"] = entry

            out_path = base_workspace.joinpath(
                f"fd_T_persistent_benchmark_{nx}x{ny}_N{n_cases}_W{n_workers_i}.json")
            with out_path.open("w") as f:
                json.dump(entry, f, indent=2)

        return results

    finally:
        T_shm.close()
        T_shm.unlink()
        R_shm.close()
        R_shm.unlink()


def benchmark_mf6_ensemble(
        nx: int,
        ny: int,
        dx: float,
        n_cases: int,
        n_workers_list: list[int],
        base_workspace,
        ghb: bool = False,
        T_stack: np.ndarray | None = None,
        R_default: np.ndarray | float = 1.0e-4,
):
    """
    Benchmark MF6 by launching independent per-case runs in parallel (no persistent workers).

    This is simpler than the persistent-worker approach and works for varying transmissivity
    fields where updating NPF conductance in-place is not implemented in the persistent worker.
    """
    base_workspace = Path(base_workspace)
    base_workspace.mkdir(parents=True, exist_ok=True)

    domain, dem, T_field_ugly, R_field_default = build_base_fields(nx, ny, dx)

    if T_stack is not None:
        T_stack = np.asarray(T_stack, dtype=np.float64)
        expected = (int(n_cases), int(ny), int(nx))
        if T_stack.shape != expected:
            raise ValueError(f"T_stack shape {T_stack.shape} expected {expected}")

    results: dict[str, dict] = {}

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

        print(f"\n=== MF6 benchmark: {nx}x{ny}, {n_cases} cases, {int(n_workers_i)} workers ===")

        t_start = time.perf_counter()
        per_case_times: list[float] = []

        with ProcessPoolExecutor(max_workers=int(n_workers_i)) as pool:
            futures = []
            for case_id in range(int(n_cases)):
                if T_stack is None:
                    T_case = T_field_ugly
                else:
                    T_case = T_stack[int(case_id), :, :]

                futures.append(
                    pool.submit(
                        run_single_mf6_case,
                        int(case_id),
                        int(nx),
                        int(ny),
                        float(dx),
                        T_case,
                        R_default,
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
        results[f"mf6_workers_{int(n_workers_i)}"] = result_entry

        out_path = base_workspace.joinpath(f"mf6_benchmark_{nx}x{ny}_N{n_cases}_W{int(n_workers_i)}.json")
        with out_path.open("w") as f:
            json.dump(result_entry, f, indent=2)

    return results


def run_fd_persistent_worker_batch_shm(
        worker_id: int,
        nx: int,
        ny: int,
        dx: float,
        T_shm_name: str,
        T_shape: tuple[int, int, int],
        T_dtype_str: str,
        R_shm_name: str,
        R_shape: tuple[int, int],
        R_dtype_str: str,
        i0: int,
        i1: int,
        ghb: bool,
):
    import time
    import numpy as np
    from multiprocessing.shared_memory import SharedMemory
    from DARCY_WARP_PACKAGE.CPU_FD import run_fd_truth_forward

    t_shm = SharedMemory(name=str(T_shm_name))
    r_shm = SharedMemory(name=str(R_shm_name))

    try:
        T_stack = np.ndarray(T_shape, dtype=np.dtype(T_dtype_str), buffer=t_shm.buf)
        R_default = np.ndarray(R_shape, dtype=np.dtype(R_dtype_str), buffer=r_shm.buf)

        per_case = []
        for k in range(int(i0), int(i1)):
            t0 = time.perf_counter()
            _head, _elapsed_internal = run_fd_truth_forward(
                nx=int(nx),
                ny=int(ny),
                dx=float(dx),
                T_truth=T_stack[int(k), :, :],
                R_truth=R_default,
                use_ghb=bool(ghb),
            )
            t1 = time.perf_counter()
            per_case.append(float(t1 - t0))

        return {
            "worker_id": int(worker_id),
            "i0": int(i0),
            "i1": int(i1),
            "n_cases": int(i1 - i0),
            "per_case_wall_seconds": per_case,
        }

    finally:
        t_shm.close()
        r_shm.close()


def run_mf6_persistent_worker_batch_shm_T(
        worker_id: int,
        nx: int,
        ny: int,
        dx: float,
        T_shm_name: str,
        T_shape: tuple[int, int, int],
        T_dtype: str,
        R_shm_name: str,
        R_shape: tuple[int, int],
        R_dtype: str,
        case_start: int,
        case_end: int,
        run_root,
        ghb: bool,
        extract_heads: bool,
):
    """
    MF6 persistent worker that updates NPF (hk) derived from transmissivity T in-place.

    This mirrors the pattern used by the recharge persistent worker but updates the
    NPF package conductance (hk) for each case instead of the RCHA package.
    """
    import time
    import numpy as np
    from multiprocessing.shared_memory import SharedMemory
    from DARCY_WARP_PACKAGE.project_base import require_mf6
    import flopy

    t_shm = SharedMemory(name=str(T_shm_name))
    r_shm = SharedMemory(name=str(R_shm_name))

    try:
        T_stack = np.ndarray(tuple(T_shape), dtype=np.dtype(T_dtype), buffer=t_shm.buf)
        R_default = np.ndarray(tuple(R_shape), dtype=np.dtype(R_dtype), buffer=r_shm.buf)

        if T_stack.shape[1:] != (int(ny), int(nx)):
            raise ValueError(f"T_stack has shape {T_stack.shape}, expected (n_cases, {int(ny)}, {int(nx)})")

        run_root = Path(run_root)
        ws_worker = run_root.joinpath(f"mf6_T_worker_{int(worker_id):03d}")
        ws_worker.mkdir(parents=True, exist_ok=True)

        if int(case_end) <= int(case_start):
            raise ValueError(f"Empty case range: case_start={case_start}, case_end={case_end}")

        # Build template once using the first case's T -> hk conversion
        t_template0 = time.perf_counter()

        T0 = np.asarray(T_stack[int(case_start), :, :], dtype=float)
        thickness = 300.0
        hk0 = (T0 / float(thickness)).astype(float)

        _ = None
        # reuse existing make_mf_model builder if available
        try:
            from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model
            _ = make_mf_model(
                nx=int(nx),
                ny=int(ny),
                grid_size=float(dx),
                nper=1,
                workspace=ws_worker,
                hk=hk0,
                recharge=np.asarray(R_default, dtype=float),
                run=False,
                use_ghb=bool(ghb),
            )
        except Exception:
            # If builder not available, raise -- persistent approach requires writer
            raise

        mf6_exe = str(require_mf6())
        try:
            sim = flopy.mf6.MFSimulation.load(
                sim_ws=str(ws_worker),
                exe_name=mf6_exe,
                verbosity_level=0,
            )
        except TypeError:
            sim = flopy.mf6.MFSimulation.load(
                sim_ws=str(ws_worker),
                exe_name=mf6_exe,
            )

        model_names = list(sim.model_names)
        if len(model_names) < 1:
            raise RuntimeError("MFSimulation.load found no models in the workspace.")
        gwf = sim.get_model(model_names[0])

        npf = gwf.get_package("npf")
        if npf is None:
            raise RuntimeError("Could not find NPF package in loaded simulation.")

        # Keep timings
        template_build_time = float(time.perf_counter() - t_template0)

        update_times: list[float] = []
        write_times: list[float] = []
        run_times: list[float] = []
        extract_times: list[float] = []
        total_times: list[float] = []

        name = gwf.name

        for case_id in range(int(case_start), int(case_end)):
            T_arr = np.asarray(T_stack[int(case_id), :, :], dtype=float)
            hk_arr = (T_arr / float(thickness)).astype(float)

            # (1) update timing
            t_update0 = time.perf_counter()

            if hk_arr.shape != (int(ny), int(nx)):
                raise ValueError(f"hk has shape {hk_arr.shape}, expected {(int(ny), int(nx))}")

            updated = False
            update_errors: list[str] = []

            # Try common flopy patterns to set npf k values in-place.
            try:
                npf.k.set_data(hk_arr, key=0)
                updated = True
            except Exception as exc:
                update_errors.append(f"npf.k.set_data(array, key=0) failed: {exc}")

            if not updated:
                try:
                    npf.k.set_data({0: hk_arr})
                    updated = True
                except Exception as exc:
                    update_errors.append(f"npf.k.set_data({{0: array}}) failed: {exc}")

            if not updated:
                if hasattr(npf, "k") and hasattr(npf.k, "array"):
                    try:
                        npf.k.array[:] = hk_arr
                        updated = True
                    except Exception as exc:
                        update_errors.append(f"npf.k.array[:] assignment failed: {exc}")
                else:
                    update_errors.append("npf.k.array assignment path unavailable")

            if not updated:
                if hasattr(npf, "k"):
                    try:
                        npf.k = hk_arr
                        updated = True
                    except Exception as exc:
                        update_errors.append(f"npf.k direct assignment failed: {exc}")
                else:
                    update_errors.append("npf.k attribute unavailable")

            if not updated:
                details = "; ".join(update_errors)
                if len(details) > 1200:
                    details = details[:1200] + " ... [truncated]"
                raise RuntimeError(
                    "Failed to update MF6 NPF hk in persistent worker "
                    f"for case_id={int(case_id)}. {details}"
                )

            t_update1 = time.perf_counter()
            update_times.append(float(t_update1 - t_update0))

            # (2) write timing
            t_write0 = time.perf_counter()
            wrote = False
            try:
                npf.write()
                wrote = True
            except Exception:
                wrote = False

            if not wrote:
                try:
                    npf.write_file()
                    wrote = True
                except Exception:
                    wrote = False

            if not wrote:
                sim.write_simulation()

            t_write1 = time.perf_counter()
            write_times.append(float(t_write1 - t_write0))

            # (3) run timing
            t_run0 = time.perf_counter()
            try:
                ok, _ = sim.run_simulation(silent=True, report=False)
            except TypeError:
                ok, _ = sim.run_simulation()
            t_run1 = time.perf_counter()
            run_times.append(float(t_run1 - t_run0))

            if not bool(ok):
                raise RuntimeError(f"MF6 failed for case_id={case_id}")

            # (4) extract timing
            t_ex0 = time.perf_counter()
            if bool(extract_heads):
                model_ws = Path(gwf.model_ws)
                try:
                    head_obj = gwf.output.head()
                    head_filename = Path(str(head_obj.filename))
                    if head_filename.is_absolute():
                        heads_path = head_filename
                    else:
                        heads_path = model_ws.joinpath(head_filename)
                except Exception:
                    heads_path = model_ws.joinpath(f"{gwf.name}.hds")

                if not heads_path.exists():
                    found = list(model_ws.glob("*.hds"))
                    raise FileNotFoundError(
                        f"Head file not found. expected={heads_path} model_ws={model_ws} "
                        f"found_hds={[p.name for p in found]}"
                    )

                _ = None
                try:
                    from DARCY_WARP_PACKAGE.modflow_truth import _extract_heads
                    _ = _extract_heads(heads_path)
                except Exception:
                    # best-effort fallback using flopy utils
                    try:
                        hdobj = flopy.utils.HeadFile(str(heads_path))
                        _ = hdobj.get_data(totim=1.0)[0, :, :]
                    except Exception:
                        _ = None

            t_ex1 = time.perf_counter()
            extract_times.append(float(t_ex1 - t_ex0))

            total_times.append(float(t_ex1 - t_update0))

        return {
            "worker_idx": int(worker_id),
            "workspace": str(ws_worker),
            "template_build_time": float(template_build_time),
            "n_cases": int(int(case_end) - int(case_start)),
            "case_start": int(case_start),
            "case_end": int(case_end),
            "update_times": update_times,
            "write_times": write_times,
            "run_times": run_times,
            "extract_times": extract_times,
            "total_times": total_times,
        }

    finally:
        t_shm.close()
        r_shm.close()


def benchmark_mf6_ensemble_persistent_T(
        nx: int,
        ny: int,
        dx: float,
        n_cases: int,
        n_workers_list: list[int],
        base_workspace,
        ghb: bool = False,
        T_stack: np.ndarray | None = None,
        R_default: np.ndarray | float = 1.0e-4,
        extract_heads: bool = True,
):
    """
    Benchmark MF6 using persistent workers that update transmissivity (T) in-place by
    converting T -> hk and modifying the NPF package per-case.
    """
    base_workspace = Path(base_workspace)
    base_workspace.mkdir(parents=True, exist_ok=True)

    domain, dem, T_field_ugly, R_field_default = build_base_fields(nx, ny, dx)

    if T_stack is None:
        T_stack = np.empty((int(n_cases), int(ny), int(nx)), dtype=np.float64)
        T_stack[:, :, :] = np.asarray(T_field_ugly, dtype=np.float64)[None, :, :]
    else:
        T_stack = np.asarray(T_stack, dtype=np.float64)
        if T_stack.shape != (int(n_cases), int(ny), int(nx)):
            raise ValueError(f"T_stack shape {T_stack.shape} expected {(int(n_cases), int(ny), int(nx))}")

    R_default = np.asarray(R_field_default if R_default is None else R_default, dtype=np.float64)
    expected_R = (int(ny), int(nx))
    if R_default.shape != expected_R:
        # allow scalar
        if R_default.size == 1:
            R_default = np.full(expected_R, float(R_default))
        else:
            raise ValueError(f"R_default shape {R_default.shape} expected {expected_R}")

    # Shared memory buffers
    T_shm = SharedMemory(create=True, size=int(T_stack.nbytes))
    R_shm = SharedMemory(create=True, size=int(R_default.nbytes))

    try:
        T_view = np.ndarray(T_stack.shape, dtype=T_stack.dtype, buffer=T_shm.buf)
        T_view[:, :, :] = T_stack[:, :, :]

        R_view = np.ndarray(R_default.shape, dtype=R_default.dtype, buffer=R_shm.buf)
        R_view[:, :] = R_default[:, :]

        results: dict[str, dict] = {}

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

            batches: list[tuple[int, int, int]] = []
            start = 0
            for w in range(int(n_workers_i)):
                end = (w + 1) * int(n_cases) // int(n_workers_i)
                if end > start:
                    batches.append((w, start, end))
                start = end

            t_wall0 = time.perf_counter()
            worker_payloads = []

            run_root = base_workspace.joinpath(f"mf6_T_persistent_workers_{n_workers_i}")
            run_root.mkdir(parents=True, exist_ok=True)

            with ProcessPoolExecutor(max_workers=n_workers_i) as pool:
                futures = []
                for w, i0, i1 in batches:
                    futures.append(
                        pool.submit(
                            run_mf6_persistent_worker_batch_shm_T,
                            w,
                            int(nx),
                            int(ny),
                            float(dx),
                            str(T_shm.name),
                            T_stack.shape,
                            str(T_stack.dtype),
                            str(R_shm.name),
                            R_view.shape,
                            str(R_view.dtype),
                            int(i0),
                            int(i1),
                            str(run_root),
                            bool(ghb),
                            bool(extract_heads),
                        )
                    )

                for fut in as_completed(futures):
                    worker_payloads.append(fut.result())

            t_wall1 = time.perf_counter()
            total_wall = float(t_wall1 - t_wall0)

            # aggregate
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

            results[f"mf6_T_persistent_workers_{n_workers_i}"] = entry

            out_path = base_workspace.joinpath(f"mf6_T_persistent_benchmark_{nx}x{ny}_N{n_cases}_W{n_workers_i}.json")
            with out_path.open("w") as f:
                json.dump(entry, f, indent=2)

        return results

    finally:
        T_shm.close()
        T_shm.unlink()
        R_shm.close()
        R_shm.unlink()


def _benchmark_warp_multigrid_class_solvers_T_core(
        nx: int,
        ny: int,
        dx: float,
        n_cases: int,
        base_workspace,
        T_stack: np.ndarray | None = None,
        warp_device: str = "cuda:0",
        n_iter: int = 200,
        nu_pre: int = 2,
        nu_post: int = 2,
        nu_coarse: int = 2,
        rel_tol: float = 5.0e-7,
        abs_tol_min: float = 5.0e-7,
        use_ghb: bool = False,
        gh_alpha: float = 1.0,
        update_mode: str = "full",
        update_diag_preconditioner: bool = False,
        warmup: int = 1,
        verify_t_upload: bool = True,
        allow_experimental_update_modes: bool = False,
        solver_module: str = "DARCY_WARP_PACKAGE.warped_darcy",
        solver_variant: str = "warp_kcycle_default",
        solve_kwargs: dict | None = None,
        mg_min_coarse_cells: int | None = 500,
) -> dict:
    """
    Benchmark WarpDarcySolver by reusing one solver instance and updating the transmissivity
    in place for each case.

    Parameters
    ----------
    update_mode:
        One of {"full", "fast", "ultrafast"}:
          - full: update all MG levels + per-level diagonal preconditioner
          - fast: fine-level update only (skip coarse rebuild, keep preconditioner unless requested)
          - ultrafast: device-only upload of fine-level T (no host staging) when possible
        fast/ultrafast are experimental; requires allow_experimental_update_modes=True.
    update_diag_preconditioner:
        If True, update fine-level diagonal preconditioner during fast/ultrafast updates.
    warmup:
        Number of warmup iterations (not timed) to trigger kernel compilation + caching.
    """
    os.environ["DARCY_FLOAT"] = "float64"
    WarpDarcySolver = import_module(str(solver_module)).WarpDarcySolver

    base_workspace = Path(base_workspace)
    base_workspace.mkdir(parents=True, exist_ok=True)

    domain, dem, T_field_ugly, R_field_ugly = build_base_fields(nx, ny, dx)

    if T_stack is not None:
        T_stack = np.asarray(T_stack, dtype=np.float64, order="C")
        expected = (int(n_cases), int(ny), int(nx))
        if T_stack.shape != expected:
            raise ValueError(f"T_stack shape {T_stack.shape} expected {expected}")

    update_mode = str(update_mode).lower().strip()
    if update_mode not in {"full", "fast", "ultrafast"}:
        raise ValueError("update_mode must be one of {'full','fast','ultrafast'}")
    if update_mode in {"fast", "ultrafast"} and (not allow_experimental_update_modes):
        raise ValueError(
            "update_mode=fast/ultrafast is experimental. "
            "Pass allow_experimental_update_modes=True to proceed."
        )
    if update_mode in {"fast", "ultrafast"}:
        print(
            f"[warn] update_mode={update_mode} is experimental and approximate; "
            "coarse/MG levels are stale. Best avoided for production results."
        )

    print(
        f"\n=== Warp class benchmark ({solver_variant}): {nx}x{ny}, {n_cases} cases, "
        f"device={warp_device}, update_mode={update_mode} ==="
    )

    single_solver = WarpDarcySolver(
        nx=int(nx),
        ny=int(ny),
        dx=float(dx),
        device=str(warp_device),
        use_ghb=bool(use_ghb),
        solver_type="pcg",
        aq_thickness=300
    )

    single_solver.build_from_truth_inputs(
        T_truth=T_field_ugly,
        R_truth=R_field_ugly,
        gh_alpha=float(gh_alpha),
    )

    mg_min_coarse_cells_norm = None if mg_min_coarse_cells is None else int(mg_min_coarse_cells)
    if mg_min_coarse_cells_norm is not None and mg_min_coarse_cells_norm < 1:
        raise ValueError("mg_min_coarse_cells must be >= 1 when provided")

    # build hierarchy once
    single_solver.build_hierarchy(
        max_levels=6,
        min_coarse_n=4,
        min_coarse_cells=mg_min_coarse_cells_norm,
    )

    if single_solver.T_field_host is None:
        raise RuntimeError("Expected single_solver.T_field_host to be initialized after build_from_truth_inputs().")

    target_dtype = single_solver.T_field_host.dtype
    env_verify = str(os.environ.get("DARCY_VERIFY_T_UPLOAD", "")).strip().lower()
    if env_verify in {"0", "false", "no", "n"}:
        verify_upload = False
    elif env_verify in {"1", "true", "yes", "y"}:
        verify_upload = True
    else:
        verify_upload = bool(verify_t_upload)
    verified_once = False

    def _verify_t_upload(T_arr: np.ndarray) -> None:
        nonlocal verified_once
        host_ok = np.allclose(single_solver.T_field_host, T_arr)
        dev_arr = single_solver.T_wp.numpy()
        dev_ok = np.allclose(dev_arr, T_arr)
        max_abs = float(np.max(np.abs(dev_arr - T_arr)))
        print(
            f"[verify] T upload host_ok={host_ok} dev_ok={dev_ok} max_abs_diff={max_abs:.3e} "
            f"host_min={float(T_arr.min()):.3e} host_max={float(T_arr.max()):.3e}"
        )
        if not (host_ok and dev_ok):
            raise RuntimeError("T upload verification failed: device/host mismatch.")
        verified_once = True

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

    # Warmup (not timed)
    for k in range(int(max(0, warmup))):
        T_case = T_field_ugly if T_stack is None else T_stack[int(k) % int(n_cases), :, :]
        T_arr = np.asarray(T_case, dtype=target_dtype, order="C")
        if update_mode == "full":
            single_solver.update_T_in_place(T_arr)
        elif update_mode == "fast":
            single_solver.update_T_in_place_fast(T_arr, update_diag_preconditioner=bool(update_diag_preconditioner))
        else:
            single_solver.update_T_in_place_ultrafast(T_arr, update_diag_preconditioner=bool(update_diag_preconditioner))

        if verify_upload and not verified_once:
            _verify_t_upload(T_arr)
        _ = single_solver.solve_multigrid_kcycle(**warmup_kwargs)

    per_case_times_single: list[float] = []
    per_case_update_times: list[float] = []
    per_case_solve_times: list[float] = []

    if verify_upload and not verified_once:
        T_case0 = T_field_ugly if T_stack is None else T_stack[0, :, :]
        T_arr0 = np.asarray(T_case0, dtype=target_dtype, order="C")
        if update_mode == "full":
            single_solver.update_T_in_place(T_arr0)
        elif update_mode == "fast":
            single_solver.update_T_in_place_fast(T_arr0, update_diag_preconditioner=bool(update_diag_preconditioner))
        else:
            single_solver.update_T_in_place_ultrafast(T_arr0, update_diag_preconditioner=bool(update_diag_preconditioner))
        _verify_t_upload(T_arr0)

    t_start_single = time.perf_counter()

    for k in range(int(n_cases)):
        T_case = T_field_ugly if T_stack is None else T_stack[int(k), :, :]
        T_arr = np.asarray(T_case, dtype=target_dtype, order="C")

        t0 = time.perf_counter()

        t_up0 = time.perf_counter()
        if update_mode == "full":
            single_solver.update_T_in_place(T_arr)
        elif update_mode == "fast":
            single_solver.update_T_in_place_fast(T_arr, update_diag_preconditioner=bool(update_diag_preconditioner))
        else:
            single_solver.update_T_in_place_ultrafast(T_arr, update_diag_preconditioner=bool(update_diag_preconditioner))
        t_up1 = time.perf_counter()

        _head = single_solver.solve_multigrid_kcycle(**solve_call_kwargs)
        t_sol1 = time.perf_counter()

        per_case_update_times.append(float(t_up1 - t_up0))
        per_case_solve_times.append(float(t_sol1 - t_up1))
        per_case_times_single.append(float(t_sol1 - t0))

    total_wall_single = float(time.perf_counter() - t_start_single)
    per_case_single = np.asarray(per_case_times_single, dtype=float)
    update_arr = np.asarray(per_case_update_times, dtype=float)
    solve_arr = np.asarray(per_case_solve_times, dtype=float)

    single_entry = {
        "nx": int(nx),
        "ny": int(ny),
        "n_cells_total": int(nx * ny),
        "n_cases": int(n_cases),
        "device": str(warp_device),
        "update_mode": str(update_mode),
        "solver_module": str(solver_module),
        "solver_variant": str(solver_variant),
        "update_diag_preconditioner": bool(update_diag_preconditioner),
        "warmup": int(max(0, warmup)),
        "total_wall_seconds": float(total_wall_single),
        "mean_case_seconds": float(per_case_single.mean()),
        "mean_update_seconds": float(update_arr.mean()),
        "mean_solve_seconds": float(solve_arr.mean()),
        "min_case_seconds": float(per_case_single.min()),
        "max_case_seconds": float(per_case_single.max()),
        "throughput_cases_per_second": float(n_cases / total_wall_single) if total_wall_single > 0.0 else float("nan"),
    }

    results = {"warp_class_single": single_entry}

    results_path = base_workspace.joinpath(f"warp_class_benchmark_T_{nx}x{ny}_N{n_cases}.json")
    with results_path.open("w") as f:
        json.dump(results, f, indent=2)

    return results


def benchmark_warp_multigrid_class_solvers_T(
        nx: int,
        ny: int,
        dx: float,
        n_cases: int,
        base_workspace,
        T_stack: np.ndarray | None = None,
        warp_device: str = "cuda:0",
        n_iter: int = 200,
        nu_pre: int = 2,
        nu_post: int = 2,
        nu_coarse: int = 2,
        rel_tol: float = 5.0e-7,
        abs_tol_min: float = 5.0e-7,
        use_ghb: bool = False,
        gh_alpha: float = 1.0,
        update_mode: str = "full",
        update_diag_preconditioner: bool = False,
        warmup: int = 1,
        verify_t_upload: bool = True,
        allow_experimental_update_modes: bool = False,
        mg_min_coarse_cells: int | None = 500,
) -> dict:
    """
    Benchmark the existing Warp K-cycle path (`warped_darcy`) for T-change ensembles.
    """
    return _benchmark_warp_multigrid_class_solvers_T_core(
        nx=nx,
        ny=ny,
        dx=dx,
        n_cases=n_cases,
        base_workspace=base_workspace,
        T_stack=T_stack,
        warp_device=warp_device,
        n_iter=n_iter,
        nu_pre=nu_pre,
        nu_post=nu_post,
        nu_coarse=nu_coarse,
        rel_tol=rel_tol,
        abs_tol_min=abs_tol_min,
        use_ghb=use_ghb,
        gh_alpha=gh_alpha,
        update_mode=update_mode,
        update_diag_preconditioner=update_diag_preconditioner,
        warmup=warmup,
        verify_t_upload=verify_t_upload,
        allow_experimental_update_modes=allow_experimental_update_modes,
        solver_module="DARCY_WARP_PACKAGE.warped_darcy",
        solver_variant="warp_kcycle_default",
        solve_kwargs=None,
        mg_min_coarse_cells=mg_min_coarse_cells,
    )


def main(argv: list[str] | None = None) -> int:
    """
    CLI entry point for T-variation ensemble benchmarks.

    Builds a deterministic T_stack from seed, then runs any selected
    benchmark components and writes JSON summaries to data_store.
    """
    parser = argparse.ArgumentParser(description="Run ensemble benchmarks varying transmissivity (T).")
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=1000)
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--n_cases", type=int, default=48)
    parser.add_argument("--workers", type=str, default=_default_workers_arg())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ghb", action="store_true")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--warp_update_mode", type=str, default="full", choices=["full", "fast", "ultrafast"])
    parser.add_argument(
        "--warp_allow_experimental_update_modes",
        action="store_true",
        help="Allow experimental warp update modes (fast/ultrafast).",
    )
    parser.add_argument("--warp_update_diag", action="store_true")
    parser.add_argument("--warp_warmup", type=int, default=1)
    parser.add_argument("--warp_no_verify_t_upload", action="store_true")
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

    # Build a deterministic stack of T fields by applying small multiplicative
    # lognormal perturbations to a base heterogeneous field.
    domain, dem, T_base_field, R_default = build_base_fields(nx_truth, ny_truth, dx_truth)

    # multiplicative lognormal noise parameters (moderate variability)
    sigma = 0.25

    T_stack = np.empty((int(n_cases), int(ny_truth), int(nx_truth)), dtype=np.float64)
    for case_id in range(int(n_cases)):
        # draw per-cell multiplicative factor from lognormal around 1.0
        mult = rng.lognormal(mean=0.0, sigma=sigma, size=(int(ny_truth), int(nx_truth)))
        T_stack[int(case_id), :, :] = np.asarray(T_base_field, dtype=np.float64) * mult

    written_outputs: dict[str, str] = {}

    if run_warp:
        warp_class_ws = data_store.joinpath(f"warp_class_T_ensemble_benchmark_{cells}")
        warp_class_ws.mkdir(exist_ok=True)

        warp_class_results = benchmark_warp_multigrid_class_solvers_T(
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
            rel_tol=1.0e-7,
            abs_tol_min=1.0e-7,
            use_ghb=ghb,
            gh_alpha=1.0,
            T_stack=T_stack,
            update_mode=str(args.warp_update_mode),
            update_diag_preconditioner=bool(args.warp_update_diag),
            warmup=int(args.warp_warmup),
            verify_t_upload=not bool(args.warp_no_verify_t_upload),
            allow_experimental_update_modes=bool(args.warp_allow_experimental_update_modes),
            mg_min_coarse_cells=mg_min_coarse_cells,
        )

        warp_class_results_path = data_store.joinpath(f"warp_class_T_ensemble_benchmark_results_{cells}.json")
        with warp_class_results_path.open("w") as f:
            json.dump(warp_class_results, f, indent=4)
        written_outputs["warp_summary_json"] = str(warp_class_results_path)

        print(f"Saved Warp class benchmark results to {warp_class_results_path}")

    if run_mf6:
        mp.set_start_method("spawn", force=True)

        mf6_bench_ws = data_store.joinpath("mf6_T_ensemble_benchmark")
        mf6_bench_ws.mkdir(exist_ok=True)

        mf6_results = benchmark_mf6_ensemble_persistent_T(
            nx=nx_truth,
            ny=ny_truth,
            dx=dx_truth,
            n_cases=n_cases,
            n_workers_list=n_workers_list,
            base_workspace=mf6_bench_ws,
            ghb=ghb,
            T_stack=T_stack,
            R_default=R_default,
            extract_heads=True,
        )

        mf6_results_path = data_store.joinpath(f"mf6_T_ensemble_benchmark_results_{cells}.json")
        with mf6_results_path.open("w") as f:
            json.dump(mf6_results, f, indent=4)
        written_outputs["mf6_summary_json"] = str(mf6_results_path)

        print(f"Saved MF6 benchmark results to {mf6_results_path}")

    if run_fd:
        fd_bench_ws = data_store.joinpath("fd_T_ensemble_benchmark")
        fd_bench_ws.mkdir(exist_ok=True)

        fd_results = benchmark_fd_ensemble_persistent(
            nx=nx_truth,
            ny=ny_truth,
            dx=dx_truth,
            n_cases=n_cases,
            n_workers_list=n_workers_list,
            base_workspace=fd_bench_ws,
            T_stack=T_stack,
            R_default=R_default,
            ghb=ghb,
        )

        fd_results_path = data_store.joinpath(f"fd_T_ensemble_benchmark_results_{cells}.json")
        with fd_results_path.open("w") as f:
            json.dump(fd_results, f, indent=4)
        written_outputs["fd_summary_json"] = str(fd_results_path)

        print(f"Saved FD benchmark results to {fd_results_path}")

    metadata_payload = {
        "suite": "t_change",
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
            "t_stack_shape": [int(n_cases), int(ny_truth), int(nx_truth)],
            "t_stack_sigma_lognormal": float(sigma),
        },
        "solver_parameters": {
            "warp": {
                "device": str(args.device),
                "max_cycles": 200,
                "nu_pre": 2,
                "nu_post": 2,
                "nu_coarse": 2,
                "rel_tol": 1.0e-7,
                "abs_tol_min": 1.0e-7,
                "warmup": int(args.warp_warmup),
                "check_every_no": 1,
                "update_mode": str(args.warp_update_mode),
                "update_diag_preconditioner": bool(args.warp_update_diag),
                "verify_t_upload": not bool(args.warp_no_verify_t_upload),
                "allow_experimental_update_modes": bool(args.warp_allow_experimental_update_modes),
                "min_coarse_cells": mg_min_coarse_cells,
            },
            "mf6": {
                "persistent_workers": True,
                "extract_heads": True,
                "workers": [int(w) for w in n_workers_list],
            },
            "fd": {
                "persistent_workers": True,
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
    metadata_path = data_store.joinpath(f"benchmark_metadata_T_{cells}.json")
    _write_metadata_if_enabled(metadata_path, metadata_payload, bool(args.write_metadata))

    return 0


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    raise SystemExit(main())
