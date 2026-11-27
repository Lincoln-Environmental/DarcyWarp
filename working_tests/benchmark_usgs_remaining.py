"""Run the remaining archived USGS regional cases through MF6 and DarcyWarp.

Run from the repository root with ``python -m working_tests.benchmark_usgs_remaining``.
Each native MF6 run uses a copied workspace; DarcyWarp reads the archive itself.
"""
import csv
import json
from pathlib import Path
import threading
from time import perf_counter, sleep

import numpy as np

from working_tests.benchmark_regional_mf6 import audit, native_run


def sample_gpu_memory(*, stop, samples, device_index):
    """Sample process-independent GPU use while a model stage is active.

    :param stop: Event indicating the stage has ended.
    :param samples: List to receive MiB samples.
    :param device_index: NVIDIA device index used by DarcyWarp.
    """
    import subprocess

    while not stop.is_set():
        result = subprocess.run(
            ["nvidia-smi", f"--id={device_index}",
             "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
        if result.returncode == 0:
            try:
                samples.append(float(result.stdout.strip().splitlines()[0]))
            except (IndexError, ValueError):
                pass
        sleep(0.25)


def measured_darcywarp(*, archive, output, device, device_index):
    """Run the importer with its current default mixed-fast CUDA settings.

    :param archive: Untouched archived MF6 workspace.
    :param output: Separate DarcyWarp output directory.
    :param device: CUDA device string.
    :param device_index: NVIDIA device index for external memory sampling.
    :return: DarcyWarp report and sampled total GPU memory values.
    """
    from DARCY_WARP_PACKAGE.mf6_import import run_mf6_model

    samples = []
    stop = threading.Event()
    monitor = threading.Thread(target=sample_gpu_memory,
                                kwargs=dict(stop=stop, samples=samples,
                                            device_index=device_index), daemon=True)
    monitor.start()
    try:
        report = run_mf6_model(
            model_directory=archive,
            output_directory=output,
            device=device,
            precision="mixed_fast",
            solver_options={},
            overwrite=True,
        )
    finally:
        stop.set()
        monitor.join()
        output.mkdir(parents=True, exist_ok=True)
        (output / "gpu_memory_samples.json").write_text(
            json.dumps(samples) + "\n")
    return report, samples


def run_case(*, archive, native_directory, output, executable, device,
             device_index, tolerances):
    """Run native MF6, input/physical audit, DarcyWarp and solution audit.

    :param archive: Original MF6 input workspace.
    :param native_directory: Fresh copied workspace for native execution.
    :param output: Directory for this case's durable diagnostics.
    :param executable: Native MODFLOW 6 executable path.
    :param device: CUDA device string.
    :param device_index: NVIDIA device index.
    :param tolerances: Existing qualified parity tolerances.
    :return: Case status and timing/memory measurements.
    """
    result = dict(model=archive.name, status="running")
    output.mkdir(parents=True, exist_ok=True)
    try:
        native_record = native_directory / "native_run.json"
        if native_record.is_file():
            result["native"] = json.loads(native_record.read_text())
        else:
            result["native"] = native_run(
                archive=archive,
                directory=native_directory,
                executable=executable,
                save_saturation=True,
            )
        audit(archive=archive, directory=native_directory, output=output,
              device=device, **tolerances)
        started = perf_counter()
        report, memory_samples = measured_darcywarp(
            archive=archive, output=output, device=device,
            device_index=device_index,
        )
        result["darcywarp_wall_seconds"] = perf_counter() - started
        result["peak_gpu_mib"] = max(memory_samples, default=None)
        result["darcywarp"] = report
        audit(archive=archive, directory=native_directory, output=output,
              device=device, use_darcywarp_heads=True, **tolerances)
        result["status"] = "passed"
    except Exception as exception:
        result["status"] = "failed"
        result["failure"] = f"{type(exception).__name__}: {exception}"
        report_path = output / "darcywarp_report.json"
        if report_path.is_file():
            result["darcywarp"] = json.loads(report_path.read_text())
        memory_path = output / "gpu_memory_samples.json"
        if memory_path.is_file():
            samples = json.loads(memory_path.read_text())
            result["peak_gpu_mib"] = max(samples, default=None)
    (output / "case_status.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    return result


def consolidate(*, results, output):
    """Write a single timing, iteration, accuracy and budget results table.

    :param results: Per-model stage outcomes.
    :param output: Benchmark output directory.
    """
    columns = [
        "model", "status", "layers", "rows", "columns", "total_cells", "active_cells",
        "native_version", "native_seconds", "darcywarp_import_seconds",
        "operator_seconds", "reference_setup_seconds", "solver_setup_seconds",
        "solver_seconds", "darcywarp_wall_seconds", "end_to_end_seconds", "peak_gpu_mib",
        "newton_iterations", "krylov_iterations", "head_rmse_m", "head_max_abs_m",
        "head_rmse_raw_m", "head_max_abs_raw_m", "head_mae_m", "head_bias_m",
        "head_p01_m", "head_p50_m", "head_p95_m", "head_p99_m",
        "head_p01_raw_m", "head_p50_raw_m", "head_p95_raw_m", "head_p99_raw_m",
        "face_flow_rmse", "face_flow_max_abs", "physical_residual_rms",
        "relative_cell_imbalance", "chd_inflow", "chd_outflow", "ghb_inflow",
        "ghb_outflow", "drn_inflow", "drn_outflow", "rcha_inflow", "rcha_outflow",
        "chd_inflow_delta", "chd_outflow_delta", "ghb_inflow_delta", "ghb_outflow_delta",
        "drn_inflow_delta", "drn_outflow_delta", "rcha_inflow_delta", "rcha_outflow_delta",
        "tight_native_seconds", "tight_head_rmse_m", "tight_head_max_abs_m",
        "tight_solution_status",
        "failure",
    ]
    rows = []
    for result in results:
        row = {name: "" for name in columns}
        row.update(model=result["model"], status=result["status"],
                   peak_gpu_mib=result.get("peak_gpu_mib", ""),
                   failure=result.get("failure", ""))
        native = result.get("native", {})
        row.update(native_version=native.get("version", ""),
                   native_seconds=native.get("seconds", ""))
        report = result.get("darcywarp", {})
        if not report:
            report_path = output / result["model"] / "darcywarp_report.json"
            if report_path.is_file():
                report = json.loads(report_path.read_text())
        steps = report.get("steps", [])
        if report.get("shape"):
            row.update(layers=report["shape"][0], rows=report["shape"][1],
                       columns=report["shape"][2],
                       total_cells=int(np.prod(report["shape"])),
                       active_cells=report.get("active_cells", ""))
            row.update(darcywarp_import_seconds=report.get("import_seconds", ""),
                       darcywarp_wall_seconds=result.get(
                           "darcywarp_wall_seconds", report.get("run_seconds", "")),
                       end_to_end_seconds=report.get("run_seconds", ""))
        if steps:
            step = steps[-1]
            row.update(
                operator_seconds=step.get("operator_seconds", ""),
                reference_setup_seconds=step.get("reference_setup_seconds", ""),
                solver_setup_seconds=step.get("solver_setup_seconds", ""),
                solver_seconds=step.get("solve_seconds", ""),
                darcywarp_wall_seconds=result.get("darcywarp_wall_seconds", ""),
                newton_iterations=step.get("outer_iterations", ""),
                krylov_iterations=step.get("krylov_iterations", ""),
            )
        failed_step = report.get("failed_step", {})
        if failed_step:
            row.update(newton_iterations=failed_step.get("outer_iterations", ""),
                       krylov_iterations=failed_step.get("krylov_iterations", ""))
        solution_path = output / result["model"] / "solution.json"
        if solution_path.is_file():
            solution = json.loads(solution_path.read_text())
            raw_difference = solution.get("head_difference", {})
            difference = solution.get("datum_aligned_head_difference", raw_difference)
            flows = solution.get("cpu_face_difference", {})
            cpu = solution.get("cpu", {})
            row.update(head_rmse_m=difference.get("rms", ""),
                       head_max_abs_m=difference.get("max_abs", ""),
                       head_rmse_raw_m=raw_difference.get("rms", ""),
                       head_max_abs_raw_m=raw_difference.get("max_abs", ""),
                       head_mae_m=difference.get("mae", ""),
                       head_bias_m=difference.get("mean", ""),
                       face_flow_rmse=flows.get("rms", ""),
                       face_flow_max_abs=flows.get("max_abs", ""),
                       physical_residual_rms=cpu.get("physical_rms", ""),
                       relative_cell_imbalance=cpu.get("relative_cell_imbalance", ""))
            for percentile in ("p01", "p50", "p95", "p99"):
                row[f"head_{percentile}_m"] = difference.get("percentiles", {}).get(percentile, "")
                row[f"head_{percentile}_raw_m"] = raw_difference.get("percentiles", {}).get(percentile, "")
            for package in ("CHD", "GHB", "DRN", "RCHA"):
                summary = solution.get("packages", {}).get(package, {})
                row[f"{package.lower()}_inflow"] = summary.get("inflow", "")
                row[f"{package.lower()}_outflow"] = summary.get("outflow", "")
                if summary:
                    row[f"{package.lower()}_inflow_delta"] = (
                        summary.get("computed_inflow", summary.get("inflow", 0.))
                        - summary.get("inflow", 0.))
                    row[f"{package.lower()}_outflow_delta"] = (
                        summary.get("computed_outflow", summary.get("outflow", 0.))
                        - summary.get("outflow", 0.))
        tight_path = output / "tight_references" / result["model"] / "tight_reference.json"
        if tight_path.is_file():
            tight = json.loads(tight_path.read_text())
            row["tight_native_seconds"] = tight.get("native", {}).get("seconds", "")
            row["tight_solution_status"] = tight.get("stages", {}).get("solution", "")
            tight_solution = tight_path.parent / "solution" / "solution.json"
            if tight_solution.is_file():
                comparison = json.loads(tight_solution.read_text()).get("head_difference", {})
                row["tight_head_rmse_m"] = comparison.get("rms", "")
                row["tight_head_max_abs_m"] = comparison.get("max_abs", "")
        rows.append(row)
    with (output / "usgs_regional_results.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    model_root = repository / "USGS_examples"
    output_root = repository / "benchmark_results" / "usgs_remaining_20261127"
    native_root = output_root / "native"
    executable = Path("/bin/modflow/mf6")
    device = "cuda:0"
    device_index = 0
    already_qualified = {"0101_0102_0103_0105_MF6_SS_Unconfined_250",
                         "0104_0106_0107_0109_MF6_SS_Unconfined_250",
                         "0108_0110_0202_0203_MF6_SS_Unconfined_250"}
    archives = [path.parent for path in sorted(model_root.glob("models.*/**/mfsim.nam")
                 ) if path.parent.name not in already_qualified]
    tolerances = dict(head_gate=0.005, head_rmse_gate=1e-4,
                      saturation_head_gate=0.1, flow_atol=1e-6,
                      flow_rtol=1e-10, budget_rtol=1e-3)
    results = []
    for archive in archives:
        case_output = output_root / archive.name
        prior_status = case_output / "case_status.json"
        if prior_status.is_file():
            prior = json.loads(prior_status.read_text())
            results.append(prior)
            continue
        print(f"\n=== {archive.name} ===", flush=True)
        native_directory = native_root / archive.name
        result = run_case(archive=archive, native_directory=native_directory,
                          output=case_output, executable=executable,
                          device=device, device_index=device_index,
                          tolerances=tolerances)
        results.append(result)
        consolidate(results=results, output=output_root)
        print(f"=== {archive.name}: {result['status']} ===", flush=True)
    consolidate(results=results, output=output_root)
