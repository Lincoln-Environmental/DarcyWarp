"""Compare borderline DarcyWarp heads against tightly converged MF6 copies.

The original MF6 workspace and the first native run remain unchanged.
Configure diagnostic cases and tolerances in the execution guard below.
"""
import json
from pathlib import Path

from working_tests.benchmark_regional_mf6 import audit, native_run


def run_tight_reference(*, archive, solver_output, native_directory, output,
                        executable, device, tolerances, ims_dvclose):
    """Run and audit a separate MF6 reference with tighter IMS head closure.

    :param archive: Original unmodified MF6 workspace.
    :param solver_output: DarcyWarp output from the original initial heads.
    :param native_directory: New workspace for the diagnostic native run.
    :param output: Directory for native-head and final-head comparisons.
    :param executable: Native MF6 executable.
    :param device: CUDA device used by the physical-equation audit.
    :param tolerances: Existing parity thresholds, unchanged.
    :param ims_dvclose: Tighter diagnostic outer and inner head closure.
    :return: Native and audit stage outcomes.
    """
    result = {"model": archive.name, "ims_dvclose": ims_dvclose, "stages": {}}
    try:
        result["native"] = native_run(
            archive=archive,
            directory=native_directory,
            executable=executable,
            save_saturation=True,
            ims_dvclose=ims_dvclose,
        )
    except Exception as exception:
        result["stages"]["native"] = f"{type(exception).__name__}: {exception}"
        output.mkdir(parents=True, exist_ok=True)
        (output / "tight_reference.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n")
        return result

    native_output = output / "native_heads"
    try:
        audit(archive=archive, directory=native_directory, output=native_output,
              device=device, **tolerances)
        result["stages"]["native_heads"] = "passed"
    except Exception as exception:
        result["stages"]["native_heads"] = f"{type(exception).__name__}: {exception}"

    solution_output = output / "solution"
    solution_output.mkdir(parents=True, exist_ok=True)
    head_link = solution_output / "darcywarp.hds"
    if not head_link.exists():
        head_link.symlink_to(solver_output / "darcywarp.hds")
    try:
        audit(archive=archive, directory=native_directory, output=solution_output,
              device=device, use_darcywarp_heads=True, **tolerances)
        result["stages"]["solution"] = "passed"
    except Exception as exception:
        result["stages"]["solution"] = f"{type(exception).__name__}: {exception}"

    output.mkdir(parents=True, exist_ok=True)
    (output / "tight_reference.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    model_root = repository / "USGS_examples"
    benchmark_root = repository / "benchmark_results" / "usgs_remaining_20261127"
    native_root = benchmark_root / "native_tight"
    output_root = benchmark_root / "tight_references"
    executable = Path("/bin/modflow/mf6")
    device = "cuda:0"
    ims_dvclose = 1e-10
    model_names = [
        "0308_0309_0310_MF6_SS_Unconfined_250",
        "0405_0409_0410_MF6_SS_Unconfined_250",
        "0406_0407_0408_0416_MF6_SS_Unconfined_250",
    ]
    tolerances = dict(head_gate=0.005, head_rmse_gate=1e-4,
                      saturation_head_gate=0.1, flow_atol=1e-6,
                      flow_rtol=1e-10, budget_rtol=1e-3)
    for model_name in model_names:
        archive = next(model_root.glob(f"models.*/{model_name}"))
        native_directory = native_root / model_name
        output = output_root / model_name
        solver_output = benchmark_root / model_name
        result = run_tight_reference(
            archive=archive,
            solver_output=solver_output,
            native_directory=native_directory,
            output=output,
            executable=executable,
            device=device,
            tolerances=tolerances,
            ims_dvclose=ims_dvclose,
        )
        print(json.dumps(result, indent=2), flush=True)
