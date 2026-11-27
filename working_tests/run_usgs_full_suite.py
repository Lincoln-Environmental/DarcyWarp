"""Qualify every USGS workspace with native MF6 and DarcyWarp CUDA."""

import json
from pathlib import Path
from time import sleep

from working_tests.benchmark_usgs_remaining import consolidate, run_case


def find_native_reference(*, model_name, native_roots):
    """Return the first completed native MF6 workspace for a model.

    :param model_name: Archived workspace folder name.
    :param native_roots: Native output roots to search, in priority order.
    :return: Completed native workspace, or ``None`` while MF6 is running.
    """
    for root in native_roots:
        candidate = root / model_name
        if (candidate / "native_run.json").is_file():
            return candidate
    return None


def run_suite(*, model_root, suite_root, native_roots, executable, device,
              device_index, tolerances, poll_seconds, retry_failed):
    """Run/resume all archived workspaces as native references become ready.

    :param model_root: Root containing the archived USGS workspaces.
    :param suite_root: Root containing native and DarcyWarp benchmark outputs.
    :param native_roots: Native output roots to search, in priority order.
    :param executable: Native MODFLOW 6 executable.
    :param device: DarcyWarp Warp device.
    :param device_index: CUDA device index for peak-memory sampling.
    :param tolerances: Qualified parity gates shared by every model.
    :param poll_seconds: Delay before checking for a missing native receipt.
    :param retry_failed: Rerun terminal failures instead of reusing their receipts.
    :return: Per-model result records.
    """
    archives = sorted(path.parent for path in model_root.rglob("mfsim.nam"))
    if len(archives) != 74:
        raise RuntimeError(f"Expected 74 model workspaces, found {len(archives)}")
    output_root = suite_root / "darcywarp_qualification"
    results = []
    for index, archive in enumerate(archives, start=1):
        output = output_root / archive.name
        receipt = output / "case_status.json"
        if receipt.is_file():
            result = json.loads(receipt.read_text())
            if result["status"] == "passed" or not retry_failed:
                results.append(result)
                print(f"[{index}/74] reused {archive.name}: {result['status']}", flush=True)
                continue
            print(f"[{index}/74] retrying previous failure: {archive.name}", flush=True)
        native_directory = find_native_reference(model_name=archive.name,
            native_roots=native_roots)
        while native_directory is None:
            print(f"[{index}/74] waiting for original-settings MF6: {archive.name}",
                  flush=True)
            sleep(poll_seconds)
            native_directory = find_native_reference(model_name=archive.name,
                native_roots=native_roots)
        print(f"[{index}/74] CUDA mixed-fast: {archive.name}", flush=True)
        result = run_case(archive=archive, native_directory=native_directory,
            output=output, executable=executable, device=device,
            device_index=device_index, tolerances=tolerances)
        results.append(result)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "gpu_batch_status.json").write_text(
            json.dumps(results, indent=2, allow_nan=False) + "\n")
        consolidate(results=results, output=output_root)
        print(f"[{index}/74] {archive.name}: {result['status']}", flush=True)
    consolidate(results=results, output=output_root)
    return results


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    model_root = repository / "USGS_examples"
    suite_root = repository / "benchmark_results" / "usgs_full_suite_20261127"
    native_roots = [suite_root / "native",
        repository / "benchmark_results" / "usgs_remaining_20261127" / "native"]
    executable = Path("/bin/modflow/mf6")
    device = "cuda:0"
    device_index = 0
    tolerances = dict(head_gate=0.005, head_rmse_gate=1e-4,
        saturation_head_gate=0.1, flow_atol=1e-6, flow_rtol=1e-10,
        budget_rtol=1e-3)
    poll_seconds = 30
    retry_failed = False
    run_suite(model_root=model_root, suite_root=suite_root,
        native_roots=native_roots, executable=executable, device=device,
        device_index=device_index, tolerances=tolerances,
        poll_seconds=poll_seconds, retry_failed=retry_failed)
