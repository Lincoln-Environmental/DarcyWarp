from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
import csv
import time

import warp as wp


def _clean_kernel_name(activity_name: str) -> str:
    # Examples seen in Warp timing:
    # "forward kernel inc_loop"
    # "get_acceleration_a9fb4286_cuda_kernel_forward"
    # "memcpy DtoH"
    name = str(activity_name)

    # Strip the common prefix Warp prints
    name = name.replace("forward kernel ", "")
    name = name.replace("backward kernel ", "")

    # Drop common suffix patterns (keep it conservative)
    name = name.replace("_cuda_kernel_forward", "")
    name = name.replace("_cuda_kernel_backward", "")
    name = name.replace("_cpu_kernel_forward", "")
    name = name.replace("_cpu_kernel_backward", "")

    # Drop an 8 hex hash chunk often appended by Warp (example: _a9fb4286)
    name = re.sub(r"_[0-9a-fA-F]{8}\b", "", name)

    return name


def _filter_label(filter_value: int) -> str:
    if filter_value == wp.TIMING_KERNEL:
        return "kernel"
    if filter_value == wp.TIMING_KERNEL_BUILTIN:
        return "builtin"
    if filter_value == wp.TIMING_MEMCPY:
        return "memcpy"
    if filter_value == wp.TIMING_MEMSET:
        return "memset"
    if filter_value == wp.TIMING_GRAPH:
        return "graph"
    return f"filter={filter_value}"


def print_warp_bottlenecks(results: list[wp.TimingResult], indent: str = "", top_k: int = 30) -> None:
    # Aggregate totals
    total_ms = 0.0
    by_filter_ms = defaultdict(float)
    by_filter_count = defaultdict(int)

    by_kernel_ms = defaultdict(float)
    by_kernel_count = defaultdict(int)

    # Keep a per device split too, it is sometimes revealing
    by_device_ms = defaultdict(float)

    for r in results:
        ms = float(r.elapsed)
        total_ms += ms

        filt = int(r.filter)
        filt_label = _filter_label(filt)

        by_filter_ms[filt_label] += ms
        by_filter_count[filt_label] += 1

        dev = str(r.device)
        by_device_ms[dev] += ms

        if filt_label in ("kernel", "builtin"):
            kname = _clean_kernel_name(r.name)
            by_kernel_ms[kname] += ms
            by_kernel_count[kname] += 1

    print(f"{indent}=== Warp GPU activity summary ===")
    print(f"{indent}Total recorded GPU time: {total_ms:.3f} ms")

    print(f"{indent}\n{indent}By activity type:")
    for key in sorted(by_filter_ms.keys(), key=lambda k: by_filter_ms[k], reverse=True):
        ms = by_filter_ms[key]
        pct = 0.0 if total_ms <= 0.0 else 100.0 * ms / total_ms
        cnt = by_filter_count[key]
        print(f"{indent}  {key:8s}  {ms:12.3f} ms  {pct:6.2f}%  (count={cnt})")

    print(f"{indent}\n{indent}By device:")
    for dev in sorted(by_device_ms.keys(), key=lambda d: by_device_ms[d], reverse=True):
        ms = by_device_ms[dev]
        pct = 0.0 if total_ms <= 0.0 else 100.0 * ms / total_ms
        print(f"{indent}  {dev:8s}  {ms:12.3f} ms  {pct:6.2f}%")

    # Top kernels
    print(f"{indent}\n{indent}Top kernels (kernel + builtin) by total GPU time:")
    items = sorted(by_kernel_ms.items(), key=lambda kv: kv[1], reverse=True)
    for i, (kname, ms) in enumerate(items[:top_k], start=1):
        pct = 0.0 if total_ms <= 0.0 else 100.0 * ms / total_ms
        cnt = by_kernel_count[kname]
        print(f"{indent}  {i:2d}. {kname:50s}  {ms:12.3f} ms  {pct:6.2f}%  (count={cnt})")


def profile_one_solve(
    solve_callable,
    out_csv: Path | None = None,
    warmup_runs: int = 1,
) -> list[wp.TimingResult]:
    # Warmup to exclude JIT compilation and first time allocations
    for _ in range(warmup_runs):
        solve_callable()
        wp.synchronize()

    wp.timing_begin(cuda_filter=wp.TIMING_ALL, synchronize=True)  #
    solve_callable()
    results = wp.timing_end(synchronize=True)  #

    print_warp_bottlenecks(results, indent="", top_k=30)

    if out_csv is not None:
        out_csv.parent.mkdir(exist_ok=True)
        with out_csv.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["device", "filter", "filter_label", "name", "clean_name", "elapsed_ms"])
            for r in results:
                filt = int(r.filter)
                label = _filter_label(filt)
                writer.writerow([str(r.device), filt, label, str(r.name), _clean_kernel_name(r.name), float(r.elapsed)])

        print(f"\nWrote raw timing events to: {out_csv}")

    return results
