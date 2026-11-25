# SPDX-License-Identifier: AGPL-3.0-or-later
"""Worker selection and optional metadata output for the ensemble benchmarks."""

import json
import os
from pathlib import Path


def default_workers_arg() -> str:
    """Build the default worker counts for a benchmark sweep.

    :returns: Comma-separated worker counts up to the smaller of 24 and the
        available CPU count. An unavailable CPU count is treated as one.
    """
    cpu_count = max(1, os.cpu_count() or 1)
    candidates = (2, 4, 8, 16, 24)
    worker_counts = [count for count in candidates if count <= cpu_count]
    if cpu_count < candidates[-1] and cpu_count not in worker_counts:
        worker_counts.append(cpu_count)
    return ",".join(str(count) for count in worker_counts)


def effective_worker_count(requested_workers: int, n_cases: int) -> int:
    """Limit the process pool to the requested workers, cases and CPUs.

    :param requested_workers: Requested pool size, at least one.
    :param n_cases: Number of model cases in the ensemble.
    :returns: The smallest of the three limits, with a minimum of one.
    :raises ValueError: If the requested worker count is less than one.
    """
    requested_workers = int(requested_workers)
    if requested_workers < 1:
        raise ValueError("n_workers must be >= 1")
    cpu_count = os.cpu_count() or 1
    return max(1, min(requested_workers, int(n_cases), cpu_count))


def write_metadata_if_enabled(out_path: Path, payload: dict, enabled: bool) -> None:
    """Save a benchmark's metadata when requested by the caller.

    :param out_path: Destination JSON file. Its parent directories are created
        when writing is enabled.
    :param payload: JSON-serialisable metadata for the benchmark run.
    :param enabled: Whether to write the file and print its location.
    :returns: None.
    """
    if not enabled:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=4)
    print(f"Saved optional benchmark metadata to {out_path}")
