from __future__ import annotations

import json
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path


def _select_warp_entry(warp_summary: dict) -> Optional[dict]:
    """
    Pick a Warp summary entry from known schemas.

    Preference order:
      1) in-place recharge benchmark entry
      2) generic single entry used by T benchmark
      3) host-only recharge entry
      4) first dict-like value containing throughput/wall fields
    """
    preferred_keys = (
        "warp_class_single_in_place",
        "warp_class_single",
        "warp_class_single_host_only",
    )

    for key in preferred_keys:
        val = warp_summary.get(key)
        if isinstance(val, dict):
            return val

    for val in warp_summary.values():
        if not isinstance(val, dict):
            continue
        if (
            "throughput_cases_per_second" in val
            or "total_wall_seconds" in val
            or "n_cases" in val
        ):
            return val
    return None


def plot_throughput_vs_workers(
    mf6_summary_json: Path,
    out_dir: Path,
    warp_summary_json: Optional[Path] = None,
    title: str = "Ensemble throughput vs MF6 worker processes",
    filename: str = "throughput_vs_workers.png",
) -> None:
    """
    Plot MF6 throughput (cases/s) against process count, with optional Warp reference line.

    :param mf6_summary_json: Path to MF6 summary JSON (dict keyed by mf6_persistent_workers_*).
    :param out_dir: Output directory for plot PNG.
    :param warp_summary_json: Optional Warp summary JSON path to overlay throughput reference.
    :param title: Plot title.
    :param filename: Output PNG filename.
    """
    mf6_summary_json = Path(mf6_summary_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mf6_summary_json.open("r", encoding="utf-8") as f:
        mf6 = json.load(f)

    warp_throughput = None
    if warp_summary_json is not None:
        warp_summary_json = Path(warp_summary_json)
        with warp_summary_json.open("r", encoding="utf-8") as f:
            warp = json.load(f)
        warp_entry = _select_warp_entry(warp)
        if isinstance(warp_entry, dict) and "throughput_cases_per_second" in warp_entry:
            warp_throughput = float(warp_entry["throughput_cases_per_second"])

    workers: list[int] = []
    throughput: list[float] = []

    for key in mf6:
        entry = mf6[key]
        workers.append(int(entry["n_workers"]))
        throughput.append(float(entry["throughput_cases_per_second"]))

    workers_arr = np.asarray(workers, dtype=int)
    order = np.argsort(workers_arr)
    workers_arr = workers_arr[order]
    throughput_arr = np.asarray(throughput, dtype=float)[order]

    plt.figure()
    plt.plot(workers_arr, throughput_arr, marker="o", label="MF6 (persistent workers)")
    if warp_throughput is not None:
        plt.axhline(float(warp_throughput), linestyle="--", label="Warp (single GPU)")

    plt.xlabel("MF6 worker processes")
    plt.ylabel("Throughput (cases per second)")
    plt.yscale("log")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(filename), dpi=200)
    plt.close()


def plot_walltime_vs_workers(
    mf6_summary_json: Path,
    out_dir: Path,
    warp_summary_json: Optional[Path] = None,
    title: str = "Total wall time to complete N cases vs MF6 worker processes",
    filename: str = "walltime_vs_workers.png",
) -> None:
    """
    Plot MF6 total wall time against process count, with optional Warp reference line.

    :param mf6_summary_json: Path to MF6 summary JSON.
    :param out_dir: Output directory for plot PNG.
    :param warp_summary_json: Optional Warp summary JSON path to overlay wall time reference.
    :param title: Plot title.
    :param filename: Output PNG filename.
    """
    mf6_summary_json = Path(mf6_summary_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mf6_summary_json.open("r", encoding="utf-8") as f:
        mf6 = json.load(f)

    warp_wall = None
    if warp_summary_json is not None:
        warp_summary_json = Path(warp_summary_json)
        with warp_summary_json.open("r", encoding="utf-8") as f:
            warp = json.load(f)
        warp_entry = _select_warp_entry(warp)
        if isinstance(warp_entry, dict) and "total_wall_seconds" in warp_entry:
            warp_wall = float(warp_entry["total_wall_seconds"])

    workers: list[int] = []
    wall: list[float] = []

    for key in mf6:
        entry = mf6[key]
        workers.append(int(entry["n_workers"]))
        wall.append(float(entry["total_wall_seconds"]))

    workers_arr = np.asarray(workers, dtype=int)
    order = np.argsort(workers_arr)
    workers_arr = workers_arr[order]
    wall_arr = np.asarray(wall, dtype=float)[order]

    plt.figure()
    plt.plot(workers_arr, wall_arr, marker="o", label="MF6 (persistent workers)")
    if warp_wall is not None:
        plt.axhline(float(warp_wall), linestyle="--", label="Warp (single GPU)")

    plt.xlabel("MF6 worker processes")
    plt.ylabel("Total wall time (s)")
    plt.yscale("log")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(filename), dpi=200)
    plt.close()


def plot_idealised_completion_curves(
    mf6_summary_json: Path,
    out_dir: Path,
    warp_summary_json: Optional[Path] = None,
    title: str = "Idealised completion curves derived from total wall time",
    filename: str = "idealised_completion_curves.png",
) -> None:
    """
    Plot elapsed time (y) versus completed cases (x) using only summary totals.
    Each curve is linear: t(k) = total_wall * k / n_cases.

    :param mf6_summary_json: Path to MF6 summary JSON.
    :param out_dir: Output directory for plot PNG.
    :param warp_summary_json: Optional Warp summary JSON to overlay an idealised Warp curve.
    :param title: Plot title.
    :param filename: Output PNG filename.
    """
    mf6_summary_json = Path(mf6_summary_json)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with mf6_summary_json.open("r", encoding="utf-8") as f:
        mf6 = json.load(f)

    warp_wall = None
    warp_n_cases = None
    if warp_summary_json is not None:
        warp_summary_json = Path(warp_summary_json)
        with warp_summary_json.open("r", encoding="utf-8") as f:
            warp = json.load(f)
        warp_entry = _select_warp_entry(warp)
        if isinstance(warp_entry, dict):
            if "total_wall_seconds" in warp_entry:
                warp_wall = float(warp_entry["total_wall_seconds"])
            if "n_cases" in warp_entry:
                warp_n_cases = int(warp_entry["n_cases"])

    workers: list[int] = []
    wall: list[float] = []
    n_cases_list: list[int] = []

    for key in mf6:
        entry = mf6[key]
        workers.append(int(entry["n_workers"]))
        wall.append(float(entry["total_wall_seconds"]))
        n_cases_list.append(int(entry["n_cases"]))

    n_cases_arr = np.asarray(n_cases_list, dtype=int)
    if int(n_cases_arr.min()) != int(n_cases_arr.max()):
        raise ValueError(f"n_cases differs across MF6 entries: {n_cases_arr.tolist()}")
    n_cases = int(n_cases_arr[0])

    workers_arr = np.asarray(workers, dtype=int)
    order = np.argsort(workers_arr)
    workers_arr = workers_arr[order]
    wall_arr = np.asarray(wall, dtype=float)[order]

    x = np.arange(1, n_cases + 1, dtype=int)

    cmap = plt.get_cmap("viridis")
    n_curves = workers_arr.size
    colors = cmap(np.linspace(0.0, 1.0, n_curves))

    plt.figure()
    for i, w in enumerate(workers_arr):
        y = wall_arr[i] * (x.astype(float) / float(n_cases))
        plt.plot(x, y, color=colors[i], label=f"MF6 W={int(w)}")

    if warp_wall is not None:
        if warp_n_cases is None or int(warp_n_cases) != int(n_cases):
            raise ValueError(f"Warp n_cases {warp_n_cases} does not match MF6 n_cases {n_cases}")
        y_warp = float(warp_wall) * (x.astype(float) / float(n_cases))
        plt.plot(x, y_warp, linestyle="--", label="Warp (single GPU)")

    plt.xlabel("Completed cases")
    plt.ylabel("Elapsed wall time (s)")
    # log scale for better visibility of early completions
    plt.yscale("log")

    # plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(filename), dpi=200)
    plt.close()




def main(argv: list[str] | None = None) -> int:
    """
    Main function to parse arguments and generate plots.
    :param argv: Optional list of command-line arguments.
    :return: Exit code (0 for success).
    """
    parser = argparse.ArgumentParser(description="Generate plots from benchmark summary JSON.")
    parser.add_argument("--mf6_summary", type=str, required=True)
    parser.add_argument("--warp_summary", type=str, default="")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--title_prefix", type=str, default="")

    args = parser.parse_args(argv)

    mf6_summary = Path(args.mf6_summary)
    warp_summary = Path(args.warp_summary) if str(args.warp_summary).strip() else None
    out_dir = Path(args.out_dir)

    plot_throughput_vs_workers(
        mf6_summary_json=mf6_summary,
        out_dir=out_dir,
        warp_summary_json=warp_summary,
        title=f"{args.title_prefix}throughput vs MF6 worker processes",
    )

    plot_walltime_vs_workers(
        mf6_summary_json=mf6_summary,
        out_dir=out_dir,
        warp_summary_json=warp_summary,
        title=f"{args.title_prefix}total wall time vs MF6 worker processes",
    )

    plot_idealised_completion_curves(
        mf6_summary_json=mf6_summary,
        out_dir=out_dir,
        warp_summary_json=warp_summary,
        title=f"{args.title_prefix}idealised completion curves from total wall time",
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
# def main() -> None:
#     base = Path(".")
#     out_dir = Path('<repo_root>/paper/tables_figures/')
#
#     mf6_summary = '<repo_root>/data/mf6_ensemble_benchmark_results_1000000.json'
#     warp_summary = '<repo_root>/data/warp_class_ensemble_benchmark_results_1000000.json'
#
#     plot_throughput_vs_workers(
#         mf6_summary_json=mf6_summary,
#         out_dir=out_dir,
#         warp_summary_json=warp_summary,
#         title="1000x1000, N=48: throughput vs MF6 worker processes",
#     )
#
#     plot_walltime_vs_workers(
#         mf6_summary_json=mf6_summary,
#         out_dir=out_dir,
#         warp_summary_json=warp_summary,
#         title="1000x1000, N=48: total wall time vs MF6 worker processes",
#     )
#
#     plot_idealised_completion_curves(
#         mf6_summary_json=mf6_summary,
#         out_dir=out_dir,
#         warp_summary_json=warp_summary,
#         title="1000x1000, N=48: idealised completion curves from total wall time",
#     )
#
#
# if __name__ == "__main__":
#     main()
