from __future__ import annotations

import argparse
from pathlib import Path

from DARCY_WARP_PACKAGE.project_base import data_store
from DARCY_WARP_PACKAGE.model_benchmarking_recharge_change import (
    main as bench_recharge_main,
    _default_workers_arg as _default_workers_arg,
)
from DARCY_WARP_PACKAGE.model_benchmarking_T_change import main as bench_t_main
from DARCY_WARP_PACKAGE.benchmark_plots import main as plots_main

p = Path(__file__).resolve()
repo_root = p.parents[1]  # DarcyWarp project root (parent of DARCY_WARP_PACKAGE)

outdir = repo_root.joinpath("paper", "tables_figures")
outdir.mkdir(parents=True, exist_ok=True)


print(str(outdir))


def main(argv: list[str] | None = None) -> int:
    """
    Run the benchmark CLI and then generate plots from its summary JSON outputs.

    :param argv: Optional argv list for programmatic calls. If None, argparse uses sys.argv.
    :return: Process exit code (0 is success).
    """
    parser = argparse.ArgumentParser(description="Run benchmarks and generate plots.")
    parser.add_argument("--nx", type=int, default=1000)
    parser.add_argument("--ny", type=int, default=1000)
    parser.add_argument("--dx", type=float, default=100.0)
    parser.add_argument("--n_cases", type=int, default=48)
    parser.add_argument("--workers", type=str, default=_default_workers_arg())
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--ghb", action="store_true")

    parser.add_argument("--run_warp", action="store_true")
    parser.add_argument("--mg_min_coarse_cells", type=int, default=1000)
    parser.add_argument("--run_mf6", action="store_true")
    parser.add_argument("--run_fd", action="store_true")
    parser.add_argument("--run_recharge", action="store_true")
    parser.add_argument("--run_t", action="store_true")
    parser.add_argument(
        "--write_metadata",
        action="store_true",
        help="Write optional benchmark metadata JSON for each suite.",
    )
    parser.add_argument(
        "--plots_only",
        action="store_true",
        help="Skip running benchmarks and only generate plots from existing summaries.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-run benchmarks even if their summary JSON files already exist.",
    )

    parser.add_argument("--out_dir", type=str, default=str(outdir))

    args = parser.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    cells = int(args.nx * args.ny)

    any_component_flag = bool(args.run_warp or args.run_mf6 or args.run_fd)
    run_flags_common: list[str] = []
    if not any_component_flag:
        run_flags_common = ["--run_warp", "--run_mf6", "--run_fd"]
    else:
        if args.run_warp:
            run_flags_common.append("--run_warp")
        if args.run_mf6:
            run_flags_common.append("--run_mf6")
        if args.run_fd:
            run_flags_common.append("--run_fd")

    if (not args.run_recharge) and (not args.run_t):
        run_recharge = True
        run_t = True
    else:
        run_recharge = bool(args.run_recharge)
        run_t = bool(args.run_t)

    bench_argv_base = [
        "--nx", str(args.nx),
        "--ny", str(args.ny),
        "--dx", str(args.dx),
        "--n_cases", str(args.n_cases),
        "--workers", str(args.workers),
        "--seed", str(args.seed),
        "--device", str(args.device),
    ]
    if args.ghb:
        bench_argv_base.append("--ghb")
    if args.write_metadata:
        bench_argv_base.append("--write_metadata")

    suite_bench_argv_base = list(bench_argv_base)
    suite_bench_argv_base.extend(["--mg_min_coarse_cells", str(args.mg_min_coarse_cells)])

    def _first_existing(candidates: list[Path]) -> Path | None:
        for path in candidates:
            if path.exists():
                return path
        return None

    def _pick_existing(label: str, candidates: list[Path]) -> Path | None:
        existing = _first_existing(candidates)
        if existing is not None:
            return existing
        if candidates:
            print(f"{label} summary JSON not found. Tried: {[str(p) for p in candidates]}")
        return None

    def _recharge_summary_candidates() -> dict[str, list[Path]]:
        return {
            "--run_mf6": [
                Path(data_store).joinpath(f"mf6_ensemble_benchmark_results_recharge{cells}.json"),
                Path(data_store).joinpath(f"mf6_ensemble_benchmark_results_{cells}.json"),
            ],
            "--run_warp": [
                Path(data_store).joinpath(f"warp_class_ensemble_benchmark_results_recharge_{cells}.json"),
                Path(data_store).joinpath(f"warp_class_ensemble_benchmark_results_{cells}.json"),
            ],
            "--run_fd": [
                Path(data_store).joinpath(f"fd_ensemble_benchmark_results_recharge{cells}.json"),
            ],
        }

    def _t_summary_candidates() -> dict[str, list[Path]]:
        return {
            "--run_mf6": [
                Path(data_store).joinpath(f"mf6_T_ensemble_benchmark_results_{cells}.json"),
            ],
            "--run_warp": [
                Path(data_store).joinpath(f"warp_class_T_ensemble_benchmark_results_{cells}.json"),
            ],
            "--run_fd": [
                Path(data_store).joinpath(f"fd_T_ensemble_benchmark_results_{cells}.json"),
            ],
        }

    def _suite_run_flags(summary_candidates: dict[str, list[Path]]) -> list[str]:
        """Return run flags for components whose summary JSON does not yet exist.

        DarcyWarp (--run_warp) always re-runs; only MF6/FD results are reused.
        """
        flags: list[str] = []
        for flag in run_flags_common:
            if flag == "--run_warp":
                flags.append(flag)
                continue
            candidates = summary_candidates.get(flag, [])
            existing = _first_existing(candidates)
            if existing is not None and not args.force:
                print(f"Skipping {flag}: summary already exists: {existing} (use --force to re-run).")
                continue
            flags.append(flag)
        return flags

    def _plot_from_summaries(mf6_summary: Path | None, warp_summary: Path | None, plot_out_dir: Path, title_prefix: str) -> int:
        if mf6_summary is None:
            return 0

        if not mf6_summary.exists():
            print(f"MF6 summary JSON not found, skipping plots: {mf6_summary}")
            return 0

        plot_argv = [
            "--mf6_summary", str(mf6_summary),
            "--out_dir", str(plot_out_dir),
            "--title_prefix", title_prefix,
        ]
        if warp_summary is not None and warp_summary.exists():
            plot_argv.extend(["--warp_summary", str(warp_summary)])

        return int(plots_main(plot_argv))

    if run_recharge:
        summary_candidates = _recharge_summary_candidates()
        if not args.plots_only:
            suite_flags = _suite_run_flags(summary_candidates)
            if suite_flags:
                rc = int(bench_recharge_main(suite_bench_argv_base + suite_flags))
                if rc != 0:
                    return rc
            else:
                print("All requested recharge benchmark summaries already exist; skipping benchmark run.")

        mf6_summary = _pick_existing("MF6 recharge", summary_candidates["--run_mf6"])
        warp_summary = _pick_existing("Warp recharge", summary_candidates["--run_warp"])
        plot_out_dir = out_dir.joinpath("recharge_change")
        plot_out_dir.mkdir(exist_ok=True)
        rc = _plot_from_summaries(
            mf6_summary=mf6_summary,
            warp_summary=warp_summary,
            plot_out_dir=plot_out_dir,
            title_prefix=f"Recharge change: {args.nx}x{args.ny}, N={args.n_cases}: ",
        )
        if rc != 0:
            return rc

    if run_t:
        summary_candidates = _t_summary_candidates()
        if not args.plots_only:
            suite_flags = _suite_run_flags(summary_candidates)
            if suite_flags:
                rc = int(bench_t_main(suite_bench_argv_base + suite_flags))
                if rc != 0:
                    return rc
            else:
                print("All requested T benchmark summaries already exist; skipping benchmark run.")

        mf6_summary = _pick_existing("MF6 T", summary_candidates["--run_mf6"])
        warp_summary = _pick_existing("Warp T", summary_candidates["--run_warp"])
        plot_out_dir = out_dir.joinpath("t_change")
        plot_out_dir.mkdir(exist_ok=True)
        rc = _plot_from_summaries(
            mf6_summary=mf6_summary,
            warp_summary=warp_summary,
            plot_out_dir=plot_out_dir,
            title_prefix=f"T change: {args.nx}x{args.ny}, N={args.n_cases}: ",
        )
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
