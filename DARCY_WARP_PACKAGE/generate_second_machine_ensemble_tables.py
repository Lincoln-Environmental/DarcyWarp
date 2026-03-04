from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _resolve_repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if parent.joinpath("paper").is_dir() and parent.joinpath("data").is_dir():
            return parent
    raise RuntimeError(f"Could not infer repository root from {here}")


REPO_ROOT = _resolve_repo_root()
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "second_machine_test" / 'DARCY_WARP_PACKAGE' / "data"
DEFAULT_OUT_DIR = REPO_ROOT / "paper" / "tables_figures" / "second_machine"

DEFAULT_T_MF6_CANDIDATES = (
    "mf6_T_ensemble_benchmark_results_1000000.json",
    "mf6_ensemble_benchmark_results_1000000.json",
)
DEFAULT_T_WARP_CANDIDATES = (
    "warp_class_T_ensemble_benchmark_results_1000000.json",
    "warp_class_ensemble_benchmark_results_1000000.json",
)
DEFAULT_RECHARGE_MF6_CANDIDATES = (
    "mf6_ensemble_benchmark_results_recharge1000000.json",
    "mf6_ensemble_benchmark_results_1000000.json",
)
DEFAULT_RECHARGE_WARP_CANDIDATES = (
    "warp_class_ensemble_benchmark_results_recharge_1000000.json",
    "warp_class_ensemble_benchmark_results_1000000.json",
)
# DEFAULT_FD_CANDIDATES = ("fd_ensemble_benchmark_results_1000000.json",)


@dataclass(frozen=True)
class BenchmarkSpec:
    key: str
    label: str
    mf6_candidates: tuple[str, ...]
    warp_candidates: tuple[str, ...]


@dataclass(frozen=True)
class EnsembleRow:
    solver: str
    variant: str
    n_workers: int | None
    n_cases: int
    total_wall_seconds: float
    throughput_cases_per_second: float
    mean_case_seconds: float
    source_file: str


def _parse_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _fmt(value: float) -> str:
    return f"{value:.3g}" if math.isfinite(value) else "-"


def _escape_latex(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_")


def _split_candidates(raw: str) -> tuple[str, ...]:
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one candidate filename is required.")
    return tuple(tokens)


def _first_existing_or_none(input_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    for name in candidates:
        path = input_dir / name
        if path.is_file():
            return path
    return None


def _load_rows(summary_path: Path, *, solver: str, default_workers: int | None) -> list[EnsembleRow]:
    raw = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Expected object at top level in {summary_path}")

    rows: list[EnsembleRow] = []
    for key, payload in raw.items():
        if not isinstance(payload, dict):
            continue

        n_cases = _parse_int(payload.get("n_cases"), default=0)
        n_workers_raw = payload.get("n_workers")
        n_workers = _parse_int(n_workers_raw, default=0) if n_workers_raw is not None else default_workers
        total_wall = _parse_float(payload.get("total_wall_seconds"))
        throughput = _parse_float(payload.get("throughput_cases_per_second"))
        mean_case = _parse_float(payload.get("mean_case_seconds"))
        if (not math.isfinite(mean_case)) and math.isfinite(total_wall) and n_cases > 0:
            mean_case = total_wall / float(n_cases)

        rows.append(
            EnsembleRow(
                solver=solver,
                variant=str(key),
                n_workers=n_workers,
                n_cases=n_cases,
                total_wall_seconds=total_wall,
                throughput_cases_per_second=throughput,
                mean_case_seconds=mean_case,
                source_file=summary_path.name,
            )
        )

    if not rows:
        raise ValueError(f"No valid benchmark entries found in {summary_path}")
    return rows


def _solver_rank(name: str) -> int:
    order = {
        "Warp": 0,
        "MODFLOW 6": 1,
        # "SciPy FD": 2,
    }
    return order.get(name, 99)


def _sort_rows(rows: list[EnsembleRow]) -> list[EnsembleRow]:
    return sorted(
        rows,
        key=lambda row: (
            _solver_rank(row.solver),
            row.n_workers if row.n_workers is not None else -1,
            row.variant,
        ),
    )


def _write_all_csv(path: Path, rows: list[EnsembleRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "solver",
                "variant",
                "n_workers",
                "n_cases",
                "total_wall_seconds",
                "throughput_cases_per_second",
                "mean_case_seconds",
                "source_file",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.solver,
                    row.variant,
                    row.n_workers if row.n_workers is not None else "",
                    row.n_cases,
                    row.total_wall_seconds,
                    row.throughput_cases_per_second,
                    row.mean_case_seconds,
                    row.source_file,
                ]
            )


def _write_worker_table_tex(
    path: Path,
    rows: list[EnsembleRow],
    *,
    table_prefix: str,
    benchmark_label: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = f"tab:{table_prefix}_worker_scaling"
    caption = (
        f"Ensemble {benchmark_label} benchmark results on the second machine configuration, "
        "reported from summary JSON outputs."
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated from ensemble benchmark summary JSON files\n")
        handle.write("\\begin{table}[htbp]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write("\\setlength{\\tabcolsep}{3pt}\n")
        handle.write("\\renewcommand{\\arraystretch}{1.05}\n")
        handle.write(f"\\caption{{{caption}}}\n")
        handle.write(f"\\label{{{label}}}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{llrrrrr}\n")
        handle.write("\\hline\n")
        handle.write(
            "Solver & Variant & Workers & Cases & Total wall (s) & Throughput (cases s$^{-1}$) & Mean case (s) \\\\\n"
        )
        handle.write("\\hline\n")
        for row in rows:
            workers = str(row.n_workers) if row.n_workers is not None else "-"
            handle.write(
                f"{_escape_latex(row.solver)} & "
                f"{_escape_latex(row.variant)} & "
                f"{workers} & "
                f"{row.n_cases} & "
                f"{_fmt(row.total_wall_seconds)} & "
                f"{_fmt(row.throughput_cases_per_second)} & "
                f"{_fmt(row.mean_case_seconds)} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}%\n")
        handle.write("}\n")
        handle.write("\\end{table}\n")


def _best_rows_by_solver(rows: list[EnsembleRow]) -> list[EnsembleRow]:
    grouped: dict[str, list[EnsembleRow]] = {}
    for row in rows:
        grouped.setdefault(row.solver, []).append(row)

    best: list[EnsembleRow] = []
    for solver, items in grouped.items():
        def score(item: EnsembleRow) -> tuple[float, float]:
            throughput = item.throughput_cases_per_second
            wall = item.total_wall_seconds
            throughput_score = throughput if math.isfinite(throughput) else float("-inf")
            wall_score = -wall if math.isfinite(wall) else float("-inf")
            return throughput_score, wall_score

        best.append(max(items, key=score))

    return sorted(best, key=lambda item: _solver_rank(item.solver))


def _write_best_csv(path: Path, rows: list[EnsembleRow], *, mf6_best_throughput: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "solver",
                "best_variant",
                "n_workers",
                "n_cases",
                "total_wall_seconds",
                "throughput_cases_per_second",
                "throughput_ratio_vs_best_mf6",
                "source_file",
            ]
        )
        for row in rows:
            ratio = (
                row.throughput_cases_per_second / mf6_best_throughput
                if math.isfinite(row.throughput_cases_per_second) and mf6_best_throughput > 0.0
                else math.nan
            )
            writer.writerow(
                [
                    row.solver,
                    row.variant,
                    row.n_workers if row.n_workers is not None else "",
                    row.n_cases,
                    row.total_wall_seconds,
                    row.throughput_cases_per_second,
                    ratio,
                    row.source_file,
                ]
            )


def _write_best_table_tex(
    path: Path,
    rows: list[EnsembleRow],
    *,
    table_prefix: str,
    benchmark_label: str,
    mf6_best_throughput: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    label = f"tab:{table_prefix}_best_solver"
    caption = (
        f"Best-performing ensemble {benchmark_label} configuration per solver on the second machine, "
        "ranked by throughput."
    )
    with path.open("w", encoding="utf-8") as handle:
        handle.write("% Auto-generated from ensemble benchmark summary JSON files\n")
        handle.write("\\begin{table}[htbp]\n")
        handle.write("\\centering\n")
        handle.write("\\small\n")
        handle.write("\\setlength{\\tabcolsep}{3pt}\n")
        handle.write("\\renewcommand{\\arraystretch}{1.05}\n")
        handle.write(f"\\caption{{{caption}}}\n")
        handle.write(f"\\label{{{label}}}\n")
        handle.write("\\resizebox{\\textwidth}{!}{%\n")
        handle.write("\\begin{tabular}{llrrrrr}\n")
        handle.write("\\hline\n")
        handle.write(
            "Solver & Best variant & Workers & Cases & Total wall (s) & Throughput (cases s$^{-1}$) & Ratio vs best MF6 \\\\\n"
        )
        handle.write("\\hline\n")
        for row in rows:
            workers = str(row.n_workers) if row.n_workers is not None else "-"
            ratio = (
                row.throughput_cases_per_second / mf6_best_throughput
                if math.isfinite(row.throughput_cases_per_second) and mf6_best_throughput > 0.0
                else math.nan
            )
            handle.write(
                f"{_escape_latex(row.solver)} & "
                f"{_escape_latex(row.variant)} & "
                f"{workers} & "
                f"{row.n_cases} & "
                f"{_fmt(row.total_wall_seconds)} & "
                f"{_fmt(row.throughput_cases_per_second)} & "
                f"{_fmt(ratio)} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}%\n")
        handle.write("}\n")
        handle.write("\\end{table}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate supplementary ensemble benchmark tables from second-machine summary JSON files."
    )
    parser.add_argument(
        "--input_dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input directory containing ensemble summary JSON files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--out_dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory for CSV and LaTeX tables (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="quickflow_ensemble_machine2",
        help="Prefix for generated output filenames and LaTeX labels.",
    )
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Comma-separated benchmark suites to generate: all, t_change, recharge_change.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if required summary JSON files for a selected benchmark are missing.",
    )
    parser.add_argument(
        "--t_mf6_candidates",
        type=str,
        default=",".join(DEFAULT_T_MF6_CANDIDATES),
        help="Comma-separated candidate MF6 summary JSON filenames for the transmissivity-change benchmark.",
    )
    parser.add_argument(
        "--t_warp_candidates",
        type=str,
        default=",".join(DEFAULT_T_WARP_CANDIDATES),
        help="Comma-separated candidate Warp summary JSON filenames for the transmissivity-change benchmark.",
    )
    parser.add_argument(
        "--recharge_mf6_candidates",
        type=str,
        default=",".join(DEFAULT_RECHARGE_MF6_CANDIDATES),
        help="Comma-separated candidate MF6 summary JSON filenames for the recharge-change benchmark.",
    )
    parser.add_argument(
        "--recharge_warp_candidates",
        type=str,
        default=",".join(DEFAULT_RECHARGE_WARP_CANDIDATES),
        help="Comma-separated candidate Warp summary JSON filenames for the recharge-change benchmark.",
    )
    # parser.add_argument(
    #     "--fd_candidates",
    #     type=str,
    #     default=",".join(DEFAULT_FD_CANDIDATES),
    #     help="Comma-separated candidate FD summary JSON filenames.",
    # )
    args = parser.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    benchmark_specs = (
        BenchmarkSpec(
            key="t_change",
            label="transmissivity change",
            mf6_candidates=_split_candidates(args.t_mf6_candidates),
            warp_candidates=_split_candidates(args.t_warp_candidates),
        ),
        BenchmarkSpec(
            key="recharge_change",
            label="recharge change",
            mf6_candidates=_split_candidates(args.recharge_mf6_candidates),
            warp_candidates=_split_candidates(args.recharge_warp_candidates),
        ),
    )
    spec_by_key = {spec.key: spec for spec in benchmark_specs}

    benchmark_tokens = [token.strip().lower() for token in str(args.benchmarks).split(",") if token.strip()]
    if not benchmark_tokens:
        raise ValueError("No benchmarks specified.")
    if len(benchmark_tokens) == 1 and benchmark_tokens[0] == "all":
        selected_specs = list(benchmark_specs)
    else:
        unknown = sorted({token for token in benchmark_tokens if token not in spec_by_key})
        if unknown:
            raise ValueError(
                f"Unknown benchmark(s): {unknown}. Valid options: {sorted(spec_by_key)}"
            )
        selected_specs = [spec_by_key[token] for token in benchmark_tokens]

    generated = 0
    for spec in selected_specs:
        mf6_path = _first_existing_or_none(input_dir, spec.mf6_candidates)
        if mf6_path is None:
            msg = (
                f"MF6 summary JSON missing for benchmark '{spec.key}' in {input_dir} "
                f"(tried: {list(spec.mf6_candidates)})"
            )
            if args.strict:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}")
            continue
        warp_path = _first_existing_or_none(input_dir, spec.warp_candidates)
        if warp_path is None:
            msg = (
                f"Warp summary JSON missing for benchmark '{spec.key}' in {input_dir} "
                f"(tried: {list(spec.warp_candidates)})"
            )
            if args.strict:
                raise FileNotFoundError(msg)
            print(f"[skip] {msg}")
            continue

        rows: list[EnsembleRow] = []
        rows.extend(_load_rows(mf6_path, solver="MODFLOW 6", default_workers=None))
        rows.extend(_load_rows(warp_path, solver="Warp", default_workers=1))
        # fd_path = _first_existing_or_none(input_dir, _split_candidates(args.fd_candidates))
        # if fd_path is not None:
        #     rows.extend(_load_rows(fd_path, solver="SciPy FD", default_workers=None))
        rows = _sort_rows(rows)

        run_prefix = f"{args.prefix}_{spec.key}"
        all_csv_path = out_dir / f"{run_prefix}_all_fields.csv"
        worker_tex_path = out_dir / f"{run_prefix}_worker_scaling_table.tex"
        best_csv_path = out_dir / f"{run_prefix}_best_solver_table.csv"
        best_tex_path = out_dir / f"{run_prefix}_best_solver_table.tex"

        _write_all_csv(all_csv_path, rows)
        _write_worker_table_tex(
            worker_tex_path,
            rows,
            table_prefix=run_prefix,
            benchmark_label=spec.label,
        )

        best_rows = _best_rows_by_solver(rows)
        mf6_best = next((row for row in best_rows if row.solver == "MODFLOW 6"), None)
        if mf6_best is None or not math.isfinite(mf6_best.throughput_cases_per_second):
            raise RuntimeError(
                f"Could not determine best MF6 throughput for normalization in benchmark '{spec.key}'."
            )
        mf6_best_throughput = mf6_best.throughput_cases_per_second

        _write_best_csv(best_csv_path, best_rows, mf6_best_throughput=mf6_best_throughput)
        _write_best_table_tex(
            best_tex_path,
            best_rows,
            table_prefix=run_prefix,
            benchmark_label=spec.label,
            mf6_best_throughput=mf6_best_throughput,
        )

        print(f"[ok] benchmark: {spec.key}")
        print(f"[ok] MF6 summary:  {mf6_path}")
        print(f"[ok] Warp summary: {warp_path}")
        # if fd_path is not None:
        #     print(f"[ok] FD summary:   {fd_path}")
        print(f"[out] {all_csv_path}")
        print(f"[out] {worker_tex_path}")
        print(f"[out] {best_csv_path}")
        print(f"[out] {best_tex_path}")
        generated += 1

    if generated == 0:
        raise RuntimeError("No ensemble benchmark tables were generated.")
    print(f"Generated {generated} ensemble benchmark table suite(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
