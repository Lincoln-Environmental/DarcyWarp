"""Generate CSV and LaTeX ensemble timing tables from saved second-machine runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "second_machine_test" / "DARCY_WARP_PACKAGE" / "data"
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


@dataclass(frozen=True)
class BenchmarkSpec:
    """Describe the inputs for one ensemble experiment.

    :param key: Suite identifier used in CLI selection and output filenames.
    :param label: Human-readable experiment name used in table captions.
    :param mf6_candidates: MODFLOW summary filenames in order of preference.
    :param warp_candidates: DarcyWarp summary filenames in order of preference.
    """

    key: str
    label: str
    mf6_candidates: tuple[str, ...]
    warp_candidates: tuple[str, ...]


@dataclass(frozen=True)
class EnsembleRow:
    """Store the measured timings for one solver configuration.

    :param solver: Solver display name.
    :param variant: Configuration identifier from the summary JSON.
    :param n_workers: Parallel worker count, or ``None`` when unknown.
    :param n_cases: Number of ensemble cases in the run.
    :param total_wall_seconds: Elapsed wall-clock time for all cases, in seconds.
    :param throughput_cases_per_second: Cases completed per second of wall time.
    :param mean_case_seconds: Reported mean time per case in seconds, or total
        wall time divided by case count when the mean was unavailable.
    :param source_file: Filename of the summary containing this configuration.
    """

    solver: str
    variant: str
    n_workers: int | None
    n_cases: int
    total_wall_seconds: float
    throughput_cases_per_second: float
    mean_case_seconds: float
    source_file: str


def _parse_float(value: Any) -> float:
    """Read a timing field that can be missing in older benchmark records.

    :param value: Raw value from the JSON record.
    :returns: Floating-point value, or NaN when conversion fails.
    """
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _parse_int(value: Any, default: int = 0) -> int:
    """Read an integer count from a benchmark record.

    :param value: Raw worker or case count.
    :param default: Count to use when the value cannot be converted to an integer.
    :returns: Converted count or the supplied default.
    """
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _fmt(value: float) -> str:
    """Format a measurement for the printed tables.

    :param value: Timing, throughput or dimensionless throughput ratio.
    :returns: Three significant digits for a finite value, otherwise a dash.
    """
    return f"{value:.3g}" if math.isfinite(value) else "-"


def _escape_latex(text: str) -> str:
    """Escape characters used in solver and variant names for a LaTeX table.

    :param text: Solver or variant name from the benchmark.
    :returns: Text with ampersands and underscores escaped.
    """
    return text.replace("&", "\\&").replace("_", "\\_")


def _split_candidates(raw: str) -> tuple[str, ...]:
    """Read an ordered list of candidate summary files from the CLI.

    :param raw: Comma-separated filenames, highest preference first.
    :returns: Nonempty filenames in their original order, with whitespace stripped.
    :raises ValueError: If the list contains no filenames.
    """
    tokens = [token.strip() for token in raw.split(",") if token.strip()]
    if not tokens:
        raise ValueError("At least one candidate filename is required.")
    return tuple(tokens)


def _first_existing_or_none(input_dir: Path, candidates: tuple[str, ...]) -> Path | None:
    """Locate the first available summary for a solver and suite.

    :param input_dir: Directory containing the benchmark summaries.
    :param candidates: Filenames in order of preference.
    :returns: Path to the first matching file, or ``None`` if none exists.
    """
    for name in candidates:
        path = input_dir / name
        if path.is_file():
            return path
    return None


def _load_rows(summary_path: Path, *, solver: str, default_workers: int | None) -> list[EnsembleRow]:
    """Read one ensemble timing record for each configuration in a summary.

    :param summary_path: UTF-8 JSON file mapping variant names to timing records.
    :param solver: Solver name to attach to each row.
    :param default_workers: Worker count to use when a record omits it.
    :returns: Timing records in the order used by the summary.
    :raises ValueError: If the file contains invalid JSON, does not contain a
        mapping, or contains no configuration records.
    """
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
    """Return the solver ordering used in both output formats.

    :param name: Solver display name.
    :returns: Sort rank placing Warp first, MODFLOW 6 second and other names last.
    """
    return {"Warp": 0, "MODFLOW 6": 1}.get(name, 99)


def _sort_rows(rows: list[EnsembleRow]) -> list[EnsembleRow]:
    """Order the complete results by solver, worker count and variant.

    :param rows: Timing records from the selected summaries.
    :returns: A new list in table order, with unknown worker counts sorted as -1.
    """
    def table_order(row: EnsembleRow) -> tuple[int, int, str]:
        """Construct the display order for an individual configuration.

        :param row: Timing record to order.
        :returns: Solver rank, worker count sort key and variant name.
        """
        workers = -1 if row.n_workers is None else row.n_workers
        return _solver_rank(row.solver), workers, row.variant

    return sorted(rows, key=table_order)


def _write_all_csv(path: Path, rows: list[EnsembleRow]) -> None:
    """Write the timing and source fields for every benchmark configuration.

    :param path: Destination CSV file. Parent directories are created as needed.
    :param rows: Configuration records in the desired output order.
    """
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
    """Write a LaTeX table of throughput and wall time across worker counts.

    :param path: Destination LaTeX file.
    :param rows: Configuration records in the desired output order.
    :param table_prefix: Prefix for the LaTeX cross-reference label.
    :param benchmark_label: Description of the ensemble experiment for the caption.
    """
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
    """Choose each solver's highest-throughput variant, breaking ties by wall time.

    :param rows: Configuration records from the selected experiment.
    :returns: One record per solver in table order. A complete tie retains the
        first matching record.
    """
    def score(item: EnsembleRow) -> tuple[float, float]:
        """Rank a solver configuration by throughput and then by shortest duration.

        :param item: Timing record to score.
        :returns: Maximisation keys, using negative infinity for missing metrics.
        """
        throughput = item.throughput_cases_per_second
        wall = item.total_wall_seconds
        throughput_score = throughput if math.isfinite(throughput) else -math.inf
        wall_score = -wall if math.isfinite(wall) else -math.inf
        return throughput_score, wall_score

    grouped: dict[str, list[EnsembleRow]] = {}
    for row in rows:
        grouped.setdefault(row.solver, []).append(row)

    return [max(grouped[solver], key=score) for solver in sorted(grouped, key=_solver_rank)]


def _throughput_ratio(throughput: float, mf6_best_throughput: float) -> float:
    """Normalise the measured throughput to the best MODFLOW 6 run.

    :param throughput: Selected solver configuration's throughput in cases per second.
    :param mf6_best_throughput: Reference throughput in the same units.
    :returns: Dimensionless ratio, or NaN for a missing measurement or a
        nonpositive reference throughput.
    """
    if math.isfinite(throughput) and mf6_best_throughput > 0.0:
        return throughput / mf6_best_throughput
    return math.nan


def _write_best_csv(path: Path, rows: list[EnsembleRow], *, mf6_best_throughput: float) -> None:
    """Write the best configuration per solver and its throughput relative to MF6.

    :param path: Destination CSV file.
    :param rows: Best configuration records in the desired output order.
    :param mf6_best_throughput: Best MODFLOW 6 throughput in cases per second.
    """
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
            ratio = _throughput_ratio(row.throughput_cases_per_second, mf6_best_throughput)
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
    """Write the LaTeX comparison of the best-performing solver configurations.

    :param path: Destination LaTeX file.
    :param rows: Best configuration records in the desired output order.
    :param table_prefix: Prefix for the LaTeX cross-reference label.
    :param benchmark_label: Description of the ensemble experiment for the caption.
    :param mf6_best_throughput: Best MODFLOW 6 throughput in cases per second.
    """
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
            ratio = _throughput_ratio(row.throughput_cases_per_second, mf6_best_throughput)
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
    """Generate worker-scaling and best-solver tables for the selected ensembles.

    :param argv: Command-line arguments excluding the program name. Passing
        ``None`` reads the process command line.
    :returns: Zero after writing the CSV and LaTeX tables.
    :raises FileNotFoundError: If the input directory is missing, or a required
        summary is missing when ``--strict`` is set.
    :raises ValueError: If a suite name, candidate list or summary is invalid.
    :raises RuntimeError: If no suites can be generated or the best MF6
        configuration lacks a finite throughput.
    """
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
