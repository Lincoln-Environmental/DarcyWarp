from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


PACKAGE_DIR = Path(__file__).resolve().parent
ROOT_DIR = PACKAGE_DIR.parent
DATA_DIR = PACKAGE_DIR / "data"
OUT_DIR = ROOT_DIR / "paper" / "tables_figures"

DEFAULT_JSON = DATA_DIR / "mass_balance_results_ghb_True_t_isotropic_False.json"
DEFAULT_COMPARE_CSV = DATA_DIR / "mass_balance_compare_vs_mf6.csv"
DEFAULT_FLOAT_FMT = ".3g"

CASE_RE = re.compile(r"^\s*(\d+)\s*x\s*(\d+)\s*$")


def _case_cells(case: str) -> float:
    match = CASE_RE.match(case.strip())
    if match is None:
        return float("nan")
    nx = int(match.group(1))
    ny = int(match.group(2))
    return float(nx * ny)


def _sort_cases(df: pd.DataFrame) -> pd.DataFrame:
    case_cells = df["case"].astype(str).map(_case_cells)
    df = df.assign(case_cells_nominal=case_cells)
    df = df.sort_values(["case_cells_nominal", "case"]).reset_index(drop=True)
    return df.drop(columns=["case_cells_nominal"])


def _latex_escape(text: str) -> str:
    return text.replace("&", "\\&").replace("_", "\\_")


def _format_value(value: object, fmt: str | None) -> str:
    if fmt is None:
        return _latex_escape(str(value))
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(val):
        return "-"
    return format(val, fmt)


def _parse_conditions(path: Path) -> tuple[str | None, str | None]:
    name = path.name
    ghb = None
    t_iso = None

    ghb_match = re.search(r"ghb_(True|False)", name)
    if ghb_match is not None:
        ghb = ghb_match.group(1)

    t_match = re.search(r"t_isotropic_(True|False)", name)
    if t_match is not None:
        t_iso = t_match.group(1)
    elif "heterogeneous" in name:
        t_iso = "False"
    elif "anisotropic" in name:
        t_iso = "False"
    elif "isotropic" in name:
        t_iso = "True"

    return ghb, t_iso


def _format_case_caption(path: Path) -> str:
    ghb, t_iso = _parse_conditions(path)
    parts = []
    if ghb is not None:
        parts.append(f"GHB: {ghb}")
    if t_iso is not None:
        t_label = "Isotropic" if t_iso == "True" else "Horizontally heterogeneous"
        parts.append(f"Transmissivity: {t_label}")
    return ", ".join(parts) if parts else "GHB/Transmissivity not specified"


def _derive_prefix(path: Path) -> str:
    ghb, t_iso = _parse_conditions(path)
    ghb_label = ghb if ghb is not None else "unknown"
    if t_iso == "True":
        t_label = "isotropic"
    elif t_iso == "False":
        t_label = "heterogeneous"
    else:
        t_label = "unknown"
    return f"quickflow_ghb_{ghb_label}_{t_label}"


def _load_mass_balance_json(json_path: Path) -> pd.DataFrame:
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object at the top level (dict keyed by case name).")

    rows: list[dict] = []
    for case_key, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        bud = payload.get("mass_balance_budget_kcycle", None)
        if not isinstance(bud, dict):
            continue
        row = {
            "case": str(bud.get("case", case_key)),
            "total_in": bud.get("total_in", np.nan),
            "total_out": bud.get("total_out", np.nan),
            "in_minus_out": bud.get("in_minus_out", np.nan),
            "percent_discrepancy": bud.get("percent_discrepancy", np.nan),
            "imbalance_fraction": bud.get("imbalance_fraction", np.nan),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No mass_balance_budget_kcycle entries found in JSON.")
    return _sort_cases(df)


def _load_compare_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Comparison CSV is empty.")
    if "mf6_found" in df.columns:
        df = df[df["mf6_found"]].copy()
    return _sort_cases(df)


def _build_closure_columns(float_fmt: str) -> list[tuple[str, str, str | None]]:
    return [
        ("case", "Case", None),
        ("total_in", "$Q_{\\mathrm{IN}}$", float_fmt),
        ("total_out", "$Q_{\\mathrm{OUT}}$", float_fmt),
        ("in_minus_out", "$\\Delta Q$", float_fmt),
        ("percent_discrepancy", "$\\epsilon$ (\\%)", float_fmt),
        ("imbalance_fraction", "$\\Delta Q / Q_{\\mathrm{tf}}$", float_fmt),
    ]


def _build_compare_columns(float_fmt: str) -> list[tuple[str, str, str | None]]:
    return [
        ("case", "Case", None),
        ("q_in_minus_out", "\\shortstack{DarcyWarp\\\\$\\Delta Q$}", float_fmt),
        ("mf6_in_minus_out", "\\shortstack{MF6\\\\$\\Delta Q$}", float_fmt),
        ("diff_in_minus_out", "\\shortstack{$\\Delta Q$\\\\Diff}", float_fmt),
        ("q_percent_discrepancy", "\\shortstack{DarcyWarp\\\\$\\epsilon$ (\\%)}", float_fmt),
        ("mf6_percent_discrepancy", "\\shortstack{MF6\\\\$\\epsilon$ (\\%)}", float_fmt),
        ("diff_percent_discrepancy", "\\shortstack{$\\epsilon$\\\\Diff}", float_fmt),
    ]


def _write_table(
    out_path: Path,
    df: pd.DataFrame,
    columns: list[tuple[str, str, str | None]],
    caption: str,
    label: str,
    resizebox: bool = True,
) -> None:
    col_spec = "l" + ("r" * (len(columns) - 1))

    with out_path.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write("\\scriptsize\n")
        f.write("\\setlength{\\tabcolsep}{3pt}\n")
        f.write("\\renewcommand{\\arraystretch}{1.05}\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")

        if resizebox:
            f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(f"\\begin{{tabular}}{{{col_spec}}}\n")
        f.write("\\hline\n")
        header = " & ".join([col[1] for col in columns]) + " \\\\\n"
        f.write(header)
        f.write("\\hline\n")

        for _, row in df.iterrows():
            cells = [_format_value(row.get(key, ""), fmt) for key, _, fmt in columns]
            f.write(" & ".join(cells) + " \\\\\n")

        f.write("\\hline\n")
        if resizebox:
            f.write("\\end{tabular}%\n")
            f.write("}\n")
        else:
            f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


def write_mass_balance_tables(
    json_path: Path,
    compare_csv_path: Path,
    out_dir: Path,
    float_fmt: str,
    prefix: str | None = None,
) -> tuple[Path, Path]:
    json_path = json_path.expanduser().resolve()
    compare_csv_path = compare_csv_path.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()

    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    if not compare_csv_path.exists():
        raise FileNotFoundError(f"Comparison CSV not found: {compare_csv_path}")

    out_dir.mkdir(parents=True, exist_ok=True)

    closure_df = _load_mass_balance_json(json_path)
    compare_df = _load_compare_csv(compare_csv_path)

    case_caption = _format_case_caption(json_path)
    prefix = prefix or _derive_prefix(json_path)

    closure_caption = (
        "Mass-balance closure metrics for DarcyWarp "
        f"({case_caption}). "
        "Imbalance fraction uses $Q_{\\mathrm{tf}} = 0.5\\,(Q_{\\mathrm{IN}} + Q_{\\mathrm{OUT}})$."
    )
    closure_label = f"tab:{prefix}_mass_balance_closure"
    closure_tex = out_dir / f"{prefix}_mass_balance_closure_table.tex"

    compare_caption = (
        "DarcyWarp vs MODFLOW 6 mass-balance closure comparison "
        f"({case_caption})."
    )
    compare_label = f"tab:{prefix}_mass_balance_compare_mf6"
    compare_tex = out_dir / f"{prefix}_mass_balance_compare_mf6_table.tex"

    _write_table(
        closure_tex,
        closure_df,
        _build_closure_columns(float_fmt),
        closure_caption,
        closure_label,
        resizebox=True,
    )
    _write_table(
        compare_tex,
        compare_df,
        _build_compare_columns(float_fmt),
        compare_caption,
        compare_label,
        resizebox=True,
    )

    return closure_tex, compare_tex


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables for mass-balance closure and MF6 comparison."
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=DEFAULT_JSON,
        help="Path to mass_balance_results_*.json",
    )
    parser.add_argument(
        "--compare-csv",
        type=Path,
        default=DEFAULT_COMPARE_CSV,
        help="Path to mass_balance_compare_vs_mf6.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for LaTeX tables.",
    )
    parser.add_argument(
        "--float-fmt",
        default=DEFAULT_FLOAT_FMT,
        help="Float format string for table values (e.g., .3g).",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        help="Optional output filename prefix override.",
    )

    args = parser.parse_args()
    closure_path, compare_path = write_mass_balance_tables(
        json_path=args.json,
        compare_csv_path=args.compare_csv,
        out_dir=args.out_dir,
        float_fmt=args.float_fmt,
        prefix=args.prefix,
    )
    print(f"Wrote: {closure_path}")
    print(f"Wrote: {compare_path}")
