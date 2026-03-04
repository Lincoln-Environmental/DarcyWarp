from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd




def _as_float(x: object) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (np.floating, np.integer)):
        return float(x.item())
    if isinstance(x, (int, float)):
        return float(x)
    try:
        return float(x)  # string numeric
    except Exception:
        return float("nan")


def _parse_mf6_last_budget(lst_path: Path) -> dict:
    """
    Best-effort parse of the last MF6 volumetric budget block from model_truth.lst.

    Returns keys:
      rcha_in, rcha_out, chd_in, chd_out, ghb_in, ghb_out,
      total_in, total_out, in_minus_out, percent_discrepancy
    """
    text = lst_path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()

    last: dict = {}
    cur: dict = {}
    state = ""

    pat_term = re.compile(r"^\s*([A-Z0-9_]+)\s*=\s*([-+0-9Ee\.]+)")
    pat_total_in = re.compile(r"^\s*TOTAL\s+IN\s*=\s*([-+0-9Ee\.]+)", re.IGNORECASE)
    pat_total_out = re.compile(r"^\s*TOTAL\s+OUT\s*=\s*([-+0-9Ee\.]+)", re.IGNORECASE)
    pat_in_out = re.compile(r"^\s*IN\s*-\s*OUT\s*=\s*([-+0-9Ee\.]+)", re.IGNORECASE)
    pat_pct = re.compile(r"^\s*PERCENT\s+DISCREPANCY\s*=\s*([-+0-9Ee\.]+)", re.IGNORECASE)

    for raw in lines:
        s = raw.strip()

        if s.startswith("IN:"):
            state = "IN"
            cur = {}
            continue

        if s.startswith("OUT:"):
            state = "OUT"
            continue

        m_total_in = pat_total_in.match(raw)
        if m_total_in is not None:
            cur["total_in"] = _as_float(m_total_in.group(1))
            continue

        m_total_out = pat_total_out.match(raw)
        if m_total_out is not None:
            cur["total_out"] = _as_float(m_total_out.group(1))
            continue

        m_in_out = pat_in_out.match(raw)
        if m_in_out is not None:
            cur["in_minus_out"] = _as_float(m_in_out.group(1))
            continue

        m_pct = pat_pct.match(raw)
        if m_pct is not None:
            cur["percent_discrepancy"] = _as_float(m_pct.group(1))
            # Ensure expected keys exist
            for k in ("rcha_in", "rcha_out", "chd_in", "chd_out", "ghb_in", "ghb_out"):
                if k not in cur:
                    cur[k] = 0.0
            if "total_in" not in cur:
                cur["total_in"] = float("nan")
            if "total_out" not in cur:
                cur["total_out"] = float("nan")
            if "in_minus_out" not in cur:
                cur["in_minus_out"] = float("nan")
            last = dict(cur)
            state = ""
            cur = {}
            continue

        m_term = pat_term.match(raw)
        if m_term is not None and state in ("IN", "OUT"):
            name = m_term.group(1)
            val = _as_float(m_term.group(2))
            if name in ("RCHA", "CHD", "GHB"):
                key = f"{name.lower()}_{'in' if state == 'IN' else 'out'}"
                cur[key] = val

    return last


if __name__ == "__main__":

    # Edit these if needed
    JSON_REL = Path('data/mass_balance_results_ghb_True_t_isotropic_False.json')
    # MF6_BASE_REL = Path("DARCY_WARP_PACKAGE/data")
    WRITE_CSV = True
    OUT_CSV_REL = Path("data/mass_balance_compare_vs_mf6.csv")

    cwd = Path.cwd()
    here = Path(__file__).resolve()

    json_path = cwd.joinpath(JSON_REL)
    if not json_path.exists():
        json_path = here.parent.parent.joinpath(JSON_REL)

    mf6_base = cwd.joinpath('data')
    if not mf6_base.exists():
        mf6_base = here.parent.parent.joinpath(cwd)

    if not json_path.exists():
        raise FileNotFoundError(f"JSON not found: {json_path}")
    if not mf6_base.exists():
        raise FileNotFoundError(f"MF6 base not found: {mf6_base}")

    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    rows: list[dict] = []

    keys = [
        "rcha_in",
        "rcha_out",
        "chd_in",
        "chd_out",
        "ghb_in",
        "ghb_out",
        "total_in",
        "total_out",
        "in_minus_out",
        "percent_discrepancy",
    ]

    for case_key, payload in data.items():
        bud = payload.get("mass_balance_budget_kcycle", None)
        if bud is None:
            continue

        case = str(bud.get("case", case_key))
        lst_path = mf6_base.joinpath(f"Paper_mf6_truth_{case}", "model_truth.lst")

        mf6_bud = {}
        mf6_found = False
        if lst_path.exists():
            mf6_bud = _parse_mf6_last_budget(lst_path)
            mf6_found = len(mf6_bud) > 0

        row = {"case": case, "mf6_found": bool(mf6_found), "mf6_lst": str(lst_path)}

        for k in keys:
            row[f"q_{k}"] = _as_float(bud.get(k, None))
            row[f"mf6_{k}"] = _as_float(mf6_bud.get(k, None))

            qv = row[f"q_{k}"]
            mv = row[f"mf6_{k}"]
            if np.isfinite(qv) and np.isfinite(mv):
                row[f"diff_{k}"] = qv - mv
            else:
                row[f"diff_{k}"] = float("nan")

        rows.append(row)

    if len(rows) == 0:
        print("No mass_balance_budget_kcycle entries found in JSON.")


    df = pd.DataFrame(rows)

    # Sort by nominal cell count if case is like "3000x2999"
    cell_counts = []
    for c in df["case"].astype(str).tolist():
        m = re.match(r"^\s*(\d+)\s*x\s*(\d+)\s*$", c)
        if m is None:
            cell_counts.append(float("nan"))
        else:
            cell_counts.append(float(int(m.group(1)) * int(m.group(2))))
    df["case_cells_nominal"] = cell_counts
    df = df.sort_values(["case_cells_nominal", "case"]).reset_index(drop=True)

    show_cols = [
        "case",
        "mf6_found",
        "q_total_in",
        "mf6_total_in",
        "diff_total_in",
        "q_total_out",
        "mf6_total_out",
        "diff_total_out",
        "q_in_minus_out",
        "mf6_in_minus_out",
        "diff_in_minus_out",
        "q_percent_discrepancy",
        "mf6_percent_discrepancy",
        "diff_percent_discrepancy",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    with pd.option_context("display.max_rows", 200, "display.max_columns", 200, "display.width", 200):
        print(df[show_cols].to_string(index=False))

    if WRITE_CSV:
        out_csv = cwd.joinpath(OUT_CSV_REL)
        if not out_csv.parent.exists():
            out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\nWrote: {out_csv}")





