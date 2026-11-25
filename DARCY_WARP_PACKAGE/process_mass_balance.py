from __future__ import annotations

import json
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd




def _as_float(x: object) -> float:
    """Convert a reported budget value to float.

    :param x: Numeric field or missing value.
    :returns: Numeric value, or NaN for missing/unparseable values.
    """
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

    :param lst_path: Native MODFLOW listing file.
    :returns: Last complete budget block, or an empty dictionary if none is found.

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


def compare_mass_balance(json_path: Path, mf6_base: Path) -> pd.DataFrame:
    """Compare reported DarcyWarp budgets with their matching native MF6 listings.

    :param json_path: Archived budget JSON or current steady benchmark JSON.
    :param mf6_base: Directory for archived listing workspaces, used only for
        records predating explicit listing provenance.
    :returns: One row per budget with closure and package rates and differences.
    :raises FileNotFoundError: If the budget JSON is missing.
    :raises ValueError: If no budget records exist.
    """
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

        case = str(bud.get("case") or case_key)
        if "mf6_listing_path" in payload:
            listing = payload["mf6_listing_path"]
            lst_path = Path(listing) if listing else None
        else:
            lst_path = mf6_base / f"Paper_mf6_truth_{case}" / "model_truth.lst"

        mf6_bud = {}
        mf6_found = False
        if lst_path is not None and lst_path.is_file():
            matches_reference = True
            if "mf6_listing_path" in payload:
                with lst_path.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                matches_reference = digest == payload.get("mf6_listing_sha256")
            if matches_reference:
                mf6_bud = _parse_mf6_last_budget(lst_path)
                mf6_found = len(mf6_bud) > 0

        row = {"case": case, "mf6_found": bool(mf6_found), "mf6_lst": str(lst_path) if lst_path is not None else ""}

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

    if not rows:
        raise ValueError("No mass_balance_budget_kcycle entries found in JSON.")

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

    return df


if __name__ == "__main__":
    from DARCY_WARP_PACKAGE.project_base import data_store

    # Set the input and output explicitly; the archived CSV is not overwritten by default.
    json_path = data_store / "mass_balance_results_ghb_True_t_isotropic_False.json"
    mf6_base = data_store
    out_csv = data_store / "mass_balance_compare_recomputed.csv"
    df = compare_mass_balance(json_path, mf6_base)

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

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"\nWrote: {out_csv}")
