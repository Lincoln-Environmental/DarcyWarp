from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
INPUT_GLOB = "comparison_results_*.json"
OUT_DIR = BASE_DIR / "paper" / "tables_figures"

PLOT_LOGLOG = True


def _transmissivity_token(name: str) -> str:
    if "isotropic" in name:
        return "isotropic"
    if "heterogeneous" in name:
        return "heterogeneous"
    if "anisotropic" in name:
        return "heterogeneous"
    return "heterogeneous"


def _transmissivity_label(token: str, title_case: bool = False) -> str:
    if token == "isotropic":
        return "Isotropic" if title_case else "isotropic"
    return "Horizontally heterogeneous" if title_case else "horizontally heterogeneous"


def derive_prefix(input_json: Path) -> str:
    ghb_status = "ghb_True" if "ghb_True" in input_json.name else "ghb_False"
    transmissivity_case = _transmissivity_token(input_json.name)
    return f"quickflow_{ghb_status}_{transmissivity_case}"


def format_case_label(input_json: Path) -> str:
    ghb_status = "True" if "ghb_True" in input_json.name else "False"
    transmissivity_case = _transmissivity_label(_transmissivity_token(input_json.name))
    return f"GHB={ghb_status}, T={transmissivity_case}"


def format_case_caption(input_json: Path) -> str:
    ghb_token = "True" if "ghb_True" in input_json.name else "False"
    t_token = _transmissivity_label(_transmissivity_token(input_json.name), title_case=True)
    return f"GHB: {ghb_token}, Transmissivity: {t_token}"


def derive_suffix(input_json: Path) -> str:
    suffix = input_json.stem.replace("comparison_results_", "")
    for token in ("ghb_True", "ghb_False", "isotropic", "heterogeneous", "anisotropic"):
        suffix = suffix.replace(token, "")
    suffix = suffix.replace("__", "_").strip("_")
    return suffix or "extra"


def process_validation_file(input_json: Path, out_dir: Path, prefix: str | None = None) -> None:
    input_json = input_json.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if prefix is None:
        prefix = derive_prefix(input_json)

    raw = json.loads(input_json.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Expected a JSON object at the top level (dict keyed by case name).")

    rows: list[dict] = []
    for case_name, entry in raw.items():
        if not isinstance(entry, dict):
            continue

        nx = int(entry.get("nx", 0))
        ny = int(entry.get("ny", 0))
        n_cells_total = int(entry.get("n_cells_total", nx * ny))
        n_cells_active = int(entry.get("n_cells_active", n_cells_total))

        row: dict = {
            "case": str(case_name),
            "nx": nx,
            "ny": ny,
            "n_cells_total": n_cells_total,
            "n_cells_active": n_cells_active,
        }

        diagnostics = entry.get("diagnostics", {})
        if isinstance(diagnostics, dict):
            for k, v in diagnostics.items():
                row["diag_" + str(k)] = v

        timings = entry.get("timings", {})
        if isinstance(timings, dict):
            for k, v in timings.items():
                row["time_" + str(k)] = v

        mf_vs_fd = entry.get("mf_vs_fd", {})
        if isinstance(mf_vs_fd, dict):
            for k, v in mf_vs_fd.items():
                row["mf_vs_fd_" + str(k)] = v

        kcycle_vs_mf = entry.get("k_cycle_vs_mf", {})
        if isinstance(kcycle_vs_mf, dict):
            for k, v in kcycle_vs_mf.items():
                row["kcycle_vs_mf_" + str(k)] = v

        fd_vs_kcycle = entry.get("fd_vs_k_cycle", {})
        if isinstance(fd_vs_kcycle, dict):
            for k, v in fd_vs_kcycle.items():
                row["fd_vs_kcycle_" + str(k)] = v

        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No valid cases found in the JSON.")

    df = df.sort_values(["n_cells_active", "nx", "ny"], ascending=[True, True, True]).reset_index(drop=True)

    # Full flattened table
    out_all = out_dir.joinpath(f"{prefix}_all_fields.csv")
    df.to_csv(out_all, index=False)

    # Runtime-focused table
    desired_runtime_cols = [
        "case",
        "nx",
        "ny",
        "n_cells_active",
        "time_warp_seconds_cold_start",
        "time_warp_seconds_warm_start",
        "time_mf6_seconds",
        "time_fd_seconds",
        "diag_n_cycles_used",
        "diag_n_levels",
        "diag_converged",
    ]
    runtime_cols = []
    for c in desired_runtime_cols:
        if c in df.columns:
            runtime_cols.append(c)

    runtime_df = df[runtime_cols].copy()

    # Vectorized speedups
    if "time_warp_seconds_warm_start" in df.columns and "time_mf6_seconds" in df.columns:
        warp_warm = df["time_warp_seconds_warm_start"].to_numpy(dtype=float)
        mf6_t = df["time_mf6_seconds"].to_numpy(dtype=float)
        speedup = np.full(warp_warm.shape, np.nan, dtype=float)
        valid = (warp_warm > 0.0) & np.isfinite(warp_warm) & np.isfinite(mf6_t)
        speedup[valid] = mf6_t[valid] / warp_warm[valid]
        runtime_df["speedup_mf6_over_warp_warm"] = speedup

    if "time_warp_seconds_warm_start" in df.columns and "time_fd_seconds" in df.columns:
        warp_warm = df["time_warp_seconds_warm_start"].to_numpy(dtype=float)
        fd_t = df["time_fd_seconds"].to_numpy(dtype=float)
        speedup = np.full(warp_warm.shape, np.nan, dtype=float)
        valid = (warp_warm > 0.0) & np.isfinite(warp_warm) & np.isfinite(fd_t)
        speedup[valid] = fd_t[valid] / warp_warm[valid]
        runtime_df["speedup_fd_over_warp_warm"] = speedup

    out_runtime = out_dir.joinpath(f"{prefix}_runtime_table.csv")
    runtime_df.to_csv(out_runtime, index=False)

    # Accuracy-focused table (if present)
    desired_acc_cols = [
        "case",
        "nx",
        "ny",
        "n_cells_active",
        "kcycle_vs_mf_rmse",
        "kcycle_vs_mf_max_abs_diff",
        "mf_vs_fd_rmse",
        "mf_vs_fd_max_abs_diff",
        "fd_vs_kcycle_rmse",
        "fd_vs_kcycle_max_abs_diff",
    ]
    acc_cols = []
    for c in desired_acc_cols:
        if c in df.columns:
            acc_cols.append(c)

    if len(acc_cols) >= 5:
        acc_df = df[acc_cols].copy()

        out_acc = out_dir.joinpath(f"{prefix}_accuracy_table.csv")
        acc_df.to_csv(out_acc, index=False, float_format="%.3g")

        out_acc_tex = out_dir.joinpath(f"{prefix}_accuracy_table.tex")
        case_caption = format_case_caption(input_json)
        caption = (
            "Accuracy comparison of DarcyWarp against MODFLOW 6 and SciPy "
            f"finite-difference solutions on synthetic 2D steady-state groundwater flow problems ({case_caption})"
        )
        label = f"tab:{prefix}_accuracy"

        with out_acc_tex.open("w", encoding="utf-8") as f:
            f.write("% Auto-generated table\n")
            f.write("\\begin{table}[htbp]\n")
            f.write("\\centering\n")
            f.write("\\scriptsize\n")
            f.write("\\setlength{\\tabcolsep}{3pt}\n")
            f.write("\\renewcommand{\\arraystretch}{1.05}\n")
            f.write(f"\\caption{{{caption}}}\n")
            f.write(f"\\label{{{label}}}\n")

            # Wrap tabular in resizebox to force it to fit the page width
            f.write("\\resizebox{\\textwidth}{!}{%\n")
            f.write("\\begin{tabular}{lrrrrrrrrr}\n")
            f.write("\\hline\n")
            f.write(
                "Case & $n_x$ & $n_y$ & $N_{act}$ & "
                "\\shortstack{DarcyWarp\\\\vs \\\\MF6\\\\RMSE (m)} & "
                "\\shortstack{DarcyWarp\\\\vs \\\\MF6\\\\Max (m)} & "
                "\\shortstack{MF6\\\\vs\\\\SciPy FD\\\\RMSE (m)} & "
                "\\shortstack{MF6\\\\vs\\\\SciPy FD\\\\Max (m)} & "
                "\\shortstack{SciPy FD\\\\vs \\\\DarcyWarp\\\\RMSE (m)} & "
                "\\shortstack{SciPy FD\\\\vs \\\\DarcyWarp\\\\Max (m)} \\\\\n"
            )
            f.write("\\hline\n")

            n_rows = int(acc_df.shape[0])
            for i in range(n_rows):
                case = str(acc_df.loc[i, "case"])
                case_tex = case.replace("_", "\\_").replace("&", "\\&")

                nx_val = int(acc_df.loc[i, "nx"])
                ny_val = int(acc_df.loc[i, "ny"])
                n_act = int(acc_df.loc[i, "n_cells_active"])

                kcycle_mf_rmse = float(
                    acc_df.loc[i, "kcycle_vs_mf_rmse"]) if "kcycle_vs_mf_rmse" in acc_df.columns else np.nan
                kcycle_mf_max = float(acc_df.loc[
                                          i, "kcycle_vs_mf_max_abs_diff"]) if "kcycle_vs_mf_max_abs_diff" in acc_df.columns else np.nan
                mf_fd_rmse = float(acc_df.loc[i, "mf_vs_fd_rmse"]) if "mf_vs_fd_rmse" in acc_df.columns else np.nan
                mf_fd_max = float(
                    acc_df.loc[i, "mf_vs_fd_max_abs_diff"]) if "mf_vs_fd_max_abs_diff" in acc_df.columns else np.nan
                fd_kcycle_rmse = float(
                    acc_df.loc[i, "fd_vs_kcycle_rmse"]) if "fd_vs_kcycle_rmse" in acc_df.columns else np.nan
                fd_kcycle_max = float(acc_df.loc[
                                          i, "fd_vs_kcycle_max_abs_diff"]) if "fd_vs_kcycle_max_abs_diff" in acc_df.columns else np.nan

                kcycle_mf_rmse_str = f"{kcycle_mf_rmse:.3g}" if np.isfinite(kcycle_mf_rmse) else "-"
                kcycle_mf_max_str = f"{kcycle_mf_max:.3g}" if np.isfinite(kcycle_mf_max) else "-"
                mf_fd_rmse_str = f"{mf_fd_rmse:.3g}" if np.isfinite(mf_fd_rmse) else "-"
                mf_fd_max_str = f"{mf_fd_max:.3g}" if np.isfinite(mf_fd_max) else "-"
                fd_kcycle_rmse_str = f"{fd_kcycle_rmse:.3g}" if np.isfinite(fd_kcycle_rmse) else "-"
                fd_kcycle_max_str = f"{fd_kcycle_max:.3g}" if np.isfinite(fd_kcycle_max) else "-"

                f.write(
                    f"{case_tex} & {nx_val} & {ny_val} & {n_act} & "
                    f"{kcycle_mf_rmse_str} & {kcycle_mf_max_str} & "
                    f"{mf_fd_rmse_str} & {mf_fd_max_str} & "
                    f"{fd_kcycle_rmse_str} & {fd_kcycle_max_str} \\\\\n"
                )

            f.write("\\hline\n")
            f.write("\\end{tabular}%\n")
            f.write("}\n")  # end resizebox
            f.write("\\end{table}\n")


    # Compact LaTeX table for runtime
    out_tex = out_dir.joinpath(f"{prefix}_runtime_table.tex")
    case_caption = format_case_caption(input_json)
    caption = (
        "Performance comparison of DarcyWarp and MODFLOW 6 on synthetic 2D "
        f"steady-state groundwater flow problems ({case_caption})"
    )
    label = f"tab:{prefix}_runtime"

    with out_tex.open("w", encoding="utf-8") as f:
        f.write("% Auto-generated table\n")
        f.write("\\begin{table}[htbp]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write(f"\\label{{{label}}}\n")
        f.write("\\begin{tabular}{lrrrrrr}\n")
        f.write("\\hline\n")
        f.write("Case & $n_x$ & $n_y$ & $N_{act}$ & Warp warm (s) & MF6 (s) & Speedup \\\\\n")
        f.write("\\hline\n")

        have_warp_warm = "time_warp_seconds_warm_start" in runtime_df.columns
        have_mf6 = "time_mf6_seconds" in runtime_df.columns
        have_speed = "speedup_mf6_over_warp_warm" in runtime_df.columns

        n_rows = int(runtime_df.shape[0])
        for i in range(n_rows):
            case = str(runtime_df.loc[i, "case"])
            nx_val = int(runtime_df.loc[i, "nx"])
            ny_val = int(runtime_df.loc[i, "ny"])
            n_act = int(runtime_df.loc[i, "n_cells_active"])

            warp_warm_s = float(runtime_df.loc[i, "time_warp_seconds_warm_start"]) if have_warp_warm else np.nan
            mf6_s = float(runtime_df.loc[i, "time_mf6_seconds"]) if have_mf6 else np.nan
            sp = float(runtime_df.loc[i, "speedup_mf6_over_warp_warm"]) if have_speed else np.nan

            warp_warm_str = f"{warp_warm_s:.3g}" if np.isfinite(warp_warm_s) else "-"
            mf6_str = f"{mf6_s:.3g}" if np.isfinite(mf6_s) else "-"
            sp_str = f"{sp:.3g}" if np.isfinite(sp) else "-"

            f.write(f"{case} & {nx_val} & {ny_val} & {n_act} & {warp_warm_str} & {mf6_str} & {sp_str} \\\\\n")

        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    # Plots
    x = runtime_df["n_cells_active"].to_numpy(dtype=float)
    case_label = format_case_label(input_json)

    # Runtime vs active cells
    plt.figure()
    if "time_warp_seconds_warm_start" in runtime_df.columns:
        plt.plot(x, runtime_df["time_warp_seconds_warm_start"].to_numpy(dtype=float), marker="o", linestyle="-", label="Warp (warm)")
    if "time_warp_seconds_cold_start" in runtime_df.columns:
        plt.plot(x, runtime_df["time_warp_seconds_cold_start"].to_numpy(dtype=float), marker="o", linestyle="--", label="Warp (cold)")
    if "time_mf6_seconds" in runtime_df.columns:
        plt.plot(x, runtime_df["time_mf6_seconds"].to_numpy(dtype=float), marker="o", linestyle="-", label="MODFLOW 6")
    if "time_fd_seconds" in runtime_df.columns:
        plt.plot(x, runtime_df["time_fd_seconds"].to_numpy(dtype=float), marker="o", linestyle="-", label="SciPy FD direct")

    plt.xlabel("Active cells")
    plt.ylabel("Runtime (s)")
    plt.title(f"Runtime vs active cells ({case_label})")
    plt.legend()
    if PLOT_LOGLOG:
        plt.xscale("log")
        plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir.joinpath(f"{prefix}_runtime_vs_cells.png"), dpi=200)
    plt.savefig(out_dir.joinpath(f"{prefix}_runtime_vs_cells.pdf"))
    plt.close()

    # Speedup MF6 over Warp warm
    if "time_warp_seconds_warm_start" in runtime_df.columns and "time_mf6_seconds" in runtime_df.columns:
        warp_warm = runtime_df["time_warp_seconds_warm_start"].to_numpy(dtype=float)
        mf6_t = runtime_df["time_mf6_seconds"].to_numpy(dtype=float)
        speedup = np.full(warp_warm.shape, np.nan, dtype=float)
        valid = (warp_warm > 0.0) & np.isfinite(warp_warm) & np.isfinite(mf6_t)
        speedup[valid] = mf6_t[valid] / warp_warm[valid]

        plt.figure()
        plt.plot(x, speedup, marker="o", linestyle="-")
        plt.xlabel("Active cells")
        plt.ylabel("Speedup (MF6 time / Warp warm time)")
        plt.title(f"Speedup vs active cells ({case_label})")
        if PLOT_LOGLOG:
            plt.xscale("log")
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(out_dir.joinpath(f"{prefix}_speedup_mf6_over_warp_warm.png"), dpi=200)
        plt.savefig(out_dir.joinpath(f"{prefix}_speedup_mf6_over_warp_warm.pdf"))
        plt.close()

    # RMSE plots
    if "kcycle_vs_mf_rmse" in runtime_df.columns:
        plt.figure()
        plt.plot(x, runtime_df["kcycle_vs_mf_rmse"].to_numpy(dtype=float), marker="o", linestyle="-")
        plt.xlabel("Active cells")
        plt.ylabel("RMSE (m)")
        plt.title(f"Warp vs MF6 head RMSE ({case_label})")
        if PLOT_LOGLOG:
            plt.xscale("log")
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(out_dir.joinpath(f"{prefix}_rmse_warp_vs_mf6.png"), dpi=200)
        plt.savefig(out_dir.joinpath(f"{prefix}_rmse_warp_vs_mf6.pdf"))
        plt.close()

    if "fd_vs_kcycle_rmse" in runtime_df.columns:
        plt.figure()
        plt.plot(x, runtime_df["fd_vs_kcycle_rmse"].to_numpy(dtype=float), marker="o", linestyle="-")
        plt.xlabel("Active cells")
        plt.ylabel("RMSE (m)")
        plt.title(f"Warp vs FD head RMSE ({case_label})")
        if PLOT_LOGLOG:
            plt.xscale("log")
            plt.yscale("log")
        plt.tight_layout()
        plt.savefig(out_dir.joinpath(f"{prefix}_rmse_warp_vs_fd.png"), dpi=200)
        plt.savefig(out_dir.joinpath(f"{prefix}_rmse_warp_vs_fd.pdf"))
        plt.close()

    print(f"Processed: {input_json}")
    print(f"Wrote: {out_all}")
    print(f"Wrote: {out_runtime}")
    if len(acc_cols) >= 5:
        print(f"Wrote: {out_dir.joinpath(f'{prefix}_accuracy_table.csv')}")
    print(f"Wrote: {out_tex}")
    print(f"Wrote plots to: {out_dir}")


if __name__ == "__main__":
    data_dir = DATA_DIR.expanduser().resolve()
    out_dir = OUT_DIR.expanduser().resolve()
    input_files = sorted(data_dir.glob(INPUT_GLOB))
    if not input_files:
        raise FileNotFoundError(f"No files matched {INPUT_GLOB} in {data_dir}")

    files_by_prefix: dict[str, list[Path]] = {}
    for input_json in input_files:
        base_prefix = derive_prefix(input_json)
        files_by_prefix.setdefault(base_prefix, []).append(input_json)

    for base_prefix in sorted(files_by_prefix):
        prefix_files = files_by_prefix[base_prefix]
        if len(prefix_files) == 1:
            process_validation_file(prefix_files[0], out_dir, prefix=base_prefix)
            continue

        # Keep the unsuffixed prefix tied to the most recently updated source file.
        canonical_file = max(prefix_files, key=lambda p: (p.stat().st_mtime_ns, p.name))
        process_validation_file(canonical_file, out_dir, prefix=base_prefix)

        for input_json in sorted(prefix_files, key=lambda p: p.name):
            if input_json == canonical_file:
                continue
            suffix = derive_suffix(input_json)
            process_validation_file(input_json, out_dir, prefix=f"{base_prefix}_{suffix}")
