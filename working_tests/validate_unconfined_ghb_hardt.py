#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Validation for GHB + hard-T (ugly_t) options in the steady 2D unconfined
runner (``run_2d_unconfined_warp_vs_mf6.py``).

Runs cases for:
  (a) uniform + GHB, (b) ugly_t (no GHB), (c) ugly_t + GHB,
  (d) ugly_t with a second seed, (e) a non-square grid,
  (f) a weak-GHB DRAINING case whose heads approach bottom + min_sat,
each with ``inner_implementation`` classic AND fast, plus the default config
(uniform, no GHB) regression.  MF6 runs ONCE per case (shared across impls;
reused on re-runs via fingerprint-matched artifacts).  GHB conductances come
from the runner's MF6-side fixed point (Warp heads are never used).

Gates (per case/impl):
  * converged with strict status "Nonlinear head-change and inner residual
    tolerances met."
  * max_abs vs MF6 < 5e-4 m (the runner's agreement tol)
  * fast-vs-classic head diff < 5e-6 m at RUNNER DEFAULTS (acceptance-basin
    bound), plus a hard-T pair at hclose=1e-6 which must agree < 1e-6 m.

Usage:
    python working_tests/validate_unconfined_ghb_hardt.py \
        [--workspace /tmp/dw_unconf_ghb_hardt] [--nx 500] [--ny 500] \
        [--combos uniform+ghb,ugly_t] [--skip-tight]
    (re-running with the same workspace reuses fingerprint-matching MF6 heads)
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import working_tests.run_2d_unconfined_warp_vs_mf6 as R  # noqa: E402

STRICT_STATUS = "Nonlinear head-change and inner residual tolerances met."
MF6_TOL = 5.0e-4
DEFAULT_FAST_CLASSIC_TOL = 5.0e-6  # hclose=1e-4 acceptance-basin width (documented)
TIGHT_FAST_CLASSIC_TOL = 1.0e-6  # hclose=1e-6 operator-equivalence gate

# Combo overrides: any key accepted by run_case / build_simple_unconfined_case.
COMBOS = [
    ("uniform+ghb", dict(t_field_kind="uniform", use_ghb=True)),
    ("ugly_t", dict(t_field_kind="ugly_t", use_ghb=False)),
    ("ugly_t+ghb", dict(t_field_kind="ugly_t", use_ghb=True)),
    ("ugly_t_s123", dict(t_field_kind="ugly_t", t_field_seed=123, use_ghb=False)),
    ("nonsquare+ghb", dict(t_field_kind="uniform", use_ghb=True, nx_scale=1.4, ny_scale=0.6)),
    # Weak-GHB draining case: low K, near-zero recharge, narrow river width
    # and a GHB stage just above the bottom, so the centre GHB row drains the
    # aquifer hard (min saturated thickness ~31 m vs 100 m initial / 300 m
    # full at the GHB row) while the GHB does NOT clamp the row (coupling
    # ratio << 1, reported in the case summary).
    #
    # DESIGN NOTE (measured): this boundary family (Dirichlet at land
    # surface on 3 edges) has a drying bifurcation — any stronger drainage
    # (larger ghb_width, taller domain, lower stage) makes MF6 DEACTIVATE
    # cells (HDRY sentinel), which the runner's finite-heads trust gate
    # rejects and which is a different truth semantics than Warp's min_sat
    # floor (cells never deactivate).  A case literally reaching
    # bottom+min_sat (0.1 m) is therefore impossible without MF6 drying;
    # this is the strongest non-drying drainage in the family.  The grid is
    # PINNED at 100x120 (probed: 100x140 already dries MF6 cells), so this
    # combo ignores --nx/--ny.
    (
        "drain_weak_ghb",
        dict(
            t_field_kind="uniform",
            use_ghb=True,
            hydraulic_conductivity=1.0,
            recharge=1.0e-7,
            ghb_width=1.0,
            ghb_head_elevation=0.3,
            nx=100,
            ny=120,
        ),
    ),
]
IMPLS = ("classic", "fast")


def _warp_kwargs(impl: str) -> dict:
    return dict(
        device="auto",
        do_double_solve=False,
        diag_preconditioner_backend="device",
        check_every_no=5,
        inner_implementation=impl,
    )


def _case_kwargs(combo: dict, nx: int, ny: int) -> dict:
    """Merge combo overrides with the base grid/physics parameters."""
    kw = dict(
        nx=int(combo.get("nx", round(nx * float(combo.get("nx_scale", 1.0))))),
        ny=int(combo.get("ny", round(ny * float(combo.get("ny_scale", 1.0))))),
        hydraulic_conductivity=float(combo.get("hydraulic_conductivity", 100.0)),
        recharge=float(combo.get("recharge", 1.0e-4)),
        t_field_kind=str(combo.get("t_field_kind", "uniform")),
        t_field_seed=int(combo.get("t_field_seed", 42)),
        use_ghb=bool(combo.get("use_ghb", False)),
        ghb_width=float(combo.get("ghb_width", 100.0)),
        ghb_head_elevation=combo.get("ghb_head_elevation"),
        # This validator deliberately exercises the independent MF6-side
        # conductance-law fixed point. The main benchmark defaults to the
        # faster warp-matched equation-equivalence comparison.
        ghb_conductance_mode="fixed_point",
    )
    return kw


def _load_summary(ws: Path) -> dict:
    import json

    return json.load(open(ws / "unconfined_benchmark_summary.json"))


def run_validation(
    workspace: str | Path = "/tmp/dw_unconf_ghb_hardt",
    nx: int = 500,
    ny: int = 500,
    combos: list[str] | None = None,
    skip_tight: bool = False,
) -> bool:
    """Run the validation suite.  Returns True when every gate passes."""
    base = Path(workspace)
    base.mkdir(parents=True, exist_ok=True)

    selected = COMBOS
    if combos:
        wanted = {c.strip() for c in combos}
        selected = [c for c in COMBOS if c[0] in wanted]
        missing = wanted - {c[0] for c in selected}
        if missing:
            raise ValueError(f"unknown combos: {sorted(missing)}")

    all_ok = True
    rows = []
    for case_name, combo in selected:
        kw = _case_kwargs(combo, nx, ny)
        mf6_dir = base / case_name.replace("+", "_")
        mf6_path = mf6_dir / "mf6_heads.npz"
        case = R.build_simple_unconfined_case(workspace=mf6_dir, **kw)
        if not R.validate_mf6_artifact(mf6_path, case):
            if case.use_ghb:
                # Cold-cache GHB path: the runner's MF6-side fixed point
                # generates the truth (no Warp heads involved).
                R.run_case(
                    workspace=mf6_dir,
                    do_run_mf6=True,
                    do_run_warp=False,
                    **_warp_kwargs("fast"),
                    **kw,
                )
            else:
                R.run_mf6_unconfined(case, out_path=mf6_path)
        heads = {}
        for impl in IMPLS:
            ws = base / f"{case_name.replace('+', '_')}_{impl}"
            ws.mkdir(parents=True, exist_ok=True)
            ws_case = R.build_simple_unconfined_case(workspace=ws, **kw)
            if not R.validate_mf6_artifact(ws / "mf6_heads.npz", ws_case):
                # Validate the source artifact against THIS case before
                # copying (never copy a stale/incompatible artifact).
                R.validate_mf6_artifact(mf6_path, ws_case, raise_on_mismatch=True)
                shutil.copy(mf6_path, ws / "mf6_heads.npz")
            summary = R.run_case(
                workspace=ws,
                do_run_mf6=False,
                do_run_warp=True,
                **_warp_kwargs(impl),
                **kw,
            )
            report = summary.get("convergence_report") or {}
            comp = summary.get("comparison") or {}
            status = str(report.get("status"))
            ok = (
                bool(summary.get("solve2_converged"))
                and status == STRICT_STATUS
                and comp.get("max_abs_diff") is not None
                and float(comp["max_abs_diff"]) < MF6_TOL
            )
            all_ok = all_ok and ok
            rows.append((case_name, impl, summary, comp, ok))
            with np.load(ws / "warp_heads.npz", allow_pickle=False) as npz:
                heads[impl] = np.asarray(npz["heads"], dtype=np.float64)
        diff = float(np.max(np.abs(heads["fast"] - heads["classic"])))
        basin_ok = diff < DEFAULT_FAST_CLASSIC_TOL
        all_ok = all_ok and basin_ok
        rows.append((case_name, "f-vs-c", {"warp_solve2_time": None}, {"max_abs_diff": diff}, basin_ok))

    print(f"\n{'case':15s} | {'impl':6s} | conv | outer | solve2(s) | RMSE vs MF6 | max|dh| vs MF6 | gate")
    print("-" * 108)
    for case_name, impl, summary, comp, ok in rows:
        if impl == "f-vs-c":
            print(f"{case_name:15s} | fast-vs-classic head diff: {comp['max_abs_diff']:.3e} m "
                  f"(basin tol {DEFAULT_FAST_CLASSIC_TOL:.0e}) {'OK' if ok else 'FAIL'}")
            continue
        print(
            f"{case_name:15s} | {impl:6s} | {str(summary.get('solve2_converged')):4s} | "
            f"{int(summary.get('solve2_outer_iterations') or -1):5d} | "
            f"{float(summary.get('warp_solve2_time') or 0.0):9.2f} | "
            f"{(comp.get('rmse') or float('nan')):11.3e} | "
            f"{(comp.get('max_abs_diff') or float('nan')):15.3e} | {'OK' if ok else 'FAIL'}"
        )

    if not skip_tight:
        # Hard-T operator equivalence at a tight fixed point (hclose=1e-6).
        print("\nhard-T tightened cross-check (hclose=1e-6):")
        old_dh_tol = R.DEFAULT_DH_TOL
        R.DEFAULT_DH_TOL = 1.0e-6
        try:
            case = R.build_simple_unconfined_case(
                nx=int(nx), ny=int(ny), t_field_kind="ugly_t",
                workspace=base / "ugly_t_tight",
            )
            tight = {}
            for impl in IMPLS:
                ws = base / f"ugly_t_hclose_{impl}"
                ws.mkdir(parents=True, exist_ok=True)
                R.run_warp_unconfined(case, out_path=ws / "warp_heads.npz",
                                      inner_head_residual_tol_min=1.0e-7,
                                      inner_head_residual_tol_max=1.0e-5,
                                      **_warp_kwargs(impl))
                with np.load(ws / "warp_heads.npz", allow_pickle=False) as npz:
                    tight[impl] = np.asarray(npz["heads"], dtype=np.float64)
        finally:
            R.DEFAULT_DH_TOL = old_dh_tol
        tight_diff = float(np.max(np.abs(tight["fast"] - tight["classic"])))
        tight_ok = tight_diff < TIGHT_FAST_CLASSIC_TOL
        all_ok = all_ok and tight_ok
        print(f"  ugly_t fast-vs-classic @hclose=1e-6: {tight_diff:.3e} m "
              f"(tol {TIGHT_FAST_CLASSIC_TOL:.0e}) {'OK' if tight_ok else 'FAIL'}")

    # Default-config regression (uniform, no GHB) — gated like every other case.
    print("\ndefault-config regression (uniform, no GHB):")
    for impl in IMPLS:
        ws = base / f"default_{impl}"
        summary = R.run_case(nx=int(nx), ny=int(ny), hydraulic_conductivity=100.0, workspace=ws,
                             do_run_mf6=not R.validate_mf6_artifact(
                                 ws / "mf6_heads.npz",
                                 R.build_simple_unconfined_case(
                                     nx=int(nx), ny=int(ny),
                                     hydraulic_conductivity=100.0, workspace=ws)),
                             do_run_warp=True, **_warp_kwargs(impl))
        comp = summary.get("comparison") or {}
        report = summary.get("convergence_report") or {}
        ok = (
            bool(summary.get("solve2_converged"))
            and str(report.get("status")) == STRICT_STATUS
            and comp.get("max_abs_diff") is not None
            and float(comp["max_abs_diff"]) < MF6_TOL
        )
        all_ok = all_ok and ok
        print(
            f"  {impl:8s}: solve2={summary.get('warp_solve2_time'):.2f}s, "
            f"outer={summary.get('solve2_outer_iterations')}, "
            f"rmse={(comp.get('rmse') or float('nan')):.3e}, "
            f"max_abs={(comp.get('max_abs_diff') or float('nan')):.3e}, "
            f"mf6_engine={summary.get('mf6_engine_time') or 0.0:.1f}s "
            f"{'OK' if ok else 'FAIL'}"
        )

    print("\nOVERALL:", "PASS" if all_ok else "FAIL")
    return all_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="/tmp/dw_unconf_ghb_hardt")
    parser.add_argument("--nx", type=int, default=500)
    parser.add_argument("--ny", type=int, default=500)
    parser.add_argument(
        "--combos",
        default=None,
        help="comma-separated subset of combo names to run (default: all)",
    )
    parser.add_argument("--skip-tight", action="store_true",
                        help="skip the hclose=1e-6 tightened cross-check")
    args = parser.parse_args()
    ok = run_validation(
        workspace=args.workspace,
        nx=args.nx,
        ny=args.ny,
        combos=args.combos.split(",") if args.combos else None,
        skip_tight=args.skip_tight,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
