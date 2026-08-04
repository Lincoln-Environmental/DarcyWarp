from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np

from DARCY_WARP_PACKAGE.model_builder import _build_domain, _build_dem, make_ugly_T_field
from DARCY_WARP_PACKAGE.modflow_truth import make_mf_model
from DARCY_WARP_PACKAGE.project_base import data_store
from DARCY_WARP_PACKAGE.sanity_case_config import (
    GRID_CASES,
    SPATIAL_GRID_CASES,
    DEFAULT_DX,
    DEFAULT_R_TRUTH,
    DEFAULT_THICKNESS,
    DEFAULT_T_SEED,
    DEFAULT_ISOTROPIC_T,
)


def _parse_cases(cases_arg: str) -> list[str]:
    if not cases_arg:
        return sorted(GRID_CASES.keys())
    return [part.strip() for part in cases_arg.split(",") if part.strip()]


def _run_case_variant(
    label: str,
    nx: int,
    ny: int,
    dx_truth: float,
    r_truth: float,
    thickness: float,
    width: float,
    isotropic: bool,
    t_isotropic_value: float,
    ghb: bool,
    seed: int,
    out_path: str,
    workspace: str,
    output_dtype: str,
) -> str:
    domain = _build_domain(nx=nx, ny=ny)
    _ = _build_dem(domain)

    if isotropic:
        t_field = np.full_like(domain, t_isotropic_value, dtype=np.float64)
    else:
        t_field = make_ugly_T_field(
            nx=nx,
            ny=ny,
            domain=domain,
            seed=int(seed),
        )

    r_field = np.full_like(domain, r_truth, dtype=np.float64)
    hk_field = t_field / thickness

    heads, _ = make_mf_model(
        nx=nx,
        ny=ny,
        grid_size=dx_truth,
        nper=1,
        workspace=Path(workspace),
        hk=hk_field,
        recharge=r_field,
        run=True,
        use_ghb=ghb,
    )

    float_dtype = np.dtype(output_dtype)
    np.savez_compressed(
        out_path,
        heads=np.asarray(heads, dtype=float_dtype),
        nx=np.int32(nx),
        ny=np.int32(ny),
        dx=np.asarray(dx_truth, dtype=float_dtype),
        ghb=np.int32(1 if ghb else 0),
        t_isotropic=np.int32(1 if isotropic else 0),
        t_isotropic_value=np.asarray(t_isotropic_value, dtype=float_dtype),
        thickness=np.asarray(thickness, dtype=float_dtype),
        width=np.asarray(width, dtype=float_dtype),
        r_truth=np.asarray(r_truth, dtype=float_dtype),
        seed=np.int32(seed),
        label=np.array(label),
    )

    return out_path


def main(argv: list[str] | None = None) -> int:
    workers_default = max(1, int((os.cpu_count() or 1) // 1.5))
    parser = argparse.ArgumentParser(description="Export MF6 truth heads to compressed NPZ.")
    parser.add_argument("--out_dir", type=str, default=str(data_store.joinpath("mf6_truth_npz")))
    parser.add_argument("--cases", type=str, default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=int, default=workers_default)
    parser.add_argument("--t_isotropic_value", type=float, default=float(DEFAULT_ISOTROPIC_T))
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Float dtype for saved arrays.",
    )
    ghb_group = parser.add_mutually_exclusive_group()
    ghb_group.add_argument("--ghb", action="store_true", help="Only run GHB=True cases.")
    ghb_group.add_argument("--no_ghb", action="store_true", help="Only run GHB=False cases.")
    t_group = parser.add_mutually_exclusive_group()
    t_group.add_argument("--isotropic", action="store_true", help="Only run isotropic T cases.")
    t_group.add_argument(
        "--heterogeneous",
        action="store_true",
        help="Only run horizontally heterogeneous T cases.",
    )
    t_group.add_argument(
        "--anisotropic",
        action="store_true",
        help="Legacy flag for horizontally heterogeneous T cases.",
    )

    args = parser.parse_args(argv)

    if args.ghb:
        ghb_flags = [True]
    elif args.no_ghb:
        ghb_flags = [False]
    else:
        ghb_flags = [True, False]

    if args.isotropic:
        isotropic_flags = [True]
    elif args.heterogeneous or args.anisotropic:
        isotropic_flags = [False]
    else:
        isotropic_flags = [True, False]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dx_truth = float(DEFAULT_DX)
    r_truth = float(DEFAULT_R_TRUTH)
    thickness = float(DEFAULT_THICKNESS)
    width = dx_truth
    t_isotropic_value = float(args.t_isotropic_value)
    seed = int(DEFAULT_T_SEED)
    output_dtype = str(args.dtype)

    labels = _parse_cases(args.cases)
    jobs: list[tuple] = []
    for label in labels:
        # Defaults come from the automatic steady view (GRID_CASES); explicit
        # labels may name any catalog grid, including manual_only ones.
        if label not in SPATIAL_GRID_CASES:
            raise KeyError(f"Unknown case label: {label}")
        cfg = SPATIAL_GRID_CASES[label]
        nx = int(cfg["nx"])
        ny = int(cfg["ny"])

        for isotropic in isotropic_flags:
            for ghb in ghb_flags:
                out_name = f"mf6_truth_{label}_ghb_{ghb}_t_isotropic_{isotropic}.npz"
                out_path = out_dir.joinpath(out_name)
                if out_path.exists() and not args.overwrite:
                    print(f"Skipping existing {out_path}")
                    continue

                ws_name = f"Paper_mf6_truth_{label}_ghb_{ghb}_t_isotropic_{isotropic}"
                ws = data_store.joinpath(ws_name)

                jobs.append(
                    (
                        label,
                        nx,
                        ny,
                        dx_truth,
                        r_truth,
                        thickness,
                        width,
                        isotropic,
                        t_isotropic_value,
                        ghb,
                        seed,
                        str(out_path),
                        str(ws),
                        output_dtype,
                    )
                )

    if not jobs:
        print("No jobs to run.")
        return 0

    if int(args.workers) <= 1 or len(jobs) == 1:
        for job in jobs:
            out_path = _run_case_variant(*job)
            print(f"Wrote {out_path}")
        return 0

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=int(args.workers), mp_context=ctx) as pool:
        futures = [pool.submit(_run_case_variant, *job) for job in jobs]
        for fut in as_completed(futures):
            out_path = fut.result()
            print(f"Wrote {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
