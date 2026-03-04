from __future__ import annotations

import argparse
from pathlib import Path
from DARCY_WARP_PACKAGE.project_base import proj_root

import numpy as np


def _default_out_dir(in_dir: Path, output_dtype: str) -> Path:
    suffix = "f32" if output_dtype == "float32" else "f64"
    return in_dir.with_name(f"{in_dir.name}_{suffix}")


def _iter_npz_paths(in_dir: Path, pattern: str, recursive: bool) -> list[Path]:
    if recursive:
        return sorted(in_dir.rglob(pattern))
    return sorted(in_dir.glob(pattern))


def _downcast_file(
    in_path: Path,
    out_path: Path,
    output_dtype: str,
    overwrite: bool,
) -> bool:
    if out_path.exists() and not overwrite:
        return False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    float_dtype = np.dtype(output_dtype)

    with np.load(in_path) as data:
        arrays: dict[str, np.ndarray] = {}
        for key in data.files:
            arr = data[key]
            if np.issubdtype(arr.dtype, np.floating):
                arrays[key] = arr.astype(float_dtype, copy=False)
            else:
                arrays[key] = arr

    np.savez_compressed(out_path, **arrays)
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Downcast float arrays in NPZ files to reduce file size."
    )
    parser.add_argument("--in_dir", required=True, help="Directory of .npz files to downcast.")
    parser.add_argument("--out_dir", default="", help="Output directory for downcast files.")
    parser.add_argument("--pattern", default="*.npz", help="Glob pattern for .npz files.")
    parser.add_argument("--recursive", action="store_true", help="Search for files recursively.")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help="Target float dtype for saved arrays.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs.")
    args = parser.parse_args(argv)

    in_dir = Path(args.in_dir)
    if in_dir.is_file():
        suffix = "f32" if args.dtype == "float32" else "f64"
        if args.out_dir:
            out_path = Path(args.out_dir)
            if out_path.is_dir():
                out_path = out_path.joinpath(in_dir.name)
        else:
            out_path = in_dir.with_name(f"{in_dir.stem}_{suffix}.npz")
        wrote = _downcast_file(in_dir, out_path, args.dtype, args.overwrite)
        print(f"{'Wrote' if wrote else 'Skipping'} {out_path}")
        return 0

    if not in_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {in_dir}")

    out_dir = Path(args.out_dir) if args.out_dir else _default_out_dir(in_dir, args.dtype)

    if out_dir.resolve() == in_dir.resolve() and not args.overwrite:
        raise ValueError("out_dir matches in_dir; use --overwrite to replace in-place.")

    paths = _iter_npz_paths(in_dir, args.pattern, args.recursive)
    if not paths:
        print("No files matched.")
        return 0

    for path in paths:
        rel_path = path.relative_to(in_dir)
        out_path = out_dir.joinpath(rel_path)
        wrote = _downcast_file(path, out_path, args.dtype, args.overwrite)
        print(f"{'Wrote' if wrote else 'Skipping'} {out_path}")

    return 0


if __name__ == "__main__":
    in_dir = proj_root.joinpath('data/mf6_truth_npz/')
    raise SystemExit(main(["--in_dir", str(in_dir)]))
