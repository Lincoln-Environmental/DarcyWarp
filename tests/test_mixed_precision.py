"""Minimal regression coverage for the EXPERIMENTAL mixed-precision solver.

The mixed-precision defect-correction solver
(``DARCY_WARP_PACKAGE.solvers.mixed_precision``) is retained as an
experimental, opt-in, non-production reference.  These tests protect only the
essentials:

1. it runs and returns a finite, correctly shaped head on a small
   heterogeneous steady confined case with GHB;
2. it agrees closely with the production FP64 K-cycle backend on that case;
3. it remains experimental, opt-in, and unreachable through the production
   solver registry/aliases.

The mixed path requires a model built under ``DARCY_FLOAT=float32`` while the
FP64 reference requires ``DARCY_FLOAT=float64``; both are pinned at import
time, so the two solves run in child processes with the env var set before
any DarcyWarp import.  No benchmark runtimes are encoded here.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


def _cuda_available() -> bool:
    if not _warp_available():
        return False
    try:
        import warp as wp

        return bool(wp.is_cuda_available())
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _warp_available(), reason="warp is not available")

requires_cuda = pytest.mark.skipif(not _cuda_available(), reason="CUDA is not available")

# Small deterministic case: heterogeneous T (ugly_t, seed 123) + GHB.
# min_coarse_cells=100 keeps the multigrid hierarchy multilevel on this small
# grid (the default 500 would collapse it to a single level and stall).
_NX, _NY = 48, 40

_CHILD_PROGRAM = r"""
import json
import sys

import numpy as np

mode, out_path = sys.argv[1], sys.argv[2]

from DARCY_WARP_PACKAGE.model_builder import (
    _build_dem,
    _build_domain,
    build_truth_inputs,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

NX, NY = 48, 40
DX, THICKNESS, R_TRUTH, SEED = 100.0, 300.0, 1.0e-4, 123

domain = _build_domain(nx=NX, ny=NY)
dem = _build_dem(domain)
T_field = make_ugly_T_field(nx=NX, ny=NY, domain=domain, seed=SEED)
R_field = np.full_like(domain, R_TRUTH, dtype=np.float64)

with WarpDarcySolver(
    nx=NX, ny=NY, dx=DX, device="cuda:0",
    use_ghb=True, solver_type="pcg", aq_thickness=THICKNESS,
) as solver:
    solver.build_from_truth_inputs(T_truth=T_field, R_truth=R_field, width=DX)
    if mode == "fp64":
        head, info = solver.solve_multigrid_kcycle(
            max_cycles=200, nu_pre=2, nu_post=2, nu_coarse=2, omega=0.7,
            rel_tol=5.0e-7, abs_tol_min=5.0e-7, initial_head=dem,
            return_info=True, max_levels=6, check_every_no=5, min_coarse_cells=100,
        )
        head = np.asarray(head, dtype=np.float64)
    elif mode == "mixed":
        from DARCY_WARP_PACKAGE.solvers.mixed_precision import (
            MixedPrecisionDefectCorrectionSession,
        )

        (_, _, _, _, bc_values64, _, gh_head64, _) = build_truth_inputs(
            nx=NX, ny=NY, dx=DX, T_truth=T_field, R_truth=R_field,
            use_ghb=True, width=DX,
        )
        session = MixedPrecisionDefectCorrectionSession(
            solver,
            bc_values_f64=bc_values64,
            gh_head_f64=gh_head64,
            R_f64=R_field,
            max_levels=6,
            min_coarse_cells=100,
        )
        head, info = session.solve(
            dem, inner_kcycles=5, max_outer=40,
            rel_tol=5.0e-7, abs_tol_min=5.0e-7,
        )
        head = np.asarray(head, dtype=np.float64)
    else:
        raise SystemExit(f"unknown mode {mode!r}")

np.savez(out_path, heads=head)
print("RESULT_JSON:" + json.dumps({"converged": bool(info["converged"])}))
"""


def _run_child(mode: str, out_path: Path) -> dict:
    env = dict(os.environ)
    env["DARCY_FLOAT"] = "float32" if mode == "mixed" else "float64"
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", _CHILD_PROGRAM, mode, str(out_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mixed-precision child ({mode}) failed:\n{proc.stderr[-3000:]}"
        )
    marker = "RESULT_JSON:"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    raise RuntimeError(f"mixed-precision child ({mode}) emitted no result")


@pytest.fixture(scope="module")
def mixed_and_fp64_heads(tmp_path_factory):
    if not _cuda_available():
        pytest.skip("CUDA is not available")
    tmp = tmp_path_factory.mktemp("mixed_precision")
    fp64_path = tmp / "fp64.npz"
    mixed_path = tmp / "mixed.npz"
    info_fp64 = _run_child("fp64", fp64_path)
    info_mixed = _run_child("mixed", mixed_path)
    return {
        "fp64": np.load(fp64_path)["heads"],
        "mixed": np.load(mixed_path)["heads"],
        "info_fp64": info_fp64,
        "info_mixed": info_mixed,
    }


@requires_cuda
def test_mixed_precision_runs_and_returns_finite_head(mixed_and_fp64_heads):
    head = mixed_and_fp64_heads["mixed"]
    assert head.shape == (_NY, _NX)
    assert np.all(np.isfinite(head))
    assert mixed_and_fp64_heads["info_mixed"]["converged"] is True


@requires_cuda
def test_mixed_precision_matches_fp64_backend(mixed_and_fp64_heads):
    assert mixed_and_fp64_heads["info_fp64"]["converged"] is True
    # Benchmarks show <= 2.5e-08 m agreement on large cases; the small-case
    # regression tolerance is kept well above that but far below the 2e-4 m
    # production MF6 accuracy gate.
    np.testing.assert_allclose(
        mixed_and_fp64_heads["mixed"],
        mixed_and_fp64_heads["fp64"],
        rtol=0.0,
        atol=1.0e-5,
    )


def test_mixed_precision_remains_experimental_opt_in_non_default():
    from DARCY_WARP_PACKAGE.solver_capabilities import ALIASES, CAPABILITIES, canonical_name

    # Not registered as a backend and not reachable through any alias.
    assert not any("mixed" in name for name in CAPABILITIES)
    assert not any("mixed" in name for name in ALIASES)
    assert not any("mixed" in target for target in ALIASES.values())
    for formulation in ("confined", "unconfined"):
        with pytest.raises(ValueError, match="unknown 2D solver backend"):
            canonical_name(
                "mixed_defect_correction", formulation=formulation, default="kcycle"
            )

    from DARCY_WARP_PACKAGE.solvers import mixed_precision

    assert mixed_precision.EXPERIMENTAL is True
    assert "non-production" in mixed_precision._EXPERIMENTAL_WARNING
