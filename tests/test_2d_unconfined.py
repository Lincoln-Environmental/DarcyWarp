import os
from pathlib import Path

import numpy as np
import pytest


os.environ.setdefault("WARP_CACHE_PATH", str(Path("/tmp/darcywarp-warp-cache")))


def _warp_available() -> bool:
    try:
        import warp  # noqa: F401
    except Exception:
        return False
    return True


pytestmark = pytest.mark.skipif(not _warp_available(), reason="warp is not available")


# ---------------------------------------------------------------------------
# Steady-state MF6 truth comparison across the benchmark grids.
#
# Re-runs the unconfined Warp solver for every truth fixture produced by
# working_tests/regenerate_unconfined_2d_truth.py and checks the heads agree
# with the stored MF6 reference. Each fixture freezes the exact solver
# settings used by run_2d_unconfined_warp_vs_mf6.py (captured from its
# solve2_settings at generation time), so "the same solver settings" are
# guaranteed without this test importing the runner. No MF6 binary is needed
# at test time. Tests skip per-grid when a fixture is absent (e.g. the large
# grids are gitignored on a fresh checkout).
#
# These fixtures are steady-state unconfined truth. They are not evidence that
# the transient 2D unconfined MF6 truth generator has completed; that separate
# artifact is produced by working_tests/run_2d_transient_vs_mf6.py.
# ---------------------------------------------------------------------------

_TRUTH_FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "unconfined_2d"
# Observed Warp-vs-MF6 agreement is ~1e-5 m; ~100x headroom for device/float drift.
_MF6_TRUTH_TOL_M = 1.0e-3


def _warp_device() -> str:
    import warp as wp

    wp.init()
    return "cuda:0" if wp.is_cuda_available() else "cpu"


def _discover_truth_fixtures() -> list[tuple[int, Path]]:
    if not _TRUTH_FIXTURE_DIR.exists():
        return []
    out: list[tuple[int, Path]] = []
    for path in sorted(_TRUTH_FIXTURE_DIR.glob("truth_*x*.npz.lzma")):
        stem = path.name[len("truth_") : -len(".npz.lzma")]
        try:
            nx_str, _ny_str = stem.split("x")
            out.append((int(nx_str), path))
        except ValueError:
            continue
    return out


def _head_metrics(heads_warp, heads_truth, active) -> dict[str, float]:
    from DARCY_WARP_PACKAGE.validation.head_checks import invalid_head_values

    heads_warp = np.asarray(heads_warp, dtype=np.float64)
    heads_truth = np.asarray(heads_truth, dtype=np.float64)
    mask = np.asarray(active) != 0
    assert np.any(mask), "Empty active domain in steady MF6 qualification"
    assert not np.any(mask & invalid_head_values(head=heads_truth)), "Invalid reference head at an active cell"
    assert not np.any(mask & invalid_head_values(head=heads_warp)), "Invalid Warp head at an active cell"
    diff = (heads_warp - heads_truth)[mask]
    abs_diff = np.abs(diff)
    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "n_active": int(mask.sum()),
    }


_TRUTH_FIXTURES = _discover_truth_fixtures()


@pytest.mark.parametrize(
    ("nx", "fixture_path"),
    _TRUTH_FIXTURES,
    ids=[f"{n}x{n}" for n, _ in _TRUTH_FIXTURES],
)
def test_unconfined_warp_matches_mf6_truth_all_grids(nx: int, fixture_path: Path) -> None:
    if not fixture_path.exists():
        pytest.skip(f"truth fixture not present: {fixture_path}")

    from DARCY_WARP_PACKAGE.unconfined_truth_io import build_solve_kwargs, load_truth_artifact
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    artifact = load_truth_artifact(fixture_path)
    cons = artifact["constructor_settings"]
    device = _warp_device()

    with WarpDarcySolver(
        nx=int(cons["nx"]),
        ny=int(cons["ny"]),
        dx=float(cons["dx"]),
        device=device,
        solver_type=str(cons.get("solver_type", "kcycle")),
        diag_preconditioner_backend=str(cons.get("diag_preconditioner_backend", "auto")),
    ) as solver:
        solver.build_from_fields(
            T_field=artifact["initial_transmissivity"],
            R_field=artifact["rhs_recharge"],
            active=artifact["active"],
            bc_mask=artifact["bc_mask"],
            bc_values=artifact["bc_values"],
        )
        heads, info = solver.solve(**build_solve_kwargs(artifact))

    assert info["converged"], "Steady Warp solve did not converge"
    assert heads.shape == artifact["heads"].shape
    metrics = _head_metrics(heads, artifact["heads"], artifact["active"])
    original = artifact["provenance"].get("original_warp_vs_mf6_max_abs_diff")
    print(
        f"[{nx}x{nx}] device={device} converged={bool(info.get('converged'))} "
        f"outer_iter={info.get('outer_iterations')} "
        f"max_abs_diff={metrics['max_abs_diff']:.3e} m (orig={original:.2e}) "
        f"rmse={metrics['rmse']:.3e} tol={_MF6_TRUTH_TOL_M:.0e}"
    )
    assert metrics["max_abs_diff"] <= _MF6_TRUTH_TOL_M, (
        f"{nx}x{nx}: re-run Warp vs MF6 truth max_abs_diff="
        f"{metrics['max_abs_diff']:.3e} m exceeds tol={_MF6_TRUTH_TOL_M:.0e}"
    )


def test_truth_fixtures_present() -> None:
    """Losing a reference must fail collection coverage even on a fresh clone."""
    assert {size for size, path in _TRUTH_FIXTURES} == {50, 100, 250, 500, 1000, 2000, 3000}
