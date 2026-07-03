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


def _build_solver(nx: int = 8, ny: int = 5, *, use_ghb: bool = False, diag_preconditioner_backend: str = "auto"):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=nx,
        ny=ny,
        dx=100.0,
        device="cpu",
        use_ghb=use_ghb,
        solver_type="kcycle",
        diag_preconditioner_backend=diag_preconditioner_backend,
    )
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values[:, 0] = 12.0
    bc_values[:, -1] = 9.0
    recharge = np.full((ny, nx), 1.0e-5, dtype=np.float64)
    transmissivity = np.full((ny, nx), 10.0, dtype=np.float64)
    build_kwargs = {
        "T_field": transmissivity,
        "R_field": recharge,
        "active": active,
        "bc_mask": bc_mask,
        "bc_values": bc_values,
    }
    if use_ghb:
        gh_mask = np.zeros((ny, nx), dtype=np.int32)
        gh_head = np.zeros((ny, nx), dtype=np.float64)
        gh_width = np.zeros((ny, nx), dtype=np.float64)
        gh_mask[1:-1, 1] = 1
        gh_head[1:-1, 1] = 10.5
        gh_width[1:-1, 1] = 25.0
        build_kwargs["gh_mask"] = gh_mask
        build_kwargs["gh_head"] = gh_head
        build_kwargs["gh_width"] = gh_width
    solver.build_from_fields(**build_kwargs)
    return solver, active


def test_confined_2d_solve_still_runs():
    solver, _active = _build_solver()

    head, info = solver.solve_multigrid_kcycle(
        max_cycles=5,
        max_levels=2,
        min_coarse_cells=1,
        check_every_no=1,
        return_info=True,
    )

    assert head.shape == (solver.ny, solver.nx)
    assert info.get("unconfined", False) is False
    assert "outer_history" not in info


def test_diag_preconditioner_device_matches_host_no_ghb():
    import warp as wp

    from DARCY_WARP_PACKAGE.warped_darcy import build_diag_preconditioner

    solver, _active = _build_solver(diag_preconditioner_backend="device")
    m_inv_wp = wp.empty((solver.ny, solver.nx), dtype=wp.float64, device="cpu")
    solver._update_diag_preconditioner_device(
        T_wp=solver.T_wp,
        active_wp=solver.active_wp,
        bc_mask_wp=solver.bc_mask_wp,
        gh_mask_wp=solver.gh_mask_wp,
        ghb_factor_wp=solver.ghb_factor_wp,
        M_inv_wp=m_inv_wp,
        nx=solver.nx,
        ny=solver.ny,
        use_ghb=False,
    )

    m_inv_host = build_diag_preconditioner(
        T_field=solver.T_field_host,
        active=solver.active_host,
        bc_mask=solver.bc_mask_host,
    )
    np.testing.assert_allclose(m_inv_wp.numpy(), m_inv_host, atol=1.0e-12, rtol=1.0e-12)


def test_diag_preconditioner_device_matches_host_with_ghb():
    import warp as wp

    from DARCY_WARP_PACKAGE.warped_darcy import build_diag_preconditioner

    solver, _active = _build_solver(use_ghb=True, diag_preconditioner_backend="device")
    m_inv_wp = wp.empty((solver.ny, solver.nx), dtype=wp.float64, device="cpu")
    solver._update_diag_preconditioner_device(
        T_wp=solver.T_wp,
        active_wp=solver.active_wp,
        bc_mask_wp=solver.bc_mask_wp,
        gh_mask_wp=solver.gh_mask_wp,
        ghb_factor_wp=solver.ghb_factor_wp,
        M_inv_wp=m_inv_wp,
        nx=solver.nx,
        ny=solver.ny,
        use_ghb=True,
    )

    m_inv_host = build_diag_preconditioner(
        T_field=solver.T_field_host,
        active=solver.active_host,
        bc_mask=solver.bc_mask_host,
        gh_mask=solver.gh_mask_host,
        ghb_factor=solver.ghb_factor_host,
        dx=solver.dx,
    )
    np.testing.assert_allclose(m_inv_wp.numpy(), m_inv_host, atol=1.0e-12, rtol=1.0e-12)


def test_confined_heads_match_between_host_and_device_diag_backend():
    solver_host, _active = _build_solver(diag_preconditioner_backend="host")
    head_host, info_host = solver_host.solve_multigrid_kcycle(
        max_cycles=6,
        max_levels=2,
        min_coarse_cells=1,
        check_every_no=1,
        return_info=True,
    )

    solver_device, _active = _build_solver(diag_preconditioner_backend="device")
    head_device, info_device = solver_device.solve_multigrid_kcycle(
        max_cycles=6,
        max_levels=2,
        min_coarse_cells=1,
        check_every_no=1,
        return_info=True,
    )

    np.testing.assert_allclose(head_device, head_host, atol=1.0e-12, rtol=1.0e-12)
    assert info_host["converged"] == info_device["converged"]


def test_unconfined_2d_updates_saturated_thickness_and_transmissivity():
    solver, active_before = _build_solver(diag_preconditioner_backend="device")
    K = np.full((solver.ny, solver.nx), 1.0, dtype=np.float64)
    bottom = np.zeros((solver.ny, solver.nx), dtype=np.float64)
    initial = bottom + 2.0

    head, info = solver.solve(
        formulation="unconfined",
        K_field=K,
        zbot_field=bottom,
        initial_head=initial,
        max_cycles=8,
        max_levels=2,
        min_coarse_cells=1,
        check_every_no=1,
        abs_tol_min=1.0e6,
        max_outer_iterations=6,
        hclose=1.0e-3,
        chebyshev_enabled=True,
        unconfined_startup_mode="confined_pre_solve",
        return_info=True,
    )

    assert head.shape == (solver.ny, solver.nx)
    assert info["unconfined"] is True
    assert info["outer_iterations"] >= 1
    assert len(info["outer_history"]) == info["outer_iterations"]
    assert np.array_equal(solver.active_host, active_before)
    sat_mins = [row["min_saturated_thickness"] for row in info["outer_history"]]
    sat_maxs = [row["max_saturated_thickness"] for row in info["outer_history"]]
    t_mins = [row["min_transmissivity"] for row in info["outer_history"]]
    t_maxs = [row["max_transmissivity"] for row in info["outer_history"]]
    assert all(v > 0.0 for v in sat_mins)
    assert all(hi >= lo for lo, hi in zip(sat_mins, sat_maxs))
    assert all(v > 0.0 for v in t_mins)
    assert all(hi >= lo for lo, hi in zip(t_mins, t_maxs))
    assert any(float(row["picard_update_max"]) > 0.0 for row in info["outer_history"])
    assert any(row["chebyshev_used"] for row in info["outer_history"])
    assert np.all(np.isfinite(head))


def test_unconfined_chebyshev_can_be_disabled():
    solver, _active = _build_solver()
    K = np.full((solver.ny, solver.nx), 1.0, dtype=np.float64)
    bottom = np.zeros((solver.ny, solver.nx), dtype=np.float64)

    _head, info = solver.solve(
        formulation="unconfined",
        K_field=K,
        zbot_field=bottom,
        max_cycles=6,
        max_levels=2,
        min_coarse_cells=1,
        check_every_no=1,
        max_outer_iterations=4,
        chebyshev_enabled=False,
        return_info=True,
    )

    assert info["chebyshev_enabled"] is False
    assert all(not row["chebyshev_used"] for row in info["outer_history"])


def test_unconfined_near_dry_cells_remain_active_and_are_flagged():
    solver, active_before = _build_solver()
    K = np.full((solver.ny, solver.nx), 1.0, dtype=np.float64)
    bottom = np.full((solver.ny, solver.nx), 11.95, dtype=np.float64)
    initial = bottom + 0.01

    _head, info = solver.solve(
        formulation="unconfined",
        K_field=K,
        zbot_field=bottom,
        initial_head=initial,
        max_cycles=4,
        max_levels=2,
        min_coarse_cells=1,
        check_every_no=1,
        max_outer_iterations=3,
        min_saturated_thickness=0.1,
        max_head_change_per_outer_iteration=0.05,
        dry_cell_flag_threshold=0.2,
        return_info=True,
    )

    assert np.array_equal(solver.active_host, active_before)
    assert info["effectively_dry_cell_count"] > 0
    assert info["chebyshev_resets"] >= 0


# ---------------------------------------------------------------------------
# MF6 truth comparison across the benchmark grids.
#
# Re-runs the unconfined Warp solver for every truth fixture produced by
# working_tests/regenerate_unconfined_2d_truth.py and checks the heads agree
# with the stored MF6 reference. Each fixture freezes the exact solver
# settings used by run_2d_unconfined_warp_vs_mf6.py (captured from its
# solve2_settings at generation time), so "the same solver settings" are
# guaranteed without this test importing the runner. No MF6 binary is needed
# at test time. Tests skip per-grid when a fixture is absent (e.g. the large
# grids are gitignored on a fresh checkout).
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
    heads_warp = np.asarray(heads_warp, dtype=np.float64)
    heads_truth = np.asarray(heads_truth, dtype=np.float64)
    mask = (np.asarray(active) != 0) & np.isfinite(heads_truth) & np.isfinite(heads_warp)
    diff = (heads_warp - heads_truth)[mask]
    abs_diff = np.abs(diff)
    return {
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan"),
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
    """Fail loudly (not skip) if the fixture dir is missing/empty, which means
    the regen step was not run."""
    if not _warp_available():
        pytest.skip("warp is not available")
    if not _TRUTH_FIXTURE_DIR.exists():
        pytest.fail(
            f"Truth fixture dir not found: {_TRUTH_FIXTURE_DIR}. Run "
            f"`python working_tests/regenerate_unconfined_2d_truth.py`."
        )
    assert _TRUTH_FIXTURES, f"No truth fixtures found in {_TRUTH_FIXTURE_DIR}"
