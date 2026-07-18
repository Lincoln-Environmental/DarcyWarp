"""Extreme-failure synthetic for the adaptive-dt failure path.

Builds a case deliberately harder than the production benchmark: a low-K
barrier strip (100x contrast) with a narrow throat, plus recharge sign swings.

NOTE (2026-07-19): this case is currently NON-CONVERGENT — dh_max stays pinned
at the 10 m per-iteration update clip at every dt down to dt_min, so every
period ends unaccepted. Its value is exercising the *hopeless sub-step* path:
early shrink reproduces the shrink sequence at ~28% fewer outer iterations
than paying the full strict budget (144 vs 200 per period), and the driver
degrades gracefully (practical fallback at dt_min, no retry storm). It is a
machinery fixture, not a physics validation — do not "fix" the case to make
it converge; use the production replay for accuracy validation.

Compares:
  A) new economics  (early shrink + budget extension ON)
  B) old economics  (both OFF), same strict budget
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_case(nx=200, ny=200, dx=1.0):
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

    solver = WarpDarcySolver(
        nx=nx, ny=ny, dx=dx, device="cuda:0",
        use_ghb=False, solver_type="kcycle", diag_preconditioner_backend="device",
    )
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values[:, 0] = 110.0
    bc_values[:, -1] = 90.0
    # Realistic calibration matching the production benchmark (day units):
    # 300 m thick aquifer, K background 100 m/day with a 100x low-K barrier.
    zbot = np.full((ny, nx), -150.0, dtype=np.float64)
    ztop = np.full((ny, nx), 150.0, dtype=np.float64)
    k = np.full((ny, nx), 100.0, dtype=np.float64)
    # Low-K barrier strip across most of the domain (100x contrast).
    k[:, nx // 2 - 3 : nx // 2 + 3] = 1.0
    # Gap in the barrier -> forced channelization through a narrow throat.
    k[ny // 2 - 10 : ny // 2 + 10, nx // 2 - 3 : nx // 2 + 3] = 100.0
    thickness0 = 250.0
    solver.build_from_fields(
        T_field=k * thickness0,
        R_field=np.zeros((ny, nx), dtype=np.float64),
        active=active, bc_mask=bc_mask, bc_values=bc_values,
    )
    h0 = np.full((ny, nx), 100.0, dtype=np.float64)
    h0[:, 0] = 110.0
    h0[:, -1] = 90.0
    return solver, k, zbot, ztop, h0


def run(config_name: str, overrides: dict, n_periods=3):
    from working_tests.transient_replay_settings import default_solve_controls

    solver, k, zbot, ztop, h0 = build_case()
    # The fast path applies one uniform rate per period, so stress the case
    # with rate swings (rise / discharge / rise) through the K-barrier
    # channelization — stiffer than the homogeneous benchmark in both
    # directions of the transient.
    rates = np.array([3.0e-4, -3.0e-4, 3.0e-4][:n_periods], dtype=np.float64)
    sc = default_solve_controls()
    sc["use_device_transient_fast_path"] = True
    sc["allow_unaccepted_transient_period"] = True
    sc.update(overrides)
    t0 = time.perf_counter()
    heads, info = solver.solve_transient_2d_unconfined(
        initial_head=h0,
        recharge_rates=rates,
        k_field=k, zbot_field=zbot, ztop_field=ztop,
        sy=0.1, ss=1.0e-5, dt=7.0,
        storage_mode="mf6_convertible_secant_sy",
        storage_reference="current_picard",
        solve_controls=sc,
        return_info=True,
    )
    wall = time.perf_counter() - t0
    print(f"\n=== {config_name} (wall {wall:.1f}s) ===")
    print(f"{'per':>3} {'subs':>5} {'retry':>6} {'eshk':>5} {'ext':>4} {'pfb':>4} {'outer':>6} {'strict':>7} {'dh_max':>9}")
    for i, p in enumerate(info.get("period_infos") or [], 1):
        print(
            f"{i:>3} {p.get('adaptive_dt_substep_count', 0):>5} "
            f"{p.get('adaptive_dt_retry_count', 0):>6} "
            f"{p.get('adaptive_dt_early_shrink_count', 0):>5} "
            f"{p.get('adaptive_dt_extension_count', 0):>4} "
            f"{p.get('adaptive_dt_practical_fallback_count', 0):>4} "
            f"{p.get('adaptive_dt_total_outer_iterations', p.get('outer_iterations', 0)):>6} "
            f"{str(p.get('strict_picard_convergence_passed')):>7} "
            f"{p.get('final_max_abs_head_change', float('nan')):9.2e}"
        )
    assert np.all(np.isfinite(heads)), f"{config_name}: non-finite heads"
    return info


def main():
    run("A: new economics (early+extension)", {})
    run("B: old economics (both off)", {
        "adaptive_dt_early_shrink_enabled": False,
        "adaptive_dt_extension_enabled": False,
    })


if __name__ == "__main__":
    main()
