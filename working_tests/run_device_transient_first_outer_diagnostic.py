#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare host and device first-outer transient unconfined linearizations."""

from __future__ import annotations

from inspect import signature
from pathlib import Path
import sys

import numpy as np
import warp as wp


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from DARCY_WARP_PACKAGE.warped_darcy import (  # noqa: E402
    WP_FLOAT,
    WarpDarcySolver,
    build_diag_preconditioner,
    build_transient_rhs_from_storage_kernel,
    copy_field_kernel,
    update_secant_sy_storage_kernel,
    update_unconfined_transmissivity_from_head_kernel,
    zero_scalar_kernel,
)
from working_tests.transient_artifacts import (  # noqa: E402
    FORMULATION_UNCONFINED,
    WARM_START_UNCONFINED_STEADY_MF6,
    default_artifact_path,
    load_transient_artifact,
    select_artifact_warm_start,
    spatial_fields_from_artifact,
)
from working_tests.transient_replay_settings import default_solve_controls  # noqa: E402
from working_tests.transient_replay_storage import compute_unconfined_storage_components  # noqa: E402


def _auto_device(requested_device: str) -> str:
    """
    Pick a Warp device for diagnostics.

    :param requested_device: ``auto``, ``cpu``, or an explicit Warp device name.
    :return: Device string usable by Warp.
    """
    mode = str(requested_device).strip().lower()
    if mode != "auto":
        return requested_device
    try:
        if wp.get_cuda_device_count() > 0:
            return "cuda:0"
    except Exception:
        return "cpu"
    return "cpu"


def _field_stats(name: str, array: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """
    Summarize a field.

    :param name: Field name.
    :param array: Field array.
    :param mask: Optional mask.
    :return: Summary dictionary.
    """
    arr = np.asarray(array, dtype=np.float64)
    vals = arr[np.asarray(mask, dtype=bool)] if mask is not None else arr.reshape(-1)
    if vals.size == 0:
        return {"name": name, "min": None, "max": None, "mean": None}
    return {
        "name": name,
        "min": float(np.nanmin(vals)),
        "max": float(np.nanmax(vals)),
        "mean": float(np.nanmean(vals)),
    }


def _diff_stats(name: str, reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """
    Compare two fields.

    :param name: Field name.
    :param reference: Reference array.
    :param candidate: Candidate array.
    :param mask: Optional comparison mask.
    :return: Difference summary.
    """
    diff = np.asarray(candidate, dtype=np.float64) - np.asarray(reference, dtype=np.float64)
    vals = diff[np.asarray(mask, dtype=bool)] if mask is not None else diff.reshape(-1)
    if vals.size == 0:
        return {"name": name, "max_abs": None, "rms": None, "mean": None}
    return {
        "name": name,
        "max_abs": float(np.nanmax(np.abs(vals))),
        "rms": float(np.sqrt(np.nanmean(vals * vals))),
        "mean": float(np.nanmean(vals)),
    }


def _print_stats_table(title: str, rows: list[dict]) -> None:
    """
    Print compact metric rows.

    :param title: Table title.
    :param rows: Metric rows.
    """
    print(f"\n{title}")
    for row in rows:
        parts = [str(row.get("name", ""))]
        for key in ("min", "max", "mean", "max_abs", "rms"):
            if key in row:
                value = row[key]
                text = "not_available" if value is None else f"{float(value):.6g}"
                parts.append(f"{key}={text}")
        print("  " + " ".join(parts))


def _load_problem_from_artifact(artifact_path: Path) -> dict:
    """
    Load the first-period production replay problem from an artifact.

    :param artifact_path: MF6 replay artifact.
    :return: Problem dictionary.
    """
    artifact = load_transient_artifact(path=artifact_path)
    spatial = spatial_fields_from_artifact(artifact=artifact)
    warm_start_head, _warm_start_used = select_artifact_warm_start(
        artifact=artifact,
        spatial=spatial,
        warm_start_mode=WARM_START_UNCONFINED_STEADY_MF6,
    )
    return {
        "nx": int(spatial["nx"]),
        "ny": int(spatial["ny"]),
        "dx": float(spatial["dx"]),
        "active": np.asarray(spatial["active"], dtype=np.int32),
        "bc_mask": np.asarray(spatial["bc_mask"], dtype=np.int32),
        "bc_values": np.asarray(spatial["bc_values"], dtype=np.float64),
        "top": np.asarray(spatial["top"], dtype=np.float64),
        "bottom": np.asarray(spatial["bottom"], dtype=np.float64),
        "k": np.asarray(spatial["k"], dtype=np.float64),
        "head_prev": np.asarray(warm_start_head, dtype=np.float64),
        "recharge": float(np.asarray(artifact["recharge_rates"], dtype=np.float64).reshape(-1)[0]),
        "sy": float(np.asarray(artifact["sy"], dtype=np.float64).reshape(())),
        "ss": float(np.asarray(artifact["ss"], dtype=np.float64).reshape(())),
        "dt": float(np.asarray(artifact["dt_days"], dtype=np.float64).reshape(())),
    }


def _build_synthetic_problem() -> dict:
    """
    Build a deterministic small first-outer problem.

    :return: Problem dictionary.
    """
    ny, nx = 8, 10
    dx = 50.0
    active = np.ones((ny, nx), dtype=np.int32)
    bc_mask = np.zeros((ny, nx), dtype=np.int32)
    bc_mask[:, 0] = 1
    bc_mask[:, -1] = 1
    bc_values = np.zeros((ny, nx), dtype=np.float64)
    bc_values[:, 0] = 112.0
    bc_values[:, -1] = 108.0
    bottom = np.full((ny, nx), 10.0, dtype=np.float64)
    top = np.full((ny, nx), 110.0, dtype=np.float64)
    k = np.full((ny, nx), 30.0, dtype=np.float64)
    head_prev = np.full((ny, nx), 109.5, dtype=np.float64)
    head_prev[:, 0] = bc_values[:, 0]
    head_prev[:, -1] = bc_values[:, -1]
    return {
        "nx": nx,
        "ny": ny,
        "dx": dx,
        "active": active,
        "bc_mask": bc_mask,
        "bc_values": bc_values,
        "top": top,
        "bottom": bottom,
        "k": k,
        "head_prev": head_prev,
        "recharge": 2.5e-4,
        "sy": 0.2,
        "ss": 1.0e-5,
        "dt": 7.0,
    }


def _assemble_host_linearization(problem: dict, *, min_sat: float) -> dict:
    """
    Assemble first-outer host reference fields.

    :param problem: Problem dictionary.
    :param min_sat: Minimum saturated thickness.
    :return: Host fields.
    """
    head_ref = np.asarray(problem["head_prev"], dtype=np.float64)
    bottom = np.asarray(problem["bottom"], dtype=np.float64)
    top = np.asarray(problem["top"], dtype=np.float64)
    k = np.asarray(problem["k"], dtype=np.float64)
    active = np.asarray(problem["active"], dtype=np.int32)
    bc_mask = np.asarray(problem["bc_mask"], dtype=np.int32)
    bc_values = np.asarray(problem["bc_values"], dtype=np.float64)
    full = np.maximum(top - bottom, float(min_sat))
    sat = np.clip(head_ref - bottom, float(min_sat), full)
    transmissivity = (k * sat).astype(np.float64)
    transmissivity[active == 0] = 0.0
    components = compute_unconfined_storage_components(
        sy=float(problem["sy"]),
        ss=float(problem["ss"]),
        head_ref=head_ref,
        head_old=np.asarray(problem["head_prev"], dtype=np.float64),
        bottom=bottom,
        top=top,
        active=active,
        bc_mask=bc_mask,
        min_sat=float(min_sat),
        storage_mode="mf6_convertible_secant_sy",
        secant_eps=1.0e-12,
    )
    storage_coeff = np.asarray(components["storage_coeff"], dtype=np.float64)
    storage_diag = storage_coeff * float(problem["dx"]) * float(problem["dx"]) / float(problem["dt"])
    rhs = float(problem["recharge"]) * float(problem["dx"]) * float(problem["dx"]) + storage_diag * head_ref
    rhs[active == 0] = 0.0
    rhs[bc_mask != 0] = bc_values[bc_mask != 0]
    m_inv = build_diag_preconditioner(
        T_field=transmissivity,
        active=active,
        bc_mask=bc_mask,
        storage_diag=storage_diag,
    ).astype(np.float64)
    return {
        "T": transmissivity,
        "storage_diag": storage_diag,
        "storage_coeff": storage_coeff,
        "rhs_eff": rhs,
        "M_inv": m_inv,
    }


def _solver_for_problem(problem: dict, *, device: str, diag_backend: str = "device") -> WarpDarcySolver:
    """
    Create a solver initialized with first-outer transmissivity.

    :param problem: Problem dictionary.
    :param device: Warp device string.
    :param diag_backend: Diagonal preconditioner backend.
    :return: Initialized solver.
    """
    fields = _assemble_host_linearization(problem=problem, min_sat=0.1)
    solver = WarpDarcySolver(
        nx=int(problem["nx"]),
        ny=int(problem["ny"]),
        dx=float(problem["dx"]),
        device=device,
        solver_type="kcycle",
        use_ghb=False,
        diag_preconditioner_backend=diag_backend,
    )
    solver.build_from_fields(
        T_field=fields["T"],
        R_field=np.full((int(problem["ny"]), int(problem["nx"])), float(problem["recharge"]), dtype=np.float64),
        active=np.asarray(problem["active"], dtype=np.int32),
        bc_mask=np.asarray(problem["bc_mask"], dtype=np.int32),
        bc_values=np.asarray(problem["bc_values"], dtype=np.float64),
    )
    return solver


def _accepted_kwargs(function, controls: dict) -> dict:
    """
    Keep only controls accepted by a callable.

    :param function: Callable.
    :param controls: Candidate controls.
    :return: Accepted keyword dictionary.
    """
    params = signature(function).parameters
    return {key: value for key, value in controls.items() if key in params}


def _host_linear_candidate(problem: dict, *, controls: dict, max_levels: int) -> dict:
    """
    Compute the host first-outer linear candidate.

    :param problem: Problem dictionary.
    :param controls: Solve controls.
    :param max_levels: Multigrid levels.
    :return: Candidate and diagnostics.
    """
    fields = _assemble_host_linearization(problem=problem, min_sat=float(controls.get("min_saturated_thickness", 0.1)))
    solver = _solver_for_problem(problem=problem, device="cpu", diag_backend="device")
    solve_kwargs = _accepted_kwargs(solver.solve_multigrid_kcycle, controls)
    solve_kwargs["max_levels"] = int(max_levels)
    solve_kwargs["return_info"] = True
    solve_kwargs["initial_head"] = np.asarray(problem["head_prev"], dtype=np.float64)
    with solver:
        candidate, info = solver.solve_multigrid_kcycle(
            transient=True,
            storage_coeff=fields["storage_coeff"],
            dt=float(problem["dt"]),
            head_prev=np.asarray(problem["head_prev"], dtype=np.float64),
            refresh_diag_with_transient_storage=True,
            **solve_kwargs,
        )
    return {"candidate": np.asarray(candidate, dtype=np.float64), "info": info, "fields": fields}


def _device_linear_candidate(
    problem: dict,
    *,
    controls: dict,
    device: str,
    max_levels: int,
    checked_inner: bool,
) -> dict:
    """
    Compute the device first-outer linear candidate.

    :param problem: Problem dictionary.
    :param controls: Solve controls.
    :param device: Warp device string.
    :param max_levels: Multigrid levels.
    :param checked_inner: Whether to request per-cycle scalar diagnostics.
    :return: Candidate and diagnostics.
    """
    fields = _assemble_host_linearization(problem=problem, min_sat=float(controls.get("min_saturated_thickness", 0.1)))
    solver = _solver_for_problem(problem=problem, device=device, diag_backend="device")
    with solver:
        solver._storage_active = True
        solver.storage_diag_wp = wp.zeros((int(problem["ny"]), int(problem["nx"])), dtype=WP_FLOAT, device=device)
        solver.storage_diag_host = np.zeros((int(problem["ny"]), int(problem["nx"])), dtype=np.float32)
        solver.build_hierarchy(
            max_levels=int(max_levels),
            min_coarse_n=4,
            min_coarse_cells=controls.get("min_coarse_cells", 4),
        )
        head_prev_wp = wp.array(np.asarray(problem["head_prev"], dtype=np.float32), dtype=WP_FLOAT, device=device)
        head_iter_wp = wp.array(np.asarray(problem["head_prev"], dtype=np.float32), dtype=WP_FLOAT, device=device)
        bottom_wp = wp.array(np.asarray(problem["bottom"], dtype=np.float32), dtype=WP_FLOAT, device=device)
        top_wp = wp.array(np.asarray(problem["top"], dtype=np.float32), dtype=WP_FLOAT, device=device)
        k_wp = wp.array(np.asarray(problem["k"], dtype=np.float32), dtype=WP_FLOAT, device=device)
        storage_coeff_wp = wp.zeros((int(problem["ny"]), int(problem["nx"])), dtype=WP_FLOAT, device=device)
        sy_coeff_wp = wp.zeros((int(problem["ny"]), int(problem["nx"])), dtype=WP_FLOAT, device=device)
        ss_coeff_wp = wp.zeros((int(problem["ny"]), int(problem["nx"])), dtype=WP_FLOAT, device=device)
        storage_diag_wp = wp.zeros((int(problem["ny"]), int(problem["nx"])), dtype=WP_FLOAT, device=device)
        storage_diag_prev_wp = wp.zeros((int(problem["ny"]), int(problem["nx"])), dtype=WP_FLOAT, device=device)
        storage_change_sum_sq = wp.zeros(1, dtype=wp.float64, device=device)
        storage_change_max = wp.zeros(1, dtype=wp.float64, device=device)
        rhs_eff_wp = wp.zeros((int(problem["ny"]), int(problem["nx"])), dtype=WP_FLOAT, device=device)
        wp.launch(
            kernel=update_unconfined_transmissivity_from_head_kernel,
            dim=(int(problem["ny"]), int(problem["nx"])),
            inputs=[
                head_iter_wp,
                k_wp,
                bottom_wp,
                top_wp,
                solver.active_wp,
                float(controls.get("min_saturated_thickness", 0.1)),
                int(problem["nx"]),
                int(problem["ny"]),
                solver.T_wp,
            ],
            device=device,
        )
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_sum_sq], device=device)
        wp.launch(kernel=zero_scalar_kernel, dim=1, inputs=[storage_change_max], device=device)
        wp.launch(
            kernel=update_secant_sy_storage_kernel,
            dim=(int(problem["ny"]), int(problem["nx"])),
            inputs=[
                head_iter_wp,
                head_prev_wp,
                bottom_wp,
                top_wp,
                solver.active_wp,
                solver.bc_mask_wp,
                float(problem["sy"]),
                float(problem["ss"]),
                float(problem["dx"]),
                float(problem["dt"]),
                float(controls.get("min_saturated_thickness", 0.1)),
                1.0e-12,
                int(problem["nx"]),
                int(problem["ny"]),
                storage_coeff_wp,
                sy_coeff_wp,
                ss_coeff_wp,
                storage_diag_wp,
                storage_diag_prev_wp,
                storage_change_sum_sq,
                storage_change_max,
            ],
            device=device,
        )
        solver._update_diag_preconditioner_device(
            T_wp=solver.T_wp,
            active_wp=solver.active_wp,
            bc_mask_wp=solver.bc_mask_wp,
            gh_mask_wp=solver.mg_levels[0].gh_mask_wp,
            ghb_factor_wp=solver.mg_levels[0].ghb_factor_wp,
            M_inv_wp=solver.mg_levels[0].M_inv_wp,
            nx=int(problem["nx"]),
            ny=int(problem["ny"]),
            use_ghb=False,
            storage_diag_wp=storage_diag_wp,
        )
        solver.mg_levels[0].T_wp = solver.T_wp
        solver.mg_levels[0].storage_diag_wp = storage_diag_wp
        solver._refresh_transient_device_hierarchy_values(levels=solver.mg_levels)
        wp.launch(
            kernel=build_transient_rhs_from_storage_kernel,
            dim=(int(problem["ny"]), int(problem["nx"])),
            inputs=[
                solver.R_wp,
                storage_diag_wp,
                head_prev_wp,
                solver.active_wp,
                solver.bc_mask_wp,
                solver.bc_values_wp,
                float(problem["dx"]),
                int(problem["nx"]),
                int(problem["ny"]),
                rhs_eff_wp,
            ],
            device=device,
        )
        info = solver._solve_multigrid_kcycle_device_buffers(
            x_wp=head_iter_wp,
            rhs_wp=rhs_eff_wp,
            T_wp=solver.T_wp,
            storage_diag_wp=storage_diag_wp,
            active_wp=solver.active_wp,
            bc_mask_wp=solver.bc_mask_wp,
            bc_values_wp=solver.bc_values_wp,
            levels=solver.mg_levels,
            solve_controls=dict(controls),
            return_scalar_info=checked_inner,
        )
        return {
            "candidate": np.asarray(head_iter_wp.numpy(), dtype=np.float64),
            "info": info,
            "fields": {
                "T": np.asarray(solver.T_wp.numpy(), dtype=np.float64),
                "storage_diag": np.asarray(storage_diag_wp.numpy(), dtype=np.float64),
                "rhs_eff": np.asarray(rhs_eff_wp.numpy(), dtype=np.float64),
                "M_inv": np.asarray(solver.mg_levels[0].M_inv_wp.numpy(), dtype=np.float64),
            },
        }


def _picard_after_clip(
    *,
    previous: np.ndarray,
    candidate: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
    bc_values: np.ndarray,
    omega: float,
    max_update: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Apply the production Picard relaxation and cap on host for diagnostics.

    :param previous: Previous Picard head.
    :param candidate: Linear candidate head.
    :param active: Active mask.
    :param bc_mask: Dirichlet mask.
    :param bc_values: Dirichlet values.
    :param omega: Relaxation factor.
    :param max_update: Maximum absolute update.
    :return: Accepted head, raw update, clipped fraction.
    """
    raw_update = float(omega) * (np.asarray(candidate, dtype=np.float64) - np.asarray(previous, dtype=np.float64))
    clipped = np.clip(raw_update, -float(max_update), float(max_update))
    accepted = np.asarray(previous, dtype=np.float64) + clipped
    accepted[np.asarray(active, dtype=np.int32) == 0] = 0.0
    accepted[np.asarray(bc_mask, dtype=np.int32) != 0] = np.asarray(bc_values, dtype=np.float64)[
        np.asarray(bc_mask, dtype=np.int32) != 0
    ]
    free = (np.asarray(active, dtype=np.int32) != 0) & (np.asarray(bc_mask, dtype=np.int32) == 0)
    clipped_fraction = float(np.count_nonzero(clipped[free] != raw_update[free]) / max(np.count_nonzero(free), 1))
    return accepted, raw_update, clipped_fraction


def run_first_outer_diagnostic(
    *,
    use_artifact: bool,
    artifact_path: Path,
    device: str,
    max_levels_values: tuple[int, ...],
    checked_inner: bool,
) -> None:
    """
    Run first-outer host/device diagnostics.

    :param use_artifact: Use the production artifact when true; otherwise use a synthetic tiny case.
    :param artifact_path: Artifact path.
    :param device: Device string or ``auto``.
    :param max_levels_values: Hierarchy depths to test.
    :param checked_inner: Use checked K-cycle diagnostics in device candidate.
    """
    wp.init()
    device_name = _auto_device(requested_device=device)
    problem = _load_problem_from_artifact(artifact_path=artifact_path) if use_artifact else _build_synthetic_problem()
    controls = default_solve_controls()
    controls["use_device_transient_fast_path"] = True
    controls["max_cycles"] = int(controls.get("unconfined_inner_max_cycles_early", 2))
    controls["check_every_no"] = 1
    controls["coarse_operator_mode"] = "device_refreshed_dynamic_coarse_operator"
    controls["min_coarse_cells"] = 4 if not use_artifact else controls.get("min_coarse_cells", 500)

    active = np.asarray(problem["active"], dtype=np.int32)
    bc_mask = np.asarray(problem["bc_mask"], dtype=np.int32)
    free = (active != 0) & (bc_mask == 0)
    omega = float(controls.get("omega", 0.7))
    omega = min(max(omega, float(controls.get("omega_min", 0.1))), float(controls.get("omega_max", 0.9)))
    max_update = float(controls.get("max_head_change_per_outer_iteration", 10.0))

    print("Device transient first-outer diagnostic")
    print(f"  problem={'artifact' if use_artifact else 'synthetic'}")
    print(f"  device={device_name}")
    print(f"  checked_inner={checked_inner}")
    print(f"  grid={int(problem['ny'])}x{int(problem['nx'])}")

    for max_levels in max_levels_values:
        host = _host_linear_candidate(
            problem=problem,
            controls=controls,
            max_levels=int(max_levels),
        )
        device_result = _device_linear_candidate(
            problem=problem,
            controls=controls,
            device=device_name,
            max_levels=int(max_levels),
            checked_inner=checked_inner,
        )
        host_accepted, host_update, host_clip_fraction = _picard_after_clip(
            previous=np.asarray(problem["head_prev"], dtype=np.float64),
            candidate=host["candidate"],
            active=active,
            bc_mask=bc_mask,
            bc_values=np.asarray(problem["bc_values"], dtype=np.float64),
            omega=omega,
            max_update=max_update,
        )
        device_accepted, device_update, device_clip_fraction = _picard_after_clip(
            previous=np.asarray(problem["head_prev"], dtype=np.float64),
            candidate=device_result["candidate"],
            active=active,
            bc_mask=bc_mask,
            bc_values=np.asarray(problem["bc_values"], dtype=np.float64),
            omega=omega,
            max_update=max_update,
        )
        print(f"\nmax_levels={max_levels}")
        _print_stats_table(
            title="host fields",
            rows=[
                _field_stats("T", host["fields"]["T"], free),
                _field_stats("storage_diag", host["fields"]["storage_diag"], free),
                _field_stats("rhs_eff", host["fields"]["rhs_eff"], free),
                _field_stats("M_inv", host["fields"]["M_inv"], free),
                _field_stats("linear_candidate_head", host["candidate"], free),
                _field_stats("raw_linear_update", host_update, free),
                _field_stats("Picard_candidate_after_relax_clip", host_accepted, free),
            ],
        )
        _print_stats_table(
            title="device fields",
            rows=[
                _field_stats("T", device_result["fields"]["T"], free),
                _field_stats("storage_diag", device_result["fields"]["storage_diag"], free),
                _field_stats("rhs_eff", device_result["fields"]["rhs_eff"], free),
                _field_stats("M_inv", device_result["fields"]["M_inv"], free),
                _field_stats("linear_candidate_head", device_result["candidate"], free),
                _field_stats("raw_linear_update", device_update, free),
                _field_stats("Picard_candidate_after_relax_clip", device_accepted, free),
            ],
        )
        _print_stats_table(
            title="host-device differences",
            rows=[
                _diff_stats("T", host["fields"]["T"], device_result["fields"]["T"], free),
                _diff_stats("storage_diag", host["fields"]["storage_diag"], device_result["fields"]["storage_diag"], free),
                _diff_stats("rhs_eff", host["fields"]["rhs_eff"], device_result["fields"]["rhs_eff"], free),
                _diff_stats("M_inv", host["fields"]["M_inv"], device_result["fields"]["M_inv"], free),
                _diff_stats("linear_candidate_head", host["candidate"], device_result["candidate"], free),
                _diff_stats("raw_linear_update", host_update, device_update, free),
                _diff_stats("Picard_candidate_after_relax_clip", host_accepted, device_accepted, free),
            ],
        )
        print(f"  host_clipped_update_fraction={host_clip_fraction:.6g}")
        print(f"  device_clipped_update_fraction={device_clip_fraction:.6g}")
        print(f"  host_info={host['info']}")
        print(f"  device_info={device_result['info']}")


if __name__ == "__main__":
    use_artifact = False
    artifact_path = default_artifact_path(formulation=FORMULATION_UNCONFINED)
    device = "auto"
    max_levels_values = (1, 6)
    checked_inner = True

    run_first_outer_diagnostic(
        use_artifact=use_artifact,
        artifact_path=artifact_path,
        device=device,
        max_levels_values=max_levels_values,
        checked_inner=checked_inner,
    )
