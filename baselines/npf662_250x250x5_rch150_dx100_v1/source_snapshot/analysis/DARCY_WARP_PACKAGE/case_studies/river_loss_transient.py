# SPDX-License-Identifier: AGPL-3.0-only
"""Transient saturated-Darcy envelope for the OGS river cross-section case.

The OGS benchmark uses ``RICHARDS_FLOW`` and obtains transient storage and
conductivity changes from its unsaturated constitutive laws.  This module does
not reproduce those laws.  It applies the same time-dependent heads to the
saturated Darcy operator so final/late-time differences can be attributed to
model structure.  With zero specific storage (the OGS materials' saturated
storage setting), each requested time is a quasi-steady Darcy state.  Positive
specific-storage values optionally exercise DarcyWarp's confined transient
operator as a separate sensitivity experiment.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section import (
    AnisotropyTarget,
    CrossSectionModel,
    _free_active_source_sink,
    _robust_solver_kwargs,
    _solver_kwargs,
)
from DARCY_WARP_PACKAGE.physics.budget_3d import (
    BoundaryInterfaceFluxPlan,
    prepare_boundary_interface_flux,
)
from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k


@dataclass(frozen=True)
class TransientStepResult:
    """One reported state from a saturated river-boundary sequence."""

    time_days: float
    dt_from_previous_output_days: float
    channel_inflow: float
    outlet_outflow: float
    other_source_sink: float
    storage_rate: float
    budget_residual: float
    relative_mass_imbalance: float
    cumulative_channel_leakage: float
    storage_change_from_initial: float
    solved: bool
    converged: bool
    cycles: int
    residual_rms: float
    channel_ramp_fraction: float


@dataclass(frozen=True)
class OGSComparisonResult:
    """DarcyWarp-to-OGS diagnostics at one DarcyWarp output time."""

    time_days: float
    compared_cells: int
    total_head_rmse: float
    total_head_mean_bias: float
    total_head_max_abs: float
    darcywarp_channel_inflow: float
    ogs_channel_inflow: float
    channel_inflow_relative_difference: float
    darcywarp_cumulative_channel_leakage: float
    ogs_cumulative_channel_leakage: float
    cumulative_leakage_relative_difference: float


def validate_output_times(output_times_days: Iterable[float]) -> np.ndarray:
    """Return finite, strictly increasing output times beginning at zero."""

    times = np.asarray(tuple(output_times_days), dtype=np.float64)
    if times.ndim != 1 or times.size < 2:
        raise ValueError("output_times_days must contain at least two values.")
    if not np.all(np.isfinite(times)) or np.any(times < 0.0):
        raise ValueError("output_times_days must be finite and non-negative.")
    if not np.isclose(times[0], 0.0, rtol=0.0, atol=1.0e-14):
        raise ValueError("output_times_days must begin at 0.")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("output_times_days must be strictly increasing.")
    return times


def initial_linear_total_head(model: CrossSectionModel) -> np.ndarray:
    """Construct the OGS linear-gradient initial total-head field."""

    cfg = model.config
    x_fraction = np.clip(model.x / cfg.domain_length, 0.0, 1.0)
    head_x = cfg.channel_head + (cfg.discharge_head - cfg.channel_head) * x_fraction
    head = np.broadcast_to(head_x[None, None, :], model.active.shape).copy()
    head[model.outlet_mask] = cfg.discharge_head
    head[model.active == 0] = 0.0
    return head


def boundary_values_at_time(
    model: CrossSectionModel,
    time_days: float,
    channel_ramp_days: float,
) -> tuple[np.ndarray, float]:
    """Return OGS-style channel-ramp and fixed far-field Dirichlet values."""

    if not np.isfinite(time_days) or time_days < 0.0:
        raise ValueError("time_days must be finite and >= 0.")
    if not np.isfinite(channel_ramp_days) or channel_ramp_days < 0.0:
        raise ValueError("channel_ramp_days must be finite and >= 0.")
    initial = initial_linear_total_head(model=model)
    if channel_ramp_days == 0.0:
        fraction = 1.0
    else:
        fraction = float(np.clip(time_days / channel_ramp_days, 0.0, 1.0))
    values = model.bc_values.copy()
    values[model.channel_mask] = (
        initial[model.channel_mask]
        + fraction * (model.config.channel_head - initial[model.channel_mask])
    )
    values[model.outlet_mask] = model.config.discharge_head
    return values, fraction


def build_specific_storage_field(
    model: CrossSectionModel,
    braidplain_specific_storage: float,
    regional_specific_storage: float,
) -> np.ndarray:
    """Map material specific storage [1/m] onto free aquifer cells."""

    for name, value in (
        ("braidplain_specific_storage", braidplain_specific_storage),
        ("regional_specific_storage", regional_specific_storage),
    ):
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and >= 0.")
    cfg = model.config
    xx = np.broadcast_to(model.x[None, None, :], model.active.shape)
    zz = np.broadcast_to(model.z[:, None, None], model.active.shape)
    free = (model.active != 0) & (model.bc_mask == 0)
    braidplain = (
        free
        & (xx < cfg.braidplain_half_width)
        & (zz >= cfg.braidplain_base)
    )
    storage = np.zeros(model.active.shape, dtype=np.float64)
    storage[free] = float(regional_specific_storage)
    storage[braidplain] = float(braidplain_specific_storage)
    return storage


def _boundary_budget(
    head: np.ndarray,
    channel_plan: BoundaryInterfaceFluxPlan,
    outlet_plan: BoundaryInterfaceFluxPlan,
    other_source_sink: float,
) -> tuple[float, float, float]:
    channel = channel_plan.evaluate(head=head)
    outlet = -outlet_plan.evaluate(head=head)
    return float(channel), float(outlet), float(other_source_sink)


def _relative_budget_error(
    channel: float,
    outlet: float,
    other: float,
    storage_rate: float,
) -> tuple[float, float]:
    residual = channel - outlet + other - storage_rate
    scale = max(abs(channel), abs(outlet), abs(other), abs(storage_rate))
    if scale == 0.0:
        relative = 0.0 if residual == 0.0 else float("inf")
    else:
        relative = abs(residual) / scale
    return float(residual), float(relative)


def _solve_state(
    *,
    solver: Any,
    model: CrossSectionModel,
    initial_head: np.ndarray,
    head_prev: np.ndarray,
    storage: np.ndarray,
    storage_volume_coeff: np.ndarray,
    channel_plan: BoundaryInterfaceFluxPlan,
    outlet_plan: BoundaryInterfaceFluxPlan,
    other_source_sink: float,
    dt_days: float,
    quasi_steady: bool,
    transient_abs_tol_min: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    cfg = model.config
    simple = _solver_kwargs(cfg=cfg)
    transient_kwargs: dict[str, Any] = {}
    if not quasi_steady:
        transient_kwargs = {
            "transient": True,
            "storage_coeff": storage,
            "dt": float(dt_days),
            "head_prev": head_prev,
        }

    def evaluate(candidate_head: np.ndarray, candidate_info: dict[str, Any]) -> bool:
        candidate = np.asarray(candidate_head, dtype=np.float64)
        channel, outlet, other = _boundary_budget(
            head=candidate,
            channel_plan=channel_plan,
            outlet_plan=outlet_plan,
            other_source_sink=other_source_sink,
        )
        storage_rate = (
            0.0
            if quasi_steady
            else float(np.sum(storage_volume_coeff * (candidate - head_prev)) / dt_days)
        )
        residual, relative = _relative_budget_error(
            channel=channel,
            outlet=outlet,
            other=other,
            storage_rate=storage_rate,
        )
        scale = max(abs(channel), abs(outlet), abs(other), abs(storage_rate))
        candidate_info["river_channel_inflow"] = channel
        candidate_info["river_outlet_outflow"] = outlet
        candidate_info["river_other_source_sink"] = other
        candidate_info["river_storage_rate"] = storage_rate
        candidate_info["river_budget_residual"] = residual
        candidate_info["river_budget_scale"] = scale
        candidate_info["river_relative_mass_imbalance"] = relative
        return bool(
            candidate_info.get("converged", False)
            and np.isfinite(relative)
            and relative <= cfg.practical_mass_imbalance_tol
        )

    strict_first = bool(not quasi_steady and cfg.robust_retry_enabled)
    primary_controls = (
        _robust_solver_kwargs(cfg=cfg, simple_kwargs=simple)
        if strict_first
        else simple
    )
    if strict_first and transient_abs_tol_min is not None:
        if (
            not np.isfinite(transient_abs_tol_min)
            or transient_abs_tol_min <= 0.0
        ):
            raise ValueError("transient_abs_tol_min must be finite and > 0.")
        primary_controls["abs_tol_min"] = min(
            float(primary_controls["abs_tol_min"]),
            float(transient_abs_tol_min),
        )
    head, info = solver.solve(
        **primary_controls,
        **transient_kwargs,
        initial_head=initial_head,
    )
    if evaluate(candidate_head=head, candidate_info=info):
        info["river_strict_controls_used"] = strict_first
        info["river_budget_refinements"] = 0
        return np.asarray(head, dtype=np.float64), info
    if not cfg.robust_retry_enabled:
        raise RuntimeError(
            "River transient step failed convergence/mass acceptance: "
            f"relative imbalance={float(info.get('river_relative_mass_imbalance', np.nan)):.3e}, "
            f"converged={bool(info.get('converged', False))}, "
            f"residual_rms={float(info.get('r_rms_end', np.nan)):.3e}, "
            f"cycles={int(info.get('n_cycles_used', info.get('n_iter_used', -1)))}, "
            f"dt={dt_days:g} days."
        )
    if strict_first:
        # A long section can have a locally tiny RMS residual but still fail
        # its integrated mass gate. In particular, a warm start just below the
        # absolute residual floor performs zero cycles. Refine by successive
        # residual decades only when the independently evaluated budget asks
        # for it; this avoids paying for an unnecessarily strict solve on
        # every stored timestep.
        refinement_controls = dict(primary_controls)
        refined_head = np.asarray(head, dtype=np.float64)
        refined_info = info
        for refinement in range(1, 4):
            residual_rms = float(refined_info.get("r_rms_end", np.nan))
            previous_floor = float(refinement_controls["abs_tol_min"])
            if np.isfinite(residual_rms) and residual_rms > 0.0:
                refined_floor = min(previous_floor * 0.1, residual_rms * 0.1)
            else:
                refined_floor = previous_floor * 0.1
            refinement_controls["abs_tol_min"] = max(
                float(np.finfo(np.float64).tiny),
                float(refined_floor),
            )
            print(
                "Transient river budget refinement "
                f"{refinement}/3: imbalance="
                f"{float(refined_info.get('river_relative_mass_imbalance', np.nan)):.3e}, "
                f"residual target={refinement_controls['abs_tol_min']:.3e}."
            )
            refined_head, refined_info = solver.solve(
                **refinement_controls,
                **transient_kwargs,
                initial_head=refined_head,
            )
            if evaluate(candidate_head=refined_head, candidate_info=refined_info):
                refined_info["river_strict_controls_used"] = True
                refined_info["river_budget_refinements"] = refinement
                refined_info["river_recommended_abs_tol_min"] = float(
                    refinement_controls["abs_tol_min"]
                )
                return np.asarray(refined_head, dtype=np.float64), refined_info
        raise RuntimeError(
            "River transient step failed convergence/mass acceptance after "
            "three budget refinements: "
            f"relative imbalance={float(refined_info.get('river_relative_mass_imbalance', np.nan)):.3e}, "
            f"converged={bool(refined_info.get('converged', False))}, "
            f"residual_rms={float(refined_info.get('r_rms_end', np.nan)):.3e}, "
            f"cycles={int(refined_info.get('n_cycles_used', refined_info.get('n_iter_used', -1)))}, "
            f"dt={dt_days:g} days."
        )
    robust = _robust_solver_kwargs(cfg=cfg, simple_kwargs=simple)
    print(
        "Initial quasi-steady river solve failed convergence or mass-balance "
        "acceptance; retrying with robust solver settings."
    )
    retry_head, retry_info = solver.solve(
        **robust,
        **transient_kwargs,
        initial_head=np.asarray(head, dtype=np.float64),
    )
    if not evaluate(candidate_head=retry_head, candidate_info=retry_info):
        raise RuntimeError(
            "River transient step failed convergence/mass acceptance: "
            f"relative imbalance={float(retry_info.get('river_relative_mass_imbalance', np.nan)):.3e}, "
            f"converged={bool(retry_info.get('converged', False))}, "
            f"residual_rms={float(retry_info.get('r_rms_end', np.nan)):.3e}, "
            f"cycles={int(retry_info.get('n_cycles_used', retry_info.get('n_iter_used', -1)))}, "
            f"dt={dt_days:g} days."
        )
    retry_info["river_strict_controls_used"] = True
    retry_info["river_budget_refinements"] = 0
    return np.asarray(retry_head, dtype=np.float64), retry_info


def run_transient_case(
    model: CrossSectionModel,
    *,
    output_times_days: Iterable[float],
    channel_ramp_days: float,
    braidplain_specific_storage: float,
    regional_specific_storage: float,
    maximum_timestep_days: float,
    progress_every_steps: int = 0,
) -> tuple[list[TransientStepResult], np.ndarray, dict[str, Any]]:
    """Run one saturated transient or zero-storage quasi-steady sequence."""

    times = validate_output_times(output_times_days=output_times_days)
    if not np.isfinite(maximum_timestep_days) or maximum_timestep_days <= 0.0:
        raise ValueError("maximum_timestep_days must be finite and > 0.")
    if progress_every_steps < 0:
        raise ValueError("progress_every_steps must be >= 0.")
    storage = build_specific_storage_field(
        model=model,
        braidplain_specific_storage=braidplain_specific_storage,
        regional_specific_storage=regional_specific_storage,
    )
    quasi_steady = not np.any(storage > 0.0)
    faces = build_7point_face_conductance_from_k(
        kx_field=model.kx,
        ky_field=model.ky,
        kz_field=model.kz,
        active=model.active,
        dx=model.config.dx,
        dy=model.config.out_of_plane_width,
        dz=model.config.dz,
    )
    channel_plan = prepare_boundary_interface_flux(
        boundary_mask=model.channel_mask,
        active=model.active,
        tx_p=faces[0],
        tx_m=faces[1],
        ty_p=faces[2],
        ty_m=faces[3],
        tz_p=faces[4],
        tz_m=faces[5],
    )
    outlet_plan = prepare_boundary_interface_flux(
        boundary_mask=model.outlet_mask,
        active=model.active,
        tx_p=faces[0],
        tx_m=faces[1],
        ty_p=faces[2],
        ty_m=faces[3],
        tz_p=faces[4],
        tz_m=faces[5],
    )
    other_source_sink = _free_active_source_sink(
        rhs=model.rhs,
        active=model.active,
        bc_mask=model.bc_mask,
    )
    solver = model.inputs.build_solver(
        device=model.config.device,
        solver=model.config.solver,
        implementation=model.config.implementation,
        face_conductance=faces,
    )
    head = initial_linear_total_head(model=model)
    initial_head = head.copy()
    cell_volume = (
        model.config.dx * model.config.out_of_plane_width * model.config.dz
    )
    storage_volume_coeff = storage * cell_volume
    initial_storage = float(np.sum(storage_volume_coeff * initial_head))
    heads = [head.copy()]
    rows: list[TransientStepResult] = []
    cumulative_leakage = 0.0
    bc0, ramp0 = boundary_values_at_time(
        model=model,
        time_days=0.0,
        channel_ramp_days=channel_ramp_days,
    )
    solver.update_boundary_values(bc0)
    channel, outlet, other = _boundary_budget(
        head=head,
        channel_plan=channel_plan,
        outlet_plan=outlet_plan,
        other_source_sink=other_source_sink,
    )
    residual, relative = _relative_budget_error(
        channel=channel,
        outlet=outlet,
        other=other,
        storage_rate=0.0,
    )
    rows.append(
        TransientStepResult(
            time_days=0.0,
            dt_from_previous_output_days=0.0,
            channel_inflow=channel,
            outlet_outflow=outlet,
            other_source_sink=other,
            storage_rate=0.0,
            budget_residual=residual,
            relative_mass_imbalance=relative,
            cumulative_channel_leakage=0.0,
            storage_change_from_initial=0.0,
            solved=False,
            converged=False,
            cycles=0,
            residual_rms=float("nan"),
            channel_ramp_fraction=ramp0,
        )
    )

    current_time = 0.0
    last_output_time = 0.0
    internal_steps = 0
    total_budget_refinements = 0
    transient_abs_tol_min: float | None = None
    current_boundary_values = bc0.copy()
    state_is_solved = False
    try:
        for target_time in times[1:]:
            last_info: dict[str, Any] = {}
            last_previous = head.copy()
            last_dt_days = float("nan")
            while current_time < target_time - 1.0e-14:
                if quasi_steady:
                    # With S_s=0 there is no time derivative. Only the end of
                    # the channel ramp and requested output times are distinct
                    # operator states; artificial substeps would just replay
                    # the same steady solve and rebuild fast workspaces.
                    next_event = float(target_time)
                    if current_time < channel_ramp_days < target_time:
                        next_event = float(channel_ramp_days)
                    dt_days = next_event - current_time
                else:
                    dt_days = min(
                        maximum_timestep_days,
                        float(target_time - current_time),
                    )
                next_time = current_time + dt_days
                boundary_values, _ = boundary_values_at_time(
                    model=model,
                    time_days=next_time,
                    channel_ramp_days=channel_ramp_days,
                )
                boundary_changed = not np.array_equal(
                    boundary_values,
                    current_boundary_values,
                )
                if boundary_changed:
                    solver.update_boundary_values(boundary_values)
                    current_boundary_values = boundary_values.copy()
                previous = head.copy()
                last_previous = previous
                last_dt_days = dt_days
                if quasi_steady and state_is_solved and not boundary_changed:
                    last_info = {
                        "converged": True,
                        "n_cycles_used": 0,
                        "r_rms_end": 0.0,
                    }
                else:
                    head, last_info = _solve_state(
                        solver=solver,
                        model=model,
                        initial_head=previous,
                        head_prev=previous,
                        storage=storage,
                        storage_volume_coeff=storage_volume_coeff,
                        channel_plan=channel_plan,
                        outlet_plan=outlet_plan,
                        other_source_sink=other_source_sink,
                        dt_days=dt_days,
                        quasi_steady=quasi_steady,
                        transient_abs_tol_min=transient_abs_tol_min,
                    )
                    refinements = int(last_info.get("river_budget_refinements", 0))
                    total_budget_refinements += refinements
                    recommended_floor = last_info.get(
                        "river_recommended_abs_tol_min"
                    )
                    if recommended_floor is not None:
                        recommended_floor_f = float(recommended_floor)
                        transient_abs_tol_min = (
                            recommended_floor_f
                            if transient_abs_tol_min is None
                            else min(transient_abs_tol_min, recommended_floor_f)
                        )
                    if not bool(last_info.get("converged", False)):
                        raise RuntimeError(
                            f"River transient solve did not converge at t={next_time:g} days."
                        )
                    state_is_solved = True
                if "river_channel_inflow" in last_info:
                    channel_step = float(last_info["river_channel_inflow"])
                else:
                    channel_step, _, _ = _boundary_budget(
                        head=head,
                        channel_plan=channel_plan,
                        outlet_plan=outlet_plan,
                        other_source_sink=other_source_sink,
                    )
                cumulative_leakage += channel_step * dt_days
                current_time = next_time
                internal_steps += 1
                if (
                    progress_every_steps > 0
                    and internal_steps % progress_every_steps == 0
                ):
                    print(
                        f"  transient progress outlet={model.outlet_mode:16s} "
                        f"t={current_time:9.2f}/{times[-1]:g} days "
                        f"steps={internal_steps:5d} "
                        f"cycles={int(last_info.get('n_cycles_used', last_info.get('n_iter_used', -1))):4d} "
                        f"imbalance={float(last_info.get('river_relative_mass_imbalance', np.nan)):.3e}"
                    )

            if "river_channel_inflow" in last_info:
                channel = float(last_info["river_channel_inflow"])
                outlet = float(last_info["river_outlet_outflow"])
                other = float(last_info["river_other_source_sink"])
            else:
                channel, outlet, other = _boundary_budget(
                    head=head,
                    channel_plan=channel_plan,
                    outlet_plan=outlet_plan,
                    other_source_sink=other_source_sink,
                )
            output_storage = float(np.sum(storage_volume_coeff * head))
            storage_rate = (
                0.0
                if quasi_steady
                else float(
                    np.sum(storage_volume_coeff * (head - last_previous))
                    / last_dt_days
                )
            )
            residual, relative = _relative_budget_error(
                channel=channel,
                outlet=outlet,
                other=other,
                storage_rate=storage_rate,
            )
            _, ramp_fraction = boundary_values_at_time(
                model=model,
                time_days=float(target_time),
                channel_ramp_days=channel_ramp_days,
            )
            rows.append(
                TransientStepResult(
                    time_days=float(target_time),
                    dt_from_previous_output_days=float(target_time - last_output_time),
                    channel_inflow=channel,
                    outlet_outflow=outlet,
                    other_source_sink=other,
                    storage_rate=storage_rate,
                    budget_residual=residual,
                    relative_mass_imbalance=relative,
                    cumulative_channel_leakage=float(cumulative_leakage),
                    storage_change_from_initial=output_storage - initial_storage,
                    solved=True,
                    converged=True,
                    cycles=int(last_info.get("n_cycles_used", last_info.get("n_iter_used", -1))),
                    residual_rms=float(last_info.get("r_rms_end", np.nan)),
                    channel_ramp_fraction=ramp_fraction,
                )
            )
            heads.append(head.copy())
            last_output_time = float(target_time)
    finally:
        solver.close()

    metadata = {
        "scientific_scope": "saturated_darcy_structural_envelope",
        "time_formulation": "quasi_steady_zero_storage" if quasi_steady else "confined_backward_euler",
        "comparison_warning": (
            "OGS uses Richards-flow saturation and relative permeability; "
            "early-time differences are model-form effects, not solver error."
        ),
        "channel_ramp_days": float(channel_ramp_days),
        "maximum_timestep_days": float(maximum_timestep_days),
        "braidplain_specific_storage": float(braidplain_specific_storage),
        "regional_specific_storage": float(regional_specific_storage),
        "internal_steps": int(internal_steps),
        "budget_refinements": int(total_budget_refinements),
        "final_transient_abs_tol_min": (
            None
            if transient_abs_tol_min is None
            else float(transient_abs_tol_min)
        ),
        "progress_every_steps": int(progress_every_steps),
    }
    return rows, np.asarray(heads), metadata


def save_transient_case(
    *,
    path: Path,
    model: CrossSectionModel,
    rows: list[TransientStepResult],
    heads: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """Persist one complete DarcyWarp sequence as a compressed NPZ artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        times_days=np.asarray([row.time_days for row in rows]),
        heads=np.asarray(heads),
        x=model.x,
        z=model.z,
        active=model.active,
        bc_mask=model.bc_mask,
        channel_mask=model.channel_mask,
        outlet_mask=model.outlet_mask,
        physical_mask=model.physical_mask,
        kx=model.kx,
        kz=model.kz,
        step_results_json=np.asarray(json.dumps([asdict(row) for row in rows])),
        metadata_json=np.asarray(json.dumps(metadata)),
    )


def load_ogs_reference(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the standardized OGS reference artifact."""

    with np.load(path, allow_pickle=False) as data:
        required = {
            "times_days",
            "pressure_heads",
            "coordinates",
            "channel_inflow",
        }
        missing = sorted(required.difference(data.files))
        if missing:
            raise ValueError(f"OGS reference is missing fields: {missing}.")
        reference = {name: np.asarray(data[name]) for name in data.files}
    times = np.asarray(reference["times_days"], dtype=np.float64)
    heads = np.asarray(reference["pressure_heads"], dtype=np.float64)
    coordinates = np.asarray(reference["coordinates"], dtype=np.float64)
    if times.ndim != 1 or np.any(np.diff(times) <= 0.0):
        raise ValueError("OGS reference times_days must be strictly increasing.")
    if heads.shape != (times.size, coordinates.shape[0]):
        raise ValueError("OGS pressure_heads must have shape (n_times, n_nodes).")
    if coordinates.ndim != 2 or coordinates.shape[1] < 2:
        raise ValueError("OGS coordinates must have shape (n_nodes, >=2).")
    for name in ("times_days", "pressure_heads", "coordinates", "channel_inflow"):
        if not np.all(np.isfinite(reference[name])):
            raise ValueError(f"OGS reference field {name!r} must be finite.")
    return reference


def compare_to_ogs_reference(
    *,
    model: CrossSectionModel,
    rows: list[TransientStepResult],
    heads: np.ndarray,
    reference_path: Path,
) -> list[OGSComparisonResult]:
    """Compare total heads and one-sided leakage without imposing pass gates."""

    from scipy.spatial import Delaunay
    from scipy.interpolate import LinearNDInterpolator

    reference = load_ogs_reference(path=reference_path)
    ogs_times = np.asarray(reference["times_days"], dtype=np.float64)
    pressure_heads = np.asarray(reference["pressure_heads"], dtype=np.float64)
    coordinates = np.asarray(reference["coordinates"], dtype=np.float64)
    elevation_datum = float(model.config.domain_top)
    if "total_heads" in reference:
        ogs_total_heads = np.asarray(reference["total_heads"], dtype=np.float64)
        if ogs_total_heads.shape != pressure_heads.shape:
            raise ValueError("OGS total_heads must match pressure_heads shape.")
    else:
        ogs_total_heads = pressure_heads + coordinates[None, :, 1] + elevation_datum
    query_mask = (
        (model.active != 0)
        & model.physical_mask
        & (model.bc_mask == 0)
    )
    iz, iy, ix = np.nonzero(query_mask)
    query = np.column_stack((model.x[ix], model.z[iz] - elevation_datum))
    triangulation = Delaunay(coordinates[:, :2])

    raw_ogs_channel = np.asarray(reference["channel_inflow"], dtype=np.float64)
    if raw_ogs_channel.size == ogs_times.size - 1:
        ogs_channel_times = ogs_times[1:]
        interval_ogs_channel = raw_ogs_channel
    elif raw_ogs_channel.size == ogs_times.size:
        ogs_channel_times = ogs_times
        interval_ogs_channel = raw_ogs_channel[1:]
    else:
        raise ValueError("OGS channel_inflow must have n_times or n_times-1 values.")
    ogs_cumulative = np.concatenate(
        ([0.0], np.cumsum(interval_ogs_channel * np.diff(ogs_times)))
    )
    ogs_cumulative_times = ogs_times

    comparisons: list[OGSComparisonResult] = []
    for index, row in enumerate(rows):
        time = float(row.time_days)
        if time < ogs_times[0] - 1.0e-12 or time > ogs_times[-1] + 1.0e-12:
            raise ValueError(
                f"DarcyWarp output time {time:g} lies outside the OGS "
                f"reference range [{ogs_times[0]:g}, {ogs_times[-1]:g}] days."
            )
        upper = int(np.searchsorted(ogs_times, time, side="left"))
        if upper == 0:
            field = ogs_total_heads[0]
        elif upper >= ogs_times.size:
            field = ogs_total_heads[-1]
        elif np.isclose(ogs_times[upper], time):
            field = ogs_total_heads[upper]
        else:
            lower = upper - 1
            weight = (time - ogs_times[lower]) / (ogs_times[upper] - ogs_times[lower])
            field = (1.0 - weight) * ogs_total_heads[lower] + weight * ogs_total_heads[upper]
        interpolator = LinearNDInterpolator(triangulation, field, fill_value=np.nan)
        ogs_at_cells = np.asarray(interpolator(query), dtype=np.float64)
        dw_at_cells = np.asarray(heads[index][iz, iy, ix], dtype=np.float64)
        valid = np.isfinite(ogs_at_cells) & np.isfinite(dw_at_cells)
        if not np.any(valid):
            raise RuntimeError(f"No overlapping OGS/DarcyWarp cells at t={time:g} days.")
        difference = dw_at_cells[valid] - ogs_at_cells[valid]
        ogs_channel = (
            float("nan")
            if time < ogs_channel_times[0]
            else float(np.interp(time, ogs_channel_times, raw_ogs_channel))
        )
        ogs_cum = float(np.interp(time, ogs_cumulative_times, ogs_cumulative))
        q_scale = max(abs(ogs_channel), 1.0e-30)
        cum_scale = max(abs(ogs_cum), 1.0e-30)
        comparisons.append(
            OGSComparisonResult(
                time_days=time,
                compared_cells=int(np.count_nonzero(valid)),
                total_head_rmse=float(np.sqrt(np.mean(difference * difference))),
                total_head_mean_bias=float(np.mean(difference)),
                total_head_max_abs=float(np.max(np.abs(difference))),
                darcywarp_channel_inflow=float(row.channel_inflow),
                ogs_channel_inflow=ogs_channel,
                channel_inflow_relative_difference=(
                    float("nan")
                    if not np.isfinite(ogs_channel)
                    else float((row.channel_inflow - ogs_channel) / q_scale)
                ),
                darcywarp_cumulative_channel_leakage=float(row.cumulative_channel_leakage),
                ogs_cumulative_channel_leakage=ogs_cum,
                cumulative_leakage_relative_difference=float((row.cumulative_channel_leakage - ogs_cum) / cum_scale),
            )
        )
    return comparisons
