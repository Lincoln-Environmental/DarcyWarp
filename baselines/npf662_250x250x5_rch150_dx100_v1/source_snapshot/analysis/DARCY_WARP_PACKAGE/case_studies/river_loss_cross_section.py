#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Example anisotropic river-loss cross-section built on generic 3D inputs.

The case-study builder in this module assembles a physically 2D river section
as an ``(nz, 1, nx)`` structured model.  The resulting fields are validated by
``Model3DInputs`` and solved through the public ``WarpDarcySolver3D`` API.
River geometry and outlet policies remain local to this example; the generic
input and boundary-budget utilities are reusable by other case studies.

The physically 2D x-z section is represented as an ``(nz, 1, nx)`` grid so it
uses DarcyWarp's directional 7-point finite-volume operator.  The script is
intended as a transparent study entry point for testing whether inferred river
loss sensitivity to vertical anisotropy depends on the far-field outlet
structure. The experiment runner and output reporting live in
``working_tests/run_river_loss_cross_section.py``.

Geometry represented by the defaults
------------------------------------
* Physical section: x = 0..5000 m, z = 0..100 m.
* Half-channel: x = 0..50 m, bed z = 98 m, water surface z = 99 m.
* Braidplain: x = 0..500 m, upper 5 m of the section.
* Channel total head: 99 m.
* Far-field total head: 96 m.
* Out-of-plane model width: 1 m.

Three outlet structures are provided:
* ``full_depth``: fixed head over the complete right boundary.
* ``lower_only``: fixed head only below the braidplain/regional interface.
* ``water_table_only``: fixed head only below the prescribed far-field
  water table. This is the boundary structure used by the OGS braidplain case.

Boundary placement
------------------
Channel-water cells are fixed-head cells.  Air above the channel is inactive.
A one-column fixed-head ghost region is added beyond x = domain_length.  Large
conductivity in fixed-head water/ghost cells removes their half-cell resistance,
placing the effective prescribed-head boundary on the aquifer-facing cell face.

Units
-----
The defaults assume metres and days. Reported boundary flows are m3/day per
metre of out-of-plane channel length (numerically equivalent to m2/day).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Literal

import numpy as np

from DARCY_WARP_PACKAGE.model_3d_inputs import Model3DInputs
from DARCY_WARP_PACKAGE.physics.budget_3d import boundary_interface_flux
from DARCY_WARP_PACKAGE.solvers_3d import build_7point_face_conductance_from_k


OutletMode = Literal["full_depth", "lower_only", "water_table_only"]
AnisotropyTarget = Literal["braidplain", "regional", "both"]


@dataclass(frozen=True)
class CrossSectionConfig:
    """Geometry, materials, boundaries, and solver controls."""

    # Physical geometry.
    domain_length: float = 5000.0
    domain_bottom: float = 0.0
    domain_top: float = 100.0
    dx: float = 10.0
    dz: float = 1.0
    out_of_plane_width: float = 1.0

    # Channel and braidplain geometry.
    channel_half_width: float = 50.0
    channel_bed_elevation: float = 98.0
    channel_water_surface: float = 99.0
    braidplain_half_width: float = 500.0
    braidplain_thickness: float = 5.0

    # Prescribed heads.
    channel_head: float = 99.0
    discharge_head: float = 96.0

    # Material properties. Replace these example values with study values.
    braidplain_kh: float = 100.0
    regional_kh: float = 10.0
    braidplain_anisotropy: float = 1.0  # Kh / Kv
    regional_anisotropy: float = 1.0   # Kh / Kv

    # A large K in fixed-head water/ghost cells makes the boundary resistance
    # approach the aquifer half-cell resistance at the shared face.
    fixed_head_k_multiplier: float = 1.0e6

    # Solver controls.
    device: str = "cuda:0"
    solver: str = "kcycle"
    implementation: str = "classic"
    max_cycles: int = 200
    max_levels: int = 6
    # The 3D flow residual is conductivity-weighted; for this thin section its
    # absolute scale is much larger than the head-equivalent residual.  The
    # practical value below still drives the head-equivalent residual to the
    # micrometre range while avoiding a false non-converged status.
    rel_tol: float = 5.0e-5
    abs_tol_min: float = 5.0e-5
    dh_rms_tol: float = 1.0e-4
    check_every_no: int = 1
    nu_pre: int = 6
    nu_post: int = 6
    nu_coarse: int = 2
    omega: float = 0.8
    min_coarse_n: int = 1
    line_omega: float = 0.8
    line_sweeps_pre: int = 1
    line_sweeps_post: int = 1
    line_sweeps_coarse: int = 1
    vertical_line_max_nz: int = 128
    # This is an acceptance gate, separate from the iterative residual.  A
    # local residual can be small while a long, thin section still has a
    # detectable integrated boundary-flux defect.
    # The outlet-structure signal is below 1e-3 for the default geometry, so
    # the budget gate must be materially smaller than that comparison signal.
    practical_mass_imbalance_tol: float = 2.0e-6
    robust_retry_enabled: bool = True
    robust_max_cycles: int = 800
    robust_nu_pre: int = 13
    robust_nu_post: int = 13
    robust_nu_coarse: int = 3
    robust_omega: float = 0.7
    robust_rel_tol: float = 1.0e-10
    robust_abs_tol_min: float = 1.0e-10
    robust_dh_rms_tol: float = 1.0e-8
    smoother: str = "chebyshev_vertical_line"

    @property
    def braidplain_base(self) -> float:
        return self.domain_top - self.braidplain_thickness


@dataclass(frozen=True)
class CrossSectionModel:
    """Fully assembled arrays and coordinate metadata for one model case."""

    config: CrossSectionConfig
    outlet_mode: OutletMode
    inputs: Model3DInputs
    x: np.ndarray
    z: np.ndarray
    channel_mask: np.ndarray
    outlet_mask: np.ndarray
    physical_mask: np.ndarray

    @property
    def kx(self) -> np.ndarray:
        return self.inputs.kx

    @property
    def ky(self) -> np.ndarray:
        return self.inputs.ky

    @property
    def kz(self) -> np.ndarray:
        return self.inputs.kz

    @property
    def active(self) -> np.ndarray:
        return self.inputs.active

    @property
    def bc_mask(self) -> np.ndarray:
        return self.inputs.bc_mask

    @property
    def bc_values(self) -> np.ndarray:
        return self.inputs.bc_values

    @property
    def rhs(self) -> np.ndarray:
        return self.inputs.rhs

    @property
    def initial_head(self) -> np.ndarray | None:
        return self.inputs.initial_head


@dataclass(frozen=True)
class CaseResult:
    """Scalar diagnostics from one anisotropy/outlet run."""

    outlet_mode: str
    anisotropy_target: str
    varied_ratio: float
    braidplain_anisotropy: float
    regional_anisotropy: float
    braidplain_kh: float
    braidplain_kv: float
    regional_kh: float
    regional_kv: float
    channel_inflow: float
    outlet_outflow: float
    net_fixed_head_inflow: float
    relative_mass_imbalance: float
    converged: bool
    cycles: int
    residual_rms: float
    head_min: float
    head_max: float
    head_file: str | None = None
    retry_attempted: bool = False
    accepted_attempt: int = 0
    other_source_sink: float = 0.0


def _require_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0; got {value!r}.")


def _require_face_aligned(name: str, value: float, origin: float, spacing: float) -> None:
    index = (float(value) - float(origin)) / float(spacing)
    if not np.isclose(index, round(index), rtol=0.0, atol=1.0e-9):
        raise ValueError(
            f"{name}={value} must align with a grid face from origin={origin} "
            f"using spacing={spacing}; index={index}."
        )


def validate_config(cfg: CrossSectionConfig) -> None:
    """Fail early when geometry cannot be represented exactly by the grid."""

    for name in (
        "domain_length",
        "dx",
        "dz",
        "out_of_plane_width",
        "channel_half_width",
        "braidplain_half_width",
        "braidplain_thickness",
        "braidplain_kh",
        "regional_kh",
        "braidplain_anisotropy",
        "regional_anisotropy",
        "fixed_head_k_multiplier",
    ):
        _require_positive(name, float(getattr(cfg, name)))

    if cfg.domain_top <= cfg.domain_bottom:
        raise ValueError("domain_top must exceed domain_bottom.")
    for name in ("domain_bottom", "domain_top", "channel_bed_elevation", "channel_water_surface", "channel_head", "discharge_head"):
        if not np.isfinite(float(getattr(cfg, name))):
            raise ValueError(f"{name} must be finite.")
    if cfg.channel_head < cfg.discharge_head:
        raise ValueError("channel_head must be greater than or equal to discharge_head for a loss experiment.")
    if not (cfg.domain_bottom < cfg.channel_bed_elevation < cfg.channel_water_surface <= cfg.domain_top):
        raise ValueError("Channel bed/water-surface elevations lie outside the model domain.")
    if cfg.channel_water_surface - cfg.channel_bed_elevation < cfg.dz:
        raise ValueError("Channel water thickness must contain at least one vertical cell.")
    if cfg.channel_half_width >= cfg.braidplain_half_width:
        raise ValueError("channel_half_width must be less than braidplain_half_width.")
    if cfg.braidplain_half_width >= cfg.domain_length:
        raise ValueError("braidplain_half_width must be less than domain_length.")
    if cfg.braidplain_base <= cfg.domain_bottom:
        raise ValueError("Braidplain base must lie above the model bottom.")
    if cfg.channel_half_width >= cfg.domain_length:
        raise ValueError("channel_half_width must fit inside the physical domain.")
    if cfg.braidplain_thickness >= cfg.domain_top - cfg.domain_bottom:
        raise ValueError("braidplain_thickness must fit inside the vertical domain.")
    if str(cfg.solver).strip().lower() not in {"kcycle", "chebyshev"}:
        raise ValueError("solver must be 'kcycle' or 'chebyshev'.")
    if str(cfg.implementation).strip().lower() not in {"classic", "fast"}:
        raise ValueError("implementation must be 'classic' or 'fast'.")
    if str(cfg.implementation).strip().lower() == "fast" and str(cfg.solver).strip().lower() != "kcycle":
        raise ValueError("implementation='fast' supports solver='kcycle' only.")
    if str(cfg.implementation).strip().lower() == "fast" and str(cfg.smoother).strip().lower() not in {"jacobi", "chebyshev"}:
        raise ValueError("implementation='fast' supports only smoother='jacobi' or 'chebyshev'.")
    for name in (
        "rel_tol",
        "abs_tol_min",
        "dh_rms_tol",
        "practical_mass_imbalance_tol",
        "robust_omega",
        "robust_rel_tol",
        "robust_abs_tol_min",
        "robust_dh_rms_tol",
    ):
        _require_positive(name, float(getattr(cfg, name)))
    for name in (
        "max_cycles",
        "max_levels",
        "min_coarse_n",
        "check_every_no",
        "nu_pre",
        "nu_post",
        "nu_coarse",
        "robust_max_cycles",
        "robust_nu_pre",
        "robust_nu_post",
        "robust_nu_coarse",
    ):
        if int(getattr(cfg, name)) < 1:
            raise ValueError(f"{name} must be an integer >= 1.")

    _require_face_aligned("domain_length", cfg.domain_length, 0.0, cfg.dx)
    _require_face_aligned("channel_half_width", cfg.channel_half_width, 0.0, cfg.dx)
    _require_face_aligned("braidplain_half_width", cfg.braidplain_half_width, 0.0, cfg.dx)
    _require_face_aligned("domain_top", cfg.domain_top, cfg.domain_bottom, cfg.dz)
    _require_face_aligned("channel_bed_elevation", cfg.channel_bed_elevation, cfg.domain_bottom, cfg.dz)
    _require_face_aligned("channel_water_surface", cfg.channel_water_surface, cfg.domain_bottom, cfg.dz)
    _require_face_aligned("braidplain_base", cfg.braidplain_base, cfg.domain_bottom, cfg.dz)


def _broadcast_xz(mask_xz: np.ndarray) -> np.ndarray:
    """Insert the singleton out-of-plane dimension: (nz, nx) -> (nz, 1, nx)."""

    return np.asarray(mask_xz)[:, None, :]


def _require_connected_active_domain(
    active_xz: np.ndarray,
    channel_xz: np.ndarray,
    outlet_xz: np.ndarray,
) -> None:
    """Require every active cell to share the channel's connected component."""

    active_bool = np.asarray(active_xz, dtype=bool)
    channel_bool = np.asarray(channel_xz, dtype=bool)
    outlet_bool = np.asarray(outlet_xz, dtype=bool)
    if active_bool.ndim != 2:
        raise ValueError("Connectivity fields must be two-dimensional x-z arrays.")
    if channel_bool.shape != active_bool.shape or outlet_bool.shape != active_bool.shape:
        raise ValueError("Connectivity fields must have matching shapes.")
    channel_seeds = np.argwhere(channel_bool & active_bool)
    if channel_seeds.size == 0:
        raise ValueError("The active domain contains no channel seed cell.")

    seen = np.zeros_like(active_bool, dtype=bool)
    stack = [tuple(int(value) for value in channel_seeds[0])]
    while stack:
        iz, ix = stack.pop()
        if seen[iz, ix]:
            continue
        seen[iz, ix] = True
        for dz_step, dx_step in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            neighbour = (iz + dz_step, ix + dx_step)
            if (
                0 <= neighbour[0] < active_bool.shape[0]
                and 0 <= neighbour[1] < active_bool.shape[1]
                and active_bool[neighbour]
                and not seen[neighbour]
            ):
                stack.append(neighbour)

    if not np.all(seen[active_bool]) or not np.all(seen[outlet_bool & active_bool]):
        raise ValueError(
            "Active domain must be one connected component linking the channel and outlet boundaries."
        )


def _free_active_source_sink(
    rhs: np.ndarray,
    active: np.ndarray,
    bc_mask: np.ndarray,
) -> float:
    """Return sources/sinks represented in equations for free active cells."""

    rhs_array = np.asarray(rhs, dtype=np.float64)
    active_array = np.asarray(active)
    boundary_array = np.asarray(bc_mask)
    if active_array.shape != rhs_array.shape or boundary_array.shape != rhs_array.shape:
        raise ValueError("rhs, active, and bc_mask must have matching shapes.")
    free_active = (active_array != 0) & (boundary_array == 0)
    return float(np.sum(rhs_array[free_active], dtype=np.float64))


def build_cross_section(
    cfg: CrossSectionConfig,
    outlet_mode: OutletMode,
) -> CrossSectionModel:
    """Assemble an anisotropic steady saturated x-z cross-section.

    A ghost column beyond the physical right boundary carries the prescribed
    head.  For ``lower_only``, the upper ghost cells are inactive, making that
    part of the physical right boundary no-flow.
    """

    validate_config(cfg)
    if outlet_mode not in {"full_depth", "lower_only", "water_table_only"}:
        raise ValueError(f"Unsupported outlet_mode={outlet_mode!r}.")

    n_physical_x = int(round(cfg.domain_length / cfg.dx))
    nx = n_physical_x + 1  # one right-side ghost column
    nz = int(round((cfg.domain_top - cfg.domain_bottom) / cfg.dz))
    ny = 1
    shape = (nz, ny, nx)

    # k=0 is the top layer; k increases downward, matching the 3D solver's
    # layered/unconfined conventions.
    x = (np.arange(nx, dtype=np.float64) + 0.5) * cfg.dx
    z = cfg.domain_top - (np.arange(nz, dtype=np.float64) + 0.5) * cfg.dz
    xx, zz = np.meshgrid(x, z)

    physical_xz = xx < cfg.domain_length
    ghost_xz = ~physical_xz

    channel_column_xz = physical_xz & (xx < cfg.channel_half_width)
    channel_water_xz = (
        channel_column_xz
        & (zz >= cfg.channel_bed_elevation)
        & (zz < cfg.channel_water_surface)
    )
    channel_air_xz = channel_column_xz & (zz >= cfg.channel_water_surface)

    # The physical aquifer is active except for air above the channel. Ghost
    # cells are activated only where a prescribed-head outlet is requested.
    active_xz = physical_xz & (~channel_air_xz)
    if outlet_mode == "full_depth":
        outlet_xz = ghost_xz
    elif outlet_mode == "lower_only":
        outlet_xz = ghost_xz & (zz < cfg.braidplain_base)
    else:
        # Cell centres below discharge_head represent a boundary extending
        # from the model base to the face-aligned far-field water table.
        outlet_xz = ghost_xz & (zz < cfg.discharge_head)
    active_xz |= outlet_xz

    braidplain_xz = (
        physical_xz
        & (xx < cfg.braidplain_half_width)
        & (zz >= cfg.braidplain_base)
        & (~channel_water_xz)
        & (~channel_air_xz)
    )
    aquifer_xz = active_xz & (~channel_water_xz) & (~outlet_xz)
    regional_xz = aquifer_xz & (~braidplain_xz)

    braidplain_kv = cfg.braidplain_kh / cfg.braidplain_anisotropy
    regional_kv = cfg.regional_kh / cfg.regional_anisotropy

    kx_xz = np.zeros((nz, nx), dtype=np.float64)
    ky_xz = np.zeros((nz, nx), dtype=np.float64)
    kz_xz = np.zeros((nz, nx), dtype=np.float64)

    kx_xz[braidplain_xz] = cfg.braidplain_kh
    ky_xz[braidplain_xz] = cfg.braidplain_kh
    kz_xz[braidplain_xz] = braidplain_kv

    kx_xz[regional_xz] = cfg.regional_kh
    ky_xz[regional_xz] = cfg.regional_kh
    kz_xz[regional_xz] = regional_kv

    # Fixed-head water/ghost cells use a high conductivity to remove their
    # artificial half-cell resistance while retaining a shared FV face flux.
    reference_k = max(cfg.braidplain_kh, cfg.regional_kh)
    fixed_k = cfg.fixed_head_k_multiplier * reference_k
    fixed_xz = channel_water_xz | outlet_xz
    kx_xz[fixed_xz] = fixed_k
    ky_xz[fixed_xz] = fixed_k
    kz_xz[fixed_xz] = fixed_k

    active = _broadcast_xz(active_xz.astype(np.int32))
    channel_mask = _broadcast_xz(channel_water_xz)
    outlet_mask = _broadcast_xz(outlet_xz)
    physical_mask = _broadcast_xz(physical_xz)

    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape, dtype=np.float64)
    bc_mask[channel_mask] = 1
    bc_values[channel_mask] = cfg.channel_head
    bc_mask[outlet_mask] = 1
    bc_values[outlet_mask] = cfg.discharge_head

    rhs = np.zeros(shape, dtype=np.float64)

    # Linear x-gradient initialisation for free aquifer cells.
    x_fraction = np.clip(xx / cfg.domain_length, 0.0, 1.0)
    h0_xz = cfg.channel_head + (cfg.discharge_head - cfg.channel_head) * x_fraction
    h0 = _broadcast_xz(h0_xz.astype(np.float64))
    h0[bc_mask != 0] = bc_values[bc_mask != 0]
    h0[active == 0] = 0.0

    if np.any(channel_mask & outlet_mask):
        raise ValueError("channel and outlet boundary masks overlap.")
    if not np.any(channel_mask):
        raise ValueError("The assembled grid contains no channel boundary cells.")
    if not np.any(outlet_mask):
        raise ValueError(f"The {outlet_mode} outlet contains no boundary cells.")
    _require_connected_active_domain(
        active_xz=active[:, 0, :] != 0,
        channel_xz=channel_mask[:, 0, :],
        outlet_xz=outlet_mask[:, 0, :],
    )

    inputs = Model3DInputs(
        kx=_broadcast_xz(kx_xz),
        ky=_broadcast_xz(ky_xz),
        kz=_broadcast_xz(kz_xz),
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        rhs=rhs,
        initial_head=h0,
        dx=cfg.dx,
        dy=cfg.out_of_plane_width,
        dz=cfg.dz,
        named_masks={
            "channel": channel_mask,
            "outlet": outlet_mask,
            "physical": physical_mask,
        },
        metadata={
            "case": "river_loss_cross_section",
            "outlet_mode": outlet_mode,
            "scientific_scope": "saturated steady Darcy-flow structural-sensitivity experiment",
            "braidplain_anisotropy": float(cfg.braidplain_anisotropy),
            "regional_anisotropy": float(cfg.regional_anisotropy),
            "channel_cells": int(np.count_nonzero(channel_mask)),
            "outlet_cells": int(np.count_nonzero(outlet_mask)),
        },
    )

    return CrossSectionModel(
        config=cfg,
        outlet_mode=outlet_mode,
        inputs=inputs,
        x=x,
        z=z,
        channel_mask=channel_mask,
        outlet_mask=outlet_mask,
        physical_mask=physical_mask,
    )


def _robust_solver_kwargs(
    cfg: CrossSectionConfig,
    simple_kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Return the deliberate, materially tighter retry configuration.

    The normal settings provide an economical first solve. The retry target is
    intentionally absolute and sufficiently tight to resolve the small
    difference between the two outlet structures.
    """

    robust_kwargs = dict(simple_kwargs)
    robust_kwargs.update({
        "rel_tol": cfg.robust_rel_tol,
        "abs_tol_min": cfg.robust_abs_tol_min,
        "dh_rms_tol": cfg.robust_dh_rms_tol,
    })
    if str(cfg.solver).strip().lower() == "chebyshev":
        robust_kwargs["max_iter"] = max(cfg.max_cycles, cfg.robust_max_cycles)
    else:
        robust_kwargs.update({
            "max_cycles": max(cfg.max_cycles, cfg.robust_max_cycles),
            "nu_pre": cfg.robust_nu_pre,
            "nu_post": cfg.robust_nu_post,
            "nu_coarse": cfg.robust_nu_coarse,
            "omega": cfg.robust_omega,
        })
    return robust_kwargs


def _solver_kwargs(cfg: CrossSectionConfig) -> dict[str, Any]:
    """Return only controls accepted by the selected linear backend."""

    common = {
        "rel_tol": cfg.rel_tol,
        "abs_tol_min": cfg.abs_tol_min,
        "dh_rms_tol": cfg.dh_rms_tol,
    }
    if str(cfg.solver).strip().lower() == "chebyshev":
        return {"max_iter": cfg.max_cycles, **common}

    kwargs: dict[str, Any] = {
        "max_cycles": cfg.max_cycles,
        "max_levels": cfg.max_levels,
        "min_coarse_n": cfg.min_coarse_n,
        "check_every_no": cfg.check_every_no,
        "smoother": cfg.smoother,
        "nu_pre": cfg.nu_pre,
        "nu_post": cfg.nu_post,
        "nu_coarse": cfg.nu_coarse,
        "omega": cfg.omega,
        **common,
    }
    if str(cfg.implementation).strip().lower() == "classic":
        kwargs.update({
            "line_omega": cfg.line_omega,
            "line_sweeps_pre": cfg.line_sweeps_pre,
            "line_sweeps_post": cfg.line_sweeps_post,
            "line_sweeps_coarse": cfg.line_sweeps_coarse,
            "vertical_line_max_nz": cfg.vertical_line_max_nz,
        })
    return kwargs


def _attempt_summary(attempt: dict[str, object], attempt_number: int) -> str:
    """Return compact diagnostics suitable for an acceptance-failure error."""

    info = attempt["info"]
    assert isinstance(info, dict)
    return (
        f"attempt {attempt_number}: solver_converged={bool(info.get('converged', False))}, "
        f"r_rms={float(info.get('r_rms_end', np.nan)):.3e}, "
        f"dh_rms={float(info.get('dh_rms_lastcheck', np.nan)):.3e}, "
        f"relative_mass_imbalance={float(attempt['relative']):.3e}"
    )


def solve_case(
    model: CrossSectionModel,
    *,
    varied_ratio: float,
    anisotropy_target: AnisotropyTarget,
    output_directory: Path | None = None,
    save_head: bool = False,
) -> tuple[CaseResult, np.ndarray]:
    """Build face conductances, solve one case, and calculate boundary budgets."""

    cfg = model.config
    tx_p, tx_m, ty_p, ty_m, tz_p, tz_m = build_7point_face_conductance_from_k(
        kx_field=model.kx,
        ky_field=model.ky,
        kz_field=model.kz,
        active=model.active,
        dx=cfg.dx,
        dy=cfg.out_of_plane_width,
        dz=cfg.dz,
    )

    solver = model.inputs.build_solver(
        device=cfg.device,
        solver=cfg.solver,
        implementation=cfg.implementation,
        face_conductance=(tx_p, tx_m, ty_p, ty_m, tz_p, tz_m),
    )
    simple_kwargs = _solver_kwargs(cfg=cfg)
    if str(cfg.implementation).lower() == "fast" and str(cfg.solver).lower() != "kcycle":
        raise ValueError("implementation='fast' supports solver='kcycle' only.")

    def evaluate(candidate_head: np.ndarray, candidate_info: dict[str, object]) -> dict[str, object]:
        candidate = np.asarray(candidate_head, dtype=np.float64)
        channel = boundary_interface_flux(
            head=candidate, boundary_mask=model.channel_mask, active=model.active,
            tx_p=tx_p, tx_m=tx_m, ty_p=ty_p, ty_m=ty_m, tz_p=tz_p, tz_m=tz_m,
        )
        outlet = -boundary_interface_flux(
            head=candidate, boundary_mask=model.outlet_mask, active=model.active,
            tx_p=tx_p, tx_m=tx_m, ty_p=ty_p, ty_m=ty_m, tz_p=tz_p, tz_m=tz_m,
        )
        other = _free_active_source_sink(
            rhs=model.rhs,
            active=model.active,
            bc_mask=model.bc_mask,
        )
        net = channel - outlet + other
        scale = max(abs(channel), abs(outlet), abs(other))
        relative = 0.0 if scale == 0.0 and net == 0.0 else (abs(net) / scale if scale > 0.0 else float("inf"))
        solver_ok = bool(candidate_info.get("converged", False))
        mass_ok = bool(np.isfinite(relative) and relative <= cfg.practical_mass_imbalance_tol)
        return {
            "head": candidate,
            "info": candidate_info,
            "channel": channel,
            "outlet": outlet,
            "other": other,
            "net": net,
            "relative": relative,
            "accepted": solver_ok and mass_ok,
        }

    attempts: list[dict[str, object]] = []
    try:
        first_head, first_info = solver.solve(
            **simple_kwargs,
            initial_head=(None if model.initial_head is None else model.initial_head.copy()),
        )
        first = evaluate(first_head, first_info)
        attempts.append(first)
        if cfg.robust_retry_enabled and not bool(first["accepted"]):
            robust_kwargs = _robust_solver_kwargs(
                cfg=cfg,
                simple_kwargs=simple_kwargs,
            )
            print("Initial river solve failed convergence or mass-balance acceptance; retrying with robust solver settings.")
            retry_head, retry_info = solver.solve(
                **robust_kwargs,
                initial_head=np.asarray(first_head, dtype=np.float64).copy(),
            )
            attempts.append(evaluate(retry_head, retry_info))
    finally:
        solver.close()

    selected_index = next((index for index, item in enumerate(attempts) if bool(item["accepted"])), None)
    accepted = None if selected_index is None else attempts[selected_index]
    if accepted is None:
        diagnostics = "; ".join(
            _attempt_summary(attempt=attempt, attempt_number=index + 1)
            for index, attempt in enumerate(attempts)
        )
        raise RuntimeError(
            "River case failed numerical convergence and mass-balance "
            f"acceptance after {len(attempts)} attempt(s): {diagnostics}."
        )
    head = np.asarray(accepted["head"], dtype=np.float64)
    info = accepted["info"]
    channel_inflow = float(accepted["channel"])
    outlet_outflow = float(accepted["outlet"])
    other_source_sink = float(accepted["other"])
    net_fixed_head_inflow = float(accepted["net"])
    relative_mass_imbalance = float(accepted["relative"])
    actual_ratio = _actual_anisotropy_ratio(cfg, anisotropy_target)

    solver_converged = bool(info.get("converged", False))

    active_head = head[model.active != 0]
    head_file: str | None = None
    if save_head:
        if output_directory is None:
            raise ValueError("save_head=True requires output_directory.")
        output_directory.mkdir(parents=True, exist_ok=True)
        ratio_tag = f"{actual_ratio:g}".replace(".", "p")
        filename = f"head_{model.outlet_mode}_{anisotropy_target}_ratio_{ratio_tag}.npz"
        path = output_directory / filename
        np.savez_compressed(
            path,
            head=head,
            x=model.x,
            z=model.z,
            active=model.active,
            bc_mask=model.bc_mask,
            channel_mask=model.channel_mask,
            outlet_mask=model.outlet_mask,
            kx=model.kx,
            kz=model.kz,
        )
        head_file = str(path)

    result = CaseResult(
        outlet_mode=model.outlet_mode,
        anisotropy_target=anisotropy_target,
        varied_ratio=float(actual_ratio),
        braidplain_anisotropy=float(cfg.braidplain_anisotropy),
        regional_anisotropy=float(cfg.regional_anisotropy),
        braidplain_kh=float(cfg.braidplain_kh),
        braidplain_kv=float(cfg.braidplain_kh / cfg.braidplain_anisotropy),
        regional_kh=float(cfg.regional_kh),
        regional_kv=float(cfg.regional_kh / cfg.regional_anisotropy),
        channel_inflow=float(channel_inflow),
        outlet_outflow=float(outlet_outflow),
        net_fixed_head_inflow=float(net_fixed_head_inflow),
        relative_mass_imbalance=float(relative_mass_imbalance),
        converged=bool(solver_converged and relative_mass_imbalance <= cfg.practical_mass_imbalance_tol),
        cycles=int(info.get("n_cycles_used", info.get("n_iter_used", -1))),
        residual_rms=float(info.get("r_rms_end", np.nan)),
        head_min=float(np.min(active_head)),
        head_max=float(np.max(active_head)),
        head_file=head_file,
        retry_attempted=bool(len(attempts) > 1),
        accepted_attempt=int(selected_index + 1),
        other_source_sink=float(other_source_sink),
    )
    return result, head


def _config_for_ratio(
    base: CrossSectionConfig,
    ratio: float,
    target: AnisotropyTarget,
) -> CrossSectionConfig:
    _require_positive("anisotropy ratio", ratio)
    if target == "braidplain":
        return replace(base, braidplain_anisotropy=float(ratio))
    if target == "regional":
        return replace(base, regional_anisotropy=float(ratio))
    if target == "both":
        return replace(
            base,
            braidplain_anisotropy=float(ratio),
            regional_anisotropy=float(ratio),
        )
    raise ValueError(f"Unsupported anisotropy target {target!r}.")


def _actual_anisotropy_ratio(cfg: CrossSectionConfig, target: AnisotropyTarget) -> float:
    """Return the ratio represented by the assembled case configuration."""

    if target == "braidplain":
        return float(cfg.braidplain_anisotropy)
    if target == "regional":
        return float(cfg.regional_anisotropy)
    if target == "both":
        if not np.isclose(cfg.braidplain_anisotropy, cfg.regional_anisotropy):
            raise ValueError("anisotropy_target='both' requires matching case anisotropy values.")
        return float(cfg.braidplain_anisotropy)
    raise ValueError(f"Unsupported anisotropy target {target!r}.")


def run_sweep(
    base_config: CrossSectionConfig,
    ratios: Iterable[float],
    outlet_modes: Iterable[OutletMode],
    anisotropy_target: AnisotropyTarget,
    output_directory: Path,
    save_heads: bool,
) -> list[CaseResult]:
    """Run all ratio/outlet combinations."""

    results: list[CaseResult] = []
    for outlet_mode in outlet_modes:
        for ratio in ratios:
            case_config = _config_for_ratio(base_config, float(ratio), anisotropy_target)
            model = build_cross_section(case_config, outlet_mode)
            result, _ = solve_case(
                model,
                varied_ratio=float(ratio),
                anisotropy_target=anisotropy_target,
                output_directory=output_directory,
                save_head=save_heads,
            )
            results.append(result)
            print(
                f"outlet={outlet_mode:11s} target={anisotropy_target:10s} "
                f"Kh/Kv={ratio:8g} Qriver={result.channel_inflow: .8e} "
                f"Qout={result.outlet_outflow: .8e} "
                f"imbalance={result.relative_mass_imbalance: .3e} "
                f"cycles={result.cycles:4d} converged={result.converged}"
            )
    return results
