#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-only
"""
Build a MODFLOW 6 truth artifact for the 2D transient path.

The runner creates a single-layer transient MF6 model with weekly stress
periods and seasonal recharge. ``formulation="unconfined"`` uses a convertible
cell with specific yield and specific storage; ``formulation="confined"`` uses a
confined cell with specific storage only. Its default output is formulation
specific:
``DARCY_WARP_PACKAGE/data/working_tests/mf6_transient_2d_<formulation>/
mf6_transient_heads.npz.lzma``. That compressed artifact stores every input
needed for a future Warp-vs-MF6 transient replay: per-period heads, final heads,
steady-state warm-start heads, active and boundary masks, bottom/top elevations,
hydraulic conductivity, recharge time series, and storage parameters.

The current Warp transient tests in ``tests/test_2d_transient.py`` exercise the
2D confined and unconfined Warp code paths directly. This runner is the separate
external MF6 truth-generation step; a completed run is evidenced by the
compressed ``mf6_transient_heads.npz.lzma`` file plus the MF6 ``.hds`` file in
the sibling ``mf6`` workspace.

Usage:
    python working_tests/run_2d_transient_vs_mf6.py
"""

from __future__ import annotations

import io
import json
import lzma
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # working_tests on path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import flopy  # noqa: E402

from DARCY_WARP_PACKAGE.model_builder import (  # noqa: E402
    _build_dem,
    _build_dirichlet_boundary_mask,
    _build_domain,
    _create_chd_single_period,
    _model_bottom,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.project_base import data_store, require_mf6  # noqa: E402
from working_tests.transient_artifacts import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    all_active_transient_heads_finite,
    compute_transient_case_fingerprint,
)


# ---- defaults --------------------------------------------------------------
N_WEEKS = 30
RECHARGE_SCHEDULE_WEEKS = 52
DT_DAYS = 7.0
ANNUAL_RECHARGE_M = 0.3
DEFAULT_NX = 1000
DEFAULT_NY = 1000
DEFAULT_DX = 100.0
DEFAULT_K = 100.0
DEFAULT_SY = 0.10           # specific yield (unconfined storage)
DEFAULT_SS = 1.0e-5         # specific storage (1/m), confined/saturated portion
DEFAULT_INIT_SAT_THICKNESS = 100.0
MF6_MODEL_NAME = "tr2d_truth"
FORMULATION_CONFINED = "confined"
FORMULATION_UNCONFINED = "unconfined"
FORMULATION_MODES = {FORMULATION_CONFINED, FORMULATION_UNCONFINED}
# Transmissivity/conductivity field specification. ``ugly_t`` adopts the hard
# heterogeneous T field from the confined steady-state benchmarks
# (``model_builder.make_ugly_T_field``), converted to K via
# ``K = T / initial_saturated_thickness`` — the same convention as
# ``export_mf6_truth_npz.py`` (``hk = t_field / thickness``).
T_FIELD_HOMOGENEOUS = "homogeneous"
T_FIELD_UGLY = "ugly_t"
T_FIELD_MODES = {T_FIELD_HOMOGENEOUS, T_FIELD_UGLY}
DEFAULT_T_FIELD_KIND = T_FIELD_HOMOGENEOUS
DEFAULT_T_FIELD_SEED = 42
WARM_START_ARTIFACT_INITIAL = "artifact_initial"
WARM_START_CONFINED_STEADY_MF6 = "confined_steady_mf6"
WARM_START_UNCONFINED_STEADY_MF6 = "unconfined_steady_mf6"
WARM_START_MODES = {
    WARM_START_ARTIFACT_INITIAL,
    WARM_START_CONFINED_STEADY_MF6,
    WARM_START_UNCONFINED_STEADY_MF6,
}


def build_seasonal_recharge(
    n_weeks: int = N_WEEKS,
    annual_depth_m: float = ANNUAL_RECHARGE_M,
    recharge_schedule_weeks: int = RECHARGE_SCHEDULE_WEEKS,
    peak_week: float | None = None,
    floor: float = 0.05,
    dt_days: float = DT_DAYS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build a winter-dominated recharge time series for transient MF6 runs.

    A single-peak cosine shape floored so summer recharge is small but
    positive, then scaled over ``recharge_schedule_weeks`` so the full schedule
    depths sum to exactly ``annual_depth_m``. Returns the first ``n_weeks``
    entries as ``(depths_m, rates_m_per_day)``; ``rate_k = depth_k / dt_days``.

    Parameters
    ----------
    annual_depth_m : total recharge depth applied over the year (m).
    recharge_schedule_weeks : number of weekly periods represented by the
        annual recharge total. Use 52 when running a shorter transient sample.
    peak_week : week index of maximum recharge on the full schedule.
    floor : minimum recharge as a fraction of the peak (summer floor).
    dt_days : days per stress period (used to convert depth -> rate).
    """
    n_weeks = int(n_weeks)
    recharge_schedule_weeks = int(recharge_schedule_weeks)
    if n_weeks < 1:
        raise ValueError("n_weeks must be positive.")
    if recharge_schedule_weeks < n_weeks:
        raise ValueError("recharge_schedule_weeks must be at least n_weeks.")
    if peak_week is None:
        peak_week = float(recharge_schedule_weeks) / 2.0

    weeks = np.arange(recharge_schedule_weeks, dtype=np.float64)
    shape = floor + (1.0 - floor) * 0.5 * (
        1.0 + np.cos(2.0 * np.pi * (weeks - float(peak_week)) / float(recharge_schedule_weeks))
    )
    schedule_depths = shape * (float(annual_depth_m) / float(shape.sum()))
    depths = schedule_depths[:n_weeks].copy()
    rates = depths / float(dt_days)
    return depths, rates


@dataclass(frozen=True)
class TransientCase:
    nx: int
    ny: int
    dx: float
    active: np.ndarray
    bc_mask: np.ndarray
    bc_values: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    hydraulic_conductivity: np.ndarray
    initial_head: np.ndarray
    sy: float
    ss: float
    n_weeks: int
    recharge_schedule_weeks: int
    dt_days: float
    steady_recharge_rate: float
    recharge_depths: np.ndarray   # (n_weeks,) m
    recharge_rates: np.ndarray    # (n_weeks,) m/day
    initial_saturated_thickness: float = DEFAULT_INIT_SAT_THICKNESS
    t_field_kind: str = T_FIELD_HOMOGENEOUS
    t_field_seed: int = DEFAULT_T_FIELD_SEED
    ghb_mask: np.ndarray | None = None
    ghb_head: np.ndarray | None = None
    ghb_width: np.ndarray | None = None
    ghb_alpha: float = 1.0
    ghb_aq_thickness: float | None = None
    ghb_head_offset: float = 2.0
    ghb_conductance_mode: str = "none"


def build_transient_unconfined_case(
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    dx: float = DEFAULT_DX,
    hydraulic_conductivity: float = DEFAULT_K,
    initial_saturated_thickness: float = DEFAULT_INIT_SAT_THICKNESS,
    sy: float = DEFAULT_SY,
    ss: float = DEFAULT_SS,
    n_weeks: int = N_WEEKS,
    annual_recharge_m: float = ANNUAL_RECHARGE_M,
    recharge_schedule_weeks: int = RECHARGE_SCHEDULE_WEEKS,
    dt_days: float = DT_DAYS,
    t_field_kind: str = DEFAULT_T_FIELD_KIND,
    t_field_seed: int = DEFAULT_T_FIELD_SEED,
    ghb_conductance_mode: str = "none",
    ghb_head_offset: float = 2.0,
    ghb_width_value: float = 1.0,
    ghb_alpha: float = 1.0,
    workspace: str | Path | None = None,
) -> TransientCase:
    """
    Build the spatial and temporal inputs for the transient unconfined model.

    The spatial grid, boundary conditions, elevations, and hydraulic
    conductivity match the steady unconfined benchmark case, but this function
    builds them directly so MF6 truth generation does not require importing
    the Warp solver. The returned ``TransientCase`` is intentionally plain
    NumPy data so it can be used both by this MF6 generator and by future Warp
    replay/comparison code without importing Flopy.
    """
    ghb_conductance_mode = str(ghb_conductance_mode).strip().lower()
    if ghb_conductance_mode not in {"none", "warp_matched", "mf6_fixed_point"}:
        raise ValueError("ghb_conductance_mode must be none, warp_matched, or mf6_fixed_point")
    if ghb_conductance_mode == "warp_matched":
        # Warp-matched conductance (C_gh evaluated at Warp's converged heads)
        # is not implemented for the transient generator.  Refuse the mode
        # rather than silently writing an initial-head-conductance artifact
        # under a misleading label; use mf6_fixed_point for independent truth.
        raise NotImplementedError(
            "ghb_conductance_mode='warp_matched' is not implemented for transient "
            "truth generation; use 'mf6_fixed_point' (independent) or 'none'."
        )
    if workspace is not None:
        Path(workspace).mkdir(parents=True, exist_ok=True)

    active = _build_domain(nx=int(nx), ny=int(ny)).astype(np.int32)
    top = np.asarray(_build_dem(active), dtype=np.float64)
    bottom = np.asarray(_model_bottom(top), dtype=np.float64)
    bc_bool = _build_dirichlet_boundary_mask(active)

    bc_mask = bc_bool.astype(np.int32)
    bc_values = np.zeros((int(ny), int(nx)), dtype=np.float64)
    bc_values[bc_bool] = top[bc_bool]

    t_field_kind = str(t_field_kind).strip().lower()
    if t_field_kind not in T_FIELD_MODES:
        raise ValueError(f"t_field_kind must be one of {sorted(T_FIELD_MODES)}.")
    if t_field_kind == T_FIELD_UGLY:
        # Hard heterogeneous T from the confined steady-state benchmarks
        # (make_ugly_T_field: lognormal correlated noise, high-T diagonal
        # channel, low-T lenses, clipped to [1, 1e5] m2/d), converted to K
        # with the benchmark convention hk = T / thickness.
        t_field = make_ugly_T_field(nx=int(nx), ny=int(ny), domain=active, seed=int(t_field_seed))
        thickness = max(float(initial_saturated_thickness), 0.1)
        k_field = np.asarray(t_field, dtype=np.float64) / thickness
    else:
        k_field = np.full((int(ny), int(nx)), float(hydraulic_conductivity), dtype=np.float64)
    k_field[active == 0] = 0.0

    initial_head = bottom + max(float(initial_saturated_thickness), 0.1)
    initial_head = np.minimum(initial_head, top)
    initial_head[bc_bool] = bc_values[bc_bool]
    initial_head[active == 0] = 0.0

    ghb_mask = np.zeros_like(active, dtype=np.int32)
    ghb_head = np.zeros_like(initial_head, dtype=np.float64)
    ghb_width = np.zeros_like(initial_head, dtype=np.float64)
    ghb_aq_thickness = float(np.nanmedian(np.maximum(top - bottom, 0.1)))
    if ghb_conductance_mode != "none":
        ghb_row = int(ny) // 2
        ghb_mask[ghb_row, 1:max(int(nx) - 1, 1)] = 1
        ghb_mask[(bc_mask != 0) | (active == 0)] = 0
        ghb_head[ghb_mask != 0] = initial_head[ghb_mask != 0] - float(ghb_head_offset)
        ghb_width[ghb_mask != 0] = float(ghb_width_value)

    depths, rates = build_seasonal_recharge(
        n_weeks=n_weeks,
        annual_depth_m=annual_recharge_m,
        recharge_schedule_weeks=recharge_schedule_weeks,
        dt_days=dt_days,
    )
    steady_recharge_rate = float(annual_recharge_m) / (float(recharge_schedule_weeks) * float(dt_days))
    return TransientCase(
        nx=int(nx),
        ny=int(ny),
        dx=float(dx),
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        top=top,
        bottom=bottom,
        hydraulic_conductivity=k_field,
        initial_head=initial_head.astype(np.float64, copy=False),
        sy=float(sy),
        ss=float(ss),
        n_weeks=int(n_weeks),
        recharge_schedule_weeks=int(recharge_schedule_weeks),
        dt_days=float(dt_days),
        steady_recharge_rate=float(steady_recharge_rate),
        recharge_depths=depths,
        recharge_rates=rates,
        initial_saturated_thickness=float(initial_saturated_thickness),
        t_field_kind=t_field_kind,
        t_field_seed=int(t_field_seed),
        ghb_mask=ghb_mask,
        ghb_head=ghb_head,
        ghb_width=ghb_width,
        ghb_alpha=float(ghb_alpha),
        ghb_aq_thickness=ghb_aq_thickness,
        ghb_head_offset=float(ghb_head_offset),
        ghb_conductance_mode=ghb_conductance_mode,
    )


def _save_compressed_npz(out_path: Path, arrays: dict[str, np.ndarray], preset: int = 9) -> None:
    """
    Save an ``np.savez`` payload through stdlib LZMA compression.

    The repository already uses ``.npz.lzma`` fixtures for large truth arrays.
    This helper keeps the transient MF6 artifact in the same lossless format
    without adding another compression dependency.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    payload = lzma.compress(buf.getvalue(), preset=preset)
    staging_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{out_path.name}.staging-",
            dir=out_path.parent,
            delete=False,
        ) as staging_file:
            staging_path = Path(staging_file.name)
            staging_file.write(payload)
            staging_file.flush()
            os.fsync(staging_file.fileno())
        os.replace(staging_path, out_path)
        staging_path = None
    finally:
        if staging_path is not None:
            staging_path.unlink(missing_ok=True)


def _mf6_normal_termination(mf6_workspace: Path) -> bool:
    """Require explicit MF6 normal-termination text, not only ``ok=True``."""
    for listing_path in (mf6_workspace / f"{MF6_MODEL_NAME}.lst", mf6_workspace / "mfsim.lst"):
        if listing_path.exists() and "normal termination" in listing_path.read_text(errors="replace").lower():
            return True
    return False


def _budget_gate_from_artifact(budget_path: Path) -> tuple[float, float]:
    """Return max absolute parsed budget discrepancy and its row tolerance."""
    with np.load(budget_path, allow_pickle=True) as budget:
        if "mf6_mass_balance_json" not in budget.files:
            raise RuntimeError(f"MF6 budget artifact {budget_path} has no parsed mass-balance rows.")
        rows = json.loads(str(np.asarray(budget["mf6_mass_balance_json"]).reshape(())))
    discrepancies = [abs(float(row["percent_discrepancy"])) for row in rows]
    if not discrepancies:
        raise RuntimeError(f"MF6 budget artifact {budget_path} has no period rows.")
    return float(max(discrepancies)), 1.0


def _validate_mf6_transient_outputs(
    *,
    mf6_workspace: Path,
    heads_per_period: np.ndarray,
    initial_head: np.ndarray,
    active: np.ndarray,
    budget_path: Path,
) -> tuple[float, float]:
    """Apply the transient truth gates before an artifact can be written."""
    if not _mf6_normal_termination(mf6_workspace):
        raise RuntimeError(f"MF6 workspace {mf6_workspace} lacks NORMAL TERMINATION.")
    if heads_per_period.ndim != 3 or heads_per_period.shape[0] < 1:
        raise RuntimeError("MF6 produced no complete saved transient periods.")
    if not all_active_transient_heads_finite(
        heads_per_period=heads_per_period,
        active=active,
    ):
        raise FloatingPointError("MF6 transient heads contain non-finite active-cell values.")
    if not np.isfinite(initial_head[active]).all():
        raise FloatingPointError("MF6 transient warm-start head contains non-finite active-cell values.")
    if float(np.max(np.abs(heads_per_period[-1][active] - initial_head[active]))) <= 1.0e-12:
        raise RuntimeError("MF6 transient response is indistinguishable from its initial condition.")
    discrepancy, tolerance = _budget_gate_from_artifact(budget_path)
    if discrepancy > tolerance:
        raise RuntimeError(f"MF6 transient budget discrepancy {discrepancy:.6g}% exceeds {tolerance:.6g}%.")
    return discrepancy, tolerance


def _load_mf6_last_head(
    *,
    hds_path: Path,
    case: TransientCase,
    label: str,
) -> np.ndarray | None:
    """
    Load a completed MF6 head file when it matches the requested case shape.
    """
    if not hds_path.exists():
        return None
    if not _mf6_normal_termination(hds_path.parent):
        print(
            f"Existing {label} warm-start workspace did not terminate normally; "
            "recomputing warm start."
        )
        return None
    heads = flopy.utils.HeadFile(str(hds_path)).get_alldata()
    if heads.size == 0:
        return None
    head = np.asarray(heads[-1, 0, :, :], dtype=np.float64)
    if head.shape != (case.ny, case.nx):
        print(
            f"Existing {label} warm-start head has shape {head.shape}; "
            f"expected {(case.ny, case.nx)}. Recomputing warm start."
        )
        return None
    head[case.active == 0] = 0.0
    head[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    if not np.all(np.isfinite(head)):
        print(f"Existing {label} warm-start head contains non-finite values. Recomputing warm start.")
        return None
    print(f"Reusing existing {label} warm-start head: {hds_path}")
    return head


def _transient_failure_message(*, mf6_ws: Path, name: str) -> str:
    """
    Include the useful MF6 listing-file failure line without requiring reruns.
    """
    listing_paths = [mf6_ws.joinpath(f"{name}.lst"), mf6_ws.joinpath("mfsim.lst")]
    markers: list[str] = []
    for path in listing_paths:
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            text = line.strip()
            if (
                "FAILED TO MEET SOLVER CONVERGENCE" in text
                or "did not converge" in text
                or "Simulation convergence failure" in text
            ):
                markers.append(f"{path.name}: {text}")
    detail = "; ".join(markers[-4:])
    return "MF6 transient run failed." if not detail else f"MF6 transient run failed: {detail}"


def _decode_budget_record_name(name: object) -> str:
    """
    Normalize an MF6 budget record name to text.

    :param name: Raw record name from Flopy.
    :return: Decoded text.
    """
    if isinstance(name, bytes):
        return name.decode("utf-8", errors="ignore").strip()
    return str(name).strip()


def _budget_array_2d(
    *,
    budget_file: "flopy.utils.CellBudgetFile",
    record_name: str,
    kstpkper: tuple[int, int],
) -> np.ndarray | None:
    """
    Read one structured-grid cell-budget record as a 2D array.

    :param budget_file: Flopy cell-budget reader.
    :param record_name: MF6 record name.
    :param kstpkper: ``(kstp, kper)`` tuple.
    :return: ``(ny, nx)`` float64 array or ``None`` when the record is absent.
    """
    try:
        data = budget_file.get_data(kstpkper=kstpkper, text=record_name, full3D=True)
    except TypeError:
        data = budget_file.get_data(kstpkper=kstpkper, text=record_name)
    if not data:
        return None
    arr = np.asarray(data[0], dtype=np.float64)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(
            f"budget record '{record_name}' at {kstpkper} has shape {arr.shape}, expected 2D structured data"
        )
    return arr.astype(np.float64, copy=False)


def save_mf6_storage_budget_terms(
    *,
    cbb_path: Path,
    out_path: Path,
    case: TransientCase,
    formulation: str,
) -> Path:
    """
    Extract MF6 storage cell-budget terms and save them as a deterministic artifact.

    :param cbb_path: MF6 cell-budget path.
    :param out_path: Output ``.npz`` path.
    :param case: Transient case metadata.
    :param formulation: ``confined`` or ``unconfined``.
    :return: Written path.
    """
    cbc = flopy.utils.CellBudgetFile(str(cbb_path), precision="double")
    unique_names = [_decode_budget_record_name(name) for name in cbc.get_unique_record_names()]
    storage_record_names = [name for name in unique_names if "STO" in name.upper() or "STORAGE" in name.upper()]
    kstpkper_list = list(cbc.get_kstpkper())
    nper = int(case.n_weeks)

    per_record_arrays: dict[str, np.ndarray] = {}
    total_storage = np.zeros((nper, case.ny, case.nx), dtype=np.float64)
    package_terms = {
        "recharge": np.zeros((nper, case.ny, case.nx), dtype=np.float64),
        "chd": np.zeros((nper, case.ny, case.nx), dtype=np.float64),
        "ghb": np.zeros((nper, case.ny, case.nx), dtype=np.float64),
        "storage": np.zeros((nper, case.ny, case.nx), dtype=np.float64),
    }

    for record_name in storage_record_names:
        record_arrays = np.zeros((nper, case.ny, case.nx), dtype=np.float64)
        for period_index in range(nper):
            arr = _budget_array_2d(
                budget_file=cbc,
                record_name=record_name,
                kstpkper=kstpkper_list[period_index],
            )
            if arr is None:
                continue
            record_arrays[period_index] = arr
            total_storage[period_index] += arr
        per_record_arrays[record_name] = record_arrays

    for record_name in unique_names:
        package_key = None
        record_name_upper = record_name.upper()
        if "RCH" in record_name_upper:
            package_key = "recharge"
        elif "CHD" in record_name_upper:
            package_key = "chd"
        elif "GHB" in record_name_upper:
            package_key = "ghb"
        elif "STO" in record_name_upper or "STORAGE" in record_name_upper:
            package_key = "storage"
        if package_key is None:
            continue
        record_arrays = np.zeros((nper, case.ny, case.nx), dtype=np.float64)
        for period_index in range(nper):
            arr = _budget_array_2d(
                budget_file=cbc,
                record_name=record_name,
                kstpkper=kstpkper_list[period_index],
            )
            if arr is None:
                continue
            record_arrays[period_index] = arr
            package_terms[package_key][period_index] += arr
        per_record_arrays.setdefault(record_name, record_arrays)

    mass_balance_rows = []
    for period_index in range(nper):
        row = {"period": int(period_index + 1)}
        total_in = 0.0
        total_out = 0.0
        for package_key, values in package_terms.items():
            period_values = np.asarray(values[period_index], dtype=np.float64)
            gross_in = float(np.sum(np.maximum(period_values, 0.0)))
            gross_out = float(np.sum(np.maximum(-period_values, 0.0)))
            row[f"{package_key}_in"] = gross_in
            row[f"{package_key}_out"] = gross_out
            total_in += gross_in
            total_out += gross_out
        in_minus_out = total_in - total_out
        denom = abs(total_in) + abs(total_out)
        row["total_in"] = float(total_in)
        row["total_out"] = float(total_out)
        row["in_minus_out"] = float(in_minus_out)
        row["percent_discrepancy"] = 0.0 if denom == 0.0 else float(100.0 * in_minus_out / denom)
        mass_balance_rows.append(row)

    out_arrays: dict[str, np.ndarray] = {
        "unique_record_names": np.asarray(unique_names, dtype=object),
        "storage_record_names": np.asarray(storage_record_names, dtype=object),
        "selected_storage_record_name": np.asarray(
            "+".join(storage_record_names) if storage_record_names else "",
            dtype=object,
        ),
        "storage_total_per_period": total_storage,
        "period_count": np.asarray(nper, dtype=np.int32),
        "dt_days": np.asarray(case.dt_days, dtype=np.float64),
        "sy": np.asarray(case.sy, dtype=np.float64),
        "ss": np.asarray(case.ss, dtype=np.float64),
        "top": np.asarray(case.top, dtype=np.float64),
        "bottom": np.asarray(case.bottom, dtype=np.float64),
        "icelltype": np.asarray([1 if formulation == FORMULATION_UNCONFINED else 0], dtype=np.int32),
        "iconvert": np.asarray(1 if formulation == FORMULATION_UNCONFINED else 0, dtype=np.int32),
        "mf6_mass_balance_json": np.asarray(json.dumps(mass_balance_rows), dtype=object),
    }
    for record_name, values in per_record_arrays.items():
        safe_name = record_name.lower().replace(" ", "_").replace("-", "_")
        out_arrays[f"record_{safe_name}_per_period"] = values
    for package_key, values in package_terms.items():
        out_arrays[f"package_{package_key}_per_period"] = values

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out_arrays)
    print(f"MF6 budget records: {unique_names}")
    print(f"MF6 storage records: {storage_record_names}")
    print(f"MF6 storage budget terms saved to {out_path}")
    return out_path


def build_confined_transmissivity(
    case: TransientCase,
    mode: str = "initial_saturated_thickness",
    min_sat: float = 0.1,
) -> np.ndarray:
    """
    Build the transmissivity convention used for confined steady warm starts.

    ``initial_saturated_thickness`` matches the unconfined pre-solve convention:
    ``K * max(initial_head-bottom, min_sat)``. ``full_thickness`` matches an MF6
    confined transient layer with ``icelltype=0``: ``K * (top-bottom)``.
    """
    mode = str(mode).strip().lower()
    if mode == "initial_saturated_thickness":
        thickness = np.maximum(case.initial_head - case.bottom, float(min_sat))
    elif mode == "full_thickness":
        thickness = np.maximum(case.top - case.bottom, float(min_sat))
    else:
        raise ValueError("mode must be 'initial_saturated_thickness' or 'full_thickness'.")
    transmissivity = np.asarray(case.hydraulic_conductivity, dtype=np.float64) * thickness
    transmissivity[case.active == 0] = 0.0
    return transmissivity.astype(np.float64, copy=False)


def run_mf6_confined_steady_warm_start(
    case: TransientCase,
    recharge_rate: float | None = None,
    mf6_workspace: str | Path | None = None,
    transmissivity_mode: str = "initial_saturated_thickness",
) -> tuple[np.ndarray, float]:
    """
    Run the matching confined steady MF6 problem used as transient warm start.

    MF6 receives a confined single-layer model with unit thickness and
    ``k = transmissivity``. That is equivalent to a fixed-transmissivity
    confined flow solve and matches the Warp warm-start convention.
    """
    mf6_ws = Path(mf6_workspace) if mf6_workspace is not None else data_store.joinpath(
        "working_tests", "mf6_transient_2d_unconfined", "mf6_confined_steady_warm_start"
    )
    mf6_ws.mkdir(parents=True, exist_ok=True)

    recharge_rate = (
        float(case.steady_recharge_rate)
        if recharge_rate is None
        else float(recharge_rate)
    )
    transmissivity = build_confined_transmissivity(
        case=case,
        mode=transmissivity_mode,
    )

    name = "tr2d_c_ss"
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=str(require_mf6()),
        version="mf6",
        sim_ws=str(mf6_ws),
    )
    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    )
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=f"{name}.nam",
        save_flows=True,
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        # The hard heterogeneous T field needs the same robust iteration
        # budget as the production transient solve.  The older 100/300 caps
        # caused this steady warm start to terminate after exhausting both
        # iteration limits without converging.
        outer_maximum=500,
        outer_dvclose=1.0e-6,
        inner_maximum=500,
        inner_dvclose=1.0e-8,
        rcloserecord=[1.0e-6, "RELATIVE_RCLOSE"],
        scaling_method="DIAGONAL",
    )
    sim.register_ims_package(ims, [gwf.name])

    confined_top = np.ones((case.ny, case.nx), dtype=np.float64)
    confined_bottom = np.zeros((case.ny, case.nx), dtype=np.float64)
    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=1,
        nrow=case.ny,
        ncol=case.nx,
        delr=case.dx,
        delc=case.dx,
        top=confined_top,
        botm=confined_bottom,
        idomain=case.active,
    )
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=case.initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=[0],
        k=transmissivity,
        k33=transmissivity,
        k33overk=False,
        save_specific_discharge=True,
        save_flows=True,
    )

    fixed_head_cells = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
    fixed_head_cells[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    chd_spd = _create_chd_single_period(boundary_heads=fixed_head_cells, active=case.active)
    flopy.mf6.ModflowGwfchd(gwf, pname="chd", stress_period_data=chd_spd, save_flows=True)

    recharge = np.full((case.ny, case.nx), recharge_rate, dtype=np.float64)
    recharge[case.active == 0] = 0.0
    flopy.mf6.ModflowGwfrcha(gwf, pname="recharge", recharge=recharge, save_flows=True)
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "LAST")],
        head_filerecord=[f"{name}.hds"],
        budget_filerecord=[f"{name}.cbb"],
        printrecord=[],
    )

    t_engine_start = time.perf_counter()
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True, report=False)
    engine_time = time.perf_counter() - t_engine_start
    if not ok:
        raise RuntimeError("MF6 confined steady warm-start run failed.")

    hds_path = mf6_ws.joinpath(f"{name}.hds")
    heads = flopy.utils.HeadFile(str(hds_path)).get_alldata()
    confined_head = np.asarray(heads[-1, 0, :, :], dtype=np.float64)
    confined_head[case.active == 0] = 0.0
    confined_head[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    if not np.all(np.isfinite(confined_head)):
        raise FloatingPointError("MF6 confined steady warm-start head contains non-finite values.")
    return confined_head, float(engine_time)


def run_mf6_unconfined_steady_warm_start(
    case: TransientCase,
    recharge_rate: float | None = None,
    mf6_workspace: str | Path | None = None,
) -> tuple[np.ndarray, float]:
    """
    Run a steady unconfined MF6 solve used as the transient initial condition.
    """
    mf6_ws = Path(mf6_workspace) if mf6_workspace is not None else data_store.joinpath(
        "working_tests", "mf6_transient_2d_unconfined", "mf6_unconfined_steady_warm_start"
    )
    mf6_ws.mkdir(parents=True, exist_ok=True)

    recharge_rate = (
        float(case.steady_recharge_rate)
        if recharge_rate is None
        else float(recharge_rate)
    )

    name = "tr2d_u_ss"
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=str(require_mf6()),
        version="mf6",
        sim_ws=str(mf6_ws),
    )
    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=1,
        perioddata=[(1.0, 1, 1.0)],
    )
    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=f"{name}.nam",
        save_flows=True,
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        outer_maximum=300,
        outer_dvclose=1.0e-7,
        inner_maximum=500,
        inner_dvclose=1.0e-9,
        rcloserecord=[1.0e-7, "RELATIVE_RCLOSE"],
        scaling_method="DIAGONAL",
    )
    sim.register_ims_package(ims, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=1,
        nrow=case.ny,
        ncol=case.nx,
        delr=case.dx,
        delc=case.dx,
        top=case.top,
        botm=case.bottom,
        idomain=case.active,
    )
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=case.initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=[1],
        k=case.hydraulic_conductivity,
        k33=case.hydraulic_conductivity,
        k33overk=False,
        save_specific_discharge=True,
        save_saturation=True,
        save_flows=True,
    )

    fixed_head_cells = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
    fixed_head_cells[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    chd_spd = _create_chd_single_period(boundary_heads=fixed_head_cells, active=case.active)
    flopy.mf6.ModflowGwfchd(gwf, pname="chd", stress_period_data=chd_spd, save_flows=True)

    recharge = np.full((case.ny, case.nx), recharge_rate, dtype=np.float64)
    recharge[case.active == 0] = 0.0
    flopy.mf6.ModflowGwfrcha(gwf, pname="recharge", recharge=recharge)
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "LAST")],
        head_filerecord=[f"{name}.hds"],
        budget_filerecord=[f"{name}.cbb"],
        printrecord=[],
    )

    t_engine_start = time.perf_counter()
    sim.write_simulation(silent=True)
    ok, _ = sim.run_simulation(silent=True, report=False)
    engine_time = time.perf_counter() - t_engine_start
    if not ok:
        raise RuntimeError("MF6 unconfined steady warm-start run failed.")

    hds_path = mf6_ws.joinpath(f"{name}.hds")
    heads = flopy.utils.HeadFile(str(hds_path)).get_alldata()
    unconfined_head = np.asarray(heads[-1, 0, :, :], dtype=np.float64)
    unconfined_head[case.active == 0] = 0.0
    unconfined_head[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    if not np.all(np.isfinite(unconfined_head)):
        raise FloatingPointError("MF6 unconfined steady warm-start head contains non-finite values.")
    return unconfined_head, float(engine_time)


def run_mf6_transient(
    case: TransientCase,
    out_path: str | Path | None = None,
    mf6_workspace: str | Path | None = None,
    warm_start_workspace: str | Path | None = None,
    reuse_existing_warm_start: bool = False,
    warm_start_mode: str = WARM_START_CONFINED_STEADY_MF6,
    confined_steady_head: np.ndarray | None = None,
    unconfined_steady_head: np.ndarray | None = None,
    formulation: str = FORMULATION_UNCONFINED,
) -> Path:
    """
    Run MODFLOW 6 for one transient case and write the truth artifact.

    Parameters
    ----------
    case:
        Complete spatial, boundary, storage, and recharge inputs for the model.
    out_path:
        Optional compressed ``.npz.lzma`` output path. When omitted, the
        artifact is written under
        ``data/working_tests/mf6_transient_2d_<formulation>``.
    mf6_workspace:
        Optional directory for MF6 input/output files. When omitted, a sibling
        ``mf6`` directory next to ``out_path`` is used.

    Returns
    -------
    Path
        Path to the compressed truth artifact containing per-period MF6 heads
        and all replay inputs.
    """
    formulation = str(formulation).strip().lower()
    if formulation not in FORMULATION_MODES:
        raise ValueError(f"formulation must be one of {sorted(FORMULATION_MODES)}.")
    if out_path is None:
        out_path = data_store.joinpath(
            "working_tests", f"mf6_transient_2d_{formulation}", "mf6_transient_heads.npz.lzma"
        )
    out_path = Path(out_path)
    mf6_ws = Path(mf6_workspace) if mf6_workspace is not None else out_path.parent.joinpath("mf6")
    mf6_ws.mkdir(parents=True, exist_ok=True)
    warm_start_ws = Path(warm_start_workspace) if warm_start_workspace is not None else out_path.parent

    warm_start_mode = str(warm_start_mode).strip().lower()
    if warm_start_mode not in WARM_START_MODES:
        raise ValueError(f"warm_start_mode must be one of {sorted(WARM_START_MODES)}.")
    confined_steady_engine_time = float("nan")
    unconfined_steady_engine_time = float("nan")
    warm_start_transmissivity_mode = "not_used"
    if warm_start_mode == WARM_START_CONFINED_STEADY_MF6:
        confined_workspace = warm_start_ws.joinpath("mf6_confined_steady_warm_start")
        if confined_steady_head is None:
            if bool(reuse_existing_warm_start):
                confined_steady_head = _load_mf6_last_head(
                    hds_path=confined_workspace.joinpath("tr2d_c_ss.hds"),
                    case=case,
                    label="confined steady",
                )
                if confined_steady_head is not None:
                    confined_steady_engine_time = 0.0
                    warm_start_transmissivity_mode = "reused_existing"
        if confined_steady_head is None:
            warm_start_transmissivity_mode = (
                "full_thickness"
                if formulation == FORMULATION_CONFINED
                else "initial_saturated_thickness"
            )
            confined_steady_head, confined_steady_engine_time = run_mf6_confined_steady_warm_start(
                case=case,
                recharge_rate=float(case.steady_recharge_rate),
                mf6_workspace=confined_workspace,
                transmissivity_mode=warm_start_transmissivity_mode,
            )
        else:
            confined_steady_head = np.asarray(confined_steady_head, dtype=np.float64)
            if warm_start_transmissivity_mode == "not_used":
                warm_start_transmissivity_mode = "provided"
        if confined_steady_head.shape != (case.ny, case.nx):
            raise ValueError(
                f"confined_steady_head shape {confined_steady_head.shape} expected {(case.ny, case.nx)}."
            )
        if not np.all(np.isfinite(confined_steady_head)):
            raise ValueError("confined_steady_head must be finite.")
        transient_initial_head = confined_steady_head.copy()
        unconfined_steady_head = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
    elif warm_start_mode == WARM_START_UNCONFINED_STEADY_MF6:
        if formulation != FORMULATION_UNCONFINED:
            raise ValueError("warm_start_mode='unconfined_steady_mf6' requires formulation='unconfined'.")
        unconfined_workspace = warm_start_ws.joinpath("mf6_unconfined_steady_warm_start")
        if unconfined_steady_head is None:
            if bool(reuse_existing_warm_start):
                unconfined_steady_head = _load_mf6_last_head(
                    hds_path=unconfined_workspace.joinpath("tr2d_u_ss.hds"),
                    case=case,
                    label="unconfined steady",
                )
                if unconfined_steady_head is not None:
                    unconfined_steady_engine_time = 0.0
            if unconfined_steady_head is None:
                unconfined_steady_head, unconfined_steady_engine_time = run_mf6_unconfined_steady_warm_start(
                    case=case,
                    recharge_rate=float(case.steady_recharge_rate),
                    mf6_workspace=unconfined_workspace,
                )
        else:
            unconfined_steady_head = np.asarray(unconfined_steady_head, dtype=np.float64)
        if unconfined_steady_head.shape != (case.ny, case.nx):
            raise ValueError(
                f"unconfined_steady_head shape {unconfined_steady_head.shape} expected {(case.ny, case.nx)}."
            )
        if not np.all(np.isfinite(unconfined_steady_head)):
            raise ValueError("unconfined_steady_head must be finite.")
        transient_initial_head = unconfined_steady_head.copy()
        confined_steady_head = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
        warm_start_transmissivity_mode = "not_used"
    else:
        transient_initial_head = np.asarray(case.initial_head, dtype=np.float64).copy()
        confined_steady_head = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
        unconfined_steady_head = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
        warm_start_transmissivity_mode = "not_used"

    transient_initial_head[case.active == 0] = 0.0
    transient_initial_head[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]

    name = MF6_MODEL_NAME
    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=str(require_mf6()),
        version="mf6",
        sim_ws=str(mf6_ws),
    )
    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=case.n_weeks,
        perioddata=[(case.dt_days, 1, 1.0)] * case.n_weeks,
    )
    gwf = flopy.mf6.ModflowGwf(
        sim, modelname=name, model_nam_file=f"{name}.nam", save_flows=True
    )
    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="COMPLEX",
        linear_acceleration="BICGSTAB",
        outer_maximum=150,
        outer_dvclose=1.0e-6,
        inner_maximum=500,
        inner_dvclose=1.0e-8,
        rcloserecord=[1.0e-6, "RELATIVE_RCLOSE"],
        scaling_method="DIAGONAL",
    )
    sim.register_ims_package(ims, [gwf.name])

    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=1,
        nrow=case.ny,
        ncol=case.nx,
        delr=case.dx,
        delc=case.dx,
        top=case.top,
        botm=case.bottom,
        idomain=case.active,
    )
    flopy.mf6.ModflowGwfic(gwf, pname="ic", strt=transient_initial_head)
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=[1 if formulation == FORMULATION_UNCONFINED else 0],
        k=case.hydraulic_conductivity,
        k33=case.hydraulic_conductivity,
        k33overk=False,
        save_specific_discharge=True,
        save_saturation=True,
        save_flows=True,
    )
    flopy.mf6.ModflowGwfsto(
        gwf,
        pname="sto",
        ss=case.ss,
        sy=(case.sy if formulation == FORMULATION_UNCONFINED else 0.0),
        iconvert=(1 if formulation == FORMULATION_UNCONFINED else 0),
        transient={0: True},                 # all periods transient (inherited)
        save_flows=True,
    )

    fixed_head_cells = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
    fixed_head_cells[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    chd_spd = _create_chd_single_period(boundary_heads=fixed_head_cells, active=case.active)
    flopy.mf6.ModflowGwfchd(gwf, pname="chd", stress_period_data=chd_spd, save_flows=True)

    ghb_package = None
    ghb_mask = np.zeros((case.ny, case.nx), dtype=np.int32) if case.ghb_mask is None else np.asarray(case.ghb_mask, dtype=np.int32)
    ghb_head = np.zeros((case.ny, case.nx), dtype=np.float64) if case.ghb_head is None else np.asarray(case.ghb_head, dtype=np.float64)
    ghb_width = np.zeros((case.ny, case.nx), dtype=np.float64) if case.ghb_width is None else np.asarray(case.ghb_width, dtype=np.float64)
    ghb_factor = np.zeros_like(ghb_width, dtype=np.float64)
    ghb_factor[ghb_mask != 0] = (
        float(case.ghb_alpha) * ghb_width[ghb_mask != 0] * float(case.dx)
        / max(float(case.ghb_aq_thickness or np.nanmedian(np.maximum(case.top - case.bottom, 0.1))), 1.0e-12)
    )
    ghb_conductance = np.zeros((case.n_weeks, case.ny, case.nx), dtype=np.float64)
    if np.any(ghb_mask != 0):
        initial_sat = np.clip(transient_initial_head - case.bottom, 0.1, np.maximum(case.top - case.bottom, 0.1))
        ghb_conductance[:] = case.hydraulic_conductivity[None, :, :] * initial_sat[None, :, :] * ghb_factor[None, :, :]
        ghb_records = {
            period: [((0, int(j), int(i)), float(ghb_head[j, i]), float(ghb_conductance[period, j, i]))
                     for j, i in zip(*np.where(ghb_mask != 0))]
            for period in range(case.n_weeks)
        }
        ghb_package = flopy.mf6.ModflowGwfghb(
            gwf, pname="ghb", stress_period_data=ghb_records, save_flows=True
        )

    # Per-period recharge: uniform-in-space, seasonal-in-time. MF6 reuses the
    # previous period's CHD; recharge is specified explicitly for every period.
    recharge_spd = {
        per: np.full((case.ny, case.nx), float(case.recharge_rates[per]), dtype=np.float64)
        for per in range(case.n_weeks)
    }
    flopy.mf6.ModflowGwfrcha(gwf, pname="recharge", recharge=recharge_spd, save_flows=True)

    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
        head_filerecord=[f"{name}.hds"],
        budget_filerecord=[f"{name}.cbb"],
        printrecord=[],
    )

    t_total_start = time.perf_counter()
    sim.write_simulation(silent=True)
    t_engine_start = time.perf_counter()
    ok, _ = sim.run_simulation(silent=True, report=False)
    engine_time = time.perf_counter() - t_engine_start
    total_time = time.perf_counter() - t_total_start
    if not ok:
        raise RuntimeError(_transient_failure_message(mf6_ws=mf6_ws, name=name))

    hds_path = mf6_ws.joinpath(f"{name}.hds")
    heads_per_period = np.asarray(
        flopy.utils.HeadFile(str(hds_path)).get_alldata(mflay=0),
        dtype=np.float64,
    )  # (n_weeks, ny, nx)
    fixed_point_iterations = 1
    fixed_point_head_change = 0.0
    fixed_point_conductance_change = 0.0
    fixed_point_converged = case.ghb_conductance_mode != "mf6_fixed_point"
    if ghb_package is not None and case.ghb_conductance_mode == "mf6_fixed_point":
        for iteration in range(1, 51):
            sat_endpoint = np.clip(
                heads_per_period - case.bottom[None, :, :],
                0.1,
                np.maximum(case.top - case.bottom, 0.1)[None, :, :],
            )
            updated_conductance = case.hydraulic_conductivity[None, :, :] * sat_endpoint * ghb_factor[None, :, :]
            conductance_change = float(np.max(np.abs(updated_conductance - ghb_conductance)))
            ghb_conductance = updated_conductance
            ghb_records = {
                period: [((0, int(j), int(i)), float(ghb_head[j, i]), float(ghb_conductance[period, j, i]))
                         for j, i in zip(*np.where(ghb_mask != 0))]
                for period in range(case.n_weeks)
            }
            ghb_package.stress_period_data.set_data(ghb_records)
            sim.write_simulation(silent=True)
            ok, _ = sim.run_simulation(silent=True, report=False)
            if not ok:
                raise RuntimeError("MF6 GHB fixed-point iteration failed")
            updated_heads = np.asarray(
                flopy.utils.HeadFile(str(hds_path)).get_alldata(mflay=0),
                dtype=np.float64,
            )
            head_change = float(np.max(np.abs(updated_heads - heads_per_period)[:, ghb_mask != 0]))
            heads_per_period = updated_heads
            fixed_point_iterations = iteration + 1
            fixed_point_head_change = head_change
            fixed_point_conductance_change = conductance_change
            conductance_scale = max(float(np.max(np.abs(ghb_conductance))), 1.0)
            if head_change <= 1.0e-6 and conductance_change <= max(1.0e-6, 1.0e-6 * conductance_scale):
                fixed_point_converged = True
                break
    cbb_path = mf6_ws.joinpath(f"{name}.cbb")
    mf6_storage_budget_path = out_path.with_name("mf6_storage_budget_terms.npz")
    save_mf6_storage_budget_terms(
        cbb_path=cbb_path,
        out_path=mf6_storage_budget_path,
        case=case,
        formulation=formulation,
    )
    budget_discrepancy, budget_tolerance = _validate_mf6_transient_outputs(
        mf6_workspace=mf6_ws,
        heads_per_period=heads_per_period,
        initial_head=transient_initial_head,
        active=case.active,
        budget_path=mf6_storage_budget_path,
    )

    annual_recharge_schedule_m = float(case.steady_recharge_rate) * float(case.recharge_schedule_weeks) * float(case.dt_days)
    arrays = {
        "heads_per_period": heads_per_period,
        "heads_final": heads_per_period[-1],
        "initial_head": np.asarray(transient_initial_head, dtype=np.float64),
        "raw_initial_head": np.asarray(case.initial_head, dtype=np.float64),
        "confined_steady_head": np.asarray(confined_steady_head, dtype=np.float64),
        "unconfined_steady_head": np.asarray(unconfined_steady_head, dtype=np.float64),
        "active": np.asarray(case.active, dtype=np.int32),
        "bc_mask": np.asarray(case.bc_mask, dtype=np.int32),
        "bc_values": np.asarray(case.bc_values, dtype=np.float64),
        "top": np.asarray(case.top, dtype=np.float64),
        "bottom": np.asarray(case.bottom, dtype=np.float64),
        "k_field": np.asarray(case.hydraulic_conductivity, dtype=np.float64),
        "recharge_depths": np.asarray(case.recharge_depths, dtype=np.float64),
        "recharge_rates": np.asarray(case.recharge_rates, dtype=np.float64),
        "recharge_schedule_weeks": np.asarray(case.recharge_schedule_weeks, dtype=np.int32),
        "steady_recharge_rate": np.asarray(case.steady_recharge_rate, dtype=np.float64),
        "sy": np.asarray(case.sy, dtype=np.float64),
        "ss": np.asarray(case.ss, dtype=np.float64),
        "formulation": np.asarray(formulation),
        "nx": np.asarray(case.nx, dtype=np.int32),
        "ny": np.asarray(case.ny, dtype=np.int32),
        "dx": np.asarray(case.dx, dtype=np.float64),
        "n_weeks": np.asarray(case.n_weeks, dtype=np.int32),
        "n_periods": np.asarray(case.n_weeks, dtype=np.int32),
        "dt_days": np.asarray(case.dt_days, dtype=np.float64),
        "initial_saturated_thickness": np.asarray(case.initial_saturated_thickness, dtype=np.float64),
        "t_field_kind": np.asarray(case.t_field_kind),
        "t_field_seed": np.asarray(case.t_field_seed, dtype=np.int32),
        "warm_start_mode": np.asarray(warm_start_mode),
        "ghb_mask": np.asarray(ghb_mask, dtype=np.int32),
        "ghb_head": np.asarray(ghb_head, dtype=np.float64),
        "ghb_width": np.asarray(ghb_width, dtype=np.float64),
        "ghb_alpha": np.asarray(case.ghb_alpha, dtype=np.float64),
        "ghb_aq_thickness": np.asarray(case.ghb_aq_thickness or 0.0, dtype=np.float64),
        "ghb_conductance": np.asarray(ghb_conductance, dtype=np.float64),
        "ghb_conductance_mode": np.asarray(case.ghb_conductance_mode),
        "ghb_conductance_settings": np.asarray(json.dumps({
            "head_offset": float(case.ghb_head_offset),
            "fixed_point_iterations": fixed_point_iterations,
            "terminal_head_change": fixed_point_head_change,
            "terminal_conductance_change": fixed_point_conductance_change,
            "converged": fixed_point_converged,
        }, sort_keys=True)),
        "mf6_normal_termination": np.asarray(1, dtype=np.int8),
        "mf6_budget_discrepancy_max": np.asarray(budget_discrepancy, dtype=np.float64),
        "mf6_budget_discrepancy_tol": np.asarray(budget_tolerance, dtype=np.float64),
        "ghb_fixed_point_converged": np.asarray(int(fixed_point_converged), dtype=np.int8),
        "artifact_schema_version": np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int32),
        "engine_time": np.asarray(engine_time, dtype=np.float64),
        "confined_steady_engine_time": np.asarray(confined_steady_engine_time, dtype=np.float64),
        "unconfined_steady_engine_time": np.asarray(unconfined_steady_engine_time, dtype=np.float64),
        "confined_steady_transmissivity_mode": np.asarray(warm_start_transmissivity_mode),
        "total_time": np.asarray(total_time, dtype=np.float64),
        "mf6_storage_budget_artifact": np.asarray(str(mf6_storage_budget_path)),
        "provenance": np.asarray(json.dumps({
            "kind": f"2d_{formulation}_transient_mf6_truth",
            "formulation": formulation,
            "warm_start_mode": warm_start_mode,
            "confined_steady_transmissivity_mode": warm_start_transmissivity_mode,
            "n_weeks": case.n_weeks,
            "recharge_schedule_weeks": case.recharge_schedule_weeks,
            "dt_days": case.dt_days,
            "annual_recharge_m": float(annual_recharge_schedule_m),
            "simulated_recharge_m": float(case.recharge_depths.sum()),
            "steady_recharge_rate": float(case.steady_recharge_rate),
            "sy": case.sy,
            "ss": case.ss,
            "t_field_kind": case.t_field_kind,
            "t_field_seed": case.t_field_seed,
            "hydraulic_conductivity": case.hydraulic_conductivity.flat[0],
            "initial_saturated_thickness": case.initial_saturated_thickness,
            "generator": "working_tests/run_2d_transient_vs_mf6.py",
        }, default=str)),
    }
    arrays["case_fingerprint"] = np.asarray(compute_transient_case_fingerprint(arrays))
    _save_compressed_npz(out_path, arrays)

    simulated_recharge = float(case.recharge_depths.sum())
    print(f"MF6 transient heads saved to {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
    print(
        f"  n_weeks={case.n_weeks} dt={case.dt_days}d  simulated recharge={simulated_recharge:.4f} m "
        f"from {case.recharge_schedule_weeks}-week annual schedule target {annual_recharge_schedule_m:.4f}; "
        f"steady recharge rate={case.steady_recharge_rate:.8g} m/d; Sy={case.sy} Ss={case.ss}"
    )
    print(f"  formulation: {formulation}")
    print(f"  warm start: {warm_start_mode}")
    print(f"  MF6 total time: {total_time:.2f}s (engine {engine_time:.2f}s)")
    finite = np.isfinite(heads_per_period)
    print(
        f"  heads per-period shape={heads_per_period.shape} "
        f"range=[{np.nanmin(heads_per_period):.3f}, {np.nanmax(heads_per_period):.3f}] m "
        f"(all finite={bool(finite.all())})"
    )
    return out_path


def main(
    nx: int = DEFAULT_NX,
    ny: int = DEFAULT_NY,
    dx: float = DEFAULT_DX,
    hydraulic_conductivity: float = DEFAULT_K,
    sy: float = DEFAULT_SY,
    ss: float = DEFAULT_SS,
    n_weeks: int = N_WEEKS,
    annual_recharge_m: float = ANNUAL_RECHARGE_M,
    recharge_schedule_weeks: int = RECHARGE_SCHEDULE_WEEKS,
    initial_saturated_thickness: float = DEFAULT_INIT_SAT_THICKNESS,
    t_field_kind: str = DEFAULT_T_FIELD_KIND,
    t_field_seed: int = DEFAULT_T_FIELD_SEED,
    ghb_conductance_mode: str = "none",
    out_path: str | Path | None = None,
    mf6_workspace: str | Path | None = None,
    warm_start_workspace: str | Path | None = None,
    reuse_existing_warm_start: bool = False,
    warm_start_mode: str = WARM_START_UNCONFINED_STEADY_MF6,
    formulation: str = FORMULATION_UNCONFINED,
) -> Path:
    """
    Build the default transient MF6 case and write its truth file.

    Parameters mirror the user-adjustable case settings in the module defaults.
    The returned path is the compressed truth artifact created by
    ``run_mf6_transient``.
    """
    formulation = str(formulation).strip().lower()
    if formulation not in FORMULATION_MODES:
        raise ValueError(f"formulation must be one of {sorted(FORMULATION_MODES)}.")
    case = build_transient_unconfined_case(
        nx=nx,
        ny=ny,
        dx=dx,
        hydraulic_conductivity=hydraulic_conductivity,
        initial_saturated_thickness=initial_saturated_thickness,
        sy=sy,
        ss=ss,
        n_weeks=n_weeks,
        annual_recharge_m=annual_recharge_m,
        recharge_schedule_weeks=recharge_schedule_weeks,
        t_field_kind=t_field_kind,
        t_field_seed=t_field_seed,
        ghb_conductance_mode=ghb_conductance_mode,
    )
    k_desc = (
        f"ugly_t(seed={t_field_seed}) K=T/{initial_saturated_thickness:g}"
        if str(t_field_kind).strip().lower() == T_FIELD_UGLY
        else f"K={hydraulic_conductivity}"
    )
    print(f"Transient 2D {formulation} MF6 truth: {nx}x{ny}, {n_weeks} weekly steps, {k_desc}")
    return run_mf6_transient(
        case=case,
        out_path=out_path,
        mf6_workspace=mf6_workspace,
        warm_start_workspace=warm_start_workspace,
        reuse_existing_warm_start=reuse_existing_warm_start,
        warm_start_mode=warm_start_mode,
        formulation=formulation,
    )


if __name__ == "__main__":
    # The replay script owns the case setup (single source of truth): pull it
    # and generate exactly the artifact the replay expects. Standalone usage
    # with explicit parameters remains available via ``main(...)`` above.
    from working_tests.run_2d_transient_warp_replay import (
        build_case_setup,
        ensure_case_artifact,
    )

    case_setup = build_case_setup()
    print(
        f"Generator pulled case setup from run_2d_transient_warp_replay: "
        f"{case_setup['nx']}x{case_setup['ny']} {case_setup['n_periods']}w "
        f"t_field={case_setup['t_field_kind']}(seed={case_setup['t_field_seed']})"
    )
    ensure_case_artifact(case_setup)
