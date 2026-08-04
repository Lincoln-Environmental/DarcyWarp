from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["DARCY_FLOAT"] = "float64"
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import flopy  # noqa: E402

from DARCY_WARP_PACKAGE.model_builder import (  # noqa: E402
    _build_dem,
    _build_dirichlet_boundary_mask,
    _build_domain,
    _build_ghb_boundary_masks,
    _create_chd_single_period,
    _model_bottom,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.project_base import data_store, require_mf6  # noqa: E402
from DARCY_WARP_PACKAGE.sanity_case_config import GRID_CASES  # noqa: E402
from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver  # noqa: E402


DEFAULT_DH_TOL = 1.0e-4
DEFAULT_RESIDUAL_FLOOR_TOL = 1.0e-4
DEFAULT_MF6_AGREEMENT_TOL = 5.0e-4
DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN = 1.0e-4
# Single source of truth for benchmark grids. Preserve the insertion order in
# sanity_case_config.py so commenting/uncommenting entries there directly
# controls this runner, including rectangular cases.
BENCHMARK_GRID_SIZES = tuple(
    (int(case["nx"]), int(case["ny"]))
    for case in GRID_CASES.values()
)

# Artifact schema version.  Bump whenever the set of fields written to the
# MF6/Warp NPZ artifacts changes in a way that invalidates older caches.
ARTIFACT_SCHEMA_VERSION = 3

# MF6 trust gates (Task: "trustworthy MF6").  MF6 can return ok=True with a
# ~200 % budget discrepancy and mostly-initial-condition heads on hard-T
# fields (see the comment in _build_and_run_mf6), so every MF6 run is gated
# on (a) normal termination, (b) finite heads, (c) the max absolute
# "PERCENT BUDGET DISCREPANCY" from the .lst files, and (d) nontrivial head
# movement from the initial condition.
DEFAULT_MF6_BUDGET_DISCREPANCY_TOL = 1.0  # percent
DEFAULT_MF6_HEAD_CHANGE_MIN = 1.0e-3  # m; set to 0/None to skip (near-initial cases)

# MF6-side GHB conductance fixed point (never uses Warp heads).
DEFAULT_GHB_COND_RTOL = 1.0e-6  # max relative conductance change
DEFAULT_GHB_HEAD_ATOL = 1.0e-6  # m, max absolute head change
# ~20 per the design; 30 leaves headroom for the Aitken-accelerated weak-GHB
# draining case (contraction ~0.85/iteration before acceleration).
DEFAULT_GHB_MAX_FIXED_POINT_ITERATIONS = 30


@dataclass(frozen=True)
class Unconfined2DCase:
    nx: int
    ny: int
    dx: float
    workspace: Path
    active: np.ndarray
    bc_mask: np.ndarray
    bc_values: np.ndarray
    top: np.ndarray
    bottom: np.ndarray
    hydraulic_conductivity: np.ndarray
    recharge: np.ndarray
    initial_head: np.ndarray
    # GHB state (zeros when use_ghb=False) and case-option metadata.
    gh_mask: np.ndarray | None = None
    gh_head: np.ndarray | None = None
    gh_width: np.ndarray | None = None
    t_field_kind: str = "uniform"
    t_field_seed: int = 42
    use_ghb: bool = False
    ghb_width: float = 100.0
    # How the MF6 GHB conductance is determined: "fixed_point" (MF6-side
    # fixed point of the head-dependent law; independent of Warp) or
    # "warp_matched" (C_gh evaluated at Warp's converged head; a deliberate
    # cross-solver equation-equivalence test, not an independent truth).
    ghb_conductance_mode: str = "warp_matched"


def _warp_device(preferred: str = "cuda:0") -> str:
    import warp as wp

    if preferred != "auto":
        return preferred
    try:
        return "cuda:0" if wp.is_cuda_available() else "cpu"
    except AttributeError:
        return "cuda:0"


def _finite_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _load_npz_json(npz_path: Path, name: str) -> dict | list:
    if not npz_path.exists():
        return {}
    with np.load(npz_path, allow_pickle=False) as data:
        if name not in data:
            return {}
        raw = str(np.asarray(data[name]).reshape(()))
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw_info": raw}


def _load_npz_scalar(npz_path: Path, name: str, default: float | None = None) -> float | None:
    if not npz_path.exists():
        return default
    with np.load(npz_path, allow_pickle=False) as data:
        if name not in data:
            return default
        return float(np.asarray(data[name]).reshape(()))


def _load_npz_str(npz_path: Path, name: str, default: str | None = None) -> str | None:
    if not npz_path.exists():
        return default
    with np.load(npz_path, allow_pickle=False) as data:
        if name not in data:
            return default
        return str(np.asarray(data[name]).reshape(()))


def _format_optional_float(value: object, spec: str, missing: str = "n/a") -> str:
    value = _finite_float(value)
    if value is None:
        return missing
    return format(value, spec)


def case_fingerprint(case: Unconfined2DCase) -> str:
    """
    Deterministic SHA-256 fingerprint of everything that defines the case.

    Covers the artifact schema version, grid (nx/ny/dx), all physical and
    boundary fields (active/bc masks+values, top/bottom, K field, recharge,
    initial head — i.e. initial-head provenance) and the GHB fields/options
    plus the t_field_kind/seed metadata.  Cached artifacts are only reused
    when their stored fingerprint matches this value.
    """
    h = hashlib.sha256()
    h.update(f"schema={ARTIFACT_SCHEMA_VERSION}\n".encode())
    scalars = {
        "nx": int(case.nx),
        "ny": int(case.ny),
        "dx": float(case.dx),
        "t_field_kind": str(case.t_field_kind),
        "t_field_seed": int(case.t_field_seed),
        "use_ghb": bool(case.use_ghb),
        "ghb_width": float(case.ghb_width),
        "ghb_conductance_mode": str(getattr(case, "ghb_conductance_mode", "fixed_point")),
    }
    h.update(json.dumps(scalars, sort_keys=True).encode())
    for name in (
        "active",
        "bc_mask",
        "bc_values",
        "top",
        "bottom",
        "hydraulic_conductivity",
        "recharge",
        "initial_head",
        "gh_mask",
        "gh_head",
        "gh_width",
    ):
        arr = np.ascontiguousarray(np.asarray(getattr(case, name)))
        h.update(name.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(arr.tobytes())
    return h.hexdigest()


def _validate_artifact(
    npz_path: str | Path,
    case: Unconfined2DCase,
    raise_on_mismatch: bool = False,
    kind: str = "MF6",
) -> bool:
    """
    Check that a cached NPZ artifact carries the current schema version and
    the case fingerprint.  Returns True on a match.  On mismatch either
    returns False (caller may regenerate) or raises when
    ``raise_on_mismatch`` is set (caller cannot regenerate / is about to
    reuse or copy the artifact).
    """
    path = Path(npz_path)
    if not path.exists():
        if raise_on_mismatch:
            raise FileNotFoundError(f"{kind} artifact {path} does not exist.")
        return False
    stored_version = _load_npz_scalar(path, "schema_version")
    stored_fingerprint = _load_npz_str(path, "case_fingerprint")
    ok = (
        stored_version is not None
        and int(stored_version) == int(ARTIFACT_SCHEMA_VERSION)
        and stored_fingerprint is not None
        and stored_fingerprint == case_fingerprint(case)
    )
    if not ok and raise_on_mismatch:
        raise RuntimeError(
            f"{kind} artifact {path} does not match the current case "
            f"(stored schema_version={stored_version}, "
            f"fingerprint match={stored_fingerprint == case_fingerprint(case)}). "
            "Refusing to reuse/copy a stale artifact; regenerate it first."
        )
    return ok


def validate_mf6_artifact(
    npz_path: str | Path,
    case: Unconfined2DCase,
    raise_on_mismatch: bool = False,
) -> bool:
    """Fingerprint/schema validation for cached MF6 head artifacts."""
    return _validate_artifact(npz_path, case, raise_on_mismatch=raise_on_mismatch, kind="MF6")


def validate_warp_artifact(
    npz_path: str | Path,
    case: Unconfined2DCase,
    raise_on_mismatch: bool = False,
) -> bool:
    """Fingerprint/schema validation for cached Warp head artifacts."""
    return _validate_artifact(npz_path, case, raise_on_mismatch=raise_on_mismatch, kind="Warp")


_BUDGET_DISCREPANCY_RE = re.compile(
    r"PERCENT DISCREPANCY\s*=?\s*([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)"
)


def _parse_mf6_lst(mf6_ws: str | Path) -> dict:
    """
    Parse all MF6 ``.lst`` files in ``mf6_ws``.

    :return: dict with ``budget_discrepancy_max`` (max absolute
        "PERCENT DISCREPANCY" over both in/out columns and all budget
        tables; None when no budget table was found) and
        ``normal_termination`` (True when any .lst reports NORMAL
        TERMINATION).
    """
    discrepancies: list[float] = []
    normal_termination = False
    for lst_path in sorted(Path(mf6_ws).glob("*.lst")):
        try:
            text = lst_path.read_text(errors="replace")
        except OSError:
            continue
        if "normal termination" in text.lower():
            normal_termination = True
        for match in _BUDGET_DISCREPANCY_RE.finditer(text):
            discrepancies.append(abs(float(match.group(1))))
    return {
        "budget_discrepancy_max": max(discrepancies) if discrepancies else None,
        "normal_termination": bool(normal_termination),
    }


def _check_mf6_budget_discrepancy(
    discrepancy: float | None,
    tol: float | None,
    context: str = "MF6 run",
) -> None:
    """Raise when the parsed MF6 budget discrepancy is missing or exceeds ``tol`` (percent)."""
    if discrepancy is None:
        raise RuntimeError(
            f"{context}: no 'PERCENT DISCREPANCY' budget table found in the MF6 "
            ".lst output; cannot verify the MF6 water budget."
        )
    if tol is not None and float(tol) >= 0.0 and float(discrepancy) > float(tol):
        raise RuntimeError(
            f"{context}: MF6 PERCENT BUDGET DISCREPANCY {float(discrepancy):.6g}% "
            f"exceeds the configured tolerance {float(tol):.6g}%. The MF6 heads "
            "are NOT a trustworthy truth (this is the silent-stall failure mode "
            "documented for heterogeneous K fields)."
        )


def _check_mf6_heads_finite(heads: np.ndarray, active: np.ndarray, context: str = "MF6 run") -> None:
    """Raise when any active-cell head is NaN/inf or the MF6 HDRY dry-cell sentinel."""
    mask = np.asarray(active) != 0
    vals = np.asarray(heads, dtype=np.float64)[mask]
    bad = (~np.isfinite(vals)) | (np.abs(vals) >= 1.0e29)
    if np.any(bad):
        raise RuntimeError(
            f"{context}: {int(np.count_nonzero(bad))} active cells have non-finite "
            "or dry-sentinel (HDRY) heads; the MF6 truth is unusable."
        )


def _mf6_head_change_metrics(heads: np.ndarray, case: Unconfined2DCase) -> dict:
    """Max/RMS absolute change of the MF6 heads from the case initial head (active cells)."""
    mask = case.active != 0
    delta = np.asarray(heads, dtype=np.float64)[mask] - np.asarray(case.initial_head, dtype=np.float64)[mask]
    return {
        "head_change_max_from_initial": float(np.max(np.abs(delta))),
        "head_change_rms_from_initial": float(np.sqrt(np.mean(delta * delta))),
    }


def _check_mf6_head_change(
    metrics: dict,
    min_change: float | None,
    context: str = "MF6 run",
) -> None:
    """
    Case-aware sanity gate: require nontrivial head movement from the initial
    condition.  Catches the documented silent-stall mode where MF6 returns
    ok=True with mostly-initial-condition heads.  ``min_change`` is an
    absolute epsilon in metres; set it to 0/None ONLY for cases explicitly
    designed to sit at/near the initial condition (e.g. an equilibrium
    control case).
    """
    if min_change is None or float(min_change) <= 0.0:
        return
    if float(metrics["head_change_max_from_initial"]) < float(min_change):
        raise RuntimeError(
            f"{context}: max head change from the initial condition is "
            f"{metrics['head_change_max_from_initial']:.6g} m, below the sanity "
            f"floor {float(min_change):.6g} m. MF6 likely stalled at the initial "
            "condition (see the hard-T silent-stall comment in "
            "_build_and_run_mf6). If this case is intentionally near-initial, "
            "pass mf6_head_change_min=0."
        )


def build_simple_unconfined_case(
    nx: int = 250,
    ny: int = 250,
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
    t_field_kind: str = "uniform",
    t_field_seed: int = 42,
    use_ghb: bool = False,
    ghb_width: float = 100.0,
    ghb_head_elevation: float | None = None,
    ghb_conductance_mode: str = "warp_matched",
) -> Unconfined2DCase:
    """
    Build a shared 2D unconfined benchmark case for MF6 and Warp.

    :param t_field_kind: ``"uniform"`` (scalar ``hydraulic_conductivity``) or
        ``"ugly_t"`` (hard heterogeneous K field:
        ``make_ugly_T_field(nx, ny, domain, seed) / 100.0`` — the transient
        replay convention, K ~ 4-535 m/day at dx=100).
    :param t_field_seed: random seed for the ugly_t field.
    :param use_ghb: add a center-row GHB boundary (mask from
        ``model_builder._build_ghb_boundary_masks``, intersected with active).
    :param ghb_width: GHB river width [m] at GHB cells; the conductance is
        ``C_gh = T_c * gh_alpha * gh_width * dx / aq_thickness`` with
        ``gh_alpha=1`` and ``aq_thickness = top - bottom`` (so at full
        saturation ``C_gh = K * ghb_width * dx``, the MF6
        ``kriv_factor * hk * grid_size * width`` convention with factor 1).
    :param ghb_head_elevation: optional GHB stage as a height above the model
        bottom [m].  ``None`` (default) keeps the historical behaviour
        (stage = land-surface DEM, i.e. an injecting/holding boundary).
        Small values (e.g. 0.3 m) design DRAINING cases whose heads approach
        ``bottom + min_sat`` near the GHB row when combined with low K, low
        recharge and a small ``ghb_width`` (so the GHB row is not clamped).
    """
    if workspace is None:
        workspace = data_store.joinpath("working_tests", "mf6_vs_warp_2d_unconfined")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    active = _build_domain(nx=int(nx), ny=int(ny)).astype(np.int32)
    top = np.asarray(_build_dem(active), dtype=np.float64)
    bottom = np.asarray(_model_bottom(top), dtype=np.float64)
    bc_bool = _build_dirichlet_boundary_mask(active)

    bc_mask = bc_bool.astype(np.int32)
    bc_values = np.zeros((int(ny), int(nx)), dtype=np.float64)
    bc_values[bc_bool] = top[bc_bool]

    t_field_kind = str(t_field_kind).strip().lower()
    if t_field_kind == "uniform":
        k_field = np.full((int(ny), int(nx)), float(hydraulic_conductivity), dtype=np.float64)
    elif t_field_kind == "ugly_t":
        if _build_ghb_boundary_masks is None and make_ugly_T_field is None:
            raise RuntimeError("ugly_t field generator unavailable.")
        k_field = np.asarray(
            make_ugly_T_field(int(nx), int(ny), active, int(t_field_seed)),
            dtype=np.float64,
        ) / 100.0
    else:
        raise ValueError(f"t_field_kind must be 'uniform' or 'ugly_t', got {t_field_kind!r}.")
    k_field[active == 0] = 0.0

    recharge_field = np.full((int(ny), int(nx)), float(recharge), dtype=np.float64)
    recharge_field[active == 0] = 0.0

    gh_mask = np.zeros((int(ny), int(nx)), dtype=np.int32)
    gh_head = np.zeros((int(ny), int(nx)), dtype=np.float64)
    gh_width_field = np.zeros((int(ny), int(nx)), dtype=np.float64)
    if use_ghb:
        if _build_ghb_boundary_masks is None:
            raise RuntimeError("GHB boundary mask builder unavailable (legacy_code import failed).")
        gh_mask_bool = np.asarray(_build_ghb_boundary_masks(active), dtype=bool) & (active != 0)
        gh_mask[gh_mask_bool] = 1
        if ghb_head_elevation is None:
            gh_head[gh_mask_bool] = top[gh_mask_bool]
        else:
            gh_head[gh_mask_bool] = bottom[gh_mask_bool] + float(ghb_head_elevation)
        gh_width_field[gh_mask_bool] = float(ghb_width)

    initial_head = bottom + max(float(initial_saturated_thickness), 0.1)
    initial_head = np.minimum(initial_head, top)
    initial_head[bc_bool] = bc_values[bc_bool]
    initial_head[active == 0] = 0.0

    return Unconfined2DCase(
        nx=int(nx),
        ny=int(ny),
        dx=float(dx),
        workspace=workspace,
        active=active,
        bc_mask=bc_mask,
        bc_values=bc_values,
        top=top,
        bottom=bottom,
        hydraulic_conductivity=k_field,
        recharge=recharge_field,
        initial_head=initial_head.astype(np.float64, copy=False),
        gh_mask=gh_mask,
        gh_head=gh_head,
        gh_width=gh_width_field,
        t_field_kind=str(t_field_kind),
        t_field_seed=int(t_field_seed),
        use_ghb=bool(use_ghb),
        ghb_width=float(ghb_width),
        ghb_conductance_mode=str(ghb_conductance_mode),
    )


def _build_and_run_mf6(
    case: Unconfined2DCase,
    ghb_conductance: np.ndarray | None = None,
    strt_override: np.ndarray | None = None,
    mf6_budget_discrepancy_tol: float | None = DEFAULT_MF6_BUDGET_DISCREPANCY_TOL,
) -> dict:
    """
    Build, run and VERIFY one MF6 single-layer unconfined solve.

    Returns a dict with ``heads``, ``engine_time``, ``total_time`` and
    ``budget_discrepancy_max``.  Raises unless the run terminates normally,
    all active-cell heads are finite, and the max absolute "PERCENT BUDGET
    DISCREPANCY" from the .lst output is within
    ``mf6_budget_discrepancy_tol`` (percent).

    When ``case.use_ghb`` is set, a GHB package is added at ``case.gh_mask``
    cells with stage ``case.gh_head`` and FIXED conductance
    ``ghb_conductance`` (required).  MF6's conductance is head-independent
    while Warp's ``C_gh = T_c(h) * ghb_factor`` varies with saturation in
    unconfined mode, so GHB truth cases go through
    ``run_mf6_ghb_fixed_point`` (an MF6-side fixed point that never uses
    Warp heads).

    :param strt_override: optional starting-head array replacing
        ``case.initial_head`` for THIS run only (used by the GHB fixed point
        to warm each re-solve from the previous iterate; it does not change
        the case definition or fingerprint).
    """
    mf6_ws = case.workspace.joinpath("mf6")
    mf6_ws.mkdir(parents=True, exist_ok=True)

    name = "unconf2d_truth"
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
    # IMS selection: uniform cases converge fine under MODERATE/CG (the
    # historical configuration, ~7 s at 500x500).  Heterogeneous (ugly_t) K
    # fields silently stall it — MF6 returns ok=True with a mostly
    # initial-condition head field and a ~200 % budget discrepancy — so
    # they get COMPLEX/BICGSTAB with a generous outer budget (~230 s at
    # 500x500; contraction-limited, insensitive to dvclose/DBD).  The
    # budget-discrepancy and head-change gates applied to every run below
    # are the hard guards against that silent-stall mode.
    if str(case.t_field_kind) == "ugly_t":
        ims = flopy.mf6.ModflowIms(
            sim,
            pname="ims",
            print_option="SUMMARY",
            complexity="COMPLEX",
            linear_acceleration="BICGSTAB",
            outer_maximum=500,
            outer_dvclose=1.0e-6,
            inner_maximum=500,
            inner_dvclose=1.0e-8,
            rcloserecord=[1.0e-6, "RELATIVE_RCLOSE"],
            scaling_method="DIAGONAL",
        )
    else:
        ims = flopy.mf6.ModflowIms(
            sim,
            pname="ims",
            print_option="SUMMARY",
            complexity="MODERATE",
            linear_acceleration="CG",
            outer_maximum=100,
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
    strt = case.initial_head if strt_override is None else np.asarray(strt_override, dtype=np.float64)
    flopy.mf6.ModflowGwfic(
        gwf,
        pname="ic",
        strt=strt,
    )
    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=[1],
        k=case.hydraulic_conductivity,
        k33=case.hydraulic_conductivity,
        k33overk=False,
        save_specific_discharge=True,
        save_saturation=True,
    )

    fixed_head_cells = np.full((case.ny, case.nx), np.nan, dtype=np.float64)
    fixed_head_cells[case.bc_mask != 0] = case.bc_values[case.bc_mask != 0]
    chd_spd = _create_chd_single_period(boundary_heads=fixed_head_cells, active=case.active)
    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="chd",
        stress_period_data=chd_spd,
        save_flows=True,
    )
    if case.use_ghb:
        if ghb_conductance is None:
            raise ValueError(
                "ghb_conductance is required when case.use_ghb is set; use "
                "run_mf6_ghb_fixed_point (MF6-side fixed point over the "
                "head-dependent conductance law) to generate GHB truth."
            )
        gh_cells = (case.gh_mask != 0) & (case.active != 0) & (case.bc_mask == 0)
        cond = np.asarray(ghb_conductance, dtype=np.float64)
        if cond.shape != (case.ny, case.nx):
            raise ValueError(f"ghb_conductance shape {cond.shape} expected {(case.ny, case.nx)}.")
        ghb_spd = [
            ((0, int(j), int(i)), float(case.gh_head[j, i]), float(cond[j, i]))
            for j, i in zip(*np.where(gh_cells))
            if cond[j, i] > 0.0
        ]
        if ghb_spd:
            flopy.mf6.ModflowGwfghb(
                gwf,
                pname="ghb",
                stress_period_data={0: ghb_spd},
                save_flows=True,
            )
    flopy.mf6.ModflowGwfrcha(
        gwf,
        pname="recharge",
        recharge=case.recharge,
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "LAST")],
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
        raise RuntimeError("MF6 unconfined run failed (run_simulation returned ok=False).")

    # Trust gates on the MF6 output: normal termination, budget discrepancy,
    # finite heads.  Applied after EVERY MF6 run.
    lst_stats = _parse_mf6_lst(mf6_ws)
    if not lst_stats["normal_termination"]:
        raise RuntimeError(
            f"MF6 run in {mf6_ws} did not report NORMAL TERMINATION in any .lst "
            "file despite ok=True; refusing to trust the heads."
        )
    _check_mf6_budget_discrepancy(
        lst_stats["budget_discrepancy_max"],
        mf6_budget_discrepancy_tol,
        context=f"MF6 run in {mf6_ws}",
    )

    hds_path = mf6_ws.joinpath(f"{name}.hds")
    heads_raw = flopy.utils.HeadFile(str(hds_path)).get_data()
    heads = np.asarray(heads_raw[0], dtype=np.float64) if heads_raw.ndim == 3 else np.asarray(heads_raw)
    _check_mf6_heads_finite(heads, case.active, context=f"MF6 run in {mf6_ws}")

    return {
        "heads": heads,
        "engine_time": float(engine_time),
        "total_time": float(total_time),
        "budget_discrepancy_max": float(lst_stats["budget_discrepancy_max"]),
    }


def _save_mf6_npz(
    case: Unconfined2DCase,
    out_path: str | Path,
    result: dict,
    mf6_head_change_min: float | None = DEFAULT_MF6_HEAD_CHANGE_MIN,
    extra: dict | None = None,
) -> Path:
    """
    Write the MF6 truth NPZ: heads, timings, budget-discrepancy and
    from-initial head-change metrics, artifact schema version and the case
    fingerprint.  Applies the head-change sanity gate before writing.
    """
    out_path = Path(out_path)
    heads = np.asarray(result["heads"], dtype=np.float64)
    change = _mf6_head_change_metrics(heads, case)
    _check_mf6_head_change(change, mf6_head_change_min, context=f"MF6 truth for {out_path}")

    payload = dict(
        heads=heads,
        engine_time=np.asarray(result["engine_time"], dtype=np.float64),
        total_time=np.asarray(result["total_time"], dtype=np.float64),
        budget_discrepancy_max=np.asarray(result["budget_discrepancy_max"], dtype=np.float64),
        head_change_max_from_initial=np.asarray(change["head_change_max_from_initial"], dtype=np.float64),
        head_change_rms_from_initial=np.asarray(change["head_change_rms_from_initial"], dtype=np.float64),
        nx=np.asarray(case.nx, dtype=np.int32),
        ny=np.asarray(case.ny, dtype=np.int32),
        dx=np.asarray(case.dx, dtype=np.float64),
        schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int32),
        case_fingerprint=np.asarray(case_fingerprint(case)),
    )
    if extra:
        payload.update(extra)
    np.savez_compressed(out_path, **payload)
    print(f"MF6 unconfined heads saved to {out_path}")
    print(
        f"MF6 metrics - Total time: {result['total_time']:.4f}s, "
        f"Engine time: {result['engine_time']:.4f}s, "
        f"budget discrepancy: {result['budget_discrepancy_max']:.3g}%, "
        f"max head change from initial: {change['head_change_max_from_initial']:.6g} m\n"
    )
    return out_path


def run_mf6_unconfined(
    case: Unconfined2DCase,
    out_path: str | Path | None = None,
    ghb_conductance: np.ndarray | None = None,
    mf6_budget_discrepancy_tol: float | None = DEFAULT_MF6_BUDGET_DISCREPANCY_TOL,
    mf6_head_change_min: float | None = DEFAULT_MF6_HEAD_CHANGE_MIN,
    artifact_extra: dict | None = None,
) -> Path:
    """
    Run the MF6 single-layer unconfined truth model once and save heads to NPZ.

    For ``case.use_ghb`` cases prefer ``run_mf6_ghb_fixed_point``, which
    determines the fixed MF6 conductance by an MF6-side fixed point (Warp
    heads are never used).  Passing ``ghb_conductance`` explicitly here is
    only for callers that already have a converged conductance array.
    """
    out_path = Path(out_path) if out_path is not None else case.workspace.joinpath("mf6_heads.npz")
    result = _build_and_run_mf6(
        case,
        ghb_conductance=ghb_conductance,
        mf6_budget_discrepancy_tol=mf6_budget_discrepancy_tol,
    )
    extra = dict(artifact_extra or {})
    if ghb_conductance is not None:
        # Persist the exact fixed array used by MF6. Reconstructing it later
        # from MF6 heads is only approximately equivalent and weakens cache
        # provenance in warp-matched equation-equivalence runs.
        extra["ghb_conductance"] = np.asarray(ghb_conductance, dtype=np.float64)
    return _save_mf6_npz(
        case,
        out_path,
        result,
        mf6_head_change_min=mf6_head_change_min,
        extra=extra or None,
    )


def run_mf6_ghb_fixed_point(
    case: Unconfined2DCase,
    out_path: str | Path | None = None,
    cond_rtol: float = DEFAULT_GHB_COND_RTOL,
    head_atol: float = DEFAULT_GHB_HEAD_ATOL,
    max_iterations: int = DEFAULT_GHB_MAX_FIXED_POINT_ITERATIONS,
    mf6_budget_discrepancy_tol: float | None = DEFAULT_MF6_BUDGET_DISCREPANCY_TOL,
    mf6_head_change_min: float | None = DEFAULT_MF6_HEAD_CHANGE_MIN,
) -> tuple[Path, dict]:
    """
    MF6-side fixed point for the head-dependent GHB conductance.

    MF6's GHB conductance is a FIXED input while the shared conductance law
    ``C_gh = K * clip(h - bottom, min_sat, top - bottom) * gh_width * dx /
    (top - bottom)`` (see ``_ghb_conductance_from_heads``) depends on the
    unknown head.  This routine iterates ENTIRELY on the MF6 side — Warp
    heads are never used:

      1. initialise ``C_gh`` from the CASE INITIAL HEAD (documented,
         deterministic; any reasonable initial head works because the
         iteration below is contractive for these cases),
      2. run MF6 with that fixed conductance,
      3. re-evaluate ``C_gh`` at the MF6 heads,
      4. repeat (warm-started from the previous MF6 heads) until BOTH the
         max relative conductance change <= ``cond_rtol`` AND the max
         absolute head change <= ``head_atol`` between iterations.

    Slowly-contracting cases (e.g. the weak-GHB draining validator combo,
    contraction ~0.85/iteration) are handled with Aitken/Steffensen
    Delta^2 acceleration: after every two plain map iterates the sequence
    is extrapolated (elementwise, only where the extrapolation is finite,
    non-negative and below the full-saturation conductance cap; the
    acceptance criteria above are ALWAYS evaluated on plain iterates, so
    acceleration cannot loosen the tolerances).

    Raises on non-convergence after ``max_iterations``.  Returns
    ``(out_path, fixed_point_info)``.
    """
    if not case.use_ghb:
        raise ValueError("run_mf6_ghb_fixed_point requires case.use_ghb=True.")
    out_path = Path(out_path) if out_path is not None else case.workspace.joinpath("mf6_heads.npz")

    conductance = _ghb_conductance_from_heads(case, case.initial_head)
    window: list[np.ndarray] = [conductance]  # map iterates for Aitken acceleration
    heads_prev: np.ndarray | None = None
    result: dict | None = None
    terminal_cond_change = float("inf")
    terminal_head_change = float("inf")
    iterations = 0
    converged = False
    cumulative_engine_time = 0.0
    cumulative_total_time = 0.0
    fixed_point_start = time.perf_counter()
    for iteration in range(1, int(max_iterations) + 1):
        iterations = iteration
        strt = None
        if heads_prev is not None:
            # Warm start from the previous iterate (clipped to the model
            # interval for robustness); this does not alter the case.
            strt = np.clip(heads_prev, case.bottom, case.top)
        print(f"Starting GHB fixed point MF6 solve {iteration}/{int(max_iterations)}...", flush=True)
        result = _build_and_run_mf6(
            case,
            ghb_conductance=conductance,
            strt_override=strt,
            mf6_budget_discrepancy_tol=mf6_budget_discrepancy_tol,
        )
        cumulative_engine_time += float(result["engine_time"])
        cumulative_total_time += float(result["total_time"])
        heads = np.asarray(result["heads"], dtype=np.float64)
        cond_new = _ghb_conductance_from_heads(case, heads)
        scale = max(1.0, float(np.max(np.abs(cond_new))))
        terminal_cond_change = float(np.max(np.abs(cond_new - conductance))) / scale
        if heads_prev is not None:
            terminal_head_change = float(np.max(np.abs(heads - heads_prev)))
        else:
            terminal_head_change = float("inf")
        print(
            f"GHB fixed point iter {iteration}: max rel conductance change "
            f"{terminal_cond_change:.3e} (tol {cond_rtol:.1e}), max head change "
            f"{terminal_head_change:.3e} m (tol {head_atol:.1e})"
        )
        heads_prev = heads
        if terminal_cond_change <= float(cond_rtol) and terminal_head_change <= float(head_atol):
            conductance = cond_new
            converged = True
            break

        # Aitken/Steffensen Delta^2 acceleration on the conductance map
        # iterates c_{n+1} = F(c_n).  Acceptance above always uses plain
        # iterates; acceleration only picks the next MF6 conductance.
        window.append(cond_new)
        next_conductance = cond_new
        if len(window) >= 3:
            c0, c1, c2 = window[-3], window[-2], window[-1]
            d1 = c1 - c0
            denom = c2 - 2.0 * c1 + c0
            with np.errstate(divide="ignore", invalid="ignore"):
                acc = c0 - d1 * d1 / np.where(np.abs(denom) > 1.0e-300, denom, np.nan)
            gh_cells = (case.gh_mask != 0) & (case.active != 0) & (case.bc_mask == 0)
            # Full-saturation cap: C_gh <= K * gh_width * dx per cell.
            cap = case.hydraulic_conductivity * np.asarray(case.gh_width, dtype=np.float64) * float(case.dx)
            valid = np.isfinite(acc) & (acc >= 0.0) & (acc <= cap)
            n_gh = int(np.count_nonzero(gh_cells))
            n_valid = int(np.count_nonzero(valid & gh_cells))
            if n_gh > 0 and n_valid >= max(1, (n_gh + 1) // 2):
                next_conductance = np.where(valid, acc, c2)
                window = [next_conductance]
                print(f"  Aitken acceleration applied ({n_valid}/{n_gh} GHB cells)")
        conductance = next_conductance

    if not converged or result is None:
        raise RuntimeError(
            f"MF6 GHB conductance fixed point did not converge in {iterations} "
            f"iterations (terminal rel conductance change {terminal_cond_change:.3e} "
            f"> {cond_rtol:.1e} or head change {terminal_head_change:.3e} m > "
            f"{head_atol:.1e} m). Refusing to use a non-converged GHB truth."
        )

    fixed_point_info = {
        "method": "mf6_side_fixed_point",
        "initialisation": "conductance evaluated at the case initial head (no Warp heads)",
        "iterations": int(iterations),
        "converged": True,
        "terminal_rel_conductance_change": float(terminal_cond_change),
        "terminal_head_change": float(terminal_head_change),
        "cond_rtol": float(cond_rtol),
        "head_atol": float(head_atol),
        "max_iterations": int(max_iterations),
        "cumulative_engine_time": float(cumulative_engine_time),
        "cumulative_total_time": float(cumulative_total_time),
        "wall_time": float(time.perf_counter() - fixed_point_start),
        "terminal_run_engine_time": float(result["engine_time"]),
        "terminal_run_total_time": float(result["total_time"]),
    }
    cumulative_result = dict(result)
    cumulative_result["engine_time"] = float(cumulative_engine_time)
    cumulative_result["total_time"] = float(cumulative_total_time)
    _save_mf6_npz(
        case,
        out_path,
        cumulative_result,
        mf6_head_change_min=mf6_head_change_min,
        extra={
            "ghb_conductance": np.asarray(conductance, dtype=np.float64),
            "ghb_fixed_point": np.asarray(json.dumps(fixed_point_info)),
        },
    )
    return out_path, fixed_point_info


def _save_outer_history_first25(info: dict, out_path: Path) -> None:
    """
    Save the first 25 Picard outer iterations as a CSV for debugging early-phase behaviour.

    :param info: solver info dictionary containing ``outer_history``.
    :param out_path: path to write the CSV file.
    """
    history = info.get("outer_history", []) if isinstance(info, dict) else []
    if not isinstance(history, list) or not history:
        return

    rows = history[:25]
    columns = [
        "outer_iteration",
        "inner_max_cycles_used",
        "inner_converged",
        "inner_head_change_converged",
        "inner_usable_for_picard",
        "h_rms_end",
        "inner_head_residual_tol_used",
        "picard_update_max",
        "picard_update_rms",
        "picard_scale",
        "omega",
        "chebyshev_ready",
        "chebyshev_used",
        "chebyshev_reset",
        "chebyshev_rejected",
        "trial_measure",
        "previous_measure",
        "clipped_update",
    ]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            if not isinstance(row, dict):
                continue
            writer.writerow({key: row.get(key) for key in columns})


def _solve_summary(info: object, elapsed: float, settings: dict) -> dict:
    summary = {
        "time": float(elapsed),
        "settings": dict(settings),
        "converged": False,
        "outer_iterations": None,
        "final_max_abs_head_change": None,
        "final_residual": None,
        "inner_solve_failures": None,
        "strict_inner_nonconvergence_count": None,
        "unusable_inner_solve_count": None,
        "practical_inner_acceptance_count": None,
        "accepted_picard_update_count": None,
        "outer_chebyshev_ready_count": None,
        "outer_chebyshev_used_count": None,
        "outer_chebyshev_reset_count": None,
        "chebyshev_rejections": None,
        "chebyshev_resets": None,
    }
    if isinstance(info, dict):
        summary["converged"] = bool(info.get("converged", False))
        for key in (
            "outer_iterations",
            "final_max_abs_head_change",
            "final_residual",
            "inner_solve_failures",
            "strict_inner_nonconvergence_count",
            "unusable_inner_solve_count",
            "practical_inner_acceptance_count",
            "accepted_picard_update_count",
            "outer_chebyshev_ready_count",
            "outer_chebyshev_used_count",
            "outer_chebyshev_reset_count",
            "chebyshev_rejections",
            "chebyshev_resets",
            "effectively_dry_cell_count",
        ):
            if info.get(key) is not None:
                value = info[key]
                summary[key] = int(value) if isinstance(value, (int, np.integer)) else float(value)
    return summary


def _ghb_conductance_from_heads(case: Unconfined2DCase, heads: np.ndarray) -> np.ndarray:
    """Head-dependent GHB conductance law evaluated at ``heads``.

    ``C_gh = T_c * ghb_factor`` with
    ``T_c = K * clip(h - bottom, min_sat, top - bottom)`` (the Picard
    saturated-thickness update, min_sat=0.1) and
    ``ghb_factor = gh_alpha * gh_width * dx / aq_thickness`` with
    ``gh_alpha=1`` and ``aq_thickness = top - bottom`` (the
    ``physics.operator_data.compute_ghb_factor_from_raw_fields``
    convention used by ``build_from_fields``).

    This is the discrete law Warp applies every Picard iteration.  MF6 only
    accepts a FIXED conductance, so ``run_mf6_ghb_fixed_point`` iterates
    MF6 with this law re-evaluated at MF6's own heads until conductance and
    heads stop changing — the fixed point of the same law, with no Warp
    heads involved.
    """
    full = np.maximum(case.top - case.bottom, 0.1)
    sat = np.clip(np.asarray(heads, dtype=np.float64) - case.bottom, 0.1, full)
    T_c = case.hydraulic_conductivity * sat
    ghb_factor = np.asarray(case.gh_width, dtype=np.float64) * float(case.dx) / full
    cond = T_c * ghb_factor
    gh_cells = (case.gh_mask != 0) & (case.active != 0) & (case.bc_mask == 0)
    out = np.zeros((case.ny, case.nx), dtype=np.float64)
    out[gh_cells] = cond[gh_cells]
    return out


def run_warp_unconfined(
    case: Unconfined2DCase,
    out_path: str | Path | None = None,
    device: str = "auto",
    chebyshev_enabled: bool = True,
    inner_smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_double_solve: bool = True,
    solver_backend: str | None = None,
    inner_implementation: str = "fast",
) -> Path:
    """
    Run the same unconfined problem in the main 2D Warp solver and save heads to NPZ.
    """
    out_path = Path(out_path) if out_path is not None else case.workspace.joinpath("warp_heads.npz")
    device = _warp_device(device)

    initial_transmissivity = case.hydraulic_conductivity * np.maximum(case.initial_head - case.bottom, 0.1)
    initial_transmissivity[case.active == 0] = 0.0
    rhs_recharge = np.asarray(case.recharge, dtype=np.float64)

    t0 = time.perf_counter()
    with WarpDarcySolver(
        nx=case.nx,
        ny=case.ny,
        dx=case.dx,
        device=device,
        use_ghb=bool(case.use_ghb),
        solver_type="kcycle",
        diag_preconditioner_backend=diag_preconditioner_backend,
        aq_thickness=(case.top - case.bottom),
    ) as warp_solver:
        warp_solver.build_from_fields(
            T_field=initial_transmissivity,
            R_field=rhs_recharge,
            active=case.active,
            bc_mask=case.bc_mask,
            bc_values=case.bc_values,
            gh_mask=case.gh_mask,
            gh_head=case.gh_head,
            gh_width=case.gh_width,
            gh_alpha=1.0,
            aq_thickness=(case.top - case.bottom),
        )
        solve1_kwargs = {
            "formulation": "unconfined",
            "solver": solver_backend,
            "K_field": case.hydraulic_conductivity,
            "zbot_field": case.bottom,
            "ztop_field": case.top,
            "initial_head": case.initial_head.copy(),
            "max_cycles": 80,
            "max_levels": 5,
            "min_coarse_cells": 500,
            "rel_tol": 5.0e-7,
            "abs_tol_min": 5.0e-7,
            "dh_rms_tol": DEFAULT_DH_TOL,
            "residual_floor_tol": DEFAULT_RESIDUAL_FLOOR_TOL,
            "inner_forcing_eta": float(inner_forcing_eta),
            "inner_head_residual_tol_min": float(inner_head_residual_tol_min),
            "inner_head_residual_tol_max": float(inner_head_residual_tol_max),
            "chebyshev_reset_factor": float(chebyshev_reset_factor),
            "transmissivity_relaxation_enabled": bool(transmissivity_relaxation_enabled),
            "unconfined_startup_mode": str(unconfined_startup_mode),
            "smoother": str(inner_smoother),
            "cheby_lambda_min": float(cheby_lambda_min),
            "cheby_lambda_max": float(cheby_lambda_max),
            "max_outer_iterations": 60,
            "omega": 0.7,
            "omega_min": 0.1,
            "omega_max": 0.9,
            "hclose": DEFAULT_DH_TOL,
            "min_saturated_thickness": 0.1,
            "initial_saturated_thickness": 100.0,
            "max_head_change_per_outer_iteration": 10.0,
            "chebyshev_enabled": bool(chebyshev_enabled),
            "chebyshev_order": 3,
            "chebyshev_rejection_factor": 1.2,
            "inner_implementation": str(inner_implementation),
        }
        if check_every_no is not None:
            solve1_kwargs["check_every_no"] = int(check_every_no)
        solve2_kwargs = dict(solve1_kwargs)

        if do_double_solve:
            t_solve1 = time.perf_counter()
            heads1, info1 = warp_solver.solve(**solve1_kwargs)
            solve1_time = time.perf_counter() - t_solve1
        else:
            heads1, info1 = None, None
            solve1_time = 0.0

        # Benchmark timing uses solve 2 from the same initial condition, matching
        # the 3D runner convention.
        t_solve2 = time.perf_counter()
        heads, info = warp_solver.solve(**solve2_kwargs)
        solve2_time = time.perf_counter() - t_solve2

    total_time = time.perf_counter() - t0
    solve1_summary = _solve_summary(info1, solve1_time, solve1_kwargs) if do_double_solve else {}
    solve2_summary = _solve_summary(info, solve2_time, solve2_kwargs)

    first25_path = Path(out_path).parent.joinpath("outer_history_first25.csv")
    _save_outer_history_first25(info, first25_path)

    np.savez_compressed(
        out_path,
        heads=np.asarray(heads, dtype=np.float64),
        heads_solve1=np.asarray(heads1, dtype=np.float64),
        total_time=np.asarray(total_time, dtype=np.float64),
        solve1_time=np.asarray(solve1_time, dtype=np.float64),
        solve2_time=np.asarray(solve2_time, dtype=np.float64),
        info=np.asarray(json.dumps(info, default=str)),
        info_solve1=np.asarray(json.dumps(info1, default=str) if info1 else "{}"),
        info_solve2=np.asarray(json.dumps(info, default=str)),
        summary_solve1=np.asarray(json.dumps(solve1_summary, default=str)),
        summary_solve2=np.asarray(json.dumps(solve2_summary, default=str)),
        solve1_settings=np.asarray(json.dumps(solve1_kwargs, default=str)),
        solve2_settings=np.asarray(json.dumps(solve2_kwargs, default=str)),
        nx=np.asarray(case.nx, dtype=np.int32),
        ny=np.asarray(case.ny, dtype=np.int32),
        dx=np.asarray(case.dx, dtype=np.float64),
        device=np.asarray(device),
        schema_version=np.asarray(ARTIFACT_SCHEMA_VERSION, dtype=np.int32),
        case_fingerprint=np.asarray(case_fingerprint(case)),
    )
    print(f"Warp unconfined heads saved to {out_path}")
    print(
        f"Warp metrics - Total time: {total_time:.4f}s, "
        f"solve1: {solve1_time:.4f}s, solve2: {solve2_time:.4f}s"
    )
    if isinstance(info, dict):
        print("Warp nonlinear solve 2 summary:")
        for key in (
            "converged",
            "outer_iterations",
            "smoother",
            "final_max_abs_head_change",
            "final_residual",
            "final_h_rms_inner_residual",
            "chebyshev_rejections",
            "chebyshev_resets",
            "accepted_picard_update_count",
            "strict_inner_nonconvergence_count",
            "unusable_inner_solve_count",
            "practical_inner_acceptance_count",
            "outer_chebyshev_ready_count",
            "outer_chebyshev_used_count",
            "outer_chebyshev_reset_count",
            "inner_forcing_eta",
            "inner_head_residual_tol_min",
            "inner_head_residual_tol_max",
            "inner_solve_failures",
            "effectively_dry_cell_count",
        ):
            if key == "final_h_rms_inner_residual":
                print(f"  {key}: {info.get('inner_h_rms_end')}")
            else:
                print(f"  {key}: {info.get(key)}")
    print()
    return out_path


def load_results(mf6_path: str | Path, warp_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(mf6_path, allow_pickle=False) as mf6_npz:
        mf6_heads = np.asarray(mf6_npz["heads"], dtype=np.float64)
    with np.load(warp_path, allow_pickle=False) as warp_npz:
        warp_heads = np.asarray(warp_npz["heads"], dtype=np.float64)
    return mf6_heads, warp_heads


def compare_results(
    mf6_path: str | Path,
    warp_path: str | Path,
    active: np.ndarray | None = None,
) -> dict[str, float]:
    mf6_heads, warp_heads = load_results(mf6_path, warp_path)
    if mf6_heads.shape != warp_heads.shape:
        raise ValueError(f"Shape mismatch: MF6 {mf6_heads.shape}, Warp {warp_heads.shape}")

    if active is None:
        mask = np.isfinite(mf6_heads) & np.isfinite(warp_heads)
    else:
        mask = (np.asarray(active) != 0) & np.isfinite(mf6_heads) & np.isfinite(warp_heads)
        if mask.shape != mf6_heads.shape:
            raise ValueError(f"active shape {mask.shape} does not match heads {mf6_heads.shape}")

    diff = warp_heads - mf6_heads
    diff_masked = diff[mask]
    abs_diff = np.abs(diff_masked)
    metrics = {
        "rmse": float(np.sqrt(np.mean(diff_masked * diff_masked))),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_bias_warp_minus_mf6": float(np.mean(diff_masked)),
        "percent_within_0_01m": float(np.mean(abs_diff <= 0.01) * 100.0),
        "percent_within_0_1m": float(np.mean(abs_diff <= 0.1) * 100.0),
        "percent_within_1_0m": float(np.mean(abs_diff <= 1.0) * 100.0),
    }
    print("\nWarp vs MF6 unconfined head comparison, active cells")
    for key, value in metrics.items():
        print(f"  {key}: {value:.6g}")
    return metrics


def _convergence_report(
    info: dict,
    comparison: dict[str, float] | None = None,
    mf6_agreement_tol: float = DEFAULT_MF6_AGREEMENT_TOL,
) -> dict:
    final_dh = _finite_float(info.get("final_max_abs_head_change"))
    hclose = None
    history = info.get("outer_history", [])
    if isinstance(history, list) and history:
        hclose = _finite_float(info.get("picard_head_tol"))
    max_abs_diff = _finite_float((comparison or {}).get("max_abs_diff"))

    head_change_converged = None if final_dh is None or hclose is None else bool(final_dh <= hclose)
    inner_residual_converged = bool(info.get("inner_residual_converged", False))
    inner_head_change_converged = bool(info.get("inner_head_change_converged", False))
    inner_practically_converged = bool(info.get("inner_practically_converged", False))
    agrees_with_mf6 = None if max_abs_diff is None else bool(max_abs_diff < float(mf6_agreement_tol))

    if bool(info.get("converged", False)):
        if inner_residual_converged:
            status = "Nonlinear head-change and inner residual tolerances met."
        elif inner_practically_converged:
            status = "Nonlinear head-change tolerance met via practical inner convergence."
        else:
            status = "Nonlinear head-change tolerance met."
    else:
        # Hard failure: a reported non-converged solve is NEVER an acceptable
        # result, no matter how small the final head change or how well it
        # happens to agree with MF6 (the previous "acceptable despite
        # non-convergence" branch was removed deliberately).
        status = "Convergence criteria not met."

    return {
        "head_change_converged": head_change_converged,
        "inner_residual_converged": inner_residual_converged,
        "inner_head_change_converged": inner_head_change_converged,
        "inner_practically_converged": inner_practically_converged,
        "agrees_with_mf6": agrees_with_mf6,
        "status": status,
        "final_max_abs_head_change": final_dh,
        "hclose": hclose,
        "residual_floor_tol": _finite_float(info.get("residual_floor_tol")),
        "max_abs_diff": max_abs_diff,
        "mf6_agreement_tol": float(mf6_agreement_tol),
    }


def _require_warp_converged(info: dict, context: str = "Warp solve") -> dict:
    """
    Refuse to use Warp heads unless the solve reported convergence AND the
    strict convergence status passes (``_convergence_report``).  Returns the
    convergence report on success; raises RuntimeError otherwise.
    """
    report = _convergence_report(info)
    status = str(report.get("status"))
    # Strict gate: info['converged'] must be set AND the strict Picard flag
    # must pass.  Backends that do not report a strict flag fall back to the
    # strictest status string.
    strict_flag = info.get("strict_picard_convergence_passed")
    if strict_flag is not None:
        strict_ok = bool(strict_flag)
    else:
        strict_ok = status == "Nonlinear head-change and inner residual tolerances met."
    if not (bool(info.get("converged", False)) and strict_ok):
        raise RuntimeError(
            f"{context}: Warp solve did not converge to the required standard "
            f"(info['converged']={info.get('converged')}, "
            f"strict_picard_convergence_passed={strict_flag}, status={status!r}). "
            "Refusing to use these heads for comparison or GHB conductance "
            "construction."
        )
    return report


def _ghb_cell_comparison(
    case: Unconfined2DCase,
    mf6_heads: np.ndarray,
    warp_heads: np.ndarray,
) -> dict:
    """Separate Warp-vs-MF6 error metrics restricted to the GHB cells."""
    gh_cells = (case.gh_mask != 0) & (case.active != 0) & (case.bc_mask == 0)
    if not np.any(gh_cells):
        return {}
    diff = np.asarray(warp_heads, dtype=np.float64)[gh_cells] - np.asarray(mf6_heads, dtype=np.float64)[gh_cells]
    out = {
        "n_ghb_cells": int(np.count_nonzero(gh_cells)),
        "rmse": float(np.sqrt(np.mean(diff * diff))),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_bias_warp_minus_mf6": float(np.mean(diff)),
    }
    print("\nWarp vs MF6 head comparison, GHB cells only")
    for key, value in out.items():
        print(f"  {key}: {value:.6g}")
    return out


def _ghb_coupling_ratio(
    case: Unconfined2DCase,
    heads: np.ndarray,
    conductance: np.ndarray,
) -> dict:
    """
    Conductance-to-neighbour-coupling ratio at GHB cells.

    For each GHB cell the inter-cell harmonic conductance to each active
    neighbour is ``C_n = 2*T*T_nb/(T + T_nb)`` (unit width/length factor:
    ``delr = delc = dx`` cancels, ``T = K * sat(h)`` evaluated at ``heads``).
    The reported ratio is ``C_gh / mean(C_n)`` per GHB cell (max and mean
    over GHB cells).  A ratio >> 1 means the GHB clamps the row; << 1 means
    it barely couples (the draining validator case targets the latter).
    """
    heads = np.asarray(heads, dtype=np.float64)
    full = np.maximum(case.top - case.bottom, 0.1)
    sat = np.clip(heads - case.bottom, 0.1, full)
    trans = case.hydraulic_conductivity * sat
    active = (case.active != 0)
    gh_cells = (case.gh_mask != 0) & active & (case.bc_mask == 0)
    if not np.any(gh_cells):
        return {}

    padded_t = np.pad(trans, 1, mode="edge")
    padded_active = np.pad(active, 1, mode="constant", constant_values=False)
    neighbour_conds = []
    for dy, dxs in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        nb_t = padded_t[1 + dy:1 + dy + case.ny, 1 + dxs:1 + dxs + case.nx]
        nb_ok = padded_active[1 + dy:1 + dy + case.ny, 1 + dxs:1 + dxs + case.nx]
        denom = trans + nb_t
        with np.errstate(divide="ignore", invalid="ignore"):
            c = np.where(denom > 0.0, 2.0 * trans * nb_t / np.where(denom > 0.0, denom, 1.0), np.nan)
        neighbour_conds.append(np.where(nb_ok, c, np.nan))
    with np.errstate(invalid="ignore"):
        mean_coupling = np.nanmean(np.stack(neighbour_conds), axis=0)
    ratio = np.asarray(conductance, dtype=np.float64)[gh_cells] / mean_coupling[gh_cells]
    ratio = ratio[np.isfinite(ratio)]
    out = {
        "n_ghb_cells": int(np.count_nonzero(gh_cells)),
        "max": float(np.max(ratio)),
        "mean": float(np.mean(ratio)),
    }
    print("\nGHB conductance-to-neighbour-coupling ratio (C_gh / mean inter-cell C)")
    for key, value in out.items():
        print(f"  {key}: {value:.6g}")
    return out


def run_case(
    nx: int = 250,
    ny: int = 250,
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
    device: str = "auto",
    chebyshev_enabled: bool = True,
    inner_smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
    do_double_solve: bool = True,
    solver_backend: str | None = None,
    inner_implementation: str = "fast",
    t_field_kind: str = "uniform",
    t_field_seed: int = 42,
    use_ghb: bool = False,
    ghb_width: float = 100.0,
    ghb_head_elevation: float | None = None,
    ghb_conductance_mode: str = "warp_matched",
    mf6_budget_discrepancy_tol: float | None = DEFAULT_MF6_BUDGET_DISCREPANCY_TOL,
    mf6_head_change_min: float | None = DEFAULT_MF6_HEAD_CHANGE_MIN,
    ghb_cond_rtol: float = DEFAULT_GHB_COND_RTOL,
    ghb_head_atol: float = DEFAULT_GHB_HEAD_ATOL,
    ghb_max_fixed_point_iterations: int = DEFAULT_GHB_MAX_FIXED_POINT_ITERATIONS,
) -> dict:
    case = build_simple_unconfined_case(
        nx=nx,
        ny=ny,
        dx=dx,
        hydraulic_conductivity=hydraulic_conductivity,
        recharge=recharge,
        initial_saturated_thickness=initial_saturated_thickness,
        workspace=workspace,
        t_field_kind=t_field_kind,
        t_field_seed=t_field_seed,
        use_ghb=use_ghb,
        ghb_width=ghb_width,
        ghb_head_elevation=ghb_head_elevation,
        ghb_conductance_mode=ghb_conductance_mode,
    )

    print(f"Running 2D unconfined case: nx={case.nx}, ny={case.ny}, dx={case.dx}")
    print(f"Workspace: {case.workspace}")
    print(
        f"Options: t_field_kind={case.t_field_kind} (seed={case.t_field_seed}), "
        f"use_ghb={case.use_ghb} (width={case.ghb_width}), "
        f"ghb_conductance_mode={case.ghb_conductance_mode}\n"
    )

    mf6_path = case.workspace.joinpath("mf6_heads.npz")
    warp_path = case.workspace.joinpath("warp_heads.npz")
    fingerprint = case_fingerprint(case)

    def _run_warp() -> None:
        run_warp_unconfined(
            case,
            out_path=warp_path,
            device=device,
            chebyshev_enabled=chebyshev_enabled,
            inner_smoother=inner_smoother,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
            do_double_solve=do_double_solve,
            solver_backend=solver_backend,
            inner_implementation=inner_implementation,
        )

    # --- MF6 truth: reuse a fingerprint-matching cache, else regenerate. ---
    # do_run_mf6=True permits regeneration on fingerprint mismatch;
    # do_run_mf6=False means the cached artifact must match or we raise.
    ghb_conductance = None
    ghb_fixed_point_info: dict | None = None
    warp_already_run = False
    conductance_mode = str(case.ghb_conductance_mode).strip().lower()
    if conductance_mode not in {"fixed_point", "warp_matched"}:
        raise ValueError(
            f"ghb_conductance_mode must be 'fixed_point' or 'warp_matched', "
            f"got {case.ghb_conductance_mode!r}."
        )
    if do_run_mf6:
        if validate_mf6_artifact(mf6_path, case):
            print(f"Reusing cached MF6 artifact {mf6_path} (schema + case fingerprint match).")
        elif case.use_ghb and conductance_mode == "warp_matched":
            # Equation-equivalence mode (benchmark default): solve Warp FIRST, evaluate
            # C_gh at its converged head, and hand MF6 that fixed
            # conductance.  Both solvers then discretise the SAME linearised
            # operator, so head agreement validates numerical consistency of
            # the two equation solvers — NOT the physical conductance
            # formulation (that is what the fixed_point mode tests).
            if not do_run_warp:
                raise ValueError(
                    "ghb_conductance_mode='warp_matched' requires do_run_warp=True "
                    "(the conductance is evaluated at Warp's converged head)."
                )
            _run_warp()
            warp_already_run = True
            _pre_info = _load_npz_json(warp_path, "info_solve2")
            _require_warp_converged(
                _pre_info,
                context=f"Warp pre-solve for warp_matched conductance (workspace={case.workspace})",
            )
            with np.load(warp_path, allow_pickle=False) as _warp_npz:
                _warp_heads = np.asarray(_warp_npz["heads"], dtype=np.float64)
            ghb_conductance = _ghb_conductance_from_heads(case, _warp_heads)
            ghb_fixed_point_info = {
                "method": "warp_matched_equation_equivalence",
                "initialisation": "C_gh evaluated at Warp's converged head",
                "interpretation": (
                    "equation-equivalence test: both solvers discretise the "
                    "same operator; validates numerical consistency, not the "
                    "physical conductance formulation"
                ),
            }
            run_mf6_unconfined(
                case,
                out_path=mf6_path,
                ghb_conductance=ghb_conductance,
                mf6_budget_discrepancy_tol=mf6_budget_discrepancy_tol,
                mf6_head_change_min=mf6_head_change_min,
                artifact_extra={
                    "ghb_fixed_point": np.asarray(json.dumps(ghb_fixed_point_info)),
                },
            )
        elif case.use_ghb:
            # MF6-side fixed point for the head-dependent GHB conductance;
            # Warp heads are NEVER used to build the MF6 truth.
            _, ghb_fixed_point_info = run_mf6_ghb_fixed_point(
                case,
                out_path=mf6_path,
                cond_rtol=ghb_cond_rtol,
                head_atol=ghb_head_atol,
                max_iterations=ghb_max_fixed_point_iterations,
                mf6_budget_discrepancy_tol=mf6_budget_discrepancy_tol,
                mf6_head_change_min=mf6_head_change_min,
            )
        else:
            run_mf6_unconfined(
                case,
                out_path=mf6_path,
                mf6_budget_discrepancy_tol=mf6_budget_discrepancy_tol,
                mf6_head_change_min=mf6_head_change_min,
            )
    elif mf6_path.exists():
        validate_mf6_artifact(mf6_path, case, raise_on_mismatch=True)
        print(f"Validated cached MF6 artifact {mf6_path} (schema + case fingerprint match).")

    if case.use_ghb and mf6_path.exists():
        # Recover the converged fixed conductance recorded with the artifact
        # (fallback: re-evaluate the terminal law at the MF6 heads, which is
        # identical at the converged fixed point).  In warp_matched mode the
        # conductance evaluated at Warp's head is already in hand and is the
        # value MF6 actually used — keep it.
        if ghb_conductance is None:
            with np.load(mf6_path, allow_pickle=False) as _mf6_npz:
                if "ghb_conductance" in _mf6_npz:
                    ghb_conductance = np.asarray(_mf6_npz["ghb_conductance"], dtype=np.float64)
                else:
                    ghb_conductance = _ghb_conductance_from_heads(
                        case, np.asarray(_mf6_npz["heads"], dtype=np.float64)
                    )
        if ghb_fixed_point_info is None:
            loaded_fp = _load_npz_json(mf6_path, "ghb_fixed_point")
            ghb_fixed_point_info = loaded_fp if isinstance(loaded_fp, dict) and loaded_fp else None

    # --- Warp solve. ---
    if do_run_warp and not warp_already_run:
        _run_warp()

    # --- Acceptance: cached/fresh Warp heads must match this case AND come
    # from a converged solve before they may be used for anything. ---
    warp_info: dict = {}
    solve2_report: dict = _convergence_report({})
    if warp_path.exists():
        validate_warp_artifact(warp_path, case, raise_on_mismatch=True)
        warp_info = _load_npz_json(warp_path, "info_solve2")
        if not warp_info:
            raise RuntimeError(
                f"Warp artifact {warp_path} carries no solve info; refusing to "
                "use unverifiable Warp heads."
            )
        solve2_report = _require_warp_converged(
            warp_info, context=f"Warp solve (workspace={case.workspace})"
        )

    metrics = {}
    ghb_cell_metrics: dict = {}
    ghb_coupling_ratio: dict = {}
    if mf6_path.exists() and warp_path.exists():
        metrics = compare_results(mf6_path, warp_path, active=case.active)
        solve2_report = _convergence_report(warp_info, comparison=metrics)
        if case.use_ghb:
            mf6_heads, warp_heads = load_results(mf6_path, warp_path)
            ghb_cell_metrics = _ghb_cell_comparison(case, mf6_heads, warp_heads)
            if ghb_conductance is not None:
                ghb_coupling_ratio = _ghb_coupling_ratio(case, mf6_heads, ghb_conductance)
    else:
        print("Skipping comparison because both MF6 and Warp heads were not generated or found.")

    if case.use_ghb and conductance_mode == "warp_matched":
        ghb_conductance_note = (
            "warp_matched equation-equivalence mode: MF6 fixed conductance = "
            "C_gh evaluated at Warp's converged head.  Both solvers discretise "
            "the same operator, so head agreement validates numerical "
            "consistency between the Warp and MF6 equation solvers; it is NOT "
            "an independent test of the head-dependent conductance law (use "
            "ghb_conductance_mode='fixed_point' for that)."
        )
    elif case.use_ghb:
        ghb_conductance_note = (
            "MF6-side fixed point: conductance initialised at the case initial "
            "head, MF6 re-run with C_gh re-evaluated at its own converged heads "
            "until the max relative conductance change and max head change meet "
            "their tolerances (Warp heads are never used).  Because MF6 takes a "
            "fixed conductance while Warp updates C_gh(h) every Picard "
            "iteration, the two models solve slightly different discrete "
            "problems away from the fixed point; the residual Warp-vs-MF6 head "
            "difference at GHB cells is reported separately in "
            "comparison_ghb_cells."
        )
    else:
        ghb_conductance_note = None

    solve1_info = _load_npz_json(warp_path, "info_solve1")
    summary = {
        "nx": int(nx),
        "ny": int(ny),
        "n_cells": int(nx * ny),
        "dx": float(dx),
        "workspace": str(case.workspace),
        "artifact_schema_version": int(ARTIFACT_SCHEMA_VERSION),
        "case_fingerprint": fingerprint,
        "chebyshev_enabled": bool(chebyshev_enabled),
        "inner_smoother": str(inner_smoother),
        "cheby_lambda_min": float(cheby_lambda_min),
        "cheby_lambda_max": float(cheby_lambda_max),
        "diag_preconditioner_backend": str(diag_preconditioner_backend),
        "check_every_no": None if check_every_no is None else int(check_every_no),
        "unconfined_startup_mode": str(unconfined_startup_mode),
        "inner_implementation": str(inner_implementation),
        "t_field_kind": str(case.t_field_kind),
        "t_field_seed": int(case.t_field_seed),
        "use_ghb": bool(case.use_ghb),
        "ghb_width": float(case.ghb_width),
        "ghb_cell_count": int(np.count_nonzero(case.gh_mask)) if case.use_ghb else 0,
        "ghb_conductance_mode": conductance_mode if case.use_ghb else None,
        "ghb_conductance_note": ghb_conductance_note,
        "ghb_fixed_point": ghb_fixed_point_info,
        "ghb_conductance_min": (
            float(np.min(ghb_conductance[case.gh_mask != 0]))
            if ghb_conductance is not None and np.any(case.gh_mask != 0) else None
        ),
        "ghb_conductance_max": (
            float(np.max(ghb_conductance[case.gh_mask != 0]))
            if ghb_conductance is not None and np.any(case.gh_mask != 0) else None
        ),
        "ghb_coupling_ratio": ghb_coupling_ratio,
        "mf6_budget_discrepancy_max": _load_npz_scalar(mf6_path, "budget_discrepancy_max"),
        "mf6_budget_discrepancy_tol": (
            None if mf6_budget_discrepancy_tol is None else float(mf6_budget_discrepancy_tol)
        ),
        "mf6_head_change_max_from_initial": _load_npz_scalar(mf6_path, "head_change_max_from_initial"),
        "mf6_head_change_rms_from_initial": _load_npz_scalar(mf6_path, "head_change_rms_from_initial"),
        "mf6_engine_time": _load_npz_scalar(mf6_path, "engine_time"),
        "mf6_total_time": _load_npz_scalar(mf6_path, "total_time"),
        "warp_total_time": _load_npz_scalar(warp_path, "total_time"),
        "warp_solve1_time": _load_npz_scalar(warp_path, "solve1_time"),
        "warp_solve2_time": _load_npz_scalar(warp_path, "solve2_time"),
        "warp_benchmark_time": _load_npz_scalar(warp_path, "solve2_time"),
        "solve1_converged": bool(solve1_info.get("converged", False)) if solve1_info else None,
        "solve2_converged": bool(warp_info.get("converged", False)) if warp_info else None,
        "solve2_inner_smoother": warp_info.get("smoother"),
        "solve2_outer_iterations": warp_info.get("outer_iterations"),
        "solve2_final_max_abs_head_change": _finite_float(warp_info.get("final_max_abs_head_change")),
        "solve2_final_residual": _finite_float(warp_info.get("final_residual")),
        "solve2_final_h_rms_inner_residual": _finite_float(warp_info.get("inner_h_rms_end")),
        "solve2_chebyshev_rejections": warp_info.get("chebyshev_rejections"),
        "solve2_chebyshev_resets": warp_info.get("chebyshev_resets"),
        "solve2_strict_inner_nonconvergence_count": warp_info.get("strict_inner_nonconvergence_count"),
        "solve2_unusable_inner_solve_count": warp_info.get("unusable_inner_solve_count"),
        "solve2_practical_inner_acceptance_count": warp_info.get("practical_inner_acceptance_count"),
        "solve2_accepted_picard_update_count": warp_info.get("accepted_picard_update_count"),
        "solve2_outer_chebyshev_ready_count": warp_info.get("outer_chebyshev_ready_count"),
        "solve2_outer_chebyshev_used_count": warp_info.get("outer_chebyshev_used_count"),
        "solve2_outer_chebyshev_reset_count": warp_info.get("outer_chebyshev_reset_count"),
        "solve2_inner_forcing_eta": warp_info.get("inner_forcing_eta"),
        "solve2_inner_head_residual_tol_min": warp_info.get("inner_head_residual_tol_min"),
        "solve2_inner_head_residual_tol_max": warp_info.get("inner_head_residual_tol_max"),
        "solve2_inner_solve_failures": warp_info.get("inner_solve_failures"),
        "solve2_effectively_dry_cell_count": warp_info.get("effectively_dry_cell_count"),
        "convergence_report": solve2_report,
        "comparison": metrics,
        "comparison_ghb_cells": ghb_cell_metrics,
    }

    summary_path = case.workspace.joinpath("unconfined_benchmark_summary.json")
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=4)
    print(f"Benchmark summary saved to {summary_path}")
    print(f"Solve 2 convergence report: {solve2_report['status']}")
    return summary


def run_grid_benchmark(
    grid_sizes: list[int | tuple[int, int]] | tuple[int | tuple[int, int], ...] = tuple(BENCHMARK_GRID_SIZES),
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
    device: str = "auto",
    chebyshev_enabled: bool = True,
    inner_smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
    do_double_solve: bool = True,
    solver_backend: str | None = None,
    inner_implementation: str = "fast",
    t_field_kind: str = "uniform",
    t_field_seed: int = 42,
    use_ghb: bool = False,
    ghb_width: float = 100.0,
    ghb_head_elevation: float | None = None,
    ghb_conductance_mode: str = "warp_matched",
    mf6_budget_discrepancy_tol: float | None = DEFAULT_MF6_BUDGET_DISCREPANCY_TOL,
    mf6_head_change_min: float | None = DEFAULT_MF6_HEAD_CHANGE_MIN,
    ghb_cond_rtol: float = DEFAULT_GHB_COND_RTOL,
    ghb_head_atol: float = DEFAULT_GHB_HEAD_ATOL,
    ghb_max_fixed_point_iterations: int = DEFAULT_GHB_MAX_FIXED_POINT_ITERATIONS,
) -> list[dict]:
    """
    Run the 2D unconfined MF6-vs-Warp benchmark over a range of grid sizes.
    """
    if workspace is None:
        workspace = data_store.joinpath("working_tests", "mf6_vs_warp_2d_unconfined_grid_benchmark")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    normalized_sizes: list[tuple[int, int]] = []
    for item in grid_sizes:
        if isinstance(item, tuple):
            nx_i, ny_i = item
        else:
            nx_i = int(item)
            ny_i = int(item)
        normalized_sizes.append((int(nx_i), int(ny_i)))

    print("\n" + "=" * 72)
    print("2D unconfined Warp vs MF6 grid-size benchmark")
    print(f"grid sizes: {normalized_sizes}")
    print(f"dx: {dx}")
    print(f"inner smoother: {inner_smoother}")
    print(f"workspace: {workspace}")
    print("=" * 72)

    current_keys = [(int(nx), int(ny)) for nx, ny in normalized_sizes]
    summary_path = workspace.joinpath("grid_benchmark_summary.json")
    previous_results: dict[tuple[int, int], dict] = {}
    results_dict: dict[tuple[int, int], dict] = {}
    if summary_path.exists():
        try:
            with summary_path.open("r") as f:
                existing = json.load(f)
            for row in existing:
                previous_results[(int(row["nx"]), int(row["ny"]))] = row
        except Exception:
            pass

    for nx, ny in normalized_sizes:
        print("\n" + "-" * 72)
        print(f"Benchmark grid: nx={nx}, ny={ny}")
        print("-" * 72)

        case_workspace = workspace.joinpath(f"grid_{nx:04d}x{ny:04d}")
        row = run_case(
            nx=nx,
            ny=ny,
            dx=dx,
            hydraulic_conductivity=hydraulic_conductivity,
            recharge=recharge,
            initial_saturated_thickness=initial_saturated_thickness,
            workspace=case_workspace,
            device=device,
            chebyshev_enabled=chebyshev_enabled,
            inner_smoother=inner_smoother,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
            do_run_mf6=do_run_mf6,
            do_run_warp=do_run_warp,
            do_double_solve=do_double_solve,
            solver_backend=solver_backend,
            inner_implementation=inner_implementation,
            t_field_kind=t_field_kind,
            t_field_seed=t_field_seed,
            use_ghb=use_ghb,
            ghb_width=ghb_width,
            ghb_head_elevation=ghb_head_elevation,
            ghb_conductance_mode=ghb_conductance_mode,
            mf6_budget_discrepancy_tol=mf6_budget_discrepancy_tol,
            mf6_head_change_min=mf6_head_change_min,
            ghb_cond_rtol=ghb_cond_rtol,
            ghb_head_atol=ghb_head_atol,
            ghb_max_fixed_point_iterations=ghb_max_fixed_point_iterations,
        )

        key = (int(nx), int(ny))
        old_row = previous_results.get(key)
        if old_row is not None and old_row.get("case_fingerprint") != row.get("case_fingerprint"):
            # Never inherit comparison/convergence metrics from an
            # incompatible prior configuration: carried-forward data is
            # keyed on the case fingerprint; on mismatch it is dropped.
            print(
                "  Prior summary row has a different case fingerprint; "
                "dropping all carried-forward data for this grid."
            )
            old_row = None
        if old_row is not None:
            if not do_run_mf6:
                row["mf6_engine_time"] = old_row.get("mf6_engine_time")
                row["mf6_total_time"] = old_row.get("mf6_total_time")
            if not do_run_warp:
                for name in (
                    "warp_total_time",
                    "warp_solve1_time",
                    "warp_solve2_time",
                    "warp_benchmark_time",
                    "solve1_converged",
                    "solve2_converged",
                    "solve2_inner_smoother",
                    "solve2_outer_iterations",
                    "solve2_final_max_abs_head_change",
                    "solve2_final_residual",
                    "solve2_chebyshev_rejections",
                    "solve2_chebyshev_resets",
                    "solve2_inner_solve_failures",
                    "solve2_effectively_dry_cell_count",
                    "convergence_report",
                ):
                    row[name] = old_row.get(name)
            if not (do_run_mf6 and do_run_warp) and not row.get("comparison"):
                row["comparison"] = old_row.get("comparison", {})

        results_dict[key] = row
        results = [results_dict[k] for k in current_keys if k in results_dict]
        with summary_path.open("w") as f:
            json.dump(results, f, indent=4)
        print(f"Updated grid benchmark summary: {summary_path}")

    print("\nGrid benchmark complete.")
    return [results_dict[k] for k in current_keys if k in results_dict]


def run_diag_preconditioner_backend_matrix(
    grid_sizes: list[int | tuple[int, int]] | tuple[int | tuple[int, int], ...] = tuple(BENCHMARK_GRID_SIZES),
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
    device: str = "auto",
    chebyshev_enabled: bool = True,
    inner_smoother: str = "chebyshev",
    cheby_lambda_min: float = 0.1,
    cheby_lambda_max: float = 2.0,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    do_run_mf6: bool = True,
    do_run_warp: bool = True,
    solver_backend: str | None = None,
) -> list[dict]:
    """
    Run the tuned backend/check-frequency benchmark matrix for unconfined 2D cases.
    """
    if workspace is None:
        workspace = data_store.joinpath("working_tests", "mf6_vs_warp_2d_unconfined_backend_matrix")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    normalized_sizes: list[tuple[int, int]] = []
    for item in grid_sizes:
        if isinstance(item, tuple):
            nx_i, ny_i = item
        else:
            nx_i = int(item)
            ny_i = int(item)
        normalized_sizes.append((int(nx_i), int(ny_i)))

    scenarios = (
        {"case_id": "A", "diag_preconditioner_backend": "host", "check_every_no": 1},
        {"case_id": "B", "diag_preconditioner_backend": "host", "check_every_no": 5},
        {"case_id": "C", "diag_preconditioner_backend": "device", "check_every_no": 5},
        {"case_id": "D", "diag_preconditioner_backend": "device", "check_every_no": 10},
    )

    results: list[dict] = []
    for scenario in scenarios:
        case_id = str(scenario["case_id"])
        backend = str(scenario["diag_preconditioner_backend"])
        check_every_no = int(scenario["check_every_no"])
        for nx, ny in normalized_sizes:
            case_workspace = workspace.joinpath(
                f"{case_id}_backend_{backend}_check_{check_every_no}_grid_{nx:04d}x{ny:04d}"
            )
            row = run_case(
                nx=nx,
                ny=ny,
                dx=dx,
                hydraulic_conductivity=hydraulic_conductivity,
                recharge=recharge,
                initial_saturated_thickness=initial_saturated_thickness,
                workspace=case_workspace,
                device=device,
                chebyshev_enabled=chebyshev_enabled,
                inner_smoother=inner_smoother,
                cheby_lambda_min=cheby_lambda_min,
                cheby_lambda_max=cheby_lambda_max,
                inner_forcing_eta=inner_forcing_eta,
                inner_head_residual_tol_min=inner_head_residual_tol_min,
                inner_head_residual_tol_max=inner_head_residual_tol_max,
                chebyshev_reset_factor=chebyshev_reset_factor,
                transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
                unconfined_startup_mode=unconfined_startup_mode,
                diag_preconditioner_backend=backend,
                check_every_no=check_every_no,
                do_run_mf6=do_run_mf6,
                do_run_warp=do_run_warp,
                solver_backend=solver_backend,
            )
            row["case_id"] = case_id
            results.append(row)

    summary_json_path = workspace.joinpath("backend_matrix_summary.json")
    with summary_json_path.open("w") as f:
        json.dump(results, f, indent=4)

    if results:
        summary_csv_path = workspace.joinpath("backend_matrix_summary.csv")
        columns = list(results[0].keys())
        with summary_csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)

    print(f"Backend matrix summary saved to {summary_json_path}")
    return results


def run_chebyshev_lambda_sweep(
    nx: int = 500,
    ny: int = 500,
    dx: float = 100.0,
    hydraulic_conductivity: float = 10.0,
    recharge: float = 1.0e-4,
    initial_saturated_thickness: float = 100.0,
    cheby_lambda_min_values: list[float] | tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.5),
    cheby_lambda_max_values: list[float] | tuple[float, ...] = (1.7, 1.8, 1.95, 2.0, 2.1, 2.2, 2.5),
    workspace: str | Path | None = None,
    device: str = "auto",
    do_run_mf6: bool = True,
    inner_forcing_eta: float = 0.10,
    inner_head_residual_tol_min: float = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max: float = 1.0e-2,
    chebyshev_reset_factor: float = 1.2,
    transmissivity_relaxation_enabled: bool = False,
    unconfined_startup_mode: str = "confined_pre_solve",
    diag_preconditioner_backend: str = "auto",
    check_every_no: int | None = None,
    do_double_solve: bool = True,
    solver_backend: str | None = None,
) -> list[dict]:
    """
    Run a single 2D unconfined case across a range of Chebyshev lambda bounds.

    MF6 is run once and reused for comparison. A CSV summary of the sweep is
    written to the workspace root.

    :param nx: number of columns.
    :param ny: number of rows.
    :param dx: cell size.
    :param hydraulic_conductivity: uniform K value.
    :param recharge: recharge rate.
    :param initial_saturated_thickness: initial saturated thickness.
    :param cheby_lambda_min_values: iterable of lower Chebyshev bounds to test.
    :param cheby_lambda_max_values: iterable of upper Chebyshev bounds to test.
    :param workspace: root directory for the sweep outputs.
    :param device: Warp device.
    :param do_run_mf6: whether to run the MF6 truth model.
    :param inner_forcing_eta: dynamic inner tolerance fraction.
    :param inner_head_residual_tol_max: dynamic inner tolerance ceiling.
    :param chebyshev_reset_factor: residual-increase reset threshold.
    :param transmissivity_relaxation_enabled: optional T-relaxation flag.
    :param unconfined_startup_mode: "initial_head" or "confined_pre_solve".
    :return: list of per-combination result dictionaries.
    """
    if workspace is None:
        workspace = data_store.joinpath(
            "working_tests",
            "mf6_vs_warp_2d_unconfined_lambda_sweep",
        )
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    base_case = build_simple_unconfined_case(
        nx=int(nx),
        ny=int(ny),
        dx=float(dx),
        hydraulic_conductivity=float(hydraulic_conductivity),
        recharge=float(recharge),
        initial_saturated_thickness=float(initial_saturated_thickness),
        workspace=workspace,
    )

    mf6_path = workspace.joinpath("mf6_heads.npz")
    if do_run_mf6:
        run_mf6_unconfined(base_case, out_path=mf6_path)

    results: list[dict] = []
    for lambda_min in cheby_lambda_min_values:
        for lambda_max in cheby_lambda_max_values:
            if float(lambda_min) >= float(lambda_max):
                print(f"Skipping invalid lambda pair: min={lambda_min}, max={lambda_max}")
                continue

            combo_workspace = workspace.joinpath(f"lambda_min_{float(lambda_min):.4f}_max_{float(lambda_max):.4f}")
            combo_workspace.mkdir(parents=True, exist_ok=True)
            combo_case = build_simple_unconfined_case(
                nx=int(nx),
                ny=int(ny),
                dx=float(dx),
                hydraulic_conductivity=float(hydraulic_conductivity),
                recharge=float(recharge),
                initial_saturated_thickness=float(initial_saturated_thickness),
                workspace=combo_workspace,
            )

            warp_path = combo_workspace.joinpath("warp_heads.npz")
            run_warp_unconfined(
                combo_case,
                out_path=warp_path,
                device=device,
                chebyshev_enabled=True,
                inner_smoother="chebyshev",
                cheby_lambda_min=float(lambda_min),
                cheby_lambda_max=float(lambda_max),
                inner_forcing_eta=float(inner_forcing_eta),
                inner_head_residual_tol_min=float(inner_head_residual_tol_min),
                inner_head_residual_tol_max=float(inner_head_residual_tol_max),
                chebyshev_reset_factor=float(chebyshev_reset_factor),
                transmissivity_relaxation_enabled=bool(transmissivity_relaxation_enabled),
                unconfined_startup_mode=str(unconfined_startup_mode),
                diag_preconditioner_backend=str(diag_preconditioner_backend),
                check_every_no=check_every_no,
                do_double_solve=do_double_solve,
            )

            metrics = {}
            if mf6_path.exists() and warp_path.exists():
                metrics = compare_results(mf6_path, warp_path, active=combo_case.active)

            warp_info = _load_npz_json(warp_path, "info_solve2")
            row = {
                "nx": int(nx),
                "ny": int(ny),
                "dx": float(dx),
                "cheby_lambda_min": float(lambda_min),
                "cheby_lambda_max": float(lambda_max),
                "diag_preconditioner_backend": str(diag_preconditioner_backend),
                "check_every_no": None if check_every_no is None else int(check_every_no),
                "unconfined_startup_mode": str(unconfined_startup_mode),
                "converged": bool(warp_info.get("converged", False)) if warp_info else None,
                "outer_iterations": warp_info.get("outer_iterations"),
                "final_max_abs_head_change": _finite_float(warp_info.get("final_max_abs_head_change")),
                "final_residual": _finite_float(warp_info.get("final_residual")),
                "inner_h_rms_end": _finite_float(warp_info.get("inner_h_rms_end")),
                "unusable_inner_solve_count": warp_info.get("unusable_inner_solve_count"),
                "practical_inner_acceptance_count": warp_info.get("practical_inner_acceptance_count"),
                "accepted_picard_update_count": warp_info.get("accepted_picard_update_count"),
                "outer_chebyshev_ready_count": warp_info.get("outer_chebyshev_ready_count"),
                "outer_chebyshev_used_count": warp_info.get("outer_chebyshev_used_count"),
                "outer_chebyshev_reset_count": warp_info.get("outer_chebyshev_reset_count"),
                "solve2_time": _load_npz_scalar(warp_path, "solve2_time"),
                "rmse": metrics.get("rmse"),
                "max_abs_diff": metrics.get("max_abs_diff"),
                "workspace": str(combo_workspace),
            }
            results.append(row)
            print(
                f"lambda_min={lambda_min:.4f} lambda_max={lambda_max:.4f} -> "
                f"converged={row['converged']} outer_iter={row['outer_iterations']} "
                f"time={_format_optional_float(row['solve2_time'], '.4f')}s "
                f"rmse={_format_optional_float(row['rmse'], '.6g')} "
                f"max_abs_diff={_format_optional_float(row['max_abs_diff'], '.6g')}"
            )

    summary_path = workspace.joinpath("lambda_sweep_summary.csv")
    if results:
        columns = list(results[0].keys())
        with summary_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nLambda sweep summary saved to {summary_path}")

    return results


def main(
    grid_sizes=BENCHMARK_GRID_SIZES,
    dx=100.0,
    hydraulic_conductivity=100.0,
    recharge=1.0e-4,
    initial_saturated_thickness=100.0,
    workspace=None,
    device="auto",
    chebyshev_enabled=True,
    inner_smoother="chebyshev",
    cheby_lambda_min=0.1,
    cheby_lambda_max=2.0,
    inner_forcing_eta=0.10,
    inner_head_residual_tol_min=DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN,
    inner_head_residual_tol_max=1.0e-2,
    chebyshev_reset_factor=1.2,
    transmissivity_relaxation_enabled=False,
    unconfined_startup_mode="confined_pre_solve",
    diag_preconditioner_backend="device",
    check_every_no=5,
    do_run_mf6=False,
    do_run_warp=True,
    run_lambda_sweep=False,
    run_backend_matrix=False,
    do_double_solve=False,
    solver_backend=None,
    inner_implementation="fast",
    t_field_kind="uniform",
    t_field_seed=42,
    use_ghb=False,
    ghb_width=100.0,
    ghb_head_elevation=None,
    ghb_conductance_mode="warp_matched",
    mf6_budget_discrepancy_tol=DEFAULT_MF6_BUDGET_DISCREPANCY_TOL,
    mf6_head_change_min=DEFAULT_MF6_HEAD_CHANGE_MIN,
    ghb_cond_rtol=DEFAULT_GHB_COND_RTOL,
    ghb_head_atol=DEFAULT_GHB_HEAD_ATOL,
    ghb_max_fixed_point_iterations=DEFAULT_GHB_MAX_FIXED_POINT_ITERATIONS,
):
    if run_backend_matrix:
        results = run_diag_preconditioner_backend_matrix(
            grid_sizes=grid_sizes,
            dx=dx,
            hydraulic_conductivity=hydraulic_conductivity,
            recharge=recharge,
            initial_saturated_thickness=initial_saturated_thickness,
            workspace=workspace,
            device=device,
            chebyshev_enabled=chebyshev_enabled,
            inner_smoother=inner_smoother,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            do_run_mf6=do_run_mf6,
            do_run_warp=do_run_warp,
            solver_backend=solver_backend,
        )
    elif run_lambda_sweep:
        results = run_chebyshev_lambda_sweep(
            nx=500,
            ny=500,
            dx=dx,
            hydraulic_conductivity=hydraulic_conductivity,
            recharge=recharge,
            initial_saturated_thickness=initial_saturated_thickness,
            cheby_lambda_min_values=(0.05, 0.1, 0.15, 0.2, 0.25, 0.5),
            cheby_lambda_max_values=(1.7, 1.8, 1.95, 2.0, 2.1, 2.2, 2.5),
            workspace=None,
            device=device,
            do_run_mf6=do_run_mf6,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
            solver_backend=solver_backend,
        )
    else:
        results = run_grid_benchmark(
            grid_sizes=grid_sizes,
            dx=dx,
            hydraulic_conductivity=hydraulic_conductivity,
            recharge=recharge,
            initial_saturated_thickness=initial_saturated_thickness,
            workspace=workspace,
            device=device,
            chebyshev_enabled=chebyshev_enabled,
            inner_smoother=inner_smoother,
            cheby_lambda_min=cheby_lambda_min,
            cheby_lambda_max=cheby_lambda_max,
            inner_forcing_eta=inner_forcing_eta,
            inner_head_residual_tol_min=inner_head_residual_tol_min,
            inner_head_residual_tol_max=inner_head_residual_tol_max,
            chebyshev_reset_factor=chebyshev_reset_factor,
            transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
            unconfined_startup_mode=unconfined_startup_mode,
            diag_preconditioner_backend=diag_preconditioner_backend,
            check_every_no=check_every_no,
            do_run_mf6=do_run_mf6,
            do_run_warp=do_run_warp,
            do_double_solve=do_double_solve,
            solver_backend=solver_backend,
            inner_implementation=inner_implementation,
            t_field_kind=t_field_kind,
            t_field_seed=t_field_seed,
            use_ghb=use_ghb,
            ghb_width=ghb_width,
            ghb_head_elevation=ghb_head_elevation,
            ghb_conductance_mode=ghb_conductance_mode,
            mf6_budget_discrepancy_tol=mf6_budget_discrepancy_tol,
            mf6_head_change_min=mf6_head_change_min,
            ghb_cond_rtol=ghb_cond_rtol,
            ghb_head_atol=ghb_head_atol,
            ghb_max_fixed_point_iterations=ghb_max_fixed_point_iterations,
        )
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    # Configuration parameters
    grid_sizes = BENCHMARK_GRID_SIZES
    dx = 100.0
    hydraulic_conductivity = 100.0
    recharge = 1.0e-4
    initial_saturated_thickness = 100.0
    workspace = None
    device = "auto"
    chebyshev_enabled = True
    inner_smoother = "chebyshev"
    cheby_lambda_min = 0.1
    cheby_lambda_max = 2.0
    inner_forcing_eta = 0.10
    inner_head_residual_tol_min = DEFAULT_INNER_HEAD_RESIDUAL_TOL_MIN
    inner_head_residual_tol_max = 1.0e-2
    chebyshev_reset_factor = 1.2
    transmissivity_relaxation_enabled = False
    unconfined_startup_mode = "confined_pre_solve"  # or "initial_head"
    diag_preconditioner_backend = "device"
    check_every_no = 5
    do_run_mf6 = True
    do_run_warp = True
    run_lambda_sweep = False
    run_backend_matrix = False
    do_double_solve = False
    solver_backend = None  # Use None for default picard, or "unconfined_fas", or "unconfined_semismooth_newton_kcycle"
    inner_implementation = "fast"  # "classic" or "fast" (face-array inner K-cycle; steady only)
    use_ghb = True  # center-row GHB boundary (gh_head=DEM unless ghb_head_elevation is set, width=ghb_width)
    ghb_width = 100.0
    ghb_head_elevation = None  # m above bottom; None -> stage = DEM (historical)
    # Primary benchmark intent: give MF6 the exact conductance evaluated at
    # Warp's converged head and compare the same discrete operator. Select
    # "fixed_point" explicitly for the slower independent conductance-law test.
    ghb_conductance_mode = "warp_matched"
    t_field_kind = "ugly_t"  # "uniform" or "ugly_t" (hard heterogeneous K ~ 4-535 m/day)
    t_field_seed = 42
    mf6_budget_discrepancy_tol = DEFAULT_MF6_BUDGET_DISCREPANCY_TOL
    mf6_head_change_min = DEFAULT_MF6_HEAD_CHANGE_MIN
    ghb_cond_rtol = DEFAULT_GHB_COND_RTOL
    ghb_head_atol = DEFAULT_GHB_HEAD_ATOL
    ghb_max_fixed_point_iterations = DEFAULT_GHB_MAX_FIXED_POINT_ITERATIONS

    main(
        grid_sizes=grid_sizes,
        dx=dx,
        hydraulic_conductivity=hydraulic_conductivity,
        recharge=recharge,
        initial_saturated_thickness=initial_saturated_thickness,
        workspace=workspace,
        device=device,
        chebyshev_enabled=chebyshev_enabled,
        inner_smoother=inner_smoother,
        cheby_lambda_min=cheby_lambda_min,
        cheby_lambda_max=cheby_lambda_max,
        inner_forcing_eta=inner_forcing_eta,
        inner_head_residual_tol_min=inner_head_residual_tol_min,
        inner_head_residual_tol_max=inner_head_residual_tol_max,
        chebyshev_reset_factor=chebyshev_reset_factor,
        transmissivity_relaxation_enabled=transmissivity_relaxation_enabled,
        unconfined_startup_mode=unconfined_startup_mode,
        diag_preconditioner_backend=diag_preconditioner_backend,
        check_every_no=check_every_no,
        do_run_mf6=do_run_mf6,
        do_run_warp=do_run_warp,
        run_lambda_sweep=run_lambda_sweep,
        run_backend_matrix=run_backend_matrix,
        do_double_solve=do_double_solve,
        solver_backend=solver_backend,
        inner_implementation=inner_implementation,
        t_field_kind=t_field_kind,
        t_field_seed=t_field_seed,
        use_ghb=use_ghb,
        ghb_width=ghb_width,
        ghb_head_elevation=ghb_head_elevation,
        ghb_conductance_mode=ghb_conductance_mode,
        mf6_budget_discrepancy_tol=mf6_budget_discrepancy_tol,
        mf6_head_change_min=mf6_head_change_min,
        ghb_cond_rtol=ghb_cond_rtol,
        ghb_head_atol=ghb_head_atol,
        ghb_max_fixed_point_iterations=ghb_max_fixed_point_iterations,
    )
