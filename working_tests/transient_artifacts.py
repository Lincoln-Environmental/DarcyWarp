from __future__ import annotations

import io
import hashlib
import json
import lzma
from pathlib import Path

import numpy as np

from DARCY_WARP_PACKAGE.project_base import data_store
from DARCY_WARP_PACKAGE.model_builder import _build_dirichlet_boundary_mask, _build_domain

# Local copies keep this module independent of the replay facade.
FORMULATION_CONFINED = "confined"
FORMULATION_UNCONFINED = "unconfined"
FORMULATION_MODES = {FORMULATION_CONFINED, FORMULATION_UNCONFINED}

UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY = "mf6_convertible_secant_sy"
UNCONFINED_STORAGE_MODES = {UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY}

DEFAULT_ARTIFACT_NAME = "mf6_transient_heads.npz.lzma"
WARM_START_ARTIFACT_INITIAL = "artifact_initial"
WARM_START_CONFINED_STEADY_MF6 = "confined_steady_mf6"
WARM_START_CONFINED_STEADY_WARP = "confined_steady_warp"
WARM_START_UNCONFINED_STEADY_MF6 = "unconfined_steady_mf6"
WARM_START_UNCONFINED_STEADY_WARP = "unconfined_steady_warp"
WARM_START_MODES = {
    WARM_START_ARTIFACT_INITIAL,
    WARM_START_CONFINED_STEADY_MF6,
    WARM_START_CONFINED_STEADY_WARP,
    WARM_START_UNCONFINED_STEADY_MF6,
    WARM_START_UNCONFINED_STEADY_WARP,
}
WARM_START_ARTIFACT_MODES = {
    WARM_START_ARTIFACT_INITIAL,
    WARM_START_CONFINED_STEADY_MF6,
    WARM_START_UNCONFINED_STEADY_MF6,
}
WARM_START_WARP_SOLVE_MODES = {
    WARM_START_CONFINED_STEADY_WARP,
    WARM_START_UNCONFINED_STEADY_WARP,
}

# Bump whenever the serialized truth contract changes.  Old artifacts are
# intentionally rejected instead of being silently compared as if they were
# generated from the current equations.
ARTIFACT_SCHEMA_VERSION = 2

_FINGERPRINT_KEYS = (
    "nx", "ny", "dx", "active", "bc_mask", "bc_values", "top", "bottom",
    "k_field", "t_field_kind", "t_field_seed", "sy", "ss", "dt_days",
    "recharge_rates", "n_periods", "n_weeks", "initial_head",
    "initial_saturated_thickness",
    "warm_start_mode", "formulation", "ghb_mask", "ghb_head",
    "ghb_conductance", "ghb_conductance_mode", "ghb_conductance_settings",
)


def _load_compressed_npz(path: str | Path) -> dict:
    path = Path(path)
    buf = io.BytesIO(lzma.decompress(path.read_bytes()))
    with np.load(buf, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def _fingerprint_value(hasher: "hashlib._Hash", key: str, value: object) -> None:
    """Add a typed, shape-aware value to a deterministic case hash."""
    hasher.update(key.encode("utf-8"))
    hasher.update(b"\0")
    if value is None:
        hasher.update(b"<missing>\0")
        return
    array = np.asarray(value)
    hasher.update(str(array.dtype).encode("utf-8"))
    hasher.update(b"\0")
    hasher.update(repr(tuple(array.shape)).encode("ascii"))
    hasher.update(b"\0")
    if array.dtype.kind in {"U", "S", "O"}:
        text = json.dumps(array.reshape(-1).tolist(), ensure_ascii=False, sort_keys=True, default=str)
        hasher.update(text.encode("utf-8"))
    else:
        hasher.update(np.ascontiguousarray(array).tobytes(order="C"))
    hasher.update(b"\0")


def compute_transient_case_fingerprint(case_inputs: dict) -> str:
    """Return the SHA-256 identity of all transient equation inputs.

    Runtime settings, output heads, and timing values are deliberately absent.
    Missing optional GHB fields are hashed as missing, so adding or changing a
    GHB mode cannot reuse a no-GHB artifact.
    """
    hasher = hashlib.sha256()
    hasher.update(f"artifact-schema={ARTIFACT_SCHEMA_VERSION}\0".encode("ascii"))
    for key in _FINGERPRINT_KEYS:
        _fingerprint_value(hasher, key, case_inputs.get(key))
    return hasher.hexdigest()


case_fingerprint = compute_transient_case_fingerprint


def all_active_transient_heads_finite(
    *,
    heads_per_period: np.ndarray,
    active: np.ndarray,
) -> bool:
    """Return whether all active cells are finite without a large broadcast."""
    heads = np.asarray(heads_per_period)
    active_mask = np.asarray(active, dtype=bool)
    if heads.ndim != 3 or heads.shape[1:] != active_mask.shape:
        return False
    return all(np.isfinite(period[active_mask]).all() for period in heads)


def validate_transient_artifact(
    path: str | Path,
    *,
    expected_fingerprint: str | None = None,
    expected_periods: int | None = None,
    require_mf6_gates: bool = True,
) -> dict:
    """Load and validate a current, internally consistent truth artifact."""
    artifact_path = Path(path)
    artifact = _load_compressed_npz(artifact_path)
    stored_version = int(np.asarray(artifact.get("artifact_schema_version", -1)).reshape(()))
    if stored_version != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"stale transient artifact {artifact_path}: schema {stored_version}, "
            f"expected {ARTIFACT_SCHEMA_VERSION}."
        )
    stored_fingerprint = _scalar_string(artifact.get("case_fingerprint", ""))
    computed_fingerprint = compute_transient_case_fingerprint(artifact)
    if stored_fingerprint != computed_fingerprint:
        raise ValueError(f"corrupt or stale transient artifact {artifact_path}: fingerprint mismatch.")
    if expected_fingerprint is not None and stored_fingerprint != str(expected_fingerprint):
        raise ValueError(f"stale transient artifact {artifact_path}: requested case fingerprint differs.")
    required = (
        "heads_per_period", "heads_final", "initial_head", "active", "bc_mask",
        "bc_values", "top", "bottom", "k_field", "recharge_rates", "sy", "ss",
        "nx", "ny", "dx", "dt_days",
    )
    missing = [name for name in required if name not in artifact]
    if missing:
        raise ValueError(f"transient artifact {artifact_path} missing keys: {missing}")
    n_periods = int(np.asarray(artifact["heads_per_period"]).shape[0])
    expected = int(np.asarray(artifact.get("n_periods", artifact.get("n_weeks", n_periods))).reshape(()))
    if n_periods != expected or (expected_periods is not None and n_periods != int(expected_periods)):
        raise ValueError(f"transient artifact {artifact_path} has {n_periods} saved periods, expected {expected}.")
    active = np.asarray(artifact["active"], dtype=np.int32) != 0
    heads = np.asarray(artifact["heads_per_period"], dtype=np.float64)
    shape = (int(np.asarray(artifact["ny"]).reshape(())), int(np.asarray(artifact["nx"]).reshape(())))
    if heads.shape != (n_periods, *shape) or not all_active_transient_heads_finite(
        heads_per_period=heads,
        active=active,
    ):
        raise ValueError(f"transient artifact {artifact_path} has invalid head shape or non-finite active heads.")
    initial = np.asarray(artifact["initial_head"], dtype=np.float64)
    if initial.shape != shape or not np.isfinite(initial[active]).all():
        raise ValueError(f"transient artifact {artifact_path} has an invalid warm-start head.")
    if np.max(np.abs(heads[-1][active] - initial[active])) <= 1.0e-12:
        raise ValueError(f"transient artifact {artifact_path} has no nontrivial transient response.")
    if require_mf6_gates:
        if not bool(int(np.asarray(artifact.get("mf6_normal_termination", 0)).reshape(()))):
            raise ValueError(f"transient artifact {artifact_path} lacks MF6 normal-termination proof.")
        discrepancy = float(np.asarray(artifact.get("mf6_budget_discrepancy_max", np.nan)).reshape(()))
        tolerance = float(np.asarray(artifact.get("mf6_budget_discrepancy_tol", np.nan)).reshape(()))
        if not np.isfinite(discrepancy) or not np.isfinite(tolerance) or discrepancy > tolerance:
            raise ValueError(f"transient artifact {artifact_path} fails the MF6 budget gate.")
        ghb_mode = _scalar_string(artifact.get("ghb_conductance_mode", "none")).strip().lower()
        if ghb_mode == "mf6_fixed_point" and not bool(
            int(np.asarray(artifact.get("ghb_fixed_point_converged", 0)).reshape(()))
        ):
            raise ValueError(f"transient artifact {artifact_path} lacks a converged MF6 GHB fixed point.")
    return artifact


def load_transient_artifact(path: str | Path) -> dict:
    arrays = validate_transient_artifact(path)
    required = (
        "heads_per_period",
        "heads_final",
        "initial_head",
        "active",
        "bc_mask",
        "bc_values",
        "top",
        "bottom",
        "k_field",
        "recharge_rates",
        "sy",
        "ss",
        "nx",
        "ny",
        "dx",
        "dt_days",
    )
    missing = [name for name in required if name not in arrays]
    if missing:
        raise KeyError(f"transient artifact {path} missing keys: {missing}")
    return arrays


def _scalar_string(array: object) -> str:
    return str(np.asarray(array).reshape(()))


def artifact_formulation(artifact: dict, artifact_path: str | Path | None = None) -> str | None:
    if "formulation" in artifact:
        formulation = _scalar_string(artifact["formulation"]).strip().lower()
        if formulation in FORMULATION_MODES:
            return formulation
    provenance = artifact.get("provenance")
    if provenance is not None:
        try:
            provenance_data = json.loads(_scalar_string(provenance))
        except (TypeError, ValueError, json.JSONDecodeError):
            provenance_data = {}
        provenance_formulation = str(provenance_data.get("formulation", "")).strip().lower()
        if provenance_formulation in FORMULATION_MODES:
            return provenance_formulation
        kind = str(provenance_data.get("kind", "")).strip().lower()
        if "2d_confined_transient" in kind:
            return FORMULATION_CONFINED
        if "2d_unconfined_transient" in kind:
            return FORMULATION_UNCONFINED
    if artifact_path is not None:
        path_text = str(artifact_path).lower()
        if "mf6_transient_2d_confined" in path_text:
            return FORMULATION_CONFINED
        if "mf6_transient_2d_unconfined" in path_text:
            return FORMULATION_UNCONFINED
    return None


def require_matching_artifact_formulation(
    artifact: dict,
    requested_formulation: str,
    artifact_path: str | Path,
) -> str | None:
    artifact_mode = artifact_formulation(
        artifact=artifact,
        artifact_path=artifact_path,
    )
    if artifact_mode is not None and artifact_mode != requested_formulation:
        raise ValueError(
            f"requested formulation '{requested_formulation}' does not match MF6 artifact "
            f"formulation '{artifact_mode}' at {artifact_path}."
        )
    return artifact_mode


def build_synthetic_spatial_fields(
    nx: int = 16,
    ny: int = 12,
    dx: float = 100.0,
    hydraulic_conductivity: float = 100.0,
    initial_saturated_thickness: float = 100.0,
    workspace: str | Path | None = None,
) -> dict:
    active = _build_domain(nx=int(nx), ny=int(ny)).astype(np.int32)
    bc_bool = _build_dirichlet_boundary_mask(active)
    bc_mask = bc_bool.astype(np.int32)

    top = np.full((int(ny), int(nx)), 110.0, dtype=np.float64)
    bottom = np.full((int(ny), int(nx)), 10.0, dtype=np.float64)
    bc_values = np.full((int(ny), int(nx)), 100.0, dtype=np.float64)

    k_field = np.full((int(ny), int(nx)), float(hydraulic_conductivity), dtype=np.float64)
    k_field[active == 0] = 0.0

    initial_head = np.minimum(bottom + float(initial_saturated_thickness), top)
    initial_head[bc_mask != 0] = bc_values[bc_mask != 0]
    initial_head[active == 0] = 0.0

    return {
        "nx": int(nx),
        "ny": int(ny),
        "dx": float(dx),
        "active": active,
        "bc_mask": bc_mask,
        "bc_values": bc_values,
        "top": top,
        "bottom": bottom,
        "k": k_field,
        "initial_head": initial_head.astype(np.float64, copy=False),
        "workspace": Path(workspace) if workspace is not None else None,
    }


def spatial_fields_from_artifact(artifact: dict) -> dict:
    shape = (int(artifact["ny"]), int(artifact["nx"]))
    return {
        "nx": int(artifact["nx"]),
        "ny": int(artifact["ny"]),
        "dx": float(artifact["dx"]),
        "active": np.asarray(artifact["active"], dtype=np.int32),
        "bc_mask": np.asarray(artifact["bc_mask"], dtype=np.int32),
        "bc_values": np.asarray(artifact["bc_values"], dtype=np.float64),
        "top": np.asarray(artifact["top"], dtype=np.float64),
        "bottom": np.asarray(artifact["bottom"], dtype=np.float64),
        "k": np.asarray(artifact["k_field"], dtype=np.float64),
        "initial_head": np.asarray(artifact["initial_head"], dtype=np.float64),
        "ghb_mask": np.asarray(artifact.get("ghb_mask", np.zeros(shape, dtype=np.int32)), dtype=np.int32),
        "ghb_head": np.asarray(artifact.get("ghb_head", np.zeros(shape)), dtype=np.float64),
        "ghb_width": np.asarray(artifact.get("ghb_width", np.zeros(shape)), dtype=np.float64),
        "ghb_alpha": float(np.asarray(artifact.get("ghb_alpha", 1.0)).reshape(())),
        "ghb_aq_thickness": float(np.asarray(artifact.get("ghb_aq_thickness", 0.0)).reshape(())),
        "ghb_conductance_mode": str(np.asarray(artifact.get("ghb_conductance_mode", "none")).reshape(())),
        "ghb_conductance": np.asarray(artifact.get("ghb_conductance", np.zeros((0, *shape))), dtype=np.float64),
        "workspace": None,
    }


def validate_warm_start_head(
    head: np.ndarray,
    spatial: dict,
    label: str = "warm_start_head",
) -> np.ndarray:
    h = np.asarray(head, dtype=np.float64).copy()
    shape = (int(spatial["ny"]), int(spatial["nx"]))
    if h.shape != shape:
        raise ValueError(f"{label} shape {h.shape} expected {shape}.")
    if not np.all(np.isfinite(h)):
        raise ValueError(f"{label} contains non-finite values.")
    active = np.asarray(spatial["active"], dtype=np.int32)
    bc_mask = np.asarray(spatial["bc_mask"], dtype=np.int32)
    bc_values = np.asarray(spatial["bc_values"], dtype=np.float64)
    bottom = np.asarray(spatial["bottom"], dtype=np.float64)
    top = np.asarray(spatial["top"], dtype=np.float64)
    h[active == 0] = 0.0
    h[bc_mask != 0] = bc_values[bc_mask != 0]
    free = (active != 0) & (bc_mask == 0)
    if np.any(free):
        min_allowed = bottom[free] - 1.0e-6
        max_allowed = top[free] + max(100.0, 0.1 * float(np.nanmax(top[free] - bottom[free])))
        if np.any(h[free] < min_allowed):
            raise ValueError(f"{label} has active free-cell heads below cell bottom.")
        if np.any(h[free] > max_allowed):
            raise ValueError(f"{label} has implausibly high active free-cell heads.")
    return h.astype(np.float64, copy=False)


def select_artifact_warm_start(
    artifact: dict,
    spatial: dict,
    warm_start_mode: str,
) -> tuple[np.ndarray, str]:
    mode = str(warm_start_mode).strip().lower()
    if mode not in WARM_START_ARTIFACT_MODES:
        raise ValueError(f"artifact warm_start_mode must be one of {sorted(WARM_START_ARTIFACT_MODES)}.")
    if mode == WARM_START_ARTIFACT_INITIAL:
        return validate_warm_start_head(
            head=artifact["initial_head"],
            spatial=spatial,
            label="artifact initial_head",
        ), WARM_START_ARTIFACT_INITIAL
    if mode == WARM_START_CONFINED_STEADY_MF6:
        if "confined_steady_head" not in artifact:
            raise KeyError("warm_start_mode='confined_steady_mf6' requires artifact key 'confined_steady_head'.")
        return validate_warm_start_head(
            head=artifact["confined_steady_head"],
            spatial=spatial,
            label="artifact confined_steady_head",
        ), WARM_START_CONFINED_STEADY_MF6
    if "unconfined_steady_head" not in artifact:
        raise KeyError("warm_start_mode='unconfined_steady_mf6' requires artifact key 'unconfined_steady_head'.")
    return validate_warm_start_head(
        head=artifact["unconfined_steady_head"],
        spatial=spatial,
        label="artifact unconfined_steady_head",
    ), WARM_START_UNCONFINED_STEADY_MF6


def artifact_warm_start_provenance(artifact: dict) -> str | None:
    provenance = artifact.get("provenance")
    if provenance is None:
        return None
    try:
        data = json.loads(_scalar_string(provenance))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    mode = str(data.get("warm_start_mode", "")).strip().lower()
    return mode or None


def validate_warm_start_comparability(
    artifact_warm_start: str | None,
    warp_warm_start_mode: str,
    allow_warm_start_mismatch: bool = False,
) -> None:
    if artifact_warm_start is None:
        return
    if warp_warm_start_mode == WARM_START_ARTIFACT_INITIAL:
        return
    if artifact_warm_start == warp_warm_start_mode:
        return
    if allow_warm_start_mismatch:
        return
    raise ValueError(
        f"Warp warm_start_mode='{warp_warm_start_mode}' does not match the MF6 artifact warm-start provenance "
        f"'{artifact_warm_start}'."
    )


def default_artifact_path(formulation: str = FORMULATION_UNCONFINED) -> Path:
    formulation = str(formulation).strip().lower()
    if formulation not in FORMULATION_MODES:
        raise ValueError(f"formulation must be one of {sorted(FORMULATION_MODES)}.")
    return data_store.joinpath("working_tests", f"mf6_transient_2d_{formulation}", DEFAULT_ARTIFACT_NAME)
