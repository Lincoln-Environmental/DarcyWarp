from __future__ import annotations

import io
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

UNCONFINED_STORAGE_PHREATIC_ONLY = "phreatic_only"
UNCONFINED_STORAGE_INTEGRATED_SY_SS = "integrated_sy_ss"
UNCONFINED_STORAGE_MF6_CONVERTIBLE = "mf6_convertible"
UNCONFINED_STORAGE_MF6_CONVERTIBLE_TOP_SWITCH = "mf6_convertible_top_switch"
UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY = "mf6_convertible_secant_sy"
UNCONFINED_STORAGE_MF6_CONVERTIBLE_CROSSING_VOLUME_SY = "mf6_convertible_crossing_volume_sy"
UNCONFINED_STORAGE_MODES = {
    UNCONFINED_STORAGE_PHREATIC_ONLY,
    UNCONFINED_STORAGE_INTEGRATED_SY_SS,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_TOP_SWITCH,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_SECANT_SY,
    UNCONFINED_STORAGE_MF6_CONVERTIBLE_CROSSING_VOLUME_SY,
}

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


def _load_compressed_npz(path: str | Path) -> dict:
    path = Path(path)
    buf = io.BytesIO(lzma.decompress(path.read_bytes()))
    with np.load(buf, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def load_transient_artifact(path: str | Path) -> dict:
    arrays = _load_compressed_npz(path)
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
