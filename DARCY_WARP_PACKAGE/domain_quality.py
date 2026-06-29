"""Domain topology QA and conservative cleaning utilities.

This module inspects 2D raster groundwater model domains for geometric
features that often behave poorly under structured-grid coarsening. The
workflow is built around two boolean-like masks on the same grid:

``active_mask``
    Cells that belong to the simulated model domain.

``chd_mask``
    Cells carrying a fixed-head / CHD style boundary condition.

The public API is centered on :func:`audit_domain`,
:func:`clean_domain`, and :func:`run_domain_qa_and_clean`. The audit step
flags disconnected regions, thin features, bottlenecks, appendages, and
components that disappear or lose CHD connectivity when repeatedly
coarsened. The cleaning step then applies a small set of explicit,
auditable edits based on those diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage


STRUCTURE4 = np.array(
    [
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0],
    ],
    dtype=np.int8,
)


DEFAULT_DOMAIN_QA_CONFIG: dict[str, Any] = {
    "coarsening_levels": 6,
    "thin_width_threshold_cells": 2.0,
    "bottleneck_width_threshold_cells": 2.0,
    "remove_all_disconnected_without_chd": True,
    "remove_disconnected_max_cells": 16,
    "fill_holes_max_cells": 8,
    "prune_appendages": True,
    "appendage_max_cells": 24,
    "save_pngs": True,
    "png_dpi": 180,
    "report_component_limit": 20,
}


@dataclass
class DomainAuditResult:
    """Audit outputs for the original domain mask.

    :ivar active_mask: Validated boolean copy of the active domain mask.
    :ivar chd_mask: Validated boolean CHD mask clipped to ``active_mask``.
    :ivar component_labels: Integer labels for 4-neighbour connected active components.
    :ivar thinness_map: Estimated local feature width in model units for each cell.
    :ivar thin_mask: Cells whose estimated width is at or below the thin-feature threshold.
    :ivar disconnected_mask: Active cells that are not connected to any CHD-bearing component.
    :ivar bottleneck_mask: Narrow cells that look like one-cell necks or corridors.
    :ivar appendage_candidate_mask: Small non-CHD branches hanging off a candidate neck.
    :ivar problem_score_map: Integer score map combining the individual issue flags.
    :ivar component_summary: Per-component diagnostics and recommended action table.
    :ivar suspect_regions: Subset of ``component_summary`` containing flagged components.
    :ivar component_coarsening: Per-component survival/connectivity status by coarsening level.
    :ivar coarsening_levels: Raw mask and connectivity state stored for each coarsening level.
    :ivar summary: High-level audit metrics suitable for logs or reports.
    :ivar config: Resolved configuration after defaults and unit conversions are applied.
    :ivar dx: Grid spacing used for width and area calculations.
    """

    active_mask: np.ndarray
    chd_mask: np.ndarray
    component_labels: np.ndarray
    thinness_map: np.ndarray
    thin_mask: np.ndarray
    disconnected_mask: np.ndarray
    bottleneck_mask: np.ndarray
    appendage_candidate_mask: np.ndarray
    problem_score_map: np.ndarray
    component_summary: pd.DataFrame
    suspect_regions: pd.DataFrame
    component_coarsening: pd.DataFrame
    coarsening_levels: list[dict[str, Any]]
    summary: dict[str, Any]
    config: dict[str, Any]
    dx: float


@dataclass
class DomainCleanResult:
    """Cleaning outputs derived from an audit result.

    :ivar active_mask_cleaned: Cleaned boolean active-domain mask.
    :ivar chd_mask_cleaned: Cleaned boolean CHD mask clipped to the cleaned domain.
    :ivar removed_cells_mask: Cells removed as disconnected components.
    :ivar filled_holes_mask: Cells activated while filling small enclosed holes.
    :ivar pruned_appendages_mask: Cells removed while pruning small appendages.
    :ivar changed_cells_mask: Cells whose active/inactive state changed during cleaning.
    :ivar post_audit: Audit result produced from the cleaned masks.
    :ivar summary: High-level cleaning metrics suitable for logs or reports.
    :ivar config: Resolved configuration used for the cleaning pass.
    :ivar dx: Grid spacing used for the workflow.
    :ivar report_paths: Paths written by :func:`export_domain_report`, if any.
    """

    active_mask_cleaned: np.ndarray
    chd_mask_cleaned: np.ndarray
    removed_cells_mask: np.ndarray
    filled_holes_mask: np.ndarray
    pruned_appendages_mask: np.ndarray
    changed_cells_mask: np.ndarray
    post_audit: DomainAuditResult
    summary: dict[str, Any]
    config: dict[str, Any]
    dx: float
    report_paths: dict[str, str]


@dataclass
class DomainQAWorkflowResult:
    """Combined result returned by :func:`run_domain_qa_and_clean`.

    :ivar audit_result: Audit diagnostics for the original masks.
    :ivar clean_result: Cleaning output and post-clean audit.
    :ivar report_paths: Paths written to disk when ``outdir`` is supplied.
    """

    audit_result: DomainAuditResult
    clean_result: DomainCleanResult
    report_paths: dict[str, str]


def _resolve_config(config: dict[str, Any] | None, dx: float) -> dict[str, Any]:
    """Merge user configuration with defaults and normalize value types.

    Cell-based thresholds are converted to physical-width thresholds using
    ``dx`` unless the corresponding ``*_m`` key is supplied explicitly.

    :param config: Optional partial configuration dictionary.
    :param dx: Grid spacing in model units.
    :return: Resolved configuration dictionary with validated numeric and boolean values.
    """

    resolved = dict(DEFAULT_DOMAIN_QA_CONFIG)
    if config:
        resolved.update(config)

    resolved["coarsening_levels"] = max(2, int(resolved["coarsening_levels"]))
    resolved["thin_width_threshold_m"] = float(
        resolved.get("thin_width_threshold_m", float(resolved["thin_width_threshold_cells"]) * float(dx))
    )
    resolved["bottleneck_width_threshold_m"] = float(
        resolved.get(
            "bottleneck_width_threshold_m",
            float(resolved["bottleneck_width_threshold_cells"]) * float(dx),
        )
    )
    resolved["remove_all_disconnected_without_chd"] = bool(
        resolved["remove_all_disconnected_without_chd"]
    )
    resolved["fill_holes_max_cells"] = max(0, int(resolved["fill_holes_max_cells"]))
    resolved["appendage_max_cells"] = max(0, int(resolved["appendage_max_cells"]))
    resolved["prune_appendages"] = bool(resolved["prune_appendages"])
    resolved["save_pngs"] = bool(resolved["save_pngs"])
    resolved["png_dpi"] = max(72, int(resolved["png_dpi"]))
    resolved["report_component_limit"] = max(1, int(resolved["report_component_limit"]))

    remove_disconnected_max_cells = resolved.get("remove_disconnected_max_cells")
    if remove_disconnected_max_cells is None:
        resolved["remove_disconnected_max_cells"] = None
    else:
        resolved["remove_disconnected_max_cells"] = max(0, int(remove_disconnected_max_cells))

    return resolved


def _validate_masks(active_mask: np.ndarray, chd_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate, coerce, and align the active and CHD masks.

    Both masks are coerced to boolean arrays. The CHD mask is clipped so
    that CHD cells can only exist where the domain is active.

    :param active_mask: Candidate active-domain mask.
    :param chd_mask: Candidate CHD mask with the same grid shape.
    :return: Tuple ``(active_bool, chd_bool)`` of validated boolean arrays.
    :raises ValueError: If ``active_mask`` is not 2D or the input shapes do not match.
    """

    active_bool = np.asarray(active_mask, dtype=bool)
    chd_bool = np.asarray(chd_mask, dtype=bool)

    if active_bool.ndim != 2:
        raise ValueError(f"active_mask must be 2D; got shape {active_bool.shape}.")
    if chd_bool.shape != active_bool.shape:
        raise ValueError(
            f"chd_mask shape {chd_bool.shape} must match active_mask shape {active_bool.shape}."
        )

    chd_bool = chd_bool & active_bool
    return active_bool, chd_bool


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Label 4-neighbour connected components in a boolean mask.

    :param mask: Boolean-like mask to label.
    :return: Tuple ``(labels, n_components)`` where ``labels`` is an ``int32`` array and
        ``n_components`` excludes label 0.
    """

    labels, n_components = ndimage.label(np.asarray(mask, dtype=bool), structure=STRUCTURE4)
    return labels.astype(np.int32, copy=False), int(n_components)


def _coarsen_mask_majority(mask: np.ndarray) -> np.ndarray:
    """Coarsen a mask by 2x2 majority vote with edge-aware padding.

    A coarse cell becomes active when at least half of its valid child
    cells are active. Ties remain active.

    :param mask: Fine-grid boolean-like mask.
    :return: Coarsened boolean mask with shape approximately half the input size.
    """

    mask_bool = np.asarray(mask, dtype=bool)
    ny_f, nx_f = mask_bool.shape
    ny_c = (int(ny_f) + 1) // 2
    nx_c = (int(nx_f) + 1) // 2

    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)

    mask_pad = np.pad(mask_bool.astype(np.int32), ((0, pad_y), (0, pad_x)), mode="constant")
    valid_pad = np.pad(
        np.ones((ny_f, nx_f), dtype=np.int32),
        ((0, pad_y), (0, pad_x)),
        mode="constant",
    )

    mask_block = mask_pad.reshape(ny_c, 2, nx_c, 2)
    valid_block = valid_pad.reshape(ny_c, 2, nx_c, 2)

    active_count = mask_block.sum(axis=(1, 3), dtype=np.int32)
    valid_count = valid_block.sum(axis=(1, 3), dtype=np.int32)

    coarse_mask = np.zeros((ny_c, nx_c), dtype=bool)
    on = valid_count > 0
    coarse_mask[on] = (2 * active_count[on]) >= valid_count[on]
    return coarse_mask


def _coarsen_mask_any(mask: np.ndarray) -> np.ndarray:
    """Coarsen a mask by 2x2 logical OR.

    This is used for CHD propagation and component-presence tracking, where
    any active child should keep the coarse representation alive.

    :param mask: Fine-grid boolean-like mask.
    :return: Coarsened boolean mask with shape approximately half the input size.
    """

    mask_bool = np.asarray(mask, dtype=bool)
    ny_f, nx_f = mask_bool.shape
    ny_c = (int(ny_f) + 1) // 2
    nx_c = (int(nx_f) + 1) // 2

    pad_y = int(2 * ny_c - ny_f)
    pad_x = int(2 * nx_c - nx_f)

    mask_pad = np.pad(mask_bool.astype(np.int32), ((0, pad_y), (0, pad_x)), mode="constant")
    mask_block = mask_pad.reshape(ny_c, 2, nx_c, 2)
    coarse_mask = mask_block.max(axis=(1, 3)) != 0
    return coarse_mask


def _neighbour_masks(active_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean masks for the four orthogonal neighbours of each cell.

    :param active_mask: Boolean-like active mask.
    :return: Tuple ``(north, south, west, east)`` aligned with the input grid.
    """

    active_bool = np.asarray(active_mask, dtype=bool)
    north = np.zeros_like(active_bool, dtype=bool)
    south = np.zeros_like(active_bool, dtype=bool)
    west = np.zeros_like(active_bool, dtype=bool)
    east = np.zeros_like(active_bool, dtype=bool)

    north[1:, :] = active_bool[:-1, :]
    south[:-1, :] = active_bool[1:, :]
    west[:, 1:] = active_bool[:, :-1]
    east[:, :-1] = active_bool[:, 1:]
    return north, south, west, east


def _identify_bottlenecks(
    active_mask: np.ndarray,
    thinness_map: np.ndarray,
    config: dict[str, Any],
) -> np.ndarray:
    """Identify narrow one-cell necks that may destabilize coarsening.

    A bottleneck candidate must be active, narrower than the configured
    width threshold, connected through an opposite neighbour pair, and have
    no more than two orthogonal neighbours.

    :param active_mask: Boolean active-domain mask.
    :param thinness_map: Estimated local widths in model units.
    :param config: Resolved QA configuration.
    :return: Boolean mask of bottleneck cells.
    """

    north, south, west, east = _neighbour_masks(active_mask)
    neighbour_count = (
        north.astype(np.int32)
        + south.astype(np.int32)
        + west.astype(np.int32)
        + east.astype(np.int32)
    )
    opposite_pairs = (north & south) | (west & east)
    narrow_enough = np.asarray(thinness_map, dtype=np.float64) <= float(
        config["bottleneck_width_threshold_m"]
    )
    bottleneck_mask = (
        np.asarray(active_mask, dtype=bool)
        & narrow_enough
        & opposite_pairs
        & (neighbour_count <= 2)
    )
    return bottleneck_mask


def _identify_appendage_candidates(
    active_mask: np.ndarray,
    chd_mask: np.ndarray,
    bottleneck_mask: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Identify small branches attached through non-CHD bottleneck cells.

    The algorithm removes candidate neck cells, labels the remaining
    components, and marks small components that are adjacent to a removed
    neck and do not carry CHD cells.

    :param active_mask: Boolean active-domain mask.
    :param chd_mask: Boolean CHD mask clipped to the active domain.
    :param bottleneck_mask: Bottleneck mask from :func:`_identify_bottlenecks`.
    :param config: Resolved QA configuration.
    :return: Tuple ``(appendage_mask, candidate_necks)``.
    """

    active_bool = np.asarray(active_mask, dtype=bool)
    chd_bool = np.asarray(chd_mask, dtype=bool)
    candidate_necks = np.asarray(bottleneck_mask, dtype=bool) & active_bool & (~chd_bool)

    if not np.any(candidate_necks) or int(config["appendage_max_cells"]) == 0:
        return np.zeros_like(active_bool, dtype=bool), candidate_necks

    truncated_active = active_bool & (~candidate_necks)
    labels, n_components = _label_components(truncated_active)
    if n_components == 0:
        return np.zeros_like(active_bool, dtype=bool), candidate_necks

    counts = np.bincount(labels.ravel(), minlength=n_components + 1)
    chd_labels = np.unique(labels[chd_bool & truncated_active])
    chd_connected = np.zeros(n_components + 1, dtype=bool)
    chd_connected[chd_labels] = True

    appendage_mask = np.zeros_like(active_bool, dtype=bool)
    for component_id in range(1, n_components + 1):
        if int(counts[component_id]) == 0:
            continue
        if chd_connected[component_id]:
            continue
        if int(counts[component_id]) > int(config["appendage_max_cells"]):
            continue

        component_mask = labels == component_id
        component_edge = ndimage.binary_dilation(component_mask, structure=STRUCTURE4)
        if np.any(component_edge & candidate_necks):
            appendage_mask[component_mask] = True

    return appendage_mask, candidate_necks


def _build_coarsening_levels(
    active_mask: np.ndarray,
    chd_mask: np.ndarray,
    dx: float,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Track component survival and CHD connectivity across coarsening levels.

    :param active_mask: Boolean active-domain mask.
    :param chd_mask: Boolean CHD mask clipped to the active domain.
    :param dx: Fine-grid spacing in model units.
    :param config: Resolved QA configuration.
    :return: List of dictionaries, one per coarsening level, containing masks,
        labels, and connectivity summaries.
    """

    levels: list[dict[str, Any]] = []
    current_active = np.asarray(active_mask, dtype=bool)
    current_chd = np.asarray(chd_mask, dtype=bool)

    total_levels = int(config["coarsening_levels"])
    for level in range(total_levels + 1):
        labels, n_components = _label_components(current_active)
        chd_labels = np.unique(labels[current_chd & current_active])
        chd_connected = np.zeros(n_components + 1, dtype=bool)
        chd_connected[chd_labels] = True
        chd_connected_mask = np.zeros_like(current_active, dtype=bool)
        if n_components > 0:
            chd_connected_mask = chd_connected[labels]

        levels.append(
            {
                "level": int(level),
                "dx": float(dx) * float(2 ** level),
                "active_mask": current_active.copy(),
                "chd_mask": current_chd.copy(),
                "labels": labels,
                "chd_connected_mask": chd_connected_mask,
                "n_active": int(np.count_nonzero(current_active)),
                "n_chd": int(np.count_nonzero(current_chd)),
                "n_components": int(n_components),
                "n_chd_connected_components": int(np.count_nonzero(chd_connected[1:])),
            }
        )

        if level >= total_levels:
            break
        if current_active.shape == (1, 1):
            break

        next_active = _coarsen_mask_majority(current_active)
        next_chd = _coarsen_mask_any(current_chd) & next_active
        current_active = next_active
        current_chd = next_chd

    return levels


def _build_component_coarsening(
    component_labels: np.ndarray,
    coarsening_levels: list[dict[str, Any]],
) -> pd.DataFrame:
    """Summarize how each fine-grid component behaves under coarsening.

    :param component_labels: Fine-grid connected-component labels.
    :param coarsening_levels: Coarsening diagnostics from :func:`_build_coarsening_levels`.
    :return: DataFrame with one row per component per level.
    """

    component_ids = np.unique(component_labels)
    component_ids = component_ids[component_ids > 0]

    records: list[dict[str, Any]] = []
    for component_id in component_ids:
        presence_mask = component_labels == int(component_id)
        current_presence = presence_mask.copy()

        for level_info in coarsening_levels:
            represented = current_presence & np.asarray(level_info["active_mask"], dtype=bool)
            survives = bool(np.any(represented))
            coarse_chd_connected = False
            if survives:
                coarse_chd_connected = bool(
                    np.any(represented & np.asarray(level_info["chd_connected_mask"], dtype=bool))
                )

            records.append(
                {
                    "component_id": int(component_id),
                    "level": int(level_info["level"]),
                    "represented_cells": int(np.count_nonzero(represented)),
                    "survives": survives,
                    "coarse_chd_connected": coarse_chd_connected,
                }
            )
            current_presence = _coarsen_mask_any(current_presence)

    dataframe = pd.DataFrame.from_records(records)
    if dataframe.empty:
        dataframe = pd.DataFrame(
            columns=[
                "component_id",
                "level",
                "represented_cells",
                "survives",
                "coarse_chd_connected",
            ]
        )
    return dataframe


def _build_component_summary(
    component_labels: np.ndarray,
    thinness_map: np.ndarray,
    thin_mask: np.ndarray,
    disconnected_mask: np.ndarray,
    bottleneck_mask: np.ndarray,
    appendage_candidate_mask: np.ndarray,
    chd_mask: np.ndarray,
    problem_score_map: np.ndarray,
    component_coarsening: pd.DataFrame,
    dx: float,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-component audit tables and recommended actions.

    :param component_labels: Fine-grid connected-component labels.
    :param thinness_map: Estimated local widths in model units.
    :param thin_mask: Cells narrower than the thin-feature threshold.
    :param disconnected_mask: Cells belonging to non-CHD-connected components.
    :param bottleneck_mask: Bottleneck-cell mask.
    :param appendage_candidate_mask: Small appendage mask.
    :param chd_mask: Boolean CHD mask.
    :param problem_score_map: Combined integer score map.
    :param component_coarsening: Per-component coarsening diagnostics.
    :param dx: Grid spacing in model units.
    :param config: Resolved QA configuration.
    :return: Tuple ``(component_summary, suspect_regions)``.
    """

    n_components = int(np.max(component_labels))
    records: list[dict[str, Any]] = []

    for component_id in range(1, n_components + 1):
        component_mask = component_labels == component_id
        cell_count = int(np.count_nonzero(component_mask))
        if cell_count == 0:
            continue

        component_thinness = np.asarray(thinness_map[component_mask], dtype=np.float64)
        component_thin = np.asarray(thin_mask[component_mask], dtype=bool)
        component_problem_score = np.asarray(problem_score_map[component_mask], dtype=np.int32)
        component_disconnected = bool(np.any(disconnected_mask[component_mask]))
        component_bottleneck_cells = int(np.count_nonzero(bottleneck_mask[component_mask]))
        component_appendage_cells = int(np.count_nonzero(appendage_candidate_mask[component_mask]))
        chd_connected = bool(np.any(chd_mask[component_mask]))

        coarsening_rows = component_coarsening.loc[
            component_coarsening["component_id"] == component_id
        ].sort_values("level")
        disappears_at_level = None
        loses_chd_at_level = None
        for _, row in coarsening_rows.iterrows():
            if int(row["level"]) == 0:
                continue
            if (disappears_at_level is None) and (not bool(row["survives"])):
                disappears_at_level = int(row["level"])
            if (
                chd_connected
                and bool(row["survives"])
                and (not bool(row["coarse_chd_connected"]))
                and loses_chd_at_level is None
            ):
                loses_chd_at_level = int(row["level"])

        recommended_action = "keep"
        if component_disconnected:
            remove_all = bool(config["remove_all_disconnected_without_chd"])
            max_cells = config["remove_disconnected_max_cells"]
            if remove_all:
                recommended_action = "remove_disconnected_component"
            elif max_cells is not None and cell_count <= int(max_cells):
                recommended_action = "remove_disconnected_component"
            else:
                recommended_action = "review_disconnected_component"
        elif component_appendage_cells > 0 and bool(config["prune_appendages"]):
            recommended_action = "prune_small_appendage"
        elif disappears_at_level is not None or loses_chd_at_level is not None:
            recommended_action = "review_coarsening_instability"
        elif np.any(component_thin):
            recommended_action = "review_thin_feature"

        records.append(
            {
                "component_id": int(component_id),
                "cell_count": cell_count,
                "area_m2": float(cell_count * dx * dx),
                "chd_connected": chd_connected,
                "chd_cell_count": int(np.count_nonzero(chd_mask[component_mask])),
                "thin_cell_count": int(np.count_nonzero(component_thin)),
                "thin_fraction": float(np.count_nonzero(component_thin) / cell_count),
                "min_width_m": float(np.min(component_thinness)) if cell_count > 0 else 0.0,
                "median_width_m": float(np.median(component_thinness)) if cell_count > 0 else 0.0,
                "bottleneck_cell_count": component_bottleneck_cells,
                "appendage_candidate_cell_count": component_appendage_cells,
                "problem_score_sum": int(component_problem_score.sum()),
                "problem_score_max": int(component_problem_score.max(initial=0)),
                "disconnected_without_chd": component_disconnected,
                "disappears_at_level": disappears_at_level,
                "loses_chd_at_level": loses_chd_at_level,
                "recommended_action": recommended_action,
            }
        )

    component_summary = pd.DataFrame.from_records(records)
    if component_summary.empty:
        component_summary = pd.DataFrame(
            columns=[
                "component_id",
                "cell_count",
                "area_m2",
                "chd_connected",
                "chd_cell_count",
                "thin_cell_count",
                "thin_fraction",
                "min_width_m",
                "median_width_m",
                "bottleneck_cell_count",
                "appendage_candidate_cell_count",
                "problem_score_sum",
                "problem_score_max",
                "disconnected_without_chd",
                "disappears_at_level",
                "loses_chd_at_level",
                "recommended_action",
            ]
        )
    component_summary = component_summary.sort_values(
        ["problem_score_sum", "cell_count", "component_id"],
        ascending=[False, False, True],
        ignore_index=True,
    )

    suspect_regions = component_summary.loc[
        (component_summary["problem_score_sum"] > 0)
        | (component_summary["recommended_action"] != "keep")
    ].reset_index(drop=True)
    return component_summary, suspect_regions


def audit_domain(
    active_mask: np.ndarray,
    chd_mask: np.ndarray,
    dx: float,
    config: dict[str, Any] | None,
    outdir: Path | None = None,
) -> DomainAuditResult:
    """
    Audit a rasterized groundwater model domain for topology issues that commonly
    destabilize geometric multigrid coarsening.

    The audit does not modify the inputs. It labels connected components,
    measures local width, detects narrow necks and appendages, simulates
    repeated coarsening, and aggregates those diagnostics into per-cell and
    per-component summaries.

    :param active_mask: 2D boolean-like active-domain mask.
    :param chd_mask: 2D boolean-like CHD mask defined on the same grid.
    :param dx: Cell size in model units used to convert width and area metrics.
    :param config: Optional QA configuration overrides. Missing keys are filled from
        :data:`DEFAULT_DOMAIN_QA_CONFIG`.
    :param outdir: Optional report directory. This is recorded in the audit summary
        and used by ``run_domain_qa_and_clean`` when exporting files.
    :return: DomainAuditResult with component labels, thinness, coarsening diagnostics,
        and per-component summary tables.
    :raises ValueError: If the masks are not 2D, have different shapes, or are
        otherwise inconsistent.
    """
    resolved_config = _resolve_config(config=config, dx=float(dx))
    active_bool, chd_bool = _validate_masks(active_mask=active_mask, chd_mask=chd_mask)

    component_labels, n_components = _label_components(active_bool)
    chd_component_ids = np.unique(component_labels[chd_bool & active_bool])
    chd_component_ids = chd_component_ids[chd_component_ids > 0]

    disconnected_mask = (component_labels != 0) & (~np.isin(component_labels, chd_component_ids))

    distance_map = ndimage.distance_transform_edt(active_bool, sampling=float(dx))
    thinness_map = (2.0 * distance_map).astype(np.float32, copy=False)
    thin_mask = active_bool & (thinness_map <= float(resolved_config["thin_width_threshold_m"]))

    bottleneck_mask = _identify_bottlenecks(
        active_mask=active_bool,
        thinness_map=thinness_map,
        config=resolved_config,
    )
    appendage_candidate_mask, candidate_necks = _identify_appendage_candidates(
        active_mask=active_bool,
        chd_mask=chd_bool,
        bottleneck_mask=bottleneck_mask,
        config=resolved_config,
    )

    coarsening_levels = _build_coarsening_levels(
        active_mask=active_bool,
        chd_mask=chd_bool,
        dx=float(dx),
        config=resolved_config,
    )
    component_coarsening = _build_component_coarsening(
        component_labels=component_labels,
        coarsening_levels=coarsening_levels,
    )

    problem_score_map = np.zeros_like(component_labels, dtype=np.int16)
    problem_score_map[thin_mask] = problem_score_map[thin_mask] + np.int16(1)
    problem_score_map[bottleneck_mask] = problem_score_map[bottleneck_mask] + np.int16(2)
    problem_score_map[appendage_candidate_mask] = problem_score_map[appendage_candidate_mask] + np.int16(2)
    problem_score_map[disconnected_mask] = problem_score_map[disconnected_mask] + np.int16(4)

    disappearing_components = component_coarsening.loc[
        (component_coarsening["level"] > 0) & (~component_coarsening["survives"]),
        "component_id",
    ].drop_duplicates()
    if not disappearing_components.empty:
        disappearing_mask = np.isin(component_labels, disappearing_components.to_numpy(dtype=np.int32))
        problem_score_map[disappearing_mask] = problem_score_map[disappearing_mask] + np.int16(2)

    lost_connectivity_components = component_coarsening.loc[
        (component_coarsening["level"] > 0)
        & (component_coarsening["survives"])
        & (~component_coarsening["coarse_chd_connected"]),
        "component_id",
    ].drop_duplicates()
    if not lost_connectivity_components.empty:
        lost_connectivity_mask = np.isin(
            component_labels,
            lost_connectivity_components.to_numpy(dtype=np.int32),
        )
        initial_chd_connected = np.isin(component_labels, chd_component_ids)
        lost_connectivity_mask = lost_connectivity_mask & initial_chd_connected
        problem_score_map[lost_connectivity_mask] = problem_score_map[lost_connectivity_mask] + np.int16(3)

    component_summary, suspect_regions = _build_component_summary(
        component_labels=component_labels,
        thinness_map=thinness_map,
        thin_mask=thin_mask,
        disconnected_mask=disconnected_mask,
        bottleneck_mask=bottleneck_mask,
        appendage_candidate_mask=appendage_candidate_mask,
        chd_mask=chd_bool,
        problem_score_map=problem_score_map,
        component_coarsening=component_coarsening,
        dx=float(dx),
        config=resolved_config,
    )

    disappearing_component_count = int(
        suspect_regions["disappears_at_level"].notna().sum()
    ) if not suspect_regions.empty else 0
    lost_connectivity_count = int(
        suspect_regions["loses_chd_at_level"].notna().sum()
    ) if not suspect_regions.empty else 0

    summary = {
        "grid_shape": [int(active_bool.shape[0]), int(active_bool.shape[1])],
        "dx": float(dx),
        "n_active": int(np.count_nonzero(active_bool)),
        "n_chd": int(np.count_nonzero(chd_bool)),
        "n_components": int(n_components),
        "n_chd_connected_components": int(chd_component_ids.size),
        "n_disconnected_components": int(n_components - chd_component_ids.size),
        "n_disconnected_cells": int(np.count_nonzero(disconnected_mask)),
        "n_thin_cells": int(np.count_nonzero(thin_mask)),
        "n_bottleneck_cells": int(np.count_nonzero(bottleneck_mask)),
        "n_appendage_candidate_cells": int(np.count_nonzero(appendage_candidate_mask)),
        "n_components_disappearing_under_coarsening": disappearing_component_count,
        "n_components_losing_chd_under_coarsening": lost_connectivity_count,
        "coarsening_rule": (
            "Each coarse active cell is active when at least half of its valid 2x2 child "
            "cells are active; ties stay active. A coarse CHD cell is active when any child "
            "CHD cell is present and the coarse active cell survives."
        ),
        "actual_coarsening_levels": int(len(coarsening_levels) - 1),
        "thin_width_threshold_m": float(resolved_config["thin_width_threshold_m"]),
        "bottleneck_width_threshold_m": float(resolved_config["bottleneck_width_threshold_m"]),
        "report_outdir": str(Path(outdir)) if outdir is not None else None,
        "n_candidate_neck_cells": int(np.count_nonzero(candidate_necks)),
    }

    return DomainAuditResult(
        active_mask=active_bool,
        chd_mask=chd_bool,
        component_labels=component_labels,
        thinness_map=thinness_map,
        thin_mask=thin_mask,
        disconnected_mask=disconnected_mask,
        bottleneck_mask=bottleneck_mask,
        appendage_candidate_mask=appendage_candidate_mask,
        problem_score_map=problem_score_map,
        component_summary=component_summary,
        suspect_regions=suspect_regions,
        component_coarsening=component_coarsening,
        coarsening_levels=coarsening_levels,
        summary=summary,
        config=resolved_config,
        dx=float(dx),
    )


def clean_domain(
    active_mask: np.ndarray,
    chd_mask: np.ndarray,
    dx: float,
    audit_result: DomainAuditResult,
    config: dict[str, Any] | None,
    outdir: Path | None = None,
) -> DomainCleanResult:
    """
    Apply conservative, auditable cleaning to a rasterized domain using the audit
    diagnostics as the decision basis.

    The cleaning rules are intentionally limited. The function can remove
    disconnected non-CHD components, fill small enclosed holes, and prune
    small appendages. It then re-audits the cleaned masks so callers can
    compare before and after diagnostics.

    :param active_mask: 2D boolean-like active-domain mask.
    :param chd_mask: 2D boolean-like CHD mask on the same grid.
    :param dx: Cell size in model units.
    :param audit_result: Audit output from :func:`audit_domain` for the same masks.
    :param config: Optional QA/cleaning configuration overrides.
    :param outdir: Optional report directory recorded in the cleaning summary.
    :return: DomainCleanResult with cleaned masks, explicit change masks, and a post-clean audit.
    :raises ValueError: If the input masks do not match the shapes stored in
        ``audit_result``.
    """
    resolved_config = _resolve_config(config=config, dx=float(dx))
    active_bool, chd_bool = _validate_masks(active_mask=active_mask, chd_mask=chd_mask)

    if active_bool.shape != audit_result.active_mask.shape:
        raise ValueError("active_mask shape does not match audit_result.active_mask shape.")
    if chd_bool.shape != audit_result.chd_mask.shape:
        raise ValueError("chd_mask shape does not match audit_result.chd_mask shape.")

    cleaned_active = active_bool.copy()

    component_summary = audit_result.component_summary
    remove_labels: list[int] = []
    if not component_summary.empty:
        disconnected_rows = component_summary.loc[
            component_summary["disconnected_without_chd"] == True
        ].copy()
        if bool(resolved_config["remove_all_disconnected_without_chd"]):
            remove_labels = disconnected_rows["component_id"].astype(int).tolist()
        else:
            max_cells = resolved_config["remove_disconnected_max_cells"]
            if max_cells is not None:
                remove_labels = disconnected_rows.loc[
                    disconnected_rows["cell_count"] <= int(max_cells),
                    "component_id",
                ].astype(int).tolist()

    removed_cells_mask = np.isin(audit_result.component_labels, np.asarray(remove_labels, dtype=np.int32))
    cleaned_active[removed_cells_mask] = False

    filled_all_holes = ndimage.binary_fill_holes(cleaned_active, structure=STRUCTURE4)
    hole_candidates = filled_all_holes & (~cleaned_active)
    hole_labels, n_holes = _label_components(hole_candidates)
    filled_holes_mask = np.zeros_like(cleaned_active, dtype=bool)
    if n_holes > 0 and int(resolved_config["fill_holes_max_cells"]) > 0:
        hole_counts = np.bincount(hole_labels.ravel(), minlength=n_holes + 1)
        for hole_id in range(1, n_holes + 1):
            if int(hole_counts[hole_id]) <= int(resolved_config["fill_holes_max_cells"]):
                filled_holes_mask[hole_labels == hole_id] = True
    cleaned_active[filled_holes_mask] = True

    pruned_appendages_mask = np.zeros_like(cleaned_active, dtype=bool)
    if bool(resolved_config["prune_appendages"]) and int(resolved_config["appendage_max_cells"]) > 0:
        post_fill_distance = ndimage.distance_transform_edt(cleaned_active, sampling=float(dx))
        post_fill_thinness_map = 2.0 * post_fill_distance
        post_fill_bottlenecks = _identify_bottlenecks(
            active_mask=cleaned_active,
            thinness_map=post_fill_thinness_map,
            config=resolved_config,
        )
        appendage_mask, candidate_necks = _identify_appendage_candidates(
            active_mask=cleaned_active,
            chd_mask=chd_bool & cleaned_active,
            bottleneck_mask=post_fill_bottlenecks,
            config=resolved_config,
        )
        if np.any(appendage_mask):
            pruned_appendages_mask[appendage_mask] = True
            cleaned_active[appendage_mask] = False

            north, south, west, east = _neighbour_masks(cleaned_active)
            neighbour_count = (
                north.astype(np.int32)
                + south.astype(np.int32)
                + west.astype(np.int32)
                + east.astype(np.int32)
            )
            removable_necks = candidate_necks & (~chd_bool) & (neighbour_count <= 1)
            if np.any(removable_necks):
                pruned_appendages_mask[removable_necks] = True
                cleaned_active[removable_necks] = False

    cleaned_chd = chd_bool & cleaned_active
    changed_cells_mask = cleaned_active != active_bool

    post_audit = audit_domain(
        active_mask=cleaned_active,
        chd_mask=cleaned_chd,
        dx=float(dx),
        config=resolved_config,
        outdir=outdir,
    )

    summary = {
        "n_active_original": int(np.count_nonzero(active_bool)),
        "n_active_cleaned": int(np.count_nonzero(cleaned_active)),
        "n_removed_cells": int(np.count_nonzero(removed_cells_mask)),
        "n_filled_hole_cells": int(np.count_nonzero(filled_holes_mask)),
        "n_pruned_appendage_cells": int(np.count_nonzero(pruned_appendages_mask)),
        "n_changed_cells": int(np.count_nonzero(changed_cells_mask)),
        "n_chd_original": int(np.count_nonzero(chd_bool)),
        "n_chd_cleaned": int(np.count_nonzero(cleaned_chd)),
        "report_outdir": str(Path(outdir)) if outdir is not None else None,
    }

    return DomainCleanResult(
        active_mask_cleaned=cleaned_active,
        chd_mask_cleaned=cleaned_chd,
        removed_cells_mask=removed_cells_mask,
        filled_holes_mask=filled_holes_mask,
        pruned_appendages_mask=pruned_appendages_mask,
        changed_cells_mask=changed_cells_mask,
        post_audit=post_audit,
        summary=summary,
        config=resolved_config,
        dx=float(dx),
        report_paths={},
    )


def export_domain_report(
    audit_result: DomainAuditResult,
    clean_result: DomainCleanResult,
    outdir: Path,
) -> dict[str, str]:
    """
    Export audit and cleaning diagnostics to disk for later review.

    :param audit_result: Audit result for the original domain.
    :param clean_result: Cleaning result, including the post-clean audit.
    :param outdir: Output directory for NumPy arrays, CSV summaries, text report,
        and optional PNG figures.
    :return: Dictionary of written report paths.
    :raises OSError: If the output directory or report files cannot be written.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    active_original_path = outdir.joinpath("active_mask_original.npy")
    active_cleaned_path = outdir.joinpath("active_mask_cleaned.npy")
    chd_original_path = outdir.joinpath("chd_mask_original.npy")
    thinness_path = outdir.joinpath("thinness_map.npy")
    problem_score_path = outdir.joinpath("problem_score_map.npy")
    removed_cells_path = outdir.joinpath("removed_cells_mask.npy")
    filled_holes_path = outdir.joinpath("filled_holes_mask.npy")
    bottleneck_path = outdir.joinpath("bottleneck_mask.npy")
    appendage_path = outdir.joinpath("appendage_candidate_mask.npy")
    suspect_csv_path = outdir.joinpath("suspect_regions.csv")
    report_path = outdir.joinpath("domain_report.txt")

    np.save(active_original_path, audit_result.active_mask)
    np.save(active_cleaned_path, clean_result.active_mask_cleaned)
    np.save(chd_original_path, audit_result.chd_mask)
    np.save(thinness_path, audit_result.thinness_map)
    np.save(problem_score_path, audit_result.problem_score_map)
    np.save(removed_cells_path, clean_result.removed_cells_mask)
    np.save(filled_holes_path, clean_result.filled_holes_mask)
    np.save(bottleneck_path, audit_result.bottleneck_mask)
    np.save(appendage_path, audit_result.appendage_candidate_mask)

    paths["active_mask_original"] = str(active_original_path)
    paths["active_mask_cleaned"] = str(active_cleaned_path)
    paths["chd_mask_original"] = str(chd_original_path)
    paths["thinness_map"] = str(thinness_path)
    paths["problem_score_map"] = str(problem_score_path)
    paths["removed_cells_mask"] = str(removed_cells_path)
    paths["filled_holes_mask"] = str(filled_holes_path)
    paths["bottleneck_mask"] = str(bottleneck_path)
    paths["appendage_candidate_mask"] = str(appendage_path)

    if not np.array_equal(audit_result.chd_mask, clean_result.chd_mask_cleaned):
        chd_cleaned_path = outdir.joinpath("chd_mask_cleaned.npy")
        np.save(chd_cleaned_path, clean_result.chd_mask_cleaned)
        paths["chd_mask_cleaned"] = str(chd_cleaned_path)

    audit_result.suspect_regions.to_csv(suspect_csv_path, index=False)
    paths["suspect_regions_csv"] = str(suspect_csv_path)

    report_lines: list[str] = [
        "Domain QA and Cleaning Report",
        "============================",
        "",
        f"Grid shape: {tuple(audit_result.summary['grid_shape'])}",
        f"Cell size dx: {audit_result.summary['dx']}",
        "",
        "Coarsening rule:",
        str(audit_result.summary["coarsening_rule"]),
        "",
        "Thresholds:",
        f"  thin_width_threshold_m = {audit_result.summary['thin_width_threshold_m']}",
        f"  bottleneck_width_threshold_m = {audit_result.summary['bottleneck_width_threshold_m']}",
        f"  fill_holes_max_cells = {clean_result.config['fill_holes_max_cells']}",
        f"  remove_all_disconnected_without_chd = {clean_result.config['remove_all_disconnected_without_chd']}",
        f"  remove_disconnected_max_cells = {clean_result.config['remove_disconnected_max_cells']}",
        f"  prune_appendages = {clean_result.config['prune_appendages']}",
        f"  appendage_max_cells = {clean_result.config['appendage_max_cells']}",
        "",
        "Audit summary:",
    ]
    for key, value in audit_result.summary.items():
        if key in {"grid_shape", "dx", "coarsening_rule", "report_outdir"}:
            continue
        report_lines.append(f"  {key} = {value}")

    report_lines.extend(
        [
            "",
            "Cleaning summary:",
        ]
    )
    for key, value in clean_result.summary.items():
        if key == "report_outdir":
            continue
        report_lines.append(f"  {key} = {value}")

    report_lines.extend(
        [
            "",
            "Post-clean summary:",
        ]
    )
    for key, value in clean_result.post_audit.summary.items():
        if key in {"grid_shape", "dx", "coarsening_rule", "report_outdir"}:
            continue
        report_lines.append(f"  {key} = {value}")

    report_lines.extend(["", "Top suspect components:"])
    if audit_result.suspect_regions.empty:
        report_lines.append("  none")
    else:
        top_rows = audit_result.suspect_regions.head(int(audit_result.config["report_component_limit"]))
        report_lines.append(top_rows.to_string(index=False))

    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    paths["domain_report"] = str(report_path)

    if bool(audit_result.config["save_pngs"]):
        try:
            import matplotlib

            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
        except Exception:
            plt = None

        if plt is not None:
            dpi = int(audit_result.config["png_dpi"])

            fig, ax = plt.subplots(figsize=(8.0, 6.0))
            ax.imshow(np.asarray(audit_result.active_mask, dtype=np.int8), cmap="Greys", interpolation="nearest")
            chd_y, chd_x = np.where(audit_result.chd_mask)
            if chd_y.size > 0:
                ax.scatter(chd_x, chd_y, s=4.0, c="tab:blue", label="CHD")
                ax.legend(loc="upper right")
            ax.set_title("Original Active and CHD Masks")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            original_png_path = outdir.joinpath("domain_original.png")
            fig.savefig(original_png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths["domain_original_png"] = str(original_png_path)

            fig, ax = plt.subplots(figsize=(8.0, 6.0))
            im = ax.imshow(audit_result.thinness_map, cmap="viridis", interpolation="nearest")
            ax.set_title("Thinness Map (estimated local width)")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            fig.colorbar(im, ax=ax, label="Estimated width")
            thinness_png_path = outdir.joinpath("thinness_map.png")
            fig.savefig(thinness_png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths["thinness_map_png"] = str(thinness_png_path)

            fig, ax = plt.subplots(figsize=(8.0, 6.0))
            im = ax.imshow(audit_result.problem_score_map, cmap="magma", interpolation="nearest")
            ax.set_title("Problem Score Map")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            fig.colorbar(im, ax=ax, label="Score")
            problem_score_png_path = outdir.joinpath("problem_score_map.png")
            fig.savefig(problem_score_png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths["problem_score_map_png"] = str(problem_score_png_path)

            fig, ax = plt.subplots(figsize=(8.0, 6.0))
            cleaned_plot = np.asarray(clean_result.active_mask_cleaned, dtype=np.int8)
            ax.imshow(cleaned_plot, cmap="Greys", interpolation="nearest")
            removed_y, removed_x = np.where(clean_result.removed_cells_mask)
            if removed_y.size > 0:
                ax.scatter(removed_x, removed_y, s=4.0, c="tab:red", label="Removed")
            filled_y, filled_x = np.where(clean_result.filled_holes_mask)
            if filled_y.size > 0:
                ax.scatter(filled_x, filled_y, s=4.0, c="tab:green", label="Filled holes")
            if removed_y.size > 0 or filled_y.size > 0:
                ax.legend(loc="upper right")
            ax.set_title("Cleaned Active Mask")
            ax.set_xlabel("Column")
            ax.set_ylabel("Row")
            cleaned_png_path = outdir.joinpath("domain_cleaned.png")
            fig.savefig(cleaned_png_path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            paths["domain_cleaned_png"] = str(cleaned_png_path)

    clean_result.report_paths = dict(paths)
    return paths


def run_domain_qa_and_clean(
    active_mask: np.ndarray,
    chd_mask: np.ndarray,
    dx: float,
    config: dict[str, Any] | None = None,
    outdir: Path | None = None,
) -> DomainQAWorkflowResult:
    """
    Run the full domain QA workflow: audit, conservative cleaning, and optional report export.

    This is the normal entry point for callers that want both diagnostics and
    cleaned masks in one call.

    :param active_mask: 2D boolean-like active-domain mask.
    :param chd_mask: 2D boolean-like CHD mask on the same grid.
    :param dx: Cell size in model units.
    :param config: Optional QA/cleaning configuration overrides.
    :param outdir: Optional directory for exported QA products.
    :return: DomainQAWorkflowResult containing the audit result, cleaning result,
        and any written report paths.
    :raises ValueError: Propagated from :func:`audit_domain` or :func:`clean_domain`
        when the input masks are invalid.
    """
    audit_result = audit_domain(
        active_mask=active_mask,
        chd_mask=chd_mask,
        dx=float(dx),
        config=config,
        outdir=outdir,
    )
    clean_result = clean_domain(
        active_mask=active_mask,
        chd_mask=chd_mask,
        dx=float(dx),
        audit_result=audit_result,
        config=config,
        outdir=outdir,
    )

    report_paths: dict[str, str] = {}
    if outdir is not None:
        report_paths = export_domain_report(
            audit_result=audit_result,
            clean_result=clean_result,
            outdir=Path(outdir),
        )

    clean_result.report_paths = dict(report_paths)
    return DomainQAWorkflowResult(
        audit_result=audit_result,
        clean_result=clean_result,
        report_paths=report_paths,
    )
