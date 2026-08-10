#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Run steady or transient river-loss structural-sensitivity experiments.

The reusable case builder lives in
``DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section``. This script is
configured entirely in its ``if __name__ == "__main__":`` block so it can be
run directly with the PyCharm Run button.

The transient default is FE-aligned where saturated Darcy permits: Kh/Kv=10,
Kh=(200, 20) m/day, a one-hour channel ramp, zero saturated specific storage,
and the OGS water-table outlet plus two alternative outlet structures. Export
an existing OGS run first with ``export_ogs_river_loss_reference.py``, then
set ``ogs_reference_path`` below. The comparison is diagnostic rather than an
equation-parity acceptance test because OGS uses Richards-flow saturation and
relative permeability.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def write_results(
    results: list[Any],
    config: Any,
    output_directory: Path,
) -> None:
    """Write sweep CSV and configuration metadata."""

    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "river_loss_anisotropy_sweep.csv"
    if not results:
        raise ValueError("The steady sweep produced no result rows.")
    fieldnames = list(asdict(results[0]).keys())
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    metadata = {
        "config": asdict(config),
        "flow_units": "m3/day over the configured out_of_plane_width",
        "flow_normalization": "not normalized; divide reported flows by out_of_plane_width for m2/day",
        "section_scope": "half-channel, one-bank cross-section",
        "positive_channel_inflow": "flow from channel fixed-head cells into aquifer",
        "positive_outlet_outflow": "flow from aquifer into far-field fixed-head cells",
        "sweep": {
            "ratios": list(dict.fromkeys(result.varied_ratio for result in results)),
            "outlet_modes": list(dict.fromkeys(result.outlet_mode for result in results)),
            "anisotropy_targets": list(dict.fromkeys(result.anisotropy_target for result in results)),
        },
    }
    with (output_directory / "river_loss_cross_section_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(metadata, stream, indent=2)

    print(f"Wrote {csv_path}")


def _write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write homogeneous dictionary records to CSV."""

    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def ensure_ogs_reference(
    *,
    reference_path: Path | None,
    refresh_reference: bool,
    fe_repository: Path,
    fe_config_path: Path,
    fe_run_directory: Path,
) -> Path | None:
    """Export the existing OGS run when its standardized artifact is absent."""

    if reference_path is None:
        return None

    fe_python = fe_repository / ".venv" / "bin" / "python"
    exporter = REPO_ROOT / "working_tests" / "export_ogs_river_loss_reference.py"
    fe_pvd_path = fe_run_directory / "results" / "model.pvd"
    fe_domain_mesh_path = fe_run_directory / "mesh" / "vtu" / "domain.vtu"
    for description, path in (
        ("fe_mesh_tester Python", fe_python),
        ("OGS export script", exporter),
        ("FE configuration", fe_config_path),
        ("FE run directory", fe_run_directory),
        ("FE PVD result", fe_pvd_path),
        ("FE domain mesh", fe_domain_mesh_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{description} not found: {path}")

    source_paths = (fe_pvd_path, fe_domain_mesh_path, fe_config_path)
    source_modified_time = max(path.stat().st_mtime for path in source_paths)
    reference_is_current = bool(
        reference_path.is_file()
        and reference_path.stat().st_mtime >= source_modified_time
    )
    if reference_is_current and not refresh_reference:
        return reference_path
    if reference_path.is_file() and not refresh_reference:
        print("The OGS comparison artifact is older than the FE run; refreshing it.")

    print(f"Preparing OGS comparison artifact at {reference_path}")
    environment = dict(os.environ)
    environment.setdefault("MPLCONFIGDIR", "/tmp/darcywarp_matplotlib")
    subprocess.run(
        [
            str(fe_python),
            str(exporter),
            "--fe-repository",
            str(fe_repository),
            "--config",
            str(fe_config_path),
            "--run-directory",
            str(fe_run_directory),
            "--output",
            str(reference_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    return reference_path


def run_transient_study(
    *,
    config: Any,
    ratios: list[float],
    outlet_modes: list[str],
    anisotropy_target: str,
    output_times_days: list[float],
    channel_ramp_days: float,
    braidplain_specific_storage: float,
    regional_specific_storage: float,
    maximum_timestep_days: float,
    progress_every_steps: int,
    output_directory: Path,
    ogs_reference: Path | None,
) -> None:
    """Run and report the saturated transient structural-envelope study."""

    from DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section import (
        _config_for_ratio,
        build_cross_section,
    )
    from DARCY_WARP_PACKAGE.case_studies.river_loss_transient import (
        compare_to_ogs_reference,
        run_transient_case,
        save_transient_case,
    )

    output_directory.mkdir(parents=True, exist_ok=True)
    step_records: list[dict[str, object]] = []
    comparison_records: list[dict[str, object]] = []
    case_metadata: list[dict[str, object]] = []
    for outlet_mode in outlet_modes:
        for ratio in ratios:
            case_config = _config_for_ratio(
                base=config,
                ratio=float(ratio),
                target=anisotropy_target,
            )
            model = build_cross_section(
                cfg=case_config,
                outlet_mode=outlet_mode,
            )
            rows, heads, metadata = run_transient_case(
                model=model,
                output_times_days=output_times_days,
                channel_ramp_days=channel_ramp_days,
                braidplain_specific_storage=braidplain_specific_storage,
                regional_specific_storage=regional_specific_storage,
                maximum_timestep_days=maximum_timestep_days,
                progress_every_steps=progress_every_steps,
            )
            ratio_tag = f"{ratio:g}".replace(".", "p")
            artifact_path = output_directory / (
                f"river_loss_transient_{outlet_mode}_{anisotropy_target}_"
                f"ratio_{ratio_tag}.npz"
            )
            case_identity = {
                "outlet_mode": outlet_mode,
                "anisotropy_target": anisotropy_target,
                "varied_ratio": float(ratio),
            }
            metadata = {
                **metadata,
                **case_identity,
                "config": asdict(case_config),
                "artifact": str(artifact_path),
            }
            save_transient_case(
                path=artifact_path,
                model=model,
                rows=rows,
                heads=heads,
                metadata=metadata,
            )
            for row in rows:
                step_records.append({**case_identity, **asdict(row)})
            if ogs_reference is not None:
                comparisons = compare_to_ogs_reference(
                    model=model,
                    rows=rows,
                    heads=heads,
                    reference_path=ogs_reference,
                )
                for comparison in comparisons:
                    comparison_records.append(
                        {**case_identity, **asdict(comparison)}
                    )
            case_metadata.append(metadata)
            final = rows[-1]
            print(
                f"transient outlet={outlet_mode:16s} Kh/Kv={ratio:8g} "
                f"Qriver_final={final.channel_inflow: .8e} "
                f"leakage={final.cumulative_channel_leakage: .8e} "
                f"imbalance={final.relative_mass_imbalance: .3e}"
            )

    steps_path = output_directory / "river_loss_transient_steps.csv"
    _write_dict_rows(path=steps_path, rows=step_records)
    if comparison_records:
        _write_dict_rows(
            path=output_directory / "river_loss_transient_ogs_comparison.csv",
            rows=comparison_records,
        )
    study_metadata = {
        "scientific_scope": "saturated_darcy_structural_envelope",
        "primary_use": "late/final total-head and one-sided channel-leakage comparison",
        "not_equation_parity": "OGS uses Richards flow; DarcyWarp does not model vadose-zone saturation or relative permeability here.",
        "flow_units": "m3/day over the configured out_of_plane_width",
        "output_times_days": output_times_days,
        "ogs_reference": None if ogs_reference is None else str(ogs_reference),
        "cases": case_metadata,
    }
    with (output_directory / "river_loss_transient_config.json").open(
        "w", encoding="utf-8"
    ) as stream:
        json.dump(study_metadata, stream, indent=2)
    print(f"Wrote {steps_path}")


def run_study(
    *,
    transient_enabled: bool,
    transient_storage_enabled: bool,
    config: Any,
    anisotropy_ratios: list[float],
    outlet_modes: list[str],
    anisotropy_target: str,
    output_directory: Path,
    save_steady_heads: bool,
    transient_output_times_days: list[float],
    channel_ramp_days: float,
    maximum_timestep_days: float,
    progress_every_steps: int,
    braidplain_specific_storage: float,
    regional_specific_storage: float,
    ogs_reference_path: Path | None,
    refresh_ogs_reference: bool,
    fe_repository: Path,
    fe_config_path: Path,
    fe_run_directory: Path,
) -> None:
    """Validate settings and dispatch the selected experiment."""

    import warp as wp

    from DARCY_WARP_PACKAGE.config import WP_FLOAT

    if not anisotropy_ratios:
        raise ValueError("anisotropy_ratios must contain at least one value.")
    if not outlet_modes:
        raise ValueError("outlet_modes must contain at least one value.")
    if config.implementation == "fast":
        if config.solver != "kcycle":
            raise ValueError("implementation='fast' supports solver='kcycle' only.")
        if config.smoother not in {"jacobi", "chebyshev"}:
            raise ValueError(
                "implementation='fast' supports smoother='jacobi' or "
                "'chebyshev' only."
            )
        if WP_FLOAT is not wp.float64:
            raise RuntimeError(
                "The fast 3D river solver requires precision='float64'."
            )
    if transient_storage_enabled:
        if not transient_enabled:
            raise ValueError(
                "transient_storage_enabled=True requires transient_enabled=True."
            )
        if (
            braidplain_specific_storage <= 0.0
            and regional_specific_storage <= 0.0
        ):
            raise ValueError(
                "transient_storage_enabled=True requires at least one positive "
                "specific-storage value."
            )
    if transient_enabled:
        prepared_ogs_reference = ensure_ogs_reference(
            reference_path=ogs_reference_path,
            refresh_reference=refresh_ogs_reference,
            fe_repository=fe_repository,
            fe_config_path=fe_config_path,
            fe_run_directory=fe_run_directory,
        )
        run_transient_study(
            config=config,
            ratios=[float(value) for value in anisotropy_ratios],
            outlet_modes=list(outlet_modes),
            anisotropy_target=anisotropy_target,
            output_times_days=[float(value) for value in transient_output_times_days],
            channel_ramp_days=float(channel_ramp_days),
            braidplain_specific_storage=(
                float(braidplain_specific_storage)
                if transient_storage_enabled
                else 0.0
            ),
            regional_specific_storage=(
                float(regional_specific_storage)
                if transient_storage_enabled
                else 0.0
            ),
            maximum_timestep_days=float(maximum_timestep_days),
            progress_every_steps=int(progress_every_steps),
            output_directory=output_directory,
            ogs_reference=prepared_ogs_reference,
        )
        return

    from DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section import run_sweep

    results = run_sweep(
        base_config=config,
        ratios=[float(value) for value in anisotropy_ratios],
        outlet_modes=list(outlet_modes),
        anisotropy_target=anisotropy_target,
        output_directory=output_directory,
        save_heads=save_steady_heads,
    )
    write_results(
        results=results,
        config=config,
        output_directory=output_directory,
    )


if __name__ == "__main__":
    # ------------------------------------------------------------------
    # EDITABLE RUN SETTINGS
    # ------------------------------------------------------------------

    # Study mode. ``transient_enabled`` writes a time series. The second
    # switch controls whether that series solves real backward-Euler storage
    # steps or the zero-storage quasi-steady structural envelope.
    transient_enabled = True
    transient_storage_enabled = True
    precision = "float64"

    # Cases. ``water_table_only`` is the FE-aligned far-field structure;
    # ``full_depth`` and ``lower_only`` are deliberate structure tests.
    anisotropy_ratios = [10.0]
    anisotropy_target = "both"  # "braidplain", "regional", or "both"
    outlet_modes = ["water_table_only", "full_depth", "lower_only"]

    # Geometry and FE-aligned saturated conductivities [m/day].
    dx = 1.0
    dz = 1.0
    braidplain_kh = 200.0
    regional_kh = 20.0

    # Device and linear solver.
    device = "cuda:0"
    implementation = "fast"  # "fast" or "classic"
    solver = "kcycle"         # "kcycle" or classic-only "chebyshev"
    smoother = "chebyshev"    # fast: "chebyshev" or "jacobi"
    max_cycles = 200
    max_levels = 6
    min_coarse_n = 1
    rel_tol = 5.0e-5
    abs_tol_min = 5.0e-5
    dh_rms_tol = 1.0e-4
    check_every_no = 1
    practical_mass_imbalance_tol = 2.0e-6
    nu_pre = 6
    nu_post = 6
    nu_coarse = 2
    omega = 0.8

    # Classic-only vertical-line controls (ignored by the fast backend).
    line_omega = 0.8
    line_sweeps_pre = 1
    line_sweeps_post = 1
    line_sweeps_coarse = 1
    vertical_line_max_nz = 128

    # Robust acceptance retry.
    robust_retry_enabled = True
    robust_max_cycles = 800
    robust_nu_pre = 13
    robust_nu_post = 13
    robust_nu_coarse = 3
    robust_omega = 0.7
    robust_rel_tol = 1.0e-10
    robust_abs_tol_min = 1.0e-10
    robust_dh_rms_tol = 1.0e-8

    # Transient reporting and channel ramp.
    transient_output_times_days = [
        0.0,
        1.0 / 24.0,
        1.0,
        5.0,
        10.0,
        20.0,
        30.0,
        100.0,
        365.0,
        730.0,
        1000.0,
    ]
    channel_ramp_days = 1.0 / 24.0
    maximum_timestep_days = 0.25
    progress_every_steps = 100  # Set to 0 to suppress internal-step updates.
    # These constant saturated S_s values are used only when
    # transient_storage_enabled=True. They are sensitivity parameters, not an
    # exact representation of OGS's head-dependent van Genuchten capacity;
    # OGS's explicit saturated storage property is zero.
    braidplain_specific_storage = 2.0e-4  # [1/m]
    regional_specific_storage = 1.0e-4    # [1/m]

    # Outputs.
    output_case_name = (
        "dx_1m_transient_storage"
        if transient_storage_enabled
        else "dx_1m_quasi_steady"
    )
    output_directory = REPO_ROOT / "cross_section_results" / output_case_name
    save_steady_heads = False

    # OGS comparison. With this enabled, pressing Run exports the existing
    # fe_mesh_tester forward run automatically if the standardized artifact
    # does not exist. Set ogs_reference_path = None to disable comparison.
    fe_repository = Path("/home/patrickdurney/PycharmProjects/fe_mesh_tester")
    fe_config_path = fe_repository / "configs" / "braidplain.yaml"
    fe_run_directory = fe_repository / "results" / "forward_run"
    ogs_reference_path: Path | None = (
        REPO_ROOT / "cross_section_results" / "ogs_river_loss_reference.npz"
    )
    # False still refreshes automatically when the PVD/config/mesh is newer.
    # Set True to force rebuilding even when timestamps indicate it is current.
    refresh_ogs_reference = False

    # Precision must be selected before importing DarcyWarp configuration.
    os.environ["DARCY_FLOAT"] = precision
    from DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section import (
        CrossSectionConfig,
    )

    config = CrossSectionConfig(
        braidplain_kh=braidplain_kh,
        regional_kh=regional_kh,
        dx=dx,
        dz=dz,
        device=device,
        implementation=implementation,
        solver=solver,
        max_cycles=max_cycles,
        max_levels=max_levels,
        min_coarse_n=min_coarse_n,
        rel_tol=rel_tol,
        abs_tol_min=abs_tol_min,
        dh_rms_tol=dh_rms_tol,
        check_every_no=check_every_no,
        practical_mass_imbalance_tol=practical_mass_imbalance_tol,
        nu_pre=nu_pre,
        nu_post=nu_post,
        nu_coarse=nu_coarse,
        omega=omega,
        line_omega=line_omega,
        line_sweeps_pre=line_sweeps_pre,
        line_sweeps_post=line_sweeps_post,
        line_sweeps_coarse=line_sweeps_coarse,
        vertical_line_max_nz=vertical_line_max_nz,
        robust_retry_enabled=robust_retry_enabled,
        robust_max_cycles=robust_max_cycles,
        robust_nu_pre=robust_nu_pre,
        robust_nu_post=robust_nu_post,
        robust_nu_coarse=robust_nu_coarse,
        robust_omega=robust_omega,
        robust_rel_tol=robust_rel_tol,
        robust_abs_tol_min=robust_abs_tol_min,
        robust_dh_rms_tol=robust_dh_rms_tol,
        smoother=smoother,
    )
    run_study(
        transient_enabled=transient_enabled,
        transient_storage_enabled=transient_storage_enabled,
        config=config,
        anisotropy_ratios=anisotropy_ratios,
        outlet_modes=outlet_modes,
        anisotropy_target=anisotropy_target,
        output_directory=output_directory,
        save_steady_heads=save_steady_heads,
        transient_output_times_days=transient_output_times_days,
        channel_ramp_days=channel_ramp_days,
        maximum_timestep_days=maximum_timestep_days,
        progress_every_steps=progress_every_steps,
        braidplain_specific_storage=braidplain_specific_storage,
        regional_specific_storage=regional_specific_storage,
        ogs_reference_path=ogs_reference_path,
        refresh_ogs_reference=refresh_ogs_reference,
        fe_repository=fe_repository,
        fe_config_path=fe_config_path,
        fe_run_directory=fe_run_directory,
    )
