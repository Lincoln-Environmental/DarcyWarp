#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-only
"""Export an existing fe_mesh_tester OGS run for DarcyWarp comparison.

Run this script with the ``fe_mesh_tester`` Python environment because its
post-processing stack supplies PyVista, YAML, and the benchmark constitutive
functions.  The output is a dependency-light NPZ consumed by
``run_river_loss_cross_section.py --transient --ogs-reference ...``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


def load_config(config_path: Path) -> dict:
    """Load the FE benchmark configuration."""

    import yaml

    with config_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a mapping in {config_path}.")
    return config


def load_converted_meshes(mesh_directory: Path) -> dict[str, object]:
    """Read the converted domain and boundary VTUs used by the OGS run."""

    import pyvista as pv

    paths = sorted(mesh_directory.glob("*.vtu"))
    if not paths:
        raise FileNotFoundError(f"No VTU meshes found under {mesh_directory}.")
    return {path.stem: pv.read(path) for path in paths}


def evaluate_existing_run(
    *,
    fe_repository: Path,
    config_path: Path,
    run_directory: Path,
) -> tuple[dict, object, dict]:
    """Use fe_mesh_tester's own evaluator on an already completed OGS run."""

    source_directory = fe_repository / "src"
    if str(source_directory) not in sys.path:
        sys.path.insert(0, str(source_directory))

    from fe_mesh_optimisation import ogs_pipeline
    from fe_mesh_optimisation.benchmarks import registry

    config = load_config(config_path=config_path)
    benchmark = registry.from_config(config)
    meshes = load_converted_meshes(
        mesh_directory=run_directory / "mesh" / "vtu"
    )
    mesh = ogs_pipeline.meshes_to_meshdata(meshes)
    pvd_path = run_directory / "results" / "model.pvd"
    if not pvd_path.is_file():
        raise FileNotFoundError(pvd_path)
    outcome = ogs_pipeline.evaluate_results(
        config=config,
        benchmark=benchmark,
        mesh=mesh,
        pvd_path=pvd_path,
        solver_cfg=config["solver"],
    )
    return outcome, mesh, config


def write_reference(
    *,
    output_path: Path,
    outcome: dict,
    mesh: object,
    config: dict,
    fe_repository: Path,
    run_directory: Path,
) -> None:
    """Write the standardized, self-describing comparison artifact."""

    times = np.asarray(outcome["times_days"], dtype=np.float64)
    pressure_heads = np.asarray(
        [outcome["heads"][float(time)] for time in times],
        dtype=np.float64,
    )
    saturations = np.asarray(
        [outcome["saturations"][float(time)] for time in times],
        dtype=np.float64,
    )
    coordinates = np.asarray(mesh.coordinates, dtype=np.float64)
    aquifer_thickness = float(
        config["benchmark"]["geometry"]["aquifer_thickness_m"]
    )
    total_heads = pressure_heads + coordinates[None, :, 1] + aquifer_thickness
    metadata = {
        "schema_version": 1,
        "source": "fe_mesh_tester OGS RICHARDS_FLOW post-processing",
        "fe_repository": str(fe_repository.resolve()),
        "run_directory": str(run_directory.resolve()),
        "scientific_scope": "Richards-flow reference for saturated-Darcy structural comparison",
        "head_datum": f"local OGS elevation + {aquifer_thickness:g} m",
        "channel_flow_scope": "one-sided half-channel",
        "flow_units": "m3/day per metre out of plane",
        "config": config,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        times_days=times,
        pressure_heads=pressure_heads,
        total_heads=total_heads,
        saturations=saturations,
        storage=np.asarray(outcome["storage"], dtype=np.float64),
        net_inflow=np.asarray(outcome["net_inflow"], dtype=np.float64),
        channel_inflow=np.asarray(outcome["channel_inflow"], dtype=np.float64),
        coordinates=coordinates,
        elements=np.asarray(mesh.elements, dtype=np.int64),
        element_regions=np.asarray(mesh.regions(), dtype=np.int64),
        boundary_edges=np.asarray(mesh.boundary_edges, dtype=np.int64),
        boundary_tags=np.asarray(mesh.boundary_tags, dtype=str),
        metadata_json=np.asarray(json.dumps(metadata)),
    )
    print(f"Wrote {output_path}")


def main(
    *,
    fe_repository: Path,
    config_path: Path,
    run_directory: Path,
    output_path: Path,
) -> None:
    """Export one completed FE run."""

    outcome, mesh, config = evaluate_existing_run(
        fe_repository=fe_repository,
        config_path=config_path,
        run_directory=run_directory,
    )
    write_reference(
        output_path=output_path,
        outcome=outcome,
        mesh=mesh,
        config=config,
        fe_repository=fe_repository,
        run_directory=run_directory,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--fe-repository",
        type=Path,
        default=Path("/home/patrickdurney/PycharmProjects/fe_mesh_tester"),
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--run-directory", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("cross_section_results/ogs_river_loss_reference.npz"),
    )
    return parser


if __name__ == "__main__":
    argument_parser = build_parser()
    arguments = argument_parser.parse_args()
    selected_fe_repository = arguments.fe_repository.resolve()
    selected_config_path = (
        arguments.config.resolve()
        if arguments.config is not None
        else selected_fe_repository / "configs" / "braidplain.yaml"
    )
    selected_run_directory = (
        arguments.run_directory.resolve()
        if arguments.run_directory is not None
        else selected_fe_repository / "results" / "forward_run"
    )
    main(
        fe_repository=selected_fe_repository,
        config_path=selected_config_path,
        run_directory=selected_run_directory,
        output_path=arguments.output.resolve(),
    )
