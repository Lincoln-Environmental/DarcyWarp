"""Locate ungrounded active components in rejected regional NPF states."""
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def evaluate_components(*, archive, head, device):
    """Evaluate independent NPF topology and summarize every ungrounded part.

    :param archive: Original MF6 input workspace.
    :param head: Candidate head array to diagnose.
    :param device: Warp device used to construct the physical model.
    :return: Metrics, ungrounded component summaries, and the cell mask.
    """
    from DARCY_WARP_PACKAGE.mf6_import import (
        _schedules, _stress_period, import_simulation, load_mf6_workspace,
    )
    from DARCY_WARP_PACKAGE.solvers.npf662_3d import case_from_model, independent_reference
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

    simulation = load_mf6_workspace(directory=archive)
    imported = import_simulation(simulation=simulation)
    forcing = _stress_period(model=imported, schedules=_schedules(model=imported), kper=0)
    shape = imported.shape
    with WarpDarcySolver3D(
        nx=shape[2], ny=shape[1], nz=shape[0], dx=imported.dx,
        dy=imported.dy, device=device, formulation="mf6_6_6_2_npf",
        implementation="fast", solver="kcycle",
    ) as solver:
        solver.build_from_layers(
            z_interfaces=imported.interfaces, kx_field=imported.kx,
            ky_field=imported.ky, kz_field=imported.kz, active=imported.active,
            initial_head=imported.initial, **forcing,
        )
        case = case_from_model(model=solver, top=0., cell_type=imported.cell_type)
        reference, _ = independent_reference(model=solver, case=case)
        metrics, fields = reference.evaluate(head=head)
        unused_drn_entries, drain, unused_riv_entries, river = reference.boundary_terms(head=head)
        boundary_tangent = reference.ghb_conductance + drain.diagonal + river.diagonal
        edges = fields["conductance"] > 0.
        graph = coo_matrix(
            (np.ones(int(edges.sum())),
             (reference.left[edges], reference.right[edges])),
            shape=(case.active.size, case.active.size),
        ).tocsr()
        unused_count, labels = connected_components(csgraph=graph, directed=False)
        diagonal = fields["principal_diagonal"]
        dry_zero = (
            reference.free & (case.cell_types > 0) & (head <= case.bottoms)
            & (diagonal == 0.) & (fields["residual"] == 0.)
            & (case.source == 0.) & (fields["boundary_throughput"] == 0.)
        )
        if case.source_gross is not None:
            dry_zero &= np.asarray(case.source_gross) == 0.
        active_labels = set(labels[(case.active != 0).ravel() & ~dry_zero.ravel()])
        grounded = (case.fixed != 0) | (boundary_tangent > 0.)
        grounded_labels = set(labels[(case.active != 0).ravel() & grounded.ravel()])
        missing = sorted(active_labels - grounded_labels)
        mask = np.zeros(case.active.size, dtype=bool)
        components = []
        flat_head = np.asarray(head).ravel()
        for label in missing:
            cells = np.flatnonzero((labels == label) & (case.active != 0).ravel())
            mask[cells] = True
            cell_mask = np.zeros(case.active.size, dtype=bool)
            cell_mask[cells] = True
            face_mask = edges & cell_mask[reference.left] & cell_mask[reference.right]
            values = flat_head[cells]
            residual = fields["residual"].ravel()[cells]
            recharge = case.source.ravel()[cells]
            coordinates = [list(map(int, np.unravel_index(int(index), shape)))
                           for index in cells[:10]]
            components.append(dict(
                label=int(label), cells=int(cells.size), first_cells=coordinates,
                head_min=float(values.min()), head_max=float(values.max()),
                saturated_fraction=float(np.mean(fields["saturation"].ravel()[cells] == 1.)),
                dry_cells=int(np.count_nonzero(values <= case.bottoms.ravel()[cells])),
                internal_faces=int(np.count_nonzero(face_mask)),
                total_face_conductance=float(fields["conductance"][face_mask].sum()),
                source_net=float(recharge.sum()), source_gross=float(np.abs(recharge).sum()),
                ghb_cells=int(np.count_nonzero(reference.ghb_conductance.ravel()[cells] > 0.)),
                ghb_tangent=float(reference.ghb_conductance.ravel()[cells].sum()),
                active_drn_cells=int(np.count_nonzero(drain.diagonal.ravel()[cells] > 0.)),
                active_drn_tangent=float(drain.diagonal.ravel()[cells].sum()),
                residual_rms=float(np.sqrt(np.mean(residual**2))),
                residual_max_abs=float(np.max(np.abs(residual), initial=0.)),
                head_residual_max_abs=float(np.max(
                    np.abs(fields["residual"].ravel()[cells]
                           / np.maximum(diagonal.ravel()[cells], np.finfo(float).tiny)),
                    initial=0.)),
            ))
        return dict(metrics=metrics, components=components), mask.reshape(shape)


def diagnose_case(*, archive, native_directory, rejected_path, output, device):
    """Compare native and rejected heads and persist component masks.

    :param archive: Original MF6 input workspace.
    :param native_directory: Native MF6 reference output workspace.
    :param rejected_path: DarcyWarp rejected-state NPZ.
    :param output: Destination for JSON and spatial component mask.
    :param device: Warp device used for physical reference construction.
    """
    import flopy

    with flopy.utils.HeadFile(next(native_directory.glob("*.hds"))) as reader:
        native_head = reader.get_data()
    with np.load(rejected_path) as saved:
        rejected_head = saved["head"].copy()
    native, native_mask = evaluate_components(archive=archive, head=native_head, device=device)
    rejected, rejected_mask = evaluate_components(
        archive=archive, head=rejected_head, device=device)
    output.mkdir(parents=True, exist_ok=True)
    (output / "component_diagnostic.json").write_text(json.dumps(
        dict(model=archive.name, native=native, rejected=rejected),
        indent=2, allow_nan=False) + "\n")
    np.savez_compressed(output / "ungrounded_components.npz",
                        native=native_mask, rejected=rejected_mask)


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    model_root = repository / "USGS_examples"
    benchmark_root = repository / "benchmark_results" / "usgs_remaining_20261127"
    device = "cpu"
    model_names = [
        "0403_0404_0707_0709_0712_MF6_SS_Unconfined_250",
        "0414_0415_MF6_SS_Unconfined_250",
    ]
    for model_name in model_names:
        archive = next(model_root.glob(f"models.*/{model_name}"))
        native_directory = benchmark_root / "native" / model_name
        rejected_path = benchmark_root / model_name / "darcywarp_rejected.npz"
        output = benchmark_root / model_name
        diagnose_case(archive=archive, native_directory=native_directory,
                      rejected_path=rejected_path, output=output, device=device)
