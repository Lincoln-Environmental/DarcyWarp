"""Replay the archived Canterbury fit with its recorded stage-two settings."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from DARCY_WARP_PACKAGE.canterbury_case_study.Canterbury_case_study import run_best_solution_and_save_outputs
from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import load_case_inputs
from DARCY_WARP_PACKAGE.canterbury_case_study.compute_canterbury_rmse import _compute_metrics, _load_head


def replay(*, output, archive, device, hydraulics, grid_size):
    if output.resolve().is_relative_to(archive.resolve()):
        raise ValueError("Write the replay to a separate output directory.")
    output.mkdir(parents=True, exist_ok=True)
    fit = torch.load(archive / "results.pt", weights_only=False)
    archived_case = load_case_inputs(grid_size=grid_size, boundary_policy="legacy_archive")
    report = dict(
        archive_sha256={name: hashlib.sha256((archive / name).read_bytes()).hexdigest()
            for name in ("results.pt", "best_head.npz")},
        archived_export=_compute_metrics(head=_load_head(path=archive / "best_head.npz"), case=archived_case),
        archived_stage_two_observational_weighted_sse=float(fit["stage2"]["best_wsse"]),
        replays={},
    )
    for policy in ("legacy_archive", "sea_precedence"):
        case = load_case_inputs(grid_size=grid_size, boundary_policy=policy)
        result = run_best_solution_and_save_outputs(case=case, outputs=fit, out_dir=output / policy,
            device=device, archived_hydraulic_settings=hydraulics)
        report["replays"][policy] = {key: result[key] for key in (
            "hydraulic_settings", "metrics", "water_budget_m3_day", "free_ghb_cells", "boundary_audit",
        )}
        report["replays"][policy]["solver"] = {key: result["solver_info"][key] for key in (
            "converged", "n_cycles_used", "r_rms_end",
        )}
    (output / "replay.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()
    grid_size = 100
    archived_stage_two = dict(use_ghb=True, gh_alpha=2.0, aq_thickness=300.0,
        perturbation_strength=0.0, reg_weight=1.0, sigma_dlogT=0.2,
        solve=dict(max_cycles=200, nu_pre=2, nu_post=2, nu_coarse=2, omega=0.7,
            rel_tol=5e-6, abs_tol_min=5e-6, max_levels=6, check_every_no=10),
    )
    replay(output=args.output, archive=args.archive, device=args.device,
        hydraulics=archived_stage_two, grid_size=grid_size)
