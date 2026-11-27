# SPDX-License-Identifier: AGPL-3.0-or-later
"""Run a supported existing MODFLOW 6 model using DarcyWarp.

Edit the explicit paths and device below, then run from the repository root:
python -m examples.run_mf6_model
"""

from pathlib import Path

from DARCY_WARP_PACKAGE.mf6_import import run_mf6_model


if __name__ == "__main__":
    model_directory = Path("/path/to/mf6/workspace")
    output_directory = Path("outputs/mf6_import")
    reference_head_file = None  # Optional existing native MF6 .hds file.
    device = "cuda:0"

    report = run_mf6_model(
        model_directory=model_directory,
        output_directory=output_directory,
        device=device,
        reference_head_file=reference_head_file,
        overwrite=True,
    )
    print(f"Accepted steps: {report['accepted_steps']}")
    print(f"Run time: {report['run_seconds']:.3f} s")
    print(f"Head output: {report['head_file']}")
