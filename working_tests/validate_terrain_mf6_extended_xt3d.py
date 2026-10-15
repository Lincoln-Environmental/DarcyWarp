"""Extend the closest custom XT3D/contact comparison to all size/K controls."""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from validate_terrain_mf6_feedback import run_feedback


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    executable=Path('/bin/modflow/mf6')
    cases=[(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    max_feedback=30
    contact_tolerance=1e-8
    relaxation=.5
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        path=root/'mf6_feedback_xt3d'/case.case_id
        if not path.exists():
            run_feedback(case=case,directory=path,executable=executable,max_feedback=max_feedback,
                contact_tolerance=contact_tolerance,relaxation=relaxation,xt3d=True)
