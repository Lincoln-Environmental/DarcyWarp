"""Investigate additional branches selected by the cold CPU direct-Picard solve."""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_extra_branches import extra_branch_checks


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(64,1e-4),(64,.1)]
    controls=CPUControls(max_outer=100,head_tol=1e-10,flow_tol=1e-11,relative_tol=1e-11,fallback=False,recovery_growth=1.)
    seed=1966
    amplitude=1e-4
    radius=8
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        extra_branch_checks(case=case,root=root,output=root/'extra_picard_branches'/case.case_id,
            controls=controls,seed=seed,amplitude=amplitude,radius=radius,cpu_method='picard')
