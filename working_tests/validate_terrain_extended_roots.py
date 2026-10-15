"""Verification-only CPU confirmation of extended GPU roots; no cold claim."""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_branches import branch_experiments


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(16,1e-4),(96,1e-4)]
    controls=CPUControls(max_outer=30,head_tol=1e-10,flow_tol=1e-11,relative_tol=1e-11,recovery_growth=1.,fallback=False)
    amplitudes=[0.,1e-4,.01]
    seed=721
    radius=3
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        branch_experiments(case=case,root=root,output=root/'branches'/case.case_id,
            controls=controls,amplitudes=amplitudes,seed=seed,radius=radius)
