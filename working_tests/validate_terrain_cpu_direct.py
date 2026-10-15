"""Cold direct Newton with physical merit and direction-preserving damping."""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_cpu_runs import run_cpu


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    controls=CPUControls(max_outer=500,physical_merit=True,fallback=False,recovery_growth=1.)
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        directory=root/'cpu_direct_newton'/case.case_id
        if not directory.exists():
            run_cpu(case=case,directory=directory,controls=controls)
