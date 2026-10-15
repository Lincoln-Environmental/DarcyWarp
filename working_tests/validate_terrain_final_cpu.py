"""Repeat CPU references from committed numerical source with complete run hashes."""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_cpu_runs import run_cpu


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    newton_cap=6000
    picard_cap=6000
    extended_cap=2000
    extended_stagnation_window=100
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        controls=CPUControls(max_outer=newton_cap) if size!=96 else CPUControls(max_outer=extended_cap,stagnation_window=extended_stagnation_window)
        run_cpu(case=case,directory=root/'cpu_final'/case.case_id/'newton',controls=controls)
        if (size,k) in cases[:3]:
            run_cpu(case=case,directory=root/'cpu_final'/case.case_id/'picard',
                controls=CPUControls(method='picard',max_outer=picard_cap,relaxation=.7))
