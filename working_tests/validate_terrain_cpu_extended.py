"""Resource-bounded extended CPU reference with explicit stagnation reporting."""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_cpu_runs import run_cpu


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    case=TerrainCase.load(directory=root/'cases'/'terrain_96_k0.0001')
    controls=CPUControls(max_outer=2000,stagnation_window=100,stagnation_ratio=.995)
    run_cpu(case=case,directory=root/'cpu_extended'/case.case_id,controls=controls)
