"""Native MF6 actual terrain models, explicitly formulation-different."""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    executable=Path('/bin/modflow/mf6')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    xt3d=True
    convertible=True
    outer_cap=3000
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        path=root/'mf6'/case.case_id/'native_xt3d_precise'
        if not path.exists():
            run_mf6(case=case,directory=path,executable=executable,xt3d=xt3d,convertible=convertible,outer_cap=outer_cap)
