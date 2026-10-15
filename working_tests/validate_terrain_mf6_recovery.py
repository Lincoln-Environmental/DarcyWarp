"""Labelled MF6 convergence-control investigation for n32 XT3D failure."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    executable=Path('/bin/modflow/mf6')
    case=TerrainCase.load(directory=root/'cases'/'terrain_32_k0.0001')
    options=dict(under_relaxation='SIMPLE',under_relaxation_gamma=.5)
    outer_cap=8000
    ref=CPUReference(case=case)
    h=case.arrays['initial'].copy()
    h[case.arrays['fixed']!=0]=case.arrays['values'][case.arrays['fixed']!=0]
    op,unused,unused=ref.evaluate(head=h)
    contacts=op.fluxes(head=h)[3]
    run_mf6(case=case,directory=root/'mf6_solver_recovery'/'n32_xt3d_simple',executable=executable,
        xt3d=True,convertible=True,drains=False,contact_mode='prescribed_flux',contact_fluxes=contacts,
        outer_cap=outer_cap,ims_options=options)
