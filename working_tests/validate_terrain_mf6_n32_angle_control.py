"""One-factor XT3D connection-elevation control for n32 native stagnation."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    executable=Path('/bin/modflow/mf6')
    case=TerrainCase.load(directory=root/'cases'/'terrain_32_k0.0001')
    ref=CPUReference(case=case)
    head=case.arrays['initial'].copy()
    head[case.arrays['fixed']!=0]=case.arrays['values'][case.arrays['fixed']!=0]
    op,unused,unused=ref.evaluate(head=head)
    q=op.fluxes(head=head)[3]
    path=root/'mf6_n32_angle_control'
    result=run_mf6(case=case,directory=path,executable=executable,xt3d=True,convertible=True,drains=False,
        outer_cap=4000,contact_mode='custom_contact_xt3d',contact_fluxes=q,use_connection_elevations=False)
    if (path/'extracted.npz').exists():
        h=np.load(path/'extracted.npz')['head']
        op,unused,unused=ref.evaluate(head=h)
        actual=op.fluxes(head=h)[3]
        atomic_json(path=path/'contact_check.json',value=dict(requested=q.tolist(),actual=actual.tolist(),
            maximum_mismatch=float(np.max(np.abs(q-actual))),
            meaning='Altered connection geometry, not a fix or an equivalent physical terrain model.'))
