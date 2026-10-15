"""Same-head term-removal diagnostics; not solutions of altered equations."""
import copy
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference
from DARCY_WARP_PACKAGE.validation.mf6_tpfa_reference import tpfa_fluxes
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D


def quantify(*,case,head):
    ref=CPUReference(case=case)
    fluxes={}
    for geometry in (False,True):
        metrics=copy.copy(ref.metrics)
        if not geometry:
            for name in ('zx','zy','zx_interfaces','zy_interfaces','zxf','zyf'):
                setattr(metrics,name,np.zeros_like(getattr(metrics,name)))
        for floor in (0.,case.physical['floor']):
            op=TerrainReferenceOperator3D(metrics=metrics,kx=case.arrays['kx'],ky=case.arrays['ky'],kz=case.arrays['kz'],
                head=head,min_sat=floor,contacts=())
            fluxes[(geometry,floor)]=op.fluxes(head=head)[:3]
    target=fluxes[(True,case.physical['floor'])]
    tpfa=tpfa_fluxes(case=case,head=head)
    rows={}
    for key,flow in fluxes.items():
        rows[str(key)]=dict(L1_difference_by_direction=[float(np.abs(q-r).sum()) for q,r in zip(flow,target)])
    interaction=[a-b-c+d for a,b,c,d in zip(fluxes[(True,case.physical['floor'])],fluxes[(False,case.physical['floor'])],fluxes[(True,0.)],fluxes[(False,0.)])]
    op,unused,unused=ref.evaluate(head=head)
    actual=op.fluxes(head=head)[3]
    native=[]
    for c in case.contacts:
        K=case.arrays['kx'][c.cell] if c.direction in ('west','east') else case.arrays['ky'][c.cell]
        C=K*c.area/c.distance if c.conductance is None else c.conductance
        native.append(C*max(head[c.cell]-c.head,0.))
    return dict(interpretation='same fixed head; term-removal/local sensitivity only, not altered-physics solution validation',
        terms=rows,geometry_floor_interaction_L1=[float(np.abs(q).sum()) for q in interaction],
        remaining_TPFA_upstream_vs_unfloored_harmonic_L1=[float(np.abs(q-r).sum()) for q,r in zip(tpfa,fluxes[(False,0.)])],
        natural_terrain_contacts=actual.tolist(),native_full_C_DRN=native,
        vertical_comparison='MF6 TPFA default vertical C matches full physical Kz half-cell series; terrain adds saturation-dependent slope normal/metric terms',
        integrated_source=float(case.arrays['source'].sum()))


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    rows={}
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        head=np.load(root/'gpu_before'/case.case_id/'newton'/'head.npz')['head']
        rows[case.case_id]=quantify(case=case,head=head)
    atomic_json(path=root/'formulation_terms.json',value=rows)
