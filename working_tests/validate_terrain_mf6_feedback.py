"""Custom MF6/independent-contact feedback; not native MF6 equivalence."""
from pathlib import Path
import json
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6


def run_feedback(*,case,directory,executable,max_feedback,contact_tolerance,relaxation,xt3d,initial=None):
    directory.mkdir(parents=True,exist_ok=False)
    ref=CPUReference(case=case)
    head=case.arrays['initial'].copy() if initial is None else initial.copy()
    head[case.arrays['fixed']!=0]=case.arrays['values'][case.arrays['fixed']!=0]
    op,unused,unused=ref.evaluate(head=head)
    q=op.fluxes(head=head)[3]
    history=[]
    converged=False
    for outer in range(max_feedback):
        path=directory/f'feedback_{outer:03d}'
        result=run_mf6(case=case,directory=path,executable=executable,xt3d=xt3d,convertible=True,
            drains=False,initial=head,outer_cap=4000,contact_mode='prescribed_flux',contact_fluxes=q)
        if not result['native_success'] or not (path/'extracted.npz').exists():
            history.append(dict(outer=outer,native_failure=result))
            break
        head=np.load(path/'extracted.npz')['head']
        op,unused,unused=ref.evaluate(head=head)
        actual=op.fluxes(head=head)[3]
        mismatch=float(np.max(np.abs(actual-q)))
        history.append(dict(outer=outer,requested_contact_flux=q.tolist(),recomputed_contact_flux=actual.tolist(),mismatch=mismatch))
        if mismatch<=contact_tolerance and result['independent_conservation']:
            converged=True
            break
        q=(1-relaxation)*q+relaxation*actual
    np.savez_compressed(directory/'head.npz',head=head,contacts=actual if history and 'mismatch' in history[-1] else q)
    atomic_json(path=directory/'summary.json',value=dict(input_hash=case.digest(),converged=converged,
        stop_reason='coupled_converged' if converged else 'coupled_or_native_failure',history=history,
        contact_tolerance_m3_day=contact_tolerance,relaxation=relaxation,xt3d=xt3d,
        initial_guess='canonical_cold' if initial is None else 'verification_initial_guess',
        formulation_equivalence=False,contact_law='independently recomputed exact DarcyWarp natural contacts; MF6 interior law retained'))


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    executable=Path('/bin/modflow/mf6')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    max_feedback=30
    contact_tolerance=1e-8
    relaxation=.5
    xt3d=False
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        path=root/'mf6_feedback_v3'/case.case_id
        if not path.exists():
            run_feedback(case=case,directory=path,executable=executable,max_feedback=max_feedback,
                contact_tolerance=contact_tolerance,relaxation=relaxation,xt3d=xt3d)
