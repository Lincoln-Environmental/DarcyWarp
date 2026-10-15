"""Immutable canonical GPU baselines; direct configuration below."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import time
import numpy as np
from DARCY_WARP_PACKAGE.validation.records import start_record,check_cached
from DARCY_WARP_PACKAGE.validation.terrain_case import canonical_case, TerrainCase, atomic_json


def run_gpu(*, case, directory, method, outer_cap, initial=None):
    from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
    import warp as wp
    settings=dict(method=method,outer_cap=outer_cap,initial_hash=None if initial is None else hashlib.sha256(initial.tobytes()).hexdigest(),controls=case.controls)
    if check_cached(directory=directory,case=case,kind='gpu',settings=settings):
        return np.load(directory/'head.npz')['head'],json.loads((directory/'summary.json').read_text())
    directory.mkdir(parents=True)
    request=start_record(directory=directory,case=case,kind='gpu',settings=settings)
    a=case.arrays
    nz,ny,nx=case.shape
    solver=WarpDarcySolver3D(nx=nx,ny=ny,nz=nz,dx=case.physical['dx'],dy=case.physical['dy'],
        dz=float(a['thickness'].mean()),device='cuda:0',implementation='fast')
    solver.build_from_terrain(z_interfaces=a['interfaces'],active_columns=a['active'],kx_field=a['kx'],
        ky_field=a['ky'],kz_field=a['kz'],bc_mask=a['fixed'],bc_values=a['values'],rhs=a['source'],
        initial_head=a['initial'].copy() if initial is None else initial.copy(),contacts=case.contacts)
    c=case.controls
    with solver:
        wp.synchronize()
        start=time.perf_counter()
        head,info=solver.solve(unconfined=True,device_picard=True,terrain_nonlinear_method=method,
            unconfined_min_sat=case.physical['floor'],unconfined_max_picard_iter=outer_cap,
            unconfined_relax=c['relaxation'],unconfined_head_tol=c['head_tol'],final_head_residual_tol=c['head_tol'],
            rel_tol=1e-12,abs_tol_min=c['flow_tol'],final_budget_relative_tol=c['relative_tol'],
            max_cycles=c['krylov_budget'],krylov_restart=c['restart'],max_levels=c['max_levels'],min_coarse_n=c['min_coarse_n'])
        wp.synchronize()
        seconds=time.perf_counter()-start
    history=info.pop('outer_history')
    np.savez_compressed(directory/'head.npz',head=head)
    (directory/'history.json.gz').write_bytes(gzip.compress(json.dumps(history).encode(),mtime=0))
    result=dict(run_hash=request['run_hash'],method=method,input_hash=case.digest(),source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        outer_cap=outer_cap,cold_start=initial is None,seconds=seconds,info=info)
    atomic_json(path=directory/'summary.json',value=result)
    print(case.case_id,method,info['converged'],info['n_outer_iterations'],flush=True)
    return head,result


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    dx=20.
    reference_cap=8000
    for size,basement_k in cases:
        case=canonical_case(size=size,basement_k=basement_k,dx=dx)
        case_path=root/'cases'/case.case_id
        if not case_path.exists():
            case.save(directory=case_path)
        case=TerrainCase.load(directory=case_path)
        for method in ('picard','newton'):
            path=root/'gpu_before'/case.case_id/method
            if not path.exists():
                run_gpu(case=case,directory=path,method=method,outer_cap=case.controls['outer_cap'])
        if not json.loads((root/'gpu_before'/case.case_id/'picard'/'summary.json').read_text())['info']['converged']:
            path=root/'gpu_before'/case.case_id/'picard_reference'
            if not path.exists():
                run_gpu(case=case,directory=path,method='picard',outer_cap=reference_cap)
