"""Independent CPU reference experiment matrix; all solves start cold unless labelled."""
from dataclasses import asdict
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase, atomic_json
from DARCY_WARP_PACKAGE.validation.records import start_record,check_cached,finish_record
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference, CPUControls, solve_cpu


def run_cpu(*, case, directory, controls, initial=None):
    settings=dict(controls=asdict(controls),initial_hash=None if initial is None else hashlib.sha256(initial.tobytes()).hexdigest())
    if check_cached(directory=directory,case=case,kind='cpu',settings=settings):
        return np.load(directory/'head.npz')['head'],json.loads((directory/'summary.json').read_text())
    directory.mkdir(parents=True,exist_ok=False)
    request=start_record(directory=directory,case=case,kind='cpu',settings=settings)
    head,result=solve_cpu(case=case,controls=controls,initial=initial)
    history=result.pop('history')
    np.savez_compressed(directory/'head.npz',head=head)
    (directory/'history.json.gz').write_bytes(gzip.compress(json.dumps(history).encode(),mtime=0))
    result.update(run_hash=request['run_hash'],input_hash=case.digest(),controls=asdict(controls),cold_start=initial is None,
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
    atomic_json(path=directory/'summary.json',value=result)
    finish_record(directory=directory)
    print(case.case_id,controls.method,result['converged'],result['stop_reason'],result['outers'],result['budget']['cell_residual_rms'],flush=True)
    return head,result


def derivative_check(*, case, seed, steps):
    ref=CPUReference(case=case)
    rng=np.random.default_rng(seed=seed)
    h=ref.grid.bottoms+.35*ref.grid.thickness
    h[~ref.free]=case.arrays['values'][~ref.free]
    v=rng.normal(size=h.shape)
    v[~ref.free]=0.
    op,r,b=ref.evaluate(head=h)
    jv=(ref.jacobian(operator=op,head=h)@v.ravel()).reshape(h.shape)
    rows=[]
    for eps in steps:
        unused,rp,unused_b=ref.evaluate(head=h+eps*v)
        unused,rm,unused_b=ref.evaluate(head=h-eps*v)
        fd=-(rp-rm)/(2*eps)
        rows.append(dict(epsilon=eps,relative_l2=float(np.linalg.norm((jv-fd)[ref.free])/np.linalg.norm(jv[ref.free]))))
    if min(r['relative_l2'] for r in rows)>2e-7:
        raise AssertionError(rows)
    return rows


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    seed=20261016
    steps=[1e-3,1e-4,1e-5,1e-6]
    max_outer=6000
    checks={}
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        checks[case.case_id]=derivative_check(case=case,seed=seed,steps=steps)
        atomic_json(path=root/'cpu_derivatives.json',value=checks)
        methods=['newton','picard'] if (size,k) in cases[:3] else ['newton']
        for method in methods:
            directory=root/'cpu_hybrid'/case.case_id/method
            if not directory.exists():
                run_cpu(case=case,directory=directory,controls=CPUControls(method=method,max_outer=max_outer,relaxation=1. if method=='newton' else .7))
