"""Independent branch polishing, perturbations, Jacobian diagnostics and reduction."""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
from scipy.sparse.linalg import splu,LinearOperator,onenormest,eigs
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUReference,CPUControls
from validate_terrain_cpu_runs import run_cpu


def constrained_diagnostics(*, case, head, seed=721):
    ref=CPUReference(case=case)
    op,r,b=ref.evaluate(head=head)
    f=ref.free.ravel()
    J=ref.jacobian(operator=op,head=head)[f][:,f].tocsc()
    principal=sum(op.principal_faces()).ravel()[f]
    for record,q in zip(op.contact_records,op.fluxes(head=head)[3]):
        if not record[6] or q>0:
            location=np.flatnonzero(f)==record[0]
            principal[location]+=record[1]
    from scipy import sparse
    A=(sparse.diags(1/principal)@J).tocsc()
    factor=splu(A=A)
    inverse=LinearOperator(shape=A.shape,matvec=factor.solve,rmatvec=TransposeSolve(factor=factor).solve)
    previous_random_state=np.random.get_state()
    np.random.seed(seed=seed)
    try:
        condition=float(onenormest(A=A)*onenormest(A=inverse))
    finally:
        np.random.set_state(previous_random_state)
    correction=factor.solve(rhs=r.ravel()[f]/principal)
    try:
        values=eigs(A=A,k=4,sigma=0.,return_eigenvectors=False,v0=np.random.default_rng(seed=seed).normal(size=A.shape[0]))
        near=[dict(real=float(v.real),imag=float(v.imag)) for v in values]
    except Exception as error:
        near=dict(unavailable=str(error))
    coo=A.tocoo()
    off=coo.row!=coo.col
    return dict(budget=b,scaled_condition_1_estimate=condition,minimum_abs_U_pivot=float(np.min(np.abs(factor.U.diagonal()))),
        correction_max=float(np.max(np.abs(correction))),correction_l2=float(np.linalg.norm(correction)),
        positive_offdiagonal_count=int(np.sum(coo.data[off]>0)),offdiagonal_count=int(off.sum()),
        near_zero_eigenvalues=near,free_unknowns=int(f.sum()),nnz=int(J.nnz))


class TransposeSolve:
    def __init__(self,*,factor):
        self.factor=factor

    def solve(self,vector):
        return self.factor.solve(rhs=vector,trans='T')


def branch_experiments(*, case, root, output, controls, amplitudes, seed, radius):
    output.mkdir(parents=True,exist_ok=True)
    rng=np.random.default_rng(seed=seed)
    heads={}
    rows={}
    for method in ('picard','newton'):
        source=root/'gpu_before'/case.case_id/method/'head.npz'
        if method=='picard':
            summary=json.loads(source.with_name('summary.json').read_text())
            if not summary['info']['converged']:
                source=root/'gpu_before'/case.case_id/'picard_reference'/'head.npz'
        initial=np.load(source)['head']
        for amplitude in amplitudes:
            label=f'{method}_perturb_{amplitude:g}'
            start=initial.copy()
            noise=rng.normal(size=start.shape)
            noise[case.arrays['fixed']!=0]=0.
            start+=amplitude*noise
            directory=output/label
            if not directory.exists():
                h,result=run_cpu(case=case,directory=directory,controls=controls,initial=start)
            h=np.load(directory/'head.npz')['head']
            result=json.loads((directory/'summary.json').read_text())
            heads[label]=h
            rows[label]=dict(result=result,diagnostics=constrained_diagnostics(case=case,head=h),
                max_shift_from_input=float(np.max(np.abs(h-initial))))
    atomic_json(path=output/'diagnostics.json',value=rows)
    left=heads['picard_perturb_0'];right=heads['newton_perturb_0']
    hotspot=tuple(int(i) for i in np.unravel_index(np.argmax(np.abs(left-right)),left.shape))
    unused,j,i=hotspot
    arrays={key:value.copy() for key,value in case.arrays.items()}
    # Explicit verification boundary problem: retain full geometry/gradient
    # stencils, freeze all cells outside a square to polished Picard heads.
    arrays['fixed'][:]=1
    arrays['fixed'][:,max(0,j-radius):j+radius+1,max(0,i-radius):i+radius+1]=0
    arrays['fixed'][case.arrays['fixed']!=0]=1
    arrays['values']=left.copy()
    arrays['initial']=left.copy()
    reduced=TerrainCase(case_id=case.case_id+'_local_dirichlet',arrays=arrays,physical=case.physical.copy(),controls=case.controls.copy(),contacts=())
    reduced_path=output/'reduced_case'
    if not reduced_path.exists():reduced.save(directory=reduced_path)
    reduced_heads={}
    for label,initial in [('a',left),('b',right)]:
        path=output/('reduced_'+label)
        if not path.exists():run_cpu(case=reduced,directory=path,controls=controls,initial=initial)
        reduced_heads[label]=np.load(path/'head.npz')['head']
    atomic_json(path=output/'reduction.json',value=dict(original_hotspot=hotspot,radius=radius,
        prescribed_boundary='all outside square fixed to independently polished Picard field; original full geometry retained',
        free_unknowns=int(np.sum(arrays['fixed']==0)),input_difference=float(np.max(np.abs(left-right))),
        final_difference=float(np.max(np.abs(reduced_heads['a']-reduced_heads['b']))),
        diagnostics={label:constrained_diagnostics(case=reduced,head=h) for label,h in reduced_heads.items()}))


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001)]
    controls=CPUControls(max_outer=30,head_tol=1e-10,flow_tol=1e-11,relative_tol=1e-11,recovery_growth=1.,fallback=False)
    amplitudes=[0.,1e-4,.01]
    seed=721
    radius=3
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        branch_experiments(case=case,root=root,output=root/'branches'/case.case_id,
            controls=controls,amplitudes=amplitudes,seed=seed,radius=radius)
