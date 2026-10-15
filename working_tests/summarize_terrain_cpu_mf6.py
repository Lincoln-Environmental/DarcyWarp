"""Offline saved-input/result comparison; no nonlinear solver advancement."""
from pathlib import Path
import json
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_compare import evaluate_saved_head,compare_fields,native_flux_fields,native_saved_balance


def collect_case(*,case,root,output):
    output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for group in ('gpu_before','gpu_after','cpu_hybrid','cpu_direct_newton','gpu_crossstarts','cpu_extended'):
        directory=root/group/case.case_id
        if (directory/'summary.json').exists():paths.append((group,directory))
        for path in directory.glob('*/summary.json'):
            paths.append((group+'/'+path.parent.name,path.parent))
    for path in (root/'extra_branches'/case.case_id).glob('*/summary.json'):
        if path.parent.name.startswith(('cpu_root','gpu_')):paths.append(('extra/'+path.parent.name,path.parent))
    for path in (root/'branches'/case.case_id).glob('*/summary.json'):
        if not path.parent.name.startswith('reduced'):paths.append(('branches/'+path.parent.name,path.parent))
    rows={};fields={}
    for label,directory in paths:
        metadata=json.loads((directory/'summary.json').read_text())
        if metadata['input_hash']!=case.digest():raise ValueError('stale result inputs')
        head=np.load(directory/'head.npz')['head']
        name=label.replace('/','_')
        result,state=evaluate_saved_head(case=case,head=head,output=output/(name+'.npz'))
        result['native']=metadata
        rows[label]=result
        fields[label]=state
    for path in (root/'mf6'/case.case_id).glob('native_xt3d_precise/summary.json'):
        metadata=json.loads(path.read_text())
        if (path.parent/'extracted.npz').exists():
            extracted=np.load(path.parent/'extracted.npz')
            result,state=evaluate_saved_head(case=case,head=extracted['head'],output=output/'mf6_native_darcy_evaluation.npz')
            result['native']=metadata
            rows['mf6_native']=result
            if state is not None:
                state=native_flux_fields(case=case,extracted=extracted,darcy_fields=state)
                state['contacts']=np.array([-record['inflow'] for record in metadata.get('packages',{}).get('DRN',[])])
                fields['mf6_native']=state
                rows['mf6_native']['independent_native_balance']=native_saved_balance(case=case,fields=extracted,contact_fluxes=state['contacts'])
    for group,label in [('mf6_feedback_v3','mf6_coupled'),('mf6_feedback_xt3d','mf6_xt3d_coupled'),('mf6_initial_shift','mf6_xt3d_initial_shift')]:
        directory=root/group/case.case_id
        if (directory/'summary.json').exists():
            metadata=json.loads((directory/'summary.json').read_text())
            head=np.load(directory/'head.npz')['head']
            result,state=evaluate_saved_head(case=case,head=head,output=output/(label+'_darcy_evaluation.npz'))
            result['native']=metadata
            rows[label]=result
            candidates=sorted(directory.glob('feedback_*/extracted.npz'))
            if candidates and state is not None:
                extracted=np.load(candidates[-1])
                state=native_flux_fields(case=case,extracted=extracted,darcy_fields=state)
                state['contacts']=np.load(directory/'head.npz')['contacts']
                rows[label]['independent_native_balance']=native_saved_balance(case=case,fields=extracted,contact_fluxes=state['contacts'])
                fields[label]=state
    comparisons={}
    reference='gpu_before/picard_reference' if 'gpu_before/picard_reference' in fields else 'gpu_before/picard'
    for label,state in fields.items():
        if state is not None and label!=reference:
            comparisons[label]=compare_fields(case=case,left=state,right=fields[reference])
    atomic_json(path=output/'metrics.json',value=rows)
    atomic_json(path=output/'comparisons.json',value=comparisons)
    return dict(case_id=case.case_id,runs={key:dict(independent_DarcyWarp_acceptance=row['independent_accepted'],
        head_rms=row.get('budget',{}).get('head_residual_rms'),physical_rms=row.get('budget',{}).get('cell_residual_rms')) for key,row in rows.items()},comparisons=comparisons)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    rows=[]
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        rows.append(collect_case(case=case,root=root,output=root/'analysis'/case.case_id))
    atomic_json(path=root/'comparison_matrix.json',value=rows)
