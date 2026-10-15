"""Offline saved-input/result comparison; no nonlinear solver advancement."""
from pathlib import Path
import json
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_compare import evaluate_saved_head,compare_fields,native_flux_fields,native_saved_balance


def save_native_fields(*,path,state,residual):
    np.savez_compressed(path,head=state['head'],native_physical_residual=residual,
        saturated_thickness=state['saturated_thickness'],wet=state['wet'],dry=state['dry'],
        native_saturation=state.get('native_saturation',np.full_like(state['head'],np.nan)),water_table=state['water_table'],
        water_table_kind=state['water_table_kind'],water_table_layer=state['water_table_layer'],
        common_floor_threshold=state['floor'],qx=state['qx'],qy=state['qy'],qz=state['qz'],contacts=state['contacts'])


def collect_case(*,case,root,output):
    output.mkdir(parents=True,exist_ok=True)
    paths=[]
    for group in ('gpu_before','gpu_after','cpu_hybrid','cpu_direct_newton','gpu_crossstarts','cpu_extended','cpu_final'):
        directory=root/group/case.case_id
        if (directory/'summary.json').exists():paths.append((group,directory))
        for path in directory.glob('*/summary.json'):
            paths.append((group+'/'+path.parent.name,path.parent))
    for group,prefix in [('extra_branches','extra'),('extra_picard_branches','extra_picard')]:
        for path in (root/group/case.case_id).glob('*/summary.json'):
            if path.parent.name.startswith(('cpu_root','gpu_')):paths.append((prefix+'/'+path.parent.name,path.parent))
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
            native_fields=root/'native_final_fields'/case.case_id/'native'/'extracted.npz'
            extracted=np.load(native_fields if native_fields.exists() else path.parent/'extracted.npz')
            result,state=evaluate_saved_head(case=case,head=extracted['head'],output=output/'mf6_native_darcy_evaluation.npz')
            result['native']=metadata
            rows['mf6_native']=result
            if state is not None:
                state=native_flux_fields(case=case,extracted=extracted,darcy_fields=state)
                state['contacts']=np.array([-record['inflow'] for record in metadata.get('packages',{}).get('DRN',[])])
                fields['mf6_native']=state
                balance,residual=native_saved_balance(case=case,fields=extracted,contact_fluxes=state['contacts'],return_fields=True)
                rows['mf6_native']['independent_native_balance']=balance
                save_native_fields(path=output/'mf6_native_fields.npz',state=state,residual=residual)
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
                native_label='coupled_xt3d' if group=='mf6_feedback_xt3d' else 'coupled_tpfa'
                native_fields=root/'native_final_fields'/case.case_id/native_label/'extracted.npz'
                extracted=np.load(native_fields if native_fields.exists() and group!='mf6_initial_shift' else candidates[-1])
                state=native_flux_fields(case=case,extracted=extracted,darcy_fields=state)
                state['contacts']=np.load(directory/'head.npz')['contacts']
                balance,residual=native_saved_balance(case=case,fields=extracted,contact_fluxes=state['contacts'],return_fields=True)
                rows[label]['independent_native_balance']=balance
                save_native_fields(path=output/(label+'_native_fields.npz'),state=state,residual=residual)
                fields[label]=state
    comparisons={}
    reference='gpu_before/picard_reference' if 'gpu_before/picard_reference' in fields else 'gpu_before/picard'
    for label,state in fields.items():
        if state is not None and label!=reference:
            comparison=compare_fields(case=case,left=state,right=fields[reference])
            if label.startswith('mf6'):
                for key in ('qx','qy','qz'):comparison.pop(key)
                comparison['common_floor_threshold_crossings']=comparison.pop('floor_switches')
                comparison['native_floor_applicable']=False
                comparison['flux_mapping']='Compare integrated transfer between corresponding geological units and across complete Cartesian map planes. Native stepped and terrain faces are not claimed identical.'
                comparison['interlayer_transfer']=[dict(native_net=float(q.sum()),darcy_net=float(r.sum()),net_difference=float((q-r).sum()),native_downward=float(np.maximum(q,0.).sum()),native_upward=float(np.maximum(-q,0.).sum()),darcy_downward=float(np.maximum(r,0.).sum()),darcy_upward=float(np.maximum(-r,0.).sum())) for q,r in zip(state['qz'],fields[reference]['qz'])]
                comparison['full_column_x_plane_max_difference']=float(np.max(np.abs(state['qx'].sum(axis=(0,1))-fields[reference]['qx'].sum(axis=(0,1)))))
                comparison['full_column_y_plane_max_difference']=float(np.max(np.abs(state['qy'].sum(axis=(0,2))-fields[reference]['qy'].sum(axis=(0,2)))))
            comparisons[label]=comparison
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
