"""Generate numerical tables, work records and figures from saved evidence."""
import csv
import io
import json
from pathlib import Path
import tarfile
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_compare import evaluate_saved_head,compare_fields
from DARCY_WARP_PACKAGE.validation.evidence_storage import readable_archive
from validate_terrain_branches import constrained_diagnostics
from verify_mf6_tpfa_and_precision import longdouble_residual


def read_json(*,path):
    return json.loads(path.read_text())


def csv_work(*,text):
    rows=list(csv.DictReader(io.StringIO(text)))
    if not rows:return dict(outers=0,linear_iterations=0)
    last=rows[-1]
    return dict(outers=int(last['nouter']),linear_iterations=int(last['total_inner_iterations']))


def collect_native_work(*,root):
    work={}
    for record in read_json(path=root/'archives'/'index.json'):
        with readable_archive(root=root,record=record) as archive:
            with tarfile.open(archive,'r|*') as bundle:
                for member in bundle:
                    if member.isfile() and member.name.endswith('/outer.csv'):
                        work[str(Path(member.name).parent)]=csv_work(text=bundle.extractfile(member).read().decode())
    for path in root.rglob('outer.csv'):
        work[str(path.parent.relative_to(root))]=csv_work(text=path.read_text())
    atomic_json(path=root/'native_work.json',value=work)
    return work


def root_evidence(*,case,root,seed,threshold):
    directories={'GPU Picard':root/'branches'/case.case_id/'picard_perturb_0',
        'GPU Newton/Picard':root/'branches'/case.case_id/'newton_perturb_0'}
    if (root/'extra_branches'/case.case_id/'cpu_root'/'head.npz').exists():
        directories['CPU Newton/direct-Picard']=root/'extra_branches'/case.case_id/'cpu_root'
    if (root/'extra_picard_branches'/case.case_id/'cpu_root'/'head.npz').exists():
        directories['CPU direct Picard']=root/'extra_picard_branches'/case.case_id/'cpu_root'
    heads={};fields={};rows={};clusters=[]
    for label,path in directories.items():
        head=np.load(path/'head.npz')['head']
        diagnostics=constrained_diagnostics(case=case,head=head,seed=seed)
        result,state=evaluate_saved_head(case=case,head=head)
        heads[label]=head;fields[label]=state
        match=None
        for index,representative in enumerate(clusters):
            if np.max(np.abs(head-representative))<=threshold:match=index;break
        if match is None:match=len(clusters);clusters.append(head)
        rows[label]=dict(path=str(path.relative_to(root)),independent=result,
            constrained_jacobian=diagnostics,extended_precision=longdouble_residual(case=case,head=head),
            numerical_cluster=match)
    pairs={}
    labels=list(heads)
    for i,left in enumerate(labels):
        for right in labels[i+1:]:
            diff=np.abs(heads[left]-heads[right])
            cell=tuple(int(v) for v in np.unravel_index(np.argmax(diff),diff.shape))
            pairs[left+' versus '+right]=dict(metrics=compare_fields(case=case,left=fields[left],right=fields[right]),
                maximum_head_cell=cell,coordinate_m=[(cell[2]+.5)*case.physical['dx'],(cell[1]+.5)*case.physical['dy']],
                head_minus_bottom=[float(heads[left][cell]-case.grid().bottoms[cell]),float(heads[right][cell]-case.grid().bottoms[cell])])
    return dict(roots=rows,pairs=pairs,distinct_numerical_clusters=len(clusters),
        cluster_threshold_m=threshold,meaning='strong numerical branch evidence; no mathematical uniqueness/nonuniqueness or physical-stability proof')


def draw_figures(*,root,cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(nrows=2,ncols=3,figsize=(13,7),constrained_layout=True)
    arrays={}
    for col,(size,k,group,label) in enumerate(cases):
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        a=np.load(root/'branches'/case.case_id/'picard_perturb_0'/'head.npz')['head']
        b=np.load(root/group/case.case_id/label/'head.npz')['head']
        unused,fa=evaluate_saved_head(case=case,head=a)
        unused,fb=evaluate_saved_head(case=case,head=b)
        for row,name,field in [(0,'head',(b-a)[0]),(1,'interface0_flow',(fb['qz']-fa['qz'])[0])]:
            maximum=float(np.max(np.abs(field)))
            artist=axes[row,col].imshow(field,origin='lower',extent=[0,size*case.physical["dx"],0,size*case.physical["dy"]],cmap='coolwarm',vmin=-maximum,vmax=maximum)
            axes[row,col].set_title(f'{size}×{size}; basement Kx={k:g}\n'+
                ('Layer 0 head difference' if row==0 else 'Interface 0 flow difference'),fontsize=10)
            axes[row,col].set_xlabel('x [m]');axes[row,col].set_ylabel('y [m]')
            fig.colorbar(artist,ax=axes[row,col],label='m' if row==0 else 'm³/day')
            arrays[case.case_id+'_'+name]=field
    fig.savefig(root/'branch_differences.png',dpi=150)
    plt.close(fig)
    np.savez_compressed(root/'branch_figure_arrays.npz',**arrays)
    refinement=read_json(path=root/'fixed_boundary_refinement_v2'/'matrix.json')
    sizes=[8,16,32]
    fig,ax=plt.subplots(figsize=(6,4),constrained_layout=True)
    for method in ('mf6_tpfa','mf6_xt3d'):
        errors=[refinement[f'fixed_boundaries_refinement_{n}'][method]['head_rms'] for n in sizes]
        ax.loglog([320/n for n in sizes],errors,marker='o',label=method)
    ax.set_xlabel('dx=dy [m], physical domain fixed at 320 m');ax.set_ylabel('h=z head RMS error [m]')
    ax.legend();ax.grid(True,which='both');fig.savefig(root/'fixed_domain_refinement.png',dpi=150);plt.close(fig)


def generate_tables(*,root,cases,seed,cluster_threshold):
    native_work=collect_native_work(root=root)
    evidence={};matrix=[];costs=[];table=[]
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        metrics=read_json(path=root/'analysis'/case.case_id/'metrics.json')
        comparisons=read_json(path=root/'analysis'/case.case_id/'comparisons.json')
        evidence[case.case_id]=root_evidence(case=case,root=root,seed=seed,threshold=cluster_threshold)
        entries=[]
        labels=['gpu_before/picard','gpu_before/picard_reference','gpu_before/newton',
            'gpu_after/picard','gpu_after/newton','cpu_final/newton','cpu_final/picard',
            'mf6_coupled','mf6_xt3d_coupled']
        reference='gpu_before/picard_reference' if 'gpu_before/picard_reference' in metrics else 'gpu_before/picard'
        for label in labels:
            if label not in metrics:continue
            row=metrics[label];native=row['native'];is_mf6=label.startswith('mf6')
            info=native.get('info',native)
            success=bool(info.get('converged',False))
            independent=row.get('independent_native_balance',{}).get('conservation',False) if is_mf6 else row['independent_accepted']
            comparison=comparisons.get(label,{})
            same_root='not_equation_equivalent' if is_mf6 else ('not_accepted' if not success or not independent else 'reference' if label==reference else 'same_numerical_cluster' if comparison.get('head_max',np.inf)<=cluster_threshold else 'different_or_unconverged')
            entry=dict(case=case.case_id,solver=label,native_success=success,independent_conservation=independent,
                equation_equivalent_to_DarcyWarp=not is_mf6,same_root=same_root,
                physical_output_agreement='quantified separately; no cross-formulation pass threshold',
                validation_incomplete=bool(not success or not independent or is_mf6 or same_root=='different_or_unconverged'),
                stop_reason=info.get('device_stop_reason',info.get('stop_reason')))
            matrix.append(entry)
            budget=row.get('independent_native_balance',{}) if is_mf6 else row['budget']
            if is_mf6:
                prefix=('mf6_feedback_xt3d' if label=='mf6_xt3d_coupled' else 'mf6_feedback_v3')+'/'+case.case_id+'/'
                records=[v for key,v in native_work.items() if key.startswith(prefix)]
                outer=sum(v['outers'] for v in records);linear=sum(v['linear_iterations'] for v in records)
                seconds=sum(read_json(path=p).get('seconds',0.) for p in (root/prefix).glob('feedback_*/summary.json'))
                factors=None;jv=None;evaluations=None
            else:
                outer=info.get('n_outer_iterations',info.get('outers'));linear=info.get('krylov_iterations')
                seconds=native.get('seconds',info.get('seconds'));factors=info.get('factorizations')
                jv=info.get('jacobian_vector_products');evaluations=info.get('nonlinear_residual_evaluations',info.get('residual_evaluations'))
            costs.append(dict(**entry,outers=outer,linear_iterations=linear,direct_factorizations=factors,
                jacobian_vector_products=jv,nonlinear_state_evaluations=evaluations,wall_seconds=seconds,
                candidate_evaluations=info.get('candidate_evaluations'),fallback_count=info.get('fallback_count'),
                nonlinear_backtracks=info.get('nonlinear_backtracks'),
                head_equivalent_DarcyWarp_rms=row['budget']['head_residual_rms'],physical_rms=budget.get('cell_residual_rms'),
                physical_max=budget.get('cell_residual_max'),residual_L1=budget.get('cell_residual_l1',row.get('residual_l1')),
                gross_throughput=budget.get('gross_external_throughput',budget.get('throughput')),
                signed_imbalance=budget.get('signed_imbalance',budget.get('imbalance')),
                relative_cell_imbalance=budget.get('relative_cell_imbalance')))
            entries.append(f'{label}: {"success" if success else "FAIL"}/{"balance" if independent else "FAIL balance"}')
        table.append(dict(case=case.case_id,clusters=evidence[case.case_id]['distinct_numerical_clusters'],entries=entries))
    atomic_json(path=root/'root_evidence.json',value=evidence)
    atomic_json(path=root/'validation_matrix.json',value=matrix)
    atomic_json(path=root/'run_metrics.json',value=costs)
    with (root/'run_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(costs[0]));writer.writeheader();writer.writerows(costs)
    atomic_json(path=root/'table_inputs.json',value=table)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    seed=721
    cluster_threshold=1e-4
    figure_cases=[(32,1e-4,'branches','newton_perturb_0'),(64,1e-4,'extra_picard_branches','cpu_root'),(64,.1,'extra_picard_branches','cpu_root')]
    generate_tables(root=root,cases=cases,seed=seed,cluster_threshold=cluster_threshold)
    draw_figures(root=root,cases=figure_cases)
