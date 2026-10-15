"""Independent host checks and same-state directions; direct configuration below.

Host matrices and downloads here are verification only, never solver work.
"""
import gzip
import json
from pathlib import Path
import shutil

from benchmark_terrain_newton import run_case

import numpy as np
from scipy.sparse.linalg import splu
import warp as wp

from shanz_terrain_case_study import build_case
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D, TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d
from DARCY_WARP_PACKAGE.solvers.terrain_device_3d import DeviceTerrainState, DevicePicardControls
from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import DeviceTerrainNewton
from DARCY_WARP_PACKAGE.solvers.terrain_fast_3d import pack_device_subfaces, pack_device_contact_refresh


def reference(*, case, head, floor, ky_ratio, kz_ratio):
    grid=TerrainGrid3D(z_interfaces=case['interfaces'],dx=case['dx'],dy=case['dx'],active_columns=case['active'])
    operator=TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=grid),kx=case['k'],
        ky=ky_ratio*case['k'],kz=kz_ratio*case['k'],head=head,min_sat=floor,contacts=case['contacts'])
    budget=compute_terrain_budget_3d(operator=operator,head=head,source=case['source'],fixed=case['fixed'],
        storage_change=np.zeros_like(a=head),storage_diagonal=np.zeros_like(a=head),return_fields=True)
    fields=dict(residual=budget.pop('cell_residual_field'))
    budget.pop('storage_change_field')
    fluxes=budget.pop('face_fluxes')
    fields.update(qx=fluxes[0],qy=fluxes[1],qz=fluxes[2],contacts=fluxes[3],
        floor_active=head-case['interfaces'][1:] <= floor,gates=fluxes[3]>0)
    return operator,budget,fields


def write_json(*, path, value):
    path.write_text(data=json.dumps(value,indent=2,default=float))


def verify_final_runs(*, directory, output, floor, ky_ratio, kz_ratio, head_tolerance, flow_tolerance, relative_tolerance, dx, krylov_budget):
    rows=[]
    comparisons=[]
    for path in sorted(directory.glob('*.json')):
        if path.stem.startswith('warmup') or not path.with_suffix('.npz').exists():
            continue
        result=json.loads(path.read_text())
        case=build_case(nx=result['size'],ny=result['size'],dx=dx)
        case['k'][-1]=result['basement_k']
        head=np.load(file=path.with_suffix('.npz'))['head']
        op,budget,fields=reference(case=case,head=head,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
        accepted=bool(np.all(a=np.isfinite(head)) and budget['head_residual_rms']<=head_tolerance and
            (budget['cell_residual_rms']<=flow_tolerance or budget['relative_cell_imbalance']<=relative_tolerance))
        history=json.loads(gzip.decompress(data=path.with_suffix('.history.json.gz').read_bytes()))
        max_work=max((row['krylov_iterations'] for row in history),default=0)
        rows.append(dict(run=path.stem,independent_accepted=accepted,reported_converged=result['info']['converged'],
            independent_budget=budget,max_outer_krylov=max_work,
            max_reported_budget_difference=max(abs(budget[key]-result['info']['budget'][key]) for key in
                ('head_residual_rms','cell_residual_rms','relative_cell_imbalance'))))
        np.savez_compressed(file=output/(path.stem+'_fields.npz'), **fields)
        if result['method']=='newton' and 'newton' in path.stem:
            paired=path.with_name(path.name.replace('newton','picard'))
            if paired.exists():
                old_head=np.load(file=paired.with_suffix('.npz'))['head']
                old_op,old_budget,old_fields=reference(case=case,head=old_head,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
                compare=dict(run=path.stem,reference=paired.stem,
                    head_rms_difference=float(np.sqrt(np.mean(a=(head-old_head)**2))),
                    head_max_difference=float(np.max(a=np.abs(head-old_head))),
                    residual_difference_l1=float(np.sum(a=np.abs(fields['residual']-old_fields['residual']))),
                    residual_difference_rms=float(np.sqrt(np.mean(a=(fields['residual'][case['fixed']==0]-old_fields['residual'][case['fixed']==0])**2))),
                    floor_state_differences=int(np.count_nonzero(a=fields['floor_active']!=old_fields['floor_active'])),
                    gate_state_differences=int(np.count_nonzero(a=fields['gates']!=old_fields['gates'])))
                for name in ('qx','qy','qz','contacts'):
                    difference=fields[name]-old_fields[name]
                    compare[name+'_max_difference']=float(np.max(a=np.abs(difference)))
                    compare[name+'_rms_difference']=float(np.sqrt(np.mean(a=difference**2)))
                comparisons.append(compare)
    write_json(path=output/'final_acceptance.json',value=rows)
    write_json(path=output/'solution_comparisons.json',value=comparisons)
    if not all(row['independent_accepted']==row['reported_converged'] and row['max_outer_krylov']<=krylov_budget for row in rows):
        raise AssertionError(rows)


def verify_checkpoint(*, checkpoint, output, floor, ky_ratio, kz_ratio, controls, relaxations, finite_difference_steps, seed):
    fields=np.load(file=checkpoint)
    head=fields['head']
    nz,ny,nx=head.shape
    case=build_case(nx=nx,ny=ny,dx=controls['dx'])
    case['k'][-1]=controls['basement_k']
    case['initial']=head
    operator,budget,host=reference(case=case,head=head,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
    free=case['fixed']==0
    solver=WarpDarcySolver3D(nx=nx,ny=ny,nz=nz,dx=case['dx'],dy=case['dx'],dz=controls['dz'],
        device=controls['device'],implementation='fast')
    solver.build_from_terrain(z_interfaces=case['interfaces'],kx_field=case['k'],ky_field=ky_ratio*case['k'],
        kz_field=kz_ratio*case['k'],active_columns=case['active'],bc_mask=case['fixed'],
        bc_values=case['values'],rhs=case['source'],initial_head=head,contacts=case['contacts'])
    with solver:
        solver.solve(max_cycles=0,krylov_restart=controls['restart'],max_levels=controls['max_levels'],min_coarse_n=controls['min_coarse_n'])
        session=solver._session
        subfaces,unused=pack_device_subfaces(operator=operator,datum=controls['datum'])
        s=DeviceTerrainState(session=session,workspace=session.workspace,device=controls['device'])
        s.upload_static(source=case['source'],sy=np.zeros_like(a=head),ss=np.zeros_like(a=head),free=free,
            subfaces=subfaces,contact_refresh=pack_device_contact_refresh(operator=operator,datum=controls['datum']))
        s.update_head(host_head=head)
        d=DeviceTerrainNewton(session=session,state=s,workspace=session.workspace,device=controls['device'],
            controls=DevicePicardControls(),datum=controls['datum'],restart=controls['restart'],krylov_budget=controls['budget'],
            unconfined=True,transient=False,dt=None,abs_tol_min=controls['flow_tol'],head_tolerance=controls['head_tol'],
            budget_relative_tol=controls['relative_tol'],rel_tol=controls['rel_tol'],min_sat=floor,
            outer_limit=controls['max_outer'],unconfined_relax=controls['relaxation'])
        d.evaluate(head_buffer=s.head,rel_buffer=s.head_rel)
        d.physical_residual(head=s.head_rel)
        diagonal=s.scale.numpy().reshape(head.shape)
        d.jacobian.prepare()
        rhs=host['residual']
        s.rhs.assign(src=(rhs/diagonal).ravel())
        s.delta.zero_()
        session.workspace.krylov._tol=0.
        session.workspace.krylov._atol=controls['picard_target']*d.sqrt_free
        session.workspace.krylov._maxiter=controls['budget']
        picard_work,picard_error,picard_tolerance=session.workspace.krylov()
        picard=s.delta.numpy().reshape(head.shape)
        matrix,offset=operator.sparse_matrix(head_for_contacts=head)
        direct=np.zeros_like(a=head)
        factor=splu(A=matrix[free.ravel()][:,free.ravel()].tocsc())
        direct[free]=factor.solve(rhs=rhs[free])
        s.rhs.assign(src=rhs.ravel())
        s.delta.zero_()
        d.newton_krylov._atol=d.newton_controls.physical_forcing*budget['cell_residual_rms']*d.sqrt_free
        d.newton_krylov._maxiter=controls['budget']
        work,error,tolerance=d.newton_krylov()
        newton=s.delta.numpy().reshape(head.shape)
        directions=dict(picard=picard,accurate_picard=direct,newton=newton)
        scans={}
        for name,delta in directions.items():
            scans[name]=[]
            for relaxation in relaxations:
                candidate=head+relaxation*np.clip(a=delta,a_min=-controls['max_change'],a_max=controls['max_change'])
                unused,trial,trial_fields=reference(case=case,head=candidate,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
                old_merit,new_merit=d.residual_merit(current=dict(head=budget['head_residual_rms'],flow=budget['cell_residual_rms']),
                    candidate=dict(head=trial['head_residual_rms'],flow=trial['cell_residual_rms']))
                scans[name].append(dict(relaxation=relaxation,budget=trial,old_merit=old_merit,merit=new_merit,
                    physical_ratio=trial['cell_residual_rms']/budget['cell_residual_rms'],
                    floor_switches=int(np.count_nonzero(a=trial_fields['floor_active']!=host['floor_active'])),
                    gate_switches=int(np.count_nonzero(a=trial_fields['gates']!=host['gates']))))
        rng=np.random.default_rng(seed=seed)
        v=rng.normal(size=head.shape)
        v[~free]=0.
        vector=wp.array(data=v.ravel(),dtype=wp.float64,device=controls['device'])
        jv=wp.zeros_like(src=vector)
        d.jacobian.matvec(x=vector,y=jv,z=jv,alpha=1.,beta=0.)
        actual=jv.numpy().reshape(head.shape)
        fd_errors=[]
        for eps in finite_difference_steps:
            unused,unused_budget,plus=reference(case=case,head=head+eps*v,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
            unused,unused_budget,minus=reference(case=case,head=head-eps*v,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio)
            fd=-(plus['residual']-minus['residual'])/(2*eps)
            fd_errors.append(dict(epsilon=eps,relative_l2=float(np.linalg.norm(x=(actual-fd)[free])/np.linalg.norm(x=actual[free])),
                floor_switches=int(np.count_nonzero(a=plus['floor_active']!=minus['floor_active']))))
        result=dict(checkpoint=str(checkpoint),current_budget=budget,scans=scans,finite_difference=fd_errors,
            picard_work=picard_work,picard_error_head_rms=picard_error/d.sqrt_free,
            accurate_picard_error_head_rms=float(np.sqrt(np.mean(a=((rhs.ravel()-matrix@direct.ravel())[free.ravel()]/diagonal[free])**2))),
            newton_work=work,newton_error_physical_rms=error/d.sqrt_free,newton_target_physical_rms=tolerance/d.sqrt_free)
        np.savez_compressed(file=output/'checkpoint_directions.npz',head=head,**directions,jv=actual,direction=v)
        write_json(path=output/'checkpoint_verification.json',value=result)
        d.close()
        s.close()



def polish_controls(*, directory, output, size, basement_k, controls, max_outer, device, floor, ky_ratio, kz_ratio):
    for method in ('picard','newton'):
        initial_path=directory/f'control_{method}_{size}_{basement_k:g}.npz'
        run_case(size=size,basement_k=basement_k,method='newton',max_outer=max_outer,
            output=output/(initial_path.stem+'.json'),device=device,controls=controls,
            floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio,verification_initial_path=initial_path)


if __name__ == "__main__":
    wp.init()
    output=Path('audit_outputs/verification')
    output.mkdir(parents=True,exist_ok=True)
    checkpoint_source=None  # Set a Path only to replace the delivered diagnostic input.
    checkpoint=output/'input_checkpoint_2000.npz'
    if checkpoint_source is not None:
        shutil.copyfile(src=checkpoint_source,dst=checkpoint)
    benchmark_directory=Path('audit_outputs/validated')
    floor=.01
    ky_ratio=.6
    kz_ratio=.15
    controls=dict(dx=20.,dz=5.,basement_k=.0001,device='cuda:0',restart=24,max_levels=6,min_coarse_n=16,
        datum=6.,budget=2000,flow_tol=1e-12,head_tol=1e-6,relative_tol=1e-8,rel_tol=1e-12,
        max_outer=2000,relaxation=.7,picard_target=3.969617457654214e-8,max_change=50.)
    relaxations=[1.,.7,.5,.1,.021875,.0109375]
    finite_difference_steps=[1e-3,1e-4,1e-5,1e-6]
    seed=235
    polish_cases=[(64,.1),(64,.01),(64,.001),(32,.0001)]
    polish_max_outer=100
    polish_controls_options=dict(dx=20.,unconfined_head_tol=1e-10,final_head_residual_tol=1e-10,
        rel_tol=1e-12,abs_tol_min=1e-14,final_budget_relative_tol=1e-12,
        unconfined_relax=.7,krylov_restart=24,max_levels=6,min_coarse_n=16,max_cycles=2000)
    polish_output=Path('audit_outputs/polish')
    verify_checkpoint(checkpoint=checkpoint,output=output,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio,
        controls=controls,relaxations=relaxations,finite_difference_steps=finite_difference_steps,seed=seed)
    verify_final_runs(directory=benchmark_directory,output=output,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio,
        head_tolerance=controls['head_tol'],flow_tolerance=controls['flow_tol'],relative_tolerance=controls['relative_tol'],dx=controls['dx'],krylov_budget=controls['budget'])

    for polish_size,polish_basement_k in polish_cases:
        polish_controls(directory=benchmark_directory,output=polish_output,size=polish_size,basement_k=polish_basement_k,
            controls=polish_controls_options,max_outer=polish_max_outer,device=controls['device'],floor=floor,
            ky_ratio=ky_ratio,kz_ratio=kz_ratio)
    verify_final_runs(directory=polish_output,output=polish_output,floor=floor,ky_ratio=ky_ratio,kz_ratio=kz_ratio,
        head_tolerance=polish_controls_options['final_head_residual_tol'],flow_tolerance=polish_controls_options['abs_tol_min'],
        relative_tolerance=polish_controls_options['final_budget_relative_tol'],dx=controls['dx'],krylov_budget=controls['budget'])
