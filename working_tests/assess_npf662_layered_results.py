"""Independent saved-head checks, operator probes and labelled local polishing."""
from dataclasses import replace
from itertools import combinations
from pathlib import Path
import hashlib
import json
import warnings

import numpy as np
from scipy.sparse.linalg import spsolve

from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
from DARCY_WARP_PACKAGE.validation.npf662_reference import (
    NPF662Reference, load_npf662_case, save_npf662_case)
from DARCY_WARP_PACKAGE.validation.npf662_metrics import stable_rms
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from validate_npf662_acceptance import reassess_v1, record_runtime
from validate_npf662_confined import make_gpu_model


def assess_case(*, directory, head_tolerance_m, flux_rtol, flux_atol):
    case=load_npf662_case(path=directory/'inputs.npz')
    original=json.loads((directory/'comparison.json').read_text())
    if original['input_hash']!=case.digest():raise ValueError('Stale input hash')
    reference=NPF662Reference(case=case)
    paths=dict(scipy_cpu=directory/'scipy_cpu.npz',warp_cpu=directory/'warp_cpu.npz',
        cuda=directory/'warp_cuda_0.npz',mf6=directory/'mf6/result.npz')
    methods={};heads={};flows={}
    for name,path in paths.items():
        with np.load(path,allow_pickle=False) as saved:
            head=saved['head'].copy()
            checks,fields=reference.evaluate(head=head)
            heads[name]=head;flows[name]=fields['flux']
            row=dict(independent=checks,returned_field_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if name=='mf6':
                native=json.loads((directory/'mf6/summary.json').read_text())
                row.update(native_success=native['native_success'],native_conservation=native['independent_conservation'],
                    same_head_flux_max_error=float(np.max(np.abs(saved['flux_native']-fields['flux']))),
                    same_head_source_allocation_error=native['source_allocation_max_error'])
                np.testing.assert_allclose(saved['flux_native'],fields['flux'],rtol=flux_rtol,atol=flux_atol)
            methods[name]=row
    pairs=[]
    for left,right in combinations(paths,2):
        error=float(np.max(np.abs((heads[left]-heads[right])[reference.free])))
        root_pair=methods[left]['independent']['independent_accepted'] and methods[right]['independent']['independent_accepted']
        pairs.append(dict(left=left,right=right,head_max_error_m=error,
            face_flux_max_error_m3_day=float(np.max(np.abs(flows[left]-flows[right]))),
            both_accepted=root_pair,same_root_head_gate=bool(root_pair and error<=head_tolerance_m),
            interpretation='Accepted near-root comparison' if root_pair else 'Failed iterate is not a different accepted root'))
    return dict(directory=str(directory),input_hash=case.digest(),geometry=case.geometry,methods=methods,pairs=pairs,
        physical_validation='Not established by matching a conventional DIS discretization')


def probe_device_and_differences(*, case, output, devices, seed, steps):
    import warp as wp
    from DARCY_WARP_PACKAGE.solvers.npf662_newton import NPF662Workspace
    output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(seed=seed)
    head=case.bottoms+(.45+.03*rng.random(size=case.active.shape))*case.thicknesses
    head[case.fixed!=0]=case.values[case.fixed!=0]
    reference=NPF662Reference(case=case)
    v=rng.normal(size=head.shape);v[~reference.free]=0.
    J=reference.jacobian(head=head)
    jv=(J@v.ravel()).reshape(head.shape)
    checks,fields=reference.evaluate(head=head)
    errors=[]
    for step in steps:
        unused,plus=reference.evaluate(head=head+step*v);unused,minus=reference.evaluate(head=head-step*v)
        fd=-(plus['residual']-minus['residual'])/(2*step)
        errors.append(dict(step_m=step,relative_l2=float(np.linalg.norm((fd-jv)[reference.free])/np.linalg.norm(jv[reference.free]))))
    if min(row['relative_l2'] for row in errors)>2e-7:raise AssertionError(errors)
    rows={}
    for device in devices:
        with make_gpu_model(case=case,device=device) as model:
            workspace=NPF662Workspace(model=model,case=replace(case,initial=head),restart=8,krylov_budget=24)
            metrics=workspace.evaluate(trial=False,linearize=True)
            q=workspace.q.numpy()[reference.left,2-reference.axes]
            residual=workspace.residual.numpy().reshape(head.shape)
            x=wp.array(data=v.ravel(),dtype=wp.float64,device=device);z=wp.zeros_like(src=x)
            workspace.matvec(x=x,y=z,z=z,alpha=1.,beta=0.)
            actual=(z.numpy()*workspace.scale.numpy()).reshape(head.shape)
            np.testing.assert_allclose(q,fields['flux'],rtol=3e-12,atol=2e-11)
            np.testing.assert_allclose(residual,fields['residual'],rtol=3e-12,atol=2e-11)
            np.testing.assert_allclose(actual[reference.free],jv[reference.free],rtol=3e-12,atol=2e-11)
            rows[device]=dict(metrics=metrics,flux_max_error=float(np.max(np.abs(q-fields['flux']))),
                residual_max_error=float(np.max(np.abs(residual-fields['residual']))),
                jv_max_error=float(np.max(np.abs((actual-jv)[reference.free]))))
            workspace.close()
    result=dict(finite_differences=errors,devices=rows,max_jacobian_columns=int(np.max(np.diff(J.indptr))),
        diagnostic_only=True,all_fields_downloaded_once=True)
    atomic_json(path=output/'summary.json',value=result)
    np.savez_compressed(output/'fields.npz',head=head,direction=v,jv=jv,**fields)
    return result


def diagnose_large_reference_failure(*, directory, output, seed, perturbation_m, max_outer, max_polish):
    output.mkdir(parents=True,exist_ok=False)
    case=load_npf662_case(path=directory/'inputs.npz')
    reference=NPF662Reference(case=case)
    head=np.load(directory/'warp_cuda_0.npz')['head'].copy()
    native=np.load(directory/'mf6/result.npz')['head'].copy()
    cold=json.loads((directory/'scipy_cpu.json').read_text())
    perturb=np.random.default_rng(seed=seed).normal(size=head.shape)*perturbation_m
    perturb[~reference.free]=0.
    restart=replace(case,initial=head+perturb)
    save_npf662_case(case=restart,path=output/'verification_restart_inputs.npz')
    solved,restarted,fields=NPF662Reference(case=restart).solve(max_outer=max_outer)
    np.savez_compressed(output/'verification_restart.npz',head=solved,**fields)
    initial=head.copy();history=[]
    for iteration in range(max_polish):
        checks,fields=reference.evaluate(head=head,physical_tolerance=1e-14,relative_tolerance=1e-12)
        if checks['independent_accepted']:break
        f=reference.free.ravel();A=reference.jacobian(head=head)[f][:,f].tocsc();rhs=fields['residual'].ravel()[f]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                delta=spsolve(A=A,b=rhs)
        except (RuntimeError,Warning) as error:
            history.append(dict(factorization_failure=str(error)));break
        linear=stable_rms(values=A@delta-rhs)
        if not np.isfinite(delta).all() or linear>max(1e-12,1e-8*checks['physical_rms']):
            history.append(dict(unusable_linear_solve=True,linear_residual_rms=linear));break
        direction=np.zeros_like(head);direction.ravel()[f]=delta
        accepted=False
        for attempt in range(20):
            alpha=2.**(-attempt)
            trial=head+alpha*direction
            after,unused=reference.evaluate(head=trial,physical_tolerance=1e-14,relative_tolerance=1e-12)
            if after['physical_metrics_valid'] and after['physical_rms']<checks['physical_rms']:
                head=trial;accepted=True;break
        history.append(dict(iteration=iteration+1,linear_residual_rms=linear,accepted=accepted,alpha=alpha))
        if not accepted:break
    checks,fields=reference.evaluate(head=head,physical_tolerance=1e-14,relative_tolerance=1e-12)
    np.savez_compressed(output/'verification_polished.npz',head=head,original_cuda=initial,**fields)
    result=dict(original_cold_cpu_accepted=cold['independent_accepted'],original_stop_reason=cold['stop_reason'],
        cold_final_attempts=cold['history'][-3:],verification_restart=dict(accepted=restarted['independent_accepted'],
            outers=restarted['outers'],checks=restarted,input_hash=restart.digest(),seed=seed,perturbation_m=perturbation_m),
        polishing=dict(checks=checks,history=history,head_max_change_m=float(np.max(np.abs((head-initial)[reference.free]))),
            head_max_difference_from_native_m=float(np.max(np.abs((head-native)[reference.free])))),
        interpretation='Verification-only restart/polishing; original cold failure and original head-parity failure are not relabelled')
    atomic_json(path=output/'summary.json',value=finite_diagnostics(value=result))
    return result


def assess_campaign(*, root, output, parents, devices, seed, steps, head_tolerance_m, flux_rtol, flux_atol):
    output.mkdir(parents=True,exist_ok=False)
    cases=[]
    for parent in parents:
        for path in sorted(parent.glob('*/comparison.json')):
            cases.append(assess_case(directory=path.parent,head_tolerance_m=head_tolerance_m,
                flux_rtol=flux_rtol,flux_atol=flux_atol))
    probe=load_npf662_case(path=root/'unconfined/experiments/heterogeneous_mixed/inputs.npz')
    derivatives=probe_device_and_differences(case=probe,output=output/'operator_probe',devices=devices,seed=seed,steps=steps)
    failure=diagnose_large_reference_failure(directory=root/'geology/varying_geology',output=output/'geology_failure',
        seed=seed,perturbation_m=1e-3,max_outer=200,max_polish=4)
    original=reassess_v1(original=Path('validation_artifacts/sevenpoint_v1'),output=output/'v1_reassessment')
    record_runtime(output=output,executable=Path('/bin/modflow/mf6'))
    result=dict(schema=1,cases=cases,returned_fields=sum(len(case['methods']) for case in cases),
        accepted_fields=sum(method['independent']['independent_accepted'] for case in cases for method in case['methods'].values()),
        original_v1_changed_classifications=original['changed_classifications'],derivatives=derivatives,
        larger_geology_diagnostic=failure,head_comparison_tolerance_m=head_tolerance_m,
        original_outputs_unchanged=True)
    atomic_json(path=output/'assessment.json',value=finite_diagnostics(value=result))
    return result


if __name__ == "__main__":
    root=Path('validation_artifacts/npf662_layered_v1')
    output=root/'final_assessment'
    parents=[root/'confined/experiments',root/'unconfined/experiments',root/'geology',root/'geology_4x4']
    devices=['cpu','cuda:0']
    seed=1129
    steps=(1e-3,1e-4,1e-5,1e-6,1e-7,1e-8)
    head_tolerance_m=1e-7
    flux_rtol=3e-12
    flux_atol=2e-11
    result=assess_campaign(root=root,output=output,parents=parents,devices=devices,seed=seed,steps=steps,
        head_tolerance_m=head_tolerance_m,flux_rtol=flux_rtol,flux_atol=flux_atol)
    print('Returned fields:',result['returned_fields'],'independently accepted:',result['accepted_fields'])
    print('Old V1 changed classifications:',result['original_v1_changed_classifications'])
