"""Pinned DIS/NPF writer and physical-flow decoder for Cartesian contract v1."""
from pathlib import Path
import hashlib
import subprocess
from time import perf_counter
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Reference,save_npf662_case
from DARCY_WARP_PACKAGE.validation.npf662_metrics import stable_rms,stable_sum
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics


def run_reference_mf6(*,case,directory,executable,prescribed_state=None,head_tolerance_m=1e-14,
                      native_solver_options=None,timeout_seconds=600,source_package='wel'):
    import flopy
    from flopy.mf6.utils.binarygrid_util import MfGrdFile
    if not executable.is_file():raise FileNotFoundError(f'Pinned local MF6 unavailable: {executable}; live parity unverified')
    reference=NPF662Reference(case=case)
    if source_package not in ('wel','rch'):raise ValueError('source_package must be wel or rch')
    if source_package=='rch' and (np.any(case.source[1:]!=0.) or np.any(case.source<0.) or np.any(case.source[case.fixed!=0]!=0.)):
        raise ValueError('Explicit fixed-top RCH mapping requires nonnegative source only in free top cells')
    if prescribed_state is not None:
        supplied=np.asarray(prescribed_state)
        if supplied.shape!=case.active.shape or not np.isfinite(supplied[case.active!=0]).all():
            raise ValueError('Supplied native head must be finite on the active canonical grid')
        if np.any((case.active!=0)&(case.cell_types!=0)&(supplied<case.bottoms)):
            raise ValueError('Supplied all-CHD state is below a convertible bottom; do not retype cells to force native evaluation')
    if not np.isfinite(head_tolerance_m) or head_tolerance_m<=0.:raise ValueError('Positive finite native head-change tolerance required')
    newton=bool(np.any(case.cell_types[case.active!=0]))
    version=subprocess.check_output([str(executable),'-v'],text=True).strip()
    checksum=hashlib.sha256(executable.read_bytes()).hexdigest()
    if version!='mf6: 6.6.2 05/12/2025' or checksum!='21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176':
        raise ValueError('Native executable differs from the pinned local MF6 contract')
    directory.mkdir(parents=True,exist_ok=False)
    nz,ny,nx=case.active.shape
    reverse=(slice(None),slice(None,None,-1),slice(None))
    values=case.values if prescribed_state is None else np.asarray(prescribed_state)
    fixed=case.fixed.astype(bool) if prescribed_state is None else case.active.astype(bool)
    save_npf662_case(case=case,path=directory/'canonical_inputs.npz')
    if prescribed_state is not None:np.savez_compressed(directory/'supplied_verification_head.npz',head=prescribed_state)
    sim=flopy.mf6.MFSimulation(sim_name='npf662',sim_ws=str(directory),exe_name=str(executable),verbosity_level=0)
    sim.simulation_data.float_precision=17;sim.simulation_data.float_characters=25
    flopy.mf6.ModflowTdis(sim,time_units='DAYS',nper=1,perioddata=[(1.,1,1.)])
    ims_options=dict(complexity='COMPLEX',print_option='ALL',outer_maximum=2000,
        outer_dvclose=head_tolerance_m,inner_maximum=2000,inner_dvclose=head_tolerance_m,rcloserecord=[1e-14,'STRICT'],
        linear_acceleration='BICGSTAB' if newton else 'CG',under_relaxation='DBD',relaxation_factor=.97,
        preconditioner_drop_tolerance=0.,
        scaling_method='DIAGONAL',
        preconditioner_levels=100,
        csv_outer_output_filerecord='outer.csv',csv_inner_output_filerecord='inner.csv')
    if native_solver_options is not None:
        ims_options.update(native_solver_options)
    flopy.mf6.ModflowIms(sim,**ims_options)
    gwf=flopy.mf6.ModflowGwf(sim,modelname='npf662',save_flows=True,newtonoptions='UNDER_RELAXATION' if newton else None)
    flopy.mf6.ModflowGwfdis(gwf,length_units='METERS',nlay=nz,nrow=ny,ncol=nx,
        delr=case.dx,delc=case.dy,top=case.tops[0,::-1,:],
        botm=case.bottoms[reverse],
        idomain=case.active[reverse].astype(int))
    flopy.mf6.ModflowGwfic(gwf,strt=(case.initial if prescribed_state is None else values)[reverse])
    flopy.mf6.ModflowGwfnpf(gwf,icelltype=case.cell_types[reverse],k=case.kx[reverse],k22=case.ky[reverse],k33=case.kz[reverse],
        angle1=0.,angle2=0.,angle3=0.,xt3doptions=None,save_flows=True,save_saturation=True,
        perched=False,thickstrt=False,cvoptions=None,rewet_record=None)
    bc=[];sources=[]
    for k,j,i in np.argwhere(case.active!=0):
        cell=(int(k),ny-1-int(j),int(i))
        if fixed[k,j,i]:bc.append((cell,float(values[k,j,i])))
        elif case.source[k,j,i]!=0:sources.append((cell,float(case.source[k,j,i])))
    if bc:flopy.mf6.ModflowGwfchd(gwf,stress_period_data=bc,save_flows=True)
    if sources:
        if source_package=='wel':
            flopy.mf6.ModflowGwfwel(gwf,pname='SOURCE',stress_period_data=sources,auto_flow_reduce=None,save_flows=True)
        else:
            recharge=case.source[0]/case.dx/case.dy
            flopy.mf6.ModflowGwfrcha(gwf,pname='RECHARGE',readasarrays=True,fixed_cell=True,
                irch=np.zeros((ny,nx),dtype=int),recharge=recharge[::-1,:],save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf,head_filerecord='npf662.hds',budget_filerecord='npf662.cbc',
        saverecord=[('HEAD','ALL'),('BUDGET','ALL')],printrecord=[('BUDGET','ALL')])
    sim.write_simulation(silent=True)
    start=perf_counter()
    try:
        run=subprocess.run([str(executable)],cwd=directory,capture_output=True,text=True,check=False,timeout=timeout_seconds)
        (directory/'stdout.log').write_text(run.stdout+run.stderr)
        native=run.returncode==0 and 'Normal termination of simulation' in run.stdout
        returncode=run.returncode;reason='native_converged' if native else 'native_failure_or_interrupted'
    except subprocess.TimeoutExpired as error:
        (directory/'stdout.log').write_bytes(error.stdout or b'')
        native=False;returncode=None;reason='wall_timeout'
    result=dict(native_success=native,returncode=returncode,stop_reason=reason,seconds=perf_counter()-start,
        input_hash=case.digest(),mf6_version=version,mf6_sha256=checksum,
        native_solver_options=ims_options,timeout_seconds=timeout_seconds,source_package=source_package,
        contract=case.contract+('_Newton' if newton else '_confined'),geometry=case.geometry,supplied_state_verification=prescribed_state is not None,
        settings=dict(xt3d=False,icelltype=case.cell_types.tolist(),newton=newton,linear_acceleration=ims_options['linear_acceleration'],variablecv=False,perched=False,thickstrt=False,rewet=False,
            steady=True,outer_dvclose_m=ims_options['outer_dvclose'],inner_dvclose_m=ims_options['inner_dvclose'],
            inner_rclose_m3_day=ims_options['rcloserecord'][0],scaling_method=ims_options['scaling_method'],
            preconditioner_levels=ims_options['preconditioner_levels'],
            preconditioner_drop_tolerance=ims_options['preconditioner_drop_tolerance'],
            relaxation_factor=ims_options['relaxation_factor']))
    paths=[directory/name for name in ('npf662.hds','npf662.cbc','npf662.dis.grb')]
    if not native or not all(path.is_file() and path.stat().st_size>0 for path in paths):
        result.update(independent_conservation=False,stop_reason='native_failure_or_missing_output')
        atomic_json(path=directory/'summary.json',value=result)
        return None,result,None
    raw=flopy.utils.HeadFile(str(directory/'npf662.hds'),precision='double').get_data()[reverse].copy()
    grid=MfGrdFile(str(directory/'npf662.dis.grb'),verbose=False)
    ia,ja=grid.ia,grid.ja
    budget=flopy.utils.CellBudgetFile(str(directory/'npf662.cbc'),precision='double')
    flow=budget.get_data(text='FLOW-JA-FACE')[-1].ravel()
    n=case.active.size
    active_mf6=case.active[reverse].ravel()!=0
    if len(ia)!=n+1 or len(ja)!=flow.size or not np.array_equal(np.diff(ia)>0,active_mf6):
        raise ValueError('Native full-user-grid connectivity mismatch')
    mapping=np.arange(n).reshape(case.active.shape)[reverse].ravel()
    edges={};reciprocity=0.
    for node in range(n):
        for p in range(ia[node]+1,ia[node+1]):
            other=int(ja[p]);a=int(mapping[node]);b=int(mapping[other])
            key=(min(a,b),max(a,b));q=-float(flow[p]) if a<b else float(flow[p])
            if key in edges:reciprocity=max(reciprocity,abs(edges[key]-q))
            edges[key]=q
    expected_edges={(int(a),int(b)) for a,b in zip(reference.left,reference.right)}
    if set(edges)!=expected_edges:
        raise ValueError('Native connectivity differs from canonical same-layer/no-pass-through DIS adjacency')
    flux=np.array([edges[(int(a),int(b))] for a,b in zip(reference.left,reference.right)])
    realised=np.zeros(n)
    if sources:
        source_text='WEL' if source_package=='wel' else 'RCHA'
        package_name='SOURCE' if source_package=='wel' else 'RECHARGE'
        for record in budget.get_data(text=source_text,paknam2=package_name)[-1]:
            realised[mapping[int(record['node'])-1]]+=float(record['q'])
    expected=np.where(case.active.astype(bool)&~fixed,case.source,0.)
    source_error=float(np.max(np.abs(realised.reshape(case.active.shape)-expected)))
    checks,fields=reference.evaluate(head=raw)
    native_sat=np.full(n,np.nan)
    for record in budget.get_data(text='DATA-SAT')[-1]:native_sat[mapping[int(record['node'])-1]]=float(record['sat'])
    native_sat=native_sat.reshape(case.active.shape)
    saturation_error=float(np.max(np.abs((native_sat-fields['saturation'])[case.active!=0])))
    physical=np.where(reference.free,case.source-(reference.incidence@flux).reshape(case.active.shape),0.)
    native_l1=stable_sum(values=np.abs(physical[reference.free]))
    native_relative=native_l1/max(checks['gross_throughput'],np.finfo(float).tiny)
    native_rms=stable_rms(values=physical[reference.free])
    if prescribed_state is not None:
        if not np.array_equal(raw[case.active!=0],values[case.active!=0]):raise ValueError('Supplied-head verification advanced a head')
        checks['independent_accepted']=False  # A constitutive check is not a root solve.
    result.update(independent=checks,source_allocation_max_error=source_error,reciprocity_max=reciprocity,saturation_max_error=saturation_error,
        native_flux_residual_rms=native_rms,native_flux_residual_relative=native_relative,
        independent_conservation=bool(native and source_error<=1e-14 and checks['input_valid'] and
            checks['boundary_condition_valid'] and checks['physical_metrics_valid'] and
            np.all(np.isfinite([native_rms,native_relative])) and (native_rms<=1e-12 or native_relative<=1e-8)))
    fields.update(raw_head=raw,native_saturation=native_sat,active=case.active,flux_native=flux,
        native_residual=physical,source_realised=realised.reshape(case.active.shape),node_to_user=mapping)
    np.savez_compressed(directory/'result.npz',**fields,head=raw)
    atomic_json(path=directory/'summary.json',value=finite_diagnostics(value=result))
    budget.close()
    return raw,result,fields
