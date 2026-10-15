"""Native MF6 6.6.2 DIS/NPF adapter and independently checked budget extraction.

This is explicitly a different formulation: DIS stair-step geological faces,
MF6 saturation/upstream law without DarcyWarp's thickness floor, native DRN
with fully saturated natural C and no terrain centroid/gradient feedback.
No fitted conductance, THICKSTRT, RCH relocation, storage or warm-up period.
"""
from pathlib import Path
import hashlib
import subprocess
import time

import numpy as np

from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json,mf6_cell
from DARCY_WARP_PACKAGE.validation.records import start_record,check_cached,finish_record
import json


def build_mf6(*, case, directory: Path, executable: Path, xt3d: bool, convertible: bool,
               initial=None, drains=True, outer_cap=3000, head_tol=1e-9, residual_tol=1e-7, contact_mode="confined_native", contact_fluxes=None, ims_options=None, use_connection_elevations=True):
    import flopy
    a=case.arrays
    nz,ny,nx=case.shape
    reverse=(slice(None),slice(None,None,-1),slice(None))
    sim=flopy.mf6.MFSimulation(sim_name='terrain',sim_ws=str(directory),exe_name=str(executable),verbosity_level=0)
    sim.simulation_data.float_precision=17
    sim.simulation_data.float_characters=25
    flopy.mf6.ModflowTdis(sim,time_units='DAYS',nper=1,perioddata=[(1.,1,1.)])
    linear_options=dict(under_relaxation='DBD')
    if ims_options is not None:linear_options.update(ims_options)
    flopy.mf6.ModflowIms(sim,print_option='ALL',complexity='COMPLEX',outer_maximum=outer_cap,
        outer_dvclose=head_tol,inner_maximum=2000,inner_dvclose=head_tol,rcloserecord=[residual_tol,'STRICT'],
        linear_acceleration='BICGSTAB',relaxation_factor=.97,**linear_options,
        csv_outer_output_filerecord='outer.csv',csv_inner_output_filerecord='inner.csv')
    gwf=flopy.mf6.ModflowGwf(sim,modelname='terrain',save_flows=True,
        newtonoptions='UNDER_RELAXATION' if convertible else None)
    flopy.mf6.ModflowGwfdis(gwf,length_units='METERS',nlay=nz,nrow=ny,ncol=nx,
        delr=case.physical['dx'],delc=case.physical['dy'],top=a['interfaces'][0,::-1,:],
        botm=a['interfaces'][1:,::-1,:],idomain=case.grid().active[reverse].astype(int))
    h=a['initial'].copy() if initial is None else initial.copy()
    h[a['fixed']!=0]=a['values'][a['fixed']!=0]
    flopy.mf6.ModflowGwfic(gwf,strt=h[reverse])
    celltype=np.full(case.shape,int(convertible),dtype=int)
    if drains and contact_mode=='confined_native':
        for contact in case.contacts:
            celltype[contact.cell]=0
    flopy.mf6.ModflowGwfnpf(gwf,save_flows=True,save_saturation=True,save_specific_discharge=True,
        icelltype=celltype[reverse],k=a['kx'][reverse],k22=a['ky'][reverse],k33=a['kz'][reverse],
        angle1=0.,angle2=0. if use_connection_elevations else None,angle3=0. if use_connection_elevations else None,xt3doptions=True if xt3d else None,
        perched=False,thickstrt=False,cvoptions=None)
    fixed=[]
    for cell_array in np.argwhere(a['fixed']!=0):
        cell=tuple(int(x) for x in cell_array)
        fixed.append((mf6_cell(cell=cell,shape=case.shape),float(a['values'][cell])))
    if fixed:
        flopy.mf6.ModflowGwfchd(gwf,stress_period_data=fixed,save_flows=True)
    wells=[]
    for cell_array in np.argwhere((a['source']!=0)&(a['fixed']==0)):
        cell=tuple(int(x) for x in cell_array)
        wells.append((mf6_cell(cell=cell,shape=case.shape),float(a['source'][cell])))
    if wells:
        flopy.mf6.ModflowGwfwel(gwf,pname='SOURCE',stress_period_data=wells,save_flows=True,print_flows=True,
            auto_flow_reduce=None)
    drn=[]
    ghb=[]
    if contact_fluxes is not None:
        feedback=[(mf6_cell(cell=c.cell,shape=case.shape),-float(q)) for c,q in zip(case.contacts,contact_fluxes)]
        flopy.mf6.ModflowGwfwel(gwf,pname='CONTACT',filename='contact.wel',stress_period_data=feedback,save_flows=True)
    if drains:
        for contact in case.contacts:
            normal=a['kx'][contact.cell] if contact.direction in ('west','east') else a['ky'][contact.cell]
            area=contact.area
            if contact.direction in ('top','bottom'):
                if contact.kind!='head':raise ValueError('native top/bottom DRN unsupported')
                from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainMetrics3D
                metrics=TerrainMetrics3D(grid=case.grid())
                k,j,i=contact.cell
                interface=k+int(contact.direction=='bottom')
                factor=np.sqrt(1+metrics.zx_interfaces[interface,j,i]**2+metrics.zy_interfaces[interface,j,i]**2)
                area=contact.area/factor
                normal=a['kz'][contact.cell]
            c=normal*area/contact.distance if contact.conductance is None else contact.conductance
            entry=(mf6_cell(cell=contact.cell,shape=case.shape),contact.head,c)
            if contact.kind=='head':ghb.append(entry)
            else:drn.append(entry)
        if ghb:
            flopy.mf6.ModflowGwfghb(gwf,stress_period_data=ghb,save_flows=True)
        if drn:
            flopy.mf6.ModflowGwfdrn(gwf,stress_period_data=drn,save_flows=True,print_flows=True)
    flopy.mf6.ModflowGwfoc(gwf,head_filerecord='terrain.hds',budget_filerecord='terrain.cbc',
        saverecord=[('HEAD','ALL'),('BUDGET','ALL')],printrecord=[('BUDGET','ALL')])
    sim.write_simulation(silent=True)
    settings=dict(input_hash=case.digest(),representation='native_DIS_XT3D' if xt3d else 'native_DIS_TPFA',
        contact_mode=contact_mode,contact_cell_confined_override=bool(drains and contact_mode=='confined_native'),convertible=convertible,newton=convertible,xt3d=xt3d,steady=True,storage=False,thickstrt=False,
        variablecv=False,perched=False,rewet=False,angle_degrees=[0,0,0] if use_connection_elevations else [0,None,None],source_package='WEL without AUTO_FLOW_REDUCE',
        requested_source=float(sum(w[1] for w in wells)),drains=drn,initial_hash=hashlib.sha256(h.tobytes()).hexdigest(),
        outer_cap=outer_cap,outer_dvclose_m=head_tol,inner_dvclose_m=head_tol,inner_rclose_m3_day=residual_tol,
        formulation_equivalent_to_terrain=False,
        differences=['DIS stepped horizontal/vertical interfaces, no terrain metric-defect stencil',
            'MF6 quadratic saturation/upstream weighting rather than harmonic K*clip(h-bottom,.01,T)',
            'native DRN fixed full natural C, no saturation or centroid-gradient correction'])
    atomic_json(path=directory/'adapter.json',value=settings)
    return sim


def extract_mf6(*, case, directory: Path, native_success: bool, output_directory=None):
    import flopy
    from flopy.mf6.utils.binarygrid_util import MfGrdFile
    paths=[directory/name for name in ('terrain.hds','terrain.cbc','terrain.dis.grb')]
    if not all(path.exists() and path.stat().st_size>0 for path in paths):
        return dict(native_success=native_success,independent_conservation=False,extraction='missing_output')
    adapter=directory/'adapter.json'
    if not adapter.is_file():return dict(native_success=native_success,independent_conservation=False,extraction='missing_input_manifest')
    if json.loads(adapter.read_text())['input_hash']!=case.digest():raise ValueError('MF6 input hash mismatch')
    destination=directory if output_directory is None else output_directory
    destination.mkdir(parents=True,exist_ok=True)
    heads=flopy.utils.HeadFile(str(paths[0]),precision='double').get_data()
    grid=MfGrdFile(str(paths[2]),verbose=False)
    ia,ja=grid.ia,grid.ja
    budget=flopy.utils.CellBudgetFile(str(paths[1]),precision='double')
    flowja=budget.get_data(text='FLOW-JA-FACE')[-1].ravel()
    n=heads.size
    # FLOW-JA-FACE row n contains flow INTO n from m; diagonal is residual.
    net_internal=np.zeros(n)
    qx=np.zeros((case.shape[0],case.shape[1],case.shape[2]-1))
    qy=np.zeros((case.shape[0],case.shape[1]-1,case.shape[2]))
    qz=np.zeros((case.shape[0]-1,case.shape[1],case.shape[2]))
    reciprocity=0.
    pairs={}
    for node in range(n):
        for pos in range(ia[node]+1,ia[node+1]):
            other=int(ja[pos])
            q=float(flowja[pos])
            net_internal[node]+=q
            pairs[(node,other)]=q
            if other<node:
                reciprocity=max(reciprocity,abs(q+pairs.get((other,node),0.)))
            if other<=node:
                continue
            k,j,i=np.unravel_index(node,case.shape)
            kk,jj,ii=np.unravel_index(other,case.shape)
            internal_j=case.shape[1]-1-j
            if kk==k+1:
                qz[k,internal_j,i]=-q
            elif jj==j+1:
                qy[k,internal_j-1,i]=q
            elif ii==i+1:
                qx[k,internal_j,i]=-q
            else:
                raise ValueError('unexpected DIS connectivity')
    package=np.zeros(n)
    components={}
    gross=0.
    source_allocation=np.zeros(n)
    available={text.decode().strip() for text in budget.get_unique_record_names()}
    for label in ('WEL','CHD','DRN','GHB'):
        data=budget.get_data(text=label) if label in available else []
        records=np.concatenate(data) if data else []
        entries=[]
        for record in records:
            node=int(record['node'])-1
            q=float(record['q'])
            package[node]+=q
            if label=='WEL':
                pass
            gross+=abs(q)
            entries.append(dict(node=node,inflow=q))
        components[label]=entries
    source_records=[]
    if 'WEL' in available:
        destinations={record['paknam2'].decode().strip() for record in budget.recordarray if record['text'].decode().strip()=='WEL'}
        if 'SOURCE' in destinations:source_records=budget.get_data(text='WEL',paknam2='SOURCE')
        elif 'CONTACT' not in destinations and len(destinations)==1:source_records=budget.get_data(text='WEL')
    for records in source_records:
        for record in records:
            source_allocation[int(record['node'])-1]+=float(record['q'])
    residual=(net_internal+package).reshape(case.shape)[:,::-1,:]
    fixed=case.arrays['fixed']!=0
    allocation=source_allocation.reshape(case.shape)[:,::-1,:]
    expected=np.where(fixed,0.,case.arrays['source'])
    valid=np.isfinite(heads)&(np.abs(heads)<1e20)
    cell_rms=float(np.sqrt(np.mean(residual**2)))
    relative=float(np.sum(np.abs(residual))/max(gross,np.finfo(float).tiny))
    head=heads[:,::-1,:].copy()
    sentinel=~valid[:,::-1,:]
    head[sentinel]=np.nan
    # Keep sentinels explicit; never replace them by bottom or plausible heads.
    saturation=np.full(case.shape,np.nan)
    sat_records=budget.get_data(text='DATA-SAT') if 'DATA-SAT' in available else []
    if sat_records:
        for record in sat_records[-1]:
            saturation.ravel()[int(record['node'])-1]=float(record['sat'])
        saturation=saturation[:,::-1,:]
    fields=dict(head=head,saturation=saturation,invalid_head=sentinel,residual=residual,qx=qx,qy=qy,qz=qz,
        source_realised=allocation,diagonal_residual=flowja[ia[:-1]].reshape(case.shape)[:,::-1,:])
    np.savez_compressed(destination/'extracted.npz',**fields)
    modelgrid=grid.modelgrid
    geometric_x,geometric_y,centres=modelgrid.xyzcellcenters
    geometric_z=centres[:,::-1,:]
    expected_x=(np.arange(case.shape[2])+.5)*case.physical['dx']
    expected_y=(np.arange(case.shape[1])+.5)*case.physical['dy']
    geometric_volumes=modelgrid.cell_thickness[:,::-1,:]*modelgrid.delr[None,None,:]*modelgrid.delc[None,::-1,None]
    result=dict(geometric_x_centre_max_error=float(np.max(np.abs(geometric_x[::-1,:]-expected_x[None,:]))),
        geometric_y_centre_max_error=float(np.max(np.abs(geometric_y[::-1,:]-expected_y[:,None]))),
        geometric_centre_max_error=float(np.max(np.abs(geometric_z-case.grid().elevations))),
        geometric_volume_max_error=float(np.max(np.abs(geometric_volumes-case.grid().volumes))),native_success=native_success,independent_conservation=bool(native_success and not sentinel.any()
        and (cell_rms<=1e-12 or relative<=1e-8) and np.max(np.abs(allocation-expected))<=1e-12),
        invalid_heads=int(sentinel.sum()),cell_residual_rms=cell_rms,cell_residual_max=float(np.max(np.abs(residual))),
        cell_residual_l1=float(np.sum(np.abs(residual))),gross_external_throughput=gross,
        signed_imbalance=float(residual.sum()),relative_cell_imbalance=relative,
        reciprocity_max=reciprocity,source_allocation_max_error=float(np.max(np.abs(allocation-expected))),
        packages=components,head_min=float(head.min()) if not sentinel.any() else None,
        head_max=float(head.max()) if not sentinel.any() else None,
        formulation_equivalence=False,nonlinear_cross_residual='not established; saved FLOW-JA-FACE is native final reported flux, not a fixed-head nonlinear reevaluation')
    budget.close()
    return result


def run_mf6(*, case, directory, executable, xt3d, convertible, initial=None, drains=True, outer_cap=3000, contact_mode='confined_native', contact_fluxes=None, ims_options=None, use_connection_elevations=True):
    settings=dict(executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),xt3d=xt3d,convertible=convertible,drains=drains,outer_cap=outer_cap,contact_mode=contact_mode,ims_options=ims_options,use_connection_elevations=use_connection_elevations,contact_fluxes=None if contact_fluxes is None else np.asarray(contact_fluxes).tolist(),initial_hash=None if initial is None else hashlib.sha256(initial.tobytes()).hexdigest())
    if check_cached(directory=directory,case=case,kind='mf6',settings=settings):
        return json.loads((directory/'summary.json').read_text())
    directory.mkdir(parents=True,exist_ok=False)
    request=start_record(directory=directory,case=case,kind='mf6',settings=settings)
    sim=build_mf6(case=case,directory=directory,executable=executable,xt3d=xt3d,
        convertible=convertible,initial=initial,drains=drains,outer_cap=outer_cap,contact_mode=contact_mode,contact_fluxes=contact_fluxes,ims_options=ims_options,use_connection_elevations=use_connection_elevations)
    start=time.perf_counter()
    try:
        completed=subprocess.run([str(executable)],cwd=directory,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
            text=True,timeout=1800,check=False)
        (directory/'stdout.log').write_text(completed.stdout)
        success=completed.returncode==0 and 'Normal termination of simulation' in completed.stdout
        result=extract_mf6(case=case,directory=directory,native_success=success)
        result.update(returncode=completed.returncode)
    except subprocess.TimeoutExpired as error:
        (directory/'stdout.log').write_bytes(error.stdout or b'')
        result=dict(native_success=False,independent_conservation=False,stop_reason='wall_timeout')
    if (directory/'extracted.npz').exists():
        np.savez_compressed(directory/'head.npz',head=np.load(directory/'extracted.npz')['head'])
    result.update(run_hash=request['run_hash'],seconds=time.perf_counter()-start,input_hash=case.digest(),
        executable=str(executable),executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        mf6_version=subprocess.check_output([str(executable),'-v'],text=True).strip())
    atomic_json(path=directory/'summary.json',value=result)
    finish_record(directory=directory)
    print(case.case_id,'MF6',xt3d,result.get('native_success'),result.get('independent_conservation'),flush=True)
    return result
