"""Immutable run identities and strict restart validation, independent of CUDA."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import subprocess
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def implementation_identity(*,kind):
    root=Path(__file__).resolve().parents[1]
    names=['physics/terrain_reference_3d.py','physics/terrain_grid_3d.py','physics/terrain_budget_3d.py',
        'validation/terrain_case.py']
    if kind=='cpu':names+=['validation/terrain_cpu.py']
    elif kind=='gpu':names += ['solvers/'+name for name in ('terrain_device_3d.py','terrain_device_kernels.py',
        'terrain_newton_3d.py','terrain_newton_kernels.py','terrain_fast_3d.py','terrain_gmres_3d.py','terrain_kernels_3d.py')]
    elif kind=='mf6':names+=['validation/terrain_mf6.py']
    else:raise ValueError('unknown run implementation')
    hashes={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    versions={}
    for name in ('numpy','scipy','warp-lang','flopy'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    hardware={}
    if Path('/proc/cpuinfo').exists():
        hardware['cpu_model']=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),None)
    if kind=='gpu':
        result=subprocess.run(['nvidia-smi','--query-gpu=name,uuid,driver_version','--format=csv,noheader'],capture_output=True,text=True,check=False)
        hardware['gpu']=result.stdout.strip()
        hardware['gpu_query_returncode']=result.returncode
    return dict(hardware=hardware,source_sha256=hashes,versions=versions,python=sys.version,interpreter=sys.executable,
        system=platform.platform(),precision=os.environ.get('DARCY_FLOAT'),blas_threads=os.environ.get('OPENBLAS_NUM_THREADS'))


def start_record(*,directory,case,kind,settings):
    request=dict(input_hash=case.digest(),kind=kind,settings=settings,implementation=implementation_identity(kind=kind))
    digest=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()
    request['run_hash']=digest
    atomic_json(path=directory/'request.json',value=request)
    return request


def check_cached(*,directory,case,kind,settings):
    if not directory.exists():return False
    for name in ('request.json','summary.json','complete.json'):
        if not (directory/name).is_file():raise ValueError(f'incomplete/unversioned immutable run: {directory}; select a fresh run directory')
    completion=json.loads((directory/'complete.json').read_text())
    for name,expected_hash in completion['files'].items():
        if not (directory/name).is_file() or hashlib.sha256((directory/name).read_bytes()).hexdigest()!=expected_hash:
            raise ValueError('corrupt or missing completed run output')
    old=json.loads((directory/'request.json').read_text())
    expected=dict(input_hash=case.digest(),kind=kind,settings=settings,implementation=implementation_identity(kind=kind))
    if any(old.get(key)!=value for key,value in expected.items()):
        raise ValueError(f'stale run inputs/settings/runtime/source: {directory}; select a fresh run directory')
    result=json.loads((directory/'summary.json').read_text())
    if result.get('input_hash')!=case.digest():raise ValueError('result input hash mismatch')
    if kind!='mf6' or result.get('native_success'):
        if not (directory/'head.npz').is_file():raise ValueError('missing immutable head output')
    return True


def finish_record(*,directory):
    names=['request.json','summary.json']
    for name in ('head.npz','history.json.gz','extracted.npz'):
        if (directory/name).is_file():names.append(name)
    files={name:hashlib.sha256((directory/name).read_bytes()).hexdigest() for name in names}
    atomic_json(path=directory/'complete.json',value=dict(files=files))
