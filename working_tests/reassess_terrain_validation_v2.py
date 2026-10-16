"""Offline versioned reassessment; never advances a saved solution."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import io
import tarfile
import numpy as np
from DARCY_WARP_PACKAGE.validation.evidence_storage import readable_archive
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase, atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_compare import evaluate_saved_head


def reassess_candidates(*,original,output,boundary_tolerance_m):
    if (output/'reassessment_v2.json').exists():raise FileExistsError('Immutable reassessment exists; choose a fresh output directory')
    cases={}
    unavailable=[]
    for manifest in sorted(original.rglob('manifest.json')):
        if not (manifest.parent/'inputs.npz').is_file():continue
        try:
            case=TerrainCase.load(directory=manifest.parent)
            cases[case.digest()]=case
        except (ValueError,KeyError,OSError) as error:
            unavailable.append(dict(path=str(manifest.relative_to(original)),reason=str(error)))
    rows=[]
    for path in sorted(original.rglob('head.npz')):
        summary=path.parent/'summary.json'
        label=str(path.relative_to(original))
        try:
            metadata=json.loads(summary.read_text())
            digest=metadata['input_hash']
            case=cases[digest]
            head=np.load(path,allow_pickle=False)['head']
            checks,unused=evaluate_saved_head(case=case,head=head,boundary_tolerance_m=boundary_tolerance_m)
            budget=checks.get('budget',{})
            equivalent=label.startswith(('gpu_','cpu','branches/','extra_','reduced_picard/','scalar_roots/root_'))
            rows.append(dict(path=label,input_hash=digest,head_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                checks=checks,same_DarcyWarp_boundary_value_problem=bool(equivalent and checks.get('input_valid') and checks.get('boundary_condition_valid')),
                conservation=bool(budget and (budget['cell_residual_rms']<=1e-12 or budget['relative_cell_imbalance']<=1e-8)),
                reference_formulation_compatibility='DarcyWarp transformed equations' if equivalent else 'different formulation or identity unresolved',
                physical_validation='unresolved; residual/root agreement is not physical validation'))
        except (ValueError,KeyError,OSError) as error:
            unavailable.append(dict(path=label,reason=str(error)))
    for record in json.loads((original/'archives/index.json').read_text()):
        wanted={name for name in record['files'] if name.endswith('/head.npz') and not (original/name).is_file()}
        if not wanted:continue
        try:
            with readable_archive(root=original,record=record) as archive:
                with tarfile.open(archive,'r|*') as bundle:
                    for member in bundle:
                        if member.name not in wanted:continue
                        label=member.name
                        payload=bundle.extractfile(member).read()
                        try:
                            metadata=json.loads((original/Path(label).parent/'summary.json').read_text())
                            case=cases[metadata['input_hash']]
                            head=np.load(io.BytesIO(payload),allow_pickle=False)['head']
                            checks,unused=evaluate_saved_head(case=case,head=head,boundary_tolerance_m=boundary_tolerance_m)
                            budget=checks.get('budget',{})
                            rows.append(dict(path='archive:'+label,input_hash=case.digest(),head_sha256=hashlib.sha256(payload).hexdigest(),
                                checks=checks,same_DarcyWarp_boundary_value_problem=False,
                                conservation=bool(budget and (budget['cell_residual_rms']<=1e-12 or budget['relative_cell_imbalance']<=1e-8)),
                                reference_formulation_compatibility='Archived native MF6/coupling; different formulation',
                                physical_validation='unresolved; intermediate feedback state is not final convergence'))
                        except (ValueError,KeyError,OSError) as error:
                            unavailable.append(dict(path='archive:'+label,reason=str(error)))
        except (ValueError,OSError,tarfile.TarError) as error:
            unavailable.append(dict(path=record['original_directory'],reason=str(error)))
    roots=[row for row in rows if row['path'].startswith('branches/') and row['path'].split('/')[-2] in ('picard_perturb_0','newton_perturb_0')]
    if len(roots)!=14 or not all(row['same_DarcyWarp_boundary_value_problem'] and row['checks']['independent_accepted'] for row in roots):
        raise ValueError('Original seven root-pair comparisons did not pass corrected acceptance')
    result=dict(schema=2,original_bundle=str(original),boundary_tolerance_m=boundary_tolerance_m,
        original_root_pairs_preserved=True,root_candidates_checked=len(roots),rows=rows,unavailable=unavailable,
        scope='Accessible and recoverable archived head.npz candidates; original artifacts unchanged. No saved solution advanced.')
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(path=output/'reassessment_v2.json',value=result)
    return result


def record_runtime(*,output,executable):
    names=('numpy','scipy','warp-lang','flopy','pytest')
    result=dict(python=sys.version,interpreter=sys.executable,platform=platform.platform(),
        packages={name:importlib.metadata.version(name) for name in names},precision='float64',
        mf6_path=str(executable.resolve()),mf6_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        mf6_version=subprocess.check_output([str(executable),'-v'],text=True).strip(),
        starting_revision='296ecd61663613eb1ccf8de7d029f079e9047f88',
        source_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
        cpu=platform.processor(),existing_python312='3.12.14 / Warp1.11.0; FloPy absent; not used for GMRES; no environment modified')
    atomic_json(path=output/'runtime.json',value=result)


if __name__ == "__main__":
    original=Path('validation_artifacts/terrain_2026-10-16')
    output=Path('validation_artifacts/sevenpoint_v1/stage0')
    boundary_tolerance_m=1e-10
    executable=Path('/bin/modflow/mf6')
    result=reassess_candidates(original=original,output=output,boundary_tolerance_m=boundary_tolerance_m)
    record_runtime(output=output,executable=executable)
    print('Checked',len(result['rows']),'candidates; unavailable',len(result['unavailable']),'; original root pairs retained.')
