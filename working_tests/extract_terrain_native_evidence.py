"""Read archived native binaries without executing MF6 or updating heads."""
from pathlib import Path
import json
import tarfile
from tempfile import TemporaryDirectory
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase,atomic_json
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import extract_mf6
from DARCY_WARP_PACKAGE.validation.evidence_storage import readable_archive


def extract_archived(*,root,case,model_directory,native_success,output,records):
    required=['terrain.hds','terrain.cbc','terrain.dis.grb']
    if all((root/model_directory/name).is_file() for name in required):
        return extract_mf6(case=case,directory=root/model_directory,native_success=native_success,output_directory=output)
    record=next(r for r in records if str(model_directory/'terrain.cbc') in r['files'])
    with TemporaryDirectory(dir=root) as temporary:
        directory=Path(temporary)
        (directory/'adapter.json').write_bytes((root/model_directory/'adapter.json').read_bytes())
        with readable_archive(root=root,record=record) as path:
            with tarfile.open(path,'r|*') as bundle:
                for member in bundle:
                    if member.isfile() and member.name in [str(model_directory/name) for name in required]:
                        (directory/Path(member.name).name).write_bytes(bundle.extractfile(member).read())
        return extract_mf6(case=case,directory=directory,native_success=native_success,output_directory=output)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    records=json.loads((root/'archives'/'index.json').read_text())
    for size,k in cases:
        case=TerrainCase.load(directory=root/'cases'/f'terrain_{size}_k{k:g}')
        models=[('native',Path('mf6')/case.case_id/'native_xt3d_precise')]
        for group,label in [('mf6_feedback_v3','coupled_tpfa'),('mf6_feedback_xt3d','coupled_xt3d')]:
            last=sorted((root/group/case.case_id).glob('feedback_*/summary.json'))[-1].parent.relative_to(root)
            models.append((label,last))
        for label,directory in models:
            metadata=json.loads((root/directory/'summary.json').read_text())
            output=root/'native_final_fields'/case.case_id/label
            result=extract_archived(root=root,case=case,model_directory=directory,native_success=metadata['native_success'],output=output,records=records)
            result.update(input_hash=case.digest(),original_model_directory=str(directory))
            atomic_json(path=output/'summary.json',value=result)
