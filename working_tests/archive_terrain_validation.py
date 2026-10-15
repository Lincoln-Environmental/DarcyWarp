"""Lossless per-case evidence archives; restore before reading raw MF6 files."""
from pathlib import Path
import hashlib
import json
import tarfile
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.validation.evidence_storage import verify_archive,restore_archive


def archive_evidence(*,root,groups,prune_raw):
    destination=root/'archives'
    destination.mkdir(parents=True,exist_ok=True)
    existing=json.loads((destination/"index.json").read_text()) if (destination/"index.json").exists() else []
    lookup={r["original_directory"]:r for r in existing}
    rows=[]
    raw_suffixes={'.cbc','.hds','.grb','.lst','.csv','.dis','.npf','.tdis','.nam','.ims','.ic','.wel','.chd','.drn','.ghb','.oc'}
    for group in groups:
        directory=root/group
        if not directory.exists():continue
        cases=[directory] if (directory/'adapter.json').exists() else sorted(directory.iterdir())
        for case_path in cases:
            if not case_path.is_dir():continue
            prior=lookup.get(str(case_path.relative_to(root)))
            if prior is not None:
                verify_archive(root=root,record=prior)
                rows.append(prior)
                continue
            archive=destination/(group+'__'+case_path.name+'.tar.gz')
            manifest=archive.with_suffix('.manifest.json')
            if archive.exists():
                record=json.loads(manifest.read_text())
                if hashlib.sha256(archive.read_bytes()).hexdigest()!=record['archive_sha256']:
                    raise ValueError('archive checksum mismatch')
                rows.append(record)
                continue
            files=[p for p in sorted(case_path.rglob('*')) if p.is_file()]
            hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
            temporary=archive.with_suffix('.pending')
            with tarfile.open(temporary,'w:gz',compresslevel=6) as bundle:
                bundle.add(case_path,arcname=str(case_path.relative_to(root)))
            temporary.replace(archive)
            # Verify the actual archived members, not merely the source files.
            with tarfile.open(archive,'r:gz') as bundle:
                for member in bundle.getmembers():
                    if member.isfile():
                        data=bundle.extractfile(member).read()
                        if hashlib.sha256(data).hexdigest()!=hashes[member.name]:raise ValueError('archive member corruption')
            record=dict(archive=str(archive.relative_to(root)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                original_directory=str(case_path.relative_to(root)),files=hashes,verified_members=len(files))
            atomic_json(path=manifest,value=record)
            rows.append(record)
            if prune_raw:
                for p in files:
                    if p.suffix in raw_suffixes or p.name=='stdout.log':p.unlink()
            print('verified',archive,len(files),flush=True)
    atomic_json(path=destination/'index.json',value=rows)


def restore_evidence(*,root):
    records=json.loads((root/'archives'/'index.json').read_text())
    for record in records:
        restore_archive(root=root,record=record)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    mode='archive'  # 'restore' restores all raw inputs/binaries/logs losslessly.
    prune_raw=True
    groups=['mf6','mf6_feedback','mf6_feedback_v2','mf6_feedback_v3','mf6_feedback_xt3d','mf6_initial_shift',
        'mf6_solver_recovery','geometry','physics_controls','fixed_boundary_refinement','fixed_boundary_refinement_v2','angle_default','fully_saturated_affine','fully_saturated_affine_xy','mf6_n32_angle_control','scalar_roots']
    if mode=='archive':archive_evidence(root=root,groups=groups,prune_raw=prune_raw)
    elif mode=='restore':restore_evidence(root=root)
    else:raise ValueError(mode)
