"""Lossless compaction: copy/verify first; separately opt into source pruning."""
import gzip
import hashlib
import json
import lzma
from pathlib import Path
import shutil
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.validation.evidence_storage import file_hash,verify_archive


def compact_archives(*,root,part_size):
    index=root/'archives'/'index.json'
    records=json.loads(index.read_text())
    for record in records:
        if record.get('encoding')=='tar.xz':continue
        old=root/record['archive']
        new=old.with_suffix('.xz')
        with gzip.open(old,'rb') as source,lzma.open(new,'wb',preset=9) as target:
            shutil.copyfileobj(source,target,length=1024*1024)
        updated=record.copy()
        updated.update(encoding='tar.xz',archive=str(new.relative_to(root)),archive_sha256=file_hash(path=new),
            superseded_archive=record['archive'],superseded_sha256=record['archive_sha256'])
        if new.stat().st_size>part_size:
            parts=[]
            with new.open('rb') as source:
                number=0
                while True:
                    data=source.read(part_size)
                    if not data:break
                    path=new.with_name(new.name+f'.part{number:03d}')
                    path.write_bytes(data)
                    parts.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(data).hexdigest()))
                    number+=1
            updated.update(archive=None,parts=parts,unsplit_copy=str(new.relative_to(root)))
        # Reassembly and every member are verified BEFORE changing the index.
        verify_archive(root=root,record=updated)
        record.clear();record.update(updated)
        atomic_json(path=old.with_suffix('.manifest.json'),value=updated)
        atomic_json(path=index,value=records)
        print('copied and verified',old.name,flush=True)


def prune_verified_copies(*,root):
    records=json.loads((root/'archives'/'index.json').read_text())
    for record in records:
        if record.get('encoding')!='tar.xz':raise ValueError('conversion incomplete')
        verify_archive(root=root,record=record)
        for key,checksum in [('superseded_archive','superseded_sha256'),('unsplit_copy','archive_sha256')]:
            if key in record:
                path=root/record[key]
                if path.exists():
                    if file_hash(path=path)!=record[checksum]:raise ValueError('source copy changed')
                    path.unlink()
        feedback=sorted((root/record['original_directory']).glob('feedback_*'))
        for directory in feedback[:-1]:
            for path in directory.glob('*.npz'):
                if file_hash(path=path)!=record['files'][str(path.relative_to(root))]:raise ValueError('derived output changed')
                path.unlink()
        print('retained verified recoverable archive',record['original_directory'],flush=True)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    mode='copy'  # Only after verification use 'prune'. Originals remain during copy.
    part_size=40_000_000
    if mode=='copy':compact_archives(root=root,part_size=part_size)
    elif mode=='prune':prune_verified_copies(root=root)
    else:raise ValueError(mode)
