"""Verify split evidence recovery before any source-file pruning."""
import hashlib
import io
import json
import tarfile
import pytest
from DARCY_WARP_PACKAGE.validation.evidence_storage import file_hash,verify_archive,restore_archive


def test_split_archive_restoration_and_corruption(tmp_path):
    content=b'original MODFLOW head/budget bytes\x00\xff'*100
    name='case/output.hds'
    archive=tmp_path/'test.tar.xz'
    with tarfile.open(archive,'w:xz') as bundle:
        entry=tarfile.TarInfo(name=name);entry.size=len(content)
        bundle.addfile(entry,io.BytesIO(content))
    payload=archive.read_bytes()
    middle=len(payload)//2
    parts=[]
    for number,data in enumerate((payload[:middle],payload[middle:])):
        path=tmp_path/f'part{number}'
        path.write_bytes(data)
        parts.append(dict(path=path.name,sha256=file_hash(path=path)))
    record=dict(archive=None,parts=parts,archive_sha256=hashlib.sha256(payload).hexdigest(),
        files={name:hashlib.sha256(content).hexdigest()})
    archive.unlink()  # Prove recovery uses recorded parts alone.
    assert verify_archive(root=tmp_path,record=record)==1
    restore_archive(root=tmp_path,record=record)
    assert (tmp_path/name).read_bytes()==content
    (tmp_path/'part0').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='corrupt archive part'):
        restore_archive(root=tmp_path,record=record)
