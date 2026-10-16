"""Verified lossless archive handling, including explicitly recorded split parts."""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import shutil
import tarfile
from tempfile import TemporaryDirectory


def file_hash(*,path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            block=stream.read(1024*1024)
            if not block:break
            digest.update(block)
    return digest.hexdigest()


@contextmanager
def readable_archive(*,root,record):
    if record.get('parts'):
        with TemporaryDirectory(dir=root) as temporary:
            combined=Path(temporary)/'evidence.tar.xz'
            with combined.open('wb') as target:
                for part in record['parts']:
                    source=root/part['path']
                    if file_hash(path=source)!=part['sha256']:raise ValueError('corrupt archive part')
                    with source.open('rb') as stream:shutil.copyfileobj(stream,target)
            if file_hash(path=combined)!=record['archive_sha256']:raise ValueError('corrupt assembled archive')
            yield combined
    else:
        path=root/record['archive']
        if file_hash(path=path)!=record['archive_sha256']:raise ValueError('corrupt archive')
        yield path


def verify_archive(*,root,record):
    seen=set()
    with readable_archive(root=root,record=record) as path:
        with tarfile.open(path,'r:*') as bundle:
            for member in bundle.getmembers():
                if member.issym() or member.islnk():raise ValueError('unsupported archive link')
                if not member.isfile():continue
                if member.name not in record['files']:raise ValueError('unexpected archive member')
                digest=hashlib.sha256()
                with bundle.extractfile(member) as stream:
                    while True:
                        block=stream.read(1024*1024)
                        if not block:break
                        digest.update(block)
                if digest.hexdigest()!=record['files'][member.name]:raise ValueError('corrupt archive member')
                seen.add(member.name)
    if seen!=set(record['files']):raise ValueError('missing archive member')
    return len(seen)


def restore_archive(*,root,record):
    verify_archive(root=root,record=record)
    with readable_archive(root=root,record=record) as path:
        with tarfile.open(path,'r:*') as bundle:bundle.extractall(path=root,filter='data')
    for name,expected in record['files'].items():
        if file_hash(path=root/name)!=expected:raise ValueError('corrupt restored file')
