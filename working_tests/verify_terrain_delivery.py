"""Verify saved evidence without running a solver or changing a returned head."""
import gzip
import json
from pathlib import Path
import subprocess

import numpy as np

from DARCY_WARP_PACKAGE.validation.evidence_storage import file_hash, verify_archive
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def compare_trajectories(*, root: Path) -> list:
    rows = []
    keys = ('relaxation', 'accepted', 'acceptance_mode')
    for after in sorted((root / 'gpu_after').glob('*/*/summary.json')):
        relative = after.parent.relative_to(root / 'gpu_after')
        before = root / 'gpu_before' / relative
        left = json.loads((before / 'summary.json').read_text())
        right = json.loads(after.read_text())
        if left['input_hash'] != right['input_hash']:
            raise ValueError('before/after input mismatch')
        h_before = np.load(before / 'head.npz')['head']
        h_after = np.load(after.parent / 'head.npz')['head']
        difference = float(np.max(np.abs(h_before - h_after)))
        if difference > 1e-4:
            raise ValueError(f'material before/after change: {relative}')
        history_before = json.loads(gzip.decompress((before / 'history.json.gz').read_bytes()))
        history_after = json.loads(gzip.decompress((after.parent / 'history.json.gz').read_bytes()))
        if len(history_before) != len(history_after):
            raise ValueError('before/after outer count changed')
        for old, new in zip(history_before, history_after):
            for key in keys:
                if old.get(key) != new.get(key):
                    raise ValueError(f'before/after decision changed: {relative}, {key}')
        for key in ('converged', 'device_stop_reason', 'n_outer_iterations'):
            if left['info'].get(key) != right['info'].get(key):
                raise ValueError(f'before/after status changed: {key}')
        work_changes = [dict(outer_iteration=old['outer_iteration'],
            before=old['krylov_iterations'], after=new['krylov_iterations'])
            for old, new in zip(history_before, history_after)
            if old['krylov_iterations'] != new['krylov_iterations']]
        rows.append(dict(experiment=str(relative), input_hash=left['input_hash'],
            outer_rows=len(history_before), max_head_difference=difference,
            stopping_and_acceptance_decisions_equal=True,
            krylov_work_changes=work_changes,
            investigation_threshold_m=1e-4))
    if len(rows) != 10:
        raise ValueError('expected five original cases, two methods each')
    return rows


def create_manifest(*, root: Path, repository: Path) -> None:
    excluded = {'delivery_manifest.json', 'delivery_checks.json', 'delivery_verification.log'}
    files = {str(path.relative_to(root)): file_hash(path=path)
        for path in sorted(root.rglob('*')) if path.is_file() and path.name not in excluded}
    sources = set((repository / 'DARCY_WARP_PACKAGE' / 'validation').glob('*.py'))
    sources.update((repository / 'DARCY_WARP_PACKAGE' / 'solvers').glob('terrain*.py'))
    sources.update((repository / 'DARCY_WARP_PACKAGE' / 'physics').glob('terrain*.py'))
    sources.update((repository / 'working_tests').glob('*terrain*.py'))
    sources.update((repository / 'tests').glob('*terrain*.py'))
    sources.update((repository / 'docs').glob('TERRAIN_CPU_MF6*'))
    implementation = {str(path.relative_to(repository)): file_hash(path=path) for path in sorted(sources)}
    runs = []
    for path in sorted(root.rglob('summary.json')):
        data = json.loads(path.read_text())
        if isinstance(data, dict):
            runs.append(dict(directory=str(path.parent.relative_to(root)),
                input_hash=data.get('input_hash'), source_revision=data.get('source_revision'),
                run_hash=data.get('run_hash'), summary_sha256=file_hash(path=path)))
    atomic_json(path=root / 'delivery_manifest.json', value=dict(schema=1,
        starting_dev='377ada20a92ad5f5a40af1ed9844cd8be31beba4',
        delivery_source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip(),
        files=files, implementation_sha256=implementation, experiments=runs,
        historical_cache_policy='Pre-registry experiments are immutable evidence, not restart-cache eligible.'))


def verify_delivery(*, root: Path, repository: Path, check_archived_members: bool) -> dict:
    manifest = json.loads((root / 'delivery_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        if file_hash(path=root / name) != expected:
            raise ValueError(f'changed evidence: {name}')
    for name, expected in manifest['implementation_sha256'].items():
        if file_hash(path=repository / name) != expected:
            raise ValueError(f'changed implementation: {name}')
    archived_members = 0
    archives = json.loads((root / 'archives' / 'index.json').read_text())
    if check_archived_members:
        for record in archives:
            archived_members += verify_archive(root=root, record=record)
    result = dict(evidence_files_checked=len(manifest['files']),
        implementation_files_checked=len(manifest['implementation_sha256']),
        archives=len(archives), archived_members_checked=archived_members,
        archived_members_verified=check_archived_members,
        before_after_gpu=compare_trajectories(root=root))
    atomic_json(path=root / 'delivery_checks.json', value=result)
    return result


if __name__ == "__main__":
    root = Path('validation_artifacts/terrain_2026-10-16')
    repository = Path.cwd()
    mode = 'verify'  # 'create' only when sealing a new evidence delivery.
    check_archived_members = True
    if mode == 'create':
        create_manifest(root=root, repository=repository)
    elif mode == 'verify':
        print(json.dumps(verify_delivery(root=root, repository=repository,
            check_archived_members=check_archived_members), indent=2))
    else:
        raise ValueError(mode)
