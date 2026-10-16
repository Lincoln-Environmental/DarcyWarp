"""Package native records and seal this local campaign using existing utilities."""
from pathlib import Path
import json

from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from DARCY_WARP_PACKAGE.validation.evidence_storage import verify_archive
from archive_sevenpoint_logs import archive_native_records
from seal_sevenpoint_evidence import seal_evidence


def package_native_logs(*, root, experiment_names, suffixes):
    records=[]
    for name in experiment_names:
        archive=root/'archives'/f'{name}.tar.xz'
        index=root/'archives'/f'{name}.json'
        if index.exists():
            record=json.loads(index.read_text())[0]
            verify_archive(root=root,record=record)
        else:
            archive_native_records(root=root,experiment_root=root/name,archive=archive,index=index,suffixes=suffixes)
            record=json.loads(index.read_text())[0]
        records.append(record)
    atomic_json(path=root/'archives/index.json',value=records)
    return sum(len(record['files']) for record in records)


if __name__ == "__main__":
    root=Path('validation_artifacts/npf662_layered_v1')
    manifest_name='delivery_manifest.json'
    starting_revision='8352de9c423471619c5914ce1d028120935b7f30'
    original_checkout=Path('/home/patrickdurney/PycharmProjects/DarcyWarp')
    initial_status=root/'starting_worktree_status.txt'
    source_directories=['DARCY_WARP_PACKAGE','working_tests','tests']
    experiment_names=('unconfined','geology','geology_4x4')
    suffixes=('.lst','.csv','.log')
    if (root/manifest_name).exists():raise FileExistsError('Immutable campaign is already sealed')
    count=package_native_logs(root=root,experiment_names=experiment_names,suffixes=suffixes)
    result=seal_evidence(root=root,manifest_name=manifest_name,starting_revision=starting_revision,
        original_checkout=original_checkout,initial_status=initial_status,source_directories=source_directories)
    print(json.dumps(dict(native_records_archived=count,files_sealed=len(result['files']),
        syntax=result['syntax'],preservation=result['preservation'],diff_check_passed=result['diff_check_passed']),indent=2))
