"""Lossless packaging of native iterations using the existing evidence reader."""
from pathlib import Path
import json
import tarfile

from DARCY_WARP_PACKAGE.validation.evidence_storage import file_hash, restore_archive, verify_archive
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def archive_native_records(*, root: Path, experiment_root: Path, archive: Path,
                           index: Path, suffixes: tuple[str, ...]) -> dict:
    if archive.exists() or index.exists():
        raise FileExistsError("Choose a fresh archive identity; original records are immutable")
    paths = sorted(path for path in experiment_root.rglob("*")
                   if path.is_file() and path.parent != experiment_root and path.suffix in suffixes)
    files = {str(path.relative_to(root)): file_hash(path=path) for path in paths}
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(name=archive, mode="x:xz") as bundle:
        for path in paths:
            bundle.add(name=path, arcname=str(path.relative_to(root)), recursive=False)
    record = dict(archive=str(archive.relative_to(root)), archive_sha256=file_hash(path=archive),
                  files=files, original_directory=str(experiment_root.relative_to(root)))
    verified = verify_archive(root=root, record=record)
    atomic_json(path=index, value=[record])
    # Remove only new evidence bytes already verified in a durable local archive.
    # Existing repository restore_archive restores their exact original paths.
    for path in paths:
        path.unlink()
    return dict(files=verified, archive_bytes=archive.stat().st_size)


def restore_native_records(*, root: Path, index: Path) -> None:
    for record in json.loads(index.read_text()):
        restore_archive(root=root, record=record)


if __name__ == "__main__":
    root = Path("validation_artifacts/sevenpoint_v1")
    experiment_root = root / "stage2"
    archive = root / "archives/stage2_native_iterations.tar.xz"
    index = root / "archives/index.json"
    suffixes = (".lst", ".csv", ".log")
    restore = False
    if restore:
        restore_native_records(root=root, index=index)
    else:
        result = archive_native_records(root=root, experiment_root=experiment_root,
                                       archive=archive, index=index, suffixes=suffixes)
        print(result)
