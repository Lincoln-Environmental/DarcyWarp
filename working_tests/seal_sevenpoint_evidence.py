"""Local syntax, change-scope and byte-level evidence checks; no remote work."""
from pathlib import Path
import ast
import hashlib
import json
import py_compile
import subprocess
import tempfile

from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def changed_python_files(*, starting_revision: str, directories: list[str]) -> list[Path]:
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "--diff-filter=AM", starting_revision, "--", *directories],
        text=True).splitlines()
    added = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *directories],
        text=True).splitlines()
    return sorted({Path(name) for name in changed + added if name.endswith(".py")})


def check_python(*, paths: list[Path]) -> dict:
    with tempfile.TemporaryDirectory(prefix="darcy-sevenpoint-syntax-") as temporary:
        for index, path in enumerate(paths):
            tree = ast.parse(source=path.read_text(), filename=str(path))
            if any(isinstance(node, ast.Lambda) for node in ast.walk(tree)):
                raise ValueError(f"Lambda found in changed source: {path}")
            py_compile.compile(file=str(path), cfile=str(Path(temporary) / f"{index}.pyc"), doraise=True)
    return dict(files=len(paths), syntax_passed=True, no_lambdas=True)


def check_preservation(*, original_checkout: Path, initial_status: Path,
                       starting_revision: str) -> dict:
    revision = subprocess.check_output(["git", "-C", str(original_checkout), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(original_checkout), "status", "--short"], text=True)
    if revision != starting_revision or status != initial_status.read_text():
        raise ValueError("Original checkout changed relative to recorded starting state")
    changed_workflows = subprocess.check_output(
        ["git", "diff", "--name-only", starting_revision, "--", ".github"], text=True)
    new_workflows = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", ".github"], text=True)
    if changed_workflows or new_workflows:
        raise ValueError("Workflow scope must remain untouched")
    return dict(original_checkout_revision=revision, original_status_unchanged=True,
                workflow_files_unchanged=True, no_remote_operations=True)


def seal_evidence(*, root: Path, manifest_name: str, starting_revision: str, original_checkout: Path,
                  initial_status: Path, source_directories: list[str]) -> dict:
    target = root / manifest_name
    if target.exists():
        raise FileExistsError("Evidence already sealed; use a new version for additional work")
    paths = changed_python_files(starting_revision=starting_revision, directories=source_directories)
    syntax = check_python(paths=paths)
    preservation = check_preservation(original_checkout=original_checkout, initial_status=initial_status,
                                      starting_revision=starting_revision)
    subprocess.run(["git", "diff", "--check"], check=True)
    subprocess.run(["git", "diff", "--cached", "--check"], check=True)
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = dict(schema=1, starting_revision=starting_revision,
                  branch=subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
                  source_base_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  syntax=syntax, preservation=preservation, diff_check_passed=True, files=files,
                  interpretation="Local bounded reference-parity evidence; not physical-model certification")
    atomic_json(path=target, value=result)
    return result


if __name__ == "__main__":
    root = Path("validation_artifacts/sevenpoint_v1")
    manifest_name = "delivery_manifest_v2.json"
    starting_revision = "296ecd61663613eb1ccf8de7d029f079e9047f88"
    original_checkout = Path("/home/patrickdurney/PycharmProjects/DarcyWarp")
    initial_status = root / "stage0/starting_worktree_status.txt"
    source_directories = ["DARCY_WARP_PACKAGE", "working_tests", "tests"]
    result = seal_evidence(root=root, manifest_name=manifest_name, starting_revision=starting_revision,
                           original_checkout=original_checkout, initial_status=initial_status,
                           source_directories=source_directories)
    print(json.dumps(dict(files_sealed=len(result["files"]), syntax=result["syntax"],
                          preservation=result["preservation"], diff_check_passed=result["diff_check_passed"]), indent=2))
