"""Collect measured evidence into a self-contained, hash-verified Git baseline."""
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def file_sha256(*, path: Path) -> str:
    digest = hashlib.sha256()
    with path.open(mode="rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def copy_evidence(*, source: Path, target: Path, records: dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src=source, dst=target)
    checksum = file_sha256(path=source)
    if file_sha256(path=target) != checksum:
        raise AssertionError(f"Copy mismatch: {source}")
    records[str(target)] = {"source": str(source), "sha256": checksum}


def copy_directory(*, source: Path, target: Path, records: dict) -> None:
    for path in sorted(source.rglob(pattern="*")):
        if path.is_file():
            copy_evidence(source=path, target=target / path.relative_to(source), records=records)


def archive_native(*, source: Path, target: Path, excluded_names: tuple, records: dict) -> None:
    members = {}
    with target.open(mode="wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|") as archive:
                for path in sorted(source.iterdir()):
                    if path.is_file() and path.name not in excluded_names:
                        archive.add(name=path, arcname=path.name, recursive=False)
                        members[path.name] = {"bytes": path.stat().st_size,
                                              "sha256": file_sha256(path=path)}
    with tarfile.open(name=target, mode="r:gz") as archive:
        for member in archive:
            stream = archive.extractfile(member=member)
            if stream is None:
                raise AssertionError(f"Missing native member: {member.name}")
            digest = hashlib.sha256()
            with stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
            if digest.hexdigest() != members[member.name]["sha256"]:
                raise AssertionError(f"Native archive mismatch: {member.name}")
    records[str(target)] = {"source": str(source), "members": members}


def write_manifest(*, output: Path) -> None:
    manifest_path = output / "SHA256SUMS.json"
    manifest = {str(path.relative_to(output)): {"bytes": path.stat().st_size, "sha256": file_sha256(path=path)}
                for path in sorted(output.rglob(pattern="*")) if path.is_file() and path != manifest_path}
    manifest_path.write_text(data=json.dumps(obj=manifest, indent=2) + "\n")


def build_archive(*, evidence: Path, output: Path, variants: tuple,
                  report_files: tuple, source_directories: dict,
                  excluded_native_names: tuple, documentation: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for variant in variants:
        variant_root = output / variant
        variant_root.mkdir()
        copy_evidence(source=evidence / "run_001" / variant / "inputs.npz",
                      target=variant_root / "inputs.npz", records=records)
        copy_directory(source=evidence / "run_001" / variant / "arrays",
                       target=variant_root / "inputs", records=records)
        copy_directory(source=evidence / "analysis_003" / "reference_arrays" / variant,
                       target=variant_root / "mf6", records=records)
        archive_native(source=evidence / "final" / "native" / variant,
                       target=variant_root / "native.tar.gz",
                       excluded_names=excluded_native_names, records=records)
    for name in report_files:
        copy_evidence(source=evidence / "final" / name,
                      target=output / "benchmark" / name, records=records)
    copy_evidence(source=evidence / "run_001" / "frozen_settings.json",
                  target=output / "benchmark" / "frozen_settings.json", records=records)
    copy_evidence(source=documentation, target=output / "ARRAY_CONVENTIONS.rst", records=records)
    for label, directory in source_directories.items():
        copy_directory(source=directory, target=output / "source_snapshot" / label, records=records)
    linear = evidence / "run_001" / "linear_A0"
    for path in sorted(linear.rglob(pattern="*")):
        if path.is_file() and "arrays" not in path.relative_to(linear).parts:
            copy_evidence(source=path, target=output / "linear_A0" / path.relative_to(linear), records=records)
    report_path = output / "benchmark" / "REPORT.rst"
    report = report_path.read_text()
    report = report.replace("Exact input NPZs and direct input arrays remain in ../run_001/<variant>.\n"
                            "Decoded native arrays remain in ../analysis_003/reference_arrays/<variant>.",
                            "Exact inputs are in ../<variant>/inputs.npz and ../<variant>/inputs/.\n"
                            "Decoded native arrays are in ../<variant>/mf6/.")
    report = report.replace("HEAD/CBC/GRB/decks/iteration CSVs are in native/confined and native/unconfined.",
                            "HEAD/CBC/GRB/decks/iteration CSVs are in ../<variant>/native.tar.gz.")
    report_path.write_text(data=report)
    records[str(report_path)]["archive_note"] = "Only archive location references updated; measurements unchanged."
    readme = (
        "Canonical MF6 6.6.2 and DarcyWarp baseline\n"
        "========================================\n\n"
        "250 x 250 cells, five layers, 100 m cells, 150 mm/year RCH recharge.\n"
        "Both confined/ and unconfined/ contain complete inputs.npz, direct input\n"
        "arrays in inputs/, and accepted native MF6 output arrays in mf6/.\n"
        "mf6/head.npy is the MF6 reference, not a DarcyWarp solution.\n\n"
        "Array orientation, units, signs, physical model and acceptance criteria\n"
        "are documented in ARRAY_CONVENTIONS.rst. Benchmark settings, individual\n"
        "measurements, histories, comparisons and plots are in benchmark/.\n"
        "linear_A0/ preserves the separate manufactured K-cycle experiment.\n\n"
        "Each native.tar.gz preserves exact original native file contents,\n"
        "including decks, HEAD, CBC, GRB, listings, convergence CSVs and summary.\n"
        "Extract into an empty directory, for example::\n\n"
        "  mkdir confined/native\n"
        "  tar -xzf confined/native.tar.gz -C confined/native\n\n"
        "The executable is identified by version and SHA256 in the metadata;\n"
        "its local archived copy is not duplicated in Git. Redundant decoded\n"
        "result.npz and canonical_inputs.npz are omitted from native archives;\n"
        "their input and output arrays are provided directly in each variant.\n\n"
        "source_snapshot/ retains the measured source, independent of later HEAD\n"
        "changes. Original run directories and failure evidence remain unchanged\n"
        "locally. provenance.json records source paths and native member hashes.\n"
        "SHA256SUMS.json covers every archive file except itself. Verify with::\n\n"
        "  python working_tests/archive_npf662_baseline.py\n\n"
        "The default script verifies an existing archive and builds it if absent.\n"
        "All configuration is in its execution guard. No solver is rerun.\n"
    )
    (output / "README.rst").write_text(data=readme)
    provenance = {"packaging_head": subprocess.check_output(args=["git", "rev-parse", "HEAD"], text=True).strip(),
                  "measured_source": "See benchmark/complete_results.json and source_snapshot/.",
                  "files": {str(Path(path).relative_to(output)): record for path, record in records.items()}}
    (output / "provenance.json").write_text(data=json.dumps(obj=provenance, indent=2) + "\n")
    (output / ".gitattributes").write_text(
        data="* -text\nsource_snapshot/** -whitespace\nbenchmark/summary.csv whitespace=cr-at-eol\n")
    write_manifest(output=output)


def verify_archive(*, output: Path) -> None:
    manifest = json.loads(s=(output / "SHA256SUMS.json").read_text())
    total = 0
    for relative, record in manifest.items():
        path = output / relative
        if path.stat().st_size != record["bytes"] or file_sha256(path=path) != record["sha256"]:
            raise AssertionError(f"Baseline integrity check failed: {relative}")
        total += record["bytes"]
    print(f"Verified {len(manifest)} files, {total / 1024**2:.2f} MiB: {output}")


if __name__ == "__main__":
    evidence = Path("benchmark_artifacts/npf662_250x250x5_rch150_dx100_v1")
    output = Path("baselines/npf662_250x250x5_rch150_dx100_v1")
    variants = ("confined", "unconfined")
    build = not output.exists()
    report_files = ("REPORT.rst", "summary.csv", "complete_results.json", "cold_process.json",
                    "newton_residual_history.png", "kcycle_residual_history.png",
                    "kcycle_levels_time_memory.png", "work_and_time_breakdown.png",
                    "validation_0.log", "validation_1.log", "publication.json",
                    "executed_cold_process_driver.py")
    source_directories = {"primary": evidence / "run_001" / "source_snapshot",
                          "analysis": evidence / "analysis_003" / "source_snapshot"}
    excluded_native_names = ("result.npz", "canonical_inputs.npz")
    documentation = Path("docs/NPF662_250_BASELINE.rst")
    if build:
        build_archive(evidence=evidence, output=output, variants=variants,
                      report_files=report_files, source_directories=source_directories,
                      excluded_native_names=excluded_native_names, documentation=documentation)
    verify_archive(output=output)
