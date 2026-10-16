"""Offline final evidence checks; never advance a saved nonlinear state."""
from contextlib import redirect_stdout
from pathlib import Path
import hashlib
import io
import json
import platform
import subprocess
import sys

import numpy as np

from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case, NPF662Reference
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def read_case(*, directory: Path) -> NPF662Case:
    with np.load(directory / "inputs.npz", allow_pickle=False) as saved:
        arrays = {name: saved[name] for name in NPF662Case.__dataclass_fields__}
    return NPF662Case(**arrays)


def recheck_case(*, directory: Path, supplied_head_rtol: float,
                 supplied_head_atol: float, head_tolerance_m: float) -> dict:
    case = read_case(directory=directory)
    reference = NPF662Reference(case=case)
    original = json.loads((directory / "comparison.json").read_text())
    if case.digest() != original["input_hash"]:
        raise ValueError(f"Input hash mismatch: {directory}")
    methods = {
        "scipy_cpu": directory / "scipy_cpu.npz",
        "warp_cpu": directory / "warp_cpu.npz",
        "cuda": directory / "warp_cuda_0.npz",
        "mf6": directory / "mf6/result.npz",
    }
    with np.load(methods["scipy_cpu"], allow_pickle=False) as saved:
        cpu = saved["head"].copy()
    rows = {}
    for method, path in methods.items():
        with np.load(path, allow_pickle=False) as saved:
            head = saved["head"].copy()
            checks, fields = reference.evaluate(head=head)
            free = reference.free
            error = float(np.max(np.abs(head[free] - cpu[free]))) if free.any() else None
            if not checks["independent_accepted"] or (error is not None and error > head_tolerance_m):
                raise AssertionError((directory, method, checks, error))
            row = dict(checks=checks, free_head_max_difference_from_scipy_m=error,
                       returned_head_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if method == "mf6":
                flux = saved["flux_native"]
                np.testing.assert_allclose(flux, fields["flux"], rtol=supplied_head_rtol,
                                           atol=supplied_head_atol)
                row["same_head_native_flux_max_error_m3_day"] = (
                    float(np.max(np.abs(flux - fields["flux"]))) if flux.size else None)
                native = json.loads((directory / "mf6/summary.json").read_text())
                if not native["native_success"] or not native["independent_conservation"]:
                    raise AssertionError(native)
                row["native_success"] = native["native_success"]
                row["native_independent_conservation"] = native["independent_conservation"]
            rows[method] = row
    return dict(directory=str(directory), input_hash=case.digest(), methods=rows,
                formulation_compatibility="Declared bounded NPF662 Cartesian v1 contract",
                physical_model_validation="Not established by these numerical parity checks")


def check_historical_evidence(*, original: Path) -> dict:
    manifest = json.loads((original / "delivery_manifest.json").read_text())
    for name, expected in manifest["files"].items():
        path = original / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Historical evidence changed: {path}")
    return dict(files_checked=len(manifest["files"]), all_byte_identical=True,
                original_manifest_sha256=hashlib.sha256((original / "delivery_manifest.json").read_bytes()).hexdigest())


def preserve_measured_source(*, gate: Path, output: Path) -> dict:
    """Reproduce recorded source bytes, including the subsequent CHD guard diff."""
    source = json.loads(gate.read_text())["source_sha256"]
    changes = []
    for name, expected in source.items():
        path = Path(name)
        payload = path.read_bytes()
        reconstructed = False
        whitespace_reconstructed = False
        if hashlib.sha256(payload).hexdigest() != expected:
            guard = (
                "        if np.any(active&(self.fixed!=0)&(self.cell_types!=0)&(self.values<self.bottoms)):\n"
                "            raise ValueError('Convertible CHD head below cell bottom is invalid for MF6 6.6.2; prescribed heads are not repaired')\n"
            ).encode()
            if name == "DARCY_WARP_PACKAGE/validation/npf662_reference.py" and payload.count(guard) == 1:
                payload = payload.replace(guard, b"")
                reconstructed = True
            elif name == "DARCY_WARP_PACKAGE/solvers/npf662_newton.py":
                payload = payload.rstrip(b"\n") + b"\n\n\n"
                whitespace_reconstructed = True
            else:
                raise ValueError(f"Unexplained measured-source difference: {path}")
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"Measured-source reconstruction does not match: {path}")
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        changes.append(dict(path=name, measured_sha256=expected,
                            reconstructed_by_removing_subsequent_invalid_CHD_input_guard=reconstructed,
                            reconstructed_by_restoring_measured_EOF_whitespace=whitespace_reconstructed,
                            delivered_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return dict(source_gate=str(gate), files=changes,
                interpretation="Measured source preserved byte-for-byte by recorded hash. Subsequent CHD input guard tested separately and EOF whitespace cleaned; all positive returned heads rechecked without solving.")


def record_machine(*, executable: Path) -> dict:
    capture = io.StringIO()
    with redirect_stdout(capture):
        np.show_config()
    return dict(interpreter=sys.executable, python=sys.version, os=platform.platform(),
                cpu=subprocess.check_output(["lscpu"], text=True), numpy_blas=capture.getvalue(),
                gpu=subprocess.check_output(["nvidia-smi"], text=True),
                mf6_invoked_path=str(executable), mf6_resolved_path=str(executable.resolve()),
                mf6_version=subprocess.check_output([str(executable), "-v"], text=True).strip(),
                mf6_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
                precision="float64", blas_threads=1,
                source_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())


def recheck_delivery(*, root: Path, output: Path, original: Path, executable: Path,
                     supplied_head_rtol: float, supplied_head_atol: float,
                     head_tolerance_m: float) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    parents = [root / "stage1/final_experiments", root / "stage2/final_experiments/small",
               root / "stage2/final_experiments/catchment"]
    cases = []
    for parent in parents:
        gate = json.loads((parent / "gate.json").read_text())
        if not gate["passed"]:
            raise ValueError(f"Failed parent gate: {parent}")
        for path in sorted(parent.glob("*/comparison.json")):
            cases.append(recheck_case(directory=path.parent, supplied_head_rtol=supplied_head_rtol,
                                      supplied_head_atol=supplied_head_atol, head_tolerance_m=head_tolerance_m))
    provenance = preserve_measured_source(gate=root / "stage2/final_experiments/small/gate.json",
                                         output=output / "measured_source")
    result = dict(schema=1, cases=cases, returned_heads_checked=sum(len(row["methods"]) for row in cases),
                  no_solution_advanced=True, measured_source=provenance,
                  historical_evidence=check_historical_evidence(original=original),
                  runtime=record_machine(executable=executable))
    atomic_json(path=output / "returned_head_checks.json", value=result)
    return result


if __name__ == "__main__":
    root = Path("validation_artifacts/sevenpoint_v1")
    output = root / "final_recheck_v2"
    original = Path("validation_artifacts/terrain_2026-10-16")
    executable = Path("/bin/modflow/mf6")
    supplied_head_rtol = 3e-12
    supplied_head_atol = 2e-11
    head_tolerance_m = 1e-7
    result = recheck_delivery(root=root, output=output, original=original, executable=executable,
                              supplied_head_rtol=supplied_head_rtol, supplied_head_atol=supplied_head_atol,
                              head_tolerance_m=head_tolerance_m)
    print("Independently accepted returned heads:", result["returned_heads_checked"])
    print("Historical files unchanged:", result["historical_evidence"]["files_checked"])
