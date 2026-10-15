"""Record actual runtime, source evidence and reproducible numerical settings."""
from pathlib import Path
import contextlib
import hashlib
import importlib.metadata
import io
import os
import platform
import subprocess
import sys
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def command_output(*,args):
    result=subprocess.run(args,capture_output=True,text=True,check=False)
    return dict(command=args,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)


def record_runtime(*,output,executable,evidence_paths):
    buffer=io.StringIO()
    with contextlib.redirect_stdout(buffer):np.show_config()
    metadata=dict(python=sys.version,interpreter=sys.executable,platform=platform.platform(),
        packages={key:importlib.metadata.version(key) for key in ('numpy','scipy','warp-lang','flopy','pytest')},
        blas_configuration=buffer.getvalue(),mf6_executable=str(executable.resolve()),
        mf6_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        mf6_version=command_output(args=[str(executable),'-v']),
        mf6_build_origin='pre-existing distributed executable in /bin/modflow; not built in this campaign; local build revision unavailable',
        official_source_tag='6.6.2',official_source_tag_commit='7cb7958 (release tag, not verified binary build revision)',
        revision=command_output(args=['git','rev-parse','HEAD']),branch=command_output(args=['git','branch','--show-current']),
        original_dev='377ada20a92ad5f5a40af1ed9844cd8be31beba4',
        cpu=command_output(args=['lscpu']),gpu=command_output(args=['nvidia-smi','-q']),
        os=command_output(args=['uname','-a']),environment={key:os.environ.get(key) for key in ('DARCY_FLOAT','OPENBLAS_NUM_THREADS','PYTHONPATH')},
        evidence={str(path):hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None for path in evidence_paths})
    atomic_json(path=output,value=metadata)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    executable=Path('/bin/modflow/mf6')
    evidence_paths=[Path(path) for path in ('docs/3D_PHYSICS_CORRECTION_2026-10-12.md',
        'docs/3D_SCIENTIFIC_AUDIT_2026-10-12.md','docs/3D_TERRAIN_SCIENTIFIC_ACCEPTANCE_2026-10-12.md',
        'docs/3D_VARIABLE_GEOMETRY_SCIENTIFIC_REPORT_2026-10-13.md','docs/3D_DEVICE_PICARD_AUDIT_2026-10-15.md',
        'docs/TERRAIN_NEWTON_DERIVATION_2026-10-16.md','docs/TERRAIN_NEWTON_VALIDATION_2026-10-16.md',
        'audit_outputs/benchmark_summary.json','audit_outputs/verification/solution_comparisons.json',
        'audit_outputs/polish/solution_comparisons.json','docs/terrain_picard_audit_host.json')]
    record_runtime(output=root/'runtime.json',executable=executable,evidence_paths=evidence_paths)
