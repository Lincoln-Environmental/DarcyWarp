"""Immutable acceptance examples and offline V1 reassessment, no head updates."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys

import numpy as np

from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case, NPF662Reference
from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from recheck_sevenpoint_delivery import recheck_case


def record_examples(*, output: Path, conductivities: tuple[float, ...]) -> None:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for index, conductivity in enumerate(conductivities):
        shape = (1, 1, 3)
        one = np.ones(shape)
        values = np.array([0., 1., 2.]).reshape(shape)
        head = values.copy(); head[0, 0, 1] += 1e-9 if conductivity == 1. else .001
        case = NPF662Case(kx=conductivity*one, ky=conductivity*one, kz=conductivity*one,
            active=one.astype(int), fixed=np.array([1, 0, 1]).reshape(shape), values=values,
            source=np.zeros(shape), initial=values.copy(), dx=1., dy=1., dz=1.)
        with np.errstate(over='ignore', invalid='ignore'):
            checks, fields = NPF662Reference(case=case).evaluate(head=head)
        raw = {name: checks[name] for name in ('head_residual_rms', 'physical_rms',
            'relative_cell_imbalance', 'gross_throughput', 'independent_accepted')}
        np.savez_compressed(output / f'example_{index}.npz', head=head, **fields, **raw)
        rows.append(dict(conductivity=conductivity, checks=checks))
    atomic_json(path=output / 'examples.json', value=finite_diagnostics(value=rows))


def reassess_v1(*, original: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for parent in (original / 'stage1/final_experiments', original / 'stage2/final_experiments/small',
                   original / 'stage2/final_experiments/catchment'):
        for comparison in sorted(parent.glob('*/comparison.json')):
            rows.append(recheck_case(directory=comparison.parent, supplied_head_rtol=3e-12,
                supplied_head_atol=2e-11, head_tolerance_m=1e-7))
    result = dict(schema=1, no_solution_advanced=True, cases=rows,
        returned_heads_checked=sum(len(row['methods']) for row in rows), changed_classifications=0)
    atomic_json(path=output / 'reassessment.json', value=result)
    return result


def record_runtime(*, output: Path, executable: Path) -> None:
    runtime = dict(python=sys.version, interpreter=sys.executable, platform=platform.platform(),
        packages={name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'warp-lang', 'flopy', 'pytest')},
        precision='float64', mf6_path=str(executable), mf6_resolved_path=str(executable.resolve()),
        mf6_version=subprocess.check_output([str(executable), '-v'], text=True).strip(),
        mf6_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        gpu=subprocess.check_output(['nvidia-smi'], text=True),
        source_revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in list(Path('DARCY_WARP_PACKAGE/validation').glob('npf662*.py'))
            + list(Path('DARCY_WARP_PACKAGE/solvers').glob('npf662*.py'))})
    atomic_json(path=output / 'runtime.json', value=runtime)


def reassess_older_fields(*, original: Path, output: Path) -> dict:
    """Include earlier attempts/controls without guessing missing provenance."""
    output.mkdir(parents=True,exist_ok=False)
    rows=[];unavailable=[];seen=set();invalid_inputs=[];missing_heads=[]
    inputs=sorted(original.rglob('inputs.npz'))+sorted(original.rglob('canonical_inputs.npz'))
    for path in inputs:
        with np.load(path,allow_pickle=False) as saved:
            if not {'kx','ky','kz','active','fixed','values','source','initial','dx','dy','dz','top'}<=set(saved.files):continue
        try:
            case=load_npf662_case(path=path)
            reference=NPF662Reference(case=case)
        except (ValueError,KeyError) as error:
            invalid_inputs.append(dict(path=str(path.relative_to(original)),reason=str(error)));continue
        candidates=sorted(path.parent.glob('*.npz'))+sorted(path.parent.glob('*/result.npz'))
        has_head=any(candidate in seen for candidate in candidates)
        for candidate in candidates:
            if candidate in seen:continue
            try:
                with np.load(candidate,allow_pickle=False) as saved:
                    if 'head' not in saved.files:continue
                    head=saved['head'].copy()
                has_head=True
                seen.add(candidate)
                checks,unused=reference.evaluate(head=head)
                previous=None;verification=False
                sidecars=(candidate.with_suffix('.json'),candidate.parent/'summary.json')
                for sidecar in sidecars:
                    if not sidecar.is_file():continue
                    metadata=json.loads(sidecar.read_text())
                    verification=verification or metadata.get('supplied_state_verification',False)
                    if 'input_hash' in metadata and metadata['input_hash']!=case.digest():
                        raise ValueError('Archived source/input hash mismatch')
                    for known in (metadata,metadata.get('independent',{}),metadata.get('independent_acceptance',{})):
                        if 'independent_accepted' in known:previous=bool(known['independent_accepted'])
                rows.append(dict(path=str(candidate.relative_to(original)),input_path=str(path.relative_to(original)),
                    input_hash=case.digest(),head_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    checks=checks,previous_independent_acceptance=previous,verification_only=verification,
                    classification_changed=bool(not verification and previous is not None and previous!=checks['independent_accepted'])))
            except (ValueError,KeyError,OSError) as error:
                unavailable.append(dict(path=str(candidate.relative_to(original)),reason=str(error)))
        if not has_head:missing_heads.append(str(path.relative_to(original)))
    result=dict(schema=1,no_solution_advanced=True,rows=rows,unavailable=unavailable,invalid_inputs=invalid_inputs,
        inputs_without_returned_field=missing_heads,
        changed_recorded_classifications=sum(row['classification_changed'] for row in rows),
        previous_classification_unrecorded=sum(row['previous_independent_acceptance'] is None for row in rows),
        scope='All accessible NPZ head fields with a readable seven-point canonical input, including earlier attempts; supplied-state constitutive fields labelled separately')
    atomic_json(path=output/'reassessment.json',value=finite_diagnostics(value=result))
    return result


if __name__ == "__main__":
    output = Path('validation_artifacts/npf662_layered_v1/part_a')
    original = Path('validation_artifacts/sevenpoint_v1')
    executable = Path('/bin/modflow/mf6')
    conductivities = (1., 1e308, 1e200, 1e-300)
    record_examples(output=output / 'after_examples', conductivities=conductivities)
    result = reassess_v1(original=original, output=output / 'reassessment')
    record_runtime(output=output, executable=executable)
    print('Saved V1 fields:', result['returned_heads_checked'], 'classification changes:', result['changed_classifications'])
