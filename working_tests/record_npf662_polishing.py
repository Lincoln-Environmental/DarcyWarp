"""Read and quantify existing polished evidence; perform no solve or polishing."""
import hashlib
import json
from pathlib import Path

import numpy as np

from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case, NPF662Reference


def head_record(*, head, native, free, archive, key, reference):
    difference = head[free] - native[free]
    checks, unused = reference.evaluate(head=head)
    magnitude = float(np.max(np.abs(native[free])))
    return dict(path=str(archive), npz_key=key, shape=list(head.shape), dtype=str(head.dtype),
        array_sha256=hashlib.sha256(head.tobytes(order='C')).hexdigest(),
        max_head_discrepancy_m=float(np.max(np.abs(difference))),
        rms_head_discrepancy_m=float(np.sqrt(np.mean(difference**2))),
        reference_head_scale_m=magnitude, float64_epsilon=float(np.finfo(np.float64).eps),
        epsilon_times_head_scale_m=float(np.finfo(np.float64).eps*magnitude),
        spacing_at_reference_head_scale_m=float(np.spacing(magnitude)),
        physical_rms_m3_day=checks['physical_rms'], head_residual_rms_m=checks['head_residual_rms'],
        relative_cell_imbalance=checks['relative_cell_imbalance'],
        production_acceptance=checks['independent_accepted'])


def record_evidence(*, inputs, native_archive, polished_archive, original_summary, output):
    case = load_npf662_case(path=inputs)
    reference = NPF662Reference(case=case)
    with np.load(file=native_archive, allow_pickle=False) as data:native = data['head'].copy()
    with np.load(file=polished_archive, allow_pickle=False) as data:
        original = data['original_cuda'].copy()
        polished = data['head'].copy()
    recorded = json.loads(s=original_summary.read_text())
    results = dict(scope='Existing 192-cell layered geological parity case; not the 250x250x5 performance case.',
        inputs=str(inputs), before=head_record(head=original, native=native, free=reference.free,
            archive=polished_archive, key='original_cuda', reference=reference),
        after=head_record(head=polished, native=native, free=reference.free,
            archive=polished_archive, key='head', reference=reference),
        original_polishing_history=recorded['polishing']['history'],
        original_verification_checks=recorded['polishing']['checks'],
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                       for path in (inputs, native_archive, polished_archive, original_summary)},
        note='Saved arrays only: no native solve, numerical solve, repeated polishing, or changed production tolerances.')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open(mode='x') as stream:
        json.dump(obj=results, fp=stream, indent=2)
        stream.write('\n')
    print(json.dumps(obj={key: results[key] for key in ('scope', 'before', 'after')}, indent=2))


if __name__ == "__main__":
    inputs = Path('validation_artifacts/npf662_layered_v1/geology/varying_geology/inputs.npz')
    native_archive = inputs.parent / 'mf6' / 'result.npz'
    evidence = Path('validation_artifacts/npf662_layered_v1/final_assessment/geology_failure')
    polished_archive = evidence / 'verification_polished.npz'
    original_summary = evidence / 'summary.json'
    output = Path('benchmark_results/npf662_polishing_evidence.json')
    record_evidence(inputs=inputs, native_archive=native_archive, polished_archive=polished_archive,
                    original_summary=original_summary, output=output)
