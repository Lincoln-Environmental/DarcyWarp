# SPDX-License-Identifier: AGPL-3.0-or-later
"""Measure independent-operator construction and safe cache lookup separately.

This is a CPU setup microbenchmark, not a GPU or complete-solve speed claim.
Keep other CPU validation work idle when collecting timings for publication.
"""
from pathlib import Path
from time import perf_counter
import hashlib
import json
import statistics
from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case, NPF662Reference
from DARCY_WARP_PACKAGE.solvers.npf662_3d import independent_reference, case_from_model
from validate_npf662_confined import make_gpu_model


def measure_reference_cache(*, input_path, output, repeats, device):
    case = load_npf662_case(path=input_path)
    uncached = []
    cached = []
    with make_gpu_model(case=case, device=device) as model:
        current = case_from_model(model=model, top=0., cell_type=case.cell_types)
        for unused in range(repeats):
            started = perf_counter()
            reference = NPF662Reference(case=current)
            uncached.append(perf_counter()-started)
        first, unused = independent_reference(model=model, case=current)
        for unused in range(repeats):
            started = perf_counter()
            reused, hit = independent_reference(model=model, case=current)
            cached.append(perf_counter()-started)
            if not hit or reused is not first:
                raise AssertionError('Cache lookup unexpectedly rebuilt the independent operator')
        if (reference.matrix != first.matrix).nnz:
            raise AssertionError('Cached and freshly assembled independent matrices differ')
        result = dict(shape=list(case.active.shape), input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
            device=device, repeats=repeats, scope='Independent CPU sparse operator setup only; no solve timing',
            uncached_seconds=uncached, cached_lookup_seconds=cached,
            uncached_median_seconds=statistics.median(data=uncached),
            cached_median_seconds=statistics.median(data=cached), mathematical_matrix_identity=True,
            physical_acceptance='Not timed or bypassed; every public solve still evaluates its final physical state')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == "__main__":
    input_path = Path('baselines/npf662_250x250x5_rch150_dx100_v1/unconfined/inputs.npz')
    output = Path('validation_runs/3d_release/reference_cache_profile.json')
    repeats = 3
    device = 'cpu'
    result = measure_reference_cache(input_path=input_path, output=output, repeats=repeats, device=device)
    print(json.dumps(result))
