# SPDX-License-Identifier: AGPL-3.0-or-later
"""Retain cold-reference, device and archived polishing results separately."""
from pathlib import Path
import hashlib
import json
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case, NPF662Reference
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json
from validate_npf662_confined import make_gpu_model


def reproduce(*, input_path, native_path, polished_path, output, cpu_outer_limit, devices, device_options):
    case = load_npf662_case(path=input_path)
    output.mkdir(parents=True, exist_ok=True)
    reference = NPF662Reference(case=case)
    cold, checks, unused = reference.solve(max_outer=cpu_outer_limit)
    np.savez_compressed(file=output/'cold_reference.npz', head=cold)
    result = dict(input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
                  cold_outer_limit=cpu_outer_limit, cold_reference=checks, devices={})
    with np.load(file=native_path, allow_pickle=False) as native:
        native_head = native['head'].copy()
    for device in devices:
        with make_gpu_model(case=case, device=device) as model:
            head, info = model.solve(unconfined=True, npf_cell_type=case.cell_types, **device_options)
        np.savez_compressed(file=output/('warp_'+device.replace(':', '_')+'.npz'), head=head)
        result['devices'][device] = dict(info=info,
            max_difference_from_archived_native_m=float(np.max(np.abs(head-native_head))))
    with np.load(file=polished_path, allow_pickle=False) as polished:
        result['archived_original'] = dict(max_difference_m=float(np.max(np.abs(polished['original_cuda']-native_head))),
            checks=reference.evaluate(head=polished['original_cuda'])[0])
        result['archived_polished'] = dict(max_difference_m=float(np.max(np.abs(polished['head']-native_head))),
            checks=reference.evaluate(head=polished['head'])[0])
    atomic_json(path=output/'results.json', value=result)
    return result


if __name__ == "__main__":
    input_path = Path('validation_artifacts/npf662_layered_v1/geology/varying_geology/inputs.npz')
    native_path = input_path.parent/'mf6/result.npz'
    polished_path = Path('validation_artifacts/npf662_layered_v1/final_assessment/geology_failure/verification_polished.npz')
    output = Path('release_evidence/geology')
    cpu_outer_limit = 200
    devices = ('cpu',)
    device_options = dict(max_outer=2000, max_cycles=2000, krylov_restart=24)
    result = reproduce(input_path=input_path, native_path=native_path, polished_path=polished_path,
        output=output, cpu_outer_limit=cpu_outer_limit, devices=devices, device_options=device_options)
    print(json.dumps(dict(cold_accepted=result['cold_reference']['independent_accepted'],
                         cold_outers=result['cold_reference']['outers'])))
