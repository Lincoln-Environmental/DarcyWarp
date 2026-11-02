# SPDX-License-Identifier: AGPL-3.0-or-later
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case, NPF662Reference
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def reproduce(*, input_path, output, max_outer):
    case=load_npf662_case(path=input_path)
    head, checks, unused=NPF662Reference(case=case).solve(max_outer=max_outer)
    np.savez_compressed(file=output/'baseline_head.npz',head=head)
    atomic_json(path=output/'baseline_checks.json',value=checks)
    print(checks['independent_accepted'],checks['outers'],checks['physical_rms'])


if __name__ == "__main__":
    input_path=Path('validation_artifacts/npf662_layered_v1/geology/varying_geology/inputs.npz')
    output=Path(__file__).resolve().parent
    max_outer=200
    reproduce(input_path=input_path,output=output,max_outer=max_outer)
