"""Direct CPU Picard on the recorded reduced discrepancy problems."""
from pathlib import Path
import numpy as np
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from validate_terrain_cpu_runs import run_cpu


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    reductions=[('n32',Path('extra_branches/terrain_32_k0.0001'),'reduced_case',['reduced_a','reduced_b']),
        ('n64low',Path('cpu_branch_reductions/terrain_64_k0.0001/radius_8'),'case',['gpu','cpu']),
        ('n64ordinary',Path('cpu_branch_reductions/terrain_64_k0.1/radius_3'),'case',['gpu','cpu']),
        ('n64k01',Path('branches/terrain_64_k0.01'),'reduced_case',['reduced_a','reduced_b']),
        ('n64k001',Path('cpu_branch_reductions/terrain_64_k0.001/radius_3'),'case',['gpu','cpu'])]
    controls=CPUControls(method='picard',max_outer=2000,relaxation=.7,stagnation_window=50,stagnation_ratio=.995)
    seed=822
    amplitude=1e-4
    rng=np.random.default_rng(seed=seed)
    for label,parent,case_directory,branches in reductions:
        case=TerrainCase.load(directory=root/parent/case_directory)
        for branch in branches:
            initial=np.load(root/parent/branch/'head.npz')['head']
            perturbation=rng.normal(size=initial.shape)*amplitude
            perturbation[case.arrays['fixed']!=0]=0.
            run_cpu(case=case,directory=root/'reduced_picard'/label/branch,controls=controls,initial=initial+perturbation)
