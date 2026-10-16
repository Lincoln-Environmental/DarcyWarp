"""Restartable fresh campaign. Immutable run keys include source/runtime/inputs.

Delivered historical experiments remain immutable evidence. This entry point
creates new runs and refuses stale or interrupted outputs; choose a fresh root
when the implementation/runtime/settings change. All controls are below.
"""
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import canonical_case,TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls
from DARCY_WARP_PACKAGE.validation.terrain_mf6 import run_mf6
from validate_terrain_cpu_runs import run_cpu
from validate_terrain_gpu_baselines import run_gpu
from validate_terrain_mf6_feedback import run_feedback


def run_native_contact_experiments(*,case,directory,executable,outer_cap):
    results={}
    for mode in ('original_convertible_drn','confined_contact_control'):
        results[mode]=run_mf6(case=case,directory=directory/mode,executable=executable,
            xt3d=True,convertible=True,outer_cap=outer_cap,contact_mode=mode)
    return results


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_reproduction_v2')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    dx=20.
    executable=Path('/bin/modflow/mf6')
    cpu_controls=CPUControls(max_outer=6000)
    cpu_extended_controls=CPUControls(max_outer=2000,stagnation_window=100)
    gpu_cap=2000
    reference_cap=8000
    mf6_cap=3000
    feedback_cap=30
    feedback_tolerance=1e-8
    feedback_relaxation=.5
    for size,k in cases:
        path=root/'cases'/f'terrain_{size}_k{k:g}'
        if not path.exists():
            case=canonical_case(size=size,basement_k=k,dx=dx)
            case.save(directory=path)
        case=TerrainCase.load(directory=path)
        run_cpu(case=case,directory=root/'cpu'/case.case_id/'newton',
            controls=cpu_extended_controls if size==96 else cpu_controls)
        if (size,k) in cases[:3]:
            run_cpu(case=case,directory=root/'cpu'/case.case_id/'picard',controls=CPUControls(method='picard',max_outer=6000,relaxation=.7))
        for method in ('picard','newton'):
            head,result=run_gpu(case=case,directory=root/'gpu'/case.case_id/method,method=method,outer_cap=gpu_cap)
            if method=='picard' and not result['info']['converged']:
                run_gpu(case=case,directory=root/'gpu'/case.case_id/'picard_reference',method=method,outer_cap=reference_cap)
        run_native_contact_experiments(case=case,directory=root/'mf6'/case.case_id,
            executable=executable,outer_cap=mf6_cap)

        for xt3d in (False,True):
            run_feedback(case=case,directory=root/'coupled'/case.case_id/('xt3d' if xt3d else 'tpfa'),executable=executable,max_feedback=feedback_cap,contact_tolerance=feedback_tolerance,relaxation=feedback_relaxation,xt3d=xt3d)
