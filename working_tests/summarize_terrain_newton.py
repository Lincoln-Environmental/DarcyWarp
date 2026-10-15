"""Summarize validated benchmark artifacts and generate convergence figures."""
import gzip
import json
from pathlib import Path

import matplotlib
import numpy as np


def summarize(*, directory, output, repeats, figsize, dpi):
    rows=[]
    for path in sorted(directory.glob('*.json')):
        if path.stem.startswith('warmup'):
            continue
        data=json.loads(path.read_text())
        info=data['info']
        rows.append(dict(run=path.stem,size=data['size'],basement_k=data['basement_k'],method=data['method'],
            converged=info['converged'],stop_reason=info['device_stop_reason'],outers=info['n_outer_iterations'],
            krylov=info['krylov_iterations'],jv=info['jacobian_vector_products'],
            device_residual_evaluations=info['nonlinear_residual_evaluations'],
            candidates=info['candidate_evaluations'],backtracks=info['nonlinear_backtracks'],
            fallback_outers=info['fallback_count'],newton_attempts=info['newton_attempts'],
            accepted_newton_steps=info['newton_accepted_steps'],
            head_rms=info['budget']['head_residual_rms'],flow_rms=info['budget']['cell_residual_rms'],
            relative=info['budget']['relative_cell_imbalance'],
            wall_seconds=data['solve_wall_seconds'],outer_loop_seconds=info['seconds'],
            outside_loop_seconds=data['outside_outer_loop_seconds'],build_seconds=data['build_seconds'],
            other_gpu_processes=data['other_compute_processes_before']+data['other_compute_processes_after']))
    timings={}
    for method in ('picard','newton'):
        samples=[row['wall_seconds'] for row in rows if row['run'].startswith('low_k_'+method+'_')]
        timings[method]=dict(samples=samples,median=float(np.median(a=samples)),
            minimum=min(samples),maximum=max(samples),all_converged=all(row['converged'] for row in rows if row['run'].startswith('low_k_'+method+'_')))
        if len(samples)!=repeats:
            raise AssertionError((method,samples))
    report=dict(rows=rows,low_k_timing=timings,speedup=timings['picard']['median']/timings['newton']['median'])
    (output/'benchmark_summary.json').write_text(data=json.dumps(report,indent=2))
    from matplotlib import pyplot as plt
    figure,axes=plt.subplots(nrows=2,ncols=2,figsize=figsize,layout='constrained')
    for method in ('picard','newton'):
        rows=json.loads(gzip.decompress(data=(directory/f'low_k_{method}_0.history.json.gz').read_bytes()))
        n=np.arange(len(rows))+1
        axes[0,0].semilogy(n,[row['physical_residual_rms'] for row in rows],label=method)
        axes[0,1].semilogy(n,[row['relative_cell_imbalance'] for row in rows],label=method)
        axes[1,0].semilogy(n,[row['relaxation'] for row in rows],label=method)
        axes[1,1].plot(n,np.cumsum([row['krylov_iterations'] for row in rows]),label=method)
    axes[0,0].set_ylabel(ylabel='Physical residual RMS [m3/day]')
    axes[0,1].set_ylabel(ylabel='Cell residual L1 / gross throughput')
    axes[1,0].set_ylabel(ylabel='Accepted relaxation')
    axes[1,1].set_ylabel(ylabel='Cumulative Krylov iterations')
    for axis in axes.ravel():
        axis.set_xlabel(xlabel='Outer iteration')
        axis.legend()
    figure.savefig(fname=output/'convergence.png',dpi=dpi)
    plt.close(fig=figure)


if __name__ == "__main__":
    matplotlib.use(backend='Agg')
    directory=Path('audit_outputs/validated')
    output=Path('audit_outputs')
    repeats=3
    figsize=(12,8)
    dpi=160
    summarize(directory=directory,output=output,repeats=repeats,figsize=figsize,dpi=dpi)
