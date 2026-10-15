"""Render evidence tables into the scientific assessment; no model execution."""
from pathlib import Path
import json


def number(*,value):
    return '—' if value is None else f'{value:.6g}'


def convergence_table(*,root,cases):
    rows=['| n | Basement Kx | GPU Picard (cap 2000 / reference) | GPU Newton/Picard | CPU Newton/direct-Picard | CPU direct Picard | MF6 TPFA/custom contact | MF6 XT3D/custom contact |',
        '|---:|---:|---|---|---|---|---|---|']
    for size,k in cases:
        case=f'terrain_{size}_k{k:g}'
        result=[]
        for label in ('picard','newton'):
            d=json.loads((root/'gpu_before'/case/label/'summary.json').read_text())['info']
            text=('pass ' if d['converged'] else 'FAIL ')+str(d['n_outer_iterations'])
            p=root/'gpu_before'/case/'picard_reference'/'summary.json'
            if label=='picard' and p.exists():text+=' / pass '+str(json.loads(p.read_text())['info']['n_outer_iterations'])
            result.append(text)
        for label in ('newton','picard'):
            p=root/'cpu_final'/case/label/'summary.json'
            if p.exists():
                d=json.loads(p.read_text());result.append(('pass ' if d['converged'] else 'FAIL ')+str(d['outers']))
            else:result.append('not run')
        for group in ('mf6_feedback_v3','mf6_feedback_xt3d'):
            d=json.loads((root/group/case/'summary.json').read_text())
            result.append(('native+balance pass' if d['converged'] else 'FAIL native')+f" ({len(d['history'])} solves)")
        rows.append('| '+str(size)+' | '+f'{k:g}'+' | '+' | '.join(result)+' |')
    return '\n'.join(rows)


def branch_table(*,root):
    evidence=json.loads((root/'root_evidence.json').read_text())
    rows=['| Case | Independently resolved numerical clusters | Largest separation from GPU Picard, m |',
        '|---|---:|---:|']
    for case,row in evidence.items():
        differences=[p['metrics']['head_max'] for name,p in row['pairs'].items() if name.startswith('GPU Picard')]
        rows.append(f"| {case} | {row['distinct_numerical_clusters']} | {max(differences):.9g} |")
    return '\n'.join(rows)


def physical_table(*,root):
    evidence=json.loads((root/'root_evidence.json').read_text())
    selections=[('terrain_32_k0.0001','GPU Picard versus GPU Newton/Picard'),
        ('terrain_64_k0.0001','GPU Picard versus CPU Newton/direct-Picard'),
        ('terrain_64_k0.0001','GPU Picard versus CPU direct Picard'),
        ('terrain_64_k0.1','GPU Picard versus GPU Newton/Picard'),
        ('terrain_64_k0.1','GPU Picard versus CPU Newton/direct-Picard'),
        ('terrain_64_k0.1','GPU Picard versus CPU direct Picard')]
    rows=['| Case and candidate vs GPU Picard | Max head Δ m | Wet-both head RMS Δ m | Max water-table diagnostic Δ m | Interlayer face-flow difference L1 m³/day |',
        '|---|---:|---:|---:|---:|']
    for case,key in selections:
        p=evidence[case]['pairs'][key]['metrics']
        rows.append(f"| {case}: {key.split(' versus ')[1]} | {p['head_max']:.6g} | {p['regimes']['wet_both']['rms']:.6g} | {number(value=p['water_table']['max_abs'])} | {p['qz']['l1']:.6g} |")
    return '\n'.join(rows)


def mf6_table(*,root,cases):
    rows=['| Case | Closest XT3D native/physical balance | Paired-cell max head Δ vs GPU Picard m | Aquifer→weathered net-transfer Δ m³/day | Weathered→basement net-transfer Δ m³/day |',
        '|---|---|---:|---:|---:|']
    for size,k in cases:
        case=f'terrain_{size}_k{k:g}'
        meta=json.loads((root/'mf6_feedback_xt3d'/case/'summary.json').read_text())
        c=json.loads((root/'analysis'/case/'comparisons.json').read_text())['mf6_xt3d_coupled']
        if meta['converged']:
            q=c['interlayer_transfer'];rows.append(f"| {case} | pass / pass | {c['head_max']:.6g} | {q[1]['net_difference']:.6g} | {q[2]['net_difference']:.6g} |")
        else:rows.append(f'| {case} | FAIL / FAIL | not a converged comparison | — | — |')
    return '\n'.join(rows)


def write_report(*,root,path,cases,template_path):
    template=template_path.read_text()
    sections=dict(CONVERGENCE=convergence_table(root=root,cases=cases),BRANCHES=branch_table(root=root),
        PHYSICAL=physical_table(root=root),MF6=mf6_table(root=root,cases=cases))
    for key,value in sections.items():template=template.replace('{{'+key+'}}',value)
    path.write_text(template)


if __name__ == "__main__":
    root=Path('validation_artifacts/terrain_2026-10-16')
    output=Path('docs/TERRAIN_CPU_MF6_VALIDATION_2026-10-16.md')
    template_path=Path('docs/TERRAIN_CPU_MF6_ASSESSMENT_TEMPLATE.txt')
    cases=[(32,1e-4),(64,1e-4),(64,.1),(64,.01),(64,.001),(16,1e-4),(96,1e-4)]
    write_report(root=root,path=output,cases=cases,template_path=template_path)
