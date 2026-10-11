"""Execute the opt-in steady profile on a Purau prepared steady-array archive.

Exports an accepted head only after native, independent budget and physical
checks. A numerically converged confined-overpressure state is not a hot start.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np


def solve_and_export(*, input_archive: Path, output_archive: Path, recharge_mm_year: float,
                     days_per_year: float, cell_size_m: float, max_levels: int,
                     device: str, budget_relative: float, budget_absolute: float,
                     head_tolerance_m: float) -> dict:
    from DARCY_WARP_PACKAGE.solvers.steady_config import production_unconfined_steady_controls
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver
    from DARCY_WARP_PACKAGE.nonlinear.coastal import CoastalSubfaces2D
    if output_archive.exists():
        raise FileExistsError(output_archive)
    with np.load(input_archive, allow_pickle=False) as archive:
        fields = {name: archive[name].copy() for name in archive.files}
    active = fields['active'].astype(bool)
    top = fields['dem_m']
    bottom = fields['darcy_bottom_m']
    network = fields['drainage_mask'].astype(bool)
    conductance = fields['drain_conductance_m2_day']
    reference_t = fields['reference_transmissivity_m2_day']
    if (np.any(top[active] <= bottom[active]) or not np.isfinite(top[active]).all()
            or recharge_mm_year < 0 or days_per_year <= 0 or cell_size_m <= 0):
        raise ValueError('invalid geometry or recharge configuration')
    coast = CoastalSubfaces2D(rows=fields['coastal_rows'], columns=fields['coastal_columns'],
        head_m=fields['coastal_subface_head_m'],
        conductance_m2_per_time=fields['coastal_subface_conductance_m2_day'],
        submerged=fields['coastal_subface_submerged'])
    source = np.where(active, recharge_mm_year/1000./days_per_year, 0.)
    zero = np.zeros(top.shape)
    controls = production_unconfined_steady_controls(max_levels=max_levels)
    with WarpDarcySolver(nx=top.shape[1], ny=top.shape[0], dx=cell_size_m,
            device=device, use_drn=True, diag_preconditioner_backend='device') as model:
        model.build_from_fields(T_field=reference_t, R_field=source, active=active.astype(int),
            bc_mask=zero.astype(int), bc_values=zero, drn_mask=network.astype(int),
            drn_elev=top, drn_cond=conductance, coastal_boundary=coast)
        head, info = model.solve(initial_head=top.copy(), reference_transmissivity_field=reference_t,
            zbot_field=bottom, ztop_field=top, **controls)
    assert info['solver_backend'] == 'unconfined_semismooth_newton_kcycle'
    assert info['preconditioner_implementation'] == 'fast'
    if not info['converged'] or info['newton_fallback_used']:
        raise RuntimeError(f"Native solve failed: {info['newton_failure_reason']}")
    # Compute physical external fluxes independently of solver's budget report.
    recharge = float(source[active].sum()*cell_size_m**2)
    drain = float((conductance*network*np.maximum(head-top, 0.))[active].sum())
    delta = head[coast.rows, coast.columns]-coast.head_m
    flux = coast.conductance_m2_per_time*np.where(coast.submerged != 0, delta, np.maximum(delta, 0.))
    sea_in = float(np.maximum(-flux, 0.).sum())
    sea_out = float(np.maximum(flux, 0.).sum())
    residual = recharge+sea_in-drain-sea_out
    budget = dict(recharge_m3_day=recharge, drain_out_m3_day=drain,
        coastal_in_m3_day=sea_in, coastal_out_m3_day=sea_out, residual_m3_day=residual)
    allowance = max(budget_absolute, budget_relative*(recharge+sea_in))
    if abs(residual) > allowance:
        raise RuntimeError(f'Independent budget failed: {budget}, allowance={allowance}')
    above = active & ~network & (head > top+head_tolerance_m)
    floor = active & (head-bottom < controls['min_saturated_thickness'])
    if not np.isfinite(head[active]).all() or above.any() or floor.any():
        raise RuntimeError(f'Physical hot-start rejection: {above.sum()} non-drain heads above DEM; '
            f'{floor.sum()} floor-dependent columns. Budget={budget}')
    output_archive.parent.mkdir(parents=True, exist_ok=True)
    metadata = dict(source=info['source_identity'], effective_configuration=info['effective_configuration'],
        budget=budget, controls=controls, recharge_mm_year=recharge_mm_year,
        days_per_year=days_per_year, cell_size_m=cell_size_m)
    np.savez_compressed(output_archive, head_m=head, active=active, metadata_json=json.dumps(metadata))
    return metadata


if __name__ == "__main__":
    repository = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repository))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-archive', type=Path, required=True)
    parser.add_argument('--output-archive', type=Path, required=True)
    parser.add_argument('--recharge-mm-year', type=float, default=300.)
    parser.add_argument('--days-per-year', type=float, default=365.25)
    parser.add_argument('--cell-size-m', type=float, default=5.)
    parser.add_argument('--max-levels', type=int, default=5)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--budget-relative', type=float, default=1e-6)
    parser.add_argument('--budget-absolute-m3-day', type=float, default=1e-6)
    parser.add_argument('--head-tolerance-m', type=float, default=1e-6)
    args = parser.parse_args()
    report = solve_and_export(input_archive=args.input_archive, output_archive=args.output_archive,
        recharge_mm_year=args.recharge_mm_year, days_per_year=args.days_per_year,
        cell_size_m=args.cell_size_m, max_levels=args.max_levels, device=args.device,
        budget_relative=args.budget_relative, budget_absolute=args.budget_absolute_m3_day,
        head_tolerance_m=args.head_tolerance_m)
    print(json.dumps(report, indent=2))
