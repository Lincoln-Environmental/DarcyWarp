"""Protect the benchmark's physical oracle and optional timing hook."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.validation.npf662_reference import (
    NPF662LayeredCase, NPF662Reference, load_npf662_case, save_npf662_case,
)
from DARCY_WARP_PACKAGE.solvers_3d import solve_multigrid_kcycle_7point_3d
from benchmark_npf662_baseline import make_linear_problem, hierarchy_signature, compare_solution
from DARCY_WARP_PACKAGE.validation.npf662_matrix_free import NPF662MatrixFreeReference
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6


def simple_case():
    shape = (2, 4, 6)
    k, j, i = np.indices(dimensions=shape)
    top = 8.+.2*j[0]+.1*i[0]
    thickness = 3.+k+.1*i+.05*j
    interfaces = np.concatenate([top[None], top[None]-np.cumsum(a=thickness, axis=0)], axis=0)
    fixed = np.zeros(shape=shape, dtype=int); fixed[:, :, 0] = 1
    conductivity = 1.+.1*i+.2*j+.5*k
    return NPF662LayeredCase(kx=conductivity, ky=.7*conductivity, kz=.15*conductivity,
        active=np.ones(shape=shape, dtype=int), fixed=fixed, values=np.zeros(shape=shape),
        source=np.zeros(shape=shape), initial=np.zeros(shape=shape), dx=2., dy=3.,
        z_interfaces=interfaces, cell_type=np.zeros(shape=shape, dtype=int))


def test_saved_layered_scalars_and_manufactured_oracle(tmp_path):
    case = simple_case()
    save_npf662_case(case=case, path=tmp_path/'inputs.npz')
    loaded = load_npf662_case(path=tmp_path/'inputs.npz')
    assert isinstance(loaded.dx, float) and isinstance(loaded.dy, float)
    assert loaded.digest() == case.digest()
    manufactured, exact, faces = make_linear_problem(case=loaded,
        settings=dict(amplitude=3., x_frequency=1., y_amplitude=.2, y_frequency=2., layer_amplitude=.1))
    checks, fields = NPF662Reference(case=manufactured).evaluate(head=exact)
    assert checks['independent_accepted']
    reference = NPF662Reference(case=manufactured)
    np.testing.assert_array_equal(np.asarray(reference.incidence.sum(axis=0)), 0.)
    assert sum(np.count_nonzero(face) for face in faces) == 2*reference.left.size
    native = dict(flux_native=fields['flux'].copy())
    comparison = compare_solution(case=manufactured, head=exact, fields=fields, native_head=exact.copy(), native_fields=native)
    assert comparison['head_max_abs_m'] == 0. and comparison['flux_max_abs_m3_day'] == 0.


def test_optional_observer_preserves_original_kcycle_work():
    case = simple_case()
    manufactured, exact, faces = make_linear_problem(case=case,
        settings=dict(amplitude=3., x_frequency=1., y_amplitude=.2, y_frequency=2., layer_amplitude=.1))
    events = []

    def observe(*, event, **values):
        events.append((event, values.get('cycle')))

    options = dict(tx_p=faces[0], tx_m=faces[1], ty_p=faces[2], ty_m=faces[3], tz_p=faces[4], tz_m=faces[5],
        rhs=manufactured.source, active=case.active, bc_mask=case.fixed, bc_values=case.values,
        initial_head=case.initial, max_cycles=2, max_levels=3, nu_pre=2, nu_post=2, nu_coarse=3,
        rel_tol=0., abs_tol_min=1e-14, dh_rms_tol=None, check_every_no=1, device='cpu')
    control, control_info = solve_multigrid_kcycle_7point_3d(**options)
    measured, measured_info = solve_multigrid_kcycle_7point_3d(benchmark_observer=observe, **options)
    np.testing.assert_array_equal(measured, control)
    assert control_info['n_cycles_used'] == measured_info['n_cycles_used']
    assert control_info['r_rms_end'] == measured_info['r_rms_end']
    assert [event for event in events if event[0] == 'cycle_end'] == [('cycle_end', 1), ('cycle_end', 2)]
    assert hierarchy_signature(shape=(5, 250, 250), max_levels=8, min_coarse_n=2)[-1] == (5, 2, 2)
    assert hierarchy_signature(shape=(5, 250, 250), max_levels=8, min_coarse_n=16) == hierarchy_signature(
        shape=(5, 250, 250), max_levels=7, min_coarse_n=16)


@pytest.mark.parametrize('state', ['confined', 'convertible', 'dry_top'])
def test_matrix_free_physics_preserves_existing_acceptance(state):
    case = simple_case()
    types = np.zeros_like(a=case.active)
    if state != 'confined':
        types[0] = 1
    values = case.tops+1.
    case = replace(case, values=values, initial=values.copy(), cell_type=types)
    head = case.bottoms+.4*case.thicknesses
    if state == 'dry_top':
        head[0] = case.bottoms[0]-.2
    head[case.fixed != 0] = case.values[case.fixed != 0]
    oracle, expected = NPF662Reference(case=case).evaluate(head=head)
    reference = NPF662MatrixFreeReference(case=case)
    assert not hasattr(reference, 'matrix') and not hasattr(reference, 'incidence')
    checks, fields = reference.evaluate(head=head)
    for name in ('flux', 'residual', 'principal_diagonal', 'saturation'):
        np.testing.assert_allclose(actual=fields[name], desired=expected[name], rtol=3e-14, atol=1e-12)
    for name in ('physical_rms', 'head_residual_rms', 'relative_cell_imbalance', 'boundary_inflow', 'boundary_outflow'):
        np.testing.assert_allclose(actual=checks[name], desired=oracle[name], rtol=3e-14, atol=1e-12)
    assert checks['independent_accepted'] == oracle['independent_accepted']
    assert checks['final_acceptance_thresholds'] == oracle['final_acceptance_thresholds']


def test_native_rch_package_preserves_explicit_top_allocation(tmp_path):
    case = simple_case()
    case.source[0, :, 1:] = .01
    head, info, fields = run_reference_mf6(case=case, directory=tmp_path/'rch', executable=Path('/bin/modflow/mf6'),
        source_package='rch', head_tolerance_m=1e-10,
        native_solver_options=dict(linear_acceleration='BICGSTAB', preconditioner_levels=5,
            preconditioner_drop_tolerance=1e-4, rcloserecord=[1e-12, 'STRICT']))
    assert info['native_success'] and info['independent_conservation'], info
    assert info['independent']['independent_accepted']
    np.testing.assert_allclose(actual=fields['source_realised'], desired=case.source, rtol=0., atol=1e-14)
    nam = (tmp_path/'rch'/'npf662.nam').read_text().upper()
    assert 'RCH6' in nam and 'WEL6' not in nam
    assert info['source_package'] == 'rch'
