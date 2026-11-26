"""Analytical and pinned MF6 checks of multi-entry switching and buffer replacement."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from DARCY_WARP_PACKAGE.solvers.npf662_kcycle import NPF662KCycleControls


def native_switching_case(*, directory, executable, active, fixed, cell, rivers, wells):
    """Run the same confined equations with independently declared native packages.

    :param directory: Native workspace.
    :param executable: Pinned native executable.
    :param active: Groundwater activity grid.
    :param fixed: Prescribed-head mask.
    :param cell: Nonzero-layer groundwater boundary cell.
    :param rivers: Separate river entry definitions for each period.
    :param wells: Actual volumetric pumping/injection schedule.
    :return: Native accepted heads and individual inward river rates.
    """
    import flopy
    from DARCY_WARP_PACKAGE.validation.npf662_mf6 import pinned_native_executable
    binary, unused_version, unused_checksum, unused_build = pinned_native_executable(executable=executable)
    simulation = flopy.mf6.MFSimulation(sim_name='sparse_audit', sim_ws=str(directory), exe_name=str(binary))
    flopy.mf6.ModflowTdis(simulation, nper=len(wells), perioddata=[(1., 1, 1.)] * len(wells))
    model = flopy.mf6.ModflowGwf(simulation, modelname='sparse_audit', save_flows=True)
    flopy.mf6.ModflowIms(simulation, complexity='COMPLEX', linear_acceleration='BICGSTAB',
        outer_maximum=100, outer_dvclose=1.e-12, inner_maximum=100, inner_dvclose=1.e-12,
        rcloserecord=[1.e-12, 'STRICT'])
    flopy.mf6.ModflowGwfdis(model, nlay=2, nrow=3, ncol=4, delr=1., delc=1.,
        top=10., botm=[9., 8.], idomain=active)
    flopy.mf6.ModflowGwfic(model, strt=8.)
    flopy.mf6.ModflowGwfnpf(model, icelltype=0, k=1., k33=1., save_flows=True)
    fixed_cells = np.column_stack(np.nonzero(fixed))
    flopy.mf6.ModflowGwfchd(model, stress_period_data=[(tuple(int(v) for v in item), 8.) for item in fixed_cells], save_flows=True)
    flopy.mf6.ModflowGwfghb(model, stress_period_data=[(cell, 8., 1.)], save_flows=True)
    flopy.mf6.ModflowGwfrch(model, stress_period_data=[(cell, .1)], save_flows=True)
    flopy.mf6.ModflowGwfdrn(model, stress_period_data=[(cell, 7., .7)], save_flows=True)
    flopy.mf6.ModflowGwfriv(model, maxbound=max(len(period) for period in rivers),
        stress_period_data={index: [(cell, stage, conductance, bottom) for conductance, stage, bottom in period]
            for index, period in enumerate(rivers)}, save_flows=True)
    flopy.mf6.ModflowGwfwel(model, stress_period_data={index: [(cell, rate)] for index, rate in enumerate(wells)}, save_flows=True)
    flopy.mf6.ModflowGwfoc(model, head_filerecord='sparse_audit.hds', budget_filerecord='sparse_audit.cbc',
        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    simulation.write_simulation(silent=True)
    success, lines = simulation.run_simulation(silent=True, report=True)
    assert success and any('Normal termination' in line for line in lines), lines
    reader = flopy.utils.HeadFile(str(directory / 'sparse_audit.hds'), precision='double')
    heads = reader.get_alldata()
    reader.close()
    budget = flopy.utils.CellBudgetFile(str(directory / 'sparse_audit.cbc'), precision='double')
    entry_rates = [budget.get_data(text='RIV', kstpkper=(0, index))[0]['q'].copy() for index in range(len(wells))]
    budget.close()
    return heads, entry_rates


@pytest.mark.parametrize('device', ['cpu', 'cuda:0'])
@pytest.mark.parametrize('coefficient_state', ['saturated', 'current'])
@pytest.mark.parametrize('graphs', [False, True])
def test_sparse_switching_reversal_and_entry_count_matches_analytical_and_mf6(
        tmp_path, pinned_mf6, device, coefficient_state, graphs):
    """Independent roots cover activation, drying of RIV connections and changed CSR counts."""
    shape = (2, 3, 4)
    cell = (1, 2, 2)
    active = np.ones(shape, dtype=np.int32)
    active[0, 2, 2] = 0
    fixed = active.copy()
    fixed[cell] = 0
    rivers = [[(2., 8., 4.), (3., 7.5, 5.)], [(2., 8., 4.), (3., 7.5, 5.)],
        [(2., 8., 4.), (3., 7.5, 5.), (.5, 6., 3.)], [(2., 8.5, 4.), (3., 7.9, 5.)]]
    wells = [4., -6., -40., 4.]
    # Three real groundwater faces and one GHB give 4(h-8). Add individual
    # active DRN/RIV affine terms in periods 0,1,3; all rivers disconnect in 2.
    expected_heads = [(75.4 + 4.1) / 9.7, (75.4 - 5.9) / 9.7, (49. - 39.9) / 4., (77.6 + 4.1) / 9.7]
    native_heads, native_rates = native_switching_case(directory=tmp_path, executable=pinned_mf6,
        active=active, fixed=fixed, cell=cell, rivers=rivers, wells=wells)
    ghb = np.zeros(shape)
    ghb[cell] = 1.
    controls = NPF662KCycleControls(coefficient_state=coefficient_state, use_cuda_graph=graphs,
        cached_line_factors=True)
    with WarpDarcySolver3D(nx=4, ny=3, nz=2, dx=1., dy=1., dz=1., device=device,
            solver='kcycle', implementation='fast', formulation='mf6_6_6_2_npf') as model:
        model.build_from_K_fields(kx_field=np.ones(shape), ky_field=np.ones(shape), kz_field=np.ones(shape),
            active=active, bc_mask=fixed, bc_values=np.full(shape, 8.), initial_head=np.full(shape, 8.),
            R_field=.1, wel_entries=[(cell, wells[0])], ghb_conductance=ghb, ghb_head=np.full(shape, 8.),
            drn_entries=[(cell, .7, 7.)], riv_entries=[(cell, *entry) for entry in rivers[0]])
        previous = np.full(shape, 8.)
        owned = []
        for index, rate in enumerate(wells):
            model.update_initial_head(initial_head=previous)
            model.update_wel(entries=[(cell, rate)])
            model.update_riv_entries(entries=[(cell, *entry) for entry in rivers[index]])
            head, info = model.solve(npf_kcycle_controls=controls, npf_roundoff_acceptance=True, top_elevation=10.)
            assert info['converged'] and info['session_backend'] == 'npf662_newton'
            assert info['krylov_method'] == 'FGMRES' and info['npf_preconditioner'] == 'kcycle'
            assert info['preconditioner_cycles'] > 0
            assert head[cell] == pytest.approx(expected_heads[index], abs=2.e-12)
            assert head[cell] == pytest.approx(native_heads[(index, *cell)], abs=2.e-10)
            accepted = model.accepted_boundary_entry_rates(package='RIV', head=head, info=info)
            analytical = np.asarray([conductance * (max(head[cell], bottom) - stage)
                for conductance, stage, bottom in rivers[index]])
            np.testing.assert_allclose(accepted, analytical, rtol=0., atol=2.e-12)
            np.testing.assert_allclose(accepted, -native_rates[index], rtol=0., atol=2.e-10)
            np.testing.assert_array_equal(model.accepted_wel_rates(head=head, info=info), [rate])
            if device.startswith('cuda') and graphs:
                assert info['multigrid']['graph_replays'] > 0
            if not graphs:
                assert info['multigrid']['graph_replays'] == 0
            owned.append((head.copy(), info, accepted.copy()))
            previous = head
        for historical_head, historical_info, historical_rates in owned:
            np.testing.assert_array_equal(model.accepted_boundary_entry_rates(package='RIV',
                head=historical_head, info=historical_info), historical_rates)
