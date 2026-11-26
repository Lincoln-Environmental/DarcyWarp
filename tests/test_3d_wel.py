# SPDX-License-Identifier: AGPL-3.0-or-later
"""Analytical validation for the explicit volumetric 3D WEL package.

Well rates are volumetric [L3/T] in every dimensionality with the MF6 sign
convention: negative abstracts, positive injects, zero disables.
"""
import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE.model_3d_inputs import Model3DInputs
from DARCY_WARP_PACKAGE.physics.wel import (
    signed_rate_budget,
    validate_well_entries,
    well_cell_volume_rates,
    well_entries,
)
from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

DEVICES = ["cpu", "cuda:0"]


def _cuda_available(device):
    if not device.startswith("cuda"):
        return True
    wp.init()
    return wp.is_cuda_available()


def _anchored_model(device, shape=(1, 3, 3), k=10., dz=5., well=0., recharge=None,
                    rhs=None, cell=(0, 1, 1), dx=20., dy=20., formulation="mf6_6_6_2_npf"):
    """One-layer confined model anchored by a prescribed-head column."""
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    values = np.zeros(shape)
    entries = None if well == 0. else [((shape[0] - 1, cell[1], cell[2]), well)]
    model = WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=dx, dy=dy, dz=dz,
        formulation=formulation, solver="kcycle", implementation="fast", device=device)
    model.build_from_K_fields(kx_field=np.full(shape, k), ky_field=np.full(shape, k),
        kz_field=np.full(shape, k), active=active, bc_mask=fixed, bc_values=values,
        rhs=rhs, R_field=recharge, wel_entries=entries, initial_head=np.zeros(shape))
    return model


@pytest.mark.parametrize("device", DEVICES)
def test_pumping_drawdown_matches_independent_cell_balance(device):
    """Steady confined drawdown equals an independently assembled scipy solve."""
    from scipy.sparse import coo_matrix, diags
    from scipy.sparse.linalg import spsolve
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    shape = (1, 3, 3)
    k, dx, dy, dz = 10., 20., 20., 5.
    well_volume = -400.
    model = _anchored_model(device, shape=shape, k=k, dz=dz, well=well_volume, dx=dx, dy=dy)
    try:
        head, info = model.solve()
        assert info["converged"]
        n = shape[1] * shape[2]
        ids = np.arange(n).reshape(shape[1:])
        rows, cols, values = [], [], []
        rhs = np.zeros(n)
        conductance = k * dz * dy / dx  # x faces
        for j in range(shape[1]):
            for i in range(shape[2] - 1):
                a, b = ids[j, i], ids[j, i + 1]
                rows += [a, b]; cols += [b, a]; values += [-conductance, -conductance]
        column = k * dz * dx / dy  # y faces
        for j in range(shape[1] - 1):
            for i in range(shape[2]):
                a, b = ids[j, i], ids[j + 1, i]
                rows += [a, b]; cols += [b, a]; values += [-column, -column]
        # Face couplings enter negative off the diagonal; the diagonal is the
        # positive row sum of the face conductances.
        off_diagonal = coo_matrix((values, (rows, cols)), shape=(n, n)).tocsr()
        diagonal = np.asarray(-off_diagonal.sum(axis=1)).ravel()
        matrix = (off_diagonal + diags(diagonal)).tocsr()
        free = np.ones(n, dtype=bool)
        fixed_cells = ids[:, 0]
        free[fixed_cells] = False
        rhs[fixed_cells] = -diagonal[fixed_cells] * 0.0  # prescribed heads are zero
        well_flat = np.ravel_multi_index((1, 1), shape[1:])
        rhs[well_flat] += well_volume
        fixed_rows = np.flatnonzero(~free)
        solution = np.zeros(n)
        keep = np.flatnonzero(free)
        reduced = matrix[keep][:, keep]
        solution[keep] = spsolve(reduced, (rhs[keep] - matrix[keep][:, fixed_rows] @ solution[fixed_rows]))
        expected = solution.reshape(shape[1:])
        np.testing.assert_allclose(head[0], expected, rtol=1e-9, atol=1e-12)
        budgets = info["source_budgets"]
        assert budgets["wel"]["outflow"] == pytest.approx(abs(well_volume))
        assert budgets["wel"]["net"] == pytest.approx(well_volume)
    finally:
        model.close()


@pytest.mark.parametrize("device", DEVICES)
def test_multiple_and_opposing_entries_keep_gross_budgets(device):
    """Opposing wells in one cell cancel numerically but never in the budget."""
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    shape = (1, 3, 3)
    model = _anchored_model(device, shape=shape)
    try:
        model.update_wel(well_entries((0, 1, 1), (0, 1, 1), rates=[-600., 250.]))
        head, info = model.solve()
        assert info["converged"]
        budgets = info["source_budgets"]["wel"]
        assert budgets["inflow"] == pytest.approx(250.)
        assert budgets["outflow"] == pytest.approx(600.)
        assert budgets["net"] == pytest.approx(-350.)
        source = model._session.rhs
        # Only the shared cell carries the net rate.
        assert source[0, 1, 1] == pytest.approx(-350.)
        assert np.count_nonzero(source) == 1
    finally:
        model.close()


@pytest.mark.parametrize("device", DEVICES)
def test_chd_well_budget_with_explicit_zero_storage_is_signed_once(device):
    """A requested CHD-cell WEL keeps nonnegative gross CHD accounting."""
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    shape = (1, 2, 2)
    model = _anchored_model(
        device, shape=shape, well=-100., cell=(0, 0, 0), formulation="legacy")
    try:
        zeros = np.zeros(shape)
        head, info = model.solve(
            transient=True, storage_diag=zeros, storage_coeff=0., dt=1., head_prev=zeros)
        assert info["converged"]
        np.testing.assert_allclose(head, 0., rtol=0., atol=0.)
        budget = info["budget"]
        assert budget["chd_inflow"] == pytest.approx(100.)
        assert budget["chd_outflow"] == pytest.approx(0.)
        assert budget["boundary_inflow"] == pytest.approx(100.)
        assert budget["boundary_outflow"] == pytest.approx(0.)
        assert budget["source_all_total"] == pytest.approx(-100.)
        assert budget["chd_cell_rate"].sum() == pytest.approx(100.)
        assert info["source_budgets"]["wel"] == pytest.approx(
            dict(inflow=0., outflow=100., net=-100.))
    finally:
        model.close()


def test_well_entries_require_parallel_cell_and_rate_arrays():
    """A prebuilt sparse value cannot silently truncate unequal arrays."""
    from DARCY_WARP_PACKAGE.physics.wel import WellEntries
    entries = WellEntries(cells=np.array([[0, 0, 0], [0, 1, 1]]), rates=np.array([-1.]))
    with pytest.raises(ValueError, match='one volumetric rate per well entry'):
        validate_well_entries(entries=entries, shape=(1, 2, 2), active=np.ones((1, 2, 2)))


@pytest.mark.parametrize("device", DEVICES)
def test_sources_update_independently_and_atomically(device):
    """Recharge, raw rates and wells replace independently; failures change nothing."""
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    shape = (1, 3, 3)
    model = _anchored_model(device, shape=shape, well=-300., recharge=1e-3,
                            rhs=np.zeros(shape))
    try:
        rhs = np.zeros(shape)
        rhs[0, 2, 2] = 50.
        model.update_rhs(rhs)
        assert model.wel_budget()["outflow"] == pytest.approx(300.)
        model.update_wel([((0, 2, 2), -100.)])
        source = model._session.rhs
        # The built recharge adds rate*area = 0.4 to every receiving column.
        assert source[0, 2, 2] == pytest.approx(-50. + 0.4)  # raw 50 replaced, well moved
        assert source[0, 1, 1] == pytest.approx(0.4)  # old well cell keeps only recharge
        model.update_R_in_place(None)
        source = model._session.rhs
        assert source[0, 2, 2] == pytest.approx(-50.)  # recharge cleared, others kept
        assert source[0, 1, 1] == pytest.approx(0.)
        snapshot = model._session.rhs.copy()
        with pytest.raises(ValueError):
            model.update_wel([((0, 9, 9), -10.)])  # outside the grid
        np.testing.assert_array_equal(model._session.rhs, snapshot)
        with pytest.raises(ValueError):
            model.update_wel([((0, 1, 0), float("nan"))])
        np.testing.assert_array_equal(model._session.rhs, snapshot)
    finally:
        model.close()


@pytest.mark.parametrize("device", DEVICES)
def test_transient_history_wel_schedule_and_volumes(device):
    """Scheduled wells replace built entries per step with integrated volumes."""
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    shape = (1, 3, 3)
    model = _anchored_model(device, shape=shape, well=-200.)
    try:
        initial = np.zeros(shape)
        dt = 86400.
        heads, info = model.solve_transient_npf(
            dt=[dt, dt], head_prev=initial, ss=1e-5, sy=0.0,
            wel_rates=[[((0, 1, 1), -400.)], None],
            source_factors=[1.0, 2.0])
        assert info["converged"]
        assert info["accepted_steps"] == 2
        periods = info["periods"]
        first = periods[0]["source_budgets"]["wel"]
        assert first["outflow"] == pytest.approx(400.)
        second = periods[1]["source_budgets"]["wel"]
        assert second["net"] == pytest.approx(0.)
        totals = info["accepted_exchange_volume_totals"]["sources"]["wel"]
        assert totals["outflow"] == pytest.approx(400. * dt)
        assert totals["net"] == pytest.approx(-400. * dt)
        # The built forcing is restored after the history call.
        assert model.wel_budget()["outflow"] == pytest.approx(200.)
    finally:
        model.close()


def test_invalid_entries_rejected_before_mutation():
    """Entry validation rejects nonfinite rates, bad indices and dense scalars."""
    shape = (1, 2, 2)
    active = np.ones(shape, dtype=np.int32)
    with pytest.raises(ValueError):
        validate_well_entries(entries=[((0, 0, 0), float("nan"))], shape=shape, active=active)
    with pytest.raises(ValueError):
        validate_well_entries(entries=[((0, 0), -1.)], shape=shape, active=active)  # wrong ndim
    with pytest.raises(ValueError):
        validate_well_entries(entries=-100., shape=shape, active=active)  # scalar broadcast trap
    inactive = active.copy()
    inactive[0, 1, 1] = 0
    with pytest.raises(ValueError):
        validate_well_entries(entries=[((0, 1, 1), -1.)], shape=shape, active=inactive)
    entries = validate_well_entries(entries=[((0, 1, 1), -1.)], shape=shape, active=active)
    dense = well_cell_volume_rates(entries=entries, shape=shape)
    assert dense[0, 1, 1] == -1.
    budget = signed_rate_budget(rates=np.array([-3., 1., 2.]))
    assert budget == dict(inflow=3., outflow=3., net=0.)


@pytest.mark.parametrize("device", DEVICES)
def test_model_inputs_and_devices_agree(device):
    """Model3DInputs carries wells and CPU/CUDA heads agree to roundoff."""
    if not _cuda_available(device):
        pytest.skip("CUDA is unavailable")
    shape = (1, 3, 3)
    inputs = Model3DInputs(
        kx=np.full(shape, 10.), ky=np.full(shape, 10.), kz=np.full(shape, 10.),
        active=np.ones(shape, dtype=np.int32),
        bc_mask=np.pad(np.ones((shape[1], 1)), ((0, 0), (0, shape[2] - 1))).reshape(shape),
        bc_values=np.zeros(shape), dx=20., dy=20., dz=5.,
        wel_entries=[((0, 1, 2), -250.)], R_field=5e-4,
    )
    cpu = inputs.build_solver(device="cpu", formulation="mf6_6_6_2_npf",
        solver="kcycle", implementation="fast")
    try:
        cpu_head, cpu_info = cpu.solve()
        assert cpu_info["converged"]
        cuda = inputs.build_solver(device=device, formulation="mf6_6_6_2_npf",
                solver="kcycle", implementation="fast")
        try:
            cuda_head, cuda_info = cuda.solve()
            assert cuda_info["converged"]
            np.testing.assert_allclose(cuda_head, cpu_head, rtol=1e-11, atol=1e-11)
        finally:
            cuda.close()
    finally:
        cpu.close()
