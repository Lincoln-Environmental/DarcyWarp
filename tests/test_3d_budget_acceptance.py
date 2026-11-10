# SPDX-License-Identifier: AGPL-3.0-or-later
"""Physical budget acceptance must not convert overflow or underflow to success."""
import numpy as np
import pytest

from DARCY_WARP_PACKAGE.physics.budget_3d import compute_3d_flow_budget, is_3d_budget_conserved


@pytest.mark.parametrize("scale", [1e-200, 1e200])
def test_cartesian_budget_rms_preserves_representable_magnitude(scale):
    shape = (1, 1, 1)
    zeros = np.zeros(shape)
    budget = compute_3d_flow_budget(
        head=zeros, source=np.full(shape, scale), faces=(zeros,) * 6,
        active=np.ones(shape, dtype=int), bc_mask=np.zeros(shape, dtype=int),
    )
    assert budget["cell_residual_rms"] == pytest.approx(scale, rel=1e-14, abs=0.0)
    assert not is_3d_budget_conserved(budget=budget, absolute_tolerance=0.0, relative_tolerance=1e-8)


def test_overflowed_throughput_cannot_hide_cellwise_imbalance():
    shape = (1, 1, 3)
    xp = np.array([[[1e308, 1e308, 0.]]])
    xm = np.array([[[0., 1e308, 1e308]]])
    zeros = np.zeros(shape)
    with np.errstate(over="ignore", invalid="ignore"):
        budget = compute_3d_flow_budget(
            head=np.array([[[1., 0., -1.]]]), source=np.array([[[0., 1., 0.]]]),
            faces=(xp, xm, zeros, zeros, zeros, zeros),
            active=np.ones(shape, dtype=int), bc_mask=np.array([[[1, 0, 1]]]),
        )
    assert not is_3d_budget_conserved(budget=budget, absolute_tolerance=1e-12, relative_tolerance=1e-8)


@pytest.mark.parametrize("key", ["cell_residual_rms", "relative_cell_imbalance", "throughput", "boundary_inflow"])
@pytest.mark.parametrize("invalid", [np.nan, np.inf])
def test_nonfinite_aggregate_cannot_pass_the_other_acceptance_branch(key, invalid):
    budget = {"cell_residual_rms": 0., "relative_cell_imbalance": 0., "throughput": 1., "boundary_inflow": 0.}
    budget[key] = invalid
    assert not is_3d_budget_conserved(budget=budget, absolute_tolerance=1e-12, relative_tolerance=1e-8)


@pytest.mark.parametrize("value", [np.nan, np.inf, -1.])
def test_absolute_budget_tolerance_is_validated(value):
    budget = {"cell_residual_rms": 0., "relative_cell_imbalance": 0.}
    with pytest.raises(ValueError, match="absolute|tolerance"):
        is_3d_budget_conserved(budget=budget, absolute_tolerance=value)


def test_negative_storage_is_rejected_not_clipped():
    zeros = np.zeros((1, 1, 1))
    with pytest.raises(ValueError, match="storage"):
        compute_3d_flow_budget(head=np.ones_like(zeros), source=zeros, faces=(zeros,) * 6,
            active=np.ones_like(zeros, dtype=int), bc_mask=np.zeros_like(zeros, dtype=int),
            storage_diag=-np.ones_like(zeros), head_prev=zeros)
