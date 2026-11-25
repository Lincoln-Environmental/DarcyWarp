"""Native MF6 and benchmark regressions for the November audit."""
import json
import os
from pathlib import Path

import numpy as np
import pytest
import warp as wp

from DARCY_WARP_PACKAGE import create_solver
from DARCY_WARP_PACKAGE.config import NP_FLOAT
from DARCY_WARP_PACKAGE.project_base import MF6
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662Case
from DARCY_WARP_PACKAGE.validation.npf662_transient_mf6 import run_transient_mf6
from working_tests.run_2d_transient_sanity_matrix import _apply_case_parity
from working_tests.run_2d_transient_vs_mf6 import _budget_gate_from_artifact, main
from working_tests.run_2d_transient_warp_replay import build_case_setup, ensure_case_artifact
from working_tests.transient_artifacts import validate_transient_artifact


def chd_case(*, adjacent):
    if adjacent:
        shape = (1, 1, 2)
        head = np.array([[[700., 500.]]])
        fixed = np.ones(shape, dtype=int)
        source = np.zeros(shape)
        exchange = np.array([[[200., -200.]]])
    else:
        shape = (1, 1, 3)
        head = np.array([[[700., 500., 300.]]])
        fixed = np.array([[[0, 1, 0]]])
        source = np.array([[[200., 0., -200.]]])
        exchange = np.zeros(shape)
    k = np.ones(shape)
    case = NPF662Case(kx=k, ky=k, kz=k, active=np.ones(shape, dtype=int), fixed=fixed,
        values=np.where(fixed, head, 0.), source=source, initial=head, dx=1., dy=1., dz=1.)
    return case, exchange


@pytest.fixture(scope="module", params=[True, False], ids=["adjacent", "through_flow"])
def native_chd_case(request, tmp_path_factory):
    if MF6 is None:
        pytest.skip("Pinned native MF6 required")
    case, expected = chd_case(adjacent=request.param)
    directory = tmp_path_factory.mktemp("nov25-chd") / "native"
    head, summary, fields = run_reference_mf6(case=case, executable=Path(MF6), directory=directory)
    assert summary["chd_budget_valid"]
    assert summary["independent_conservation"]
    native = fields["chd_native_inflow"]
    np.testing.assert_allclose(native, expected, rtol=0., atol=1e-12)
    return case, head, native


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("implementation,formulation", [
    ("classic", "legacy"), ("fast", "legacy"), ("fast", "mf6_6_6_2_npf"),
])
def test_gross_chd_matches_native_cell_rates(native_chd_case, device, implementation, formulation):
    if device != "cpu" and (os.environ.get("DARCYWARP_TEST_CPU_ONLY") == "1" or not wp.is_cuda_available()):
        pytest.skip("CUDA required")
    if implementation == "fast" and NP_FLOAT != np.float64:
        pytest.skip("Fast 3D requires FP64")
    case, native_head, native = native_chd_case
    nz, ny, nx = case.active.shape
    with create_solver(dim=3, nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy, dz=case.dz,
                       device=device, implementation=implementation, formulation=formulation) as model:
        model.build_from_K_fields(kx_field=case.kx, ky_field=case.ky, kz_field=case.kz,
            active=case.active, bc_mask=case.fixed, bc_values=case.values,
            rhs=case.source, initial_head=case.initial)
        head, info = model.solve()
    assert info["converged"]
    np.testing.assert_allclose(head, native_head, rtol=0., atol=1e-10)
    budget = info["independent_acceptance"] if formulation != "legacy" else info["budget"]
    for checked in (budget, info.get("device_final_metrics", budget)):
        assert checked["chd_inflow"] == pytest.approx(np.maximum(native, 0.).sum(), abs=1e-12)
        assert checked["chd_outflow"] == pytest.approx(np.maximum(-native, 0.).sum(), abs=1e-12)
    if not np.any(native):
        assert budget["free_boundary_inflow"] == pytest.approx(200.)
        assert abs(budget["free_boundary_outflow"]) == pytest.approx(200.)


@pytest.mark.parametrize("formulation", ["steady", "transient"])
def test_unapproved_executable_is_not_invoked(tmp_path, formulation):
    invoked = tmp_path / "invoked"
    fake = tmp_path / "mock-mf6"
    fake.write_text(f"#!/bin/sh\ntouch {invoked}\nprintf 'mf6: 6.6.2 05/12/2025\\n'\n")
    fake.chmod(0o755)
    case, _ = chd_case(adjacent=False)
    with pytest.raises(ValueError, match="pinned MF6"):
        options = dict(case=case, executable=fake, directory=tmp_path / "native")
        if formulation == "steady":
            run_reference_mf6(**options)
        else:
            run_transient_mf6(**options, source_factors=[1.], dt=1., ss=1e-4, sy=0.)
    assert not invoked.exists()


def test_changed_k_regenerates_native_history_and_steady_initial_head(tmp_path):
    if MF6 is None:
        pytest.skip("Pinned native MF6 required")
    original = tmp_path / "mf6_transient_heads.npz.lzma"
    main(nx=8, ny=8, dx=100., n_weeks=2, t_field_kind="homogeneous", hydraulic_conductivity=10.,
         t_field_seed=42, out_path=original, executable=MF6)
    setup = build_case_setup(nx=8, ny=8, n_periods=2, t_field_kind="homogeneous")
    setup.update(artifact_path=original, hydraulic_conductivity=100.)
    actual = ensure_case_artifact(setup, executable=MF6)
    assert actual != original
    first = validate_transient_artifact(original, verify_native_files=True)
    second = validate_transient_artifact(actual, verify_native_files=True)
    assert np.all(first["k_field"][first["active"] != 0] == 10.)
    assert np.all(second["k_field"][second["active"] != 0] == 100.)
    assert first["warm_start_fingerprint"] != second["warm_start_fingerprint"]


def test_native_budget_gate_uses_mean_flow_including_zero(tmp_path):
    path = tmp_path / "budget.npz"
    rows = [dict(total_in=0., total_out=0., percent_discrepancy=0.)]
    np.savez(path, mf6_mass_balance_json=json.dumps(rows))
    assert _budget_gate_from_artifact(path) == (0., 1.)
    rows.append(dict(total_in=101.5, total_out=100., percent_discrepancy=0.7444168734))
    np.savez(path, mf6_mass_balance_json=json.dumps(rows))
    discrepancy, limit = _budget_gate_from_artifact(path)
    assert discrepancy == pytest.approx(1.4888337469)
    assert discrepancy > limit


@pytest.mark.parametrize("second", ["missing", "partial", "same_path"])
def test_matrix_rejects_missing_or_incomplete_repeat(tmp_path, second):
    full = np.ones((3, 4, 5))
    first_path = tmp_path / "first.npz"
    second_path = tmp_path / "second.npz"
    np.savez(first_path, heads_per_period=full)
    if second == "partial":
        np.savez(second_path, heads_per_period=full[:-1])
    if second == "same_path":
        second_path = first_path
    row = dict(case_label="case", variant="classic_device_fp64", completed=True, repeats=2,
               head_history_shape=list(full.shape), head_artifact_paths=[str(first_path), str(second_path)])
    _apply_case_parity(rows={"case": row}, label="case", variants=("classic_device_fp64",))
    assert row["head_parity"]["passed"] is False


def test_npf_native_device_reports_chd_budget(native_chd_case):
    if NP_FLOAT != np.float64 or os.environ.get("DARCYWARP_TEST_CPU_ONLY") == "1" or not wp.is_cuda_available():
        pytest.skip("CUDA FP64 NPF required")
    case, expected_head, native = native_chd_case
    nz, ny, nx = case.active.shape
    with create_solver(dim=3, nx=nx, ny=ny, nz=nz, dx=case.dx, dy=case.dy, dz=case.dz,
                       device="cuda:0", implementation="fast", formulation="mf6_6_6_2_npf") as model:
        model.build_from_K_fields(kx_field=case.kx, ky_field=case.ky, kz_field=case.kz,
            active=case.active, bc_mask=case.fixed, bc_values=case.values,
            rhs=case.source, initial_head=case.initial)
        previous = case.initial.copy()
        previous[case.fixed != 0] = 1.e30
        head, info = model.solve(transient=True, dt=1., head_prev=previous, ss=1.e-4, sy=0.,
                                 npf_device_native=True)
    assert info["converged"] and info["device_native"]
    np.testing.assert_array_equal(head, expected_head)
    assert info["chd_budget"]["chd_inflow"] == pytest.approx(np.maximum(native, 0.).sum(), abs=1.e-12)
    assert info["chd_budget"]["chd_outflow"] == pytest.approx(np.maximum(-native, 0.).sum(), abs=1.e-12)
