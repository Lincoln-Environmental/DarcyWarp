"""Canterbury boundary inputs distinguish missing stages, zero heads and overlap."""

import numpy as np
import pytest

from DARCY_WARP_PACKAGE.canterbury_case_study.canterbury_data_prep import (
    load_case_inputs, resolve_fixed_heads,
)


def test_canterbury_source_masks_preserve_zero_stages_and_apply_sea_precedence():
    north = np.array([[np.nan, 0., 0.]])
    south = np.array([[0., 0., 1.3]])
    sea = np.array([[0., 0., 0.1]])
    fixed, stages, audit = resolve_fixed_heads(active=np.ones_like(north, dtype=bool),
        stages=dict(north=north, south=south, sea=sea),
        masks=dict(north=np.array([[True, True, False]])), policy="sea_precedence")
    np.testing.assert_array_equal(fixed, [[0, 1, 1]])
    np.testing.assert_allclose(stages, [[0.0, 0.0, 0.1]])
    assert audit["missing_stages"] == [dict(row=0, column=0, source="north")]
    assert audit["overlaps"][0]["selected_source"] == "sea"


def test_archive_boundary_provenance_and_free_zero_stage_ghb():
    case = load_case_inputs(grid_size=100)
    missing = [(row["row"], row["column"]) for row in case.boundary_audit["missing_stages"]]
    assert missing == [(58, 834), (59, 841), (59, 842), (59, 843), (60, 852)]
    assert all(case.bc_mask[row, col] == 0 for row, col in missing)
    assert case.gh_mask[60, 852] == 1 and case.gh_head[60, 852] == 0.0
    assert case.bc_mask[625, 455] == 1
    np.testing.assert_allclose(case.bc_values[625, 455], 0.1)


def test_export_rejects_unconverged_head_before_writing(tmp_path, monkeypatch):
    from types import SimpleNamespace


    from DARCY_WARP_PACKAGE.canterbury_case_study import Canterbury_case_study as study

    shape = (2, 2)
    ones = np.ones(shape)
    case = SimpleNamespace(nx=2, ny=2, dx=100., active=ones, model_top=ones, recharge_base=ones,
        bc_mask=ones, bc_values=ones, gh_mask=ones, gh_head=ones, gh_width=ones,
        boundary_audit={"policy": "sea_precedence"})
    fit = dict(pilot_shape=shape, T_min=1., T_max=3., rbf_epsilon=10.,
        best_params={"r_scale": 0.5}, best_T_pilots=ones, best_logT=np.zeros(shape),
        hydraulic_settings=dict(use_ghb=True, gh_alpha=2., aq_thickness=300.,
            perturbation_strength=0., boundary_policy="sea_precedence", solve=dict(max_levels=2)))

    class RejectedSolver:
        def __init__(self, **constructor):
            assert constructor["aq_thickness"] == 300.

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def build_from_fields(self, **fields):
            assert fields["gh_alpha"] == 2.

        def build_hierarchy(self, **controls):
            pass

        def solve_multigrid_kcycle(self, **controls):
            return ones, {"converged": False, "r_rms_end": 1.0}

    def no_interpolation_cache(**controls):
        return None

    def constant_transmissivity(**controls):
        return ones

    monkeypatch.setattr(study, "WarpDarcySolver", RejectedSolver)
    monkeypatch.setattr(study, "precompute_rbf_cache", no_interpolation_cache)
    monkeypatch.setattr(study, "get_T_field_from_pilots_cached", constant_transmissivity)
    with pytest.raises(RuntimeError, match="final solve did not converge"):
        study.run_best_solution_and_save_outputs(case=case, outputs=fit, out_dir=tmp_path, device="cpu")
    assert not (tmp_path / "best_head.npz").exists()


@pytest.mark.parametrize("optimizer", ["dual_annealing", "minimize"])
def test_stochastic_fit_replays_winning_realisation(tmp_path, monkeypatch, optimizer):
    import pandas as pd
    from types import SimpleNamespace

    from DARCY_WARP_PACKAGE.canterbury_case_study import Canterbury_case_study as study

    shape = (8, 8)
    ones = np.ones(shape, dtype=np.float32)
    zeros = np.zeros_like(ones)
    fixed = zeros.copy()
    fixed[:, [0, -1]] = 1
    boundary = 10 * ones
    boundary[:, -1] = 11
    case = SimpleNamespace(nx=8, ny=8, dx=1., active=ones, bc_mask=fixed, bc_values=boundary,
        gh_mask=zeros, gh_head=zeros, gh_width=zeros, model_top=10 * ones, recharge_base=ones,
        obs_df=pd.DataFrame(dict(i=[3, 5], j=[2, 4], gwl=[10., 11.], std_gwl=[1., 1.])),
        boundary_audit={"policy": "sea_precedence"})

    def evaluate_twice(func, x0, **options):
        # The two evaluations draw different fields even at identical raw pilots.
        first = func(x0)
        second = func(x0)
        return study.OptimizeResult(x=x0, fun=min(first, second), nfev=2)

    monkeypatch.setattr(study, optimizer, evaluate_twice)
    calibrate = study.dual_annealing_calibration if optimizer == "dual_annealing" else study.lbfgsb_calibration
    fit = calibrate(case=case, device="cpu", use_ghb=False,
        pilot_nx=2, pilot_ny=2, T_min=1., T_max=100., rbf_epsilon=1., perturbation_strength=1.,
        rel_tol=1e-5, abs_tol_min=1e-4)
    assert not np.array_equal(fit["best_T_pilots"], fit["best_realised_T_pilots"])
    summary = study._summarize_best(fit)
    first = study.run_best_solution_and_save_outputs(case, summary, tmp_path / "first", device="cpu")
    second = study.run_best_solution_and_save_outputs(case, summary, tmp_path / "second", device="cpu")
    with np.load(first["head_path"]) as a, np.load(second["head_path"]) as b:
        np.testing.assert_array_equal(a["head"], b["head"])
    assert first["metrics"]["observational_weighted_sse"] == pytest.approx(fit["best_wsse"], abs=1e-6)
    del summary["best_realised_T_pilots"]
    with pytest.raises(ValueError, match="no saved realised pilot"):
        study.run_best_solution_and_save_outputs(case, summary, tmp_path / "legacy", device="cpu")
