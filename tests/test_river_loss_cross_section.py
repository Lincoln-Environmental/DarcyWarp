# SPDX-License-Identifier: AGPL-3.0-only
"""CPU tests for the reusable river cross-section case builder."""

from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from DARCY_WARP_PACKAGE.model_3d_inputs import Model3DInputs
from DARCY_WARP_PACKAGE.physics.budget_3d import (
    boundary_interface_flux,
    prepare_boundary_interface_flux,
)
from DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section import (
    CrossSectionConfig,
    _free_active_source_sink,
    _require_connected_active_domain,
    _robust_solver_kwargs,
    _solver_kwargs,
    build_cross_section,
    solve_case,
    validate_config,
)
from DARCY_WARP_PACKAGE.case_studies.river_loss_transient import (
    boundary_values_at_time,
    build_specific_storage_field,
    initial_linear_total_head,
    run_transient_case,
    validate_output_times,
)


class TestModel3DInputs(unittest.TestCase):
    def _inputs(self, **overrides):
        shape = (2, 1, 3)
        values = {
            "kx": np.ones(shape),
            "ky": np.ones(shape),
            "kz": np.ones(shape),
            "active": np.ones(shape, dtype=np.int32),
            "bc_mask": np.zeros(shape, dtype=np.int32),
            "bc_values": np.zeros(shape),
            "rhs": np.zeros(shape),
            "initial_head": np.zeros(shape),
            "dx": 1.0,
            "dy": 1.0,
            "dz": 1.0,
        }
        values.update(overrides)
        return Model3DInputs(**values)

    def test_validates_spacing_shape_and_masks(self):
        with self.assertRaises(ValueError):
            self._inputs(dx=0.0)
        with self.assertRaises(ValueError):
            self._inputs(kx=np.ones((1, 1, 3)))
        bc_mask = np.zeros((2, 1, 3), dtype=np.int32)
        bc_mask[0, 0, 0] = 1
        active = np.ones((2, 1, 3), dtype=np.int32)
        active[0, 0, 0] = 0
        with self.assertRaises(ValueError):
            self._inputs(active=active, bc_mask=bc_mask)

    def test_validates_metadata_and_named_masks(self):
        with self.assertRaises(ValueError):
            self._inputs(metadata={"bad": object()})
        with self.assertRaises(ValueError):
            self._inputs(named_masks={"wrong": np.zeros((3, 1, 2))})
        model = self._inputs(named_masks={"boundary": np.array([[[1, 0, 0]], [[0, 0, 0]]])})
        self.assertEqual(model.shape, (2, 1, 3))
        self.assertEqual(model.named_masks["boundary"].dtype, np.dtype(bool))


class TestRiverCrossSection(unittest.TestCase):
    @staticmethod
    def _config() -> CrossSectionConfig:
        return CrossSectionConfig(
            domain_length=20.0,
            domain_top=10.0,
            dx=5.0,
            dz=1.0,
            channel_half_width=5.0,
            channel_bed_elevation=8.0,
            channel_water_surface=9.0,
            braidplain_half_width=10.0,
            braidplain_thickness=2.0,
            device="cpu",
            max_cycles=100,
            max_levels=3,
            check_every_no=10,
        )

    def test_default_materials_and_both_outlet_modes(self):
        cfg = self._config()
        full = build_cross_section(cfg, "full_depth")
        lower = build_cross_section(cfg, "lower_only")
        self.assertEqual(full.inputs.shape, (10, 1, 5))
        self.assertTrue(np.array_equal(full.inputs.named_masks["channel"], full.channel_mask))
        self.assertGreater(int(full.channel_mask.sum()), 0)
        self.assertEqual(int(full.outlet_mask.sum()), 10)
        self.assertEqual(int(lower.outlet_mask.sum()), 8)
        self.assertTrue(np.all(full.kx[full.channel_mask] == cfg.fixed_head_k_multiplier * cfg.braidplain_kh))
        self.assertEqual(full.inputs.metadata["case"], "river_loss_cross_section")
        self.assertIn("saturated steady Darcy-flow", full.inputs.metadata["scientific_scope"])
        braid = (full.kx == cfg.braidplain_kh) & (full.kz == cfg.braidplain_kh / cfg.braidplain_anisotropy)
        regional = (full.kx == cfg.regional_kh) & (full.kz == cfg.regional_kh / cfg.regional_anisotropy)
        self.assertGreater(int(np.count_nonzero(braid)), 0)
        self.assertGreater(int(np.count_nonzero(regional)), 0)
        self.assertFalse(np.any(full.channel_mask & full.outlet_mask))

    def test_water_table_outlet_matches_face_aligned_discharge_elevation(self):
        cfg = replace(self._config(), channel_head=9.0, discharge_head=6.0)
        model = build_cross_section(cfg, "water_table_only")
        self.assertEqual(int(model.outlet_mask.sum()), 6)
        outlet_z = np.broadcast_to(model.z[:, None, None], model.active.shape)[
            model.outlet_mask
        ]
        self.assertTrue(np.all(outlet_z < cfg.discharge_head))
        self.assertAlmostEqual(float(np.max(outlet_z) + 0.5 * cfg.dz), 6.0)

    def test_boundary_flux_sign_and_single_face_count(self):
        shape = (1, 1, 3)
        head = np.array([[[3.0, 2.0, 1.0]]])
        boundary = np.array([[[1, 0, 0]]], dtype=np.int32)
        active = np.ones(shape, dtype=np.int32)
        tx_p = np.ones(shape)
        tx_m = np.ones(shape)
        zeros = np.zeros(shape)
        self.assertAlmostEqual(
            boundary_interface_flux(
                head=head,
                boundary_mask=boundary,
                active=active,
                tx_p=tx_p,
                tx_m=tx_m,
                ty_p=zeros,
                ty_m=zeros,
                tz_p=zeros,
                tz_m=zeros,
            ),
            1.0,
        )
        right = np.zeros(shape, dtype=np.int32); right[:, :, -1] = 1
        self.assertAlmostEqual(
            boundary_interface_flux(
                head=head, boundary_mask=right, active=active,
                tx_p=tx_p, tx_m=tx_m, ty_p=zeros, ty_m=zeros, tz_p=zeros, tz_m=zeros,
            ), -1.0,
        )

    def test_boundary_flux_ignores_inactive_and_same_boundary_faces(self):
        shape = (1, 1, 3)
        head = np.array([[[3.0, 2.0, 1.0]]])
        active = np.array([[[1, 0, 1]]], dtype=np.int32)
        boundary = np.array([[[1, 1, 0]]], dtype=np.int32)
        ones = np.ones(shape)
        zeros = np.zeros(shape)
        assert boundary_interface_flux(
            head=head, boundary_mask=boundary, active=active,
            tx_p=ones, tx_m=ones, ty_p=zeros, ty_m=zeros, tz_p=zeros, tz_m=zeros,
        ) == 0.0

    def test_prepared_boundary_flux_matches_validated_general_path(self):
        shape = (2, 2, 4)
        rng = np.random.default_rng(42)
        head = rng.normal(size=shape)
        active = np.ones(shape, dtype=np.int32)
        boundary = np.zeros(shape, dtype=np.int32)
        boundary[:, :, 0] = 1
        faces = tuple(rng.uniform(0.1, 2.0, size=shape) for _ in range(6))
        expected = boundary_interface_flux(
            head=head,
            boundary_mask=boundary,
            active=active,
            tx_p=faces[0],
            tx_m=faces[1],
            ty_p=faces[2],
            ty_m=faces[3],
            tz_p=faces[4],
            tz_m=faces[5],
        )
        plan = prepare_boundary_interface_flux(
            boundary_mask=boundary,
            active=active,
            tx_p=faces[0],
            tx_m=faces[1],
            ty_p=faces[2],
            ty_m=faces[3],
            tz_p=faces[4],
            tz_m=faces[5],
        )
        self.assertAlmostEqual(plan.evaluate(head=head), expected)

    def test_case_geometry_validation_is_explicit(self):
        with self.assertRaisesRegex(ValueError, "channel_head"):
            build_cross_section(self._config().__class__(**{
                **self._config().__dict__, "channel_head": 90.0, "discharge_head": 96.0,
            }), "full_depth")
        with self.assertRaisesRegex(ValueError, "smoother"):
            build_cross_section(self._config().__class__(**{
                **self._config().__dict__, "implementation": "fast", "smoother": "vertical_line",
            }), "full_depth")
        with self.assertRaisesRegex(ValueError, "robust_abs_tol_min"):
            build_cross_section(self._config().__class__(**{
                **self._config().__dict__, "robust_abs_tol_min": 0.0,
            }), "full_depth")

    def test_robust_retry_forces_a_materially_tighter_warm_start_solve(self):
        cfg = replace(
            self._config(),
            max_cycles=600,
            robust_max_cycles=400,
            robust_rel_tol=1.0e-7,
            robust_abs_tol_min=1.0e-7,
            robust_dh_rms_tol=1.0e-5,
        )
        validate_config(cfg)
        retry = _robust_solver_kwargs(
            cfg=cfg,
            simple_kwargs={
                "max_cycles": cfg.max_cycles,
                "rel_tol": cfg.rel_tol,
                "abs_tol_min": cfg.abs_tol_min,
                "dh_rms_tol": cfg.dh_rms_tol,
                "nu_pre": cfg.nu_pre,
                "nu_post": cfg.nu_post,
                "nu_coarse": cfg.nu_coarse,
                "omega": cfg.omega,
            },
        )
        self.assertEqual(retry["max_cycles"], 600)
        self.assertLess(retry["rel_tol"], cfg.rel_tol)
        self.assertLess(retry["abs_tol_min"], cfg.abs_tol_min)
        self.assertLess(retry["dh_rms_tol"], cfg.dh_rms_tol)

    def test_chebyshev_receives_only_supported_solver_controls(self):
        cfg = replace(self._config(), solver="chebyshev")
        controls = _solver_kwargs(cfg=cfg)
        self.assertEqual(controls["max_iter"], cfg.max_cycles)
        self.assertNotIn("max_cycles", controls)
        self.assertNotIn("max_levels", controls)
        self.assertNotIn("smoother", controls)

        retry = _robust_solver_kwargs(cfg=cfg, simple_kwargs=controls)
        self.assertEqual(retry["max_iter"], cfg.robust_max_cycles)
        self.assertNotIn("nu_pre", retry)
        self.assertEqual(retry["abs_tol_min"], cfg.robust_abs_tol_min)

    def test_connectivity_requires_one_channel_linked_active_component(self):
        connected = np.ones((1, 5), dtype=bool)
        channel = np.array([[1, 0, 0, 0, 0]], dtype=bool)
        outlet = np.array([[0, 0, 0, 0, 1]], dtype=bool)
        _require_connected_active_domain(
            active_xz=connected,
            channel_xz=channel,
            outlet_xz=outlet,
        )

        disconnected = connected.copy()
        disconnected[0, 2] = False
        with self.assertRaisesRegex(ValueError, "one connected component"):
            _require_connected_active_domain(
                active_xz=disconnected,
                channel_xz=channel,
                outlet_xz=outlet,
            )

    def test_source_budget_uses_only_free_active_equations(self):
        rhs = np.array([[[1.0, 2.0, 4.0, 8.0]]])
        active = np.array([[[1, 1, 0, 1]]], dtype=np.int32)
        bc_mask = np.array([[[1, 0, 0, 0]]], dtype=np.int32)
        self.assertEqual(
            _free_active_source_sink(rhs=rhs, active=active, bc_mask=bc_mask),
            10.0,
        )

    def test_anisotropy_is_assigned_to_requested_material_zone(self):
        cfg = replace(self._config(), braidplain_anisotropy=4.0, regional_anisotropy=7.0)
        model = build_cross_section(cfg, "full_depth")
        braid = (model.kx == cfg.braidplain_kh) & (model.kz == cfg.braidplain_kh / 4.0)
        regional = (model.kx == cfg.regional_kh) & (model.kz == cfg.regional_kh / 7.0)
        assert np.any(braid) and np.any(regional)

    def test_small_cpu_solve_closes_fixed_head_budget(self):
        cfg = self._config()
        result, head = solve_case(
            build_cross_section(cfg, "full_depth"),
            varied_ratio=1.0,
            anisotropy_target="both",
        )
        self.assertEqual(head.shape, (10, 1, 5))
        self.assertTrue(np.all(np.isfinite(head)))
        self.assertTrue(result.converged)
        self.assertLessEqual(result.relative_mass_imbalance, cfg.practical_mass_imbalance_tol)

    def test_small_classic_chebyshev_solve_uses_public_backend(self):
        cfg = replace(self._config(), solver="chebyshev")
        result, head = solve_case(
            build_cross_section(cfg, "full_depth"),
            varied_ratio=1.0,
            anisotropy_target="both",
        )
        self.assertTrue(np.all(np.isfinite(head)))
        self.assertTrue(result.converged)
        self.assertLessEqual(result.relative_mass_imbalance, cfg.practical_mass_imbalance_tol)

    def test_transient_helpers_reproduce_linear_initial_head_and_channel_ramp(self):
        cfg = replace(self._config(), channel_head=9.0, discharge_head=6.0)
        model = build_cross_section(cfg, "water_table_only")
        initial = initial_linear_total_head(model=model)
        boundary0, fraction0 = boundary_values_at_time(
            model=model,
            time_days=0.0,
            channel_ramp_days=1.0,
        )
        boundary1, fraction1 = boundary_values_at_time(
            model=model,
            time_days=1.0,
            channel_ramp_days=1.0,
        )
        self.assertEqual(fraction0, 0.0)
        self.assertEqual(fraction1, 1.0)
        np.testing.assert_allclose(boundary0[model.channel_mask], initial[model.channel_mask])
        np.testing.assert_allclose(boundary1[model.channel_mask], cfg.channel_head)
        np.testing.assert_allclose(boundary1[model.outlet_mask], cfg.discharge_head)

        storage = build_specific_storage_field(
            model=model,
            braidplain_specific_storage=2.0e-4,
            regional_specific_storage=1.0e-5,
        )
        self.assertTrue(np.all(storage[model.bc_mask != 0] == 0.0))
        self.assertGreater(float(np.max(storage)), float(np.min(storage[storage > 0.0])))
        with self.assertRaises(ValueError):
            validate_output_times([1.0, 2.0])

    def test_small_zero_storage_transient_sequence_closes_final_budget(self):
        cfg = replace(
            self._config(),
            channel_head=9.0,
            discharge_head=6.0,
            check_every_no=1,
        )
        model = build_cross_section(cfg, "water_table_only")
        rows, heads, metadata = run_transient_case(
            model=model,
            output_times_days=[0.0, 0.05, 0.1],
            channel_ramp_days=0.05,
            braidplain_specific_storage=0.0,
            regional_specific_storage=0.0,
            maximum_timestep_days=0.05,
        )
        self.assertEqual(heads.shape, (3, *model.active.shape))
        self.assertEqual(metadata["time_formulation"], "quasi_steady_zero_storage")
        self.assertTrue(rows[-1].converged)
        self.assertLessEqual(
            rows[-1].relative_mass_imbalance,
            cfg.practical_mass_imbalance_tol,
        )
        self.assertGreater(rows[-1].cumulative_channel_leakage, 0.0)

    def test_small_positive_storage_sequence_uses_backward_euler(self):
        cfg = replace(
            self._config(),
            channel_head=9.0,
            discharge_head=6.0,
            check_every_no=1,
        )
        model = build_cross_section(cfg, "water_table_only")
        rows, heads, metadata = run_transient_case(
            model=model,
            output_times_days=[0.0, 0.05],
            channel_ramp_days=0.05,
            braidplain_specific_storage=2.0e-4,
            regional_specific_storage=1.0e-4,
            maximum_timestep_days=0.01,
        )
        self.assertEqual(heads.shape, (2, *model.active.shape))
        self.assertEqual(metadata["time_formulation"], "confined_backward_euler")
        self.assertTrue(rows[-1].solved)
        self.assertTrue(rows[-1].converged)
        self.assertLessEqual(
            rows[-1].relative_mass_imbalance,
            cfg.practical_mass_imbalance_tol,
        )

    def test_small_positive_storage_sequence_uses_fast_backend(self):
        cfg = replace(
            self._config(),
            channel_head=9.0,
            discharge_head=6.0,
            implementation="fast",
            smoother="jacobi",
            check_every_no=1,
        )
        model = build_cross_section(cfg, "water_table_only")
        rows, heads, metadata = run_transient_case(
            model=model,
            output_times_days=[0.0, 0.05],
            channel_ramp_days=0.05,
            braidplain_specific_storage=2.0e-4,
            regional_specific_storage=1.0e-4,
            maximum_timestep_days=0.05,
        )
        self.assertEqual(heads.shape, (2, *model.active.shape))
        self.assertEqual(metadata["time_formulation"], "confined_backward_euler")
        self.assertTrue(rows[-1].converged)
        self.assertLessEqual(
            rows[-1].relative_mass_imbalance,
            cfg.practical_mass_imbalance_tol,
        )


if __name__ == "__main__":
    unittest.main()
