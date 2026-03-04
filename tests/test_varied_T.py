import os
import sys
import gc
import unittest
from pathlib import Path

import numpy as np

# Ensure package import works when tests are run from tests/ directly
try:
    from DARCY_WARP_PACKAGE.model_builder import build_base_fields, make_ugly_T_field
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver as wds
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from DARCY_WARP_PACKAGE.model_builder import build_base_fields, make_ugly_T_field
    from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver as wds


class TestUpdateTInPlace(unittest.TestCase):
    def test_update_T_matches_rebuild(self):
        # Skip if warp not available
        try:
            import warp as wp
        except Exception as exc:
            self.skipTest(f"warp not available: {exc}")

        device = os.environ.get("WARP_DEVICE", "cuda:0")
        try:
            wp.get_device(device)
        except Exception as exc:
            self.skipTest(f"Warp device {device} unavailable: {exc}")

        # small grid for quick tests
        nx = 40
        ny = 20
        dx = 100.0

        domain, dem, T_base, R_default = build_base_fields(nx=nx, ny=ny, dx=dx)

        # Create a few transmissivity variants to test
        T_iso = np.full_like(T_base, 3000.0)
        T_scaled = (T_base * 0.5).astype(T_base.dtype)
        T_newseed = make_ugly_T_field(nx=nx, ny=ny, domain=domain, seed=456)

        variants = [T_iso, T_scaled, T_newseed]

        tol = float(os.environ.get("WARP_HEAD_TOL", "2e-4"))

        for T_var in variants:
            # Reference: build solver from scratch with T_var
            with wds(nx=nx, ny=ny, dx=dx, device=device, use_ghb=False, solver_type="pcg") as solver_ref:
                solver_ref.build_from_truth_inputs(T_truth=T_var, R_truth=R_default)

                head_ref, _info = solver_ref.solve_multigrid_kcycle(
                    max_cycles=200,
                    nu_pre=2,
                    nu_post=2,
                    nu_coarse=2,
                    rel_tol=5.0e-7,
                    abs_tol_min=5.0e-7,
                    initial_head=dem,
                    return_info=True,
                    max_levels=6,
                    check_every_no=1,
                )

                if hasattr(head_ref, "numpy"):
                    head_ref = head_ref.numpy()
                else:
                    head_ref = np.asarray(head_ref)

            # Solver that is built once at base T and then updated in-place
            with wds(nx=nx, ny=ny, dx=dx, device=device, use_ghb=False, solver_type="pcg") as solver_upd:
                solver_upd.build_from_truth_inputs(T_truth=T_base, R_truth=R_default)
                # now update transmissivity in place
                solver_upd.update_T_in_place(T_var)

                head_upd, _info2 = solver_upd.solve_multigrid_kcycle(
                    max_cycles=200,
                    nu_pre=2,
                    nu_post=2,
                    nu_coarse=2,
                    rel_tol=5.0e-7,
                    abs_tol_min=5.0e-7,
                    initial_head=dem,
                    return_info=True,
                    max_levels=6,
                    check_every_no=1,
                )

                if hasattr(head_upd, "numpy"):
                    head_upd = head_upd.numpy()
                else:
                    head_upd = np.asarray(head_upd)

            # Ensure shapes match
            self.assertEqual(head_ref.shape, head_upd.shape, "Head shape mismatch between rebuild and update")

            active_mask = domain == 1
            diff = np.abs(np.asarray(head_ref) - np.asarray(head_upd))
            diff = diff[active_mask]

            self.assertTrue(np.all(np.isfinite(diff)), "Non-finite diffs present in active cells")
            max_diff = float(diff.max()) if diff.size else 0.0
            self.assertLessEqual(max_diff, tol, f"Max diff {max_diff} exceeds tolerance {tol}")

            # cleanup
            gc.collect()
            wp.synchronize_device(device)


if __name__ == "__main__":
    unittest.main()
