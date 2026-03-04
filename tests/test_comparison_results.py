import gc
import os
from importlib import import_module
from pathlib import Path
import unittest

import numpy as np

# When running under PyCharm's test runner (or imported by unittest discovery),
# the module-level __main__ is not executed. Set default environment guards so
# the Play/Run action in PyCharm (which uses unittest discovery/import) will
# still run the expensive Warp truth comparisons unless the user overrides them.
# Force the test guard to true when this module is imported by a test runner
# (PyCharm's Play button runs tests via import/unittest discovery). If you
# truly want to skip tests, unset RUN_WARP_TRUTH_TESTS in your run config.
os.environ["RUN_WARP_TRUTH_TESTS"] = "1"
# Respect an existing DARCY_FLOAT if set, otherwise default to double
os.environ.setdefault("DARCY_FLOAT", os.environ.get("DARCY_FLOAT", "float64"))

try:
    from DARCY_WARP_PACKAGE.model_builder import _build_domain, _build_dem, make_ugly_T_field
    from DARCY_WARP_PACKAGE.sanity_case_config import GRID_CASES
except ModuleNotFoundError:
    # When running the file directly from the tests/ directory the package
    # root may not be on sys.path. Insert the repo root (parent of tests) so
    # the top-level package `DARCY_WARP_PACKAGE` can be imported.
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from DARCY_WARP_PACKAGE.model_builder import _build_domain, _build_dem, make_ugly_T_field
    from DARCY_WARP_PACKAGE.sanity_case_config import GRID_CASES

DEFAULT_WARP_HEAD_TOL = 2.0e-4

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    val = raw.strip().lower()
    if val in ("1", "true", "yes", "y", "on"):
        return True
    if val in ("0", "false", "no", "n", "off"):
        return False
    raise ValueError(f"Invalid boolean value for {name}: {raw}")


class TestWarpVsMf6Truth(unittest.TestCase):
    def _run_warp_truth_comparison(
        self,
        *,
        solver_module: str,
        solver_kwargs: dict | None = None,
        tol_env: str = "WARP_HEAD_TOL",
        tol_override: float | None = None,
        labels_override: list[str] | tuple[str, ...] | None = None,
        n_solves: int = 2,
        case_prefix: str = "",
    ) -> None:
        # Run by default when imported (PyCharm Play/Run); allow opt-out via
        # SKIP_WARP_TRUTH_TESTS=1 if you explicitly want to skip the expensive checks.
        if os.environ.get("SKIP_WARP_TRUTH_TESTS") == "1":
            self.skipTest("Skip Warp truth comparisons (SKIP_WARP_TRUTH_TESTS=1)")

        os.environ.setdefault("DARCY_FLOAT", "float64")

        try:
            import warp as wp
        except Exception as exc:
            self.skipTest(f"warp not available: {exc}")

        device = os.environ.get("WARP_DEVICE", "cuda:0")
        try:
            wp.get_device(device)
        except Exception as exc:
            self.skipTest(f"Warp device {device} unavailable: {exc}")

        try:
            wds = import_module(solver_module).WarpDarcySolver
        except Exception as exc:
            self.skipTest(f"{solver_module} unavailable: {exc}")

        repo_root = Path(__file__).resolve().parents[1]
        truth_dir = Path(
            os.environ.get(
                "MF6_TRUTH_DIR",
                str(repo_root.joinpath("DARCY_WARP_PACKAGE", "data", "mf6_truth_npz")),
            )
        )
        if tol_override is not None:
            tol = float(tol_override)
        else:
            tol = float(os.environ.get(tol_env, os.environ.get("WARP_HEAD_TOL", str(DEFAULT_WARP_HEAD_TOL))))

        if labels_override is not None:
            labels = [str(label).strip() for label in labels_override if str(label).strip()]
        else:
            cases_env = os.environ.get("WARP_TRUTH_CASES", "").strip()
            if cases_env:
                labels = [part.strip() for part in cases_env.split(",") if part.strip()]
            else:
                labels = sorted(GRID_CASES.keys())
        if not labels:
            raise ValueError("No cases selected for truth comparison.")

        n_solves_i = int(n_solves)
        if n_solves_i < 1:
            raise ValueError("n_solves must be >= 1.")

        kcycle_kwargs = {
            "max_cycles": 200,
            "nu_pre": 2,
            "nu_post": 2,
            "nu_coarse": 2,
            "omega": 0.7,
            "rel_tol": 1.0e-5,
            "abs_tol_min": 1.0e-5,
            "return_info": True,
            "max_levels": 6,
            "check_every_no": 1,
            "min_coarse_cells": 500,
        }
        if solver_kwargs:
            kcycle_kwargs.update(solver_kwargs)

        variants_env = os.environ.get("WARP_TRUTH_VARIANTS", "").strip().lower()
        if variants_env == "all":
            variants = [
                (False, False),
                (False, True),
                (True, False),
                (True, True),
            ]
        else:
            isotropic = _env_bool("WARP_TRUTH_ISOTROPIC", False)
            ghb = _env_bool("WARP_TRUTH_GHB", True)
            variants = [(isotropic, ghb)]

        for isotropic_flag, ghb_flag in variants:
            for label in labels:
                case_desc = f"{case_prefix}{label} ghb={ghb_flag} isotropic={isotropic_flag}".strip()
                cfg = GRID_CASES.get(label)
                self.assertIsNotNone(cfg, f"Unknown case label: {case_desc}")

                truth_path = truth_dir.joinpath(
                    f"mf6_truth_{label}_ghb_{ghb_flag}_t_isotropic_{isotropic_flag}.npz"
                )
                self.assertTrue(
                    truth_path.exists(),
                    f"Missing MF6 truth file {truth_path}. Run export_mf6_truth_npz.py.",
                )

                with np.load(truth_path) as truth:
                    heads_mf6 = np.asarray(truth["heads"], dtype=np.float64)
                    nx = int(truth["nx"])
                    ny = int(truth["ny"])
                    dx = float(truth["dx"])
                    ghb = bool(int(truth["ghb"]))
                    isotropic = bool(int(truth["t_isotropic"]))
                    t_isotropic_value = float(truth["t_isotropic_value"])
                    thickness = float(truth["thickness"])
                    width = float(truth["width"])
                    r_truth = float(truth["r_truth"])
                    seed = int(truth["seed"])

                self.assertEqual(nx, int(cfg["nx"]), f"{case_desc} nx mismatch")
                self.assertEqual(ny, int(cfg["ny"]), f"{case_desc} ny mismatch")
                self.assertEqual(ghb, ghb_flag, f"{case_desc} ghb mismatch")
                self.assertEqual(isotropic, isotropic_flag, f"{case_desc} isotropic mismatch")

                domain = _build_domain(nx=nx, ny=ny)
                dem = _build_dem(domain)
                active_mask = domain == 1

                if isotropic:
                    t_field = np.full_like(domain, t_isotropic_value, dtype=np.float64)
                else:
                    t_field = make_ugly_T_field(
                        nx=nx,
                        ny=ny,
                        domain=domain,
                        seed=seed,
                    )

                r_field = np.full_like(domain, r_truth, dtype=np.float64)

                with wds(
                    nx=nx,
                    ny=ny,
                    dx=dx,
                    device=device,
                    use_ghb=ghb,
                    solver_type="pcg",
                    aq_thickness=thickness,
                ) as solver:
                    solver.build_from_truth_inputs(
                        T_truth=t_field,
                        R_truth=r_field,
                        width=width,
                    )

                    head_warp = None
                    solve_info = None
                    head_guess = np.asarray(dem, dtype=np.float64)
                    for _ in range(n_solves_i):
                        solve_kwargs = dict(kcycle_kwargs)
                        solve_kwargs["initial_head"] = head_guess
                        head_warp, solve_info = solver.solve_multigrid_kcycle(
                            **solve_kwargs,
                        )
                        if hasattr(head_warp, "numpy"):
                            head_guess = np.asarray(head_warp.numpy(), dtype=np.float64)
                        else:
                            head_guess = np.asarray(head_warp, dtype=np.float64)

                    head_warp = head_guess

                wp.synchronize_device(device)
                gc.collect()
                wp.synchronize_device(device)

                self.assertEqual(
                    head_warp.shape,
                    heads_mf6.shape,
                    f"{case_desc} head shape mismatch",
                )

                abs_diff = np.abs(head_warp - heads_mf6)
                abs_diff = abs_diff[active_mask]
                self.assertTrue(
                    np.all(np.isfinite(abs_diff)),
                    f"{case_desc} non-finite diffs in active cells",
                )
                max_diff = float(abs_diff.max()) if abs_diff.size else 0.0
                self.assertLessEqual(
                    max_diff,
                    tol,
                    (
                        f"{case_desc} max abs diff {max_diff} exceeds tolerance {tol}; "
                        f"DARCY_FLOAT={os.environ.get('DARCY_FLOAT')}, "
                        f"converged={None if solve_info is None else solve_info.get('converged')}, "
                        f"cycles={None if solve_info is None else solve_info.get('n_cycles_used')}/"
                        f"{None if solve_info is None else solve_info.get('max_cycles')}, "
                        f"r_rms_end={None if solve_info is None else solve_info.get('r_rms_end')}, "
                        f"tol_abs={None if solve_info is None else solve_info.get('tol_abs')}, "
                        f"dh_rms_end={None if solve_info is None else solve_info.get('dh_rms_end')}, "
                        f"dh_max_end={None if solve_info is None else solve_info.get('dh_max_end')}"
                    ),
                )

    def test_warp_heads_within_tolerance(self) -> None:
        self._run_warp_truth_comparison(
            solver_module="DARCY_WARP_PACKAGE.warped_darcy",
            case_prefix="pcg ",
        )


if __name__ == "__main__":
    # When running this test file directly (e.g. from PyCharm Run), set
    # the environment guard so the expensive GPU truth comparisons run.
    # Use setdefault so an external environment can still override these.
    os.environ.setdefault("RUN_WARP_TRUTH_TESTS", "1")
    os.environ.setdefault("DARCY_FLOAT", "float64")
    # Call unittest main without trying to pass env vars via its args
    unittest.main()
