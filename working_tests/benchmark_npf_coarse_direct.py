"""Compare CPU or CUDA sparse direct solves with line relaxation on the NPF coarse grid."""

import argparse
from contextlib import nullcontext
from dataclasses import replace
from functools import partial
from hashlib import file_digest, sha256
from pathlib import Path
import json
import os
import resource
import subprocess
import sys
from time import perf_counter
from unittest.mock import patch


def assemble_coarse_matrix(*, level):
    import numpy as np
    from scipy.sparse import coo_matrix

    active = level["active"].numpy() != 0
    free = active & (level["bc_mask"].numpy() == 0)
    size = int(free.sum())
    if size == 0:
        raise ValueError("The coarsest grid has no free cells")

    indices = np.full(shape=free.shape, fill_value=-1, dtype=np.int32)
    indices[free] = np.arange(size, dtype=np.int32)
    diagonal = level["storage"].numpy().astype(dtype=np.float64)
    rows, cols, values = [], [], []
    for direction, face in enumerate(level["faces"]):
        axis = 2 - direction // 2
        lower, upper = [slice(None)] * 3, [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        source, target = (tuple(lower), tuple(upper)) if direction % 2 == 0 else (tuple(upper), tuple(lower))
        coefficients = face.numpy()[source].astype(dtype=np.float64)
        connected = active[target] & (coefficients > 0.0)
        diagonal[source] += np.where(connected, coefficients, 0.0)
        coupling = connected & free[source] & free[target]
        rows.append(indices[source][coupling])
        cols.append(indices[target][coupling])
        values.append(-coefficients[coupling])
    rows.append(indices[free])
    cols.append(indices[free])
    values.append(diagonal[free])
    matrix = coo_matrix(arg1=(np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(size, size)).tocsc()
    return free, matrix


def make_direct_workspace(*, base, statistics, single_coarse_pass):
    import numpy as np
    from scipy.sparse.linalg import splu

    class DirectCoarseWorkspace(base):
        def __init__(self, **options):
            self.initializing = True
            self.dirty = True
            self.factor = None
            self.matrix = None
            self.free_coarse = None
            super().__init__(**options)
            self.one_pass_controls = replace(self.controls, cycle_kind="v")
            self.initializing = False

        def refresh_storage_device(self, *, refresh_faces=False):
            super().refresh_storage_device(refresh_faces=refresh_faces)
            self.dirty = True

        def _cycle(self, *, index):
            if single_coarse_pass and not self.initializing and index == len(self.work_levels) - 2:
                # Use one exact coarse correction at its parent. All finer
                # levels retain the original K-cycle recurrence.
                controls = self.controls
                self.controls = self.one_pass_controls
                try:
                    super()._cycle(index=index)
                finally:
                    self.controls = controls
                return
            if self.initializing or index != len(self.work_levels) - 1:
                super()._cycle(index=index)
                return

            level = self.work_levels[index]
            started = perf_counter()
            if self.dirty:
                self._prepare_factor(level=level)
            rhs = level["rhs"].numpy()
            result = np.zeros_like(rhs)
            reduced_rhs = rhs[self.free_coarse].astype(dtype=np.float64)
            solution = self.factor.solve(reduced_rhs)
            result[self.free_coarse] = solution
            level["x"].assign(result)
            statistics["solves"] += 1
            statistics["coarse_wall_seconds"] += perf_counter() - started
            if self.check_factor:
                norm_rhs = np.linalg.norm(reduced_rhs, ord=np.inf)
                if norm_rhs > 0.0:
                    residual = reduced_rhs - self.matrix @ solution
                    self._residual(level=level)
                    device_residual = level["residual"].numpy()[self.free_coarse]
                    for key, value in (("lu_relative_residual_inf", residual),
                                       ("device_relative_residual_inf", device_residual)):
                        relative = float(np.linalg.norm(value, ord=np.inf) / norm_rhs)
                        statistics[key] = max(statistics[key], relative)
                    statistics["verified_factors"] += 1
                    self.check_factor = False

        def _prepare_factor(self, *, level):
            started = perf_counter()
            free, matrix = assemble_coarse_matrix(level=level)
            statistics["matrix_seconds"] += perf_counter() - started
            self.dirty = False
            if self.matrix is not None and np.array_equal(matrix.data, self.matrix.data) and (
                np.array_equal(matrix.indices, self.matrix.indices)
                and np.array_equal(matrix.indptr, self.matrix.indptr)
            ):
                return

            self.factor = None
            started = perf_counter()
            self.factor = splu(A=matrix, permc_spec="MMD_AT_PLUS_A")
            statistics["factor_seconds"] += perf_counter() - started
            self.free_coarse, self.matrix = free, matrix
            self.check_factor = True
            statistics["factorizations"] += 1
            statistics["shape"] = list(level["dim"])
            statistics["free_rows"] = matrix.shape[0]
            statistics["matrix_nnz"] = int(matrix.nnz)
            statistics["factor_nnz"] = max(statistics["factor_nnz"], int(self.factor.nnz))
            statistics["matrix_csc_bytes"] = sum(a.nbytes for a in (matrix.data, matrix.indices, matrix.indptr))
            statistics["factor_value_index_bytes_estimate"] = statistics["factor_nnz"] * (
                matrix.data.itemsize + matrix.indices.itemsize
            )

        def close(self):
            super().close()
            self.factor = self.matrix = self.free_coarse = None

    return DirectCoarseWorkspace


def build_case(*, size, variant, ratio, seed, reference_thickness, ghb):
    import numpy as np
    from benchmark_npf662_vs_mf6 import suite_settings
    from DARCY_WARP_PACKAGE.validation.npf662_benchmark_cases import (
        add_ghb_boundaries, generate_case, with_ugly_conductivity,
    )

    definition, = suite_settings(sizes=[size], variants=[variant])
    geology = dict(definition["geology"], layer_k=np.asarray(definition["geology"]["layer_k"]))
    case = generate_case(settings=geology, convertible=definition["convertible"])
    case = with_ugly_conductivity(case=case, seed=seed, reference_thickness=reference_thickness,
        layer_multipliers=geology["layer_k"] / geology["layer_k"][0], vertical_anisotropy=ratio)
    if ghb:
        case = add_ghb_boundaries(case=case, conductance_factor=1.0, draining_offset=-0.5, injecting_offset=1.0)
    return case


def observe_first_linear_solve(*, solve, diagnostics):
    def observed_solve(self, *, preconditioner, max_iterations, absolute_tolerance):
        result = solve(self, preconditioner=preconditioner, max_iterations=max_iterations,
                       absolute_tolerance=absolute_tolerance)
        if not diagnostics:
            diagnostics.update(
                hierarchy_shapes=[list(level["dim"]) for level in preconditioner.work_levels],
                graph_captures=preconditioner.graph_captures,
                graph_replays=preconditioner.graph_replays,
                first_linear_iterations=self.last_info["iterations"],
                first_linear_true_residual_history=self.last_info["history"],
                first_linear_target=self.last_info["target"],
            )
        return result

    return observed_solve


def run_case(*, case, args, profile, dt, ss, sy, factors, reference=None, native_accepted=False, gates=None):
    import warp as wp
    from DARCY_WARP_PACKAGE.solvers import npf662_kcycle as kcycle
    from DARCY_WARP_PACKAGE.solvers.npf662_fgmres import RestartedFGMRES3D
    from run_3d_sanity import measure

    controls = kcycle.NPF662KCycleControls(**profile["multigrid"])
    controls = replace(controls, levels=args.levels, use_cuda_graph=args.method in ("graph", "cudss"))
    solve = dict(profile["solve"], npf_newton_controls=profile["newton"])
    statistics = None
    workspace = kcycle.NPF662KCycleWorkspace
    if args.method in ("direct", "direct-single"):
        statistics = dict(solves=0, factorizations=0, matrix_seconds=0.0, factor_seconds=0.0,
            coarse_wall_seconds=0.0, factor_nnz=0, verified_factors=0,
            lu_relative_residual_inf=0.0, device_relative_residual_inf=0.0)
        workspace = make_direct_workspace(base=workspace, statistics=statistics,
            single_coarse_pass=args.method == "direct-single")
    if args.method == "cudss":
        from npf_coarse_cudss import make_cudss_workspace

        statistics = dict(analysis_seconds=0.0, assembly_seconds=0.0, factor_seconds=0.0,
            graph_capture_seconds=0.0, factorizations=0, change_checks=0)
        workspace = make_cudss_workspace(base=workspace, statistics=statistics, bridge_path=args.cudss_bridge)

    diagnostics = {}
    observed_solve = observe_first_linear_solve(solve=RestartedFGMRES3D.solve, diagnostics=diagnostics)
    with (
        patch.object(kcycle, "NPF662KCycleWorkspace", new=workspace),
        patch.object(RestartedFGMRES3D, "solve", new=observed_solve),
    ):
        run, heads = measure(case=case, family="npf", mode="transient" if args.periods else "steady",
            label="kcycle", reference=reference, native_accepted=native_accepted, device=args.device, profiles={},
            factors=factors, dt=dt, ss=ss, sy=sy, npf_options=solve, kcycle_controls=controls, gates=gates or {},
            progress_every=min(5, max(1, args.periods)), npf_history=bool(args.periods))
    device = wp.get_device(ident=args.device)
    run["cuda_pool_high_bytes"] = wp.get_mempool_used_mem_high(device=device) if device.is_cuda else None
    run["process_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    run["direct"] = statistics
    run["first_linear_solve"] = diagnostics
    run["krylov_iterations"] = sum(row["krylov"] for row in run["periods"])
    run["outer_iterations"] = sum(row["outer_iterations"] for row in run["periods"])
    run["max_head_residual_rms_m"] = max(row["physical"]["head_residual_rms"] for row in run["periods"])
    return run, heads


def load_saved_scenario(*, manifest_path, gates):
    import numpy as np
    from DARCY_WARP_PACKAGE.validation.npf662_reference import load_npf662_case
    from run_3d_sanity import sanity_reference_valid

    manifest = json.loads(s=manifest_path.read_text())
    inputs = manifest["inputs"]
    fingerprint = sha256(json.dumps(obj=inputs, sort_keys=True).encode()).hexdigest()
    if manifest["fingerprint"] != fingerprint or inputs["family"] != "npf":
        raise ValueError("Invalid saved NPF reference fingerprint or formulation")
    folder = manifest_path.parent
    for name in ("inputs.npz", "reference.npz"):
        with (folder / name).open(mode="rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != manifest["sha256"][name]:
                raise ValueError(f"Saved reference checksum mismatch for {folder / name}")
    case = load_npf662_case(path=folder / "inputs.npz")
    if case.digest() != inputs["case_sha256"]:
        raise ValueError("Saved NPF case digest mismatch")
    if inputs["mode"] not in ("steady", "transient"):
        raise ValueError("Saved NPF reference has an unsupported mode")
    transient = inputs["mode"] == "transient"
    factors = np.asarray(inputs["factors"], dtype=float)
    steps = len(factors) if transient else 1
    with np.load(file=folder / "reference.npz", allow_pickle=False) as saved:
        if transient:
            np.testing.assert_array_equal(actual=saved["times"], desired=inputs["dt"] * np.arange(1, steps + 1))
        reference = {key: saved[key] for key in ("heads", "fluxes", "ghb_inflows")}
    if reference["heads"].shape != (steps, *case.active.shape):
        raise ValueError("Saved NPF head history shape differs from the case and timestep schedule")
    if not sanity_reference_valid(info=manifest["native"], mode=inputs["mode"], gates=gates):
        raise ValueError("Saved MF6 reference failed the benchmark's native acceptance checks")
    return case, reference, manifest


def run_benchmark(*, args, root, profile_path, seed, reference_thickness, dt, ss, sy, source_factors, block_periods):
    import numpy as np
    import warp as wp
    from run_3d_sanity import save_json

    wp.init()
    profile = json.loads(s=profile_path.read_text())
    reference = manifest = None
    gates = dict(rmse=1e-3, maximum=1e-2, budget_percent=1.0)
    if args.mf6_manifest:
        case, reference, manifest = load_saved_scenario(manifest_path=args.mf6_manifest, gates=gates)
        inputs = manifest["inputs"]
        schedule = np.asarray(inputs["factors"], dtype=float)
        dt, ss, sy = (inputs[key] for key in ("dt", "ss", "sy"))
        args.periods = len(schedule) if inputs["mode"] == "transient" else 0
        args.size = case.active.shape[-1]
        args.variant = "convertible" if np.any(case.cell_types) else "confined"
        args.ghb = case.ghb_conductance is not None
    else:
        build = partial(build_case, variant=args.variant, ratio=args.ratio, seed=seed,
                        reference_thickness=reference_thickness, ghb=args.ghb)
        schedule = np.resize(a=np.repeat(a=source_factors, repeats=block_periods), new_shape=args.periods)
        case = build(size=args.size)
    allocation = (wp.ScopedAllocator(device=args.device, allocator=wp.CudaManagedAllocator())
                  if args.managed_memory else nullcontext())
    with allocation:
        run, heads = run_case(case=case, args=args, profile=profile, dt=dt, ss=ss, sy=sy, factors=schedule,
            reference=reference, native_accepted=manifest is not None, gates=gates)
    if manifest is not None:
        run["mf6"] = dict(manifest=str(args.mf6_manifest), fingerprint=manifest["fingerprint"],
            sha256=manifest["sha256"], engine_seconds=manifest["native_single_run_seconds"],
            inputs=manifest["inputs"], max_head_delta_m=max(row["head_max_m"] for row in run["periods"]))
    if args.compare_heads:
        target = np.load(file=args.compare_heads, allow_pickle=False)
        if target.shape != heads.shape:
            raise ValueError("Compared runs returned different head history shapes")
        run["max_head_delta_m"] = float(np.max(np.abs(heads - target)))
    run.update(method=args.method, size=args.size, variant=args.variant, levels=args.levels,
        managed_memory=args.managed_memory, periods_requested=args.periods, grid_shape=list(case.active.shape),
        anisotropy=args.ratio if manifest is None else None, ghb=args.ghb, case_sha256=case.digest(),
        dt=dt, ss=ss, sy=sy, source_factors=schedule.tolist(), controls=profile,
        device=wp.get_device(ident=args.device).name,
        warp=wp.__version__, profile=profile_path.name,
        revision=subprocess.check_output(args=["git", "rev-parse", "HEAD"], cwd=root, text=True).strip())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(file=args.output.with_suffix(".npy"), arr=heads, allow_pickle=False)
    save_json(path=args.output.with_suffix(".json"), value=run)
    print(f'{args.method}: {run["public_solve_seconds"]:.4f} s, {run["krylov_iterations"]} Krylov iterations, '
          f'physical checks {"passed" if run["gpu_physical_accepted"] else "failed"}', flush=True)
    return 0 if (run["accepted"] if manifest is not None else run["gpu_physical_accepted"]) else 1


def parse_arguments(*, defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("graph", "eager", "direct", "direct-single", "cudss"), required=True,
        help="direct-single uses one LU correction at the parent of the coarsest level")
    parser.add_argument("--cudss-bridge", type=Path, help="Shared library built from npf_coarse_cudss.c")
    parser.add_argument("--managed-memory", action="store_true",
        help="Use CUDA managed memory to allow allocations beyond device capacity")
    parser.add_argument("--size", type=int, default=defaults["size"])
    parser.add_argument("--variant", choices=("confined", "convertible"), default=defaults["variant"])
    parser.add_argument("--levels", type=int, default=defaults["levels"])
    parser.add_argument("--periods", type=int, default=defaults["periods"], help="Zero selects steady flow")
    parser.add_argument("--ratio", type=float, default=defaults["ratio"], help="Horizontal to vertical K ratio")
    parser.add_argument("--ghb", action=argparse.BooleanOptionalAction, default=defaults["ghb"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--output", type=Path, required=True, help="Output stem for JSON and heads")
    parser.add_argument("--compare-heads", type=Path, help="Saved .npy history from a paired run")
    parser.add_argument("--mf6-manifest", type=Path, help="Replay the exact inputs and schedule from a validated MF6 cache")
    args = parser.parse_args()
    if args.periods < 0 or args.levels < 2 or args.size < 2:
        parser.error("Requires nonnegative periods and at least two levels and grid cells per axis")
    if (args.method in ("graph", "cudss") or args.managed_memory) and args.device == "cpu":
        parser.error("Graph capture, cuDSS and managed memory require a CUDA device")
    if args.method == "cudss":
        if args.cudss_bridge is None or not args.cudss_bridge.is_file():
            parser.error("The cuDSS method requires an existing --cudss-bridge shared library")
        args.cudss_bridge = args.cudss_bridge.resolve()
    return args


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    os.environ["DARCY_FLOAT"] = "float64"
    defaults = dict(size=500, variant="confined", levels=5, periods=0, ratio=100.0, device="cuda:0", ghb=True)
    arguments = parse_arguments(defaults=defaults)
    profile_path = root / "working_tests" / "npf_ghb_five_level_controls.json"
    seed, reference_thickness = 42, 100.0
    dt, ss, sy = 7.0, 1e-5, 0.15
    source_factors, block_periods = (1.0, 1.5, 0.25, 1.0), 5
    raise SystemExit(run_benchmark(args=arguments, root=root, profile_path=profile_path, seed=seed,
        reference_thickness=reference_thickness, dt=dt, ss=ss, sy=sy, source_factors=source_factors,
        block_periods=block_periods))
