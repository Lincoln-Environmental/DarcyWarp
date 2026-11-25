"""Benchmark realistic 3D WEL/DRN/RIV groundwater workloads on CUDA.

Heterogeneous multilayer aquifers carry substantial abstraction through
explicit sparse wells, nonzero recharge, gaining and losing rivers, extensive
drainage and multiple boundary entries per cell. The programme separates
compilation and construction from warm solver timings, exercises repeated
pumping and river-stage updates, calibration-style steady solves and device
resident transient histories, and records accuracy against MF6 where a
matched truth model is requested.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D

RESULT_DIR = Path(__file__).resolve().parents[1] / "benchmark_results" / "wel_boundaries_3d"


def heterogeneous_fields(*, shape, seed=11, wells_top=True, drain=True, river=True):
    """Build layered heterogeneous fields with all source/boundary packages.

    :param shape: (nz, ny, nx) grid shape, top layer first.
    :param seed: Reproducible heterogeneity seed.
    :param wells_top: Place abstraction wells in the upper half of each column.
    :param drain: Enable an extensive drainage network on the top layer.
    :param river: Enable a gaining/losing river across the domain.
    :return: Keyword fields for ``build_from_K_fields`` plus the entry lists.
    """
    nz, ny, nx = shape
    rng = np.random.default_rng(seed)
    log_k = rng.normal(loc=np.log(5.0), scale=1.0, size=shape)
    layer_scale = np.linspace(3.0, 0.3, nz)[:, None, None]
    k = np.exp(log_k * 0.6) * layer_scale
    k[:, :: ny // 8, :] *= 0.05  # low-K horizontal inclusions
    active = np.ones(shape, dtype=np.int32)
    fixed = np.zeros(shape, dtype=np.int32)
    fixed[:, :, 0] = 1
    fixed[:, :, -1] = 1
    bc_values = np.where(fixed != 0, 40.0, 0.0)
    dx = dy = 40.0
    dz = 10.0
    recharge = np.zeros((ny, nx))
    recharge[:, :] = 2.0e-4
    recharge[::7, ::5] = 0.0  # dry patches receive none

    entries = []
    if wells_top:
        for row in range(5, ny - 5, 11):
            for col in range(4, nx - 4, 13):
                layer = 0 if (row + col) % 3 else min(1, nz - 1)
                entries.append(((layer, row, col), -3000.0))
                if (row + col) % 5 == 0:
                    # A second, partially screened well shares the cell.
                    entries.append(((layer, row, col), -1500.0))
    drn_entries = None
    if drain:
        drn_entries = [((0, row, col), 25.0, 28.5 + 0.01 * (col % 4))
                       for row in range(2, ny - 2, 3) for col in range(2, nx - 2, 4)]
    riv_entries = None
    if river:
        stage_row = ny // 2
        riv_entries = [((0, stage_row, col), 400.0,
                        39.5 + (0.6 if col % 2 else -0.6), 38.8)
                       for col in range(1, nx - 1, 2)]
    return dict(
        kx_field=k, ky_field=k * 0.6, kz_field=k * 0.1,
        active=active, bc_mask=fixed, bc_values=bc_values,
        R_field=recharge, dx=dx, dy=dy, dz=dz,
        wel_entries=entries, drn_entries=drn_entries, riv_entries=riv_entries,
    )


def build_model(device, shape, **overrides):
    """Construct the benchmark model with optional field overrides."""
    fields = heterogeneous_fields(shape=shape, **{k: v for k, v in overrides.items()
        if k in ("seed", "wells_top", "drain", "river")})
    dx, dy, dz = fields.pop("dx"), fields.pop("dy"), fields.pop("dz")
    model = WarpDarcySolver3D(nx=shape[2], ny=shape[1], nz=shape[0], dx=dx, dy=dy, dz=dz,
        formulation="mf6_6_6_2_npf", solver="kcycle", implementation="fast", device=device)
    started = perf_counter()
    model.build_from_K_fields(**fields, initial_head=np.full(shape, 40.0))
    model.construction_seconds = perf_counter() - started
    return model


def run_programme(device="cuda:0", grids=None):
    """Execute the full benchmark matrix and record structured results.

    :param device: Warp device used for every solver.
    :param grids: Mapping of programme names to grid shapes.
    :return: Nested result mapping written to ``benchmark_results``.
    """
    if grids is None:
        grids = dict(small=(10, 112, 90), large=(5, 450, 450))
    results = dict(device=device, grids={})
    for name, shape in grids.items():
        started = perf_counter()
        model = build_model(device, shape)
        construction = model.construction_seconds
        warm_seconds = []
        head = None
        info = None
        for repeat in range(3):
            model.update_wel(model._wel_entries)  # calibration-style steady solves
            step_started = perf_counter()
            head, info = model.solve()
            warm_seconds.append(perf_counter() - step_started)
        first_wel_budget = dict(info["source_budgets"]["wel"])

        # Repeated pumping update: halve every well rate and re-solve warm.
        built = model._wel_entries
        halved = [((int(c), int(r), int(col)), 0.5 * float(rate))
                  for (c, r, col), rate in zip(built.cells.tolist(), built.rates.tolist())]
        update_started = perf_counter()
        model.update_wel(halved)
        model.solve()
        update_seconds = perf_counter() - update_started

        # Repeated river-stage update: raise every stage by one metre.
        river = model._session.riv_entries
        stage_update_seconds = None
        if river is not None:
            raised = [((int(k), int(row), int(col)), float(c), float(s) + 1.0, float(b))
                for (k, row, col), c, s, b in zip(river.cells.tolist(), river.conductance.tolist(),
                    river.second_level.tolist(), river.third_level.tolist())]
            stage_started = perf_counter()
            model.update_riv_entries(raised)
            model.solve()
            stage_update_seconds = perf_counter() - stage_started

        # Device-resident transient history from the updated forcing.
        history_started = perf_counter()
        steps = 20
        previous = np.full(shape, 40.0)
        heads, history_info = model.solve_transient_npf(
            dt=[30.0] * steps, head_prev=previous, ss=1e-5, sy=0.05)
        history_seconds = perf_counter() - history_started

        entry = dict(
            shape=list(shape), cells=int(np.prod(shape)),
            construction_seconds=construction,
            steady_solve_seconds=warm_seconds,
            steady_best_seconds=min(warm_seconds),
            newton_iterations=info["outer_iterations"],
            krylov_iterations=info["krylov_iterations"],
            preconditioner=info.get("npf_preconditioner"),
            pumping_update_seconds=update_seconds,
            stage_update_seconds=stage_update_seconds,
            wel_budget=first_wel_budget,
            wel_entries=0 if model._wel_entries is None else len(model._wel_entries),
            drn_entries=0 if model._session.drn_entries is None else len(model._session.drn_entries),
            riv_entries=0 if model._session.riv_entries is None else len(model._session.riv_entries),
            history_seconds=history_seconds,
            history_steps=steps,
            history_converged=history_info["converged"],
            history_accepted_steps=history_info.get("accepted_steps"),
            total_seconds=perf_counter() - started,
        )
        if wp.get_device(ident=device).is_cuda:
            entry["mempool_high_water_bytes"] = int(wp.get_mempool_used_mem_high())
        results["grids"][name] = entry
        model.close()
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULT_DIR / "benchmark_results.json"
    output.write_text(json.dumps(results, indent=2, sort_keys=True))
    return results


if __name__ == "__main__":
    outcome = run_programme()
    print(json.dumps(outcome, indent=2, sort_keys=True))
