# SPDX-License-Identifier: AGPL-3.0-only
"""CPU parity checks for the experimental FP64 3D face-array kernels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.solvers.face_kernels_3d_f64 import (
    apply_A_faces_3d_f64_kernel,
    combine_partials_3d_f64_kernel,
    residual_partials_3d_f64_kernel,
)


def _prepare_warp_cpu() -> None:
    if wp.config.kernel_cache_dir is None:
        wp.config.kernel_cache_dir = str(Path("/tmp") / "darcywarp_warp_kernel_cache")
    wp.init()


def _array(value: np.ndarray, dtype):
    return wp.array(value, dtype=dtype, device="cpu")


def test_face_operator_matches_two_cell_7point_stencil():
    _prepare_warp_cpu()
    shape = (1, 1, 2)
    x = np.array([[[0.0, 1.0]]], dtype=np.float64)
    rhs = np.zeros(shape, dtype=np.float64)
    zero = np.zeros(shape, dtype=np.float64)
    tx_p = zero.copy()
    tx_m = zero.copy()
    tx_p[0, 0, 0] = 1.0
    tx_m[0, 0, 1] = 1.0
    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)

    out = wp.zeros(shape, dtype=wp.float64, device="cpu")
    common = [
        _array(x, wp.float64),
        _array(tx_p, wp.float64),
        _array(tx_m, wp.float64),
        _array(zero, wp.float64),
        _array(zero, wp.float64),
        _array(zero, wp.float64),
        _array(zero, wp.float64),
        _array(zero, wp.float64),
        _array(active, wp.int32),
        _array(bc_mask, wp.int32),
    ]
    wp.launch(
        kernel=apply_A_faces_3d_f64_kernel,
        dim=2,
        inputs=[*common, out, 2, 1, 1],
        device="cpu",
    )
    np.testing.assert_allclose(out.numpy(), [[[-1.0, 1.0]]])

    residual = wp.zeros(shape, dtype=wp.float64, device="cpu")
    partials = wp.zeros(1, dtype=wp.float64, device="cpu")
    residual_inputs = [
        _array(x, wp.float64),
        _array(rhs, wp.float64),
        _array(tx_p, wp.float64),
        _array(tx_m, wp.float64),
        _array(zero, wp.float64),
        _array(zero, wp.float64),
        _array(zero, wp.float64),
        _array(zero, wp.float64),
        _array(zero, wp.float64),
        _array(active, wp.int32),
        _array(bc_mask, wp.int32),
        residual,
        partials,
        2,
        1,
        1,
        32,
    ]
    wp.launch(
        kernel=residual_partials_3d_f64_kernel,
        dim=2,
        inputs=residual_inputs,
        device="cpu",
    )
    total = wp.zeros(1, dtype=wp.float64, device="cpu")
    wp.launch(
        kernel=combine_partials_3d_f64_kernel,
        dim=1,
        inputs=[partials, total, 1],
        device="cpu",
    )
    np.testing.assert_allclose(residual.numpy(), [[[1.0, -1.0]]])
    np.testing.assert_allclose(total.numpy(), [2.0])


def test_face_operator_ignores_conductance_to_inactive_cell():
    _prepare_warp_cpu()
    shape = (1, 1, 2)
    x = np.array([[[2.0, 9.0]]], dtype=np.float64)
    zero = np.zeros(shape, dtype=np.float64)
    tx_p = zero.copy()
    tx_p[0, 0, 0] = 1.0
    active = np.array([[[1, 0]]], dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    out = wp.zeros(shape, dtype=wp.float64, device="cpu")
    wp.launch(
        kernel=apply_A_faces_3d_f64_kernel,
        dim=2,
        inputs=[
            _array(x, wp.float64),
            _array(tx_p, wp.float64),
            _array(zero, wp.float64),
            _array(zero, wp.float64),
            _array(zero, wp.float64),
            _array(zero, wp.float64),
            _array(zero, wp.float64),
            _array(zero, wp.float64),
            _array(active, wp.int32),
            _array(bc_mask, wp.int32),
            out,
            2,
            1,
            1,
        ],
        device="cpu",
    )
    # A genuinely null free equation evaluates to zero, never to head.
    np.testing.assert_allclose(out.numpy(), [[[0.0, 0.0]]])
