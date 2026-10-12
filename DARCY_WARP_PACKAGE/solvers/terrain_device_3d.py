"""Device-resident 3D terrain Picard driver (2D fast-unconfined contract).

The nonlinear outer loop keeps the total head, transformed coefficients,
secant storage, RHS and corrections on the device: per outer iteration it
refreshes saturation, the packed 13-column face operator and the storage
diagonals with the terrain device kernels, extracts the seven-point
principal part for the multigrid preconditioner, rebuilds the row-scaled
RHS, and runs restarted GMRES in correction form (A·delta = b − A·h with
homogeneous prescribed-head constraints) under an adaptive inexact inner
controller modelled on the production 2D driver: fixed-work restart-cycle
blocks, dual-residual contraction classification with rollback of
diverging blocks, Eisenstat-Walker-style forcing targets, and acceptance
of usable inexact corrections so a stiff frozen system cannot burn its
whole Krylov budget every outer iteration. Damped clipped updates and a
snapshot rollback of rejected corrections run on device; only a few
scalars cross to the host per outer iteration. Final acceptance is
independently reconstructed on the host from the completed physical state
by the caller (the session), never from the frozen inner residual.

The host-driven path in terrain_fast_3d.TerrainOperatorSession3D.solve
remains the reference implementation and the default; this driver is
selected explicitly with device_picard=True.
"""
from dataclasses import dataclass

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.solvers.terrain_device_kernels import (
    terrain_saturation_kernel, terrain_face_coefficients_kernel,
    terrain_secant_storage_kernel, terrain_rhs_kernel, terrain_relaxed_update_kernel,
    terrain_copy_kernel, terrain_dual_check_kernel, terrain_combine_partials_kernel,
    terrain_relative_head_kernel, terrain_extract_principal_kernel, BLOCK_SIZE)


@dataclass
class DevicePicardControls:
    """Adaptive inner and nonlinear controls (2D production defaults)."""
    initial_block_cycles: int = 2          # restart cycles per block
    min_block_cycles: int = 1
    max_block_cycles: int = 4
    min_total_cycles: int = 1
    eta_initial: float = 0.05
    eta_min: float = 0.005
    eta_max: float = 0.10
    eta_gamma: float = 0.25
    eta_power: float = 1.5
    good_contraction_ratio: float = 0.40
    weak_contraction_ratio: float = 0.90
    stall_contraction_ratio: float = 0.9995
    divergence_ratio: float = 1.10
    stall_patience: int = 4
    minimum_usable_reduction: float = 0.10
    max_head_change_per_outer: float = 25.0
    block_stall_cycles: int = 40           # per-block hard Krylov ceiling factor


class DeviceTerrainState:
    """Persistent device buffers for the terrain Picard loop."""
    def __init__(self, *, session, device, shape):
        import warp as wpx
        n = int(np.prod(shape))
        nz, ny, nx = shape
        ncol = ny*nx
        self.n = n
        self.device = device
        grid = session.grid
        metrics = session.metrics

        def f64(array):
            return wp.array(np.ascontiguousarray(array, dtype=np.float64).ravel(),
                dtype=wp.float64, device=device)

        def i32(array):
            return wp.array(np.ascontiguousarray(array, dtype=np.int32).ravel(),
                dtype=wp.int32, device=device)

        # static geometry
        self.bottoms = f64(grid.bottoms)
        self.thickness = f64(grid.thickness)
        self.zx_interfaces = f64(metrics.zx_interfaces)
        self.zy_interfaces = f64(metrics.zy_interfaces)
        self.zx_centre = f64(metrics.zx)
        self.zy_centre = f64(metrics.zy)
        zxf_full = np.zeros(shape); zxf_full[:, :, :-1] = metrics.zxf
        zyf_full = np.zeros(shape); zyf_full[:, :-1, :] = metrics.zyf
        self.zxf = f64(zxf_full)
        self.zyf = f64(zyf_full)
        self.active = i32(grid.active)
        self.kx = f64(session.kx)
        self.ky = f64(session.ky)
        self.kz = f64(session.kz)
        # dynamic fields
        self.head = f64(np.zeros(shape))
        self.previous = f64(np.zeros(shape))
        self.snapshot = f64(np.zeros(shape))
        self.candidate = f64(np.zeros(shape))
        self.fraction = wp.zeros(n, dtype=wp.float64, device=device)
        self.coefficients = wp.zeros((n, 13), dtype=wp.float64, device=device)
        self.storage_diag = wp.zeros(n, dtype=wp.float64, device=device)
        self.storage_change = wp.zeros(n, dtype=wp.float64, device=device)
        self.rhs = wp.zeros(n, dtype=wp.float64, device=device)
        self.residual = wp.zeros(n, dtype=wp.float64, device=device)
        self.scale = wp.zeros(n, dtype=wp.float64, device=device)
        self.correction_rhs = wp.zeros(n, dtype=wp.float64, device=device)
        self.free = wp.zeros(n, dtype=wp.int32, device=device)
        self.fixed = i32(session.bc_mask)
        self.bc_values = f64(session.bc_values)
        blocks = (n + BLOCK_SIZE - 1)//BLOCK_SIZE
        self.partials = wp.zeros((blocks, 6), dtype=wp.float64, device=device)
        self.blocks = blocks
        self.combined = wp.zeros(2, dtype=wp.float64, device=device)
        # level-0 principal arrays mirroring the multigrid face layout
        shape3 = (nz, ny, nx)
        self.faces = tuple(wp.zeros(shape3, dtype=wp.float64, device=device) for _ in range(6))
        self.diag = wp.zeros(shape3, dtype=wp.float64, device=device)
        self.principal_storage = wp.zeros(shape3, dtype=wp.float64, device=device)
        self.level0_ready = False

    def upload_static(self, *, source, sy, ss, contacts_boundary, contact_rhs, free):
        self.source = wp.array(np.ascontiguousarray(source, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.sy = wp.array(np.ascontiguousarray(sy, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.ss = wp.array(np.ascontiguousarray(ss, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.boundary = wp.array(np.ascontiguousarray(contacts_boundary, dtype=np.float64),
            dtype=wp.float64, device=self.device)
        self.contact_rhs = wp.array(np.ascontiguousarray(contact_rhs, dtype=np.float64).ravel(),
            dtype=wp.float64, device=self.device)
        self.free.assign(np.ascontiguousarray(free.ravel(), dtype=np.int32))

    def update_k_fields(self, *, kx, ky, kz):
        self.kx.assign(np.ascontiguousarray(kx, dtype=np.float64).ravel())
        self.ky.assign(np.ascontiguousarray(ky, dtype=np.float64).ravel())
        self.kz.assign(np.ascontiguousarray(kz, dtype=np.float64).ravel())

    def close(self):
        for name in ("head", "previous", "snapshot", "candidate", "fraction",
                     "coefficients", "storage_diag", "storage_change", "rhs",
                     "residual", "scale", "correction_rhs", "partials", "combined",
                     "faces", "diag", "principal_storage", "source", "sy", "ss",
                     "boundary", "contact_rhs", "bottoms", "thickness",
                     "zx_interfaces", "zy_interfaces", "zx_centre", "zy_centre",
                     "zxf", "zyf", "active", "fixed", "bc_values", "kx", "ky", "kz",
                     "free"):
            setattr(self, name, None)


class DeviceTerrainPicard:
    """One solve invocation of the device-resident Picard loop."""
    def __init__(self, *, session, state, workspace, device, controls: DevicePicardControls,
                 datum, restart, max_cycles_total, levels, min_coarse_n):
        self.session = session
        self.state = state
        self.workspace = workspace
        self.device = device
        self.controls = controls
        self.datum = datum
        self.restart = restart
        self.max_cycles_total = max_cycles_total
        self.levels = levels
        self.min_coarse_n = min_coarse_n

    def refresh_operator(self):
        s = self.state
        wp.launch(terrain_saturation_kernel, dim=s.n,
            inputs=[s.head, s.bottoms, s.thickness, wp.float64(self.session.min_sat),
                    None, None], outputs=[None, s.fraction], device=self.device) if False else None
        # (explicit launch below; keep the placeholder out)
        wp.launch(terrain_saturation_kernel, dim=s.n,
            inputs=[s.head, s.bottoms, s.thickness, wp.float64(self.session.min_sat)],
            outputs=[s.true_sat, s.fraction] if hasattr(s, "true_sat") else [s.fraction],
            device=self.device) if False else None
