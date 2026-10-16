"""Persistent fixed-work A0 inverse using the fast 3D face-array K-cycle."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.config import WP_FLOAT
from DARCY_WARP_PACKAGE.kernels_3d import (
    vertical_line_relaxation_7point_kernel, copy_field_3d_kernel,
    restrict_blockavg_axes_3d_kernel, prolong_bilinear_axes_3d_kernel,
    add_correction_3d_kernel,
)
from .fast_3d import _build_levels, _upload_levels
from .face_kernels_3d_f64 import (
    BLOCK_SIZE_3D, residual_partials_3d_f64_kernel,
    dot_partials_3d_f64_kernel, combine_partials_3d_f64_kernel,
    apply_a_dot_partials_3d_f64_kernel, compute_safe_alpha_3d_kernel,
    axpy_active_device_scalar_3d_kernel, mask_correction_3d_kernel,
)


@dataclass(frozen=True)
class NPF662KCycleControls:
    levels: int = 5
    min_coarse_n: int = 2
    pre_sweeps: int = 2
    post_sweeps: int = 2
    coarse_sweeps: int = 20
    cycles: int = 1
    omega: float = 0.8
    use_cuda_graph: bool = True

    def validate(self):
        for name in ('levels', 'min_coarse_n', 'coarse_sweeps', 'cycles'):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('pre_sweeps', 'post_sweeps'):
            if int(getattr(self, name)) != getattr(self, name) or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        if not np.isfinite(self.omega) or not 0. < self.omega <= 1.:
            raise ValueError('Line relaxation omega must be in (0,1]')


@wp.kernel
def scaled_rhs(v: wp.array(dtype=wp.float64), diagonal: wp.array(dtype=wp.float64),
               free: wp.array(dtype=wp.int32), out: wp.array(dtype=wp.float64)):
    i = wp.tid()
    out[i] = wp.float64(0.)
    if free[i] != 0:
        out[i] = v[i] * diagonal[i]


class NPF662KCycleWorkspace:
    def __init__(self, *, faces, active, fixed, device, controls):
        controls.validate()
        if WP_FLOAT != wp.float64:
            raise ValueError('NPF K-cycle requires DARCY_FLOAT=float64')
        start = perf_counter()
        self.device = str(device)
        self.controls = controls
        self.closed = False
        self.snapshots = tuple(np.array(value, copy=True) for value in (*faces, active, fixed))
        zeros = np.zeros(shape=active.shape, dtype=np.float64)
        host = _build_levels(faces=faces, active=active, bc_mask=fixed, bc_values=zeros,
                             rhs=zeros, storage_diag=zeros, max_levels=controls.levels,
                             min_coarse_n=controls.min_coarse_n)
        self.levels = _upload_levels(host_levels=host, device=self.device)
        for level in self.levels:
            level['c_prime'] = wp.zeros(shape=level['dim'], dtype=wp.float64, device=self.device)
            level['d_prime'] = wp.zeros(shape=level['dim'], dtype=wp.float64, device=self.device)
        self.graph = None
        self.applications = self.cycles = self.graph_captures = self.graph_replays = 0
        self.graph_failed = False
        self.graph_error = None
        self.device_seconds = 0.
        self.cuda = wp.get_device(ident=self.device).is_cuda
        self.start_event = wp.Event(device=self.device, enable_timing=True) if self.cuda else None
        self.end_event = wp.Event(device=self.device, enable_timing=True) if self.cuda else None
        # Compile before capture. Graph creation is outside FGMRES applications.
        self._enqueue()
        if wp.get_device(ident=self.device).is_cuda:
            wp.synchronize_device(device=self.device)
            if controls.use_cuda_graph:
                try:
                    with wp.ScopedCapture(device=self.device) as capture:
                        self._enqueue()
                    self.graph = capture.graph
                    self.graph_captures = int(self.graph is not None)
                except Exception as error:
                    self.graph_failed = True
                    self.graph_error = str(error)
        self.construction_seconds = perf_counter() - start

    def compatible(self, *, faces, active, fixed, device, controls):
        return (not self.closed and self.device == str(device) and self.controls == controls
                and all(np.array_equal(saved, current) for saved, current
                        in zip(self.snapshots, (*faces, active, fixed))))

    def _smooth(self, *, level, sweeps):
        x, work = level['x'], level['work']
        for unused in range(sweeps):
            wp.launch(kernel=vertical_line_relaxation_7point_kernel, dim=level['nx'] * level['ny'],
                inputs=[*level['faces'], level['active'], level['bc_mask'], level['storage'],
                        level['rhs'], x, level['bc_values'], self.controls.omega,
                        level['c_prime'], level['d_prime'], level['nx'], level['ny'], level['nz'], work],
                device=self.device)
            x, work = work, x
        if x is not level['x']:
            wp.copy(dest=level['x'], src=x)

    def _residual(self, *, level):
        level['partials_a'].zero_()
        wp.launch(kernel=residual_partials_3d_f64_kernel, dim=level['n_cells'],
            inputs=[level['x'], level['rhs'], *level['faces'], level['storage'], level['active'],
                    level['bc_mask'], level['residual'], level['partials_a'],
                    level['nx'], level['ny'], level['nz'], BLOCK_SIZE_3D], device=self.device)

    def _energy_add(self, *, level, base, correction, rhs):
        level['partials_a'].zero_()
        level['partials_b'].zero_()
        wp.launch(kernel=dot_partials_3d_f64_kernel, dim=level['n_cells'],
            inputs=[rhs, correction, level['active'], level['bc_mask'], level['partials_a'],
                    level['nx'], level['ny'], level['nz'], BLOCK_SIZE_3D], device=self.device)
        wp.launch(kernel=apply_a_dot_partials_3d_f64_kernel, dim=level['n_cells'],
            inputs=[correction, *level['faces'], level['diag'], level['active'], level['bc_mask'],
                    level['partials_b'], level['nx'], level['ny'], level['nz'], BLOCK_SIZE_3D], device=self.device)
        for partial, scalar in (('partials_a', 'num'), ('partials_b', 'den')):
            wp.launch(kernel=combine_partials_3d_f64_kernel, dim=1,
                inputs=[level[partial], level[scalar], level['n_partials']], device=self.device)
        wp.launch(kernel=compute_safe_alpha_3d_kernel, dim=1,
                  inputs=[level['num'], level['den'], level['alpha']], device=self.device)
        wp.launch(kernel=axpy_active_device_scalar_3d_kernel, dim=level['dim'],
            inputs=[base, correction, level['active'], level['bc_mask'], level['alpha'],
                    level['nx'], level['ny'], level['nz']], device=self.device)

    def _cycle(self, *, index):
        level = self.levels[index]
        self._smooth(level=level, sweeps=self.controls.pre_sweeps)
        if index == len(self.levels) - 1:
            self._smooth(level=level, sweeps=self.controls.coarse_sweeps)
            return
        self._residual(level=level)
        coarse = self.levels[index + 1]
        wp.launch(kernel=restrict_blockavg_axes_3d_kernel, dim=coarse['dim'],
            inputs=[level['residual'], level['active'], level['bc_mask'], coarse['rhs'],
                    level['nx'], level['ny'], level['nz'], coarse['nx'], coarse['ny'], coarse['nz'],
                    level['coarsen_y'], level['coarsen_x']], device=self.device)
        coarse['x'].zero_()
        self._cycle(index=index + 1)
        wp.copy(dest=coarse['z1'], src=coarse['x'])
        self._residual(level=coarse)
        wp.copy(dest=coarse['rhs'], src=coarse['residual'])
        coarse['x'].zero_()
        self._cycle(index=index + 1)
        self._energy_add(level=coarse, base=coarse['z1'], correction=coarse['x'], rhs=coarse['rhs'])
        wp.launch(kernel=prolong_bilinear_axes_3d_kernel, dim=level['dim'],
            inputs=[coarse['z1'], level['correction'], level['nx'], level['ny'], level['nz'],
                    coarse['nx'], coarse['ny'], coarse['nz'], level['coarsen_y'], level['coarsen_x']], device=self.device)
        if level['project_correction']:
            wp.launch(kernel=mask_correction_3d_kernel, dim=level['dim'],
                      inputs=[level['correction'], level['active'], level['bc_mask']], device=self.device)
            self._energy_add(level=level, base=level['x'], correction=level['correction'], rhs=level['residual'])
        else:
            wp.launch(kernel=add_correction_3d_kernel, dim=level['dim'],
                inputs=[level['x'], level['correction'], level['active'], level['bc_mask'], level['bc_values'],
                        level['nx'], level['ny'], level['nz']], device=self.device)
        self._smooth(level=level, sweeps=self.controls.post_sweeps)

    def _enqueue(self):
        self.levels[0]['x'].zero_()
        for unused in range(self.controls.cycles):
            self._cycle(index=0)

    def apply(self, *, vector, out, diagonal, free):
        if self.closed:
            raise RuntimeError('K-cycle workspace is closed')
        fine = self.levels[0]
        if self.cuda:wp.record_event(event=self.start_event)
        wp.launch(kernel=scaled_rhs, dim=vector.size,
                  inputs=[vector, diagonal, free, fine['rhs'].flatten()], device=self.device)
        if self.graph is not None:
            wp.capture_launch(graph=self.graph)
            self.graph_replays += 1
        else:
            self._enqueue()
        wp.copy(dest=out, src=fine['x'].flatten())
        if self.cuda:wp.record_event(event=self.end_event)
        self.applications += 1
        self.cycles += self.controls.cycles

    def collect_timing(self):
        """Called after Arnoldi's existing norm read; adds no synchronization."""
        if self.cuda:
            self.device_seconds += .001 * wp.get_event_elapsed_time(
                start_event=self.start_event, end_event=self.end_event, synchronize=False)

    def allocated_bytes(self):
        buffers = {}
        for level in self.levels:
            for value in level.values():
                for array in value if isinstance(value, tuple) else (value,):
                    if isinstance(array, wp.array):buffers[array.ptr] = array.capacity
        return sum(buffers.values())

    def close(self):
        if not self.closed:
            self.graph = None
            self.levels = []
            self.snapshots = ()
            self.start_event = self.end_event = None
            self.closed = True
