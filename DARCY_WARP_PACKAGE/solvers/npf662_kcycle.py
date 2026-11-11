"""Persistent A0 plus optional storage inverse using the 3D face-array K-cycle."""
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
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f'{name} must be a positive integer')
        for name in ('pre_sweeps', 'post_sweeps'):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
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


@wp.kernel
def coarsen_faces_device(
    xp: wp.array3d(dtype=wp.float64), xm: wp.array3d(dtype=wp.float64),
    yp: wp.array3d(dtype=wp.float64), ym: wp.array3d(dtype=wp.float64),
    zp: wp.array3d(dtype=wp.float64), zm: wp.array3d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32), fixed: wp.array3d(dtype=wp.int32),
    cxp: wp.array3d(dtype=wp.float64), cxm: wp.array3d(dtype=wp.float64),
    cyp: wp.array3d(dtype=wp.float64), cym: wp.array3d(dtype=wp.float64),
    czp: wp.array3d(dtype=wp.float64), czm: wp.array3d(dtype=wp.float64),
    springs: wp.array3d(dtype=wp.float64), fy: int, fx: int):
    k,j,i = wp.tid()
    spring = wp.float64(0.)
    for axis in range(6):
        total = wp.float64(0.)
        for y in range(fy):
            for x in range(fx):
                jj = j*fy+y
                ii = i*fx+x
                if jj < active.shape[1] and ii < active.shape[2]:
                    if active[k,jj,ii] != 0 and fixed[k,jj,ii] == 0:
                        nk = k
                        nj = jj
                        ni = ii
                        value = wp.float64(0.)
                        if axis == 0:
                            ni += 1
                            value = xp[k,jj,ii]
                        elif axis == 1:
                            ni -= 1
                            value = xm[k,jj,ii]
                        elif axis == 2:
                            nj += 1
                            value = yp[k,jj,ii]
                        elif axis == 3:
                            nj -= 1
                            value = ym[k,jj,ii]
                        elif axis == 4:
                            nk += 1
                            value = zp[k,jj,ii]
                        else:
                            nk -= 1
                            value = zm[k,jj,ii]
                        if nk >= 0 and nk < active.shape[0] and nj >= 0 and nj < active.shape[1] and ni >= 0 and ni < active.shape[2]:
                            if active[nk,nj,ni] != 0:
                                if fixed[nk,nj,ni] != 0:
                                    spring += value
                                elif nk != k or nj/fy != j or ni/fx != i:
                                    total += value
        total = total/wp.float64(fy*fx)
        if axis == 0: cxp[k,j,i] = total
        elif axis == 1: cxm[k,j,i] = total
        elif axis == 2: cyp[k,j,i] = total
        elif axis == 3: cym[k,j,i] = total
        elif axis == 4: czp[k,j,i] = total
        else: czm[k,j,i] = total
    springs[k,j,i] = spring/wp.float64(fy*fx)


@wp.kernel
def coarsen_storage_device(fine: wp.array3d(dtype=wp.float64),
    active_f: wp.array3d(dtype=wp.int32), fixed_f: wp.array3d(dtype=wp.int32),
    coarse: wp.array3d(dtype=wp.float64), springs: wp.array3d(dtype=wp.float64),
    active_c: wp.array3d(dtype=wp.int32), fixed_c: wp.array3d(dtype=wp.int32),
    fy: int, fx: int):
    k, j, i = wp.tid()
    value = wp.float64(0.)
    if active_c[k,j,i] != 0 and fixed_c[k,j,i] == 0:
        for y in range(fy):
            for x in range(fx):
                jj = j*fy+y
                ii = i*fx+x
                if jj < fine.shape[1] and ii < fine.shape[2]:
                    if active_f[k,jj,ii] != 0 and fixed_f[k,jj,ii] == 0:
                        value += fine[k,jj,ii]
        value = value/wp.float64(fy*fx)+springs[k,j,i]
    coarse[k,j,i] = value


@wp.kernel
def refresh_storage_diagonal(storage: wp.array3d(dtype=wp.float64),
    xp: wp.array3d(dtype=wp.float64), xm: wp.array3d(dtype=wp.float64),
    yp: wp.array3d(dtype=wp.float64), ym: wp.array3d(dtype=wp.float64),
    zp: wp.array3d(dtype=wp.float64), zm: wp.array3d(dtype=wp.float64),
    active: wp.array3d(dtype=wp.int32), diagonal: wp.array3d(dtype=wp.float64)):
    k,j,i = wp.tid()
    value = wp.float64(1.)
    if active[k,j,i] != 0:
        value = storage[k,j,i]+(xp[k,j,i]+xm[k,j,i]+yp[k,j,i]+ym[k,j,i]+zp[k,j,i]+zm[k,j,i])
    diagonal[k,j,i] = value


class NPF662KCycleWorkspace:
    def __init__(self, *, faces, active, fixed, device, controls, storage_diag=None):
        controls.validate()
        if WP_FLOAT != wp.float64:
            raise ValueError('NPF K-cycle requires DARCY_FLOAT=float64')
        start = perf_counter()
        self.device = str(device)
        self.controls = controls
        self.closed = False
        zeros = np.zeros(shape=active.shape, dtype=np.float64)
        storage = zeros if storage_diag is None else np.asarray(storage_diag, dtype=float)
        if storage.shape != active.shape or np.any(~np.isfinite(storage)) or np.any(storage < 0.):
            raise ValueError('Finite nonnegative per-cell storage diagonal required')
        self.snapshots = tuple(np.array(value, copy=True) for value in (*faces, active, fixed, storage))
        host = _build_levels(faces=faces, active=active, bc_mask=fixed, bc_values=zeros,
                             rhs=zeros, storage_diag=storage, max_levels=controls.levels,
                             min_coarse_n=controls.min_coarse_n)
        self.levels = _upload_levels(host_levels=host, device=self.device)
        for level, host_level in zip(self.levels, host):
            level['boundary_springs'] = wp.array(data=host_level.get('boundary_springs',
                np.zeros(shape=level['dim'])), dtype=wp.float64, device=self.device)
            level['c_prime'] = wp.zeros(shape=level['dim'], dtype=wp.float64, device=self.device)
            level['d_prime'] = wp.zeros(shape=level['dim'], dtype=wp.float64, device=self.device)
        self.graph = None
        self.applications = self.cycles = self.graph_captures = self.graph_replays = 0
        self.graph_failed = False
        self.graph_error = None
        self.value_refresh_count = 0
        self.device_storage_refresh_count = 0
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

    def compatible(self, *, faces, active, fixed, device, controls, storage_diag=None):
        storage = np.zeros_like(active, dtype=float) if storage_diag is None else storage_diag
        return (not self.closed and self.device == str(device) and self.controls == controls
                and all(np.array_equal(saved, current) for saved, current
                        in zip(self.snapshots, (*faces, active, fixed, storage))))

    def can_refresh(self, *, faces, active, fixed, device, controls):
        """Use the 2D fast-face structure/value split; K values may change."""
        return (not self.closed and self.device == str(device) and self.controls == controls
            and np.array_equal(self.snapshots[6], active) and np.array_equal(self.snapshots[7], fixed)
            and all(np.array_equal(old > 0., new > 0.) for old, new in zip(self.snapshots[:6], faces)))

    def refresh(self, *, faces, active, fixed, storage_diag=None):
        zeros = np.zeros(shape=active.shape, dtype=float)
        storage = zeros if storage_diag is None else np.asarray(storage_diag, dtype=float)
        if storage.shape != active.shape or np.any(~np.isfinite(storage)) or np.any(storage < 0.):
            raise ValueError('Finite nonnegative per-cell storage diagonal required')
        faces_changed = any(not np.array_equal(old, new) for old, new in zip(self.snapshots[:6], faces))
        if faces_changed:
            for target, values in zip(self.levels[0]['faces'], faces):
                target.assign(values)
        self.levels[0]['storage'].assign(storage)
        self.refresh_storage_device(refresh_faces=faces_changed)
        self.snapshots = tuple(np.array(value, copy=True) for value in (*faces, active, fixed, storage))
        self.value_refresh_count += 1

    def refresh_storage_device(self, *, refresh_faces=False):
        for index, level in enumerate(self.levels):
            if index:
                fine = self.levels[index-1]
                if refresh_faces:
                    wp.launch(kernel=coarsen_faces_device, dim=level['dim'], inputs=[
                        *fine['faces'], fine['active'], fine['bc_mask'], *level['faces'],
                        level['boundary_springs'], fine['coarsen_y'], fine['coarsen_x']], device=self.device)
                wp.launch(kernel=coarsen_storage_device, dim=level['dim'], inputs=[
                    fine['storage'], fine['active'], fine['bc_mask'], level['storage'],
                    level['boundary_springs'], level['active'], level['bc_mask'],
                    fine['coarsen_y'], fine['coarsen_x']], device=self.device)
            wp.launch(kernel=refresh_storage_diagonal, dim=level['dim'], inputs=[
                level['storage'], *level['faces'], level['active'], level['diag']], device=self.device)
        self.device_storage_refresh_count += 1

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
