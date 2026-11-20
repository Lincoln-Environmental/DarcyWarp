"""GPU-resident cuDSS coarse solver for benchmark_npf_coarse_direct.py."""

import ctypes as ct
from dataclasses import replace
from time import perf_counter

import numpy as np
import warp as wp


@wp.kernel
def assemble_values(
    slots: wp.array2d(dtype=wp.int32), columns: wp.array(dtype=wp.int32),
    xp: wp.array(dtype=wp.float32), xm: wp.array(dtype=wp.float32),
    yp: wp.array(dtype=wp.float32), ym: wp.array(dtype=wp.float32),
    zp: wp.array(dtype=wp.float32), zm: wp.array(dtype=wp.float32),
    storage: wp.array(dtype=wp.float32), active: wp.array(dtype=wp.int32),
    fixed: wp.array(dtype=wp.int32), values: wp.array(dtype=wp.float64),
    changed: wp.array(dtype=wp.int32),
):
    row = wp.tid()
    free = active[row] != 0 and fixed[row] == 0
    diagonal = wp.float64(1.0)
    if free:
        diagonal = wp.max(wp.float64(storage[row]), wp.float64(0.0))
    row_changed = int(0)
    for direction in range(6):
        slot = slots[row, direction]
        if slot >= 0:
            neighbor = columns[slot]
            off_diagonal = wp.float64(0.0)
            if free and active[neighbor] != 0:
                face = xp[row]
                if direction == 1:
                    face = xm[row]
                elif direction == 2:
                    face = yp[row]
                elif direction == 3:
                    face = ym[row]
                elif direction == 4:
                    face = zp[row]
                elif direction == 5:
                    face = zm[row]
                coefficient = wp.max(wp.float64(face), wp.float64(0.0))
                diagonal += coefficient
                if fixed[neighbor] == 0:
                    off_diagonal = -coefficient
            if values[slot] != off_diagonal:
                row_changed = 1
            values[slot] = off_diagonal
    diagonal_slot = slots[row, 6]
    if values[diagonal_slot] != diagonal:
        row_changed = 1
    values[diagonal_slot] = diagonal
    if row_changed:
        wp.atomic_max(changed, 0, row_changed)


@wp.kernel
def pack_rhs(source: wp.array(dtype=wp.float32), active: wp.array(dtype=wp.int32),
             fixed: wp.array(dtype=wp.int32), rhs: wp.array(dtype=wp.float64)):
    row = wp.tid()
    rhs[row] = wp.float64(0.0)
    if active[row] != 0 and fixed[row] == 0:
        rhs[row] = wp.float64(source[row])


@wp.kernel
def unpack_solution(solution: wp.array(dtype=wp.float64), result: wp.array(dtype=wp.float32)):
    row = wp.tid()
    result[row] = wp.float32(solution[row])


@wp.kernel
def max_residual(residual: wp.array(dtype=wp.float32), maximum: wp.array(dtype=wp.float32)):
    row = wp.tid()
    value = wp.abs(residual[row])
    if not wp.isfinite(value):
        value = wp.float32(3.402823e38)
    wp.atomic_max(maximum, 0, value)


def build_csr_pattern(*, shape):
    nz, ny, nx = shape
    k, j, i = np.indices(dimensions=shape, dtype=np.int32).reshape(3, -1)
    n = i.size
    row = np.arange(n, dtype=np.int32)
    neighbors_present = np.stack(arrays=(k > 0, j > 0, i > 0, np.ones(n, dtype=bool),
                                          i + 1 < nx, j + 1 < ny, k + 1 < nz), axis=1)
    strides = np.array(object=(-nx * ny, -nx, -1, 0, 1, nx, nx * ny), dtype=np.int32)
    columns = (row[:, None] + strides)[neighbors_present]
    offsets = np.empty(shape=n + 1, dtype=np.int32)
    offsets[0] = 0
    np.cumsum(a=neighbors_present.sum(axis=1), dtype=np.int32, out=offsets[1:])
    slots = np.full(shape=neighbors_present.shape, fill_value=-1, dtype=np.int32)
    slots[neighbors_present] = np.arange(columns.size, dtype=np.int32)
    return offsets, columns, np.ascontiguousarray(slots[:, (4, 2, 5, 1, 6, 0, 3)])


def load_bridge(*, path):
    bridge = ct.CDLL(name=str(path))
    pointer, integer = ct.c_void_p, ct.c_int64
    definitions = dict(
        coarse_create=(ct.c_int, [ct.POINTER(pointer), pointer, integer, integer, *([pointer] * 5)]),
        coarse_analyze=(ct.c_int, [pointer]),
        coarse_factor=(ct.c_int, [pointer, ct.c_int]),
        coarse_solve=(ct.c_int, [pointer]),
        coarse_get_info=(ct.c_int, [pointer, ct.POINTER(ct.c_int)]),
        coarse_peak_bytes=(ct.c_size_t, [pointer]),
        coarse_destroy=(None, [pointer]),
    )
    for name, (result, arguments) in definitions.items():
        function = getattr(bridge, name)
        function.restype = result
        function.argtypes = arguments
    return bridge


def check_status(*, status, operation):
    if status:
        raise RuntimeError(f"cuDSS {operation} failed with status {status}")


class CoarseCudaDirect:
    def __init__(self, *, level, device, bridge):
        self.device = device
        self.stream = wp.get_stream(device=device)
        self.bridge = bridge
        self.level = level
        offsets, columns, slots = build_csr_pattern(shape=level["dim"])
        self.offsets = wp.array(data=offsets, dtype=wp.int32, device=device)
        self.columns = wp.array(data=columns, dtype=wp.int32, device=device)
        self.slots = wp.array(data=slots, dtype=wp.int32, device=device)
        self.values = wp.zeros(shape=columns.size, dtype=wp.float64, device=device)
        self.rhs = wp.zeros(shape=level["n_cells"], dtype=wp.float64, device=device)
        self.solution = wp.zeros_like(self.rhs)
        self.changed = wp.zeros(shape=1, dtype=wp.int32, device=device)
        self.maximum = wp.zeros(shape=1, dtype=wp.float32, device=device)
        self.faces = tuple(face.flatten() for face in level["faces"])
        self.storage, self.active, self.fixed, self.source, self.result = (
            level[name].flatten() for name in ("storage", "active", "bc_mask", "rhs", "x")
        )
        self.handle = ct.c_void_p()
        with wp.ScopedDevice(self.device):
            status = bridge.coarse_create(ct.byref(self.handle), self.stream.cuda_stream, self.rhs.size,
                self.columns.size, self.offsets.ptr, self.columns.ptr, self.values.ptr,
                self.rhs.ptr, self.solution.ptr)
        check_status(status=status, operation="create")

    def assemble(self):
        self.changed.zero_()
        wp.launch(kernel=assemble_values, dim=self.rhs.size, device=self.device,
            inputs=[self.slots, self.columns, *self.faces, self.storage, self.active, self.fixed,
                    self.values, self.changed])
        return bool(self.changed.numpy()[0])

    def analyze(self):
        with wp.ScopedDevice(self.device):
            check_status(status=self.bridge.coarse_analyze(self.handle), operation="analysis")
            wp.synchronize_stream(stream_or_device=self.stream)

    def factor(self, *, refactor):
        with wp.ScopedDevice(self.device):
            check_status(status=self.bridge.coarse_factor(self.handle, int(refactor)), operation="factorization")
            wp.synchronize_stream(stream_or_device=self.stream)
            info = ct.c_int()
            check_status(status=self.bridge.coarse_get_info(self.handle, ct.byref(info)), operation="factor info")
            if info.value:
                raise RuntimeError(f"cuDSS factorization returned numerical info {info.value}")

    def enqueue_solve(self):
        wp.launch(kernel=pack_rhs, dim=self.rhs.size, device=self.device,
                  inputs=[self.source, self.active, self.fixed, self.rhs])
        with wp.ScopedDevice(self.device):
            check_status(status=self.bridge.coarse_solve(self.handle), operation="solve")
        wp.launch(kernel=unpack_solution, dim=self.rhs.size, device=self.device,
                  inputs=[self.solution, self.result])

    def residual_max(self):
        self.maximum.zero_()
        wp.launch(kernel=max_residual, dim=self.rhs.size, device=self.device,
                  inputs=[self.level["residual"].flatten(), self.maximum])
        return float(self.maximum.numpy()[0])

    def close(self):
        if self.handle:
            with wp.ScopedDevice(self.device):
                wp.synchronize_stream(stream_or_device=self.stream)
                self.bridge.coarse_destroy(self.handle)
            self.handle = None


def make_cudss_workspace(*, base, statistics, bridge_path):
    bridge = load_bridge(path=bridge_path)

    class CudssWorkspace(base):
        def __init__(self, *, controls, **options):
            if controls.preconditioner_precision != "float32":
                raise ValueError("This benchmark expects the FP32 NPF preconditioner profile")
            self.initializing = True
            self.coarse_solver = None
            self.dirty = True
            self.leaf_solves = 0
            super().__init__(controls=replace(controls, use_cuda_graph=False), **options)
            self.controls = controls
            self.one_pass_controls = replace(controls, cycle_kind="v")
            self.initializing = False

        def refresh_storage_device(self, *, refresh_faces=False):
            super().refresh_storage_device(refresh_faces=refresh_faces)
            self.dirty = True

        def _cycle(self, *, index):
            last = len(self.work_levels) - 1
            if self.initializing or index < last - 1:
                super()._cycle(index=index)
            elif index == last - 1:
                controls = self.controls
                self.controls = self.one_pass_controls
                try:
                    super()._cycle(index=index)
                finally:
                    self.controls = controls
            else:
                self.coarse_solver.enqueue_solve()
                self.leaf_solves += 1

        def _prepare(self):
            level = self.work_levels[-1]
            started = perf_counter()
            first = self.coarse_solver is None
            if first:
                self.coarse_solver = CoarseCudaDirect(level=level, device=self.device, bridge=bridge)
            solver = self.coarse_solver
            changed = solver.assemble()
            statistics["assembly_seconds"] += perf_counter() - started
            statistics["change_checks"] += 1
            if first:
                started = perf_counter()
                solver.analyze()
                statistics["analysis_seconds"] += perf_counter() - started
                statistics["coarse_rows"] = solver.rhs.size
                statistics["matrix_nnz"] = solver.columns.size
            if first or changed:
                self.graph = None
                started = perf_counter()
                solver.factor(refactor=not first)
                statistics["factor_seconds"] += perf_counter() - started
                statistics["factorizations"] += 1
                if first:
                    # Unit RHS independently checked by Warp's original coarse stencil.
                    level["rhs"].fill_(1.0)
                    solver.enqueue_solve()
                    self._residual(level=level)
                    statistics["unit_rhs_coarse_residual_inf"] = solver.residual_max()
                started = perf_counter()
                self.leaf_solves = 0
                with wp.ScopedCapture(stream=solver.stream) as capture:
                    self._enqueue()
                self.graph = capture.graph
                if self.graph is None:
                    raise RuntimeError("cuDSS K-cycle CUDA graph is empty")
                self.graph_captures += 1
                statistics["graph_capture_seconds"] += perf_counter() - started
                statistics["coarse_solves_per_application"] = self.leaf_solves
            self.dirty = False

        def apply(self, *, vector, out, diagonal, free):
            if self.dirty:
                self._prepare()
            super().apply(vector=vector, out=out, diagonal=diagonal, free=free)

        def close(self):
            if self.coarse_solver is not None:
                self.graph = None
                statistics["cudss_allocator_peak_bytes"] = bridge.coarse_peak_bytes(self.coarse_solver.handle)
                statistics["graph_replays"] = self.graph_replays
                statistics["device_apply_seconds"] = self.device_seconds
                self.coarse_solver.close()
                self.coarse_solver = None
            super().close()

    return CudssWorkspace
