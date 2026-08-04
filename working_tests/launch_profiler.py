# SPDX-License-Identifier: AGPL-3.0-only
"""Shared launch/synchronization instrumentation for campaign benchmarks."""

from __future__ import annotations

import warp as wp


class NullCapture:
    """ScopedCapture stand-in: runs the cycle eagerly, yields graph=None.

    Used only for instrumented profile runs so per-launch accounting sees
    every kernel (CUDA-graph replay bypasses wp.launch).
    """

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    graph = None


class LaunchProfiler:
    """Per-kernel GPU timing (CUDA events) + host sync/readback counting.

    Note: in the eager host-bound regime, per-launch event windows measure
    submission cadence rather than pure kernel GPU time; use nsys for
    authoritative per-kernel GPU times.
    """

    def __init__(self):
        self._records = []  # (name, dim, ev0, ev1)
        self.sync_calls = 0
        self.numpy_readbacks = 0
        self.numpy_bytes = 0
        self._patched = []

    def _wrap_launch(self, orig):
        def launch_wrapper(kernel, dim, inputs=[], **kwargs):
            name = getattr(kernel, "key", None) or str(kernel)
            e0 = wp.Event(enable_timing=True)
            e1 = wp.Event(enable_timing=True)
            wp.record_event(e0)
            result = orig(kernel, dim, inputs=inputs, **kwargs)
            wp.record_event(e1)
            self._records.append((name, tuple(dim) if hasattr(dim, "__len__") else (dim,), e0, e1))
            return result
        return launch_wrapper

    def _wrap_sync(self, orig):
        def sync_wrapper(*a, **k):
            self.sync_calls += 1
            return orig(*a, **k)
        return sync_wrapper

    def _wrap_numpy(self, orig):
        def numpy_wrapper(self_arr, *a, **k):
            self.numpy_readbacks += 1
            try:
                self.numpy_bytes += int(self_arr.size) * int(self_arr.dtype.itemsize)
            except Exception:
                pass
            return orig(self_arr, *a, **k)
        return numpy_wrapper

    def __enter__(self):
        for obj, attr, wrap in (
            (wp, "launch", self._wrap_launch),
            (wp, "synchronize", self._wrap_sync),
            (wp, "synchronize_device", self._wrap_sync),
            (wp.array, "numpy", self._wrap_numpy),
        ):
            orig = getattr(obj, attr)
            self._patched.append((obj, attr, orig))
            setattr(obj, attr, wrap(orig))
        # Disable CUDA-graph capture so the profiled cycle runs eagerly.
        self._patched.append((wp, "ScopedCapture", wp.ScopedCapture))
        wp.ScopedCapture = lambda *a, **k: NullCapture()
        return self

    def __exit__(self, *exc):
        for obj, attr, orig in reversed(self._patched):
            setattr(obj, attr, orig)
        self._patched.clear()
        return False

    def report(self):
        wp.synchronize()
        per_kernel = {}
        total_gpu_ms = 0.0
        for name, dim, e0, e1 in self._records:
            ms = wp.get_event_elapsed_time(e0, e1, synchronize=False)
            key = f"{name}@{'x'.join(str(d) for d in dim)}"
            entry = per_kernel.setdefault(key, {"count": 0, "gpu_ms": 0.0})
            entry["count"] += 1
            entry["gpu_ms"] += ms
            total_gpu_ms += ms
        return {
            "launches_total": len(self._records),
            "gpu_busy_ms": total_gpu_ms,
            "per_kernel": dict(sorted(per_kernel.items(), key=lambda kv: -kv[1]["gpu_ms"])),
            "host_sync_calls": self.sync_calls,
            "numpy_readbacks": self.numpy_readbacks,
            "numpy_readback_bytes": self.numpy_bytes,
        }
