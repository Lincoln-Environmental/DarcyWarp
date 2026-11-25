# SPDX-License-Identifier: AGPL-3.0-or-later
"""Shared launch/synchronization instrumentation for campaign benchmarks."""

from __future__ import annotations

import warp as wp


class NullCapture:
    """ScopedCapture stand-in: runs the cycle eagerly, yields graph=None.

    Used only for instrumented profile runs so per-launch accounting sees
    every kernel (CUDA-graph replay bypasses wp.launch).
    """

    def __enter__(self):
        """Return this eager capture context."""
        return self

    def __exit__(self, *exc):
        """Leave the capture context without suppressing exceptions.

        :param exc: Context manager exception type, value and traceback.
        :returns: False, so errors propagate.
        """
        return False

    graph = None


def _eager_capture(*args, **kwargs):
    """Provide a no-op capture context that runs launches eagerly.

    :param args: Positional arguments accepted by Warp's ScopedCapture.
    :param kwargs: Keyword arguments accepted by Warp's ScopedCapture.
    :returns: Eager capture context; the capture options are not used.
    """
    return NullCapture()


def _descending_gpu_duration(item):
    """Sort kernel records by descending elapsed GPU time.

    :param item: Pair of kernel label and accumulated timing dictionary.
    :returns: Negative elapsed GPU milliseconds.
    """
    return -item[1]["gpu_ms"]


class LaunchProfiler:
    """Per-kernel GPU timing (CUDA events) + host sync/readback counting.

    Note: in the eager host-bound regime, per-launch event windows measure
    submission cadence rather than pure kernel GPU time; use nsys for
    authoritative per-kernel GPU times.
    """

    def __init__(self):
        """Initialise an empty record of launches, synchronisations and readbacks."""
        self._records = []  # (name, dim, ev0, ev1)
        self.sync_calls = 0
        self.numpy_readbacks = 0
        self.numpy_bytes = 0
        self._patched = []

    def _wrap_launch(self, orig):
        """Wrap Warp's launch function to record device timing events.

        :param orig: Original launch callable.
        :returns: Instrumented launch callable.
        """
        def launch_wrapper(kernel, dim, inputs=None, **kwargs):
            """Record events around one kernel launch.

            :param kernel: Warp kernel.
            :param dim: Launch dimensions.
            :param inputs: Optional positional kernel arguments.
            :param kwargs: Remaining keyword arguments to the original launch.
            :returns: Result of the underlying launch.
            """
            if inputs is None:
                inputs = []
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
        """Count calls to a Warp synchronisation function.

        :param orig: Original synchronisation callable.
        :returns: Instrumented synchronisation callable.
        """
        def sync_wrapper(*a, **k):
            """Count and forward one synchronisation call.

            :param a: Original positional arguments.
            :param k: Original keyword arguments.
            :returns: Result of the underlying synchronisation.
            """
            self.sync_calls += 1
            return orig(*a, **k)
        return sync_wrapper

    def _wrap_numpy(self, orig):
        """Count array readbacks and estimate their byte volume.

        :param orig: Original Warp array-to-NumPy callable.
        :returns: Instrumented array method.
        """
        def numpy_wrapper(self_arr, *a, **k):
            """Count and forward one array readback.

            :param self_arr: Source Warp array.
            :param a: Original positional arguments.
            :param k: Original keyword arguments.
            :returns: Result of the underlying conversion.
            """
            self.numpy_readbacks += 1
            try:
                self.numpy_bytes += int(self_arr.size) * int(self_arr.dtype.itemsize)
            except Exception:
                pass
            return orig(self_arr, *a, **k)
        return numpy_wrapper

    def __enter__(self):
        """Install timing wrappers and force eager kernel execution."""
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
        wp.ScopedCapture = _eager_capture
        return self

    def __exit__(self, *exc):
        """Restore the original Warp callables after profiling.

        :param exc: Context manager exception type, value and traceback.
        :returns: False, so errors propagate.
        """
        for obj, attr, orig in reversed(self._patched):
            setattr(obj, attr, orig)
        self._patched.clear()
        return False

    def report(self):
        """Synchronise timing events and report grouped durations and readbacks.

        :returns: Dictionary of activity counts, times in milliseconds,
            host synchronisations and estimated readback bytes.
        """
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
            "per_kernel": dict(sorted(per_kernel.items(), key=_descending_gpu_duration)),
            "host_sync_calls": self.sync_calls,
            "numpy_readbacks": self.numpy_readbacks,
            "numpy_readback_bytes": self.numpy_bytes,
        }
