# SPDX-License-Identifier: AGPL-3.0-or-later
"""Counters executed on CPU/CUDA, including inside captured graph replays."""
import warp as wp


@wp.kernel
def increment_work(counter: wp.array(dtype=wp.int64), slot: int):
    wp.atomic_add(counter, slot, wp.int64(1))
