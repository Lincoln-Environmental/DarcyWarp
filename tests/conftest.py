# SPDX-License-Identifier: AGPL-3.0-or-later
"""Explicit CPU-only execution and honest CUDA coverage reporting."""
import os
import pytest


def pytest_configure(config):
    if os.environ.get('DARCYWARP_TEST_CPU_ONLY') == '1':
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def pytest_collection_modifyitems(config, items):
    import warp as wp
    if wp.is_cuda_available():
        return
    for item in items:
        parameters = getattr(getattr(item, 'callspec', None), 'params', {})
        for name, value in parameters.items():
            if isinstance(value, str) and value.startswith('cuda') and name in ('device', 'backend', 'dev', 'terrain_test_device'):
                item.add_marker(pytest.mark.skip(reason='CUDA unavailable or explicitly disabled; GPU coverage not executed'))
