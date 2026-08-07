# SPDX-License-Identifier: AGPL-3.0-only
"""Compatibility exports for the river-loss case-study builder.

The reusable implementation now lives under ``DARCY_WARP_PACKAGE.case_studies``.
This module remains so existing imports and the historical ``python -m`` entry
point continue to work.
"""

from __future__ import annotations

from DARCY_WARP_PACKAGE.case_studies.river_loss_cross_section import *  # noqa: F401,F403


def build_parser():
    """Return the historical CLI parser from the working-test runner."""

    from working_tests.run_river_loss_cross_section import build_parser as _build_parser

    return _build_parser()


def write_results(*args, **kwargs):
    """Write sweep output through the working-test reporting helper."""

    from working_tests.run_river_loss_cross_section import write_results as _write_results

    return _write_results(*args, **kwargs)


def main() -> None:
    """Forward the historical module entry point to the experiment runner."""

    from working_tests.run_river_loss_cross_section import main as _main

    _main()


if __name__ == "__main__":
    main()
