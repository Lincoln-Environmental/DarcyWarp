# SPDX-License-Identifier: AGPL-3.0-only
"""Public 2D Darcy model facade.

The compatibility implementation remains in :mod:`warped_darcy` during the
migration, while new callers can depend on this stable model-facing module.
"""

from __future__ import annotations

from DARCY_WARP_PACKAGE.warped_darcy import WarpDarcySolver

__all__ = ["WarpDarcySolver"]
