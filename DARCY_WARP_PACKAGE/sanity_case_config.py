"""Shared spatial cases for steady and transient sanity suites.

The catalog deliberately contains geometry only.  Temporal schedules and
physics belong to the individual benchmark runners, so adding a grid here
does not silently change a selected simulation suite.
"""

from __future__ import annotations


def _grid(nx: int, ny: int, *, manual_only: bool = False) -> dict[str, int | bool]:
    return {"nx": int(nx), "ny": int(ny), "manual_only": bool(manual_only)}


# Complete spatial registry.  The near-capacity cases remain discoverable for
SPATIAL_GRID_CASES: dict[str, dict[str, int | bool]] = {
    "100x100": _grid(100, 100),
    "100x250": _grid(100, 250),
    "400x400": _grid(400, 400),
    "500x500": _grid(500, 500),
    "100x1000": _grid(100, 1000),
    "250x1000": _grid(250, 1000),
    "1000x1000": _grid(1000, 1000),
    "1000x1001": _grid(1000, 1001, manual_only=True),
    "2000x1000": _grid(2000, 1000, manual_only=True),
    "3000x111": _grid(3000, 111, manual_only=True),
    "3000x223": _grid(3000, 223, manual_only=True),
    "3000x333": _grid(3000, 333, manual_only=True),
    "3000x999": _grid(3000, 999, manual_only=True),
    "3000x1999": _grid(3000, 1999, manual_only=True),
    "3000x2999": _grid(3000, 2999, manual_only=True),
    "3000x3000": _grid(3000, 3000, manual_only=True),
}
# explicit/manual runs but are never selected by an automatic suite.


# Automatic suite membership.  GRID_CASES is the automatic steady view and
# deliberately excludes every manual_only grid (including the historically
# selected large cases 1000x1001/2000x1000/3000x111/223/333/999 and the
# near-capacity 3000x1999 case): those remain in SPATIAL_GRID_CASES for
# explicit/manual runs but are never selected by an automatic suite.
# Copies prevent callers from mutating the canonical catalog.
STEADY_SANITY_LABELS = (
    "100x100",
    "100x250",
    "400x400",
    "100x1000",
    "250x1000",
)

TRANSIENT_SMOKE_LABELS = ("100x100", "100x250")
TRANSIENT_SHAPE_LABELS = ("100x1000", "1000x1001", "3000x111")
TRANSIENT_PRODUCTION_LABELS = ("500x500", "1000x1000")
TRANSIENT_SCALE_LABELS = ("2000x1000", "3000x999")
TRANSIENT_CAPACITY_LABELS = ("3000x1999", "3000x2999", "3000x3000")


def _selected_grid_cases(labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    return {
        label: {
            "nx": int(SPATIAL_GRID_CASES[label]["nx"]),
            "ny": int(SPATIAL_GRID_CASES[label]["ny"]),
        }
        for label in labels
    }


GRID_CASES = _selected_grid_cases(STEADY_SANITY_LABELS)

DEFAULT_DX = 100.0
DEFAULT_R_TRUTH = 1.0e-4
DEFAULT_THICKNESS = 300.0
DEFAULT_GHB = True
DEFAULT_ISOTROPIC = False
DEFAULT_T_SEED = 123
DEFAULT_ISOTROPIC_T = 3000.0


__all__ = [
    "DEFAULT_DX",
    "DEFAULT_GHB",
    "DEFAULT_ISOTROPIC",
    "DEFAULT_ISOTROPIC_T",
    "DEFAULT_R_TRUTH",
    "DEFAULT_T_SEED",
    "DEFAULT_THICKNESS",
    "GRID_CASES",
    "SPATIAL_GRID_CASES",
    "STEADY_SANITY_LABELS",
    "TRANSIENT_CAPACITY_LABELS",
    "TRANSIENT_PRODUCTION_LABELS",
    "TRANSIENT_SCALE_LABELS",
    "TRANSIENT_SHAPE_LABELS",
    "TRANSIENT_SMOKE_LABELS",
]
