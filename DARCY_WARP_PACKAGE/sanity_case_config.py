from __future__ import annotations

GRID_CASES = {
    "100x100": {"nx": 100, "ny": 100},
    "100x250": {"nx": 100, "ny": 250},
    "400x400": {"nx": 400, "ny": 400},
    "100x1000": {"nx": 100, "ny": 1000},
    "250x1000": {"nx": 250, "ny": 1000},
    "1000x1001": {"nx": 1000, "ny": 1001},
    "2000x1000": {"nx": 2000, "ny": 1000},
    "3000x111": {"nx": 3000, "ny": 111},
    "3000x223": {"nx": 3000, "ny": 223},
    "3000x333": {"nx": 3000, "ny": 333},
    "3000x999": {"nx": 3000, "ny": 999},
    "3000x1999": {"nx": 3000, "ny": 1999},
    "3000x2999": {"nx": 3000, "ny": 2999},
    "3000x3000": {"nx": 3000, "ny": 3000},
}

DEFAULT_DX = 100.0
DEFAULT_R_TRUTH = 1.0e-4
DEFAULT_THICKNESS = 300.0
DEFAULT_GHB = True
DEFAULT_T_SEED = 123
DEFAULT_ISOTROPIC_T = 3000.0
