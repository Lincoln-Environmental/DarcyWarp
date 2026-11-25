"""Prevent solver entry points from injecting mutable per-call globals."""

import ast
from pathlib import Path


def test_no_runtime_namespace_injection():
    """Numerical backends must keep per-call state local."""
    root = Path(__file__).resolve().parents[1] / "DARCY_WARP_PACKAGE" / "solvers"
    for name in ("multigrid_kcycle.py", "picard_unconfined.py", "transient_unconfined.py", "device_picard_periods.py"):
        tree = ast.parse((root / name).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                assert node.func.id not in {"globals", "locals"}, name
