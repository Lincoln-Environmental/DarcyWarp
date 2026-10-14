"""Run existing terrain regressions with their shared backend selector on CPU."""

import importlib
import sys
from pathlib import Path

import pytest


def cpu_device():
    return "cpu"


def run_cpu_tests(*, repository_path: Path, test_arguments: list[str]) -> int:
    sys.path.insert(0, str(repository_path))
    sys.path.insert(0, str(repository_path / "tests"))
    helpers = importlib.import_module(name="test_3d_scientific_audit")
    helpers.device = cpu_device
    return pytest.main(args=test_arguments)


if __name__ == "__main__":
    repository_path = Path(__file__).resolve().parents[1]
    test_arguments = sys.argv[1:]
    exit_code = run_cpu_tests(
        repository_path=repository_path, test_arguments=test_arguments
    )
    raise SystemExit(exit_code)
