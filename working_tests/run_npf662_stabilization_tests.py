"""Run affected numerical suites while explicitly excluding native MF6 work."""
import ast
from pathlib import Path
import sys

import pytest


def native_test_names(*, path, entry_points):
    tree = ast.parse(source=path.read_text())
    calls = {}
    for node in ast.walk(node=tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            names = set()
            for child in ast.walk(node=node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):names.add(child.func.id)
                    elif isinstance(child.func, ast.Attribute):names.add(child.func.attr)
            calls[node.name] = names
    native = set(entry_points)
    while True:
        expanded = native | {name for name, names in calls.items() if names & native}
        if expanded == native:break
        native = expanded
    return {name for name in calls if name.startswith('test_') and name in native}


class NoNativeSelection:
    def __init__(self, *, entry_points):
        self.entry_points = entry_points
        self.inventory = {}

    def pytest_collection_modifyitems(self, session, config, items):
        selected, excluded = [], []
        for item in items:
            path = Path(item.path)
            if path not in self.inventory:
                self.inventory[path] = native_test_names(path=path, entry_points=self.entry_points)
            name = getattr(item, 'originalname', item.name.split('[')[0])
            if name in self.inventory[path]:excluded.append(item)
            else:selected.append(item)
        items[:] = selected
        config.hook.pytest_deselected(items=excluded)
        reporter = config.pluginmanager.getplugin('terminalreporter')
        reporter.write_line(f'No-MF6 policy: excluded {len(excluded)} native-invoking test cases.')
        for item in excluded:reporter.write_line(f'Native excluded: {item.nodeid}')


def prohibit_native_process(*, event, arguments):
    if event != 'subprocess.Popen':return
    executable, command = arguments[:2]
    values = [executable] + (list(command) if isinstance(command, (list, tuple)) else [command])
    for value in values:
        if isinstance(value, (str, bytes)):
            name = Path(value.decode() if isinstance(value, bytes) else value).name.lower()
            if name in ('mf6', 'mf6.exe', 'modflow6'):
                raise RuntimeError('Native MF6 execution is forbidden during stabilization testing')


def run_tests(*, directory, patterns, entry_points, pytest_options):
    paths = sorted({path for pattern in patterns for path in directory.glob(pattern=pattern)})

    def audit_hook(event, arguments):
        prohibit_native_process(event=event, arguments=arguments)

    sys.addaudithook(hook=audit_hook)
    selection = NoNativeSelection(entry_points=entry_points)
    return pytest.main(args=[str(path) for path in paths] + list(pytest_options), plugins=[selection])


if __name__ == "__main__":
    directory = Path('tests')
    patterns = ('test_3d*.py', 'test_terrain*.py', 'test_npf662*.py',
                'test_fast_3d*.py', 'test_validation_repairs_3d.py')
    entry_points = ('run_reference_mf6', 'run_mf6', '_build_mf6_unconfined', 'run_simulation')
    pytest_options = ('-q', '--tb=short')
    result = run_tests(directory=directory, patterns=patterns, entry_points=entry_points,
                       pytest_options=pytest_options)
    raise SystemExit(result)
