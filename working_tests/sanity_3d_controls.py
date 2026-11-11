"""Read benchmark controls from the actual 2D sanity sources without importing them."""
from pathlib import Path
import ast
import hashlib


def read_function(*, path, name):
    tree = ast.parse(source=path.read_text())
    return next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)


def resolve_function_path(*, root, path, name):
    tree = ast.parse(source=path.read_text())
    if any(isinstance(node, ast.FunctionDef) and node.name == name for node in tree.body):
        return path
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            for alias in node.names:
                if (alias.asname or alias.name) == name:
                    candidate = root.joinpath(*node.module.split('.')).with_suffix('.py')
                    if candidate.is_file():
                        return resolve_function_path(root=root, path=candidate, name=alias.name)
    raise ValueError(f'Cannot resolve source for {name} from {path}')


def literal_bindings(*, node):
    values = {}
    for item in ast.walk(node=node):
        if isinstance(item, ast.Assign) and len(item.targets) == 1 and isinstance(item.targets[0], ast.Name):
            try:
                values[item.targets[0].id] = ast.literal_eval(node_or_string=item.value)
            except (ValueError, TypeError):
                pass
    return values


def resolve_value(*, node, bindings):
    if isinstance(node, ast.Name):
        return bindings[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [resolve_value(node=value, bindings=bindings) for value in node.elts]
    return ast.literal_eval(node_or_string=node)


def call_keywords(*, node, bindings):
    values = {}
    for keyword in node.keywords:
        try:
            values[keyword.arg] = resolve_value(node=keyword.value, bindings=bindings)
        except (ValueError, TypeError, KeyError):
            pass
    return values


def function_defaults(*, node):
    arguments = node.args.args[-len(node.args.defaults):]
    return {argument.arg: ast.literal_eval(node_or_string=value)
            for argument, value in zip(arguments, node.args.defaults)
            if isinstance(value, (ast.Constant, ast.UnaryOp))}


def read_profiles(*, root):
    steady_path = root/'DARCY_WARP_PACKAGE/model_convergence_and_sanity_tests.py'
    backend_path = root/'DARCY_WARP_PACKAGE/solvers/multigrid_kcycle.py'
    truth_path = root/'DARCY_WARP_PACKAGE/modflow_truth.py'
    transient_entry_path = root/'working_tests/transient_replay_settings.py'
    transient_path = resolve_function_path(root=root, path=transient_entry_path, name='default_solve_controls')
    transient_truth_path = root/'working_tests/run_2d_transient_vs_mf6.py'
    unconfined_path = root/'working_tests/run_2d_unconfined_warp_vs_mf6.py'
    steady = read_function(path=steady_path, name='run_solve')
    call = next(node.value for node in ast.walk(node=steady) if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == 'solve_kwargs' for target in node.targets))
    common = call_keywords(node=call, bindings={})
    common.pop('return_info', None)
    defaults = function_defaults(node=read_function(path=backend_path, name='solve_multigrid_kcycle_backend'))
    steady_defaults = function_defaults(node=steady)
    common.update(dh_rms_tol=defaults['dh_rms_tol'],
                  dh_max_tol=defaults['dh_max_factor']*defaults['dh_rms_tol'],
                  check_every_no=steady_defaults['check_every_no'],
                  smoother=defaults['smoother'], cheby_lambda_min=defaults['cheby_lambda_min'],
                  cheby_lambda_max=defaults['cheby_lambda_max'])
    coarse = next(node.value for node in ast.walk(node=steady) if isinstance(node, ast.Assign)
                  and any(isinstance(target, ast.Name) and target.id == 'nu_coarse' for target in node.targets))
    steady_profiles = {implementation: dict(common, nu_coarse=ast.literal_eval(
        node_or_string=coarse.body if implementation == 'fast' else coarse.orelse))
        for implementation in ('classic', 'fast')}
    function = read_function(path=transient_path, name='default_solve_controls')
    dictionary = next(node.value for node in ast.walk(node=function) if isinstance(node, ast.Return))
    relevant = set(common)|{'nu_coarse', 'smoother', 'cheby_lambda_min', 'cheby_lambda_max'}
    transient = {ast.literal_eval(node_or_string=key): ast.literal_eval(node_or_string=value)
        for key, value in zip(dictionary.keys, dictionary.values)
        if ast.literal_eval(node_or_string=key) in relevant}
    transient['dh_max_tol'] = defaults['dh_max_factor']*transient['dh_rms_tol']
    native = {}
    for label, path, name in (
        ('steady_confined', truth_path, 'make_mf_model'),
        ('transient', transient_truth_path, 'run_mf6_transient'),
        ('steady_unconfined', unconfined_path, '_build_and_run_mf6'),
    ):
        function = read_function(path=path, name=name)
        candidates = [node for node in ast.walk(node=function) if isinstance(node, ast.Call)
                      and isinstance(node.func, ast.Attribute) and node.func.attr == 'ModflowIms']
        call = candidates[0]
        native[label] = call_keywords(node=call, bindings=literal_bindings(node=function))
        if len(native[label]) != len(call.keywords):
            raise ValueError(f'Cannot resolve complete native sanity profile: {label}')
    paths = tuple(dict.fromkeys((steady_path, backend_path, truth_path, transient_entry_path,
        transient_path, transient_truth_path, unconfined_path)))
    return dict(steady=steady_profiles, transient=transient, mf6=native,
        source_sha256={str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        notes=['Cartesian controls are taken from these 2D source functions.',
               '2D min_coarse_cells has no identical 3D min_coarse_n parameter.',
               'NPF Newton/GMRES has a distinct fixed physical contract and Krylov work units; its final head RMS remains 1e-6 m.'])
