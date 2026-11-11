# Contributing to DarcyWarp

Changes that make the model easier to use, understand and verify are welcome.
For a larger numerical or API change, describe the problem and proposed
behaviour in an issue before investing in a broad rewrite.

## Set up

From a checkout of the repository:

```bash
conda env create -f environment.yml
conda activate darcywarp
```

Run Python modules from the repository root so that
`DARCY_WARP_PACKAGE` is importable. The dependency file pins NVIDIA
Warp as `warp-lang==1.17.0`; its Python import name is `warp`.

## Make a focused change

Work on a branch based on `main` and keep documentation, examples and
numerical changes scoped to the problem. Preserve existing APIs unless the
change explicitly requires an interface update.

Use explicit functions and readable control flow. Keep parameters and physical
units visible in example scripts. Small examples should use an explicit
`if __name__ == "__main__":` entry point rather than adding a new CLI
framework.

For solver changes, retain the model's ownership of device arrays and cached
resources. Backends borrow those resources through the existing context.
Preserve attribution and licence notices.

## Check what changed

The CI workflow runs Python 3.12 with NumPy, SciPy, pandas, pytest and
Warp 1.17.0, checks imports and syntax, then runs these CPU smoke tests:

```bash
python -m compileall -q DARCY_WARP_PACKAGE tests working_tests
python -m pytest tests/test_sparse_system_consistency.py tests/test_solver_registry_2d.py -v
```

Run additional existing tests for the formulation, backend, precision and
device affected by a numerical change. Report passes, failures and skips
separately. A CPU run does not exercise CUDA-only paths. See the
[verification guide](source/validation.rst) for the test map and the
[capability guide](source/capabilities.rst) for supported combinations.

For the fast 2D release suites, including GHB, mixed precision and saved MF6
fixtures, run:

```bash
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python -m working_tests.run_2d_gpu_tests
```

The runner excludes native MF6 generation;
inspect CUDA skips and use the [2D release guide](docs/2D_GPU_RELEASE_NOTES.rst)
for separate performance and full-history MF6 checks.

For documentation changes, check the relevant example and build the docs:

```bash
python -m examples.steady_recharge
python -m sphinx -b html -n -W --keep-going source build/html
```

Review the rendered pages. New tests should exercise a meaningful behaviour or
regression, rather than repeat the implementation.

## Numerical changes and comparisons

Include a small reproducible case with explicit units, initial/boundary
conditions, forcing and solver controls. Show convergence information and
water balance as well as head error. Explain any changed tolerances using the
accuracy required by the case.

Performance comparisons should record hardware, dependency versions, grid,
precision, solver settings and timing scope. Use identical model inputs and
verify comparable numerical accuracy. Keep reference provenance when updating
saved results; do not replace a reference just to make a failing comparison
pass.

## Submit a pull request

Explain the problem, what changes and the resulting behaviour. Include the
checks performed and any device or dependency limits. Link the relevant issue
or reproduction, and keep generated benchmark outputs out of unrelated
changes.

The repository's declared licence is
[GNU AGPL v3.0 or later](LICENSE). Retain existing authorship and attribution
when editing files.
