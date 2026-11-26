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

Run these checks from the repository root with the project environment active.

```bash
python -m compileall -q DARCY_WARP_PACKAGE tests working_tests
PYTHONPATH=.:working_tests python -m pytest -q -ra
```

For a focused run, give pytest the module that covers the changed equation or
interface. The [verification guide](source/validation.rst) maps test modules to
physical checks. The [capability guide](source/capabilities.rst) lists supported
solver combinations. Record the device alongside pass, failure and skip counts.
The native parity suite uses FloPy and the pinned MODFLOW 6.6.2 executable.

For dimensional solver runs and JUnit XML reports, use these commands.

```bash
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python -m working_tests.run_2d_gpu_tests
DARCY_FLOAT=float64 PYTHONPATH=.:working_tests python -m working_tests.run_3d_gpu_tests
```

The 2D runner includes the stored unconfined reference grids. The 3D runner
includes the independent equation checks. Both can run without a native MF6
executable. The maintained solver tests select CUDA when available and fall
back to CPU. The [2D release guide](docs/2D_GPU_RELEASE_NOTES.rst) records the
separate performance and full-history MF6 qualification.

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
