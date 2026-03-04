# DarcyWarp Benchmarks

This repo contains scripts to run ensemble benchmarks (Warp solver, MODFLOW 6, optional CPU finite difference) and to generate standard summary plots from the benchmark outputs.

## Environment setup

Create the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate darcywarp
```

## Documentation (Sphinx)

The repo includes a Sphinx docs scaffold in `source/`.

Build docs with `make`:

```bash
make docs
```

Or use the Linux helper script (cleans first, then builds HTML docs):

```bash
bash scripts/generate_docs.sh
```

Generated HTML docs are written to `build/html/index.html`.

What this does

Generates a fixed set of n_cases synthetic recharge fields for a grid (nx, ny, dx)

Benchmarks the same recharge fields across:

Warp class solver (single GPU, repeated solves with in place recharge updates)

MF6 persistent worker mode (each worker builds a template once, then loops over assigned cases)

Optional CPU FD ensemble (if enabled in your project)

Writes JSON summaries to data_store

Generates plots from the JSON summaries

Outputs

Benchmark run writes summary JSON files to data_store (imported from DARCY_WARP_PACKAGE.project_base):

Recharge-change suite:

mf6_ensemble_benchmark_results_recharge{cells}.json

warp_class_ensemble_benchmark_results_recharge_{cells}.json

fd_ensemble_benchmark_results_recharge{cells}.json (only if FD is run)

Transmissivity-change suite:

mf6_T_ensemble_benchmark_results_{cells}.json

warp_class_T_ensemble_benchmark_results_{cells}.json

fd_T_ensemble_benchmark_results_{cells}.json (only if FD is run)

Where:

cells = nx * ny

Plot generation writes PNG figures to the --out_dir you choose.

Requirements

Python environment that can import:

DARCY_WARP_PACKAGE

flopy

MF6 executable available (typically mf6 on PATH, or configured in modflow_truth)

Warp solver dependencies available for GPU runs

Recommended workflow

Use the wrapper entrypoint that runs benchmarks then plots.

Run all benchmarks and generate plots
python -m DARCY_WARP_PACKAGE.bench_and_plot \
  --nx 1000 \
  --ny 1000 \
  --dx 100.0 \
  --n_cases 48 \
  --workers 2,4,8,16,24 \
  --seed 42 \
  --device cuda:0 \
  --out_dir <repo_root>/paper/tables_figures

Include GHB boundaries
python -m DARCY_WARP_PACKAGE.bench_and_plot \
  --nx 1000 \
  --ny 1000 \
  --dx 100.0 \
  --n_cases 48 \
  --workers 2,4,8,16,24 \
  --seed 42 \
  --device cuda:0 \
  --ghb \
  --out_dir <repo_root>/paper/tables_figures

Run only specific benchmarks

If you set none of --run_warp, --run_mf6, --run_fd, the wrapper runs all three by default.

Warp only:

python -m DARCY_WARP_PACKAGE.bench_and_plot \
  --run_warp \
  --nx 1000 \
  --ny 1000 \
  --dx 100.0 \
  --n_cases 48 \
  --seed 42 \
  --device cuda:0 \
  --out_dir <repo_root>/paper/tables_figures


MF6 only:

python -m DARCY_WARP_PACKAGE.bench_and_plot \
  --run_mf6 \
  --nx 1000 \
  --ny 1000 \
  --dx 100.0 \
  --n_cases 48 \
  --workers 2,4,8,16,24 \
  --seed 42 \
  --out_dir <repo_root>/paper/tables_figures


FD only:

python -m DARCY_WARP_PACKAGE.bench_and_plot \
  --run_fd \
  --nx 1000 \
  --ny 1000 \
  --dx 100.0 \
  --n_cases 48 \
  --workers 2,4,8,16,24 \
  --seed 42 \
  --out_dir <repo_root>/paper/tables_figures

Benchmark entrypoint (advanced)

You can run benchmark CLIs directly without plotting.

Recharge-change benchmark:

python -m DARCY_WARP_PACKAGE.model_benchmarking_recharge_change \
  --nx 1000 \
  --ny 1000 \
  --dx 100.0 \
  --n_cases 48 \
  --workers 2,4,8,16,24 \
  --seed 42 \
  --device cuda:0 \
  --run_warp \
  --run_mf6

Transmissivity-change benchmark:

python -m DARCY_WARP_PACKAGE.model_benchmarking_T_change \
  --nx 1000 \
  --ny 1000 \
  --dx 100.0 \
  --n_cases 48 \
  --workers 2,4,8,16,24 \
  --seed 42 \
  --device cuda:0 \
  --run_warp \
  --run_mf6

Optional metadata output:

Add --write_metadata to either benchmark CLI (or to bench_and_plot) to write an additional metadata JSON recording run configuration, seeds, solver settings, runtime flags, and output paths.

Benchmark flags

--nx: number of columns

--ny: number of rows

--dx: grid cell size

--n_cases: number of ensemble cases

--workers: comma separated MF6 or FD process counts to test (example 2,4,8,16,24)

--seed: RNG seed for recharge generation

--device: Warp device string (example cuda:0)

--ghb: enable GHB boundaries

--run_warp: run Warp solver benchmark

--run_mf6: run MF6 benchmark

--run_fd: run CPU FD benchmark

Recharge consistency guarantee

The benchmark generates a single recharge_stack of shape (n_cases, ny, nx) using seed, base_recharge, and jitter. That same stack is passed to all solver benchmarks that are enabled.

This guarantees every solver sees the exact same recharge arrays for each case index.

Plot generation entrypoint

Generate plots from existing JSON summaries.

python -m DARCY_WARP_PACKAGE.benchmark_plots \
  --mf6_summary /path/to/mf6_ensemble_benchmark_results_1000000.json \
  --warp_summary /path/to/warp_class_ensemble_benchmark_results_1000000.json \
  --out_dir <repo_root>/paper/tables_figures \
  --title_prefix "1000x1000, N=48: "

Plot flags

--mf6_summary: path to MF6 summary JSON

--warp_summary: optional path to Warp summary JSON

--out_dir: directory to write PNGs

--title_prefix: prefix used in plot titles

Plots produced

Typical outputs (filenames depend on benchmark_plots.py defaults):

throughput_vs_workers.png
MF6 throughput (cases per second) vs worker count, with optional Warp reference line.

walltime_vs_workers.png
Total wall time to complete N cases vs worker count, with optional Warp reference line.

idealised_completion_curves.png
Idealised completion curves derived from total wall time.

Troubleshooting
MF6 persistent worker crash: cannot unpack non-iterable NoneType object

This occurs if the worker tries to do:

sim, rch_pkg = make_mf_model(...)

but make_mf_model(...) returns None.

Correct approach in persistent MF6 workflows is:

Call make_mf_model(..., run=False) to write the template model to disk (ignore return value)

Load it using flopy.mf6.MFSimulation.load(...)

Get the recharge package via gwf.get_package("recharge") or gwf.get_package("rcha")

Update recharge with rcha.recharge.set_data(..., key=0) (or fallback dict form)

Missing Warp overlay in plots

The wrapper only passes --warp_summary to plotting if the Warp summary JSON exists:

warp_class_ensemble_benchmark_results_recharge_{cells}.json (recharge suite)

or

warp_class_T_ensemble_benchmark_results_{cells}.json (T suite)

If you ran only MF6, the Warp overlay will not be drawn.

## License

This repository is licensed under the GNU Affero General Public License v3.0 or later (`AGPL-3.0-or-later`).

See `LICENSE` for the full license text.
