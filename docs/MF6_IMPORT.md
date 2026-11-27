# Import an existing MODFLOW 6 model (initial development implementation)

Edit the explicit workspace, output and device settings in
[examples/run_mf6_model.py](../examples/run_mf6_model.py), then run:

    python -m examples.run_mf6_model

The reusable entry point is
DARCY_WARP_PACKAGE.mf6_import.run_mf6_model. FloPy reads the workspace,
including external array files, and DarcyWarp solves the imported model.
The original files are not rewritten and MF6 is not executed. An optional
reference_head_file points to an existing native MF6 binary head file.

The adapter automatically selects FP64 before importing the solver when
DarcyWarp precision has not yet been initialised. For a process that has
already loaded FP32 configuration or kernels, restart Python with
DARCY_FLOAT=float64. The adapter never reloads kernels or changes an
initialised solver's precision.

## Initial supported contract

- Single GWF model, structured DIS with uniform DELR/DELC and positive
  thickness in active cells. Layer interfaces may vary spatially. Invalid
  IDOMAIN=0 geometry is replaced with positive-thickness dummy geometry
  only where active interfaces can remain unchanged. An internal excluded
  gap whose bounding active elevations cannot accommodate positive dummy
  thickness is rejected. No original input files or active geometry change.
- IC, default NPF conductance and axis-aligned anisotropy; confined or NEWTON
  convertible cells; default STO specific storage and specific yield with
  matching ICONVERT and ICELLTYPE.
- CHD, GHB, WEL, RCH/RCHA, DRN and RIV. Multiple packages are combined;
  multiple GHB entries in a cell are conductance-weighted, and DRN/RIV/WEL
  entries retain individual physical inputs. Recharge receiving cells are
  resolved from the original MF6 list or IRCH array. IDOMAIN-excluded and
  prescribed-head RCHA cells receive no recharge; list recharge on excluded
  cells is rejected. Recharge is passed as volumetric RHS forcing.
- TDIS fixed timesteps (including TSMULT) and steady/transient transitions.
  Inherited periods and explicitly empty list stress periods are distinguished.
- MF6 row/layer ordering and original model length/time units are retained.
  DarcyWarp uses its own solver and acceptance controls rather than IMS.
- All accepted timesteps are written to darcywarp.hds (double-precision MF6
  binary HEAD format). darcywarp_report.json records timings, scalar budgets
  and optional head comparisons at matching time-step records. Original OC
  selection, MF6 binary CBC, observations and listing-file output are not
  reproduced.

Unsupported packages or physical options raise UnsupportedMF6Model rather
than being silently substituted. Current exclusions include DISV/DISU,
vertical pass-through cells, active-cell pinch-outs, rotated anisotropy, XT3D, alternative
conductance/storage options, time-series inputs, auxiliary multipliers, ATS,
SFR/UZF and coupled models. Non-recharge stress entries on inactive or CHD
cells are rejected. Inactive/dry head markers in native reference active
cells are not compared. Full drying/rewetting parity is outside the native
solver's current qualification.

This is a correctness-first single-step driver. It updates package data
between stress periods and rebuilds when CHD membership changes. It does
not yet group steps into a GPU-resident history. A failed timestep is not
written as an accepted head; an incomplete run leaves a .partial file.

## Regression tests

    pytest -q tests/test_mf6_import.py

Tests cover translation from native input files, stress-period reuse and
clear, geometric timesteps, unsupported input, binary head round-trip and
native MF6 parity when the pinned reference executable is available. CUDA
and CPU use the existing test device selection. This development adapter
must be qualified against independent regional workspaces before being
advertised as a general MODFLOW replacement.
