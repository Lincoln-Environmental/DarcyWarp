# Stage 1 confined parity — completed locally

Contract: [NPF662 Cartesian v1](NPF662_CARTESIAN_REFERENCE_CONTRACT_V1.md). Stage0 was committed and passed before implementation. Historical terrain code/evidence remain intact. The opt-in `formulation="mf6_6_6_2_npf"` confined path reuses the existing physical Cartesian conductances and persistent fast/classic session. Independent NumPy/SciPy assembly, component checks and final acceptance supplement that path; no terrain principal coefficients or cross-term deletion are involved. Legacy defaults remain unchanged.

**Gate passes:**21 new tests, zero skips, CPU and CUDA;292 existing Cartesian/boundary/budget/session/scientific and terrain regressions, zero skips. Eight complete controls pass independent CPU, Warp CPU, CUDA and live MF6: constant, x/y/z affine, layered series resistance, six-order heterogeneous K, a masked hole, and two grounded disconnected components. Ungrounded balanced/positive/negative-source components are rejected. Source on inactive cells is rejected; fixed-cell reservoir sources retain their explicit convention. Workspace numerical refresh and topology rebuild are tested.

The independently assembled operator agrees with GPU arbitrary-head action within the predeclared tolerance. All16 native models (eight cold roots plus eight supplied-state checks) ran locally. Maximum same-head native face difference2.84e-14m³/day; maximum native/CPU head difference1.11e-12m. Native physical RMS ranges0–5.05e-13m³/day. Every native source allocation error is0. Each native result passes the unchanged independent gate. The supplied-state models prove heads did not advance and reconstruct original-problem cell residuals; their all-CHD native solver residual is not used as the parity test.

Native high-contrast layered accuracy required investigation. Small-tolerance BICGSTAB and CG runs could return native success with independently checked physical RMS around8e-12m³/day. Tightening native tolerances alone, zero dropping, or diagonal scaling did not resolve it. Increasing ILUT fill on the same model reduced physical RMS to8.27e-15. Failed numerical attempts and actual listings/CSVs remain under stage1/. This supports a reference linear-accuracy limitation; it does not demonstrate a different physical law or justify loosening acceptance. Final reference controls are explicit in the contract and decks. The initial artifact writer also exposed legacy steady `dt=NaN`; the new contract reports null because no timestep exists, without changing physical arithmetic.

API:

```python
solver = WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz,
    device="cuda:0", implementation="fast", formulation="mf6_6_6_2_npf")
solver.build_from_K_fields(kx_field=kx, ky_field=ky, kz_field=kz, active=active,
    bc_mask=fixed, bc_values=values, rhs=volumetric_source, initial_head=cold_head)
head, info = solver.solve(max_cycles=2000)
```

Results identify formulation, geometry, reference contract, effective options, backend convergence and independently checked acceptance. Terrain builds and arbitrary face overrides are rejected in this formulation. Unsupported flags are errors, not legacy fallback.

Executed with the unchanged stage0/runtime.json interpreter and MF6:

```bash
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
export OPENBLAS_NUM_THREADS=1
PY=/home/patrickdurney/miniconda3/envs/darcywarp/bin/python
"$PY" -m pytest -q tests/test_npf662_confined.py
"$PY" working_tests/validate_npf662_confined.py
"$PY" -m pytest -q tests/test_3d_solver.py tests/test_3d_physics_validation.py tests/test_fast_3d_kcycle.py tests/test_3d_session.py tests/test_3d_operator_analytical.py tests/test_3d_face_kernels.py tests/test_3d_scientific_audit.py tests/test_terrain_cpu_validation.py tests/test_validation_repairs_3d.py tests/test_3d_terrain_gpu.py
```

Evidence and full numerical tables: `validation_artifacts/sevenpoint_v1/stage1/experiments/gate.json`, per-case arrays/comparisons and complete native decks, listings, heads, CBC/GRB and iteration CSVs. Source hashes record the actual working implementation before this local milestone commit. No performance or physical-model-validation claim follows. Stage2 is now permitted; variable elevations, local head-dependent packages and storage remain deferred.
