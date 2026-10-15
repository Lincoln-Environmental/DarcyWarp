#!/usr/bin/env bash
# Execute from the fix branch repository root. Existing interpreter, no install.
set -euo pipefail
export DARCY_FLOAT=float64
export PYTHONPATH=.:working_tests
TERRAIN_PYTHON=/home/patrickdurney/miniconda3/envs/darcywarp/bin/python
# Executed before implementation from starting dev d056eaf666268e3c5f3e0017c91d51c3160162de:
"$TERRAIN_PYTHON" working_tests/terrain_convergence_baseline.py --device-picard --device cuda:0 --variant variable --sizes 64 --dx 20 --floor 0.01 --basement-k 0.0001 --max-cycles 2000 --max-outer 2000 --out audit_outputs/low_k_64_before.json
# Direct configuration in the execution guard: two-outer warmups, 3 full
# converged cold-start primary pairs, K=.1/.01/.001 and n=16/32/96 controls.
# Newton cap=2000; separate converged Picard references cap=4000 (primary)
# or 8000 (controls). GPU synchronized; per-run other-process observations.
"$TERRAIN_PYTHON" -u working_tests/benchmark_terrain_newton.py > audit_outputs/benchmarks.log 2>&1
# Delivered checkpoint is an offline diagnostic input, never a cold-start seed.
# Stricter polishing runs are explicitly warm-started verification-only work.
"$TERRAIN_PYTHON" -u working_tests/verify_terrain_newton.py > audit_outputs/verification.log 2>&1
"$TERRAIN_PYTHON" working_tests/summarize_terrain_newton.py
"$TERRAIN_PYTHON" -m pytest -q tests/test_terrain_newton.py tests/test_terrain_device_picard.py tests/test_3d_terrain_gmres.py tests/test_3d_terrain_gpu.py tests/test_3d_terrain_reference.py tests/test_3d_scientific_audit.py tests/test_terrain_shanz_diagnostics.py tests/test_terrain_runtime.py tests/test_terrain_contact_exchange.py tests/test_terrain_contact_stencils.py tests/test_terrain_lateral_contacts.py tests/test_3d_solver.py tests/test_semismooth_newton_2d.py > audit_outputs/final_regressions.log 2>&1
