# Confined layered geometry milestone

Part A was committed as9f3ddea before formulation work:94 repaired/NPF/validation tests pass, no skips; the64 saved V1 final fields keep their classifications. The [layered contract](NPF662_LAYERED_DIS_REFERENCE_CONTRACT_V2.md) was written and live supplied-head equations checked before implementing this builder.

Production layered faces come from `physics/npf662_layers.py`, not the SciPy oracle. `NPF662LayeredCase` has an independent schema2 and no scalar dz/top. `NPF662Case` keeps exactly its V1 dataclass field order/hash interpretation; an old NPZ roundtrip/hash regression passes. `build_from_layers` constructs a new owned session with authoritative interfaces and derived tops/bottoms/thickness/centres/volumes. Legacy Cartesian geometry validation was not relaxed. Session.dz=None prevents accidental use of an unrelated scalar. Layered confined solves use the existing diagonal-preconditioned GMRES controller with cell types0; no new K-cycle/MG algorithm is introduced. The existing scalar confined backend stays unchanged.

The first test run had one fixture indexing error for a one-column series case;108 other checks passed. Correcting the fixture without changing equations or tolerances gives124 passing tests, no skips (same focused group as Part A plus layered tests). Tests cover arbitrary heads/Jv, x/y thickness variation, nonparallel interfaces, nonsquare grid/unequal spacing, anisotropy, representable intermediate overflow, vertical series resistance, uniform compatibility, masks/components, prescribed layers, scalar-dz independence, numerical reuse, same-shape geometry rebuild, topology rebuild and cold reset. Native confined supplied-head/complete comparisons pass.

Five durable cold controls (uniform, x, y, independent interfaces, masked) each pass independent SciPy CPU, Warp CPU, local CUDA and pinned native MF6. Every native supplied-head model preserves its head and matches independently reconstructed flux/residual. Exact fields, inputs, native files and source hashes are under `validation_artifacts/npf662_layered_v1/confined/experiments/`. No physical-model validation or terrain-equivalence claim follows. Public layered-unconfined activation remains deliberately gated until this confined milestone is recorded.

Local commands use the unchanged existing interpreter documented in Part A, `DARCY_FLOAT=float64 PYTHONPATH=.:working_tests OPENBLAS_NUM_THREADS=1`:

```bash
python -m pytest -q tests/test_npf662_layers.py tests/test_npf662_acceptance.py tests/test_npf662_confined.py tests/test_npf662_newton.py tests/test_validation_repairs_3d.py
python working_tests/verify_npf662_layered_contract.py
python working_tests/validate_npf662_layered.py
```
