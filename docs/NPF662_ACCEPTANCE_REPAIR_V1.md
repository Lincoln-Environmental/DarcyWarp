# NPF662 acceptance repair, before layered geometry

Baseline: local dev8352de9c423471619c5914ce1d028120935b7f30. Dedicated branch `implementation/npf662-layered-dis`; unrelated main-checkout work is preserved. No environment, workflow or remote changes.

The initial assertions produce28 failures and1 pass, including both audited defects. Raw pre-fix examples use the unchanged audited checkout and the same interpreter; `before_examples/example_1.npz` retains the overflowed quantities before serialization. The shifted centre1+1e-9 is accepted by the old wrapper under requested1e-14m³/day/1e-12relative tolerances despite failing both alternatives. The1e308 example has finite faces and overflowed diagonal/throughput but was accepted.

Confined independent evaluation now receives the caller's physical RMS and relative-cell thresholds explicitly and records them in `final_acceptance_thresholds`. Krylov `rel_tol` remains a separate iteration control. All declared thresholds must be finite nonnegative scalars; zero requests exact arithmetic equality, not disabling a check. The head-equivalent and prescribed-head defaults remain1e-6m and1e-10m. Unconfined V1 retains fixed physical1e-12m³/day OR relative1e-8 policy; unsupported overrides raise an error explicitly explaining that `npf_newton_controls` controls iteration only.

The independent evaluator requires finite active residuals, conductances/fluxes, derivatives, principal diagonals/scaling, saturation, norms, external gross throughput and reservoir diagnostics. Invalid/unrepresentable aggregates reject before `finite_diagnostics` converts nonfinite diagnostic scalars to null. Scaled RMS avoids overflowing squared residuals; stable sums and signed overflow recovery retain representable results. Genuine zero/empty norms are0; inactive marker values remain raw and excluded, and original fixed heads are never repaired. Ordinary-range conductance arithmetic is unchanged.

Device validation additionally checks row scales, principal diagonal and face operands, and explicitly checks actual acceptance quantities. Independent final evaluation cannot promote an invalid device state to convergence. This does not redesign the nonlinear controller, preconditioner, safeguards or shared budgets. The GPU squared reductions still have a smaller numerical range than the host stable reductions; they fail conservatively if they overflow, rather than being promoted to success.

Offline reassessment reads64 accessible saved seven-point final fields under unchanged defaults without advancing any heads or modifying old evidence:64 remain accepted,0 classification changes. This is equation/conservation acceptance, not physical-model validation. Runtime metadata, raw before/after examples, assertions and reassessment live in `validation_artifacts/npf662_layered_v1/part_a/`. The first repaired group gives91 passes, no skips; the final group additionally tests actual CPU/CUDA overflow rejection (see `final_tests.log`).

Executed locally with `DARCY_FLOAT=float64 PYTHONPATH=.:working_tests OPENBLAS_NUM_THREADS=1` and existing `/home/patrickdurney/miniconda3/envs/darcywarp/bin/python`:

```bash
python -m pytest -q tests/test_npf662_acceptance.py
python -m pytest -q tests/test_npf662_acceptance.py tests/test_npf662_confined.py tests/test_npf662_newton.py tests/test_validation_repairs_3d.py
python working_tests/validate_npf662_acceptance.py
```
