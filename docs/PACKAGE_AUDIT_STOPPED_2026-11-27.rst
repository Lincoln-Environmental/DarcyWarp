Stopped package audit checkpoint
================================

Work was stopped at the user request on 2026-11-27. Qualification is
unfinished; this branch must not be treated as SFR-ready or fully qualified.

The production suite at 7111579 recorded 1655 passed and 21 skipped;
17 FP32-specific checks and four later reference-contract checks passed.
Later benchmark/reference-harness changes did not complete their final
validation. The existing intermediate report predates some changes.

The latest clean resident-history results and partial native-run receipts
are retained under benchmark_results/package_audit_repair. Completed
COMPLEX reference metadata and interrupted manifests are saved under
stopped_native_checkpoint. An interrupted run is not a passing result.

All source changes and compact results are committed. Large raw arrays
and generated MF6 caches remain local. No main merge or SFR work occurred.
