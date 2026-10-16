Canonical MF6 6.6.2 and DarcyWarp baseline
========================================

250 x 250 cells, five layers, 100 m cells, 150 mm/year RCH recharge.
Both confined/ and unconfined/ contain complete inputs.npz, direct input
arrays in inputs/, and accepted native MF6 output arrays in mf6/.
mf6/head.npy is the MF6 reference, not a DarcyWarp solution.

Array orientation, units, signs, physical model and acceptance criteria
are documented in ARRAY_CONVENTIONS.rst. Benchmark settings, individual
measurements, histories, comparisons and plots are in benchmark/.
linear_A0/ preserves the separate manufactured K-cycle experiment.

Each native.tar.gz preserves exact original native file contents,
including decks, HEAD, CBC, GRB, listings, convergence CSVs and summary.
Extract into an empty directory, for example::

  mkdir confined/native
  tar -xzf confined/native.tar.gz -C confined/native

The executable is identified by version and SHA256 in the metadata;
its local archived copy is not duplicated in Git. Redundant decoded
result.npz and canonical_inputs.npz are omitted from native archives;
their input and output arrays are provided directly in each variant.

source_snapshot/ retains the measured source, independent of later HEAD
changes. Original run directories and failure evidence remain unchanged
locally. provenance.json records source paths and native member hashes.
SHA256SUMS.json covers every archive file except itself. Verify with::

  python working_tests/archive_npf662_baseline.py

The default script verifies an existing archive and builds it if absent.
All configuration is in its execution guard. No solver is rerun.
