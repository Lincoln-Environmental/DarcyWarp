# MF6 datum-aware head comparison

The MF6 benchmark now retains raw head differences and applies the head gate
to a second comparison with only physically invariant datum shifts. The
existing `0.005 m` maximum and `1e-4 m` RMS tolerances are unchanged. An
alignment is accepted only when cell residuals, intercell flows, package
exchange and activation branches, and transient storage are unchanged.
Exact-zero rows are considered individually; balanced components are shifted
one at a time. A candidate that changes any of those physical terms is kept at
its original head.

Against the tightly converged 6.6.2 reference, 0403 passes without a datum
shift: raw and aligned head RMSE are `1.055e-7 m`, and maximum error is
`1.810e-5 m`. For 0414, the raw comparison remains visible and fails the head
limits (RMSE `2.160e-4 m`, maximum `0.12135 m`). Two balanced-component datum
shifts reduce the aligned RMSE to `3.967e-8 m` and maximum error to
`2.650e-5 m`; the unchanged head gates then pass. The 0414 audit also passes
the existing face-flow, package-budget, and physical-residual checks.

Machine-readable metrics and spatial maps are retained in each model's
`datum_alignment_audit/` directory. The full 74-model qualification is still
in progress; these two audits do not establish suite-wide compatibility.
