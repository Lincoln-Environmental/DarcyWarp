# Draft 3D README addition — not published

DarcyWarp has separate legacy Cartesian, explicitly selected MF6 6.6.2-compatible
Cartesian/layered NPF, and experimental transformed-terrain implementations.
Their geometry, boundary and dry-cell laws differ. See the
[3D capability guide](3D_CAPABILITIES.md) and
[integration-readiness assessment](3D_RELEASE_READINESS.md) before selecting one.

A synthetic public-API example with exact series-resistance heads and fluxes is
available at [examples/verify_3d_series_flow.py](../examples/verify_3d_series_flow.py).
It runs on Warp CPU and saves the completed numerical verification.

Developers: Patrick Durney and Jaco Fourie. Software ownership: Lincoln Agritech
Ltd. Project code: AGPL-3.0-or-later; adapted NVIDIA GMRES retains its
Apache-2.0 attribution. This text is a future integration draft and does not
change the existing public README or promote 3D to a release.
