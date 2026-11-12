# Third-party attributions and licences

DarcyWarp is developed by Patrick Durney and Jaco Fourie at Lincoln Agritech Ltd
and is distributed under **AGPL-3.0-or-later**, except where a source file
explicitly carries a different SPDX licence identifier. The project licence
is in [LICENSE](LICENSE).

## USGS MODFLOW 6: scientific formulation and validation reference

The DarcyWarp `mf6_6_6_2_npf` option independently implements selected
groundwater-flow and storage equations and numerical conventions specified by
the USGS **MODFLOW 6** Groundwater Flow Model, including the **NPF (Node
Property Flow)** and **STO (Storage)** packages. DarcyWarp implements these
methods using Python, NumPy and NVIDIA Warp. The implementation is not an
official MODFLOW 6 distribution, a wrapper of its Fortran solver, or a claim
of exact support for every MODFLOW 6 package or option.

Scientific / software reference:

> Langevin, C. D., Hughes, J. D., Provost, A. M., and contributors (2025).
> *MODFLOW 6 Modular Hydrologic Model*, version 6.6.2, released 12 May 2025.
> U.S. Geological Survey software release.
> https://doi.org/10.5066/F76Q1VQV

The [MODFLOW 6.6.2 software citation](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/CITATION.cff)
is authoritative for the full contributor list and software DOI.
The official [NPF](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-npf.html)
and [STO](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-sto.html)
input specifications document the referenced packages.

MODFLOW 6.6.2 declares **CC0-1.0** (with the USGS public-domain statement and
an "unless otherwise noted" qualification) in its
[licence](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/LICENSE.md).
DarcyWarp's independent NPF/STO implementation remains subject to DarcyWarp's
declared AGPL-3.0-or-later licence. MODFLOW 6 and FloPy are used for independent
comparison workflows; no MODFLOW 6 Fortran implementation or executable is
included as a part of the DarcyWarp solver distribution.

This acknowledgment does **not** imply USGS endorsement, verification or
certification of DarcyWarp. Validation against MODFLOW 6 is limited to the
documented cases and numerical options.

## NVIDIA Warp: adapted GMRES code (Apache-2.0)

`DARCY_WARP_PACKAGE/solvers/terrain_gmres_3d.py` contains an adaptation of
`GMRES._run` from
[NVIDIA Warp 1.17.0, `warp/_src/optim/linear.py`](https://github.com/NVIDIA/warp/blob/v1.17.0/warp/_src/optim/linear.py).

Upstream copyright notice:

> Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Upstream SPDX identifier: **Apache-2.0**.

The DarcyWarp version modifies the original code for persistent GPU restart
graphs and Krylov arrays, reduced scalar readbacks between restarts,
short final restart blocks and buffer-handling changes. The adapted file
retains its **Apache-2.0** SPDX identifier and source attribution.

The complete upstream NVIDIA Warp Apache-2.0 licence, including the upstream
licence-file copyright notice, is preserved verbatim in
[third_party_licenses/NVIDIA-Warp-Apache-2.0.txt](third_party_licenses/NVIDIA-Warp-Apache-2.0.txt),
taken from [NVIDIA Warp v1.17.0 LICENSE.md](https://github.com/NVIDIA/warp/blob/v1.17.0/LICENSE.md).

The remainder of the DarcyWarp package does not acquire the Apache-2.0
identifier merely by importing or using the adapted class. NVIDIA Warp itself
is an external dependency installed by the environment, not vendored into
DarcyWarp.

NVIDIA does not sponsor, endorse or certify DarcyWarp.
