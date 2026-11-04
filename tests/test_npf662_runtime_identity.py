# SPDX-License-Identifier: AGPL-3.0-or-later
"""Native reference portability must not erase executable provenance."""
import pytest

from DARCY_WARP_PACKAGE.validation.npf662_mf6 import MF6_BUILDS, MF6_VERSION, validate_mf6_identity


@pytest.mark.parametrize("checksum", sorted(MF6_BUILDS))
def test_documented_exact_builds_are_labelled(checksum):
    assert validate_mf6_identity(version=MF6_VERSION, checksum=checksum) == MF6_BUILDS[checksum]


@pytest.mark.parametrize("version,checksum", [(MF6_VERSION, "0" * 64), ("mf6: 6.7.0", next(iter(MF6_BUILDS)))])
def test_other_versions_or_unrecorded_builds_are_rejected(version, checksum):
    with pytest.raises(ValueError, match="pinned MF6 contract"):
        validate_mf6_identity(version=version, checksum=checksum)
