# SPDX-License-Identifier: AGPL-3.0-or-later
"""Paths for benchmark inputs and the optional native MODFLOW executable.

Importing this module does not create directories or access network shares.
"""

from pathlib import Path
from shutil import which

proj_root = Path(__file__).parent
data_store = proj_root / "data"

_MF6_ERROR = (
    "mf6 not found. Either add /bin/modflow to PATH for your Python environment/IDE, "
    "or install the binary to a standard location. "
    "Tried '/bin/modflow/mf6' and PATH."
)


def find_mf6() -> Path | None:
    """Locate MODFLOW 6 at the installed path or on the process PATH.

    :returns: Executable path, or None when neither location exists.
    """
    installed = Path("/bin/modflow/mf6")
    if installed.exists():
        return installed
    found = which("mf6")
    return None if found is None else Path(found)


MF6 = find_mf6()


def require_mf6() -> Path:
    """Return the native MODFLOW executable selected for this process.

    :returns: Path to MODFLOW 6.
    :raises RuntimeError: If the executable was not found.
    """
    if MF6 is None:
        raise RuntimeError(_MF6_ERROR)
    return MF6
