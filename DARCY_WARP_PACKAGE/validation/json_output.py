# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write finite validation results atomically."""
import json
from pathlib import Path


def atomic_json(*, path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False, default=float))
    temporary.replace(path)
