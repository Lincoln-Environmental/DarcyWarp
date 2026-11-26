"""Extract completed exact-input native runs after a post-solve extractor failure."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np


def recover_completed(*, directory, executable, repeats):
    """Read the historical files themselves; never substitute a live model source.

    :param directory: Owned completed exact-input native workspace.
    :param executable: Pinned MF6 identity used for this campaign.
    :param repeats: Required completed native engine runs.
    :return: Extracted accepted heads, rate/volume ledgers and timing provenance.
    :raises AssertionError: If native acceptance, time grids or file ordering fail.
    """
    import flopy
    from qualify_package_mf6 import extract_completed_native
    from DARCY_WARP_PACKAGE.validation.npf662_mf6 import pinned_native_executable
    unused_binary, version, checksum, build = pinned_native_executable(executable=executable)
    logs = sorted(directory.glob('engine_*.txt'))
    assert len(logs) == repeats, (directory, len(logs), repeats)
    elapsed = []
    for path in logs:
        text = path.read_text()
        assert 'Normal termination' in text, path
        line = next(line for line in text.splitlines() if 'Elapsed run time:' in line)
        seconds = 0.
        for value, unit in re.findall(r'(\d+(?:\.\d+)?)\s+(Days|Hours|Minutes|Seconds)', line):
            multiplier = {'Days': 86400., 'Hours': 3600., 'Minutes': 60., 'Seconds': 1.}[unit]
            seconds += float(value) * multiplier
        assert seconds > 0., line
        elapsed.append(seconds)
    suffixes = {'.nam', '.tdis', '.ims', '.dis', '.ic', '.npf', '.sto', '.rch', '.wel', '.drn', '.riv', '.ghb', '.chd', '.oc'}
    inputs = [path for path in directory.iterdir() if path.suffix in suffixes]
    assert all(path.stat().st_mtime <= logs[0].stat().st_mtime for path in inputs)
    assert (directory / 'pkgqual.hds').stat().st_mtime <= logs[-1].stat().st_mtime
    reader = flopy.utils.HeadFile(str(directory / 'pkgqual.hds'), precision='double')
    times = np.asarray(reader.get_times())
    shape = list(reader.get_data(totim=float(times[0])).shape)
    reader.close()
    budget = flopy.utils.CellBudgetFile(str(directory / 'pkgqual.cbc'), precision='double')
    names = {name.decode().strip() for name in budget.get_unique_record_names()}
    budget.close()
    packages = [name for name in ('WEL', 'RCH', 'CHD', 'DRN', 'RIV', 'GHB') if name in names]
    manifest = dict(version=version, sha256=checksum, build=build, shape=shape,
        transient='STO-SS' in names, durations=np.diff(np.concatenate(([0.], times))).tolist(),
        packages=packages, engine_seconds=elapsed, engine_logs=[path.name for path in logs],
        input_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs},
        input_precision=17, construction_seconds=None, requested_repeats=repeats,
        ims_input_text=(directory / 'pkgqual.ims').read_text(),
        timing='MF6-reported engine elapsed times recovered from normal-termination logs; Python wrapper excluded',
        recovery='Completed historical native input/output files and log ordering; no live DarcyWarp inputs used')
    (directory / 'recovered_manifest.json').write_text(json.dumps(manifest, indent=2))
    return extract_completed_native(directory=directory, manifest=manifest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--repeats', type=int, required=True)
    arguments = parser.parse_args()
    from DARCY_WARP_PACKAGE.project_base import require_mf6
    executable = require_mf6()
    recover_completed(directory=arguments.directory, executable=executable, repeats=arguments.repeats)
