"""Keep reproducible controls and every endpoint audit without large solver histories."""
import argparse
import hashlib
import json
from pathlib import Path


def compact_campaign(*, input_path):
    raw = input_path.read_bytes()
    campaign = json.loads(s=raw)
    campaign['raw_results_path'] = str(input_path)
    campaign['raw_results_sha256'] = hashlib.sha256(raw).hexdigest()
    for case in campaign['cases']:
        for run in case['runs']:
            for period in run['periods']:
                period.pop('outer_history', None)
    return campaign


def save_qualification(*, input_paths, output_path):
    campaigns = [compact_campaign(input_path=path) for path in input_paths]
    evidence = dict(schema=1, campaigns=campaigns,
        retained='Every warmup/repeat endpoint audit, original controls, source hashes and reference fingerprints.',
        omitted='Per-outer solver histories and binary head/flow artifacts; regenerate with the documented runner.',
        precision='Roundoff acceptance is a componentwise backward residual test, not bitwise MF6 head equality.')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(data=json.dumps(obj=evidence, indent=2, allow_nan=False)+'\n')


def parse_arguments(*, default_output):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('inputs', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, default=default_output)
    return parser.parse_args()


if __name__ == "__main__":
    default_output = Path('benchmark_results/ghb_3d/qualification.json')
    arguments = parse_arguments(default_output=default_output)
    save_qualification(input_paths=arguments.inputs, output_path=arguments.output)
