"""Audit saved 3D sanity runs against native MF6 budgets without rerunning MF6.

First generate references/heads with run_3d_sanity.py --ghb. This command never
changes the reference solver settings. A failed head gate remains a failure
even when the two engines' operators agree at identical heads.
"""
from pathlib import Path
import argparse
import json
import subprocess

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.validation.ghb_mf6_parity import (
    build_model, compare_endpoint, compare_heads, edge_indices, evaluate_device,
    file_digest, load_reference, read_native_saturations, prescribed_flow_probe,
)
from DARCY_WARP_PACKAGE.validation.head_checks import finite_diagnostics


def parse_arguments(*, defaults):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, default=defaults["qualification"])
    parser.add_argument("--cache", type=Path, default=defaults["cache"])
    parser.add_argument("--output", type=Path, default=defaults["output"])
    parser.add_argument("--device", default=defaults["device"])
    parser.add_argument("--head-tolerance", type=float, default=defaults["head_tolerance"])
    parser.add_argument("--roundoff-factor", type=float, default=defaults["roundoff_factor"])
    parser.add_argument("--cases", nargs="+", help="Exact case identities; omitted audits all cases.")
    parser.add_argument("--raw-root", type=Path, help="Relocated parent of campaign directories containing saved GPU heads.")
    parser.add_argument("--prescribed-flow-probes", type=Path,
        help="New directory for separate convertible first/last fixed-head MF6 flow-law probes; IMS settings unchanged.")
    parser.add_argument("--mf6", type=Path, default=defaults["mf6"])
    args = parser.parse_args()
    if not np.isfinite(args.head_tolerance) or args.head_tolerance <= 0:
        parser.error("head-tolerance must be finite and positive")
    if not np.isfinite(args.roundoff_factor) or args.roundoff_factor <= 0:
        parser.error("roundoff-factor must be finite and positive")
    return args


def save_report(*, path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(data=json.dumps(obj=finite_diagnostics(value=report), indent=2, allow_nan=False) + "\n")
    temporary.replace(target=path)


def audit_case(*, case_row, campaign, args):
    directory = args.cache / case_row["reference"]["fingerprint"]
    case, native, manifest = load_reference(directory=directory)
    if case.digest() != case_row["case_sha256"] or manifest["inputs"] != case_row["reference"]["inputs"]:
        raise ValueError("Campaign and reference inputs differ")
    inputs = manifest["inputs"]
    raw = Path(campaign.get("raw_results_path", args.qualification))
    if args.raw_root is not None:
        raw = args.raw_root / raw.parent.name / raw.name
    if campaign.get("raw_results_sha256") and file_digest(path=raw) != campaign["raw_results_sha256"]:
        raise ValueError("Saved benchmark results changed")
    family = inputs["family"]
    transient = inputs["mode"] == "transient"
    saturation, cbc_digest = read_native_saturations(directory=directory, case=case,
        count=len(native["heads"]), transient=transient)
    edges = edge_indices(case=case)
    previous = case.initial if transient else None
    row = dict(case=case_row["case"], shape=list(case.active.shape),
        reference_fingerprint=manifest["fingerprint"], reference_sha256=manifest["sha256"],
        diagnostic_cbc_sha256=cbc_digest,
        mf6_settings=inputs["native_options"], mf6_settings_changed=False,
        mode=inputs["mode"], family=family, endpoints=[], solution_comparisons=[])
    with build_model(case=case, family=family, device=args.device) as model:
        for index, head in enumerate(native["heads"]):
            factor = inputs["factors"][index] if transient else 1.
            options = dict(case=case, family=family, head=head, previous=previous,
                source_factor=factor, dt=inputs["dt"], ss=inputs["ss"], sy=inputs["sy"], edges=edges)
            actual = evaluate_device(model=model, **options)
            options.pop("family")
            comparison = compare_endpoint(**options, actual=actual,
                native_flux=native["fluxes"][index], native_ghb=native["ghb_inflows"][index],
                native_storage=native["storage_rates"][index] if transient else np.zeros_like(head),
                roundoff_factor=args.roundoff_factor, native_saturation=saturation[index])
            if args.prescribed_flow_probes is not None and np.any(case.cell_types) and index in (0, len(native["heads"]) - 1):
                probe_directory = args.prescribed_flow_probes / raw.parent.name / manifest["fingerprint"] / str(index + 1)
                comparison["prescribed_flow_probe"] = prescribed_flow_probe(case=case, head=head,
                    actual=actual, edges=edges, directory=probe_directory, executable=args.mf6,
                    native_options=inputs["native_options"], roundoff_factor=args.roundoff_factor)
            row["endpoints"].append(dict(period=index + 1, **comparison))
            print(f'{row["case"]} {index + 1}/{len(native["heads"])}: operator_parity={comparison["operator_parity"]}', flush=True)
            previous = head if transient else None
    for run in case_row["runs"]:
        path = raw.parent / f'{row["case"]}_{run["implementation"]}_{run["repeat"]}.npz'
        with np.load(file=path, allow_pickle=False) as saved:
            metrics = compare_heads(actual=saved["heads"], native=native["heads"],
                active=case.active != 0, head_tolerance=args.head_tolerance)
        row["solution_comparisons"].append(dict(implementation=run["implementation"], repeat=run["repeat"],
            warmup=run["warmup"], heads_sha256=file_digest(path=path), **metrics))
    row["operator_parity"] = all(endpoint["operator_parity"] for endpoint in row["endpoints"])
    row["solution_parity"] = all(run["passed"] for run in row["solution_comparisons"])
    return row


def run_audit(*, args):
    qualification = json.loads(s=args.qualification.read_text())
    campaigns = qualification.get("campaigns", [qualification])
    selected = [(campaign, row) for campaign in campaigns for row in campaign["cases"]
        if args.cases is None or row["case"] in args.cases]
    if not selected or (args.cases is not None and set(args.cases) != {row["case"] for campaign, row in selected}):
        raise ValueError("Requested case is absent from qualification")
    report = dict(schema=1, qualification_sha256=file_digest(path=args.qualification),
        revision=subprocess.check_output(args=["git", "rev-parse", "HEAD"], text=True).strip(),
        device=args.device, warp=wp.__version__, head_tolerance_m=args.head_tolerance,
        roundoff_factor=args.roundoff_factor, mf6_settings_changed=False, complete=False, cases=[])
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), root / "DARCY_WARP_PACKAGE/validation/ghb_mf6_parity.py"]
    report["audit_source_sha256"] = {str(path.relative_to(root)): file_digest(path=path) for path in paths}
    solver_paths = sorted({name for campaign, row in selected for name in campaign.get("solver_source_sha256", {})})
    report["solver_source_sha256"] = {name: file_digest(path=root / name) for name in solver_paths}
    report["device_name"] = wp.get_device(ident=args.device).name
    for campaign, case_row in selected:
        report["cases"].append(audit_case(case_row=case_row, campaign=campaign, args=args))
        save_report(path=args.output, report=report)
    report.update(complete=True, operator_parity=all(row["operator_parity"] for row in report["cases"]),
        solution_parity=all(row["solution_parity"] for row in report["cases"]))
    report["passed"] = report["operator_parity"] and report["solution_parity"]
    save_report(path=args.output, report=report)
    print(json.dumps(obj={key: report[key] for key in ("complete", "operator_parity", "solution_parity", "passed")}), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    defaults = dict(qualification=Path("benchmark_results/ghb_3d/main_qualification.json"),
        cache=Path("benchmark_results/ghb_3d/mf6_sanity_cache"),
        output=Path("benchmark_results/ghb_3d/mf6_parity.json"), device="cuda:0",
        head_tolerance=1e-6, roundoff_factor=32., mf6=Path("/bin/modflow/mf6"))
    arguments = parse_arguments(defaults=defaults)
    raise SystemExit(run_audit(args=arguments))
