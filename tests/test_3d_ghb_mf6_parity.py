"""Native budget parity, nonlinear reporting state and corrupt-evidence guards."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pytest

from test_3d_ghb_mf6 import ghb_case
from working_tests.sanity_3d_controls import read_profiles
from DARCY_WARP_PACKAGE.validation.ghb_mf6_parity import (
    arithmetic_comparison, build_model, compare_endpoint, compare_heads,
    edge_indices, evaluate_device, file_digest, load_reference,
    prescribed_flow_probe, read_native_saturations,
)
from DARCY_WARP_PACKAGE.validation.npf662_reference import save_npf662_case
from DARCY_WARP_PACKAGE.validation.npf662_mf6 import run_reference_mf6
from DARCY_WARP_PACKAGE.validation.npf662_transient_mf6 import run_transient_mf6


@pytest.mark.parametrize("device", ["cpu", "cuda:0"])
@pytest.mark.parametrize("family,convertible", [("cartesian", False), ("npf", False), ("npf", True)])
def test_native_budget_and_prescribed_state(tmp_path, pinned_mf6, device, family, convertible):
    case = ghb_case(convertible=convertible)
    profiles = read_profiles(root=Path(__file__).resolve().parents[1])["mf6"]
    ims = profiles["steady_unconfined" if convertible else "steady_confined"]
    head, info, fields = run_reference_mf6(case=case, directory=tmp_path / "steady/native",
        executable=pinned_mf6, native_solver_options=ims, replace_native_solver_options=True,
        head_tolerance_m=ims["outer_dvclose"])
    assert info["native_success"]
    factors, dt, ss, sy = [1., .5], 2., 1e-3, .15
    native, info = run_transient_mf6(case=case, directory=tmp_path / "transient/native",
        executable=pinned_mf6, executable_sha256=file_digest(path=pinned_mf6),
        source_factors=factors, dt=dt, ss=ss, sy=sy,
        native_solver_options=profiles["transient"])
    assert info["native_success"]
    sat, unused = read_native_saturations(directory=tmp_path / "steady", case=case, count=1, transient=False)
    transient_sat, unused = read_native_saturations(directory=tmp_path / "transient", case=case, count=2, transient=True)
    states = [(head, None, 1., fields["flux_native"], fields["ghb_native_inflow"], np.zeros_like(head), sat[0])]
    previous = case.initial
    for index, current in enumerate(native["heads"]):
        states.append((current, previous, factors[index], native["fluxes"][index],
            native["ghb_inflows"][index], native["storage_rates"][index], transient_sat[index]))
        previous = current
    edges = edge_indices(case=case)
    with build_model(case=case, family=family, device=device) as model:
        for index, (current, previous, factor, flux, ghb, storage, saturation) in enumerate(states):
            inputs = dict(case=case, head=current, previous=previous, source_factor=factor,
                dt=dt, ss=ss, sy=sy, edges=edges)
            actual = evaluate_device(model=model, family=family, **inputs)
            result = compare_endpoint(**inputs, actual=actual, native_flux=flux,
                native_ghb=ghb, native_storage=storage, native_saturation=saturation, roundoff_factor=32.)
            assert result["comparisons"]["ghb_inflow"]["passed"], result
            assert result["comparisons"]["storage_rate"]["passed"], result
            assert result["reported_saturation_diagnostic"]["reconstructed_face_flux"]["passed"], result
            if not convertible:
                assert result["operator_parity"], result
            if index == 0:
                probe = prescribed_flow_probe(case=case, head=current, actual=actual, edges=edges,
                    directory=tmp_path / "prescribed", executable=pinned_mf6,
                    native_options=ims, roundoff_factor=32.)
                assert probe["face_flux"]["passed"], probe
            # A sign error in either package must not be hidden by domain cancellation.
            wrong_ghb = compare_endpoint(**inputs, actual=actual, native_flux=flux,
                native_ghb=-ghb, native_storage=storage, roundoff_factor=32.)
            assert not wrong_ghb["comparisons"]["ghb_inflow"]["passed"]
            if previous is not None:
                wrong_storage = compare_endpoint(**inputs, actual=actual, native_flux=flux,
                    native_ghb=ghb, native_storage=-storage, roundoff_factor=32.)
                assert not wrong_storage["comparisons"]["storage_rate"]["passed"]
            wrong_faces = compare_endpoint(**inputs, actual=actual, native_flux=flux[::-1],
                native_ghb=ghb, native_storage=storage, roundoff_factor=32.)
            assert not wrong_faces["comparisons"]["face_flux"]["passed"]


def test_roundoff_comparison_has_no_absolute_floor():
    assert not arithmetic_comparison(actual=np.array([1e-100]), native=np.array([0.]),
        scale=np.array([0.]), roundoff_factor=32.)["passed"]
    assert not arithmetic_comparison(actual=np.array([2e-100]), native=np.array([1e-100]),
        scale=np.array([1e-100]), roundoff_factor=32.)["passed"]
    with pytest.raises(ValueError, match="Nonfinite"):
        arithmetic_comparison(actual=np.array([np.nan]), native=np.array([0.]),
            scale=np.array([1.]), roundoff_factor=32.)


def test_head_gate_is_separate_and_strict():
    native = np.ones(shape=(2, 1, 2, 3))
    actual = native.copy()
    actual[1, 0, 0, 1] += 2e-6
    assert not compare_heads(actual=actual, native=native, active=np.ones(shape=(1, 2, 3), dtype=bool),
        head_tolerance=1e-6)["passed"]


@pytest.mark.parametrize("corruption", ["input_bytes", "reference_bytes", "input_settings", "time_grid"])
def test_reference_integrity(tmp_path, corruption):
    case = ghb_case(convertible=False)
    inputs = dict(case_sha256=case.digest(), mode="transient", factors=[1.], dt=2.,
        native_options=dict(outer_dvclose=1e-6))
    fingerprint = hashlib.sha256(json.dumps(obj=inputs, sort_keys=True).encode()).hexdigest()
    directory = tmp_path / fingerprint
    directory.mkdir()
    save_npf662_case(case=case, path=directory / "inputs.npz")
    np.savez(file=directory / "reference.npz", heads=case.initial[None],
        times=np.array([3. if corruption == "time_grid" else 2.]))
    manifest = dict(fingerprint=fingerprint, inputs=inputs, accepted=True,
        sha256={name: file_digest(path=directory / name) for name in ("inputs.npz", "reference.npz")})
    if corruption == "input_settings":
        manifest["inputs"]["native_options"]["outer_dvclose"] = 1e-4
    (directory / "manifest.json").write_text(data=json.dumps(obj=manifest))
    if corruption in ("input_bytes", "reference_bytes"):
        name = "inputs.npz" if corruption == "input_bytes" else "reference.npz"
        with (directory / name).open(mode="ab") as stream:
            stream.write(b"changed")
    with pytest.raises(ValueError, match="fingerprint|Corrupt|time grid"):
        load_reference(directory=directory)
