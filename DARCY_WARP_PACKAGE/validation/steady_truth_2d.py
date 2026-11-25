"""Physical fields and provenance for the steady 2D reference cache."""

import hashlib
import inspect
import json
from tempfile import NamedTemporaryFile
from pathlib import Path

import numpy as np

from DARCY_WARP_PACKAGE.model_builder import (
    steady_domain_2d, _build_dem, _build_dirichlet_boundary_mask, _build_ghb_boundary_masks,
    make_ugly_T_field,
)
from DARCY_WARP_PACKAGE.project_base import require_mf6
from DARCY_WARP_PACKAGE.validation.head_checks import invalid_head_values


def steady_transmissivity(*, domain, isotropic, isotropic_value, seed):
    """Construct the single physical T field used by Warp, FD and MF6."""
    if isotropic:
        return np.full(domain.shape, isotropic_value, dtype=np.float64)
    ny, nx = domain.shape
    return make_ugly_T_field(nx=nx, ny=ny, domain=domain, seed=seed)


def steady_reference_fingerprint(*, executable, runner, domain, conductivity, recharge, controls):
    """Hash physical inputs and the exact executable and adapter used to write them."""
    import flopy

    digest = hashlib.sha256()
    digest.update(json.dumps(controls, sort_keys=True, allow_nan=False).encode())
    digest.update(flopy.__version__.encode())
    with Path(executable).open("rb") as stream:
        binary_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    adapter_file = Path(inspect.getfile(runner))
    identity = dict(reference_schema=2, executable_sha256=binary_sha256, reference_adapter=runner.__qualname__,
        reference_adapter_sha256=hashlib.sha256(adapter_file.read_bytes()).hexdigest(),
        flopy_version=flopy.__version__)
    digest.update(json.dumps(identity, sort_keys=True).encode())
    dem = _build_dem(domain=domain)
    for value in (domain, conductivity, recharge, dem, dem.astype(float) - controls['thickness'],
                  _build_dirichlet_boundary_mask(domain), _build_ghb_boundary_masks(domain)):
        array = np.ascontiguousarray(value)
        digest.update(array.dtype.str.encode())
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest(), identity


def load_cached_mf6_truth(
    truth_path: Path,
    label: str,
    nx: int,
    ny: int,
    dx: float,
    ghb: bool,
    isotropic: bool,
    t_isotropic_value: float,
    thickness: float,
    width: float,
    recharge: float,
    seed: int,
    fingerprint: str,
    active: np.ndarray | None = None,
) -> tuple[np.ndarray, float | None] | None:
    """Load a matching MF6 artifact, returning ``None`` for a cache miss."""
    if not truth_path.exists():
        return None

    try:
        with np.load(truth_path, allow_pickle=False) as truth:
            if "reference_fingerprint" not in truth or str(truth["reference_fingerprint"]) != fingerprint:
                return None
            expected_scalars = {
                "nx": int(nx),
                "ny": int(ny),
                "ghb": int(bool(ghb)),
                "t_isotropic": int(bool(isotropic)),
                "seed": int(seed),
            }
            for key, expected in expected_scalars.items():
                if key not in truth.files or int(truth[key]) != expected:
                    return None

            expected_floats = {
                "dx": float(dx),
                "t_isotropic_value": float(t_isotropic_value),
                "thickness": float(thickness),
                "width": float(width),
                "r_truth": float(recharge),
            }
            for key, expected in expected_floats.items():
                if key not in truth.files or not np.isclose(
                    float(truth[key]),
                    expected,
                    rtol=1.0e-6,
                    atol=1.0e-12,
                ):
                    return None

            if "label" not in truth.files or str(truth["label"]) != str(label):
                return None
            if "heads" not in truth.files:
                return None

            heads = np.asarray(truth["heads"], dtype=np.float64)
            if heads.shape != (int(ny), int(nx)):
                return None
            checked = heads if active is None else heads[active != 0]
            if np.any(invalid_head_values(head=checked)):
                return None

            mf6_seconds = None
            if "mf6_seconds" in truth.files:
                candidate_seconds = float(truth["mf6_seconds"])
                if np.isfinite(candidate_seconds) and candidate_seconds >= 0.0:
                    mf6_seconds = candidate_seconds
    except (OSError, ValueError, KeyError, TypeError):
        return None

    return heads, mf6_seconds


def save_mf6_truth(
    truth_path: Path,
    heads: np.ndarray,
    label: str,
    nx: int,
    ny: int,
    dx: float,
    ghb: bool,
    isotropic: bool,
    t_isotropic_value: float,
    thickness: float,
    width: float,
    recharge: float,
    seed: int,
    mf6_seconds: float,
    output_dtype: np.dtype,
    fingerprint: str,
    identity: dict,
) -> None:
    """Atomically save one reusable MF6 truth artifact."""
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = NamedTemporaryFile(dir=truth_path.parent, prefix=f".{truth_path.name}.",
        suffix=".staging.npz", delete=False)
    temporary_path = Path(temporary.name)
    try:
        with temporary:
            float_dtype = np.dtype(output_dtype)
            np.savez_compressed(
                temporary,
                heads=np.asarray(heads, dtype=float_dtype),
                nx=np.int32(nx),
                ny=np.int32(ny),
                dx=np.asarray(dx, dtype=float_dtype),
                ghb=np.int32(1 if ghb else 0),
                t_isotropic=np.int32(1 if isotropic else 0),
                t_isotropic_value=np.asarray(t_isotropic_value, dtype=float_dtype),
                thickness=np.asarray(thickness, dtype=float_dtype),
                width=np.asarray(width, dtype=float_dtype),
                r_truth=np.asarray(recharge, dtype=float_dtype),
                seed=np.int32(seed),
                label=np.array(label),
                mf6_seconds=np.asarray(mf6_seconds, dtype=np.float64),
                reference_fingerprint=np.array(fingerprint),
                reference_identity_json=np.array(json.dumps(identity, sort_keys=True)),
            )
        temporary_path.replace(truth_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_or_run_mf6_truth(
    truth_path: Path,
    workspace: Path,
    label: str,
    nx: int,
    ny: int,
    dx: float,
    ghb: bool,
    isotropic: bool,
    t_isotropic_value: float,
    thickness: float,
    width: float,
    recharge_rate: float,
    recharge_field: np.ndarray,
    seed: int,
    hk_field: np.ndarray,
    output_dtype: np.dtype,
    mf6_runner,
    domain: np.ndarray | None = None,
    executable: Path | None = None,
    overwrite: bool = False,
    provenance: dict | None = None,
) -> tuple[np.ndarray, float | None, str]:
    """Load a current reference or create it alongside any historical artifact."""
    domain = steady_domain_2d(domain=domain, nx=nx, ny=ny)
    parameters = inspect.signature(mf6_runner).parameters
    supports_domain = "domain" in parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()
    )
    if not supports_domain and not np.array_equal(domain, steady_domain_2d(domain=None, nx=nx, ny=ny)):
        raise ValueError("The supplied MF6 writer cannot represent a custom domain")
    domain_option = {"domain": domain} if supports_domain else {}
    executable = require_mf6() if executable is None else executable
    fingerprint, identity = steady_reference_fingerprint(
        executable=executable, runner=mf6_runner, domain=domain, conductivity=hk_field,
        recharge=recharge_field, controls=dict(dx=dx, ghb=bool(ghb), width=width,
            thickness=thickness, output_dtype=np.dtype(output_dtype).str, nper=1),
    )
    if truth_path.exists():
        try:
            with np.load(truth_path, allow_pickle=False) as saved:
                old_fingerprint = str(saved["reference_fingerprint"]) if "reference_fingerprint" in saved else None
        except (OSError, ValueError, KeyError, TypeError):
            old_fingerprint = None
        if old_fingerprint != fingerprint:
            truth_path = truth_path.with_name(f"{truth_path.stem}.qualified-{fingerprint[:16]}.npz")
    cached = None if overwrite else load_cached_mf6_truth(
        truth_path=truth_path,
        label=label,
        nx=nx,
        ny=ny,
        dx=dx,
        ghb=ghb,
        isotropic=isotropic,
        t_isotropic_value=t_isotropic_value,
        thickness=thickness,
        width=width,
        recharge=recharge_rate,
        seed=seed,
        fingerprint=fingerprint,
        active=domain,
    )
    if cached is not None:
        heads, mf6_seconds = cached
        print(f"Loading cached MF6 truth: {truth_path}")
        if provenance is not None:
            record_reference(provenance, truth_path, fingerprint)
        return heads, mf6_seconds, "cache"

    print(f"MF6 truth cache miss; running model for {label}: {truth_path}")
    heads, mf6_seconds = mf6_runner(
        nx=nx,
        ny=ny,
        grid_size=dx,
        nper=1,
        workspace=workspace,
        hk=hk_field,
        recharge=recharge_field,
        run=True,
        use_ghb=ghb,
        ghb_width=width,
        aquifer_thickness=thickness,
        executable=executable,
        **domain_option,
    )
    if np.shape(heads) != (ny, nx) or np.any(invalid_head_values(head=np.asarray(heads)[domain != 0])):
        raise ValueError("MF6 returned invalid heads on the requested active domain")
    save_mf6_truth(
        truth_path=truth_path,
        heads=heads,
        label=label,
        nx=nx,
        ny=ny,
        dx=dx,
        ghb=ghb,
        isotropic=isotropic,
        t_isotropic_value=t_isotropic_value,
        thickness=thickness,
        width=width,
        recharge=recharge_rate,
        seed=seed,
        mf6_seconds=float(mf6_seconds),
        output_dtype=output_dtype,
        fingerprint=fingerprint,
        identity=identity,
    )
    if provenance is not None:
        record_reference(provenance, truth_path, fingerprint)
    return np.asarray(heads, dtype=output_dtype).astype(np.float64), float(mf6_seconds), "generated"


def record_reference(provenance, path, fingerprint):
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    provenance.update(path=str(path.resolve()), sha256=digest, fingerprint=fingerprint)
