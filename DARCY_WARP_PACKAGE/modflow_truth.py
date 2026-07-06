import numpy as np
import flopy
import warnings
import matplotlib.pyplot as plt
from pathlib import Path
import time
from multiprocessing.shared_memory import SharedMemory

from DARCY_WARP_PACKAGE.project_base import data_store, require_mf6
from DARCY_WARP_PACKAGE.model_builder import (
    _build_domain,
    _build_dirichlet_boundary_mask,
    _build_ghb_boundary_masks,
    _build_dem,
    _model_bottom,
    _create_chd_single_period,
)


def fill_nan_with_nearest(grid):
    """
    Fill np.nan values in a 2D array using nearest neighbour interpolation.
    :param grid: np.ndarray with np.nan values to fill
    :return: np.ndarray with NaNs filled from nearest neighbour
    """
    from scipy import ndimage

    grid = np.array(grid, copy=True)
    mask = np.isnan(grid)

    if not np.any(mask):
        return grid

    indices = ndimage.distance_transform_edt(
        mask,
        return_indices=True,
    )[1]

    filled = grid[tuple(indices)]
    return filled


def _make_ghb_spd(
    domain: np.ndarray,
    ghb_mask: np.ndarray,
    dem: np.ndarray,
    hk: np.ndarray,
    grid_size: float,
    kriv_factor: float,
    width: float = None
):
    """
    Build GHB stress period data for a single steady period.

    :param domain: 2D array of active cells (1 active, 0 inactive)
    :param ghb_mask: 2D mask where True marks GHB cells
    :param dem: 2D DEM array (used as GHB stage)
    :param hk: 2D array of aquifer hydraulic conductivity [m/d] for riverbed K scaling
    :param grid_size: cell size [m]
    :param kriv_factor: factor for riverbed K relative to hk_scalar
    :param width: (optional) width for GHB conductance, defaults to grid_size
    :return: dict {0: [((k, i, j), stage, cond), ...]}
    """
    if width is None:
        width = grid_size

    # Mask for active cells with GHB boundaries
    mask = (domain == 1) & ghb_mask & np.isfinite(dem)
    i_idx, j_idx = np.where(mask)

    if i_idx.size == 0:
        return {0: []}

    # Prepare the kij array (k, i, j) for the GHB records
    n = i_idx.size
    kij = np.empty((n, 3), dtype=int)
    kij[:, 0] = 0  # Assuming single layer (layer 0)
    kij[:, 1] = i_idx
    kij[:, 2] = j_idx

    # Stage for the GHB boundary (using DEM)
    stage = dem[i_idx, j_idx].astype(float)

    # Compute conductance: kriv_factor * hk * grid_size * width
    # For each GHB cell, use the corresponding hk value
    k_riv = kriv_factor * hk[i_idx, j_idx]  # Use the hk array at specific indices
    cond = k_riv * grid_size * width  # Conductance for each cell

    # Prepare the GHB records
    ghb_records = []
    for k in range(n):
        ghb_records.append(
            (tuple(kij[k]), float(stage[k]), float(cond[k]))
        )

    return {0: ghb_records}


def _normalize_2d_field(
    value: float | np.ndarray | None,
    shape: tuple[int, int],
    default: float,
    name: str,
) -> np.ndarray:
    """
    Normalize a scalar or 2D field to a float array of shape (nrow, ncol).
    """
    if value is None:
        return np.full(shape, float(default), dtype=float)

    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(shape, float(arr), dtype=float)
    if arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}.")
    return arr.astype(float, copy=True)


def _normalize_layer_field(
    value: float | np.ndarray,
    nlay: int,
    shape: tuple[int, int],
    name: str,
) -> np.ndarray:
    """
    Normalize a scalar, 2D field, or 3D layer field to (nlay, nrow, ncol).
    """
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full((nlay, *shape), float(arr), dtype=float)
    if arr.shape == shape:
        return np.repeat(arr[np.newaxis, :, :], nlay, axis=0).astype(float, copy=True)
    if arr.shape == (nlay, *shape):
        return arr.astype(float, copy=True)
    raise ValueError(f"{name} has shape {arr.shape}, expected {shape} or {(nlay, *shape)}.")


def _create_chd_single_period_multilayer(
    boundary_heads: np.ndarray,
    active: np.ndarray,
    nlay: int,
):
    """
    Build CHD stress period data by applying the same 2D boundary heads to every layer.
    """
    heads = np.asarray(boundary_heads, dtype=float)
    act = np.asarray(active)

    if heads.shape != act.shape:
        raise ValueError("boundary_heads and active must have the same shape")

    mask = np.isfinite(heads) & (act > 0)
    i_idx, j_idx = np.where(mask)
    if i_idx.size == 0:
        return {0: []}

    vals = heads[i_idx, j_idx].astype(float)
    chd_records = []
    for layer in range(int(nlay)):
        for idx in range(i_idx.size):
            chd_records.append(
                ((layer, int(i_idx[idx]), int(j_idx[idx])), float(vals[idx]))
            )

    return {0: chd_records}


def _make_ghb_spd_multilayer(
    domain: np.ndarray,
    ghb_mask: np.ndarray,
    dem: np.ndarray,
    hk: np.ndarray,
    grid_size: float,
    kriv_factor: float,
    width: float = None,
):
    """
    Build GHB stress period data for every layer using the same 2D GHB mask.
    """
    if width is None:
        width = grid_size

    hk_arr = np.asarray(hk, dtype=float)
    if hk_arr.ndim != 3:
        raise ValueError("hk must be a 3D array with shape (nlay, nrow, ncol).")

    mask = (domain == 1) & ghb_mask & np.isfinite(dem)
    i_idx, j_idx = np.where(mask)
    if i_idx.size == 0:
        return {0: []}

    stage = dem[i_idx, j_idx].astype(float)
    ghb_records = []
    for layer in range(hk_arr.shape[0]):
        k_riv = kriv_factor * hk_arr[layer, i_idx, j_idx]
        cond = k_riv * grid_size * width
        for idx in range(i_idx.size):
            ghb_records.append(
                (
                    (int(layer), int(i_idx[idx]), int(j_idx[idx])),
                    float(stage[idx]),
                    float(cond[idx]),
                )
            )

    return {0: ghb_records}


def make_mf_model(
    nx: int = 1000,
    ny: int = 250,
    grid_size: float = 100.0,
    nper: int = 1,
    workspace=None,
    hk: float | np.ndarray = 300.0,
    recharge: float | np.ndarray | None = None,
    run: bool = True,
    use_ghb: bool = False,
    kriv_factor: float = 1.0,
    record_full_time: bool = False,
):
    """
    Build a simple steady MF6 model using CHD and optional GHB boundaries.

    :param nx: number of columns
    :param ny: number of rows
    :param grid_size: horizontal cell size [m]
    :param nper: number of stress periods
    :param workspace: path to output folder
    :param hk: uniform horizontal K [m/d] (scalar) or 2D K field (nrow, ncol)
    :param recharge: recharge [m/d], scalar or 2D (nrow, ncol). If None, uses 1e-4.
    :param run: if True, run MF6 and return heads
    :param use_ghb: if True, add GHB boundaries along the model center line
    :param kriv_factor: factor for riverbed K relative to hk for GHB
    :param record_full_time: if True, return total time including write + run + extract
    :return: (heads2d, engine_time) if record_full_time is False, else (heads2d, total_time)
    """
    if workspace is None:
        workspace = data_store.joinpath("mf6_truth")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    name = "model_truth"

    domain = _build_domain(nx=nx, ny=ny)
    dem = _build_dem(domain=domain)
    model_bottom = _model_bottom(dem=dem)
    dirichlet_mask = _build_dirichlet_boundary_mask(domain)
    ghb_mask = _build_ghb_boundary_masks(domain)

    nrow, ncol = domain.shape
    nlay = 1

    model_top = fill_nan_with_nearest(dem).astype(float)
    perioddata = [(1.0, 1, 1.0) for _ in range(int(nper))]

    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=str(require_mf6()),
        version="mf6",
        sim_ws=str(workspace),
    )

    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=int(nper),
        perioddata=perioddata,
    )

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=f"{name}.nam",
    )

    hclose = 1.0e-4
    rclose = 1.0e-6
    nouter = 10
    ninner = 2000

    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        linear_acceleration="CG",
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=[rclose, "RELATIVE_RCLOSE"],
        scaling_method="DIAGONAL",
    )
    sim.register_ims_package(ims, [gwf.name])

    delr = float(grid_size)
    delc = float(grid_size)

    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=model_top,
        botm=model_bottom.astype(float),
        idomain=domain.astype(int),
    )

    flopy.mf6.ModflowGwfic(
        gwf,
        pname="ic",
        strt=model_top,
    )

    if isinstance(hk, np.ndarray):
        hk_use = np.asarray(hk, dtype=float)
    else:
        hk_use = float(hk)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=[0],
        k=hk_use,
        k33=hk_use,
        k33overk=False,
        save_specific_discharge=True,
        save_saturation=True,
    )

    fixed_head_cells = dirichlet_mask.astype(float)
    fixed_head_cells[fixed_head_cells == 0.0] = np.nan
    fixed_head_cells = fixed_head_cells * model_top

    chd_spd = _create_chd_single_period(
        boundary_heads=fixed_head_cells,
        active=domain,
    )

    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="chd",
        stress_period_data=chd_spd,
        save_flows=True,
    )

    if recharge is None:
        recharge_grid = np.full((nrow, ncol), 1.0e-4, dtype=float)
    else:
        if isinstance(recharge, np.ndarray):
            recharge_grid = np.asarray(recharge, dtype=float)
            if recharge_grid.shape != (nrow, ncol):
                raise ValueError(f"recharge has shape {recharge_grid.shape}, expected {(nrow, ncol)}.")
        else:
            recharge_grid = np.full((nrow, ncol), float(recharge), dtype=float)

    recharge_grid = recharge_grid.copy()
    recharge_grid[domain == 0] = 0.0

    flopy.mf6.ModflowGwfrcha(
        gwf,
        pname="recharge",
        recharge=recharge_grid,
    )

    if use_ghb:
        if isinstance(hk_use, np.ndarray):
            hk_arr = hk_use
        else:
            hk_arr = np.full((nrow, ncol), hk_use, dtype=float)

        ghb_spd = _make_ghb_spd(
            domain=domain,
            ghb_mask=ghb_mask,
            dem=dem,
            hk=hk_arr,
            grid_size=grid_size,
            kriv_factor=kriv_factor,
        )
        if ghb_spd[0]:
            flopy.mf6.ModflowGwfghb(
                gwf,
                pname="ghb",
                stress_period_data=ghb_spd,
                save_flows=True,
            )

    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "LAST")],
        head_filerecord=[f"{name}.hds"],
        budget_filerecord=[f"{name}.cbb"],
        printrecord=[],
    )

    t_total_start = time.perf_counter()

    sim.write_simulation(silent=True)

    if not run:
        return None

    t_engine_start = time.perf_counter()
    ok, _ = sim.run_simulation(silent=True, report=False)
    t_engine_end = time.perf_counter()
    engine_time = t_engine_end - t_engine_start

    if not ok:
        warnings.warn("MF6 run failed")

    heads2d = _extract_heads(workspace.joinpath(f"{name}.hds"))

    t_total_end = time.perf_counter()
    total_time = t_total_end - t_total_start

    if record_full_time:
        return heads2d, float(total_time)

    return heads2d, float(engine_time)


def make_mf_model_multilayer(
    nx: int = 1000,
    ny: int = 200,
    nlay: int = 2,
    grid_size: float = 100.0,
    nper: int = 1,
    workspace=None,
    hk: float | np.ndarray = 10.0,
    vertical_k: float | np.ndarray | None = None,
    recharge: float | np.ndarray | None = None,
    layer_thickness: float | tuple[float, ...] | list[float] | np.ndarray = 150.0,
    run: bool = True,
    use_ghb: bool = False,
    kriv_factor: float = 1.0,
    record_full_time: bool = False,
    icelltype: int | list[int] = 0,
):
    """
    Build a simple multi-layer steady MF6 truth model using the shared model-builder tools.

    :param nx: number of columns
    :param ny: number of rows
    :param nlay: number of layers
    :param grid_size: horizontal cell size [m]
    :param nper: number of stress periods
    :param workspace: path to output folder
    :param hk: horizontal K [m/d], as scalar, 2D (nrow, ncol), or 3D (nlay, nrow, ncol)
    :param vertical_k: vertical K [m/d], same accepted shapes as hk. If None, uses hk.
    :param recharge: recharge [m/d], scalar or 2D (nrow, ncol). If None, uses 1e-4.
    :param layer_thickness: scalar layer thickness for every layer, or nlay positive layer thicknesses
    :param run: if True, run MF6 and return heads
    :param use_ghb: if True, add GHB boundaries along the model center line on both layers
    :param kriv_factor: factor for riverbed K relative to hk for GHB
    :param record_full_time: if True, return total time including write + run + extract
    :return: (heads3d, engine_time) if record_full_time is False, else (heads3d, total_time)
    """
    if workspace is None:
        workspace = data_store.joinpath("mf6_truth_multilayer")
    workspace = Path(workspace)
    workspace.mkdir(parents=True, exist_ok=True)

    name = "model_ml_truth"

    domain = _build_domain(nx=nx, ny=ny)
    dem = _build_dem(domain=domain)
    dirichlet_mask = _build_dirichlet_boundary_mask(domain)
    ghb_mask = _build_ghb_boundary_masks(domain)

    nrow, ncol = domain.shape
    nlay = int(nlay)
    shape2d = (nrow, ncol)

    thickness = np.asarray(layer_thickness, dtype=float)
    if thickness.ndim == 0:
        layer_thicknesses = np.full(nlay, float(thickness), dtype=float)
    elif thickness.shape == (nlay,):
        layer_thicknesses = thickness.astype(float, copy=True)
    else:
        raise ValueError(f"layer_thickness has shape {thickness.shape}, expected scalar or {(nlay,)}.")

    if not np.all(np.isfinite(layer_thicknesses)) or np.any(layer_thicknesses <= 0.0):
        raise ValueError("layer_thickness values must be positive finite numbers.")

    model_top = fill_nan_with_nearest(dem).astype(float)
    cumulative_thickness = np.cumsum(layer_thicknesses)
    botm = np.empty((nlay, nrow, ncol), dtype=float)
    for layer in range(nlay):
        botm[layer, :, :] = model_top - cumulative_thickness[layer]

    idomain = np.repeat(domain[np.newaxis, :, :], nlay, axis=0).astype(int)
    strt = np.repeat(model_top[np.newaxis, :, :], nlay, axis=0).astype(float)
    perioddata = [(1.0, 1, 1.0) for _ in range(int(nper))]

    hk_use = _normalize_layer_field(hk, nlay=nlay, shape=shape2d, name="hk")
    if not np.all(np.isfinite(hk_use)) or np.any(hk_use <= 0.0):
        raise ValueError("hk values must be positive finite numbers.")

    if vertical_k is None:
        k33_use = hk_use.copy()
    else:
        k33_use = _normalize_layer_field(vertical_k, nlay=nlay, shape=shape2d, name="vertical_k")
        if not np.all(np.isfinite(k33_use)) or np.any(k33_use <= 0.0):
            raise ValueError("vertical_k values must be positive finite numbers.")

    sim = flopy.mf6.MFSimulation(
        sim_name=name,
        exe_name=str(require_mf6()),
        version="mf6",
        sim_ws=str(workspace),
    )

    flopy.mf6.ModflowTdis(
        sim,
        pname="tdis",
        time_units="DAYS",
        nper=int(nper),
        perioddata=perioddata,
    )

    gwf = flopy.mf6.ModflowGwf(
        sim,
        modelname=name,
        model_nam_file=f"{name}.nam",
    )

    hclose = 1.0e-4
    rclose = 1.0e-6
    nouter = 10
    ninner = 2000

    ims = flopy.mf6.ModflowIms(
        sim,
        pname="ims",
        print_option="SUMMARY",
        complexity="SIMPLE",
        linear_acceleration="CG",
        outer_maximum=nouter,
        outer_dvclose=hclose,
        inner_maximum=ninner,
        inner_dvclose=hclose,
        rcloserecord=[rclose, "RELATIVE_RCLOSE"],
        scaling_method="DIAGONAL",
    )
    sim.register_ims_package(ims, [gwf.name])

    delr = float(grid_size)
    delc = float(grid_size)

    flopy.mf6.ModflowGwfdis(
        gwf,
        pname="dis",
        nlay=nlay,
        nrow=nrow,
        ncol=ncol,
        delr=delr,
        delc=delc,
        top=model_top,
        botm=botm,
        idomain=idomain,
    )

    flopy.mf6.ModflowGwfic(
        gwf,
        pname="ic",
        strt=strt,
    )

    if isinstance(icelltype, int):
        icelltype_list = [icelltype for _ in range(nlay)]
    else:
        icelltype_list = list(icelltype)

    flopy.mf6.ModflowGwfnpf(
        gwf,
        pname="npf",
        icelltype=icelltype_list,
        k=hk_use,
        k33=k33_use,
        k33overk=False,
        save_specific_discharge=True,
        save_saturation=True,
    )

    fixed_head_cells = dirichlet_mask.astype(float)
    fixed_head_cells[fixed_head_cells == 0.0] = np.nan
    fixed_head_cells = fixed_head_cells * model_top

    chd_spd = _create_chd_single_period_multilayer(
        boundary_heads=fixed_head_cells,
        active=domain,
        nlay=nlay,
    )

    flopy.mf6.ModflowGwfchd(
        gwf,
        pname="chd",
        stress_period_data=chd_spd,
        save_flows=True,
    )

    recharge_grid = _normalize_2d_field(
        recharge,
        shape=shape2d,
        default=1.0e-4,
        name="recharge",
    )
    recharge_grid[domain == 0] = 0.0

    flopy.mf6.ModflowGwfrcha(
        gwf,
        pname="recharge",
        recharge=recharge_grid,
    )

    if use_ghb:
        ghb_spd = _make_ghb_spd_multilayer(
            domain=domain,
            ghb_mask=ghb_mask,
            dem=dem,
            hk=hk_use,
            grid_size=grid_size,
            kriv_factor=kriv_factor,
        )
        if ghb_spd[0]:
            flopy.mf6.ModflowGwfghb(
                gwf,
                pname="ghb",
                stress_period_data=ghb_spd,
                save_flows=True,
            )

    flopy.mf6.ModflowGwfoc(
        gwf,
        pname="oc",
        saverecord=[("HEAD", "ALL"), ("BUDGET", "LAST")],
        head_filerecord=[f"{name}.hds"],
        budget_filerecord=[f"{name}.cbb"],
        printrecord=[],
    )

    t_total_start = time.perf_counter()

    sim.write_simulation(silent=True)

    if not run:
        return None

    t_engine_start = time.perf_counter()
    ok, _ = sim.run_simulation(silent=True, report=False)
    t_engine_end = time.perf_counter()
    engine_time = t_engine_end - t_engine_start

    if not ok:
        warnings.warn("MF6 multilayer run failed")

    heads3d = _extract_heads_3d(workspace.joinpath(f"{name}.hds"))

    t_total_end = time.perf_counter()
    total_time = t_total_end - t_total_start

    if record_full_time:
        return heads3d, float(total_time)

    return heads3d, float(engine_time)


def _extract_heads(hds_path: Path, totim: float = 1.0, plot=False):
    """
    Extract 2D head array from MF6 head file at specified time.
    :param hds_path: Path to MF6 head file
    :param totim: time at which to extract heads
    :return: 2D array of heads for layer 0
    """
    hdobj = flopy.utils.HeadFile(str(hds_path))
    heads = hdobj.get_data(totim=totim)

    heads = np.asarray(heads, dtype=float)
    heads[heads > 1.0e6] = np.nan  # Mask out any invalid head values

    heads2d = heads[0, :, :]

    # Simple diagnostic plot
    if plot:
        png_path = str(hds_path).replace(".hds", ".png")
        plt.imshow(heads2d, cmap="viridis")
        plt.colorbar(label="Head (m)")
        plt.title("Synthetic truth MF6 heads")
        plt.xlabel("Column index")
        plt.ylabel("Row index")
        plt.tight_layout()
        plt.savefig(png_path, dpi=200)
        plt.close()

    return heads2d


def _extract_heads_3d(hds_path: Path, totim: float = 1.0):
    """
    Extract a 3D head array from an MF6 head file at a specified time.
    """
    hdobj = flopy.utils.HeadFile(str(hds_path))
    heads = hdobj.get_data(totim=totim)

    heads = np.asarray(heads, dtype=float)
    heads[heads > 1.0e6] = np.nan

    return heads


def run_mf6_persistent_worker_batch_shm(
    worker_id: int,
    nx: int,
    ny: int,
    dx: float,
    hk_shm_name: str,
    hk_shape: tuple[int, int],
    hk_dtype: str,
    rch_shm_name: str,
    rch_shape: tuple[int, int, int],
    rch_dtype: str,
    case_start: int,
    case_end: int,
    run_root,
    ghb: bool,
    extract_heads: bool,
) -> dict:
    """
        Worker side MF6 persistent batch using shared memory inputs.

        Builds a template simulation once in a worker specific workspace, then runs
        cases in the half open interval [case_start, case_end), updating only recharge.

        :param worker_idx: Worker index for workspace naming.
        :param nx: Number of columns.
        :param ny: Number of rows.
        :param dx: Cell size.
        :param hk_shm_name: Shared memory name for hk array.
        :param hk_shape: Shape of hk array, (ny, nx).
        :param hk_dtype: Dtype string for hk array.
        :param rch_shm_name: Shared memory name for recharge stack.
        :param rch_shape: Shape of recharge stack, (n_cases, ny, nx).
        :param rch_dtype: Dtype string for recharge stack.
        :param case_start: First case index to run, inclusive.
        :param case_end: Last case index to run, exclusive.
        :param base_workspace: Folder in which worker workspaces are created.
        :param ghb: If True, include GHB boundaries.
        :param extract_heads: If True, read heads after each run.
        :return: Payload with timing arrays and counts for aggregation.
        """

    hk_shm = SharedMemory(name=str(hk_shm_name))
    rch_shm = SharedMemory(name=str(rch_shm_name))

    try:
        hk_field = np.ndarray(tuple(hk_shape), dtype=np.dtype(hk_dtype), buffer=hk_shm.buf)
        recharge_all = np.ndarray(tuple(rch_shape), dtype=np.dtype(rch_dtype), buffer=rch_shm.buf)

        if hk_field.shape != (int(ny), int(nx)):
            raise ValueError(f"hk_field has shape {hk_field.shape}, expected {(int(ny), int(nx))}.")
        if recharge_all.shape[1:] != (int(ny), int(nx)):
            raise ValueError(
                f"recharge_all has shape {recharge_all.shape}, expected (n_cases, {int(ny)}, {int(nx)})."
            )

        run_root = Path(run_root)
        ws_worker = run_root.joinpath(f"mf6_worker_{int(worker_id):03d}")
        ws_worker.mkdir(parents=True, exist_ok=True)

        if int(case_end) <= int(case_start):
            raise ValueError(f"Empty case range: case_start={case_start}, case_end={case_end}")

        # ---- Build template once (write full input set once) ----
        t_template0 = time.perf_counter()

        r0 = np.asarray(recharge_all[int(case_start), :, :], dtype=float)

        _ = make_mf_model(
            nx=int(nx),
            ny=int(ny),
            grid_size=float(dx),
            nper=1,
            workspace=ws_worker,
            hk=np.asarray(hk_field, dtype=float),
            recharge=r0,
            run=False,
            use_ghb=bool(ghb),
        )

        # Load the written simulation (same as your disk worker)
        mf6_exe = str(require_mf6())
        try:
            sim = flopy.mf6.MFSimulation.load(
                sim_ws=str(ws_worker),
                exe_name=mf6_exe,
                verbosity_level=0,
            )
        except TypeError:
            sim = flopy.mf6.MFSimulation.load(
                sim_ws=str(ws_worker),
                exe_name=mf6_exe,
            )

        model_names = list(sim.model_names)
        if len(model_names) < 1:
            raise RuntimeError("MFSimulation.load found no models in the workspace.")
        gwf = sim.get_model(model_names[0])

        rcha = gwf.get_package("recharge")
        if rcha is None:
            rcha = gwf.get_package("rcha")
        if rcha is None:
            raise RuntimeError("Could not find recharge package (RCHA) in loaded simulation.")

        template_build_time = float(time.perf_counter() - t_template0)

        # ---- Per-case timing arrays ----
        update_times: list[float] = []
        write_times: list[float] = []
        run_times: list[float] = []
        extract_times: list[float] = []
        total_times: list[float] = []

        name = gwf.name

        for case_id in range(int(case_start), int(case_end)):
            recharge_arr = np.asarray(recharge_all[int(case_id), :, :], dtype=float)

            # (1) update timing
            t_update0 = time.perf_counter()

            if recharge_arr.shape != (int(ny), int(nx)):
                raise ValueError(f"Recharge has shape {recharge_arr.shape}, expected {(int(ny), int(nx))}")

            try:
                rcha.recharge.set_data(recharge_arr, key=0)
            except TypeError:
                rcha.recharge.set_data({0: recharge_arr})

            t_update1 = time.perf_counter()
            update_times.append(float(t_update1 - t_update0))

            # (2) minimal write timing (mirror disk worker)
            t_write0 = time.perf_counter()

            wrote = False
            try:
                rcha.write()
                wrote = True
            except Exception:
                wrote = False

            if not wrote:
                try:
                    rcha.write_file()
                    wrote = True
                except Exception:
                    wrote = False

            if not wrote:
                sim.write_simulation()

            t_write1 = time.perf_counter()
            write_times.append(float(t_write1 - t_write0))

            # (3) run timing
            t_run0 = time.perf_counter()
            try:
                ok, _ = sim.run_simulation(silent=True, report=False)
            except TypeError:
                ok, _ = sim.run_simulation()
            t_run1 = time.perf_counter()
            run_times.append(float(t_run1 - t_run0))

            if not bool(ok):
                raise RuntimeError(f"MF6 failed for case_id={case_id}")

            # (4) extract timing
            t_ex0 = time.perf_counter()
            if bool(extract_heads):
                model_ws = Path(gwf.model_ws)

                # Get the actual head file configured by OC (preferred)
                try:
                    head_obj = gwf.output.head()
                    head_filename = Path(str(head_obj.filename))
                    if head_filename.is_absolute():
                        heads_path = head_filename
                    else:
                        heads_path = model_ws.joinpath(head_filename)
                except Exception:
                    # Fallback only if output control is not configured in a standard way
                    heads_path = model_ws.joinpath(f"{gwf.name}.hds")

                if not heads_path.exists():
                    # Useful diagnostics if something is misconfigured
                    found = list(model_ws.glob("*.hds"))
                    raise FileNotFoundError(
                        f"Head file not found. expected={heads_path} model_ws={model_ws} "
                        f"found_hds={[p.name for p in found]}"
                    )

                _ = _extract_heads(heads_path)
            t_ex1 = time.perf_counter()
            extract_times.append(float(t_ex1 - t_ex0))

            total_times.append(float(t_ex1 - t_update0))

        return {
            "worker_idx": int(worker_id),
            "workspace": str(ws_worker),
            "template_build_time": float(template_build_time),
            "n_cases": int(int(case_end) - int(case_start)),
            "case_start": int(case_start),
            "case_end": int(case_end),
            "update_times": update_times,
            "write_times": write_times,
            "run_times": run_times,
            "extract_times": extract_times,
            "total_times": total_times,
        }

    finally:
        hk_shm.close()
        rch_shm.close()


if __name__ == "__main__":
    ws = data_store.joinpath("Paper_mf6_truth")

    heads = make_mf_model(
        grid_size=100,
        nper=1,
        workspace=ws,
        run=True,
        use_ghb=True,  # set True if you want GHB boundaries as well
    )
