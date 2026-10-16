from scipy.ndimage import gaussian_filter
import numpy as np

"""
This builds the model domain, np arrays, boundary fixed head cells, general
head boundary cells etc for use in both MODFLOW and the differentiable model.
"""


def _build_domain(nx: int = 3000, ny: int = 333):
    """
    Build a 1 layer np array of dimensions (ny, nx) representing the model domain.

    :param nx: number of columns
    :param ny: number of rows
    :return: np.array of shape (ny, nx) with 1 for active cells
    """
    domain = np.ones((ny, nx), dtype=np.int8)

    # Example: set some cells to inactive (0)
    # domain[100:150, 50:100] = 0

    return domain


def _build_dirichlet_boundary_mask(domain: np.ndarray):
    """
    Build a mask for Dirichlet boundary conditions where boundaries are
    on the north, south and east edges of the model.

    0 = no dirichlet boundary
    1 = dirichlet boundary

    :param domain: np.array of shape (ny, nx)
    :return: mask array of shape (ny, nx) with True where Dirichlet applies
    """
    ny, nx = domain.shape
    dirichlet_mask = np.zeros((ny, nx), dtype=bool)

    # North boundary
    dirichlet_mask[0, :] = True
    # South boundary
    dirichlet_mask[-1, :] = True
    # East boundary
    dirichlet_mask[:, -1] = True

    # Only on active cells
    dirichlet_mask = dirichlet_mask & (domain == 1)

    return dirichlet_mask


def _build_ghb_boundary_masks(domain: np.ndarray):
    """
    Build a mask for general head boundary conditions.

    Here we assume a "river" cell 100 m by 100 m running west to east along
    the center row of the model domain.

    :param domain: np.array of shape (ny, nx)
    :return: mask array of shape (ny, nx) with True where GHB applies
    """
    ny, nx = domain.shape
    ghb_mask = np.zeros((ny, nx), dtype=bool)

    # Center row (integer index)
    center_row = (ny - 1) // 2
    ghb_mask[center_row, :] = True

    # Only on active cells
    ghb_mask = ghb_mask & (domain == 1)

    return ghb_mask


def _build_dem(domain: np.ndarray):
    """
    Build a simple DEM for the model domain on an (ny, nx) grid.
    Max elevation is in the west and min in the east, uniform in y.

    :param domain: np.array of shape (ny, nx)
    :return: dem array of shape (ny, nx)
    """
    ny, nx = domain.shape

    # Linear gradient in x from 300 m in the west to 0 m in the east
    x_frac = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    col_elev = 300.0 * (1.0 - x_frac)  # 300 -> 0
    dem = np.tile(col_elev, (ny, 1)).astype(np.float32)

    # Optional: set DEM outside the domain to NaN
    dem[domain != 1] = np.nan

    return dem


def _model_bottom(dem: np.ndarray):
    """
    Build a simple model bottom that is 300 m below the DEM.

    :param dem: np.array of shape (ny, nx)
    :return: bottom array of shape (ny, nx)
    """
    bottom = dem - 300.0
    return bottom


def _create_chd_single_period(
    boundary_heads: np.ndarray,  # 2D (nrow, ncol) heads on layer 0; NaN = no CHD
    active: np.ndarray,          # 2D (nrow, ncol) active cell mask; > 0 = active
):
    """
    Build MODFLOW-6 CHD stress_period_data for a single stress period (nper = 1).

    :param boundary_heads: 2D array (nrow, ncol) with constant heads on layer 0.
                           Use np.nan where no CHD is defined.
    :param active: 2D array (nrow, ncol) with active-cell flags (> 0 means active).
    :return: dict {0: [((k, i, j), head), ...]} for the CHD package.
    """
    heads = np.asarray(boundary_heads, dtype=float)
    act = np.asarray(active)

    if heads.shape != act.shape:
        raise ValueError("boundary_heads and active must have the same shape")

    # CHD cells are finite heads in active cells on layer 0
    mask = np.isfinite(heads) & (act > 0)
    i_idx, j_idx = np.where(mask)

    if i_idx.size == 0:
        return {0: []}

    n = i_idx.size
    kij = np.empty((n, 3), dtype=int)
    kij[:, 0] = 0
    kij[:, 1] = i_idx
    kij[:, 2] = j_idx

    vals = heads[i_idx, j_idx].astype(float)

    # Build records for period 0
    chd_records = [
        (tuple(kij[k]), float(vals[k]))
        for k in range(n)
    ]

    return {0: chd_records}



def make_ugly_T_field(nx, ny, domain=None, seed=42):
    """
    :param nx: int number of columns
    :param ny: int number of rows
    :param domain: optional np.ndarray (ny, nx) active mask, 1 active, 0 inactive
    :param seed: int random seed for reproducibility
    :return: np.ndarray (ny, nx) heterogeneous transmissivity field [L2/T]
    """
    rng = np.random.default_rng(seed)

    y = np.linspace(0.0, 1.0, ny, dtype=np.float32)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    X, Y = np.meshgrid(x, y)

    # Base around your old uniform value 3000 m2/d
    base_logT = np.log10(3000.0)

    # Large scale gradient: higher T in west and mid domain, lower in east and corners
    large_scale = 0.5 * (1.0 - X) + 0.3 * (Y - 0.5)

    # Two scale correlated noise fields
    noise_large = rng.standard_normal((ny, nx))
    noise_small = rng.standard_normal((ny, nx))

    noise_large = gaussian_filter(noise_large, sigma=30.0)
    noise_small = gaussian_filter(noise_small, sigma=5.0)

    # Combine to log10(T)
    logT = base_logT + 0.5 * large_scale + 0.4 * noise_large + 0.2 * noise_small

    # High T channel roughly diagonal
    channel_mask = np.abs(Y - (0.3 + 0.2 * X)) < 0.02
    logT[channel_mask] = logT[channel_mask] + 1.0  # 10x boost

    # Low T lenses
    lens1 = (X - 0.25) ** 2 + (Y - 0.70) ** 2 < 0.01
    lens2 = (X - 0.80) ** 2 + (Y - 0.30) ** 2 < 0.005
    logT[lens1] = logT[lens1] - 1.0   # 0.1x
    logT[lens2] = logT[lens2] - 0.7   # ~0.2x

    # Back to linear T, with clipping
    T_field = np.power(10.0, logT)
    T_field = np.clip(T_field, 1.0, 1.0e5)

    if domain is not None:
        domain_bool = np.asarray(domain, dtype=bool)
        T_field = np.where(domain_bool, T_field, 0.0)

    return T_field.astype(np.float32)


def build_base_fields(nx, ny, dx):
    """
    :param nx: number of columns
    :param ny: number of rows
    :param dx: cell size
    :return: (domain, dem, T_field, R_field)
    """
    domain = _build_domain(nx=nx, ny=ny)
    dem = _build_dem(domain)
    _ = _model_bottom(dem)

    T_field_ugly = make_ugly_T_field(
        nx=nx,
        ny=ny,
        domain=domain,
        seed=123,
    )

    R_field = np.full_like(domain, 1.0e-4, dtype=np.float64)
    return domain, dem, T_field_ugly, R_field


def build_truth_inputs(
        nx: int,
        ny: int,
        dx: float,
        T_truth: float | np.ndarray,
        R_truth: float | np.ndarray = 1.0e-4,
        use_ghb: bool = True,
        width: float = 100.0,
) -> tuple:
    """
    Build a synthetic forward problem matching your MF6 truth setup.

    :param nx: number of columns
    :param ny: number of rows
    :param dx: cell size
    :param T_truth: scalar or array transmissivity
    :param R_truth: scalar or array recharge
    :param use_ghb: switch GHB on or off
    :return: (T_field, R_field, active, bc_mask, bc_values, gh_mask, gh_head, gh_width)
    """
    domain = _build_domain(nx=nx, ny=ny)
    active = domain.astype(np.int32)

    dem = _build_dem(domain)
    bottom = _model_bottom(dem)
    _ = bottom

    dirichlet_mask = _build_dirichlet_boundary_mask(domain)

    if use_ghb and _build_ghb_boundary_masks is not None:
        gh_mask_bool = _build_ghb_boundary_masks(domain)
    else:
        gh_mask_bool = np.zeros_like(domain, dtype=bool)

    bc_values = np.zeros_like(dem, dtype=np.float64)
    bc_values[dirichlet_mask] = dem[dirichlet_mask]

    gh_head = np.zeros_like(dem, dtype=np.float64)
    gh_width = np.zeros_like(dem, dtype=np.float64)
    if use_ghb and _build_ghb_boundary_masks is not None:
        gh_head[gh_mask_bool] = dem[gh_mask_bool]
        gh_width[gh_mask_bool] = width

    # Transmissivity
    if np.isscalar(T_truth):
        T_field = np.full_like(dem, float(T_truth), dtype=np.float64)
    else:
        T_arr = np.asarray(T_truth, dtype=np.float64)
        if T_arr.shape != dem.shape:
            raise ValueError(
                f"T_truth shape {T_arr.shape} does not match DEM/domain shape {dem.shape}"
            )
        T_field = T_arr

    # Recharge
    if np.isscalar(R_truth):
        R_field = np.full_like(dem, float(R_truth), dtype=np.float64)
    else:
        R_arr = np.asarray(R_truth, dtype=np.float64)
        if R_arr.shape != dem.shape:
            raise ValueError(
                f"R_truth shape {R_arr.shape} does not match DEM/domain shape {dem.shape}"
            )
        R_field = R_arr

    return (
        T_field,
        R_field,
        active,
        dirichlet_mask.astype(np.int32),
        bc_values,
        gh_mask_bool.astype(np.int32),
        gh_head,
        gh_width,
    )

def compare_head_fields(
        head_ref: np.ndarray,
        head_warp: np.ndarray,
        active_mask: np.ndarray | None = None,
) -> dict:
    """
    Compare a reference solution (FD or MF6) and Warp heads.

    :param head_ref: reference heads [ny, nx] or [nlay, ny, nx]
    :param head_warp: Warp heads [ny, nx] or [nlay, ny, nx]
    :param active_mask: optional active mask
    :return: metrics dict
    """
    head_ref = np.asarray(head_ref, dtype=float)
    head_warp = np.asarray(head_warp, dtype=float)

    if head_ref.ndim == 3:
        head_ref = head_ref[0]
    if head_warp.ndim == 3:
        head_warp = head_warp[0]

    if head_ref.shape != head_warp.shape:
        raise ValueError(
            f"Shape mismatch: ref {head_ref.shape}, warp {head_warp.shape}"
        )

    ny, nx = head_ref.shape

    if active_mask is None:
        mask = np.ones((ny, nx), dtype=bool)
    else:
        mask = np.asarray(active_mask).astype(bool)
        if mask.shape != (ny, nx):
            raise ValueError(
                f"active_mask shape {mask.shape} does not match heads {head_ref.shape}"
            )

    diff = head_warp - head_ref
    diff = np.where(mask, diff, np.nan)

    abs_diff = np.abs(diff)
    sq_diff = diff * diff

    rmse = float(np.sqrt(np.nanmean(sq_diff)))
    max_abs = float(np.nanmax(abs_diff))
    mean_bias = float(np.nanmean(diff))

    p_within_0_1 = float(np.nanmean(abs_diff <= 0.1) * 100.0)
    p_within_0_5 = float(np.nanmean(abs_diff <= 0.5) * 100.0)
    p_within_1_0 = float(np.nanmean(abs_diff <= 1.0) * 100.0)

    metrics = {
        "rmse": rmse,
        "max_abs_diff": max_abs,
        "mean_bias_warp_minus_ref": mean_bias,
        "percent_within_0_1m": p_within_0_1,
        "percent_within_0_5m": p_within_0_5,
        "percent_within_1_0m": p_within_1_0,
    }

    print("Reference vs Warp head comparison (Warp minus Ref):")
    print(f"  RMSE               : {rmse:.5f} m")
    print(f"  Max abs difference : {max_abs:.5f} m")
    print(f"  Mean bias          : {mean_bias:.5f} m")
    print(f"  |diff| <= 0.1 m    : {p_within_0_1:6.2f} % of active cells")
    print(f"  |diff| <= 0.5 m    : {p_within_0_5:6.2f} % of active cells")
    print(f"  |diff| <= 1.0 m    : {p_within_1_0:6.2f} % of active cells")

    return metrics