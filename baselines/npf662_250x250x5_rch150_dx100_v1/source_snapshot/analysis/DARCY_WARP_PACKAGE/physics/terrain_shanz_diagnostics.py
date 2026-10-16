"""Independent physical hydro-diagnostics for terrain groundwater models.

Pure NumPy reconstruction from a completed head field and the geometry:
a pressure-zero phreatic water table (distinguished from confined
potentiometric conditions), fixed-head coastal exchange split against
discharge-only spring outflow, and surface-discharge potential separated
into artesian pressure, water-table intersection and (when available)
actual surface exchange flux. Used by the SHANZ case study and its tests;
never consulted by the solver itself.
"""
import numpy as np


def water_table_profile(*, head, z_interfaces, surface):
    """Per-column pressure-zero water table and its interpretation.

    Scans each column top-down through the geological layers:

    * The first layer whose head reaches its top interface is fully
      saturated; if EVERY layer is saturated to its top the column is
      confined/potentiometric and no phreatic water table exists
      (``kind="potentiometric"``, elevation = NULL, the potentiometric
      surface equals the head of the shallowest saturated unit).
    * The first layer with bottom < head < top is partially saturated:
      the water table sits at z = head inside it (pressure zero).
    * The first layer with head <= bottom is dry; if every layer is dry
      the column has no water table (``kind="dry"``).

    Returns (elevation array with NaN where unresolvable, kind array of
    "water_table"|"potentiometric"|"dry", depth-to-water array with NaN
    where unresolvable).
    """
    head = np.asarray(head)
    nz, ny, nx = head.shape
    elevation = np.full((ny, nx), np.nan)
    kind = np.empty((ny, nx), dtype=object)
    for j in range(ny):
        for i in range(nx):
            column_kind = "dry"
            column_elevation = np.nan
            for k in range(nz):
                top = z_interfaces[k, j, i]
                bottom = z_interfaces[k+1, j, i]
                h = head[k, j, i]
                if h > top:
                    column_kind = "potentiometric"
                    column_elevation = np.nan
                    break
                if h > bottom:
                    # A head exactly at the top interface is the limiting
                    # phreatic case (water table at that elevation).
                    column_kind = "water_table"
                    column_elevation = float(h)
                    break
            kind[j, i] = column_kind
            elevation[j, i] = column_elevation
    depth = surface - elevation
    return elevation, kind, depth


def coastal_exchange(*, operator, head, fixed, spring_outflows=None):
    """Split coastal exchange into fixed-head and discharge-only parts.

    Fixed-head exchange is reconstructed from the authoritative face
    fluxes: every face between a free and a fixed cell transfers water
    into whichever side receives it. Returns gross fixed-head inflow and
    outflow (to the groundwater domain), optional spring outflow (the
    operator's discharge-only subfaces, outflow-positive), and the net
    coastal exchange (positive = net loss from the groundwater domain).
    """
    fluxes = operator.fluxes(head=head)
    fixed_mask = np.asarray(fixed) != 0
    inflow = 0.0
    outflow = 0.0
    for axis, flux in zip((2, 1, 0), fluxes[:3]):
        left, right = [slice(None)]*3, [slice(None)]*3
        left[axis], right[axis] = slice(None, -1), slice(1, None)
        left, right = tuple(left), tuple(right)
        # Flux sign: positive along increasing axis (toward the right cell).
        exchange_right = np.where(fixed_mask[right] & ~fixed_mask[left], flux, 0.0)
        exchange_left = np.where(fixed_mask[left] & ~fixed_mask[right], -flux, 0.0)
        # Positive exchange_right = into the fixed cell = out of the domain.
        outflow += float(np.sum(np.maximum(exchange_right, 0.0)))
        outflow += float(np.sum(np.maximum(exchange_left, 0.0)))
        inflow += float(np.sum(np.maximum(-exchange_right, 0.0)))
        inflow += float(np.sum(np.maximum(-exchange_left, 0.0)))
    springs = float(np.sum(spring_outflows)) if spring_outflows is not None \
        else float(np.sum(np.maximum(fluxes[3], 0.0)))
    net = outflow + springs - inflow
    return dict(fixed_head_inflow=inflow, fixed_head_outflow=outflow,
                spring_outflow=springs, net_coastal_exchange=net,
                fixed_head_cells=int(np.count_nonzero(fixed_mask)))


def surface_discharge_potential(*, head, z_interfaces, surface):
    """Separate artesian potential from water-table surface intersection.

    * ``artesian_cells``: columns whose shallowest head exceeds the ground
      surface (potentiometric overshoot) while every layer is saturated.
    * ``water_table_at_surface``: columns with a phreatic water table at
      or above the ground surface.
    * ``deep_head_above_surface``: columns where any DEEP layer's head
      exceeds the ground surface although a phreatic water table exists
      above it (a leaky artesian indicator, NOT actual discharge).
    """
    elevation, kind, unused_depth = water_table_profile(
        head=head, z_interfaces=z_interfaces, surface=surface)
    shallow_head = head[0]
    ny, nx = shallow_head.shape
    artesian = np.zeros((ny, nx), dtype=bool)
    deep_above = np.zeros((ny, nx), dtype=bool)
    for j in range(ny):
        for i in range(nx):
            if kind[j, i] == "potentiometric" and shallow_head[j, i] > surface[j, i]:
                artesian[j, i] = True
            if kind[j, i] == "water_table":
                if elevation[j, i] >= surface[j, i]:
                    deep_above[j, i] = True
                elif np.any(head[:, j, i] > surface[j, i]):
                    deep_above[j, i] = True
            elif kind[j, i] == "potentiometric":
                # Dry-shallow over pressurised deep layers: leaky-artesian
                # indicator, not actual discharge.
                if np.any(head[:, j, i] > surface[j, i]):
                    deep_above[j, i] = True
    at_surface = np.zeros((ny, nx), dtype=bool)
    at_surface[elevation >= surface] = True
    at_surface &= (kind == "water_table")
    return dict(artesian_cells=int(np.count_nonzero(artesian)),
        water_table_at_surface_cells=int(np.count_nonzero(at_surface)),
        deep_head_above_surface_cells=int(np.count_nonzero(deep_above & ~at_surface & ~artesian)),
        artesian_mask=artesian, water_table_at_surface_mask=at_surface,
        deep_head_mask=deep_above)
