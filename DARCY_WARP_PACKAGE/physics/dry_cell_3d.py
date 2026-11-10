"""Flow diagnostics for the quasi-3D saturation floor; not unsaturated physics."""
import numpy as np


def quantify_dry_cell_transmission(*, head, bottoms, tops, actual_faces,
                                  unfloored_faces, active, bc_mask, source, budget):
    """Evaluate floor-induced horizontal flux at the same final head field.

    Vertical dry-interface flux is reported separately: it measures the
    saturation-independent assumption, not an error against a Richards model.
    Pairwise magnitudes count each internal face once; they cannot cancel in
    a domain residual. Ratios with no recharge/abstraction are undefined.
    """
    act = np.asarray(active) != 0
    free = act & (np.asarray(bc_mask) == 0)
    true_sat = np.clip(head - bottoms, 0.0, tops - bottoms)
    horizontal = 0.0
    vertical_dry = 0.0
    for axis, index in ((2, 0), (1, 2), (0, 4)):
        here, there = [slice(None)] * 3, [slice(None)] * 3
        here[axis], there[axis] = slice(None, -1), slice(1, None)
        here, there = tuple(here), tuple(there)
        participates = act[here] & act[there] & (free[here] | free[there])
        difference = np.abs(head[here] - head[there])
        if axis != 0:
            excess = actual_faces[index][here] - unfloored_faces[index][here]
            horizontal += float(np.sum(np.where(participates, np.maximum(excess, 0.0) * difference, 0.0)))
        else:
            dry = (true_sat[here] == 0.0) | (true_sat[there] == 0.0)
            vertical_dry += float(np.sum(np.where(participates & dry, actual_faces[index][here] * difference, 0.0)))
    injection = float(np.sum(np.maximum(source[free], 0.0)))
    abstraction = float(np.sum(np.maximum(-source[free], 0.0)))
    throughput = budget["throughput"]
    return dict(horizontal_floor_excess_flow=horizontal,
                vertical_dry_interface_flow=vertical_dry,
                horizontal_floor_to_recharge=horizontal / injection if injection > 0.0 else None,
                horizontal_floor_to_abstraction=horizontal / abstraction if abstraction > 0.0 else None,
                horizontal_floor_to_budget_throughput=horizontal / throughput if throughput > 0.0 else None,
                recharge=injection, abstraction=abstraction, budget_throughput=throughput,
                basis="same-head face sensitivity; vertical dry transmission is a model-assumption diagnostic")
