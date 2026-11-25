"""Compare converged 2D unconfined heads with native MF6 face budgets."""
from pathlib import Path

import numpy as np
from flopy.mf6.utils.binarygrid_util import MfGrdFile
from flopy.utils import CellBudgetFile, HeadFile

from DARCY_WARP_PACKAGE.validation.time_grid import same_step_times
from DARCY_WARP_PACKAGE.validation.head_checks import invalid_head_values
from working_tests.run_2d_transient_vs_mf6 import MF6_MODEL_NAME
from working_tests.transient_replay_mass_balance import transmissivity_from_period_head
from working_tests.transient_replay_settings import DEFAULT_MIN_SAT


def compare_native_face_flows(*, native_workspace, heads, artifact, min_sat=DEFAULT_MIN_SAT):
    """Audit each unique active face touching a free cell at each saved endpoint.

    The single-layer DIS replay uses square cells. In this geometry face width
    cancels the centre-to-centre distance. Positive flow here goes from the
    lower user node number to the higher one; MF6 reports inflow to a node.
    """
    base = Path(native_workspace) / MF6_MODEL_NAME
    grid = MfGrdFile(str(base.with_suffix('.dis.grb')))
    active = np.asarray(artifact['active']) != 0
    fixed = np.asarray(artifact['bc_mask']) != 0
    reference_heads = artifact['heads_per_period']
    if tuple(grid.shape) != (1, *active.shape) or np.shape(heads) != np.shape(reference_heads):
        raise ValueError('Native grid or replay head history shape mismatch')
    if not np.all(grid.delr == artifact['dx']) or not np.all(grid.delc == artifact['dx']):
        raise ValueError('Native square-cell spacing differs from the replay')
    if not np.array_equal(np.asarray(grid.idomain).reshape(active.shape), active):
        raise ValueError('Native active domain differs from the replay')
    node = np.repeat(np.arange(active.size), np.diff(grid.ia))
    other = grid.ja
    eligible = active.ravel() & ~fixed.ravel()
    positions = np.flatnonzero((node < other) & (eligible[node] | eligible[other]))
    left, right = node[positions], other[positions]
    # Verify that the decoded records cover every expected horizontal and vertical face.
    ids = np.arange(active.size).reshape(active.shape)
    keys = []
    for lower, upper in ((ids[:, :-1], ids[:, 1:]), (ids[:-1], ids[1:])):
        mask = (active.ravel()[lower] & active.ravel()[upper]
                & (eligible[lower] | eligible[upper]))
        keys.append(lower[mask] * active.size + upper[mask])
    expected_edges = np.sort(np.concatenate(keys))
    if not positions.size or not np.array_equal(np.sort(left * active.size + right), expected_edges):
        raise ValueError('Native face records differ from the active free-cell connections')
    times = float(artifact['dt_days']) * np.arange(1, len(heads) + 1)
    rows = []
    with HeadFile(str(base.with_suffix('.hds'))) as native_heads, CellBudgetFile(str(base.with_suffix('.cbb'))) as budget:
        endpoints = budget.get_kstpkper()
        if (not same_step_times(saved=budget.get_times(), expected=times)
                or not same_step_times(saved=native_heads.get_times(), expected=times)
                or endpoints != [(0, i) for i in range(len(heads))]):
            raise ValueError('Native head/budget times differ from the replay')
        for index, step in enumerate(endpoints):
            source_head = native_heads.get_data(kstpkper=step, mflay=0)
            if np.any(invalid_head_values(head=source_head[active])) or not np.array_equal(source_head[active], reference_heads[index][active]):
                raise ValueError(f'Native head file differs from reference artifact at endpoint {index + 1}')
            h = np.asarray(heads[index], dtype=float)
            if np.any(invalid_head_values(head=h[active])):
                raise ValueError(f'Invalid active replay head at endpoint {index + 1}')
            t = transmissivity_from_period_head(formulation='unconfined', head=h,
                k=artifact['k_field'], top=artifact['top'], bottom=artifact['bottom'],
                active=active, min_sat=min_sat).ravel()
            denominator = t[left] + t[right]
            conductance = np.divide(2. * t[left] * t[right], denominator,
                out=np.zeros_like(denominator), where=denominator > 0.)
            candidate = conductance * (h.ravel()[left] - h.ravel()[right])
            records = budget.get_data(kstpkper=step, text='FLOW-JA-FACE')
            if len(records) != 1 or records[0].size != grid.ja.size:
                raise ValueError('Missing or ambiguous native FLOW-JA-FACE record')
            native = -records[0].ravel()[positions]
            if not np.isfinite(native).all() or not np.isfinite(candidate).all():
                raise ValueError('Non-finite native or reconstructed face flow')
            error = candidate - native
            rms = float(np.sqrt(np.mean(error ** 2)))
            reference_rms = float(np.sqrt(np.mean(native ** 2)))
            if reference_rms == 0. and rms != 0.:
                raise ValueError(f'Nonzero reconstructed flow on a zero-flow native field at endpoint {index + 1}')
            rows.append(dict(time_days=float(times[index]), rms_m3_day=rms,
                relative_l2=rms / reference_rms if reference_rms else 0.,
                max_face_error_m3_day=float(np.max(np.abs(error)))))
    return dict(compared_faces=int(positions.size), periods=rows, native_record='FLOW-JA-FACE',
        fixed_to_fixed_faces_excluded=True, worst_rms_m3_day=max(row['rms_m3_day'] for row in rows),
        worst_relative_l2=max(row['relative_l2'] for row in rows),
        worst_cell_face_error_m3_day=max(row['max_face_error_m3_day'] for row in rows))
