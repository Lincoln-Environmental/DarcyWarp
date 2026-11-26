"""Session-owned accepted NPF equations and exchange outputs.

The public token is an opaque lookup key. It grants no authority to replace
owned equation inputs or outputs with caller-provided arrays or hashes.
"""
from dataclasses import dataclass
from uuid import uuid4

import numpy as np
import warp as wp


@dataclass(frozen=True, eq=False)
class AcceptedNPFState:
    """Private snapshot retained by its session, including resident equation fields.

    :param head: Owned accepted device head.
    :param fields: Owned device equation inputs and exchange outputs.
    :param dt: Actual timestep duration, or None for steady flow.
    :param storage_factor: Actual storage volume/time multiplier.
    :param shape: Groundwater grid shape.
    """

    head: object
    fields: dict
    dt: float | None
    storage_factor: float
    shape: tuple


def own_accepted_state(*, session, workspace, dt):
    """Snapshot the solved equation without downloading resident grid fields.

    :param session: Unique owning operator session.
    :param workspace: Accepted numerical workspace; trials are excluded.
    :param dt: Actual accepted duration or None.
    :return: JSON-compatible opaque token for the session-owned snapshot.
    """
    names = ('source', 'prescribed', 'free', 'fixed', 'bottom', 'thickness',
             'types', 'c0', 'storage_previous', 'storage_ss', 'storage_sy',
             'ghb_conductance', 'ghb_head', 'drn_offsets', 'drn_entry_c',
             'drn_entry_elev', 'riv_offsets', 'riv_entry_c', 'riv_entry_stage',
             'riv_entry_bottom', 'drn_rate', 'riv_rate', 'drn_entry_rate', 'riv_entry_rate')
    fields = {}
    for name in names:
        field = getattr(workspace, name, None)
        if field is not None:
            fields[name] = wp.clone(src=field)
    state = AcceptedNPFState(head=wp.clone(src=workspace.head), fields=fields,
        dt=dt, storage_factor=workspace.storage_factor, shape=workspace.shape)
    if not hasattr(session, '_accepted_equations'):
        session._accepted_equations = {}
    identifier = uuid4().hex
    session._accepted_equations[identifier] = state
    return dict(identifier=identifier, dt=dt)


def verified_package_rates(*, session, head, info, package):
    """Verify supplied metadata against owned historical outputs on extraction.

    :param session: The session that accepted the state.
    :param head: Supplied accepted groundwater head.
    :param info: Accepted public solve metadata.
    :param package: Drain or river package name in lowercase.
    :return: Owned copy of the accepted per-cell discharge array.
    :raises ValueError: If any supplied head, discharge, entry rate or duration differs.
    :raises RuntimeError: If the state is rejected, absent or belongs to another session.
    """
    if not isinstance(info, dict) or not info.get('converged', False):
        raise RuntimeError('Package exchange requires an accepted solve')
    token = info.get('accepted_state', {})
    identifier = token.get('identifier') if isinstance(token, dict) else None
    state = getattr(session, '_accepted_equations', {}).get(identifier)
    if state is None:
        raise RuntimeError('Accepted state does not belong to this model session')
    if token.get('dt') != state.dt or ('dt' in info and info['dt'] != state.dt):
        raise ValueError('Timestep metadata does not match the accepted equation')
    accepted_head = state.head.numpy().reshape(state.shape)
    if not np.array_equal(np.asarray(head), accepted_head):
        raise ValueError('head does not match the accepted solve state')
    rate = state.fields[f'{package}_rate'].numpy().reshape(state.shape)
    supplied = info.get(f'{package}_discharge_rate_array')
    if supplied is None or not np.array_equal(np.asarray(supplied), rate):
        raise ValueError('Discharge does not match the owned accepted state')
    for name in ('drn', 'riv'):
        owned = state.fields[f'{name}_entry_rate'].numpy()
        supplied_entries = info.get(f'{name}_entry_rates')
        if owned.size and (supplied_entries is None or not np.array_equal(np.asarray(supplied_entries), owned)):
            raise ValueError('Entry rates do not match the owned accepted state')
        owned_cells = state.fields[f'{name}_rate'].numpy().reshape(state.shape)
        supplied_cells = info.get(f'{name}_discharge_rate_array')
        if supplied_cells is None or not np.array_equal(np.asarray(supplied_cells), owned_cells):
            raise ValueError('Cell rates do not match the owned accepted state')
    return rate.copy()
