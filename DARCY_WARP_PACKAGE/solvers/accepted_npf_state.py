"""Session-owned accepted NPF equations and exchange outputs.

The public token is an opaque lookup key. It grants no authority to replace
owned equation inputs or outputs with caller-provided arrays or hashes.
"""
from dataclasses import dataclass
from uuid import uuid4
import weakref

import numpy as np
import warp as wp


@dataclass(frozen=True)
class _RetainedStateOwner:
    """Shared lifetime owner for one immutable accepted-state identifier."""

    _session: object
    _identifier: str

    def __init__(self, *, session, identifier):
        object.__setattr__(self, '_session', weakref.ref(session))
        object.__setattr__(self, '_identifier', str(identifier))

    def __del__(self):
        session = self._session()
        if session is not None:
            getattr(session, '_accepted_equations', {}).pop(self._identifier, None)


class RetainedStateToken(dict):
    """Opaque public token whose lifetime owns one session snapshot."""

    def __init__(self, *, session=None, identifier=None, dt=None, owner=None):
        if owner is None:
            owner = _RetainedStateOwner(session=session, identifier=identifier)
        identifier = owner._identifier
        super().__init__(identifier=identifier, dt=dt)
        self._owner = owner

    def __copy__(self):
        """A shallow diagnostic copy shares ownership of the retained state."""
        return type(self)(owner=self._owner, dt=self.get('dt'))

    def __deepcopy__(self, memo):
        """Diagnostic copies verify the live original without acquiring ownership."""
        return dict(self)


def retained_state_identifier(*, token):
    """Return the private immutable identifier carried by an accepted token."""
    if isinstance(token, RetainedStateToken):
        return token._owner._identifier
    return token.get('identifier') if isinstance(token, dict) else None


class RetainedStateIdentifier(str):
    """String-compatible identifier owning one model-side source snapshot."""

    def __new__(cls, value, *, owner, attribute):
        instance = super().__new__(cls, value)
        instance._owner = weakref.ref(owner)
        instance._attribute = attribute
        return instance

    def __deepcopy__(self, memo):
        return self

    def __del__(self):
        owner = self._owner()
        if owner is not None:
            getattr(owner, self._attribute, {}).pop(str(self), None)


@dataclass(frozen=True, eq=False)
class AcceptedNPFState:
    """Private snapshot retained by its session, including resident equation fields.

    :param head: Owned accepted device head.
    :param fields: Owned device equation inputs and exchange outputs.
    :param dt: Actual timestep duration, or None for steady flow.
    :param storage_factor: Actual storage volume/time multiplier.
    :param shape: Groundwater grid shape.
    :param resident_outputs: True when full exchange arrays remain on device.
    """

    head: object
    fields: dict
    dt: float | None
    storage_factor: float
    shape: tuple
    resident_outputs: bool
    enabled_packages: tuple


def own_accepted_state(*, session, workspace, dt, resident_outputs):
    """Snapshot the solved equation without downloading resident grid fields.

    :param session: Unique owning operator session.
    :param workspace: Accepted numerical workspace; trials are excluded.
    :param dt: Actual accepted duration or None.
    :param resident_outputs: Keep output arrays resident until explicitly requested.
    :return: JSON-compatible opaque token for the session-owned snapshot.
    """
    static_names = ('prescribed', 'free', 'fixed', 'bottom', 'thickness',
        'types', 'c0', 'left', 'right', 'storage_ss', 'storage_sy', 'ghb_conductance',
        'ghb_head', 'drn_offsets', 'drn_entry_c', 'drn_entry_elev', 'riv_offsets',
        'riv_entry_c', 'riv_entry_stage', 'riv_entry_bottom')
    static = getattr(workspace, '_accepted_static_fields', None)
    if static is None:
        static = {}
        for name in static_names:
            field = getattr(workspace, name, None)
            if field is not None:
                static[name] = wp.clone(src=field)
        workspace._accepted_static_fields = static
    # A public load invalidates static ownership. Prepared resident steps have
    # fixed K, topology, boundary definitions, types and storage parameters, so
    # their accepted states safely share this independently owned snapshot.
    fields = dict(static)
    for name in ('source', 'source_gross', 'storage_previous', 'drn_rate', 'riv_rate', 'drn_entry_rate', 'riv_entry_rate'):
        field = getattr(workspace, name, None)
        if field is not None:
            fields[name] = wp.clone(src=field)
    state = AcceptedNPFState(head=wp.clone(src=workspace.head), fields=fields,
        dt=dt, storage_factor=workspace.storage_factor, shape=workspace.shape,
        resident_outputs=resident_outputs,
        enabled_packages=tuple(name for name, enabled in
            (('drn', workspace.has_drn), ('riv', workspace.has_riv)) if enabled))
    if not hasattr(session, '_accepted_equations'):
        session._accepted_equations = {}
    identifier = uuid4().hex
    session._accepted_equations[identifier] = state
    return RetainedStateToken(session=session, identifier=identifier, dt=dt)


def verified_package_rates(*, session, head, info, package, entries=False):
    """Verify supplied metadata against owned historical outputs on extraction.

    :param session: The session that accepted the state.
    :param head: Supplied accepted groundwater head.
    :param info: Accepted public solve metadata.
    :param package: Drain or river package name in lowercase.
    :param entries: Return per-entry rates instead of accumulated per-cell rates.
    :return: Owned copy of the accepted per-cell discharge array.
    :raises ValueError: If any supplied head, discharge, entry rate or duration differs.
    :raises RuntimeError: If the state is rejected, absent or belongs to another session.
    """
    if not isinstance(info, dict) or not info.get('converged', False):
        raise RuntimeError('Package exchange requires a converged solve')
    token = info.get('accepted_state', {})
    identifier = retained_state_identifier(token=token)
    state = getattr(session, '_accepted_equations', {}).get(identifier)
    if state is None:
        raise RuntimeError('Accepted state does not belong to this model session')
    if token.get('dt') != state.dt or ('dt' in info and info['dt'] != state.dt):
        raise ValueError('Timestep metadata does not match the accepted equation')
    if package not in state.enabled_packages:
        raise RuntimeError(f'The accepted equation has no {package.upper()} package output')
    accepted_head = state.head.numpy().reshape(state.shape)
    if not np.array_equal(np.asarray(head), accepted_head):
        raise ValueError('head does not match the accepted solve state')
    if f'{package}_rate' not in state.fields:
        raise RuntimeError('The accepted equation has no requested package output')
    materialized = {}
    for name in ('drn', 'riv'):
        owned_entries = state.fields.get(f'{name}_entry_rate')
        if owned_entries is None:
            continue
        supplied_entries = info.get(f'{name}_entry_rates')
        if supplied_entries is not None:
            if not np.array_equal(np.asarray(supplied_entries), owned_entries.numpy()):
                raise ValueError('Entry rates do not match the owned accepted state')
        elif owned_entries.shape[0] and not state.resident_outputs:
            raise ValueError('Accepted entry output is missing')
        supplied_cells = info.get(f'{name}_discharge_rate_array')
        if supplied_cells is not None:
            owned_cells = state.fields[f'{name}_rate'].numpy().reshape(state.shape)
            if not np.array_equal(np.asarray(supplied_cells), owned_cells):
                raise ValueError('Cell rates do not match the owned accepted state')
            materialized[name] = owned_cells
        elif not state.resident_outputs:
            raise ValueError('Accepted cell output is missing')
    if entries:
        return state.fields[f'{package}_entry_rate'].numpy().copy()
    rate = materialized.get(package)
    if rate is None:
        rate = state.fields[f'{package}_rate'].numpy().reshape(state.shape)
    return rate.copy()
