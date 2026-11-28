"""Independent NumPy/SciPy reference for contract NPF662 Cartesian v1.

No Warp imports or production conductance arrays. Outward flux has positive
incidence on its first endpoint. The sparse Jacobian is D(div q)=-D(R).
"""
from dataclasses import dataclass
import hashlib
import warnings
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import spsolve
from DARCY_WARP_PACKAGE.validation.head_checks import inspect_heads
from DARCY_WARP_PACKAGE.validation.npf662_metrics import (
    balanced_nullspace_components, exact_zero_equation_rows, final_tolerances,
    stable_rms, stable_sum, local_balance_valid, unbalanced_ungrounded_components,
)
from DARCY_WARP_PACKAGE.physics.ghb_3d import npf_ghb_component_anchor


@dataclass(frozen=True)
class ReferenceLinearControls:
    """Linear accuracy controls; final physical acceptance remains unchanged."""
    relative_residual: float = 1e-8
    backward_error: float = 1e-12
    refinement_steps: int = 3

    def validate(self):
        if any(not np.isfinite(value) or not 0. < value < 1. for value in
               (self.relative_residual, self.backward_error)):
            raise ValueError('Reference linear tolerances must lie in (0,1)')
        if isinstance(self.refinement_steps, (bool, np.bool_)) or not isinstance(self.refinement_steps, (int, np.integer)) or self.refinement_steps < 0:
            raise ValueError('refinement_steps must be a nonnegative integer')


def solve_reference_correction(*, matrix, rhs, controls):
    """Equilibrated SciPy solve, checked in the original physical equations.

    Extended-precision residual evaluation makes iterative refinement useful
    without sharing any production coefficients or Warp kernels. Both forcing
    and componentwise backward error must pass before nonlinear line search.
    """
    controls.validate()
    matrix = matrix.tocsr()
    if not np.all(np.isfinite(matrix.data)) or not np.all(np.isfinite(rhs)):
        raise ValueError('Reference linear system is nonfinite')
    row_scale = abs(matrix).max(axis=1).toarray().ravel()
    if np.any(row_scale <= 0.):
        raise ValueError('Reference linear system contains a zero row')
    coo = matrix.tocoo()
    row_scaled = sparse.coo_matrix((coo.data/row_scale[coo.row], (coo.row, coo.col)), shape=matrix.shape).tocsr()
    column_scale = abs(row_scaled).max(axis=0).toarray().ravel()
    if np.any(column_scale <= 0.):
        raise ValueError('Reference linear system contains a zero column')
    scaled_coo = row_scaled.tocoo()
    equilibrated = sparse.coo_matrix((scaled_coo.data/column_scale[scaled_coo.col],
        (scaled_coo.row, scaled_coo.col)), shape=matrix.shape).tocsc()
    extended = matrix.astype(np.longdouble)
    extended_rhs = np.asarray(rhs, dtype=np.longdouble)
    correction = np.zeros_like(rhs, dtype=float)
    residual = extended_rhs.copy()
    records = []
    for refinement in range(controls.refinement_steps+1):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            increment = spsolve(A=equilibrated, b=np.asarray(residual/row_scale, dtype=float))/column_scale
        correction += increment
        residual = extended_rhs-extended@np.asarray(correction, dtype=np.longdouble)
        denominator = abs(extended)@np.abs(np.asarray(correction, dtype=np.longdouble))+np.abs(extended_rhs)
        ratios = np.divide(np.abs(residual), denominator, out=np.zeros_like(residual), where=denominator > 0.)
        ratios[(denominator == 0.) & (residual != 0.)] = np.inf
        backward = float(np.max(ratios)) if ratios.size else 0.
        residual_rms = stable_rms(values=residual)
        rhs_rms = stable_rms(values=rhs)
        relative = residual_rms/max(rhs_rms, np.finfo(float).tiny)
        accepted = bool(np.all(np.isfinite(correction)) and np.isfinite(relative) and
                        backward <= controls.backward_error and relative <= controls.relative_residual)
        records.append(dict(refinement=refinement, linear_residual_rms=residual_rms,
            linear_residual_max=float(np.max(np.abs(residual))) if residual.size else 0.,
            linear_relative_residual=relative, linear_backward_error=backward, linear_accepted=accepted))
        if accepted or not np.all(np.isfinite(correction)):
            break
    return correction, dict(**records[-1], refinement_history=records,
        scaling='row and column maximum equilibration; original-equation extended residual')


def serialized_case_fields(*,case):
    names=tuple(name for name in case.__dataclass_fields__
                if name not in ('drn_entries', 'riv_entries')
                and (name not in ('ghb_conductance', 'ghb_head', 'drn_conductance',
                                  'drn_elevation', 'drn_mask', 'riv_conductance', 'riv_stage',
                                  'riv_bottom', 'riv_mask', 'source_gross')
                     or getattr(case, name) is not None))
    if getattr(case,'_legacy_implicit_cell_type',False) and np.ndim(case.cell_type)==0 and case.cell_type==0:
        return tuple(name for name in names if name!='cell_type')
    return names


def digest_with_boundary_entries(*, digest, case):
    """Mix sparse package entries into a case digest.

    :param digest: Running SHA-256 over the serialised dense fields.
    :param case: Case owning optional ``drn_entries``/``riv_entries``.
    :return: The same digest with entry cells, conductances and levels mixed in.
    """
    for name in ('drn_entries', 'riv_entries'):
        entries = getattr(case, name, None)
        if entries is None:
            continue
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(entries.cells, dtype=np.int64).tobytes())
        digest.update(np.ascontiguousarray(entries.conductance, dtype=np.float64).tobytes())
        for level in (entries.first_level, entries.second_level, entries.third_level):
            if level is not None:
                digest.update(np.ascontiguousarray(level, dtype=np.float64).tobytes())
    return digest


@dataclass
class NPF662Case:
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    active: np.ndarray
    fixed: np.ndarray
    values: np.ndarray
    source: np.ndarray
    initial: np.ndarray
    dx: float
    dy: float
    dz: float
    top: float = 0.
    cell_type: np.ndarray | int = 0
    ghb_conductance: np.ndarray | None = None
    ghb_head: np.ndarray | None = None
    drn_conductance: np.ndarray | None = None
    drn_elevation: np.ndarray | None = None
    drn_mask: np.ndarray | None = None
    riv_conductance: np.ndarray | None = None
    riv_stage: np.ndarray | None = None
    riv_bottom: np.ndarray | None = None
    riv_mask: np.ndarray | None = None
    drn_entries: object | None = None
    riv_entries: object | None = None
    source_gross: np.ndarray | None = None

    def validate(self):
        shape=self.active.shape
        if len(shape)!=3 or min(shape)<1:raise ValueError('Expected nonempty (nz,ny,nx)')
        for name in ('kx','ky','kz','fixed','values','source','initial'):
            value=np.asarray(getattr(self,name))
            if value.shape!=shape or not np.all(np.isfinite(value)):raise ValueError(f'Invalid {name}')
        if self.source_gross is not None:
            gross = np.asarray(self.source_gross)
            if gross.shape != shape or not np.all(np.isfinite(gross)) or np.any(gross < np.abs(self.source)):
                raise ValueError('Gross source must be finite and include every signed source')
        spacing=(self.dx,self.dy) if self.geometry=='structured_dis_layered' else (self.dx,self.dy,self.dz)
        if any(np.ndim(x)!=0 or not np.isfinite(x) or x<=0 for x in spacing):raise ValueError('Positive finite spacing required')
        if self.geometry!='structured_dis_layered' and (np.ndim(self.top)!=0 or not np.isfinite(self.top)):
            raise ValueError('Finite flat scalar top required; variable elevations are unsupported')
        for mask in (self.active,self.fixed):
            if not np.all((mask==0)|(mask==1)):raise ValueError('Binary masks required')
        active=self.active.astype(bool)
        if not active.any():raise ValueError('At least one active cell required')
        if np.any(self.fixed.astype(bool)&~active):raise ValueError('Fixed cells must be active')
        self.validate_active_heads()
        if np.any(self.source[~active]!=0):raise ValueError('Inactive source would be lost')
        for k in (self.kx,self.ky,self.kz):
            if np.any(k[active]<=0):raise ValueError('Positive physical Cartesian K required')
        types=np.asarray(self.cell_type)
        if types.ndim!=0 and types.shape!=shape:raise ValueError('cell_type must be scalar or full-grid 0/1 array')
        if np.any((types!=0)&(types!=1)):raise ValueError('Only ICELLTYPE 0/1 supported')
        if np.any(active&(self.fixed!=0)&(self.cell_types!=0)&(self.values<self.bottoms)):
            raise ValueError('Convertible CHD head below cell bottom is invalid for MF6 6.6.2; prescribed heads are not repaired')
        from DARCY_WARP_PACKAGE.physics.ghb_3d import validate_ghb_fields
        validate_ghb_fields(conductance=self.ghb_conductance, head=self.ghb_head,
                            active=self.active, fixed=self.fixed)
        if self.ghb_conductance is not None and np.any((np.asarray(self.ghb_conductance)>0.)
                &(self.cell_types!=0)&(np.asarray(self.ghb_head)<self.bottoms)):
            raise ValueError('Convertible GHB head below cell bottom is invalid for MF6 6.6.2')
        from DARCY_WARP_PACKAGE.physics.drn_riv_3d import validate_drn_fields, validate_riv_fields, validate_boundary_entries
        drn_entries = validate_boundary_entries(entries=self.drn_entries, package='DRN', shape=shape,
            active=self.active, fixed=self.fixed)
        riv_entries = validate_boundary_entries(entries=self.riv_entries, package='RIV', shape=shape,
            active=self.active, fixed=self.fixed)
        if drn_entries is not None and (self.drn_conductance is not None or self.drn_elevation is not None):
            raise ValueError('Supply DRN through dense fields or sparse entries, not both')
        if riv_entries is not None and (self.riv_conductance is not None or self.riv_stage is not None):
            raise ValueError('Supply RIV through dense fields or sparse entries, not both')
        self.drn_entries = drn_entries
        self.riv_entries = riv_entries
        validate_drn_fields(conductance=self.drn_conductance, elevation=self.drn_elevation,
                            active=self.active, fixed=self.fixed, mask=self.drn_mask)
        validate_riv_fields(conductance=self.riv_conductance, stage=self.riv_stage,
                            bottom=self.riv_bottom, active=self.active, fixed=self.fixed, mask=self.riv_mask)

    def validate_active_heads(self):
        active = self.active != 0
        from DARCY_WARP_PACKAGE.validation.head_checks import invalid_head_values
        if np.any(invalid_head_values(head=self.values[active & (self.fixed != 0)])):
            raise ValueError("Invalid active prescribed head")
        if np.any(invalid_head_values(head=self.initial[active])):
            raise ValueError("Invalid active initial head")

    @property
    def cell_types(self):
        return np.broadcast_to(self.cell_type,self.active.shape)

    @property
    def bottoms(self):
        return np.broadcast_to((self.top-self.dz*np.arange(1,self.active.shape[0]+1))[:,None,None],self.active.shape)

    @property
    def tops(self):
        return np.broadcast_to((self.top-self.dz*np.arange(self.active.shape[0]))[:,None,None],self.active.shape)

    @property
    def thicknesses(self):
        return np.full(self.active.shape,self.dz,dtype=float)

    @property
    def geometry(self):
        return 'orthogonal_cartesian_uniform_dz'

    @property
    def contract(self):
        return 'NPF662_CARTESIAN_V1'

    def digest(self):
        self.validate()
        digest=hashlib.sha256()
        for name in serialized_case_fields(case=self):
            value=np.ascontiguousarray(getattr(self,name))
            digest.update(name.encode());digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
        return digest_with_boundary_entries(digest=digest, case=self).hexdigest()

    def entry_digest_fields(self):
        return ()


@dataclass
class NPF662LayeredCase:
    """Separate schema2: no meaningless scalar dz/top in the canonical inputs."""
    kx: np.ndarray
    ky: np.ndarray
    kz: np.ndarray
    active: np.ndarray
    fixed: np.ndarray
    values: np.ndarray
    source: np.ndarray
    initial: np.ndarray
    dx: float
    dy: float
    z_interfaces: np.ndarray
    cell_type: np.ndarray | int = 0
    ghb_conductance: np.ndarray | None = None
    ghb_head: np.ndarray | None = None
    drn_conductance: np.ndarray | None = None
    drn_elevation: np.ndarray | None = None
    drn_mask: np.ndarray | None = None
    riv_conductance: np.ndarray | None = None
    riv_stage: np.ndarray | None = None
    riv_bottom: np.ndarray | None = None
    riv_mask: np.ndarray | None = None
    drn_entries: object | None = None
    riv_entries: object | None = None
    source_gross: np.ndarray | None = None

    validate_active_heads = NPF662Case.validate_active_heads

    def validate(self):
        from DARCY_WARP_PACKAGE.physics.npf662_layers import layered_geometry
        layered_geometry(z_interfaces=self.z_interfaces,shape=self.active.shape,dx=self.dx,dy=self.dy)
        NPF662Case.validate(self)

    @property
    def cell_types(self):
        return np.broadcast_to(self.cell_type,self.active.shape)

    @property
    def bottoms(self):
        return np.asarray(self.z_interfaces,dtype=np.float64)[1:]

    @property
    def tops(self):
        return np.asarray(self.z_interfaces,dtype=np.float64)[:-1]

    @property
    def thicknesses(self):
        return self.tops-self.bottoms

    @property
    def geometry(self):
        return 'structured_dis_layered'

    @property
    def contract(self):
        return 'NPF662_LAYERED_DIS_V2'

    def digest(self):
        self.validate()
        digest=hashlib.sha256(b'NPF662_LAYERED_DIS_V2')
        for name in serialized_case_fields(case=self):
            value=np.ascontiguousarray(getattr(self,name))
            digest.update(name.encode());digest.update(value.dtype.str.encode());digest.update(str(value.shape).encode());digest.update(value.tobytes())
        return digest_with_boundary_entries(digest=digest, case=self).hexdigest()


def save_npf662_case(*,case,path):
    case.validate()
    arrays={name:np.asarray(getattr(case,name)) for name in serialized_case_fields(case=case)}
    for name, package in (('drn_entries', 'DRN'), ('riv_entries', 'RIV')):
        entries = getattr(case, name, None)
        if entries is None:
            continue
        prefix = name + '_'
        arrays[prefix + 'cells'] = entries.cells
        arrays[prefix + 'conductance'] = entries.conductance
        if package == 'DRN':
            arrays[prefix + 'elevation'] = entries.first_level
        else:
            arrays[prefix + 'stage'] = entries.second_level
            arrays[prefix + 'bottom'] = entries.third_level
    if isinstance(case,NPF662LayeredCase):arrays['schema_version']=np.asarray(2)
    np.savez_compressed(path,**arrays)


def load_npf662_case(*,path):
    with np.load(path,allow_pickle=False) as saved:
        schema=int(saved['schema_version']) if 'schema_version' in saved else 1
        if schema not in (1,2):raise ValueError('Unsupported NPF input schema')
        cls=NPF662Case if schema==1 else NPF662LayeredCase
        optional_fields = {'ghb_conductance', 'ghb_head', 'drn_conductance', 'drn_elevation',
                           'drn_mask', 'riv_conductance', 'riv_stage', 'riv_bottom', 'riv_mask',
                           'drn_entries', 'riv_entries', 'source_gross'}
        missing=set(cls.__dataclass_fields__)-set(saved.files)-optional_fields
        implicit=bool(schema==1 and missing=={'cell_type'})
        if missing and not implicit:raise ValueError(f'Missing NPF input fields: {sorted(missing)}')
        values = {name:saved[name].copy() for name in cls.__dataclass_fields__ if name in saved}
        for name, level_names in (('drn_entries', ('elevation',)),
                                  ('riv_entries', ('stage', 'bottom'))):
            prefix = name + '_'
            if prefix + 'cells' not in saved:
                continue
            cells = saved[prefix + 'cells']
            conductance = saved[prefix + 'conductance']
            levels = [saved[prefix + level] for level in level_names]
            values[name] = [(tuple(cell), float(conductance[index]),
                *[float(level[index]) for level in levels]) for index, cell in enumerate(cells)]
        case=cls(**values)
        # NPZ scalars are zero-dimensional arrays; FloPy DIS expects actual
        # scalar spacing, not an array that it tries to index as delr/delc.
        for name in ('dx','dy','dz','top'):
            if hasattr(case,name):setattr(case,name,float(getattr(case,name)))
        if implicit:case._legacy_implicit_cell_type=True
    case.validate()
    return case


class NPF662Reference:
    def __init__(self,*,case):
        case.validate()
        self.case=case
        self.shape=case.active.shape
        self.initial_datum=np.array(case.initial,dtype=np.float64,copy=True)
        self.free=case.active.astype(bool)&~case.fixed.astype(bool)
        self.ghb_conductance = (np.zeros(self.shape) if case.ghb_conductance is None
                                else np.broadcast_to(case.ghb_conductance, self.shape).copy())
        self.ghb_head = (np.zeros(self.shape) if case.ghb_head is None
                         else np.broadcast_to(case.ghb_head, self.shape).copy())
        self.refresh_drn_riv_fields(case=case)
        self.component_anchor = npf_ghb_component_anchor(conductance=case.ghb_conductance,
            active=case.active, fixed=case.fixed)
        ids=np.arange(case.active.size).reshape(self.shape)
        left=[];right=[];conductances=[];axes=[]
        for axis,k,d,width in ((2,case.kx,case.dx,case.dy),
            (1,case.ky,case.dy,case.dx),(0,case.kz,1.,case.dx*case.dy)):
            l,u=[slice(None)]*3,[slice(None)]*3
            l[axis],u[axis]=slice(None,-1),slice(1,None)
            l,u=tuple(l),tuple(u)
            mask=case.active[l].astype(bool)&case.active[u].astype(bool)
            a,b=ids[l][mask],ids[u][mask]
            if case.geometry=='structured_dis_layered':
                if axis==0:width=np.longdouble(case.dx)*case.dy
                kl,kr=np.asarray(k[l][mask],dtype=np.longdouble),np.asarray(k[u][mask],dtype=np.longdouble)
                tl,tr=np.asarray(case.thicknesses[l][mask],dtype=np.longdouble),np.asarray(case.thicknesses[u][mask],dtype=np.longdouble)
                resistance=(.5*d/kl/tl+.5*d/kr/tr) if axis else (.5*tl/kl+.5*tr/kr)
                with np.errstate(over='ignore',under='ignore'):C=np.asarray(np.longdouble(width)/resistance,dtype=float)
            else:
                distance=case.dz if axis==0 else d
                area=width if axis==0 else width*case.dz
                C=area/(.5*distance/k[l][mask]+.5*distance/k[u][mask])
            if np.any(~np.isfinite(C)) or np.any(C<=0):raise ValueError('Unrepresentable physical conductance')
            left.append(a);right.append(b);conductances.append(C);axes.append(np.full(C.size,axis))
        self.left=np.concatenate(left);self.right=np.concatenate(right)
        self.c0=np.concatenate(conductances);self.axes=np.concatenate(axes)
        n=ids.size;m=self.left.size
        self.incidence=sparse.coo_matrix((np.concatenate([np.ones(m),-np.ones(m)]),
            (np.concatenate([self.left,self.right]),np.tile(np.arange(m),2))),shape=(n,m)).tocsr()
        self.matrix=(self.incidence@sparse.diags(self.c0)@self.incidence.T
                     + sparse.diags(self.ghb_conductance.ravel())).tocsr()
        indices=np.arange(m)
        self.select_left=sparse.csr_matrix((np.ones(m),(indices,self.left)),shape=(m,n))
        self.select_right=sparse.csr_matrix((np.ones(m),(indices,self.right)),shape=(m,n))

    def refresh_drn_riv_fields(self, *, case):
        """Refresh package values without rebuilding geometry and K operators.

        Sparse entries and dense single-entry fields both reduce to one owned
        CSR entry list per package; multiple entries per cell keep their own
        fluxes and the cell contributions sum afterwards.

        :param case: Validated steady NPF case with the current DRN/RIV fields.
        :return: None.
        :raises ValueError: If package fields are invalid for the case grid.
        """
        from DARCY_WARP_PACKAGE.physics.drn_riv_3d import (csr_from_boundary_field,
            validate_drn_fields, validate_riv_fields)

        drain_conductance, drain_elevation = validate_drn_fields(
            conductance=case.drn_conductance, elevation=case.drn_elevation,
            active=case.active, fixed=case.fixed, mask=case.drn_mask)
        river_conductance, river_stage, river_bottom = validate_riv_fields(
            conductance=case.riv_conductance, stage=case.riv_stage, bottom=case.riv_bottom,
            active=case.active, fixed=case.fixed, mask=case.riv_mask)
        self.drn_conductance = (np.zeros(self.shape) if drain_conductance is None
            else drain_conductance)
        self.drn_elevation = (np.zeros(self.shape) if drain_elevation is None
            else drain_elevation)
        self.riv_conductance = (np.zeros(self.shape) if river_conductance is None
            else river_conductance)
        self.riv_stage = np.zeros(self.shape) if river_stage is None else river_stage
        self.riv_bottom = np.zeros(self.shape) if river_bottom is None else river_bottom
        self.drn_entries = case.drn_entries
        self.riv_entries = case.riv_entries
        self.drn_csr = (self.drn_entries if self.drn_entries is not None else
            (csr_from_boundary_field(conductance=self.drn_conductance, levels=[self.drn_elevation],
                shape=self.shape) if drain_conductance is not None else None))
        self.riv_csr = (self.riv_entries if self.riv_entries is not None else
            (csr_from_boundary_field(conductance=self.riv_conductance,
                levels=[self.riv_stage, self.riv_bottom], shape=self.shape)
                if river_conductance is not None else None))
        self.case = case

    def boundary_terms(self, *, head):
        """Evaluate DRN and RIV terms with per-entry discharges.

        :param head: Current groundwater head grid.
        :return: ``(drn_entry_discharge, drn_terms, riv_entry_discharge, riv_terms)``.
        """
        from DARCY_WARP_PACKAGE.physics.drn_riv_3d import csr_entry_terms, drn_terms, riv_terms
        h_flat = np.where(self.case.active, np.asarray(head, dtype=np.float64), 0.).ravel()
        if self.drn_csr is not None:
            drn_entries, drn = csr_entry_terms(head_flat=h_flat, entries=self.drn_csr,
                package='DRN', shape=self.shape)
        else:
            drn_entries = None
            drn = drn_terms(head=head, conductance=self.drn_conductance,
                            elevation=self.drn_elevation)
        if self.riv_csr is not None:
            riv_entries, riv = csr_entry_terms(head_flat=h_flat, entries=self.riv_csr,
                package='RIV', shape=self.shape)
        else:
            riv_entries = None
            riv = riv_terms(head=head, conductance=self.riv_conductance,
                            stage=self.riv_stage, bottom=self.riv_bottom)
        return drn_entries, drn, riv_entries, riv

    def face_terms(self,*,head):
        c=self.case
        h=np.where(c.active,head,0.).ravel()
        saturation,derivative=quadratic_saturation_derivative(head=head,bottom=c.bottoms,thickness=c.thicknesses,cell_type=c.cell_types)
        choose_left=(h[self.left]>h[self.right])|((h[self.left]==h[self.right])&(self.axes==1))
        donor=np.where(choose_left,self.left,self.right)
        horizontal=self.axes!=0
        variable=horizontal&((c.cell_types.ravel()[self.left]!=0)|(c.cell_types.ravel()[self.right]!=0))
        conductance=self.c0*np.where(variable,saturation.ravel()[donor],1.)
        extra=np.where(variable,self.c0*derivative.ravel()[donor]*(h[self.left]-h[self.right]),0.)
        dl=conductance+np.where(choose_left,extra,0.)
        dr=-conductance+np.where(~choose_left,extra,0.)
        return conductance,dl,dr,saturation

    def _face_flux_subset(self, *, head, face_indices):
        """Evaluate only faces incident to one exact-zero row.

        :param head: Current candidate head field.
        :param face_indices: Flat indices into the prebuilt face arrays.
        :return: Physical face flows for the selected faces.
        """
        c = self.case
        edges = np.asarray(face_indices, dtype=np.int64)
        left = self.left[edges]
        right = self.right[edges]
        h = np.where(c.active, head, 0.).ravel()
        choose_left = (h[left] > h[right]) | (
            (h[left] == h[right]) & (self.axes[edges] == 1))
        donor = np.where(choose_left, left, right)
        variable = ((self.axes[edges] != 0) &
            ((c.cell_types.ravel()[left] != 0) | (c.cell_types.ravel()[right] != 0)))
        saturation, unused_derivative = quadratic_saturation_derivative(
            head=h[donor], bottom=c.bottoms.ravel()[donor],
            thickness=c.thicknesses.ravel()[donor], cell_type=c.cell_types.ravel()[donor])
        conductance = self.c0[edges] * np.where(variable, saturation, 1.)
        return conductance * (h[left] - h[right])

    def jacobian(self,*,head):
        unused,dl,dr,unused_saturation=self.face_terms(head=head)
        unused_drn_ed, drain, unused_riv_ed, river = self.boundary_terms(head=head)
        return (self.incidence@(self.select_left.multiply(dl[:,None])+self.select_right.multiply(dr[:,None]))
                + sparse.diags((self.ghb_conductance + drain.diagonal + river.diagonal).ravel())).tocsr()

    def boundary_occupancy(self):
        """Flat conducting-cell indicator covering dense fields and entries."""
        drn = np.zeros(self.case.active.size, dtype=bool)
        riv = np.zeros(self.case.active.size, dtype=bool)
        if self.drn_csr is not None:
            np.logical_or.at(drn, self.drn_csr.flat_cells, self.drn_csr.conductance > 0.)
        else:
            drn |= (self.drn_conductance.ravel() > 0.)
        if self.riv_csr is not None:
            np.logical_or.at(riv, self.riv_csr.flat_cells, self.riv_csr.conductance > 0.)
        else:
            riv |= (self.riv_conductance.ravel() > 0.)
        return drn, riv

    def _datum_preserves_boundary_activation(self, *, cell, old_head, new_head):
        """Check whether one exact-zero row shift crosses a DRN/RIV branch.

        :param cell: Flat cell index.
        :param old_head: Evaluated head before the datum shift.
        :param new_head: Proposed datum head.
        :return: Whether every package entry stays on its current branch.
        """
        if self.drn_csr is None:
            if (self.drn_conductance.ravel()[cell] > 0.
                    and (old_head > self.drn_elevation.ravel()[cell])
                    != (new_head > self.drn_elevation.ravel()[cell])):
                return False
        else:
            start, stop = self.drn_csr.offsets[cell:cell + 2]
            enabled = self.drn_csr.conductance[start:stop] > 0.
            elevations = self.drn_csr.first_level[start:stop][enabled]
            if np.any((old_head > elevations) != (new_head > elevations)):
                return False
        if self.riv_csr is None:
            if (self.riv_conductance.ravel()[cell] > 0.
                    and (old_head > self.riv_bottom.ravel()[cell])
                    != (new_head > self.riv_bottom.ravel()[cell])):
                return False
        else:
            start, stop = self.riv_csr.offsets[cell:cell + 2]
            enabled = self.riv_csr.conductance[start:stop] > 0.
            bottoms = self.riv_csr.third_level[start:stop][enabled]
            if np.any((old_head > bottoms) != (new_head > bottoms)):
                return False
        return True

    def _datum_state_is_unchanged(self, *, before_head, before_fields, after_head):
        """Compare equation terms before accepting one datum adjustment.

        :param before_head: Current head field.
        :param before_fields: Physical fields evaluated at ``before_head``.
        :param after_head: Proposed head field.
        :return: Whether residuals, physical flows, boundary branches and
            storage are unchanged, plus the evaluated fields at ``after_head``.
        """
        unused_checks, after_fields = self.evaluate(head=after_head)
        unchanged = all(name not in before_fields or name not in after_fields or
            np.array_equal(before_fields[name], after_fields[name])
            for name in ('residual', 'flux', 'ghb_outflow', 'drn_outflow',
                         'riv_outflow', 'boundary_throughput', 'storage_rate',
                         'storage_tangent'))
        before_drn_entries, before_drain, before_riv_entries, before_river = (
            self.boundary_terms(head=before_head))
        after_drn_entries, after_drain, after_riv_entries, after_river = (
            self.boundary_terms(head=after_head))
        for before, after in ((before_drn_entries, after_drn_entries),
                              (before_riv_entries, after_riv_entries)):
            if before is None or after is None:
                unchanged &= before is after
            else:
                unchanged &= np.array_equal(before, after)
        for before, after in ((before_drain, after_drain),
                              (before_river, after_river)):
            unchanged &= np.array_equal(before.discharge, after.discharge)
            unchanged &= np.array_equal(before.diagonal, after.diagonal)
            unchanged &= np.array_equal(before.rhs, after.rhs)
        return bool(unchanged), after_fields

    def check_components(self):
        n=self.case.active.size
        graph=sparse.coo_matrix((np.ones(self.left.size),(self.left,self.right)),shape=(n,n)).tocsr()
        unused,labels=connected_components(csgraph=graph,directed=False)
        drn_occupied, riv_occupied = self.boundary_occupancy()
        grounded=(self.case.fixed.ravel()!=0)|(self.ghb_conductance.ravel()>0.)
        grounded|=drn_occupied|riv_occupied
        unbalanced=unbalanced_ungrounded_components(labels=labels,active=self.case.active,
            grounded=grounded,source=self.case.source)
        if unbalanced:
            label,cells,net_source=unbalanced[0]
            raise ValueError(f'Ungrounded active component {label} with {cells} cells has '
                             f'nonzero net source={net_source:.17g}')

    def certify_roundoff(self, *, head, residual, conductance, storage=None, tangent=None):
        from .npf662_metrics import componentwise_roundoff
        unused_drn_ed, drain, unused_riv_ed, river = self.boundary_terms(head=head)
        boundary_c = self.ghb_conductance + drain.diagonal + river.diagonal
        boundary_head = np.zeros(self.shape)
        head_field = np.asarray(head, dtype=np.float64)
        boundary_row_scale = self.ghb_conductance * np.abs(self.ghb_head)
        drn_ed, riv_ed = self.entry_discharge_scales(head=head)
        boundary_row_scale += drn_ed
        boundary_row_scale += riv_ed
        return componentwise_roundoff(head=head,source=self.case.source,residual=residual,
            free=self.free,left=self.left,right=self.right,conductance=conductance,
            ghb_conductance=boundary_c,ghb_head=boundary_head,
            storage=storage,tangent=tangent,previous=getattr(self,'previous',None),
            previous_tangent=getattr(self,'preconditioner_storage',None),
            storage_coordinate_scale=np.where(self.case.cell_types!=0,
                np.abs(self.case.bottoms)+self.case.thicknesses,0.) if storage is not None else None,
            boundary_row_scale=boundary_row_scale)

    def entry_discharge_scales(self, *, head):
        """Per-cell rounding scales from every entry's own boundary operands.

        :param head: Current groundwater head grid.
        :return: ``(drn_scale, riv_scale)`` grids summing each entry's
            conductance times its own reference-elevation magnitudes.
        """
        h_flat = np.where(self.case.active, np.asarray(head, dtype=np.float64), 0.).ravel()
        drn_scale = np.zeros(self.case.active.size, dtype=np.float64)
        riv_scale = np.zeros(self.case.active.size, dtype=np.float64)
        for entries, target, package in ((self.drn_csr, drn_scale, 'DRN'),
                                         (self.riv_csr, riv_scale, 'RIV')):
            if entries is None:
                continue
            c = entries.conductance
            h = h_flat[entries.flat_cells]
            if package == 'DRN':
                threshold = entries.first_level
                active = (c > 0.) & (h > threshold)
                scale = np.where(active, c * (np.abs(h) + np.abs(threshold)), 0.)
            else:
                stage, bottom = entries.second_level, entries.third_level
                reference = np.maximum(h, bottom)
                scale = np.where(c > 0., c * (np.abs(reference) + np.abs(stage)), 0.)
            np.add.at(target, entries.flat_cells, scale)
        return (drn_scale.reshape(self.shape), riv_scale.reshape(self.shape))

    def seed_unanchored_boundary_components(self, *, head):
        """Move only unanchored initial guesses to a needed switching threshold.

        A positive component imbalance cannot be corrected while every DRN is
        inactive and every RIV is disconnected. A uniform head shift leaves
        confined intercell flows unchanged and opens the nearest available
        monotone boundary branch without pinning any groundwater cell.

        :param head: Initial total-head field for this solve.
        :return: A copied head field and component/threshold seed records.
        """
        candidate = np.asarray(head, dtype=np.float64).copy()
        drn_has = self.drn_csr is not None or np.any(self.drn_conductance > 0.)
        riv_has = self.riv_csr is not None or np.any(self.riv_conductance > 0.)
        if not drn_has and not riv_has and not hasattr(self, 'storage'):
            return candidate, []
        n = self.case.active.size
        current_conductance, unused_dl, unused_dr, unused_sat = self.face_terms(head=candidate)
        connected = current_conductance > 0.
        graph = sparse.coo_matrix((np.ones(int(np.sum(connected))),
            (self.left[connected], self.right[connected])), shape=(n, n)).tocsr()
        unused, labels = connected_components(csgraph=graph, directed=False)
        checks, fields = self.evaluate(head=candidate)
        unused_drn_ed, drain, unused_riv_ed, river = self.boundary_terms(head=candidate)
        package_tangent = drain.diagonal + river.diagonal
        storage_tangent = np.zeros(self.shape)
        if hasattr(self, 'storage'):
            unused_storage, storage_tangent = self.storage(head=candidate)
        seeds = []
        active = self.case.active.ravel() != 0
        active_cells = np.flatnonzero(active)
        active_labels = labels[active_cells]
        label_count = int(labels.max()) + 1
        fixed = self.case.fixed.ravel() != 0
        restrained = np.zeros(label_count, dtype=bool)
        for restraint in (fixed, self.ghb_conductance.ravel() > 0.,
                          package_tangent.ravel() > 0., storage_tangent.ravel() > 0.):
            counts = np.bincount(active_labels, weights=restraint[active_cells],
                                  minlength=label_count)
            restrained |= counts > 0.
        free_cells = active_cells[self.free.ravel()[active_cells]]
        free_labels = labels[free_cells]
        residual = fields['residual'].ravel()
        residual_mass = np.bincount(free_labels, weights=np.abs(residual[free_cells]),
                                    minlength=label_count)
        candidate_labels = np.flatnonzero((~restrained) & (residual_mass > 0.))

        # Group cell and boundary entries by component once. Scanning a full
        # grid and every package entry for every disconnected dry component
        # makes this setup path quadratic on regional models with many dry
        # cells.
        active_order = np.argsort(active_labels, kind='stable')
        sorted_active_labels = active_labels[active_order]
        sorted_active_cells = active_cells[active_order]
        package_groups = []
        for entries, package, level in ((self.drn_csr, 'DRN', 'first_level'),
                                        (self.riv_csr, 'RIV', 'third_level')):
            if entries is None:
                package_groups.append(None)
                continue
            entry_labels = labels[entries.flat_cells]
            entry_order = np.argsort(entry_labels, kind='stable')
            package_groups.append((entries, package, level,
                                   entry_labels[entry_order], entry_order))

        for label in candidate_labels:
            start = int(np.searchsorted(sorted_active_labels, label, side='left'))
            stop = int(np.searchsorted(sorted_active_labels, label, side='right'))
            component_cells = sorted_active_cells[start:stop]
            free_component_cells = component_cells[self.free.ravel()[component_cells]]
            net_need = stable_sum(values=residual[free_component_cells])
            if not np.isfinite(net_need) or net_need == 0.:
                continue
            direction = 1. if net_need > 0. else -1.
            threshold_candidates = []
            if hasattr(self, 'storage'):
                # Enter the existing constitutive transition, rather than
                # pretending a positive parameter implies a positive tangent.
                # 1e-6 is the fixed MF6 saturation smoothing width.
                cells = free_component_cells
                convertible = self.case.cell_types.ravel()[cells] != 0
                current_head = candidate.ravel()[cells]
                bottom = self.case.bottoms.ravel()[cells]
                thickness = self.case.thicknesses.ravel()[cells]
                top = bottom + thickness
                sy = self.sy.ravel()[cells]
                ss = self.ss.ravel()[cells]
                if direction > 0.:
                    selected = convertible & (current_head <= bottom) & ((ss > 0.) | (sy > 0.))
                    for cell, threshold in zip(cells[selected],
                            (bottom + 1.e-6 * thickness)[selected]):
                        threshold_candidates.append((int(cell), float(threshold), 'STO'))
                else:
                    selected = convertible & (current_head >= top) & (sy > 0.)
                    for cell, threshold in zip(cells[selected],
                            (top - 1.e-6 * thickness)[selected]):
                        threshold_candidates.append((int(cell), float(threshold), 'STO'))
            for group in package_groups:
                if group is None or direction < 0.:
                    continue
                entries, package, level, entry_labels, entry_order = group
                start = int(np.searchsorted(entry_labels, label, side='left'))
                stop = int(np.searchsorted(entry_labels, label, side='right'))
                for entry in entry_order[start:stop]:
                    cell = int(entries.flat_cells[entry])
                    if not self.free.ravel()[cell]:
                        continue
                    threshold = float(getattr(entries, level)[entry])
                    if candidate.ravel()[cell] <= threshold:
                        threshold_candidates.append((cell, threshold, package))
            if not threshold_candidates:
                continue
            target = np.inf if direction > 0. else -np.inf
            shifts = [(float(np.nextafter(threshold, target) - candidate.ravel()[index]),
                       int(index), float(threshold), package)
                      for index, threshold, package in threshold_candidates]
            directional_shifts = [(abs(record[0]), *record) for record in shifts
                                  if record[0] * direction > 0. and np.isfinite(record[0])]
            if not directional_shifts:
                continue
            unused_distance, shift, index, threshold, package = min(directional_shifts)
            candidate.ravel()[component_cells] += shift
            # The uniform shift addition can round the threshold cell back onto
            # the boundary value. Enforce the intended active branch explicitly
            # instead of trusting the representability of the shift.
            guard = 0
            while (candidate.ravel()[index] - threshold) * direction <= 0. and guard < 64:
                candidate.ravel()[index] = np.nextafter(candidate.ravel()[index], target)
                guard += 1
            if (candidate.ravel()[index] - threshold) * direction <= 0.:
                raise ValueError('Unanchored boundary seeding failed to cross the threshold')
            # The seeded entry itself defines the branch; strict crossing above
            # its own threshold is the activation condition.
            entered = bool((candidate.ravel()[index] - threshold) * direction > 0.)
            if not entered:
                raise ValueError('Unanchored boundary seeding did not enter the intended active branch')
            seeds.append(dict(component=int(label), package=package, cell=int(index),
                threshold=float(threshold), uniform_shift_m=float(shift),
                active_branch_verified=entered,
                component_net_inflow=float(net_need)))
        return candidate, seeds

    def evaluate(self,*,head,physical_tolerance=1e-12,relative_tolerance=1e-8,
                 head_tolerance_m=1e-6,boundary_tolerance_m=1e-10,roundoff_acceptance=False):
        c=self.case
        thresholds=final_tolerances(physical_tolerance=physical_tolerance,relative_tolerance=relative_tolerance,
            head_tolerance_m=head_tolerance_m,boundary_tolerance_m=boundary_tolerance_m)
        anchored_head_tolerance = getattr(c, '_anchored_cell_head_tolerance_m', None)
        if anchored_head_tolerance is not None:
            thresholds['anchored_cell_head_error_m'] = anchored_head_tolerance
        checks=inspect_heads(head=head,active=c.active,fixed=c.fixed,values=c.values,
            boundary_tolerance_m=boundary_tolerance_m)
        h=np.where(c.active,head,0.).ravel()
        conductance,dl,dr,saturation=self.face_terms(head=head)
        flux=conductance*(h[self.left]-h[self.right])
        divergence=(self.incidence@flux).reshape(self.shape)
        # Recover a representable signed sum if sparse accumulation overflowed
        # before cancellation. Do not repair unrepresentable positive diagonals.
        for index in np.flatnonzero(~np.isfinite(divergence.ravel())):
            row=self.incidence.getrow(int(index))
            divergence.ravel()[index]=stable_sum(values=row.data*flux[row.indices])
        ghb_exchange=np.zeros(self.shape)
        selected=self.ghb_conductance>0.
        ghb_exchange[selected]=self.ghb_conductance[selected]*(head[selected]-self.ghb_head[selected])
        drn_entries_discharge, drain, riv_entries_discharge, river = self.boundary_terms(head=head)
        boundary_exchange = ghb_exchange + drain.discharge + river.discharge
        boundary_tangent = self.ghb_conductance + drain.diagonal + river.diagonal
        drain_gross = (np.abs(drain.discharge) if drn_entries_discharge is None else
            np.bincount(self.drn_csr.flat_cells, weights=np.abs(drn_entries_discharge), minlength=c.active.size).reshape(self.shape))
        river_gross = (np.abs(river.discharge) if riv_entries_discharge is None else
            np.bincount(self.riv_csr.flat_cells, weights=np.abs(riv_entries_discharge), minlength=c.active.size).reshape(self.shape))
        boundary_gross = np.abs(ghb_exchange) + drain_gross + river_gross
        residual=np.where(self.free,c.source-divergence-boundary_exchange,0.)
        diagonal=np.asarray(abs(self.incidence)@conductance).reshape(self.shape)+boundary_tangent
        scaled=np.divide(residual,diagonal,out=np.zeros_like(residual),where=diagonal>0)
        scaled[(diagonal<=0)&(residual!=0)]=np.inf
        boundary=self.free.ravel()[self.left]&c.fixed.ravel()[self.right].astype(bool)
        reverse=self.free.ravel()[self.right]&c.fixed.ravel()[self.left].astype(bool)
        drain_rates = (drain.discharge[self.free] if drn_entries_discharge is None else drn_entries_discharge)
        river_rates = (river.discharge[self.free] if riv_entries_discharge is None else riv_entries_discharge)
        free_exchange=np.concatenate([flux[boundary], -flux[reverse], ghb_exchange[self.free],
            drain_rates, river_rates])
        free_in=stable_sum(values=np.maximum(-free_exchange,0.))
        free_out=stable_sum(values=np.maximum(free_exchange,0.))
        fixed=(c.active!=0)&(c.fixed!=0)
        chd_rate=divergence[fixed]-c.source[fixed]
        chd_in=stable_sum(values=np.maximum(chd_rate,0.))
        chd_out=stable_sum(values=np.maximum(-chd_rate,0.))
        ghb_in=stable_sum(values=np.maximum(-ghb_exchange[self.free],0.))
        ghb_out=stable_sum(values=np.maximum(ghb_exchange[self.free],0.))
        drn_in=stable_sum(values=np.maximum(-drain_rates,0.))
        drn_out=stable_sum(values=np.maximum(drain_rates,0.))
        riv_in=stable_sum(values=np.maximum(-river_rates,0.))
        riv_out=stable_sum(values=np.maximum(river_rates,0.))
        inflow=stable_sum(values=[chd_in,ghb_in,drn_in,riv_in])
        outflow=stable_sum(values=[chd_out,ghb_out,drn_out,riv_out])
        gross_source = np.abs(c.source) if c.source_gross is None else np.asarray(c.source_gross)
        throughput=stable_sum(values=np.concatenate([gross_source[self.free],np.abs(free_exchange)]))
        equation_throughput=stable_sum(values=np.concatenate([np.abs(c.source[self.free]),np.abs(free_exchange)]))
        rms=stable_rms(values=residual[self.free]);hrms=stable_rms(values=scaled[self.free])
        l1=stable_sum(values=np.abs(residual[self.free]));relative=l1/max(equation_throughput,np.finfo(float).tiny)
        active=c.active!=0
        metrics_valid=all(np.all(np.isfinite(value)) for value in
            (flux,conductance,dl,dr,saturation[active],diagonal[active],divergence[active],residual[self.free],scaled[self.free]))
        metrics_valid=bool(metrics_valid and np.all(np.isfinite([inflow,outflow,throughput,rms,hrms,l1,relative])))
        edges=conductance>0.
        graph=sparse.coo_matrix((np.ones(int(edges.sum())),(self.left[edges],self.right[edges])),shape=self.matrix.shape).tocsr()
        unused,labels=connected_components(csgraph=graph,directed=False)
        # Exact zero-equation rows are independent of saturation. Larger
        # floating components are excluded only after the balanced nullspace
        # check below proves their physical Jacobian has a constant-head mode.
        dry_zero = exact_zero_equation_rows(case=c, head=head, free=self.free,
            diagonal=diagonal, residual=residual, boundary_throughput=boundary_gross)
        active_labels=set(labels[active.ravel() & ~dry_zero.ravel()])
        grounded_mask=(c.active.ravel()!=0)&((c.fixed.ravel()!=0)|(boundary_tangent.ravel()>0.))
        grounded_labels=set(labels[grounded_mask])
        ungrounded_labels=active_labels-grounded_labels
        nullspaces=balanced_nullspace_components(labels=labels,active=c.active,
            grounded=grounded_mask,source=c.source,
            left=self.left,right=self.right,
            face_left_derivative=dl,face_right_derivative=dr,
            boundary_tangent=boundary_tangent,
            boundary_throughput=boundary_gross)
        balanced_labels=ungrounded_labels & set(nullspaces)
        ungrounded=len(ungrounded_labels-balanced_labels)
        result=dict(**checks,head_residual_rms=hrms,physical_rms=rms,
            physical_max=float(np.abs(residual).max()),residual_l1=l1,relative_cell_imbalance=relative,
            gross_throughput=throughput,equation_throughput=equation_throughput,signed_imbalance=stable_sum(values=residual[self.free]),boundary_inflow=inflow,boundary_outflow=outflow,
            free_boundary_inflow=free_in,free_boundary_outflow=free_out,
            ghb_inflow=ghb_in,ghb_outflow=ghb_out,drn_inflow=drn_in,drn_outflow=drn_out,
            drn_net=drn_in-drn_out,riv_inflow=riv_in,riv_outflow=riv_out,riv_net=riv_in-riv_out,
            chd_inflow=chd_in,chd_outflow=chd_out,
            fixed_cell_source_total=stable_sum(values=c.source[c.fixed!=0]),free_cells=int(self.free.sum()),
            fixed_reservoir_required_inflow=stable_sum(values=(divergence-c.source)[c.fixed!=0]),
            ungrounded_current_components=ungrounded,
            balanced_nullspace_components=len(balanced_labels),
            balanced_nullspace_cells=int(sum(nullspaces[label].size for label in balanced_labels)),
            exact_zero_equation_rows=int(dry_zero.sum()),
            exact_dry_zero_rows=int(dry_zero.sum()),
            physical_metrics_valid=metrics_valid,
            final_acceptance_thresholds=thresholds)
        result['physical_metrics_valid']=bool(metrics_valid and np.all(np.isfinite([
            result['signed_imbalance'],result['fixed_cell_source_total'],result['fixed_reservoir_required_inflow']])))
        result['independent_accepted']=bool(checks['input_valid'] and checks['boundary_condition_valid'] and
            result['physical_metrics_valid'] and ungrounded==0 and hrms<=head_tolerance_m and
            (rms<=physical_tolerance or relative<=relative_tolerance))
        result['local_balance_valid'] = None
        if result['independent_accepted'] or roundoff_acceptance:
            component_anchor = npf_ghb_component_anchor(conductance=boundary_tangent,
                active=c.active, fixed=c.fixed, left=self.left, right=self.right,
                face_conductance=conductance)
            result['local_balance_valid'] = local_balance_valid(case=c, free=self.free, left=self.left, right=self.right,
                head=head, residual=residual, flux=flux, conductance=conductance, ghb_conductance=boundary_tangent,
                ghb_exchange=boundary_exchange, thresholds=thresholds, component_anchor=component_anchor,
                boundary_throughput=boundary_gross)
            result['independent_accepted'] &= result['local_balance_valid']
        if roundoff_acceptance:
            from .npf662_metrics import apply_roundoff_acceptance
            apply_roundoff_acceptance(checks=result,certificate=self.certify_roundoff(
                head=head,residual=residual,conductance=conductance))
        return result,dict(flux=flux,residual=residual,fixed_reservoir_required_inflow=np.where(c.fixed,divergence-c.source,0.),
            chd_cell_rate=np.where(fixed,divergence-c.source,0.),
            principal_diagonal=diagonal,saturation=saturation,ghb_outflow=ghb_exchange,
            drn_outflow=drain.discharge,riv_outflow=river.discharge,boundary_throughput=boundary_gross,
            conductance=conductance,dflux_left=dl,dflux_right=dr)

    def apply_nullspace_datum(self, *, head):
        """Retain the original mean head on eligible balanced components.

        Only balanced components with an exact constant-head Jacobian mode
        are shifted to their original mean head. Each shift is accepted only
        when residuals, physical flows, boundary branches and storage remain
        unchanged.

        :param head: Converged total-head grid.
        :return: Candidate heads and a record of applied datum shifts.
        """
        original = np.array(head, dtype=np.float64, copy=True, order='C')
        candidate = original.copy()
        checks, fields = self.evaluate(head=original)
        unused_drn_entries, before_drain, unused_riv_entries, before_river = self.boundary_terms(
            head=original)
        drain, river = before_drain, before_river
        boundary_tangent = self.ghb_conductance + drain.diagonal + river.diagonal
        # TransientReference.evaluate() already returns the physical residual
        # and principal diagonal with STO added. Reuse those evaluated values;
        # subtracting storage here would count accumulation twice.
        physical_residual = fields['residual']
        diagonal = fields['principal_diagonal']
        storage = fields.get('storage_rate')
        storage_tangent = fields.get('storage_tangent')
        zero_rows = exact_zero_equation_rows(case=self.case, head=candidate,
            free=self.free, diagonal=diagonal, residual=physical_residual,
            boundary_throughput=fields['boundary_throughput'], storage=storage)
        datum_heads = (np.asarray(self.previous, dtype=np.float64) if storage is not None
                       else self.initial_datum)
        datum_heads = datum_heads.ravel()
        flat = candidate.ravel()
        records = []
        zero_cells = np.flatnonzero(zero_rows.ravel())
        for cell in zero_cells:
            previous = float(flat[cell])
            datum = float(datum_heads[cell])
            shift = datum - previous
            if shift == 0.:
                continue
            if not self._datum_preserves_boundary_activation(cell=int(cell),
                    old_head=previous, new_head=datum):
                continue
            if storage is not None:
                has_storage_capacity = (self.ss.ravel()[cell] > 0. or
                    (self.case.cell_types.ravel()[cell] != 0 and self.sy.ravel()[cell] > 0.))
                if has_storage_capacity and self.case.thicknesses.ravel()[cell] > 0.:
                    continue
            first_face = self.incidence.indptr[cell]
            last_face = self.incidence.indptr[cell + 1]
            face_indices = self.incidence.indices[first_face:last_face]
            before_flux = self._face_flux_subset(head=candidate,
                face_indices=face_indices)
            flat[cell] = datum
            after_flux = self._face_flux_subset(head=candidate,
                face_indices=face_indices)
            if not np.array_equal(before_flux, after_flux):
                flat[cell] = previous
                continue
            records.append(dict(kind='exact_zero_equation_row', cell=int(cell),
                cells=1, datum_head_m=float(datum),
                previous_mean_head_m=float(previous), applied_shift_m=float(shift)))
        current_fields = fields
        if records:
            unchanged, current_fields = self._datum_state_is_unchanged(
                before_head=original, before_fields=fields, after_head=candidate)
            if not unchanged:
                candidate = original.copy()
                records = []
                current_fields = fields
        if checks['balanced_nullspace_components'] > 0:
            conducting = np.asarray(fields['conductance']) > 0.
            graph = sparse.coo_matrix((np.ones(int(conducting.sum())),
                (self.left[conducting], self.right[conducting])),
                shape=(self.case.active.size, self.case.active.size)).tocsr()
            unused, labels = connected_components(csgraph=graph, directed=False)
            grounded = ((self.case.fixed != 0) | (boundary_tangent > 0.)).ravel()
            if storage_tangent is not None:
                grounded |= (storage_tangent > 0.).ravel()
            nullspaces = balanced_nullspace_components(labels=labels, active=self.case.active,
                grounded=grounded, source=self.case.source,
                left=self.left, right=self.right,
                face_left_derivative=fields['dflux_left'],
                face_right_derivative=fields['dflux_right'],
                boundary_tangent=boundary_tangent,
                boundary_throughput=fields['boundary_throughput'],
                storage=storage,
                storage_tangent=storage_tangent)
            for label, cells in nullspaces.items():
                if np.all(zero_rows.ravel()[cells]):
                    continue
                current_flat = candidate.ravel()
                current_mean = stable_sum(values=current_flat[cells]) / cells.size
                datum_mean = stable_sum(values=datum_heads[cells]) / cells.size
                shift = datum_mean - current_mean
                if shift == 0.:
                    continue
                trial = candidate.copy()
                trial.ravel()[cells] += shift
                unchanged, trial_fields = self._datum_state_is_unchanged(
                    before_head=candidate, before_fields=current_fields, after_head=trial)
                if not unchanged:
                    continue
                candidate = trial
                current_fields = trial_fields
                records.append(dict(kind='balanced_nullspace_component',
                    component=int(label), cells=int(cells.size),
                    datum_mean_head_m=float(datum_mean),
                    previous_mean_head_m=float(current_mean),
                    applied_shift_m=float(shift)))
        if not records:
            return candidate, []
        for record in records:
            record['flux_boundary_storage_unchanged'] = True
        return candidate, records

    def solve(self,*,max_outer=200):
        self.check_components()
        has_switching_boundary = bool(self.drn_csr is not None or self.riv_csr is not None
            or np.any(self.drn_conductance > 0.) or np.any(self.riv_conductance > 0.))
        if np.any(self.case.cell_types[self.case.active!=0]) or has_switching_boundary:
            return self.solve_newton(max_outer=max_outer)
        c=self.case;f=self.free.ravel();fixed=c.fixed.ravel()!=0
        head=np.where(c.fixed,c.values,c.initial).astype(float).ravel().copy()
        linear_max=backward=None
        if f.any():
            rhs=(c.source+self.ghb_conductance*self.ghb_head).ravel()[f]-self.matrix[f][:,fixed]@head[fixed]
            matrix=self.matrix[f][:,f].tocsc()
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                head[f]=spsolve(A=matrix,b=rhs)
            error=rhs-matrix@head[f]
            linear_max=float(np.max(np.abs(error)))
            norm=float(np.max(np.asarray(abs(matrix).sum(axis=1))))
            backward=linear_max/max(norm*np.max(np.abs(head[f]))+np.max(np.abs(rhs)),np.finfo(float).tiny)
        head=head.reshape(self.shape)
        result,fields=self.evaluate(head=head)
        result.update(linear_residual_max=linear_max,linear_backward_error=backward)
        return head,result,fields

    def solve_newton(self,*,max_outer,linear_controls=None):
        linear_controls = ReferenceLinearControls() if linear_controls is None else linear_controls
        linear_controls.validate()
        c=self.case;f=self.free.ravel()
        head=np.where(c.fixed,c.values,c.initial).astype(float).copy()
        head, active_set_seeds = self.seed_unanchored_boundary_components(head=head)
        history=[];reason='outer_budget';reseeds=0
        for outer in range(max_outer):
            checks,fields=self.evaluate(head=head)
            if checks['independent_accepted']:
                reason='converged';break
            rhs=fields['residual'].ravel()[f]
            old=checks['physical_rms']
            found=False;trials=[]
            for mode in ('newton','full_saturated_safeguard'):
                if mode == 'newton':
                    operator = self.jacobian(head=head)
                else:
                    unused_drn_ed, drain, unused_riv_ed, river = self.boundary_terms(head=head)
                    operator = self.matrix + sparse.diags((drain.diagonal + river.diagonal).ravel())
                A=operator[f][:,f].tocsc()
                try:
                    correction, linear_info = solve_reference_correction(matrix=A, rhs=rhs, controls=linear_controls)
                except (RuntimeError, Warning, ValueError) as error:
                    trials.append(dict(direction=mode,factorization_failure=str(error)));continue
                if not linear_info['linear_accepted']:
                    trials.append(dict(direction=mode, accepted=False, stopping='inadequate_linear_solve', **linear_info));continue
                delta=np.zeros_like(head);delta.ravel()[f]=correction
                alpha=min(1.,50./max(float(np.max(np.abs(correction))),np.finfo(float).tiny))
                for attempt in range(30):
                    trial=head+alpha*delta
                    check,trial_fields=self.evaluate(head=trial)
                    if check['input_valid'] and (check['physical_rms']<old or check['independent_accepted']):
                        head=trial;found=True;break
                    alpha*=.5
                trials.append(dict(direction=mode,alpha=alpha,accepted=found,**linear_info,
                    physical_rms=check['physical_rms']))
                if found:break
            history.append(dict(outer=outer+1,accepted=found,attempts=trials))
            if not found:
                # General recovery when head-dependent restraint is lost during
                # the iteration: one fresh active-set seed, then keep iterating.
                if reseeds < 2:
                    seeded, recovery_seeds = self.seed_unanchored_boundary_components(head=head)
                    if recovery_seeds:
                        head = seeded
                        active_set_seeds = active_set_seeds + recovery_seeds
                        reseeds += 1
                        history[-1]['reseeded_after_failure'] = True
                        continue
                reason='line_search_failed';break
        checks,fields=self.evaluate(head=head)
        checks.update(stop_reason='converged' if checks['independent_accepted'] else reason,
            history=history,outers=len(history),active_set_seeds=active_set_seeds,
            active_set_reseeds=reseeds)
        return head,checks,fields


def quadratic_saturation_derivative(*,head,bottom,thickness,cell_type):
    t=np.clip((np.asarray(head)-bottom)/thickness,0.,1.)
    eps=1e-6
    a=1./(1.-eps)
    saturation=np.where(t<eps,a*.5*t*t/eps,np.where(t<1.-eps,a*t+.5*(1.-a),
        np.where(t<1.,1.-a*.5*(1.-t)**2/eps,1.)))
    derivative=np.where(t<eps,a*t/eps,np.where(t<1.-eps,a,np.where(t<1.,a*(1.-t)/eps,0.)))/thickness
    return np.where(cell_type!=0,saturation,1.),np.where(cell_type!=0,derivative,0.)
