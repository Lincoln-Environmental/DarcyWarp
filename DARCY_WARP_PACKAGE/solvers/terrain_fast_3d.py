"""Persistent FP64 terrain flux operator and principal-part multigrid GMRES.

Cartesian fast_3d is untouched. The FULL transformed face operator drives
Krylov and final conservation; the conservative seven-point approximation
is solely a fixed linear V-cycle right preconditioner, never labelled RAP.
"""
from time import perf_counter

import numpy as np
import warp as wp
from warp.optim.linear import LinearOperator

from DARCY_WARP_PACKAGE.config import WP_FLOAT
from DARCY_WARP_PACKAGE.solvers.fast_3d import _build_levels, _upload_levels, _refresh_levels_in_place
from DARCY_WARP_PACKAGE.solvers.face_kernels_3d_f64 import face_jacobi_3d_f64_kernel, residual_partials_3d_f64_kernel, BLOCK_SIZE_3D
from DARCY_WARP_PACKAGE.kernels_3d import copy_field_3d_kernel, restrict_blockavg_axes_3d_kernel, prolong_bilinear_axes_3d_kernel, add_correction_3d_kernel
from DARCY_WARP_PACKAGE.solvers.terrain_kernels_3d import terrain_gradients_kernel, terrain_faces_kernel, terrain_divergence_kernel, terrain_combine_kernel, terrain_preconditioner_rhs_kernel, terrain_preconditioner_output_kernel
from DARCY_WARP_PACKAGE.solvers.terrain_gmres_3d import TerrainGMRES3D
from DARCY_WARP_PACKAGE.solvers_3d import _resolve_isolated_free_cells
from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainMetrics3D
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D, field_3d
from DARCY_WARP_PACKAGE.physics.terrain_budget_3d import compute_terrain_budget_3d, terrain_dry_diagnostics_3d
from DARCY_WARP_PACKAGE.physics.storage_contract_3d import validate_unconfined_storage_inputs
from DARCY_WARP_PACKAGE.physics.storage_3d import storage_increment_3d, seed_zero_capacity_state_3d
from DARCY_WARP_PACKAGE.physics.components_3d import hydraulic_component_labels_3d


def pack_terrain_coefficients(*, operator):
    packed = np.zeros((*operator.shape,13))
    start = 0
    for axis,values in zip((2,1,0),operator.coefficients):
        selection = [slice(None)]*3
        selection[axis] = slice(None,-1)
        for index,value in enumerate(values):
            packed[tuple(selection)+(start+index,)] = value
        start += len(values)
    return packed.reshape(-1,13)


def pack_terrain_boundaries(*, operator, head, datum):
    """Six-column per-cell boundary rows: (conductance, hz/hx/hy metric
    couplings, hb-datum, discharge-only flag). Closed discharge contacts
    contribute zero columns; the device divergence kernel additionally
    evaluates the gate itself, so an ungated pack is also valid input."""
    packed = np.zeros((np.prod(operator.shape),6))
    rhs = np.zeros(operator.shape)
    flux = operator.fluxes(head=head)[3]
    for index,record in enumerate(operator.contact_records):
        cell,c,bz,bx,by,hb,discharge = record
        if discharge and flux[index] <= 0:
            continue
        packed[cell] += (c,bz,bx,by,hb-datum,1.0)
        rhs.ravel()[cell] += c*(hb-datum)
    return packed,rhs


def pack_device_subfaces(*, operator, datum):
    """Per-SUBFACE device contact rows: (cell, c, bz, bx, by, hb-datum,
    discharge). Gating is evaluated independently per subface on device
    and only gate-open subfaces are aggregated into the per-cell boundary
    rows, matching the CPU reference for multiple contacts per cell."""
    records = operator.contact_records
    count = max(len(records), 1)
    subfaces = np.zeros((count, 7))
    for index, record in enumerate(records):
        cell,c,bz,bx,by,hb,discharge = record
        subfaces[index] = (cell, c, bz, bx, by, hb-datum, int(discharge))
    return subfaces, len(records)


def pack_device_contact_refresh(*, operator, datum):
    """Endpoint constitutive rows for exact device contact refresh.

    All rows are affine in saturated fraction, except overridden top/bottom
    metric terms (bz, bx and by): their affine numerators divide by normal K.
    """
    endpoints = []
    for head in (operator.grid.bottoms, operator.grid.tops):
        reference = TerrainReferenceOperator3D(metrics=operator.metrics,
            kx=operator.kx, ky=operator.ky, kz=operator.kz, head=head,
            min_sat=0.0, contacts=operator.contacts, trusted_k=True)
        rows, unused_count = pack_device_subfaces(operator=reference, datum=datum)
        endpoints.append(rows)
    normals = np.zeros((endpoints[0].shape[0], 3))
    for index, contact in enumerate(operator.contacts):
        if contact.direction in ("top", "bottom") and contact.conductance is not None:
            layer, j, i = contact.cell
            interface = layer + int(contact.direction == "bottom")
            zx = operator.metrics.zx_interfaces[interface, j, i]
            zy = operator.metrics.zy_interfaces[interface, j, i]
            normals[index] = (operator.kz[contact.cell],
                operator.kz[contact.cell]+operator.kx[contact.cell]*zx*zx
                + operator.ky[contact.cell]*zy*zy, 1.0)
    return (*endpoints, normals)


def seed_discharge_only_components_3d(*, operator, head, source, fixed, storage, potential_storage, topology_cache):
    """Open a physically available discharge branch with a numerical guess.

    A positive net source fixes the datum through outflow. A balanced component
    with only discharge contacts still has an undetermined datum; negative
    net source cannot be supplied by discharge-only contacts. No boundary or
    storage term is manufactured and all final flux gates remain authoritative.
    """
    records = operator.contact_records
    if not any(record[6] and record[1] > 0 for record in records):
        return 0
    labels,count,sizes,reused = hydraulic_component_labels_3d(faces=operator.principal_faces(),
        active=operator.grid.active,topology_cache=topology_cache)
    active = operator.grid.active != 0
    constrained = np.bincount(labels[active&((np.asarray(fixed) != 0)|(storage > 0))],minlength=count) > 0
    potential = np.bincount(labels[active&potential_storage],minlength=count) > 0
    for record in records:
        if not record[6] and record[1] > 0:
            constrained[labels.ravel()[record[0]]] = True
    net = np.bincount(labels[active],weights=source[active],minlength=count)
    magnitude = np.bincount(labels[active],weights=np.abs(source[active]),minlength=count)
    gradients = tuple(gradient @ head.ravel() for gradient in (operator.metrics.gz,operator.metrics.gx,operator.metrics.gy))
    candidates = {}
    for record in records:
        cell,c,bz,bx,by,hb,discharge = record
        if discharge and c > 0:
            component = int(labels.ravel()[cell])
            threshold = hb-head.ravel()[cell]+(bz*gradients[0][cell]+bx*gradients[1][cell]+by*gradients[2][cell])/c
            candidates.setdefault(component,[]).append((threshold,c))
    seeded = 0
    for component,contacts in candidates.items():
        if constrained[component]:
            continue
        opening = min(threshold for threshold,c in contacts)
        if potential[component] and opening < 0:
            continue
        tolerance = 64*np.finfo(np.float64).eps*magnitude[component]
        if abs(net[component]) <= tolerance:
            raise ValueError("source-balanced discharge-only component has an undetermined head datum; provide an explicit physical head constraint")
        if net[component] < 0:
            raise ValueError("negative net source in a discharge-only component: no boundary can provide inflow; no valid steady solution")
        if opening >= 0:
            selected = active&(labels == component)
            # A fully saturated GUESS gives fixed metric coefficients and a
            # guaranteed open branch, including bottom contacts whose ratio
            # changes during wetting. The physical old storage is untouched.
            full = TerrainReferenceOperator3D(metrics=operator.metrics,kx=operator.kx,ky=operator.ky,kz=operator.kz,contacts=operator.contacts)
            full_thresholds = []
            for record in full.contact_records:
                cell,c,bz,bx,by,hb,discharge = record
                if discharge and c > 0 and labels.ravel()[cell] == component:
                    threshold = hb-head.ravel()[cell]+(bz*gradients[0][cell]+bx*gradients[1][cell]+by*gradients[2][cell])/c
                    full_thresholds.append((threshold,c))
            lift = max(float(np.max(operator.grid.tops[selected]-head[selected])),
                min(threshold for threshold,c in full_thresholds)+net[component]/sum(c for threshold,c in full_thresholds))
            proposed = head[selected]+lift
            head[selected] = np.maximum(proposed,np.nextafter(head[selected],np.inf))
            seeded += 1
    return seeded


class TerrainGPUWorkspace3D:
    def __init__(self, *, operator, fixed, storage, boundary, device,
                 max_levels, min_coarse_n, restart, maxiter, rel_tol, abs_tol,
                 datum=0.0):
        self.shape = operator.shape
        self.nz,self.ny,self.nx = self.shape
        self.n = int(np.prod(self.shape))
        self.device = device
        self.datum = float(datum)
        self.max_levels,self.min_coarse_n = max_levels,min_coarse_n
        self.x = wp.zeros(self.n,dtype=wp.float64,device=device)
        self.b = wp.zeros_like(self.x)
        self.gx,self.gy,self.gz = (wp.zeros_like(self.x) for unused in range(3))
        self.qx,self.qy,self.qz = (wp.zeros_like(self.x) for unused in range(3))
        self.preconditioned = wp.zeros_like(self.x)
        self.scale = wp.ones_like(self.x)
        self.coefficients = wp.array(pack_terrain_coefficients(operator=operator),dtype=wp.float64,device=device)
        self.boundary = wp.array(boundary,dtype=wp.float64,device=device)
        self.storage = wp.array(storage.ravel(),dtype=wp.float64,device=device)
        self.active = wp.array(operator.grid.active.ravel(),dtype=wp.int32,device=device)
        self.fixed = wp.array(np.asarray(fixed,dtype=np.int32).ravel(),dtype=wp.int32,device=device)
        self.horizontal = wp.zeros((self.ny*self.nx,6),dtype=wp.float64,device=device)
        self.depths = wp.zeros(max(self.nz-1,1)*self.ny*self.nx,dtype=wp.float64,device=device)
        self.refresh_metrics(metrics=operator.metrics)
        self.levels = None
        self.refresh_principal(operator=operator,fixed=fixed,storage=storage,boundary=boundary)
        self.a_operator = LinearOperator(shape=(self.n,self.n),dtype=wp.float64,device=wp.get_device(device),matvec=self.krylov_matvec)
        self.m_operator = LinearOperator(shape=(self.n,self.n),dtype=wp.float64,device=wp.get_device(device),matvec=self.precondition)
        self.krylov = TerrainGMRES3D(A=self.a_operator,b=self.b,x=self.x,M=self.m_operator,
            restart=restart,maxiter=maxiter,tol=rel_tol,atol=abs_tol*np.sqrt(self.n),check_every=restart)
        self.refresh_count = 0

    def refresh_metrics(self, *, metrics):
        horizontal = np.stack((*metrics.x_coefficients,*metrics.y_coefficients),axis=-1).reshape(-1,6)
        self.horizontal.assign(horizontal)
        depths = metrics.depth_distances if self.nz > 1 else np.ones((1,self.ny,self.nx))
        self.depths.assign(depths.ravel())

    def refresh_principal(self, *, operator, fixed, storage, boundary):
        principal_storage = storage + boundary[:,0].reshape(self.shape)
        values = np.zeros(self.shape)
        diagonal = sum(operator.principal_faces())+principal_storage
        diagonal[(np.asarray(fixed) != 0)|(operator.grid.active == 0)] = 1
        if np.any(diagonal <= 0):
            raise ValueError("unconstrained zero terrain diagonal escaped topology validation")
        self.scale.assign(diagonal.ravel())
        self.scale_host = diagonal
        if self.levels is None:
            self.levels = _upload_levels(host_levels=_build_levels(faces=operator.principal_faces(),
                active=operator.grid.active,bc_mask=fixed,bc_values=values,rhs=values,
                storage_diag=principal_storage,max_levels=self.max_levels,min_coarse_n=self.min_coarse_n),device=self.device)
        else:
            _refresh_levels_in_place(levels=self.levels,faces=operator.principal_faces(),
                active=operator.grid.active,bc_mask=fixed,bc_values=values,rhs=values,
                storage_diag=principal_storage,max_levels=self.max_levels,min_coarse_n=self.min_coarse_n)

    def refresh(self, *, operator, fixed, storage, boundary, coefficients=None):
        # The caller's freshly packed coefficients are reused when provided,
        # so the nonlinear loop packs each outer iteration exactly once.
        self.coefficients.assign(pack_terrain_coefficients(operator=operator)
            if coefficients is None else coefficients)
        self.storage.assign(storage.ravel())
        self.boundary.assign(boundary)
        self.refresh_metrics(metrics=operator.metrics)
        self.refresh_principal(operator=operator,fixed=fixed,storage=storage,boundary=boundary)
        self.refresh_count += 1

    def matvec(self, x, y, z, alpha, beta):
        self.apply_operator(x=x,y=y,z=z,alpha=alpha,beta=beta,row_scaled=False)

    def krylov_matvec(self, x, y, z, alpha, beta):
        self.apply_operator(x=x,y=y,z=z,alpha=alpha,beta=beta,row_scaled=True)

    def apply_operator(self, *, x, y, z, alpha, beta, row_scaled):
        wp.launch(terrain_gradients_kernel,dim=self.n,inputs=[x,self.horizontal,self.depths,
            self.gx,self.gy,self.gz,self.nx,self.ny,self.nz],device=self.device)
        wp.launch(terrain_faces_kernel,dim=self.n,inputs=[x,self.gx,self.gy,self.gz,
            self.coefficients,self.qx,self.qy,self.qz,self.nx,self.ny,self.nz],device=self.device)
        wp.launch(terrain_divergence_kernel,dim=self.n,inputs=[x,y,self.gx,self.gy,self.gz,
            self.qx,self.qy,self.qz,self.storage,self.boundary,self.active,self.fixed,self.scale,int(row_scaled),z,
            float(alpha),float(beta),self.nx,self.ny,self.nz],device=self.device)

    def smooth(self, *, level, iterations):
        source,out = level["x"],level["work"]
        for unused in range(iterations):
            wp.launch(face_jacobi_3d_f64_kernel,dim=level["dim"],inputs=[level["rhs"],source,
                *level["faces"],level["diag"],level["active"],level["bc_mask"],level["bc_values"],
                0.65,level["nx"],level["ny"],level["nz"],out],device=self.device)
            source,out = out,source
        if source is not level["x"]:
            wp.copy(dest=level["x"],src=source)

    def vcycle(self, *, index):
        level = self.levels[index]
        if index+1 == len(self.levels):
            self.smooth(level=level,iterations=30)
            return
        self.smooth(level=level,iterations=2)
        level["partials_a"].zero_()
        wp.launch(residual_partials_3d_f64_kernel,dim=level["n_cells"],inputs=[level["x"],level["rhs"],
            *level["faces"],level["storage"],level["active"],level["bc_mask"],level["residual"],
            level["partials_a"],level["nx"],level["ny"],level["nz"],BLOCK_SIZE_3D],device=self.device)
        coarse = self.levels[index+1]
        wp.launch(restrict_blockavg_axes_3d_kernel,dim=coarse["dim"],inputs=[level["residual"],
            level["active"],level["bc_mask"],coarse["rhs"],level["nx"],level["ny"],level["nz"],
            coarse["nx"],coarse["ny"],coarse["nz"],level["coarsen_y"],level["coarsen_x"]],device=self.device)
        coarse["x"].zero_()
        self.vcycle(index=index+1)
        level["correction"].zero_()
        wp.launch(prolong_bilinear_axes_3d_kernel,dim=level["dim"],inputs=[coarse["x"],level["correction"],
            level["nx"],level["ny"],level["nz"],coarse["nx"],coarse["ny"],coarse["nz"],
            level["coarsen_y"],level["coarsen_x"]],device=self.device)
        wp.launch(add_correction_3d_kernel,dim=level["dim"],inputs=[level["x"],level["correction"],
            level["active"],level["bc_mask"],level["bc_values"],level["nx"],level["ny"],level["nz"]],device=self.device)
        self.smooth(level=level,iterations=2)

    def precondition(self, x, y, z, alpha, beta):
        fine = self.levels[0]
        wp.launch(terrain_preconditioner_rhs_kernel,dim=self.shape,inputs=[x,self.scale,fine["rhs"],fine["x"],fine["active"],fine["bc_mask"]],device=self.device)
        self.vcycle(index=0)
        wp.launch(terrain_preconditioner_output_kernel,dim=self.shape,inputs=[x,fine["x"],fine["active"],fine["bc_mask"],self.preconditioned],device=self.device)
        wp.launch(terrain_combine_kernel,dim=self.n,inputs=[self.preconditioned,y,z,float(alpha),float(beta)],device=self.device)

    def close(self):
        self.krylov._cycle_graph = None
        self.krylov._A = None
        self.krylov._M = None
        self.a_operator = None
        self.m_operator = None


class TerrainOperatorSession3D:
    """Single owner of terrain inputs, versioned geometry and GPU workspaces."""
    def __init__(self, *, grid, kx, ky, kz, bc_mask, bc_values, rhs, initial_head, contacts=()):
        self.grid = grid
        self.metrics = TerrainMetrics3D(grid=grid)
        self.shape = grid.shape
        self.kx = field_3d(value=kx,shape=self.shape,name="kx",nonnegative=True)
        self.ky = field_3d(value=ky,shape=self.shape,name="ky",nonnegative=True)
        self.kz = field_3d(value=kz,shape=self.shape,name="kz",nonnegative=True)
        self.bc_mask = np.asarray(bc_mask,dtype=np.int32).copy()
        if self.bc_mask.shape != self.shape or not np.all((self.bc_mask == 0)|(self.bc_mask == 1)) or np.any((self.bc_mask != 0)&(grid.active == 0)):
            raise ValueError("bc_mask must be binary, grid-shaped and active")
        self.bc_values = field_3d(value=bc_values,shape=self.shape,name="bc_values")
        self.rhs = field_3d(value=rhs,shape=self.shape,name="rhs")
        self.initial_head = None if initial_head is None else field_3d(value=initial_head,shape=self.shape,name="initial_head")
        self.contacts = tuple(contacts)
        if any(self.bc_mask[contact.cell] for contact in self.contacts):
            raise ValueError("a fixed-head cell cannot also own an exchange subface")
        self.workspace = None
        self.workspace_key = None
        self.snapshot = None
        self.snapshot_packed_positive = None
        self.topology_cache = {}
        self.geometry_revision = 0
        self.k_version = 0

    def update_rhs(self, rhs):
        self.rhs[...] = field_3d(value=rhs,shape=self.shape,name="rhs")

    def update_initial_head(self, initial_head):
        self.initial_head = None if initial_head is None else field_3d(value=initial_head,shape=self.shape,name="initial_head")

    def update_boundary_values(self, bc_values):
        self.bc_values[...] = field_3d(value=bc_values,shape=self.shape,name="bc_values")

    def update_contacts(self, *, contacts):
        records = tuple(contacts)
        TerrainReferenceOperator3D(metrics=self.metrics,kx=self.kx,ky=self.ky,kz=self.kz,contacts=records)
        if any(self.bc_mask[contact.cell] for contact in records):
            raise ValueError("a fixed-head cell cannot also own an exchange subface")
        self.contacts = records

    def update_geometry(self, *, grid):
        if grid.shape != self.shape:
            raise ValueError("terrain geometry updates must retain grid shape")
        if np.any((self.bc_mask != 0)&(grid.active == 0)):
            raise ValueError("geometry update would deactivate a fixed-head cell")
        self.grid = grid
        self.metrics = TerrainMetrics3D(grid=grid)
        self.geometry_revision += 1

    def close(self):
        if self.workspace is not None:
            self.workspace.close()
        self.workspace = None

    def workspace_topology_matches(self, *, fixed, active, coefficients):
        """Compare against the topology installed with the last snapshot.

        Numeric refresh preserves buffers and captured Krylov graphs only
        when both masks and positive-face connectivity remain unchanged.
        Missing metadata cannot establish that the hierarchy is compatible.
        """
        return (self.workspace is not None and self.snapshot is not None
            and self.snapshot_packed_positive is not None
            and np.array_equal(fixed, self.snapshot[6])
            and np.array_equal(active, self.snapshot[7])
            and np.array_equal(coefficients[:, (0, 3, 6)] > 0,
                self.snapshot_packed_positive))

    def solve(self, *, device, unconfined=False, transient=False, dt=None, head_prev=None,
              initial_head=None, sy=None, ss=None, storage_coeff=None, storage_diag=None,
              unconfined_storage_mode="secant_sy", unconfined_min_sat=0.01,
              unconfined_max_picard_iter=100, unconfined_relax=0.7,
              unconfined_head_tol=1e-6, final_head_residual_tol=None,
              final_budget_relative_tol=1e-8, max_cycles=300, max_levels=6,
              min_coarse_n=2, krylov_restart=12, rel_tol=1e-10, abs_tol_min=1e-12,
              dh_rms_tol=1e-6, return_info=True, device_picard=False, terrain_nonlinear_method="picard", terrain_newton_controls=None, **unsupported):
        if WP_FLOAT is not wp.float64:
            raise ValueError("terrain fast solver requires DARCY_FLOAT=float64")
        if terrain_nonlinear_method not in ("picard", "newton"):
            raise ValueError("terrain_nonlinear_method must be picard or newton")
        if terrain_nonlinear_method == "newton" and (not unconfined or transient):
            raise ValueError("terrain Newton currently supports steady unconfined solves only")
        if terrain_newton_controls is not None:
            if terrain_nonlinear_method != "newton":
                raise ValueError("terrain_newton_controls requires terrain_nonlinear_method=newton")
            from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import TerrainNewtonControls
            if not isinstance(terrain_newton_controls, TerrainNewtonControls):
                raise TypeError("terrain_newton_controls must be TerrainNewtonControls")
            terrain_newton_controls.validate()
        if unsupported:
            raise TypeError(f"unsupported terrain controls: {', '.join(sorted(unsupported))}")
        if max_cycles < 0 or krylov_restart < 1 or max_levels < 1 or unconfined_max_picard_iter < 1 or not 0 < unconfined_relax <= 1:
            raise ValueError("invalid terrain iteration controls")
        head_tolerance = (unconfined_head_tol if unconfined else dh_rms_tol) if final_head_residual_tol is None else final_head_residual_tol
        if any(not np.isfinite(v) or v < 0 for v in (rel_tol,abs_tol_min,final_budget_relative_tol)) or head_tolerance is None or not np.isfinite(head_tolerance) or head_tolerance <= 0:
            raise ValueError("terrain tolerances must be finite and nonnegative, with positive head tolerance")
        initial = self.initial_head if initial_head is None else initial_head
        if unconfined:
            if not np.isfinite(unconfined_min_sat) or unconfined_min_sat <= 0:
                raise ValueError("terrain unconfined solves require a positive numerical saturation floor")
            unconfined_storage_mode = validate_unconfined_storage_inputs(transient=transient,mode=unconfined_storage_mode,sy=sy,ss=ss,
                storage_coeff=storage_coeff,storage_diag=storage_diag,head_prev=head_prev,initial_head=initial,dt=dt)
            if unconfined_storage_mode == "phreatic_sy":
                raise ValueError("terrain supports conservative secant_sy or explicit confined_volume storage")
        elif sy is not None or ss is not None:
            raise ValueError("confined terrain uses storage_coeff, not sy/ss")
        if storage_diag is not None:
            raise ValueError("terrain preassembled storage_diag is unsupported; provide physical storage fields")
        if transient and (dt is None or not np.isfinite(dt) or dt <= 0 or (head_prev is None and initial is None)):
            raise ValueError("terrain transient requires positive dt and explicit head_prev or initial_head")
        if not transient and (storage_coeff is not None or head_prev is not None or dt is not None):
            raise ValueError("terrain storage inputs require transient=True")
        if transient and not unconfined and storage_coeff is None:
            raise ValueError("confined terrain transient requires storage_coeff")
        previous = None if not transient else field_3d(value=head_prev if head_prev is not None else initial,shape=self.shape,name="head_prev")
        head = field_3d(value=self.grid.elevations if initial is None else initial,shape=self.shape,name="initial_head")
        head[self.bc_mask != 0] = self.bc_values[self.bc_mask != 0]
        head[self.grid.active == 0] = 0
        sy_field = field_3d(value=0 if sy is None else sy,shape=self.shape,name="sy",nonnegative=True)
        ss_field = field_3d(value=0 if ss is None else ss,shape=self.shape,name="ss",nonnegative=True)
        coeff = field_3d(value=0 if storage_coeff is None else storage_coeff,shape=self.shape,name="storage_coeff",nonnegative=True)
        storage_seeds = 0
        if unconfined and transient and unconfined_storage_mode == "secant_sy":
            storage_seeds = seed_zero_capacity_state_3d(head=head,previous=previous,
                bottoms=self.grid.bottoms,thickness=self.grid.thickness,sy=sy_field,ss=ss_field,
                source=self.rhs,free=(self.grid.active != 0)&(self.bc_mask == 0),
                area=self.grid.dx*self.grid.dy,dt=dt)
        datum = float(self.grid.surface_elevation.flat[0])
        free = (self.grid.active != 0)&(self.bc_mask == 0)
        key = (device,max_levels,min_coarse_n,krylov_restart,max_cycles,rel_tol,abs_tol_min)
        total_iterations = 0
        history = []
        refresh_seconds = 0.0
        discharge_seeds = 0
        starting_workspace = self.workspace
        workspace_reused = self.workspace is not None and self.workspace_key == key
        graph_captures_before = getattr(self.workspace.krylov,"graph_captures",0) if workspace_reused else 0
        outer_limit = unconfined_max_picard_iter if unconfined or any(c.kind == "discharge_only" for c in self.contacts) else 1
        if device_picard or terrain_nonlinear_method == "newton":
            return self._solve_device(head=head,previous=previous,device=device,
                unconfined=unconfined,transient=transient,dt=dt,datum=datum,free=free,
                outer_limit=outer_limit,unconfined_relax=unconfined_relax,
                unconfined_storage_mode=unconfined_storage_mode,sy_field=sy_field,
                ss_field=ss_field,dt_unused=dt,key=key,max_levels=max_levels,
                min_coarse_n=min_coarse_n,krylov_restart=krylov_restart,
                max_cycles=max_cycles,rel_tol=rel_tol,abs_tol_min=abs_tol_min,
                head_tolerance=head_tolerance,
                final_budget_relative_tol=final_budget_relative_tol,
                unconfined_min_sat=unconfined_min_sat,storage_seeds=storage_seeds,
                return_info=return_info, terrain_nonlinear_method=terrain_nonlinear_method,
                terrain_newton_controls=terrain_newton_controls)
        for outer in range(outer_limit):
            start = perf_counter()
            operator = TerrainReferenceOperator3D(metrics=self.metrics,kx=self.kx,ky=self.ky,kz=self.kz,
                head=head if unconfined else None,min_sat=unconfined_min_sat,contacts=self.contacts,trusted_k=True)
            storage,change = self.storage_terms(head=head,previous=previous,transient=transient,unconfined=unconfined,
                mode=unconfined_storage_mode,sy=sy_field,ss=ss_field,coeff=coeff,dt=dt)
            seeded = seed_discharge_only_components_3d(operator=operator,head=head,source=self.rhs,
                fixed=self.bc_mask,storage=storage,potential_storage=bool(transient)&((sy_field > 0)|(ss_field > 0)),topology_cache=self.topology_cache)
            if seeded:
                discharge_seeds += seeded
                operator = TerrainReferenceOperator3D(metrics=self.metrics,kx=self.kx,ky=self.ky,kz=self.kz,
                    head=head if unconfined else None,min_sat=unconfined_min_sat,contacts=self.contacts,trusted_k=True)
                storage,change = self.storage_terms(head=head,previous=previous,transient=transient,unconfined=unconfined,
                    mode=unconfined_storage_mode,sy=sy_field,ss=ss_field,coeff=coeff,dt=dt)
            initial_budget = compute_terrain_budget_3d(operator=operator,head=head,source=self.rhs,fixed=self.bc_mask,
                storage_change=change,storage_diagonal=storage)
            boundary,boundary_rhs = pack_terrain_boundaries(operator=operator,head=head,datum=datum)
            rhs = self.rhs+boundary_rhs+(storage*(previous-datum) if transient else 0)
            rhs[self.bc_mask != 0] = self.bc_values[self.bc_mask != 0]-datum
            rhs[self.grid.active == 0] = 0
            topology_diag = storage + boundary[:,0].reshape(self.shape)
            isolated = _resolve_isolated_free_cells(faces=operator.principal_faces(),active=self.grid.active,
                bc_mask=self.bc_mask,rhs_eff=rhs,storage_diag=topology_diag,topology_cache=self.topology_cache)
            # Zero rows with no source retain their datum; constraints for the
            # numerical iteration are explicitly reported, carry no budget.
            fixed = self.bc_mask.copy()
            fixed[isolated] = 1
            rhs[isolated] = head[isolated]-datum
            # Coefficients are a deterministic function of the versioned
            # inputs below (K fields, geometry revision, floor, and the head
            # for unconfined constitutive updates), so equal inputs imply an
            # equal operator and neither a pack nor a device refresh is
            # needed. This replaces the per-outer 13-column array compare.
            snapshot = (self.k_version,self.geometry_revision,unconfined_min_sat,
                head.copy() if unconfined else None,storage.copy(),boundary.copy(),
                fixed.copy(),self.grid.active.copy())
            unchanged = self.snapshot is not None and all(
                a is None and b is None or np.array_equal(a,b) for a,b in zip(snapshot,self.snapshot))
            packed_coefficients = None
            if self.workspace is None or self.workspace_key != key:
                if self.workspace is not None:
                    self.workspace.close()
                packed_coefficients = pack_terrain_coefficients(operator=operator)
                self.workspace = TerrainGPUWorkspace3D(operator=operator,fixed=fixed,storage=storage,boundary=boundary,
                    device=device,max_levels=max_levels,min_coarse_n=min_coarse_n,restart=krylov_restart,
                    maxiter=max_cycles,rel_tol=rel_tol,abs_tol=abs_tol_min,datum=datum)
                self.workspace_key = key
            elif not unchanged:
                packed_coefficients = pack_terrain_coefficients(operator=operator)
                same_topology = self.workspace_topology_matches(fixed=fixed,
                    active=self.grid.active, coefficients=packed_coefficients)
                if same_topology:
                    self.workspace.refresh(operator=operator,fixed=fixed,storage=storage,boundary=boundary,
                        coefficients=packed_coefficients)
                else:
                    if self.workspace is not None:
                        self.workspace.close()
                    self.workspace = TerrainGPUWorkspace3D(operator=operator,fixed=fixed,storage=storage,boundary=boundary,
                        device=device,max_levels=max_levels,min_coarse_n=min_coarse_n,restart=krylov_restart,
                        maxiter=max_cycles,rel_tol=rel_tol,abs_tol=abs_tol_min,datum=datum)
                    self.workspace_key = key
            self.snapshot = snapshot
            if packed_coefficients is not None:
                # Persist across unchanged outers: topology comparisons on
                # later changed outers still need the last known pattern.
                self.snapshot_packed_positive = packed_coefficients[:,(0,3,6)] > 0
            self.workspace.b.assign((rhs/self.workspace.scale_host).ravel())
            relative_head = np.where(self.grid.active != 0,head-datum,0)
            self.workspace.x.assign(relative_head.ravel())
            # The inner stopping norm has head units after row scaling. Bound
            # it by both the requested head accuracy and the INDEPENDENT
            # volumetric balance target; otherwise low-K or large datum/flow
            # scales can stall Picard with an already "converged" inner solve.
            n_free = max(int(np.count_nonzero(free)),1)
            flow_target = max(abs_tol_min,final_budget_relative_tol*initial_budget["throughput"]/n_free)
            scale_max = float(np.max(self.workspace.scale_host[free])) if free.any() else 1.0
            reference_head = max(float(np.linalg.norm(rhs/self.workspace.scale_host)),float(np.linalg.norm(relative_head)))/np.sqrt(n_free)
            relative_target = rel_tol*reference_head if rel_tol > 0 else np.inf
            inner_head_target = min(relative_target,0.05*flow_target/scale_max,0.05*head_tolerance)
            self.workspace.krylov._tol = 0.0
            self.workspace.krylov._atol = inner_head_target*np.sqrt(n_free)
            self.workspace.krylov._maxiter = int(max_cycles)
            refresh_seconds += perf_counter()-start
            if max_cycles > 0:
                iterations,error,tolerance = self.workspace.krylov()
            else:
                iterations,error,tolerance = 0, float("nan"),abs_tol_min
            total_iterations += iterations
            candidate = self.workspace.x.numpy().reshape(self.shape)+datum
            candidate[self.grid.active == 0] = 0
            candidate[self.bc_mask != 0] = self.bc_values[self.bc_mask != 0]
            dh = candidate-head
            starting_head = head.copy()
            relaxation = unconfined_relax if outer_limit > 1 else 1.0
            if outer_limit > 1:
                head[free] += relaxation*dh[free]
            else:
                head = candidate
            final_operator = TerrainReferenceOperator3D(metrics=self.metrics,kx=self.kx,ky=self.ky,kz=self.kz,
                head=head if unconfined else None,min_sat=unconfined_min_sat,contacts=self.contacts,trusted_k=True)
            final_storage,final_change = self.storage_terms(head=head,previous=previous,transient=transient,unconfined=unconfined,
                mode=unconfined_storage_mode,sy=sy_field,ss=ss_field,coeff=coeff,dt=dt)
            budget = compute_terrain_budget_3d(operator=final_operator,head=head,source=self.rhs,fixed=self.bc_mask,
                storage_change=final_change,storage_diagonal=final_storage)
            # The exact final-head balance, rather than the frozen inner
            # residual, controls damping at elastic/Sy switches and rewetting.
            # Accept monotonic residual reduction or an already physical state.
            for backtrack in range(12):
                if outer_limit == 1 or budget["cell_residual_rms"] <= initial_budget["cell_residual_rms"] or budget["head_residual_rms"] <= head_tolerance:
                    break
                relaxation *= 0.5
                head[free] = starting_head[free]+relaxation*dh[free]
                final_operator = TerrainReferenceOperator3D(metrics=self.metrics,kx=self.kx,ky=self.ky,kz=self.kz,
                    head=head if unconfined else None,min_sat=unconfined_min_sat,contacts=self.contacts,trusted_k=True)
                final_storage,final_change = self.storage_terms(head=head,previous=previous,transient=transient,unconfined=unconfined,
                    mode=unconfined_storage_mode,sy=sy_field,ss=ss_field,coeff=coeff,dt=dt)
                budget = compute_terrain_budget_3d(operator=final_operator,head=head,source=self.rhs,fixed=self.bc_mask,
                    storage_change=final_change,storage_diagonal=final_storage)
            converged = np.all(np.isfinite(head)) and budget["head_residual_rms"] <= head_tolerance and (budget["cell_residual_rms"] <= abs_tol_min or budget["relative_cell_imbalance"] <= final_budget_relative_tol)
            proposed_change_max = float(np.max(np.abs(dh[free]))) if free.any() else 0
            # Convergence diagnostics must describe the state that was
            # actually accepted: a small DAMPED update is not convergence.
            accepted = head-starting_head
            accepted_change_max = float(np.max(np.abs(accepted[free]))) if free.any() else 0
            history.append(dict(outer_iteration=outer+1,krylov_iterations=iterations,
                proposed_head_change_max=proposed_change_max,accepted_head_change_max=accepted_change_max,
                head_change_max=accepted_change_max,
                physical_residual_rms=budget["cell_residual_rms"],head_residual_rms=budget["head_residual_rms"],relaxation=relaxation))
            if converged:
                break
        info = dict(implementation="fast",geometry="terrain",solver_type="terrain_gmres_3d",
            fast_operator="fp64_conservative_transformed_faces",converged=bool(converged),
            final_physical_acceptance=bool(converged),budget=budget,n_cycles_used=total_iterations,
            krylov_iterations=total_iterations,n_outer_iterations=len(history),outer_history=history,
            n_levels=len(self.workspace.levels),workspace_reused=workspace_reused and starting_workspace is self.workspace,
            operator_refreshes=self.workspace.refresh_count,operator_refresh_seconds=refresh_seconds,
            n_zero_capacity_storage_seeds=storage_seeds,n_discharge_component_seeds=discharge_seeds,geometry_revision=self.geometry_revision,n_isolated_cells_excluded=int(np.sum(isolated)),
            cuda_graph_captures=getattr(self.workspace.krylov,"graph_captures",0),
            cuda_graph_replays=getattr(self.workspace.krylov,"graph_replays",0),
            cuda_graph_reused=workspace_reused and graph_captures_before > 0 and getattr(self.workspace.krylov,"graph_captures",0) == graph_captures_before,
            transient=transient,dt=dt,unconfined_storage_mode=unconfined_storage_mode if unconfined else "confined_volume",
            min_saturated_thickness=unconfined_min_sat if unconfined else None,
            final_head_residual_rms=budget["head_residual_rms"])
        info.update(r_rms_end=budget["cell_residual_rms"],h_rms_end=budget["head_residual_rms"],
            dh_rms_lastcheck=float(np.sqrt(np.mean(accepted[free]**2))) if free.any() else 0,
            dh_max_lastcheck=accepted_change_max,
            proposed_dh_rms_lastcheck=float(np.sqrt(np.mean(dh[free]**2))) if free.any() else 0,
            proposed_dh_max_lastcheck=proposed_change_max,requested_min_saturated_thickness=unconfined_min_sat,
            effective_layer_min_saturated_thickness=np.minimum(unconfined_min_sat,self.grid.thickness.min(axis=(1,2))).tolist(),
            multigrid_cycles=(total_iterations+self.workspace.krylov._restart-1)//self.workspace.krylov._restart,
            multigrid_hierarchy="conservative approximate principal part; not exact Galerkin")
        if unconfined:
            info["dry_cell_approximation"] = terrain_dry_diagnostics_3d(operator=final_operator,head=head,source=self.rhs,fixed=self.bc_mask,budget=budget)
        return (head,info) if return_info else head

    def _solve_device(self, *, head, previous, device, unconfined, transient, dt, datum,
                      free, outer_limit, unconfined_relax, unconfined_storage_mode,
                      sy_field, ss_field, dt_unused, key, max_levels, min_coarse_n,
                      krylov_restart, max_cycles, rel_tol, abs_tol_min, head_tolerance,
                      final_budget_relative_tol, unconfined_min_sat, storage_seeds,
                      return_info, terrain_nonlinear_method="picard", terrain_newton_controls=None):
        """GPU-resident Picard: identical physics and final acceptance to the
        host path, with the constitutive refresh, RHS, corrections and
        nonlinear acceptance running on device. Requires the unconfined
        secant-Sy storage mode (steady solves carry zero storage); confined
        and legacy storage modes stay on the host path."""
        from DARCY_WARP_PACKAGE.solvers.terrain_device_3d import (
            DeviceTerrainState, DeviceTerrainPicard, DevicePicardControls)
        if not unconfined or unconfined_storage_mode != "secant_sy":
            raise ValueError("the device Picard path currently requires unconfined secant_sy storage")
        start = perf_counter()
        operator = TerrainReferenceOperator3D(metrics=self.metrics,kx=self.kx,ky=self.ky,kz=self.kz,
            head=head,min_sat=unconfined_min_sat,contacts=self.contacts,trusted_k=True)
        storage,change = self.storage_terms(head=head,previous=previous,transient=transient,
            unconfined=True,mode=unconfined_storage_mode,sy=sy_field,ss=ss_field,coeff=None,dt=dt)
        seeded = seed_discharge_only_components_3d(operator=operator,head=head,source=self.rhs,
            fixed=self.bc_mask,storage=storage,potential_storage=True,topology_cache=self.topology_cache)
        if seeded:
            operator = TerrainReferenceOperator3D(metrics=self.metrics,kx=self.kx,ky=self.ky,kz=self.kz,
                head=head,min_sat=unconfined_min_sat,contacts=self.contacts,trusted_k=True)
            storage,change = self.storage_terms(head=head,previous=previous,transient=transient,
                unconfined=True,mode=unconfined_storage_mode,sy=sy_field,ss=ss_field,coeff=None,dt=dt)
        boundary,boundary_rhs = pack_terrain_boundaries(operator=operator,head=head,datum=datum)
        subfaces,subface_count = pack_device_subfaces(operator=operator,datum=datum)
        contact_rhs = np.zeros(np.prod(self.shape))
        for record in operator.contact_records:
            contact_rhs[record[0]] += record[1]*(record[5]-datum)
        rhs = self.rhs+boundary_rhs+(storage*(previous-datum) if transient else 0)
        rhs[self.bc_mask != 0] = self.bc_values[self.bc_mask != 0]-datum
        rhs[self.grid.active == 0] = 0
        topology_diag = storage + boundary[:,0].reshape(self.shape)
        isolated = _resolve_isolated_free_cells(faces=operator.principal_faces(),active=self.grid.active,
            bc_mask=self.bc_mask,rhs_eff=rhs,storage_diag=topology_diag,topology_cache=self.topology_cache)
        fixed = self.bc_mask.copy()
        fixed[isolated] = 1
        rhs[isolated] = head[isolated]-datum
        packed_coefficients = pack_terrain_coefficients(operator=operator)
        workspace_reused = (self.workspace is not None and self.workspace_key == key
            and self.workspace_topology_matches(fixed=fixed, active=self.grid.active,
                coefficients=packed_coefficients))
        if not workspace_reused:
            if self.workspace is not None:
                self.workspace.close()
            self.workspace = TerrainGPUWorkspace3D(operator=operator,fixed=fixed,storage=storage,
                boundary=boundary,device=device,max_levels=max_levels,min_coarse_n=min_coarse_n,
                restart=krylov_restart,maxiter=max_cycles,rel_tol=rel_tol,abs_tol=abs_tol_min,datum=datum)
            self.workspace_key = key
        else:
            self.workspace.refresh(operator=operator,fixed=fixed,storage=storage,
                boundary=boundary,coefficients=packed_coefficients)
        state = DeviceTerrainState(session=self,device=device,workspace=self.workspace)
        state.upload_static(source=self.rhs,sy=sy_field,ss=ss_field,
            free=(self.grid.active != 0)&(self.bc_mask == 0)&(~isolated),
            subfaces=subfaces,
            contact_refresh=pack_device_contact_refresh(operator=operator, datum=datum))
        state.bc_values.assign(np.where(isolated, head, self.bc_values).ravel())
        state.update_head(host_head=head)
        if previous is not None:
            state.update_previous(host_previous=previous)
        else:
            state.update_previous(host_previous=head)
        driver_options = {}
        driver_class = DeviceTerrainPicard
        if terrain_nonlinear_method == "newton":
            from DARCY_WARP_PACKAGE.solvers.terrain_newton_3d import DeviceTerrainNewton
            driver_class = DeviceTerrainNewton
            driver_options["newton_controls"] = terrain_newton_controls
        driver = driver_class(session=self,state=state,workspace=self.workspace,
            device=device,controls=DevicePicardControls(),datum=datum,restart=krylov_restart,
            krylov_budget=max_cycles,unconfined=True,transient=transient,dt=dt,
            abs_tol_min=abs_tol_min,head_tolerance=head_tolerance,
            budget_relative_tol=final_budget_relative_tol,rel_tol=rel_tol,
            min_sat=unconfined_min_sat,outer_limit=outer_limit,
            unconfined_relax=unconfined_relax, **driver_options)
        result = driver.run(max_outer=outer_limit)
        head_device = state.head.numpy().reshape(self.shape)
        head_device[self.grid.active == 0] = 0
        head_device[self.bc_mask != 0] = self.bc_values[self.bc_mask != 0]
        # Independent final host acceptance at the completed physical state.
        final_operator = TerrainReferenceOperator3D(metrics=self.metrics,kx=self.kx,ky=self.ky,kz=self.kz,
            head=head_device,min_sat=unconfined_min_sat,contacts=self.contacts,trusted_k=True)
        final_storage,final_change = self.storage_terms(head=head_device,previous=previous,
            transient=transient,unconfined=True,mode=unconfined_storage_mode,sy=sy_field,
            ss=ss_field,coeff=None,dt=dt)
        budget = compute_terrain_budget_3d(operator=final_operator,head=head_device,source=self.rhs,
            fixed=fixed,storage_change=final_change,storage_diagonal=final_storage)
        # Reconcile host cache metadata with the final DEVICE state so a
        # later host solve refreshes safely instead of reading an old layout.
        final_boundary, unused_rhs = pack_terrain_boundaries(operator=final_operator,
            head=head_device, datum=datum)
        self.snapshot = (self.k_version, self.geometry_revision, unconfined_min_sat,
            head_device.copy(), final_storage.copy(), final_boundary.copy(),
            fixed.copy(), self.grid.active.copy())
        packed_final = pack_terrain_coefficients(operator=final_operator)
        self.snapshot_packed_positive = packed_final[:, (0, 3, 6)] > 0
        self.workspace.scale_host = state.scale.numpy().reshape(self.shape)
        converged = bool(np.all(np.isfinite(head_device))
            and budget["head_residual_rms"] <= head_tolerance
            and (budget["cell_residual_rms"] <= abs_tol_min
                 or budget["relative_cell_imbalance"] <= final_budget_relative_tol))
        info = dict(implementation="fast",geometry="terrain",solver_type="terrain_device_picard_3d",
            fast_operator="fp64_conservative_transformed_faces_device",
            converged=converged,final_physical_acceptance=converged,budget=budget,
            krylov_iterations=result["krylov_iterations"],
            n_outer_iterations=result["outer_iterations"],
            outer_history=result["history"],
            seconds=result["seconds"],host_scalar_reads=result["host_scalar_reads"],
            host_outer_reductions=result["host_outer_reductions"],
            gmres_norm_readbacks=result["gmres_norm_readbacks"],
            rejected_updates=result["rejected"],
            n_zero_capacity_storage_seeds=storage_seeds,
            n_discharge_component_seeds=int(seeded),
            n_discharge_subfaces=int(subface_count),
            workspace_reused=workspace_reused,operator_refreshes=result["operator_evaluations"],geometry_revision=self.geometry_revision,
            n_isolated_cells_excluded=int(np.sum(isolated)),
            transient=transient,dt=dt,unconfined_storage_mode=unconfined_storage_mode,
            min_saturated_thickness=unconfined_min_sat,
            final_head_residual_rms=budget["head_residual_rms"],
            dry_cell_approximation=terrain_dry_diagnostics_3d(operator=final_operator,
                head=head_device, source=self.rhs, fixed=self.bc_mask, budget=budget),
            device_picard=True, device_stop_reason=result["stop_reason"],
            n_free_cells=driver.n_free)
        info.update(terrain_nonlinear_method=terrain_nonlinear_method,
            jacobian_vector_products=result.get("jacobian_vector_products", 0),
            fallback_count=result.get("fallback_count", 0),
            candidate_evaluations=sum(row["backtracking_evaluations"] for row in result["history"]),
            nonlinear_residual_evaluations=driver.operator_evaluations,
            newton_attempts=sum(row.get("newton_attempted", False) for row in result["history"]),
            newton_accepted_steps=sum(row.get("direction") == "newton" for row in result["history"]),
            nonlinear_backtracks=sum(row["backtracks"] for row in result["history"]))
        if terrain_nonlinear_method == "newton":
            info["solver_type"] = "terrain_device_newton_3d"
        last_trial = result["history"][-1] if result["history"] else {}
        info.update(r_rms_end=budget["cell_residual_rms"],h_rms_end=budget["head_residual_rms"],
            dh_rms_lastcheck=float(last_trial.get("accepted_head_change_rms", 0.0)),
            dh_max_lastcheck=float(last_trial.get("accepted_head_change_max", 0.0)),
            proposed_dh_max_lastcheck=float(last_trial.get("proposed_head_change_max", 0.0)),
            gmres_restart_blocks=sum(row["restart_blocks"] for row in result["history"]),
            multigrid_cycles=result["krylov_iterations"]
                + sum(row["restart_blocks"] for row in result["history"]),
            multigrid_hierarchy="conservative approximate principal part; not exact Galerkin")
        if terrain_nonlinear_method == "newton":
            driver.close()
        state.close()
        return (head_device,info) if return_info else head_device

    def storage_terms(self, *, head, previous, transient, unconfined, mode, sy, ss, coeff, dt):
        zero = np.zeros(self.shape)
        if not transient:
            return zero,zero.copy()
        if unconfined and mode == "secant_sy":
            increment = storage_increment_3d(head=head,previous=previous,bottoms=self.grid.bottoms,
                thickness=self.grid.thickness,sy=sy,ss=ss)
            difference = head-previous
            wet = np.clip(head-self.grid.bottoms,0,self.grid.thickness)
            tangent = sy*((head > self.grid.bottoms)&(head < self.grid.tops))+ss*wet
            # A representable nonzero displacement always uses a secant;
            # there is no dimensional absolute head-change threshold.
            diagonal = np.divide(increment,difference,out=tangent.copy(),where=difference != 0)*self.grid.dx*self.grid.dy/dt
            change = increment*self.grid.dx*self.grid.dy/dt
        else:
            diagonal = coeff*self.grid.volumes/dt
            change = diagonal*(head-previous)
        free = (self.grid.active != 0)&(self.bc_mask == 0)
        return np.where(free,diagonal,0),np.where(free,change,0)
