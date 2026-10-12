"""Independent NumPy/SciPy total-head terrain finite-volume reference.

Adapted mathematical flux formulation from RichardsWarp terrain_reference
(feature/scientific-audit-fixes). No pressure-head/gravity term: h is TOTAL
hydraulic head. Positive flux points along increasing x,y,depth. Every face
is evaluated once. Sparse assembly uses gradient/incidence matrices, not GPU
stencil coefficients, enabling an independent parity check.
"""
from dataclasses import dataclass

import numpy as np
from scipy import sparse


def harmonic_resistance(*, left, right, dl, dr):
    """Stable series conductance and half-cell tangential weights."""
    tl, tr = left / dl, right / dr
    small, large = np.minimum(tl, tr), np.maximum(tl, tr)
    ratio = np.divide(small, large, out=np.zeros_like(small), where=large > 0)
    conductance = small / (1 + ratio)
    scaled_l = np.divide(tl, large, out=np.zeros_like(tl), where=large > 0)
    scaled_r = np.divide(tr, large, out=np.zeros_like(tr), where=large > 0)
    denom = scaled_l + scaled_r
    wl = np.divide(scaled_r, denom, out=np.zeros_like(tl), where=denom > 0)
    wr = np.divide(scaled_l, denom, out=np.zeros_like(tr), where=denom > 0)
    return conductance, wl, wr


def field_3d(*, value, shape, name, nonnegative=False):
    array = np.broadcast_to(np.asarray(value, dtype=np.float64), shape).copy()
    if not np.all(np.isfinite(array)) or (nonnegative and np.any(array < 0)):
        raise ValueError(f"{name} must be finite" + (" and nonnegative" if nonnegative else ""))
    return array


@dataclass(frozen=True)
class TerrainContact3D:
    """One exterior subface: total prescribed head or discharge-only head.

    area is the physical subface area; top/bottom areas are projected onto
    the map plane internally. distance is the positive logical
    centre-to-interface distance. elevation is the
    physical contact elevation, preserved for coastal/surface diagnostics.
    conductance overrides normal K*area/distance, retaining metric correction.
    """
    cell: tuple[int, int, int]
    direction: str
    area: float
    distance: float
    elevation: float
    head: float
    kind: str = "head"
    conductance: float | None = None
    x_offset: float = 0.0
    y_offset: float = 0.0


class TerrainReferenceOperator3D:
    def __init__(self, *, metrics, kx, ky, kz, head=None, min_sat=0.01, contacts=(),
                 trusted_k=False):
        self.metrics = metrics
        self.grid = metrics.grid
        self.shape = self.grid.shape
        self.active = self.grid.active.astype(bool)
        if trusted_k:
            # The session owns and validates these fields once; rebuilding
            # operators per nonlinear iteration then costs no re-validation
            # copies. The fields must not be mutated during the operator's
            # lifetime (the session only replaces them wholesale).
            self.kx, self.ky, self.kz = kx, ky, kz
        else:
            self.kx = field_3d(value=kx, shape=self.shape, name="kx", nonnegative=True)
            self.ky = field_3d(value=ky, shape=self.shape, name="ky", nonnegative=True)
            self.kz = field_3d(value=kz, shape=self.shape, name="kz", nonnegative=True)
        nonflat = np.any(metrics.zx != 0) or np.any(metrics.zy != 0) or np.any(metrics.zxf != 0) or np.any(metrics.zyf != 0)
        active_fields = (self.kx[self.active],self.ky[self.active],self.kz[self.active])
        has_contacts = bool(contacts)
        if (nonflat or has_contacts) and any(np.any(field == 0) for field in active_fields) and not (not has_contacts and all(np.all(field == 0) for field in active_fields)):
            raise ValueError("nonflat terrain or explicit coastal contacts require positive Kx/Ky/Kz on hydraulic cells; mixed exactly-zero material needs an impermeable-interface closure, which is unsupported (no small positive K cutoff)")
        if not np.isfinite(min_sat) or min_sat < 0:
            raise ValueError("unconfined_min_sat must be finite and nonnegative")
        self.true_sat = self.grid.thickness if head is None else np.clip(np.asarray(head) - self.grid.bottoms, 0, self.grid.thickness)
        self.regularized_sat = self.grid.thickness if head is None else np.clip(np.asarray(head) - self.grid.bottoms, np.minimum(min_sat, self.grid.thickness), self.grid.thickness)
        fraction = self.regularized_sat / self.grid.thickness
        # Derive the tensor from physical diag(Kx*f,Ky*f,Kz), rather than
        # multiplying the entire transformed tensor by an arbitrary factor.
        # b = local layer thickness; A00 = b*Kx*f, A02 = Kx*f*z_x with z_x
        # the in-layer centred interface slope, A22 = (Kx*f*z_x²+Ky*f*z_y²+Kz)/b.
        self.ex = self.kx * fraction
        self.ey = self.ky * fraction
        self.coefficients = self._build_face_coefficients()
        self.contacts = tuple(contacts)
        self.contact_records = self._validate_contacts()

    def _build_face_coefficients(self):
        g, m = self.grid, self.metrics
        # Horizontal faces: the normal conductance is the harmonic series of
        # half-cell TRANSMISSIVITIES (K*f*thickness), so the face area is the
        # unit-s map width and variable layer thickness enters physically.
        transmissivity_x = self.ex * g.thickness
        transmissivity_y = self.ey * g.thickness
        specs = ((2, transmissivity_x, g.dx/2, g.dx/2, g.dy, m.zxf),
                 (1, transmissivity_y, g.dy/2, g.dy/2, g.dx, m.zyf))
        result = []
        for axis, conductivity, dl, dr, area, face_slope in specs:
            left, right = [slice(None)]*3, [slice(None)]*3
            left[axis], right[axis] = slice(None, -1), slice(1, None)
            left, right = tuple(left), tuple(right)
            normal, wl, wr = harmonic_resistance(left=conductivity[left], right=conductivity[right], dl=dl, dr=dr)
            valid = self.active[left] & self.active[right]
            normal = normal * area * valid
            bl = wl * conductivity[left] * face_slope * area * valid
            br = wr * conductivity[right] * face_slope * area * valid
            result.append((normal, bl, br))
        # Vertical (s-direction) faces share one interface, so both the
        # normal conductance and the metric cross terms evaluate THAT
        # interface's own slopes. The cross split keeps the harmonic
        # resistance weights of the face: low-permeability cells never
        # receive couplings scaled to their transmissive neighbour, which
        # preserves the row structure that stiff aquifer-basement contrasts
        # rely on. Because the resistance-weighted mean of the cell-centre
        # slopes differs from the interface slope, the metric terms of an
        # affine total-head field would not cancel on nonparallel contacts;
        # the gamma terms subtract exactly that defect (resistance-weighted
        # too, and identically zero when the interface parallels the cell
        # centre surfaces), restoring exactness for uniform conductivity.
        interface_x = m.zx_interfaces[1:-1]
        interface_y = m.zy_interfaces[1:-1]
        left, right = (slice(None, -1), slice(None), slice(None)), (slice(1, None), slice(None), slice(None))
        valid = self.active[left] & self.active[right]
        azz_top = self.ex[left]*interface_x**2 + self.ey[left]*interface_y**2 + self.kz[left]
        azz_bottom = self.ex[right]*interface_x**2 + self.ey[right]*interface_y**2 + self.kz[right]
        normal, wl, wr = harmonic_resistance(left=azz_top, right=azz_bottom,
            dl=g.thickness[:-1]/2, dr=g.thickness[1:]/2)
        normal = normal*g.dx*g.dy*valid
        area = g.dx*g.dy
        result.append((normal,
            wl*self.ex[left]*interface_x*area*valid,
            wr*self.ex[right]*interface_x*area*valid,
            wl*self.ey[left]*interface_y*area*valid,
            wr*self.ey[right]*interface_y*area*valid,
            wl*(self.ex[left]*interface_x*(m.zx[:-1]-interface_x)+self.ey[left]*interface_y*(m.zy[:-1]-interface_y))*area*valid,
            wr*(self.ex[right]*interface_x*(m.zx[1:]-interface_x)+self.ey[right]*interface_y*(m.zy[1:]-interface_y))*area*valid))
        return tuple(result)

    def _validate_contacts(self):
        directions = {"west": (2,-1), "east": (2,1), "south": (1,-1), "north": (1,1), "top": (0,-1), "bottom": (0,1)}
        records = []
        areas = {}
        m = self.metrics
        for contact in self.contacts:
            cell = contact.cell
            if len(cell) != 3 or any(c < 0 or c >= s for c,s in zip(cell,self.shape)) or not self.active[cell]:
                raise ValueError(f"contact cell {cell} is outside the active terrain")
            if contact.direction not in directions or contact.kind not in {"head", "discharge_only"}:
                raise ValueError("contact direction or kind unsupported")
            if not all(np.isfinite(v) for v in (contact.area,contact.distance,contact.elevation,contact.head,contact.x_offset,contact.y_offset)) or min(contact.area,contact.distance) <= 0:
                raise ValueError("contact area/distance must be positive, all contact values finite")
            if contact.conductance is not None and (not np.isfinite(contact.conductance) or contact.conductance < 0):
                raise ValueError("contact conductance must be finite and nonnegative")
            axis, sign = directions[contact.direction]
            if abs(contact.x_offset) > self.grid.dx/2 or abs(contact.y_offset) > self.grid.dy/2 or (axis == 2 and contact.x_offset != 0) or (axis == 1 and contact.y_offset != 0):
                raise ValueError("contact offsets must be tangential and within the owning face")
            neighbor = list(cell)
            neighbor[axis] += sign
            if 0 <= neighbor[axis] < self.shape[axis] and self.active[tuple(neighbor)]:
                raise ValueError(f"contact at {cell}/{contact.direction} double-counts an internal face")
            layer = cell[0]
            column = cell[1:]
            if axis == 0:
                # Top/bottom exterior faces follow their own bounding interface.
                interface = layer if sign < 0 else layer + 1
                zx_interface = m.zx_interfaces[interface][column]
                zy_interface = m.zy_interfaces[interface][column]
                surface_factor = float(np.sqrt(1+zx_interface**2+zy_interface**2))
                available = self.grid.dx*self.grid.dy*surface_factor
            else:
                surface_factor = 1.0
                available = (self.grid.dy if axis == 2 else self.grid.dx)*self.grid.thickness[cell]
            key = (cell,contact.direction)
            areas[key] = areas.get(key,0) + contact.area
            if areas[key] > available*(1+1e-12):
                raise ValueError(f"subface areas exceed exterior face at {key}")
            if axis == 0:
                normal = self.ex[cell]*zx_interface**2 + self.ey[cell]*zy_interface**2 + self.kz[cell]
            else:
                normal = (self.ex if axis == 2 else self.ey)[cell]
            projected_area = contact.area/surface_factor
            conductance = normal*projected_area/contact.distance if contact.conductance is None else contact.conductance
            scale = projected_area if contact.conductance is None else (conductance*contact.distance/normal if normal > 0 else 0)
            # Reconstruct the layer's bounding interfaces at the physical
            # contact point; its elevation offset is measured from the layer
            # centre surface there, so coastal geometry never inherits the
            # owning cell's centre elevation.
            dx_point = sign*contact.distance if axis == 2 else contact.x_offset
            dy_point = sign*contact.distance if axis == 1 else contact.y_offset
            top_point = self.grid.z_interfaces[layer][column] + m.zx_interfaces[layer][column]*dx_point + m.zy_interfaces[layer][column]*dy_point
            bottom_point = self.grid.z_interfaces[layer+1][column] + m.zx_interfaces[layer+1][column]*dx_point + m.zy_interfaces[layer+1][column]*dy_point
            centre_point = .5*(top_point+bottom_point)
            depth_offset = centre_point-contact.elevation
            tolerance = 64*np.finfo(np.float64).eps*max(abs(centre_point),abs(contact.elevation),1.)
            if axis != 0 and not bottom_point-tolerance <= contact.elevation <= top_point+tolerance:
                raise ValueError(f"contact elevation is outside its layer at {cell}/{contact.direction}")
            if axis == 0 and abs(depth_offset-sign*contact.distance) > tolerance:
                raise ValueError("top/bottom contact elevation must match its logical distance and centroid offsets")
            if (self.shape[0] == 1 and axis != 0 and abs(depth_offset) > tolerance) or (self.shape[2] == 1 and contact.x_offset != 0) or (self.shape[1] == 1 and contact.y_offset != 0):
                raise ValueError("offset subfaces require enough cells to reconstruct tangential gradients")
            zx,zy = m.zx[cell],m.zy[cell]
            if axis == 2:
                cross = (sign*self.ex[cell]*zx*scale-conductance*depth_offset,0,-conductance*contact.y_offset)
            elif axis == 1:
                cross = (sign*self.ey[cell]*zy*scale-conductance*depth_offset,-conductance*contact.x_offset,0)
            else:
                cross = (0,sign*self.ex[cell]*zx*scale-conductance*contact.x_offset,
                         sign*self.ey[cell]*zy*scale-conductance*contact.y_offset)
            records.append((np.ravel_multi_index(cell,self.shape),conductance,*cross,contact.head,contact.kind == "discharge_only"))
        return tuple(records)

    def fluxes(self, *, head):
        h = np.asarray(head,dtype=np.float64)
        if h.shape != self.shape or not np.all(np.isfinite(h)):
            raise ValueError("head must be finite and grid-shaped")
        hz = (self.metrics.gz @ h.ravel()).reshape(self.shape)
        hx = (self.metrics.gx @ h.ravel()).reshape(self.shape)
        hy = (self.metrics.gy @ h.ravel()).reshape(self.shape)
        cx, cy, cz = self.coefficients
        qx = cx[0]*(h[:,:,:-1]-h[:,:,1:])-cx[1]*hz[:,:,:-1]-cx[2]*hz[:,:,1:]
        qy = cy[0]*(h[:,:-1]-h[:,1:])-cy[1]*hz[:,:-1]-cy[2]*hz[:,1:]
        qz = cz[0]*(h[:-1]-h[1:])-cz[1]*hx[:-1]-cz[2]*hx[1:]-cz[3]*hy[:-1]-cz[4]*hy[1:]-cz[5]*hz[:-1]-cz[6]*hz[1:]
        boundary = np.empty(len(self.contact_records))
        for index, record in enumerate(self.contact_records):
            cell,c,bz,bx,by,hb,discharge = record
            q = c*(h.ravel()[cell]-hb)-bz*hz.ravel()[cell]-bx*hx.ravel()[cell]-by*hy.ravel()[cell]
            boundary[index] = max(q,0) if discharge else q
        return qx,qy,qz,boundary

    def divergence(self, *, fluxes):
        qx,qy,qz,boundary = fluxes
        result = np.zeros(self.shape)
        result[:,:,:-1] += qx
        result[:,:,1:] -= qx
        result[:,:-1] += qy
        result[:,1:] -= qy
        result[:-1] += qz
        result[1:] -= qz
        for q, record in zip(boundary,self.contact_records):
            result.ravel()[record[0]] += q
        return result

    def sparse_matrix(self, *, head_for_contacts=None):
        """Assemble via face gradient rows and incidence; independently of GPU."""
        n = np.prod(self.shape)
        ids = np.arange(n).reshape(self.shape)
        matrix = sparse.csr_matrix((n,n))
        for axis, coeff in zip((2,1,0),self.coefficients):
            left,right = [slice(None)]*3,[slice(None)]*3
            left[axis],right[axis] = slice(None,-1),slice(1,None)
            l,r = ids[tuple(left)].ravel(),ids[tuple(right)].ravel()
            rows = np.arange(l.size)
            select_l = sparse.coo_matrix((np.ones(l.size),(rows,l)),shape=(l.size,n)).tocsr()
            select_r = sparse.coo_matrix((np.ones(l.size),(rows,r)),shape=(l.size,n)).tocsr()
            jump = select_l-select_r
            face = sparse.diags(coeff[0].ravel()) @ jump
            if axis != 0:
                face -= sparse.diags(coeff[1].ravel()) @ select_l @ self.metrics.gz
                face -= sparse.diags(coeff[2].ravel()) @ select_r @ self.metrics.gz
            else:
                for value, select, gradient in ((coeff[1],select_l,self.metrics.gx),(coeff[2],select_r,self.metrics.gx),(coeff[3],select_l,self.metrics.gy),(coeff[4],select_r,self.metrics.gy),(coeff[5],select_l,self.metrics.gz),(coeff[6],select_r,self.metrics.gz)):
                    face -= sparse.diags(value.ravel()) @ select @ gradient
            matrix += jump.T @ face
        offset = np.zeros(n)
        boundary_flux = None if head_for_contacts is None else self.fluxes(head=head_for_contacts)[3]
        for index,record in enumerate(self.contact_records):
            cell,c,bz,bx,by,hb,discharge = record
            if discharge and (boundary_flux is None or boundary_flux[index] <= 0):
                continue
            row = sparse.csr_matrix(([c],([0],[cell])),shape=(1,n))
            row -= bz*self.metrics.gz[cell] + bx*self.metrics.gx[cell] + by*self.metrics.gy[cell]
            emit = sparse.csr_matrix(([1.0],([cell],[0])),shape=(n,1))
            matrix += emit @ row
            offset[cell] -= c*hb
        return matrix.tocsr(),offset.reshape(self.shape)

    def principal_faces(self):
        faces = [np.zeros(self.shape) for unused in range(6)]
        for axis, index, coeff in zip((2,1,0),(0,2,4),self.coefficients):
            left,right = [slice(None)]*3,[slice(None)]*3
            left[axis],right[axis] = slice(None,-1),slice(1,None)
            faces[index][tuple(left)] = coeff[0]
            faces[index+1][tuple(right)] = coeff[0]
        return tuple(faces)
