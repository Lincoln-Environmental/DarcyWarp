"""Preimplementation supplied-head check of conventional DIS connection laws."""
from pathlib import Path
import hashlib
import subprocess

import flopy
from flopy.mf6.utils.binarygrid_util import MfGrdFile
import numpy as np

from DARCY_WARP_PACKAGE.validation.npf662_reference import quadratic_saturation_derivative
from DARCY_WARP_PACKAGE.validation.terrain_case import atomic_json


def verify_native_connection_laws(*, output, executable, interfaces, kx, ky, kz, dx, dy, convertible):
    version = subprocess.check_output([str(executable), '-v'], text=True).strip()
    checksum = hashlib.sha256(executable.read_bytes()).hexdigest()
    if version != 'mf6: 6.6.2 05/12/2025' or checksum != '21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176':
        raise ValueError('Pinned reference executable mismatch')
    output.mkdir(parents=True, exist_ok=False)
    thickness = interfaces[:-1] - interfaces[1:]
    bottom = interfaces[1:]
    head = bottom + .6*thickness
    shape = thickness.shape
    nz, ny, nx = shape
    reverse = (slice(None), slice(None, None, -1), slice(None))
    np.savez_compressed(output/'inputs.npz', z_interfaces=interfaces, kx=kx, ky=ky, kz=kz, head=head, dx=dx, dy=dy)
    sim = flopy.mf6.MFSimulation(sim_name='contract', sim_ws=str(output), exe_name=str(executable), verbosity_level=0)
    sim.simulation_data.float_precision = 17; sim.simulation_data.float_characters = 25
    flopy.mf6.ModflowTdis(sim, time_units='DAYS', perioddata=[(1., 1, 1.)])
    flopy.mf6.ModflowIms(sim, outer_dvclose=1e-12, inner_dvclose=1e-12,
        rcloserecord=[1e-12, 'STRICT'], linear_acceleration='BICGSTAB')
    gwf = flopy.mf6.ModflowGwf(sim, modelname='contract', newtonoptions='UNDER_RELAXATION' if convertible else None)
    flopy.mf6.ModflowGwfdis(gwf, nlay=nz, nrow=ny, ncol=nx, delr=dx, delc=dy,
        top=interfaces[0, ::-1], botm=bottom[reverse], length_units='METERS')
    flopy.mf6.ModflowGwfic(gwf, strt=head[reverse])
    flopy.mf6.ModflowGwfnpf(gwf, icelltype=int(convertible), k=kx[reverse], k22=ky[reverse], k33=kz[reverse],
        angle1=0., angle2=0., angle3=0., save_flows=True, save_saturation=True)
    bc = [((int(k), ny-1-int(j), int(i)), float(head[k,j,i])) for k,j,i in np.ndindex(shape)]
    flopy.mf6.ModflowGwfchd(gwf, stress_period_data=bc, save_flows=True)
    flopy.mf6.ModflowGwfoc(gwf, head_filerecord='contract.hds', budget_filerecord='contract.cbc',
        saverecord=[('HEAD', 'ALL'), ('BUDGET', 'ALL')])
    sim.write_simulation(silent=True)
    run = subprocess.run([str(executable)], cwd=output, capture_output=True, text=True, check=False)
    (output/'stdout.log').write_text(run.stdout+run.stderr)
    if run.returncode or 'Normal termination' not in run.stdout:
        raise AssertionError(run.stdout)
    returned = flopy.utils.HeadFile(str(output/'contract.hds'), precision='double').get_data()[reverse]
    np.testing.assert_array_equal(returned, head)
    grid = MfGrdFile(str(output/'contract.dis.grb'), verbose=False)
    budget = flopy.utils.CellBudgetFile(str(output/'contract.cbc'), precision='double')
    qja = budget.get_data(text='FLOW-JA-FACE')[-1].ravel()
    mapping = np.arange(head.size).reshape(shape)[reverse].ravel()
    actual = {}
    for n in range(head.size):
        for p in range(grid.ia[n]+1, grid.ia[n+1]):
            actual[(int(mapping[n]), int(mapping[grid.ja[p]]))] = -float(qja[p])
    saturation, unused = quadratic_saturation_derivative(head=head, bottom=bottom, thickness=thickness, cell_type=int(convertible))
    ids = np.arange(head.size).reshape(shape)
    rows = []
    for axis, conductivity, distance, width in ((2,kx,dx,dy), (1,ky,dy,dx), (0,kz,1.,dx*dy)):
        l,u = [slice(None)]*3, [slice(None)]*3
        l[axis],u[axis] = slice(None,-1),slice(1,None); l,u = tuple(l),tuple(u)
        if axis:
            c0 = width/(.5*distance/(conductivity[l]*thickness[l])+.5*distance/(conductivity[u]*thickness[u]))
        else:
            c0 = width/(.5*thickness[l]/conductivity[l]+.5*thickness[u]/conductivity[u])
        donor_left = (head[l]>head[u]) | ((head[l]==head[u]) & (axis==1))
        c = c0*np.where(donor_left,saturation[l],saturation[u]) if axis else c0
        expected = c*(head[l]-head[u])
        native = np.array([actual[(int(a),int(b))] for a,b in zip(ids[l].ravel(),ids[u].ravel())]).reshape(expected.shape)
        np.testing.assert_allclose(native, expected, rtol=3e-12, atol=2e-11)
        rows.append(dict(axis=axis, faces=expected.size, flux_max_error=float(np.max(np.abs(native-expected)))))
    budget.close()
    result = dict(native_success=True, supplied_state_only=True, head_not_advanced=True,
        convertible=convertible, comparisons=rows, mf6_version=version, mf6_sha256=checksum,
        equations='Conventional same-k harmonic full thickness; no overlap clipping, slope distance, rotation or XT3D')
    atomic_json(path=output/'comparison.json', value=result)
    return result


if __name__ == "__main__":
    output = Path('validation_artifacts/npf662_layered_v1/contract')
    executable = Path('/bin/modflow/mf6')
    nz, ny, nx = 3, 2, 4
    dx, dy = 2., 3.
    k,j,i = np.indices((nz,ny,nx))
    surface = 20. + 5.*i[0] + 2.*j[0]
    thickness = 1. + .5*k + .1*j + .2*i
    interfaces = np.concatenate([surface[None], surface[None]-np.cumsum(thickness,axis=0)],axis=0)
    kx = 1.+k+i
    ky = .6*kx
    kz = .15*kx
    verify_native_connection_laws(output=output/'confined', executable=executable, interfaces=interfaces,
        kx=kx, ky=ky, kz=kz, dx=dx, dy=dy, convertible=False)
    verify_native_connection_laws(output=output/'newton', executable=executable, interfaces=interfaces,
        kx=kx, ky=ky, kz=kz, dx=dx, dy=dy, convertible=True)
