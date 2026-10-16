"""Independent MF6 6.6.2 Newton DIS TPFA flux reevaluation, no solver update.

Supported only: positive K, unstaggered full stacks, no THICKSTRT, VARIABLECV,
PERCHED, XT3D, rotation, deactivation, or native DRN; original terrain outlets
may be applied separately as custom-coupled boundary physics.
"""
import numpy as np
from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import harmonic_resistance


def quadratic_saturation(*,head,tops,bottoms,epsilon):
    fraction=np.clip((head-bottoms)/(tops-bottoms),0.,1.)
    a=1./(1.-epsilon)
    return np.where(fraction<epsilon,a*.5*fraction**2/epsilon,
        np.where(fraction<1.-epsilon,a*fraction+.5*(1.-a),
            np.where(fraction<1.,1.-a*.5*(1.-fraction)**2/epsilon,1.)))


def tpfa_fluxes(*,case,head,epsilon=1e-6):
    grid=case.grid()
    a=case.arrays
    saturation=quadratic_saturation(head=head,tops=grid.tops,bottoms=grid.bottoms,epsilon=epsilon)
    fluxes=[]
    for axis,key,distance,width in ((2,'kx',grid.dx,grid.dy),(1,'ky',grid.dy,grid.dx)):
        l,u=[slice(None)]*3,[slice(None)]*3
        l[axis],u[axis]=slice(None,-1),slice(1,None)
        l,u=tuple(l),tuple(u)
        C,unused,unused=harmonic_resistance(left=a[key][l]*grid.thickness[l],right=a[key][u]*grid.thickness[u],dl=distance/2,dr=distance/2)
        upstream=np.where(head[l]>head[u],saturation[l],saturation[u])
        fluxes.append(C*width*upstream*(head[l]-head[u]))
    C,unused,unused=harmonic_resistance(left=a['kz'][:-1],right=a['kz'][1:],dl=grid.thickness[:-1]/2,dr=grid.thickness[1:]/2)
    fluxes.append(C*grid.dx*grid.dy*(head[:-1]-head[1:]))
    return tuple(fluxes)
