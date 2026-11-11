"""Shared deterministic layered NPF benchmark geometry, independent of Warp."""
import numpy as np
from DARCY_WARP_PACKAGE.validation.npf662_reference import NPF662LayeredCase


def generate_case(*, settings, convertible):
    nz, ny, nx = settings['shape']
    indices = np.asarray(convertible)
    if indices.ndim != 1 or (indices.size and (indices.dtype.kind not in 'iu'
            or np.any(indices < 0) or np.any(indices >= nz))):
        raise ValueError('convertible must be a sequence of layer indices, not a Boolean flag')
    rng = np.random.default_rng(seed=settings['seed'])
    y, x = np.meshgrid(np.linspace(start=0., stop=1., num=ny),
                       np.linspace(start=0., stop=1., num=nx), indexing='ij')
    surface = settings['surface_base'] + settings['surface_rise'] * x
    surface += settings['surface_wave'] * np.sin(2.*np.pi*y) * np.sin(np.pi*x)
    layers = np.arange(nz)[:, None, None]
    thickness = settings['layer_thickness'] + settings['layer_increment']*layers
    thickness = thickness + settings['thickness_wave'] * np.sin(
        2.*np.pi*x[None] + settings['layer_phase']*layers) * np.cos(
        2.*np.pi*y[None] - settings['layer_phase']*layers)
    interfaces = np.concatenate([surface[None], surface[None]-np.cumsum(a=thickness, axis=0)], axis=0)
    pattern = settings['logk_wave']*np.sin(2.*np.pi*x[None]+layers) * np.cos(2.*np.pi*y[None])
    noise = rng.uniform(low=-settings['logk_noise'], high=settings['logk_noise'], size=(nz, ny, nx))
    kx = settings['layer_k'][:, None, None] * np.exp(pattern+noise)
    ky = kx * (settings['ky_ratio'] + settings['ky_wave']*np.sin(2.*np.pi*y[None]+layers))
    kz = kx * (settings['kz_ratio'] + settings['kz_wave']*np.cos(2.*np.pi*x[None]-layers))
    active = np.ones(shape=(nz, ny, nx), dtype=np.int32)
    fixed = np.zeros_like(a=active); fixed[:, :, 0] = 1
    values = np.full(shape=active.shape, fill_value=settings['coast_head'], dtype=np.float64)
    source = np.zeros_like(a=kx)
    source[0, :, 1:] = settings['recharge']*settings['dx']*settings['dy']*(
        1.+settings['recharge_wave']*np.sin(2.*np.pi*y[:, 1:])*np.sin(np.pi*x[:, 1:]))
    initial = settings['coast_head'] + settings['cold_rise']*np.broadcast_to(array=x, shape=active.shape)
    initial[fixed != 0] = values[fixed != 0]
    types = np.zeros_like(a=active)
    types[indices.astype(np.intp)] = 1
    case = NPF662LayeredCase(kx=kx, ky=ky, kz=kz, active=active, fixed=fixed,
        values=values, source=source, initial=initial, dx=settings['dx'], dy=settings['dy'],
        z_interfaces=interfaces, cell_type=types)
    case.validate()
    return case


def make_layered_dry_source(*,initial):
    shape=(1,1,2);one=np.ones(shape)
    return NPF662LayeredCase(kx=one,ky=.6*one,kz=.15*one,active=one.astype(int),
        fixed=np.array([0,1]).reshape(shape),values=np.zeros(shape),source=np.array([1.,0.]).reshape(shape),
        initial=np.full(shape,initial),dx=1.,dy=1.,
        z_interfaces=np.array([[[2.,4.]],[[0.,0.]]]),cell_type=1)
