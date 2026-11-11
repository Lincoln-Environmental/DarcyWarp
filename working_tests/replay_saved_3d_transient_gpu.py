"""Replay an existing six-period MF6 Cartesian history without a native run."""
from pathlib import Path
from time import perf_counter
import hashlib
import json

import flopy
import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D
from sanity_3d_controls import read_profiles


def load_reference(*, reference_path):
    with np.load(file=reference_path, allow_pickle=False) as saved:
        data = {name: saved[name].copy() for name in saved.files}
    simulation = flopy.mf6.MFSimulation.load(sim_ws=str(reference_path.parent/'mf6_transient'), verbosity_level=0)
    native = simulation.get_model()
    shape = data['initial_head'].shape
    kx, kz = native.npf.k.array.copy(), native.npf.k33.array.copy()
    np.testing.assert_allclose(actual=kx, desired=np.broadcast_to(array=data['hk_2d'], shape=shape))
    np.testing.assert_allclose(actual=native.ic.strt.array, desired=data['initial_head'], atol=1e-7)
    np.testing.assert_array_equal(actual=native.dis.idomain.array, desired=data['active'])
    np.testing.assert_allclose(actual=native.sto.ss.array, desired=float(data['ss']))
    np.testing.assert_allclose(actual=simulation.tdis.perioddata.array['perlen'], desired=float(data['dt_days']))
    np.testing.assert_array_equal(actual=simulation.tdis.perioddata.array['nstp'], desired=1)
    if np.any(native.npf.icelltype.array):
        raise ValueError('This reference must be confined')
    return data, kx, kz


def replay(*, reference_path, output, device, implementations, repeats, options, tolerance):
    wp.init()
    data, kx, kz = load_reference(reference_path=reference_path)
    nz, ny, nx = kx.shape
    free = (data['active'] != 0)&(data['bc_mask'] == 0)
    result = dict(reference=str(reference_path), sha256=hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        metadata_verified=True, tolerance_m=tolerance, runs=[])
    for implementation in implementations:
        for repeat in range(repeats+1):
            previous = data['initial_head'].copy()
            rows, heads = [], []
            wp.synchronize_device(device=device)
            started = perf_counter()
            with WarpDarcySolver3D(nx=nx, ny=ny, nz=nz, dx=float(data['dx']), dz=float(data['dz']),
                    device=device, implementation=implementation) as model:
                source = np.zeros_like(a=previous)
                model.build_from_K_fields(kx_field=kx, ky_field=kx, kz_field=kz, active=data['active'],
                    bc_mask=data['bc_mask'], bc_values=data['bc_values'], rhs=source, initial_head=previous)
                for period, rate in enumerate(data['recharge_rates']):
                    source[0] = rate*float(data['dx'])**2*data['active'][0]
                    model.update_rhs(rhs=source)
                    model.update_initial_head(initial_head=previous)
                    old = previous.copy()
                    head, info = model.solve(transient=True, storage_coeff=float(data['ss']),
                        dt=float(data['dt_days']), head_prev=previous, **options)
                    np.testing.assert_array_equal(actual=previous, desired=old)
                    difference = head[free]-data['heads_per_period'][period][free]
                    maximum = float(np.max(np.abs(difference)))
                    rows.append(dict(period=period, accepted=bool(info['converged'] and maximum <= tolerance),
                        head_rmse_m=float(np.sqrt(np.mean(difference**2))), head_max_m=maximum, info=info))
                    heads.append(head.copy())
                    previous = head.copy()
                wp.synchronize_device(device=device)
                elapsed = perf_counter()-started
            row = dict(implementation=implementation, repeat=repeat, warmup=repeat == 0,
                seconds=elapsed, accepted=all(item['accepted'] for item in rows), periods=rows)
            result['runs'].append(row)
            output.joinpath('cartesian_saved_transient.json').write_text(data=json.dumps(obj=result, indent=2, default=str))
            np.savez_compressed(file=output/f'cartesian_saved_{implementation}_{repeat}.npz', heads=np.stack(arrays=heads))
            print(f'Saved transient {implementation} repeat={repeat}: {elapsed:.4f}s accepted={row["accepted"]}', flush=True)


if __name__ == "__main__":
    reference_path = Path('/home/patrickdurney/PycharmProjects/DarcyWarp/DARCY_WARP_PACKAGE/data/working_tests/mf6_vs_warp_3d/mf6_transient_heads.npz')
    output = Path('release_evidence/gpu_20261111_2d_sanity')
    device = 'cuda:0'
    implementations = ('classic', 'fast')
    repeats = 3
    profiles = read_profiles(root=Path('.'))
    options = profiles['transient']
    tolerance = 1e-2
    replay(reference_path=reference_path, output=output, device=device, implementations=implementations,
        repeats=repeats, options=options, tolerance=tolerance)
