import numpy as np
import warp as wp
import pytest

from DARCY_WARP_PACKAGE.config import NP_FLOAT, WP_FLOAT
from DARCY_WARP_PACKAGE.kernels_3d import vertical_line_relaxation_7point_kernel
from DARCY_WARP_PACKAGE.solvers_3d import solve_multigrid_kcycle_7point_3d

wp.init()

def test_column_only():
    nx, ny, nz = 1, 1, 10
    shape = (nz, ny, nx)

    tx_p = np.zeros(shape, dtype=NP_FLOAT)
    tx_m = np.zeros(shape, dtype=NP_FLOAT)
    ty_p = np.zeros(shape, dtype=NP_FLOAT)
    ty_m = np.zeros(shape, dtype=NP_FLOAT)
    tz_p = np.ones(shape, dtype=NP_FLOAT) * 2.0
    tz_m = np.ones(shape, dtype=NP_FLOAT) * 2.0
    
    # Bottom cell no tz_m
    tz_m[0, :, :] = 0.0
    # Top cell no tz_p
    tz_p[-1, :, :] = 0.0

    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape, dtype=NP_FLOAT)
    storage_diag = np.zeros(shape, dtype=NP_FLOAT)

    rhs = np.zeros(shape, dtype=NP_FLOAT)
    rhs[5, 0, 0] = 10.0 # impulse
    
    # bc at bottom
    bc_mask[0, 0, 0] = 1
    bc_values[0, 0, 0] = 5.0

    # Solve with exact vertical line smoother
    head, info = solve_multigrid_kcycle_7point_3d(
        tx_p=tx_p, tx_m=tx_m, ty_p=ty_p, ty_m=ty_m, tz_p=tz_p, tz_m=tz_m,
        rhs=rhs, active=active, bc_mask=bc_mask, bc_values=bc_values,
        storage_diag=storage_diag, max_cycles=1, nu_pre=1, nu_post=0, nu_coarse=0,
        smoother="vertical_line", line_omega=1.0, return_info=True
    )
    
    # Verify tridiagonal solution manually or check that residual is zero
    if info["r_rms_end"] >= 1e-5:
        print("Failed!")
        print("r_rms_end:", info["r_rms_end"])
        print("head:", head.ravel())
        print("info:", info)
    assert info["r_rms_end"] < 1e-5

def test_small_3d():
    nx, ny, nz = 3, 3, 4
    shape = (nz, ny, nx)

    tx_p = np.ones(shape, dtype=NP_FLOAT)
    tx_m = np.ones(shape, dtype=NP_FLOAT)
    ty_p = np.ones(shape, dtype=NP_FLOAT)
    ty_m = np.ones(shape, dtype=NP_FLOAT)
    tz_p = np.ones(shape, dtype=NP_FLOAT) * 3.0
    tz_m = np.ones(shape, dtype=NP_FLOAT) * 3.0

    active = np.ones(shape, dtype=np.int32)
    bc_mask = np.zeros(shape, dtype=np.int32)
    bc_values = np.zeros(shape, dtype=NP_FLOAT)
    storage_diag = np.zeros(shape, dtype=NP_FLOAT)

    rhs = np.random.rand(*shape).astype(NP_FLOAT)

    # set some boundaries
    bc_mask[:, 0, :] = 1
    bc_values[:, 0, :] = 1.0

    head, info = solve_multigrid_kcycle_7point_3d(
        tx_p=tx_p, tx_m=tx_m, ty_p=ty_p, ty_m=ty_m, tz_p=tz_p, tz_m=tz_m,
        rhs=rhs, active=active, bc_mask=bc_mask, bc_values=bc_values,
        storage_diag=storage_diag, max_cycles=1, nu_pre=1, nu_post=0, nu_coarse=0,
        smoother="vertical_line", line_omega=1.0, return_info=True
    )
    
    assert head.shape == shape
    
if __name__ == "__main__":
    test_column_only()
    test_small_3d()
    print("Tests passed!")
