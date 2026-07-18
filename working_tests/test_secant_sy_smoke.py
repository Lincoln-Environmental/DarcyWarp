import numpy as np
import warp as wp
import time
from DARCY_WARP_PACKAGE.warped_darcy import (
    secant_specific_storage_coeff,
    secant_specific_yield_coeff,
    specific_storage_potential,
    update_secant_sy_storage_kernel,
)

def main():
    wp.init()
    device = "cpu"

    np.random.seed(42)
    N = 10
    h_prev = np.zeros(N)
    h_iter = np.zeros(N)
    bottom = np.zeros(N)
    top = np.ones(N) * 10.0
    active = np.ones(N, dtype=np.int32)
    bc_mask = np.zeros(N, dtype=np.int32)

    # 0: below -> below (dh=1.0)
    h_prev[0] = -5.0
    h_iter[0] = -4.0

    # 1: inside -> inside
    h_prev[1] = 2.0
    h_iter[1] = 7.0

    # 2: above -> above
    h_prev[2] = 15.0
    h_iter[2] = 16.0

    # 3: crossing the bottom
    h_prev[3] = -2.0
    h_iter[3] = 2.0

    # 4: crossing the top
    h_prev[4] = 8.0
    h_iter[4] = 12.0

    # 5: falling across both bounds
    h_prev[5] = 12.0
    h_iter[5] = -2.0

    # 6: zero head change inside cell
    h_prev[6] = 5.0
    h_iter[6] = 5.0

    # 7: Dirichlet cell (moving, but should zero out)
    h_prev[7] = 5.0
    h_iter[7] = 6.0
    bc_mask[7] = 1

    # 8: Inactive cell (moving, but should zero out)
    h_prev[8] = 5.0
    h_iter[8] = 6.0
    active[8] = 0

    # 9: zero head change above top
    h_prev[9] = 15.0
    h_iter[9] = 15.0

    sy_f = 0.2
    ss_f = 1e-5
    min_sat_f = 0.1
    eps = 1e-12
    dx = 1.0
    dt = 1.0

    # Host calculation
    head_ref64 = h_iter
    head_old64 = h_prev
    zbot_arr = bottom
    ztop_arr = top
    full_thickness = np.maximum(ztop_arr - zbot_arr, 0.0)
    sat_ref_zero = np.clip(head_ref64 - zbot_arr, 0.0, full_thickness)
    sat_old_zero = np.clip(head_old64 - zbot_arr, 0.0, full_thickness)
    sy_coeff_host = secant_specific_yield_coeff(
        head_ref=head_ref64,
        head_old=head_old64,
        bottom=zbot_arr,
        top=ztop_arr,
        specific_yield=sy_f,
        secant_eps=eps,
    )
    ss_coeff_host = secant_specific_storage_coeff(
        head_ref=head_ref64,
        head_old=head_old64,
        bottom=zbot_arr,
        top=ztop_arr,
        specific_storage=ss_f,
        secant_eps=eps,
    )
    
    storage = sy_coeff_host + ss_coeff_host
    storage[active == 0] = 0.0
    storage[bc_mask != 0] = 0.0
    sy_coeff_host[active == 0] = 0.0
    sy_coeff_host[bc_mask != 0] = 0.0
    ss_coeff_host[active == 0] = 0.0
    ss_coeff_host[bc_mask != 0] = 0.0
    
    storage_diag_host = storage * (dx * dx / dt)

    # Device calculation
    h_iter_wp = wp.array(h_iter.reshape(1, N), dtype=wp.float64, device=device)
    h_prev_wp = wp.array(h_prev.reshape(1, N), dtype=wp.float64, device=device)
    bottom_wp = wp.array(bottom.reshape(1, N), dtype=wp.float64, device=device)
    top_wp = wp.array(top.reshape(1, N), dtype=wp.float64, device=device)
    active_wp = wp.array(active.reshape(1, N), dtype=wp.int32, device=device)
    bc_mask_wp = wp.array(bc_mask.reshape(1, N), dtype=wp.int32, device=device)
    
    storage_coeff_wp = wp.zeros((1, N), dtype=wp.float64, device=device)
    sy_coeff_wp = wp.zeros((1, N), dtype=wp.float64, device=device)
    ss_coeff_wp = wp.zeros((1, N), dtype=wp.float64, device=device)
    storage_diag_wp = wp.zeros((1, N), dtype=wp.float64, device=device)
    storage_diag_prev_wp = wp.zeros((1, N), dtype=wp.float64, device=device)
    storage_change_sum_sq_buf = wp.zeros(1, dtype=wp.float64, device=device)
    storage_change_max_buf = wp.zeros(1, dtype=wp.float64, device=device)

    wp.launch(
        kernel=update_secant_sy_storage_kernel,
        dim=(1, N),
        inputs=[
            h_iter_wp, h_prev_wp, bottom_wp, top_wp, active_wp, bc_mask_wp,
            float(sy_f), float(ss_f), float(dx), float(dt), float(min_sat_f), float(eps), N, 1,
            storage_coeff_wp, sy_coeff_wp, ss_coeff_wp, storage_diag_wp, storage_diag_prev_wp,
            storage_change_sum_sq_buf, storage_change_max_buf
        ],
        device=device
    )
    
    sy_coeff_device = sy_coeff_wp.numpy()[0]
    ss_coeff_device = ss_coeff_wp.numpy()[0]
    storage_diag_device = storage_diag_wp.numpy()[0]

    np.testing.assert_allclose(sy_coeff_device, sy_coeff_host, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(ss_coeff_device, ss_coeff_host, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(storage_diag_device, storage_diag_host, rtol=1e-5, atol=1e-6)

    dh_ref = head_ref64 - head_old64
    moving_free = (np.abs(dh_ref) > eps) & (active != 0) & (bc_mask == 0)
    phi_ref = specific_storage_potential(
        head=head_ref64,
        bottom=zbot_arr,
        top=ztop_arr,
        specific_storage=ss_f,
    )
    phi_old = specific_storage_potential(
        head=head_old64,
        bottom=zbot_arr,
        top=ztop_arr,
        specific_storage=ss_f,
    )
    np.testing.assert_allclose(
        sy_coeff_host[moving_free] * dh_ref[moving_free],
        sy_f * (sat_ref_zero[moving_free] - sat_old_zero[moving_free]),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ss_coeff_host[moving_free] * dh_ref[moving_free],
        phi_ref[moving_free] - phi_old[moving_free],
        rtol=1e-12,
        atol=1e-12,
    )
    print("All smoke tests passed!")

if __name__ == "__main__":
    main()
