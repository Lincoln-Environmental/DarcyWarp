import numpy as np

class DarcyMultigridKCycle:
    def __init__(self, n_levels):
        """
        :param n_levels: number of MG levels, 0 is finest, n_levels-1 is coarsest
        """
        self.n_levels = n_levels
        # you will already have level specific data attached somewhere:
        # self.dx_levels, self.T_levels, self.active_levels, etc

    def solve_multigrid_kcycle(
        self,
        max_cycles: int = 20,
        nu_pre: int = 2,
        nu_post: int = 2,
        n_krylov_coarse: int = 2,
        omega: float = 0.8,
        rel_tol: float = 1.0e-4,
        abs_tol_min: float = 5.0e-7,
        initial_head: np.ndarray | None = None,
        return_info: bool = True,
    ):
        """
        Multilevel K cycle solver.

        :param max_cycles: maximum number of K cycles
        :param nu_pre: Jacobi pre smoothers on each level
        :param nu_post: Jacobi post smoothers on each level
        :param n_krylov_coarse: Krylov iterations on each coarse level
        :param omega: Jacobi relaxation factor
        :param rel_tol: relative RMS residual tolerance (fine level)
        :param abs_tol_min: minimum absolute RMS residual tolerance
        :param initial_head: optional initial head on fine grid (2D)
        :param return_info: whether to return info dict
        :return: head_fine, info
        """
        level = 0  # finest
        rhs_fine = self._build_rhs_level(level)  # you already have this

        ny, nx = rhs_fine.shape
        b0 = rhs_fine.reshape(ny * nx)
        if initial_head is None:
            x0 = np.zeros_like(b0)
        else:
            x0 = initial_head.reshape(ny * nx).copy()

        b_norm = float(np.linalg.norm(b0))
        if b_norm == 0.0:
            b_norm = 1.0

        res_hist = []

        x = x0
        for cycle in range(max_cycles):
            r = b0 - self._apply_A_level(level, x)
            res_norm = float(np.linalg.norm(r))
            res_hist.append(res_norm)

            if res_norm <= max(rel_tol * b_norm, abs_tol_min):
                break

            x = self._k_cycle_level(
                level=level,
                x=x,
                b=b0,
                nu_pre=nu_pre,
                nu_post=nu_post,
                n_krylov_coarse=n_krylov_coarse,
                omega=omega,
            )

        head_fine = x.reshape(ny, nx)

        if not return_info:
            return head_fine

        info = {
            "n_cycles": cycle + 1,
            "res_hist": np.array(res_hist),
        }
        return head_fine, info

    def _k_cycle_level(
        self,
        level: int,
        x: np.ndarray,
        b: np.ndarray,
        nu_pre: int,
        nu_post: int,
        n_krylov_coarse: int,
        omega: float,
    ) -> np.ndarray:
        """
        Single K cycle starting on 'level' with initial guess x and rhs b.
        Everything is flattened 1D on that level.
        """

        # pre smoothing
        if nu_pre > 0:
            x = self._jacobi_smooth_level(
                level=level,
                x=x,
                b=b,
                n_sweeps=nu_pre,
                omega=omega,
            )

        A_x = self._apply_A_level(level, x)
        r = b - A_x

        # coarsest level: no recursion, just extra smoothing
        if level == self.n_levels - 1:
            if nu_post > 0:
                x = self._jacobi_smooth_level(
                    level=level,
                    x=x,
                    b=b,
                    n_sweeps=nu_post,
                    omega=omega,
                )
            return x

        # restrict residual to coarse grid
        r_c = self._restrict_residual_level(level, r)

        # coarse grid correction via PCG with MG preconditioner (recursive)
        e_c = self._coarse_pcg_with_mg(
            level=level + 1,
            rhs=r_c,
            n_krylov=n_krylov_coarse,
            nu_pre=nu_pre,
            nu_post=nu_post,
            omega=omega,
        )

        # prolongate correction and update x
        corr_fine = self._prolong_correction_level(level, e_c)
        x = x + corr_fine

        # post smoothing
        if nu_post > 0:
            x = self._jacobi_smooth_level(
                level=level,
                x=x,
                b=b,
                n_sweeps=nu_post,
                omega=omega,
            )

        return x

    def _coarse_pcg_with_mg(
        self,
        level: int,
        rhs: np.ndarray,
        n_krylov: int,
        nu_pre: int,
        nu_post: int,
        omega: float,
    ) -> np.ndarray:
        """
        Solve A_level e = rhs on coarse level using n_krylov steps of PCG.
        Preconditioner is one K cycle on the *next* level (unless already
        at coarsest level).
        """
        n = rhs.shape[0]
        e = np.zeros(n, dtype=rhs.dtype)

        A_e = self._apply_A_level(level, e)
        res = rhs - A_e

        # preconditioned residual z = M^{-1} res
        z = self._apply_mg_preconditioner(
            level=level,
            rhs=res,
            nu_pre=nu_pre,
            nu_post=nu_post,
            omega=omega,
        )
        p = z.copy()
        rho_old = float(res @ z)

        for it in range(n_krylov):
            A_p = self._apply_A_level(level, p)
            denom = float(p @ A_p)
            if denom == 0.0:
                break
            alpha = rho_old / denom

            e = e + alpha * p
            res = res - alpha * A_p

            if it == n_krylov - 1:
                break

            z = self._apply_mg_preconditioner(
                level=level,
                rhs=res,
                nu_pre=nu_pre,
                nu_post=nu_post,
                omega=omega,
            )
            rho_new = float(res @ z)
            if rho_old == 0.0:
                break
            beta = rho_new / rho_old
            p = z + beta * p
            rho_old = rho_new

        return e

    def _apply_mg_preconditioner(
        self,
        level: int,
        rhs: np.ndarray,
        nu_pre: int,
        nu_post: int,
        omega: float,
    ) -> np.ndarray:
        """
        M^{-1} rhs. If not on coarsest level, this is a K cycle on level+1.
        On the coarsest level it is just a few Jacobi sweeps.
        """
        if level == self.n_levels - 1:
            # already coarsest: simple smoothing "solve"
            z = np.zeros_like(rhs)
            z = self._jacobi_smooth_level(
                level=level,
                x=z,
                b=rhs,
                n_sweeps=nu_pre + nu_post,
                omega=omega,
            )
            return z

        # one recursive K cycle starting at the next coarser level
        z0 = np.zeros_like(rhs)
        z = self._k_cycle_level(
            level=level,
            x=z0,
            b=rhs,
            nu_pre=nu_pre,
            nu_post=nu_post,
            n_krylov_coarse=1,  # preconditioner itself can be cheap
            omega=omega,
        )
        return z

