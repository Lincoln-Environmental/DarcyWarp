import numpy as np

class DarcyMultigridKCycle:
    def __init__(self, n_levels):
        """
        :param n_levels: number of MG levels, 0 is finest, n_levels-1 is coarsest
        """
        self.n_levels = n_levels
        self.coarse_krylov_method = "fgmres"
        self.breakdown_detected = False
        self.fallback_used = False
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

        n_unknowns = int(max(b0.size, 1))
        b_l2_norm = float(np.linalg.norm(b0))
        b_rms = b_l2_norm / np.sqrt(float(n_unknowns))
        if b_rms == 0.0:
            b_norm = 1.0
        else:
            b_norm = b_rms

        self.breakdown_detected = False
        self.fallback_used = False
        residual_history = []

        x = x0
        n_cycles_applied = 0
        converged = False
        for cycle in range(max_cycles):
            r = b0 - self._apply_A_level(level, x)
            flow_l2 = float(np.linalg.norm(r))
            flow_rms = flow_l2 / np.sqrt(float(n_unknowns))
            residual_history.append(
                {
                    "cycle": int(cycle),
                    "flow_l2_norm": flow_l2,
                    "flow_residual_rms": flow_rms,
                    "head_residual_rms": flow_rms,
                    "relative_residual": flow_rms / b_norm,
                }
            )

            if flow_rms <= max(rel_tol * b_norm, abs_tol_min):
                converged = True
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
            n_cycles_applied += 1

        r_final = b0 - self._apply_A_level(level, x)
        final_l2 = float(np.linalg.norm(r_final))
        final_rms = final_l2 / np.sqrt(float(n_unknowns))
        if not residual_history:
            initial_l2 = final_l2
            initial_rms = final_rms
        else:
            initial_l2 = float(residual_history[0]["flow_l2_norm"])
            initial_rms = float(residual_history[0]["flow_residual_rms"])
        converged = bool(converged or final_rms <= max(rel_tol * b_norm, abs_tol_min))

        head_fine = x.reshape(ny, nx)

        if not return_info:
            return head_fine

        info = {
            "converged": converged,
            "n_cycles_applied": int(n_cycles_applied),
            "n_cycles": int(n_cycles_applied),
            "initial_flow_l2_norm": initial_l2,
            "final_flow_l2_norm": final_l2,
            "initial_flow_residual_rms": initial_rms,
            "final_flow_residual_rms": final_rms,
            "initial_head_residual_rms": initial_rms,
            "final_head_residual_rms": final_rms,
            "relative_residual": final_rms / b_norm,
            "residual_history": residual_history,
            "res_hist": np.asarray([row["flow_residual_rms"] for row in residual_history], dtype=float),
            "breakdown_detected": bool(self.breakdown_detected),
            "fallback_used": bool(self.fallback_used),
            "coarse_krylov_method": self.coarse_krylov_method,
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

        # FGMRES permits the recursive K-cycle/V-cycle preconditioner to vary.
        e_c = self._coarse_fgmres_with_mg(
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

    def _coarse_fgmres_with_mg(
        self,
        level: int,
        rhs: np.ndarray,
        n_krylov: int,
        nu_pre: int,
        nu_post: int,
        omega: float,
    ) -> np.ndarray:
        """
        Solve A_level e = rhs on a coarse level using restart-free FGMRES.

        The multigrid preconditioner may be variable because recursive K-cycle
        coefficients depend on the current residual, so ordinary PCG is not a
        valid wrapper here.
        """
        self.coarse_krylov_method = "fgmres"
        n = rhs.shape[0]
        if n == 0:
            return np.zeros_like(rhs)
        max_it = int(max(n_krylov, 0))
        if max_it == 0:
            return np.zeros_like(rhs)

        beta = float(np.linalg.norm(rhs))
        if (not np.isfinite(beta)) or beta <= 0.0:
            if not np.isfinite(beta):
                self.breakdown_detected = True
            return np.zeros(n, dtype=rhs.dtype)

        v0 = rhs / beta
        if not np.all(np.isfinite(v0)):
            self.breakdown_detected = True
            self.fallback_used = True
            return self._safe_smoothing_fallback(
                level=level,
                rhs=rhs,
                nu_pre=nu_pre,
                nu_post=nu_post,
                omega=omega,
            )

        V = [v0]
        Z: list[np.ndarray] = []
        H = np.zeros((max_it + 1, max_it), dtype=float)
        eps = np.finfo(float).eps
        k_done = 0

        for j in range(max_it):
            z = self._apply_mg_preconditioner(
                level=level,
                rhs=V[j],
                nu_pre=nu_pre,
                nu_post=nu_post,
                omega=omega,
            )
            if not np.all(np.isfinite(z)):
                self.breakdown_detected = True
                self.fallback_used = True
                return self._safe_smoothing_fallback(
                    level=level,
                    rhs=rhs,
                    nu_pre=nu_pre,
                    nu_post=nu_post,
                    omega=omega,
                )
            Z.append(z)
            w = self._apply_A_level(level, z)
            if not np.all(np.isfinite(w)):
                self.breakdown_detected = True
                self.fallback_used = True
                return self._safe_smoothing_fallback(
                    level=level,
                    rhs=rhs,
                    nu_pre=nu_pre,
                    nu_post=nu_post,
                    omega=omega,
                )

            for i in range(j + 1):
                H[i, j] = float(np.dot(w, V[i]))
                w = w - H[i, j] * V[i]
            h_next = float(np.linalg.norm(w))
            scale = max(float(np.linalg.norm(z)) * float(np.linalg.norm(w)), 1.0)
            H[j + 1, j] = h_next
            k_done = j + 1
            if h_next <= eps * scale:
                break
            V.append(w / h_next)

        if k_done == 0:
            return np.zeros(n, dtype=rhs.dtype)
        rhs_ls = np.zeros(k_done + 1, dtype=float)
        rhs_ls[0] = beta
        try:
            y, *_ = np.linalg.lstsq(H[: k_done + 1, :k_done], rhs_ls, rcond=None)
        except np.linalg.LinAlgError:
            self.breakdown_detected = True
            self.fallback_used = True
            return self._safe_smoothing_fallback(
                level=level,
                rhs=rhs,
                nu_pre=nu_pre,
                nu_post=nu_post,
                omega=omega,
            )
        if not np.all(np.isfinite(y)):
            self.breakdown_detected = True
            self.fallback_used = True
            return self._safe_smoothing_fallback(
                level=level,
                rhs=rhs,
                nu_pre=nu_pre,
                nu_post=nu_post,
                omega=omega,
            )
        e = np.zeros(n, dtype=rhs.dtype)
        for i in range(k_done):
            e = e + y[i] * Z[i]
        if not np.all(np.isfinite(e)):
            self.breakdown_detected = True
            self.fallback_used = True
            return self._safe_smoothing_fallback(
                level=level,
                rhs=rhs,
                nu_pre=nu_pre,
                nu_post=nu_post,
                omega=omega,
            )
        return e

    def _safe_smoothing_fallback(
        self,
        level: int,
        rhs: np.ndarray,
        nu_pre: int,
        nu_post: int,
        omega: float,
    ) -> np.ndarray:
        z = np.zeros_like(rhs)
        return self._jacobi_smooth_level(
            level=level,
            x=z,
            b=rhs,
            n_sweeps=max(1, int(nu_pre) + int(nu_post)),
            omega=omega,
        )

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
