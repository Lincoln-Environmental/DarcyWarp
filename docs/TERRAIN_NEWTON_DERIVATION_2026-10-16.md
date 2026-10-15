# Discrete terrain Newton derivative

The new `terrain_nonlinear_method="newton"` path solves the existing steady unconfined terrain equations. It changes the nonlinear direction and its safeguards, without changing constitutive laws, mesh, forcing, boundary data or final acceptance. Picard remains the default and is available explicitly as `terrain_nonlinear_method="picard"`.

For each free cell, the implemented physical residual is

\[
R_i(h)=Q_i-\sum_f B_{if}q_f(h)-\sum_{t\in i}g_t(q_t(h)).
\]

`terrain_physical_residual_kernel` uses **source minus outward flux**. `B` is the conservative face incidence matrix; storage is zero on this path. A discharge contact uses `g(q)=max(q,0)`; an ordinary head contact uses `g(q)=q`. Newton solves `J_R delta=-R`. The implemented positive-divergence operator is `A=-J_R`, with RHS `R`, which is algebraically the same free-cell equation. Prescribed/isolated/inactive rows impose `delta=0` through identity constraint rows. All dry free cells remain unknowns.

Newton GMRES works in **unscaled physical residual units**, i.e. `S=I`. Its absolute L2 target is `0.05*physical_RMS*sqrt(N_free)` by default. Picard fallback retains its existing head-scaled forcing. Final head-equivalent residuals continue to divide the physical cell residual by the existing seven-point principal diagonal, including open contact conductances. The final gate remains

\[
R_{head,RMS}\le10^{-6}\ \mathrm{m}\quad\land\quad
(R_{physical,RMS}\le10^{-12}\ \mathrm{m^3/day}\quad\lor\quad
\|R\|_1/\mathrm{gross\ throughput}\le10^{-8}).
\]

The original tiny-throughput denominator treatment is unchanged. Neither update magnitude nor signed domain imbalance substitutes for these tests.

The saturation fraction in a layer of thickness `T` and bottom `b` is

\[
f(h)=\operatorname{clip}(h-b,\min(floor,T),T)/T.
\]

The selected generalized derivative is `Df[v]=v/T` strictly between the floor and top, and zero on both constant branches and at either exact threshold. Threshold comparisons use the actual represented `h-b`, without an epsilon band or smoothing. At discharge `q=0`, the selected derivative is the closed branch, zero. These are admissible active-set selections, not ordinary smooth derivatives across a kink. Tests use one-sided perturbations into the selected branches; smooth-state tests use central differences over several step sizes. An exactly representable threshold is used to distinguish a mathematical kink from subtraction roundoff.

For every terrain face, define half-cell normal conductances `t_l=a_l/d_l`, `t_r=a_r/d_r`, series conductance `C=t_l*t_r/(t_l+t_r)`, and weights `w_l=t_r/(t_l+t_r)`, `w_r=t_l/(t_l+t_r)`. The derivative uses

\[
DC=w_l^2Dt_l+w_r^2Dt_r,\qquad
Dw_l=(w_rDt_r-w_lDt_l)/(t_l+t_r),\qquad Dw_r=-Dw_l.
\]

`harmonic_direction` evaluates these expressions in scaled arithmetic using `max(t_l,t_r)`; it does not differentiate the artificial min/max ordering used to evaluate a smooth harmonic function. Both zero materials give zero derivative; exactly-zero immutable K cannot become positive through a head perturbation.

On horizontal x faces, `a_l=Kx_l*f_l*T_l`, `a_r=Kx_r*f_r*T_r`, and `d_l=d_r=dx/2`. The three packed coefficients are `C*dy`, `w_l*a_l*zx_face*dy`, and `w_r*a_r*zx_face*dy`. Their derivatives include endpoint fraction derivatives and harmonic-weight derivatives. Y faces use the corresponding `Ky`, `dy/2`, `zy_face`, and `dx` factors. Fixed geometry has zero derivative.

On vertical faces, the normal transformed coefficients are

\[
a_l=Kx_l f_l z_{x,if}^2+Ky_l f_l z_{y,if}^2+Kz_l,
\]

and similarly for the lower cell, with `d_l=T_l/2`, `d_r=T_r/2`. `Kz` is saturation-independent, but the transformed normal coefficient and resistance weights still depend on horizontal saturated fractions and interface slopes. Every packed vertical coefficient is differentiated: the series term, four tangential-gradient weights, and the two metric-defect terms involving `(zx_centre-zx_interface)` and `(zy_centre-zy_interface)`. The reference implementation is the actual 13-column variable-interface operator, not a simpler seven-point transmissivity model.

If `G` denotes the fixed-grid gradient operations, each derivative flux is assembled as

\[
Dq[v]=C(h)Gv+DC(h)[v]Gh,
\]

with the appropriate multiple packed metric coefficients. A constant floored coefficient has zero coefficient derivative and still contributes `C Gv`. Direction vectors are projected to zero on constraint cells before **both** coefficient and gradient differentiation; their original accepted total heads remain in `Gh`. Thus fixed-head face exchange is linearized correctly.

Contact rows are the exact existing device refresh law. With prepacked dry/wet endpoints,

\[
c(f)=c_0+f(c_1-c_0),\qquad b(f)=b_0+w(f)(b_1-b_0).
\]

Natural contacts use `w=f`. For the existing supported overridden top/bottom metric rows, `n=(1-f)n_0+fn_1`, `w=f*n_1/n`, and `Dw=(n_0*n_1/n^2)Df`. `Dc=(c_1-c_0)Df`. Differentiate the entire reconstructed contact flux `c*(h-hb)-bz*hz-bx*hx-by*hy`, then apply the accepted gate derivative. This includes the coefficient times gradient derivative and coefficient derivative times accepted gradient. The low-K benchmark uses only natural lateral contacts; no override is involved there.

Each internal derivative face is evaluated once and gathered with opposite signs into its two cells. This preserves the original discrete conservation structure. Independent tests compare this device derivative against finite differences of the separately assembled NumPy/SciPy physical flux residual.

The existing principal-part multigrid V-cycle is a **fixed linear right preconditioner**, not a claim that the full Jacobian is SPD. Its original interface approximates `(D^-1 A_P)^-1=A_P^-1 D`, where `D` is the accepted principal diagonal. Newton presents the physical residual through `D^-1` to that interface, obtaining `A_P^-1` for the unscaled system. `D`, all Picard coefficients, contact gates and hierarchy values remain fixed throughout each ordinary GMRES solve. There is no differentiation of a head-dependent scaling and no variable preconditioner inside GMRES.

`TerrainJacobian3D` owns private direction, gradient, coefficient-derivative, flux and contact scratch buffers. `prepare()` saves accepted head gradients once per attempted Newton solve. Jv performs no nonlinear refresh and never writes accepted head, residual, RHS, saturation, coefficients, gates or hierarchy. GMRES workspaces and captured restart graphs persist across outer attempts within a solve. The graph's device counter counts executed Jv calls, including linear residual checks; Python capture activity is not mistaken for executed work.

Globalization first tries a Newton relaxation of one, then the existing halving search, 50 m clipping, dual merit and bounded recovery. Newton is considered adequate for continued use when its linear target is reached, accepted relaxation is at least 0.5 and accepted merit is at most 0.25 of current merit. Otherwise Picard is tried from the same original accepted head with its original 0.7 starting relaxation and unchanged safeguards. If Picard also fails, an already acceptable Newton candidate is retained. A weak attempt postpones further Newton attempts until physical RMS halves from that attempt's starting value. This retry policy uses solver/acceptance results and residual progress; it has no grid identity or fixed iteration switch.

Each outer has one shared Krylov budget. Newton may use at most 75%, while reserving at least a restart when possible. Picard receives exactly the remaining budget, including unused Newton work; retry work inside each adaptive solve is charged to the same allocation. Shortened restart blocks count actual iterations. Zero budget and unusable solves terminate honestly. Candidate evaluations, Jv calls, backtracks, attempted/accepted Newton steps and Picard fallback outers are reported separately. Selected history rows identify whether an inner RMS has metres or m3/day units.

A saved acceptable Newton candidate and its correction are retained while Picard is tested. Restoration re-evaluates the entire accepted nonlinear operator, RHS, gates and every multigrid level. Rejected searches do the same. The driver also refreshes the canonical accepted fields before returning after an unusable solve. The final session acceptance is independently reconstructed on the host at the returned head. Closing the solve clears Newton graph callbacks before releasing solve-local device state.

This path currently supports steady unconfined terrain only. Transient and confined requests for Newton raise explicitly; the existing Picard paths remain available. Control defaults are in `TerrainNewtonControls`, optionally passed as `terrain_newton_controls=...`. Scientific validation and runtime results are in `TERRAIN_NEWTON_VALIDATION_2026-10-16.md`.
