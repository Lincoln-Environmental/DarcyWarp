# Selected unconfined NPF662 law — Cartesian contract v1 addendum

Stage1 gate is complete (local milestone and stage1/experiments/gate.json). Geometry, physical K, source ownership, CHD placement, units and inactive/no-flow semantics stay as in the confined contract. This addendum intentionally defines a new law; legacy harmonic-floor Cartesian and transformed terrain laws remain untouched.

Cell types are explicitly0(confined) or1(convertible), scalar or full-grid array. Confined saturation is1 with derivative0. Convertible saturation uses MF6 6.6.2 `sQuadraticSaturation` with epsilon=1e-6, thickness b=dz, t=clip((h-bottom)/b,0,1), a=1/(1-epsilon):

```
s(t) = a*t*t/(2*epsilon)                   t<epsilon
       a*t + (1-a)/2                      epsilon<=t<1-epsilon
       1-a*(1-t)**2/(2*epsilon)            1-epsilon<=t<1
       1                                  t>=1
```

Below bottom and above top the derivative is0; at bottom/top it is0. Interior derivatives with respect to head are a*t/(epsilon*b), a/b, and a*(1-t)/(epsilon*b). The source-defined joins are C1. There is **no positive saturation/conductance floor**.

For horizontal faces, calculate full-thickness harmonic C0 first. If either cell is convertible, C=C0*s_upstream; otherwise C=C0. Upstream means higher total head. At a tie, MF6's upper-triangle loop selects the larger **MF6 user node**, which is the right endpoint for canonical x+, but the left endpoint for canonical northward y+. Equal heads with differing cell types can be a genuine selected-derivative kink. The head-gradient term remains even when coefficient derivative is zero. Vertical C=C0, independent of head and saturation; PERCHED and VARIABLECV are both off, so use the full head difference, not a perched elevation.

For outward q=C(h)*(hi-hj),

```
dq/dhi = C + dC/dhi*(hi-hj)
dq/dhj = -C + dC/dhj*(hi-hj)
J = D(div q) = -D(R),       J*delta = R.
```

Only endpoints enter a face. Incidence assembly gives at most seven spatial columns per row before prescribed-head elimination. Free dry cells remain unknowns. Fixed-head corrections are zero, while their flux contributions and conductance derivatives remain. A constant full-saturated principal row scale S=1/P0 is used on device, consistently on J and R; S is independent of the current head. The nonsymmetric scaled Jacobian uses restarted GMRES and a fixed diagonal preconditioner within each solve; CG is not used for Newton. CPU verification has independently assembled sparse derivatives/direct solves.

Device Newton tries a full direction-preserving bounded step, evaluates full candidate physical fluxes/saturation, and searches dual head/flow merits. A bounded norm recovery may be recorded. Frozen-coefficient Picard fallback occurs after rejected or unusable Newton steps, with a **shared** per-outer Krylov budget. Trials have private head/flux/residual/scale buffers; they cannot overwrite accepted Jacobian/RHS. Work counters report actual Krylov, Jv, residual/candidate, backtracking and fallback work. The unchanged independent final gate is recomputed at the returned head, including prescribed heads.

An exact-dry source control demonstrated a zero Jacobian row with nonzero RHS: the initial CPU and GPU controllers failed while native MF6 recovered. The corrected controller diagnoses such impossible frozen rows before spending Krylov work. After Newton and actual frozen Picard cannot give an accepted direction, it may solve the **full-saturated C0 preconditioner** against the unchanged nonlinear residual as a labelled `full_saturated_safeguard`. This is a numerical direction only. Every trial still uses the original zero-floor upstream law; no storage, conductivity or physical saturation is changed. All attempts share the same remaining Krylov budget. The independent CPU reference implements its own sparse/direct version. Recovery from a dry start is distinguished from changing the initial physical inputs.

At a putative root, positive **current physical conductances**, not potential C0 alone, must ground each active component. A balanced detached dry component does not determine its head. Such fields retain their zero residual but fail independent datum acceptance, with a separate count and failure reason; dry unknowns are never silently removed.

MF6 configuration differs from confined only by ICELLTYPE and model NEWTON UNDER_RELAXATION, plus BICGSTAB for the nonsymmetric linear solve. Full-thickness harmonic averaging, epsilon1e-6, explicit unrotated K, XT3D/THICKSTRT/VARIABLECV/DEWATERED/PERCHED/REWET off; no storage, RCH or native gradient-reconstructed contacts. Only CHD, no-flow and unreduced WEL are supported. Other options, terrain geometry, arbitrary conductance overrides and a terrain saturation floor are errors. The exact flat top elevation must be supplied. No implicit legacy fallback is allowed.

For convertible cells, CHD head must be at or above the owning bottom (equality is allowed). Official6.6.2 `gwf-chd.chd_ck` enforces this. A real negative experiment reproduced native model rejection while an initially permissive CPU check accepted a balanced field; the common case validator now rejects the invalid definition before CPU/GPU/MF6 solves, without altering the supplied head. Confined CHD is not subject to this convertible restriction. Source and the rejected deck are preserved under stage2/invalid_chd_before_guard.

Native nonlinear head-change tolerances must also be representable. Small controls use1e-14m; the larger flat catchment explicitly uses `max(1e-14,32*eps*max(1,abs(active initial/prescribed heads)))`, 2.13e-12m for its inputs. The initial1e-14 run stalled at a2.84e-14m update and was interrupted, not passed. Native STRICT RCLOSE remains1e-14m³/day, all governing options and the independent acceptance/head/flux gates stay unchanged. This input-based iteration policy is recorded separately from physical options and does not change initial-head-dependent conductance (THICKSTRT remains off).

Source confirmation: official6.6.2 `GwfConductanceUtils.hcond/convertible_upstream/vcond`, `gwf-npf.npf_fn/calc_initial_sat/calc_condsat`, and `SmoothingFunctions.sQuadraticSaturation/Derivative` in the preserved source bundle. The previous TPFA evaluator agrees on shared flat, all-convertible stacks, but lacks mixed-cell/tie/mask scope; it is a comparison, not the new oracle.

The separately identified scalar control uses the original geometry and Q=.8. If h<=2m, upstream H=2 has exactly s=.5, so horizontal q=2(h-2), vertical q=h+1; R=0 gives h=3.8/3=1.266666666667m, admissible below2. For h>2, both outward terms sum above3m³/day, exceeding.8, so no additional root is admissible. This establishes uniqueness only for this control. The original harmonic-law three-root control and its roots are retained without editing.
