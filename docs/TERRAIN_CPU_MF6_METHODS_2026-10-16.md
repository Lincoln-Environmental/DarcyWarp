# Independent terrain CPU and MF6 validation methods

This programme tests discrete equations and physical outputs separately. A converged solver is not a physical reference by definition. The production terrain formulation and its acceptance gate are unchanged. Newton remains experimental and opt-in.

## Canonical inputs and execution

`DARCY_WARP_PACKAGE.validation.terrain_case.TerrainCase` serializes the **exact** original `shanz_terrain_case_study.build_case` arrays, without running that script's demonstration or using its warm starts. The seven original cases use fixed dx=dy=20 m. Their different sizes change domain extent, not spatial resolution on a fixed domain.

Inputs are FP64 total head in metres, Cartesian Kx/Ky/Kz in m/day, volumetric cell source in m³/day, and interface elevations in metres. Layers run top to bottom. Arrays use C-order `(k,j,i)`, with j increasing northward and i eastward. Cell centres are `((i+.5)dx,(j+.5)dy,(top+bottom)/2)`. MF6 DIS rows run in the opposite y direction. Index conversion reverses j exactly; zero-based Python tuples become one-based MF6 input through FloPy. Tests cover every index in a nonsquare three-dimensional grid.

The manifest retains original source on fixed cells and states its reservoir semantics; free-cell source, fixed-cell source, volumes, positive thickness, active masks, contact ownership and connectivity are checked. No source is redistributed. Physical inputs and numerical controls have separate records. SHA-256 includes array names, types, shapes and bytes, physical settings and individual contacts. Loading refuses changed inputs. The CPU solver can load and solve a serialized case with all Warp imports explicitly prohibited.

Fresh runs use `execute_terrain_validation_programme.py`, with direct configuration in its execution guard. Immutable requests include input hashes, relevant implementation-file hashes, numerical settings, Python/package versions, precision and BLAS threading; MF6 requests also include the executable checksum. A stale, unversioned or interrupted directory is refused. Change the configured output root for a changed implementation/runtime. Historical developmental runs predate this registry and are identified by their command/revision logs; they are evidence, not eligible cache hits. Final cold CPU references repeat from committed numerical source with complete requests. Offline analysis never advances a saved solution.

## Independent CPU equations and Jacobian

The existing independent NumPy/SciPy `TerrainReferenceOperator3D` is retained. It assembles flux-gradient and incidence matrices independently of the GPU packed stencil. The new CPU nonlinear controller imports no Warp kernels, GPU Jv, GPU derivative arrays, or GPU nonlinear controller.

For outward-positive face/contact flow, the free-cell physical residual is

```
R(h) = original_source - divergence(q(h)) - contacts(h).
J = D(divergence + contacts) = -J_R;   J delta = R.
```

Fixed heads are imposed before iteration and their correction unknowns are eliminated. Dry free cells remain unknowns. A steady component needs a genuine head constraint or compatible positive-source discharge branch. Balanced discharge-only components have an undetermined datum; negative net source cannot be supplied by an outflow-only boundary. These are rejected explicitly. A closed positive-source discharge branch can receive an explicitly reported initial numerical lift; no equation or artificial storage is added. None of the original seven cases needs this lift.

At every accepted or trial head, the CPU reconstructs saturated fractions, all 13 face coefficients, metric terms, contact conductances/cross terms, gates, physical fluxes and the principal diagonal. Every internal face has opposite incidence in its two cells. Storage is exactly zero.

Let `L,U` select a face's two endpoint cells and `B=L-U`. With fixed gradient matrices G, the frozen flux row contains both the head jump and the actual terrain gradient/metric terms. The full sparse derivative is its frozen row **plus** the endpoint coefficient derivatives multiplied by the accepted head gradients/jumps. Applying `B.T` scatters it conservatively. This retains the complete multipoint sparsity; it is not the seven-point preconditioner stencil.

For half-cell conductances `t_l=a_l/d_l`, `t_r=a_r/d_r`, series C and resistance weights wl,wr:

```
C = 1/(d_l/a_l + d_r/a_r)
dC/dh_l = wl² (da_l/dh_l)/d_l
dC/dh_r = wr² (da_r/dh_r)/d_r
dwl/dh_l = -wl*wr*(da_l/dh_l)/a_l
dwl/dh_r = +wl*wr*(da_r/dh_r)/a_r; dwr=-dwl.
```

Horizontal `a=K*f*T`. Vertical `a=Kx*f*sx²+Ky*f*sy²+Kz` uses the **shared interface** slope, not a simplified transmissivity. Product differentiation includes every tangential coefficient and metric-defect coefficient involving the difference between centre and interface slopes. The frozen gradient matrix supplies C times the direction gradient; the extra endpoint rows supply DC times the accepted gradient. Kz stays saturation-independent.

The saturation clamp derivative is 1/T strictly between the effective floor and top, and zero on constant branches and at exact thresholds. A constant floored coefficient still transmits the head-gradient derivative. At raw discharge flux zero the selected contact derivative is the closed branch. No smoothing is used. Natural lateral contact coefficient derivatives include its entire reconstructed trace; overridden lateral coefficients are constant. Natural and overridden top/bottom contacts are differentiated directly from normal conductivity, projected area, metric defect and override/normal scaling, rather than importing GPU endpoints. Tests use central differences only in smooth regimes and the appropriate one-sided difference at kinks.

CPU linear systems use SuperLU sparse direct factorization. Histories record achieved linear residual and normwise backward error. Singular-factorization warnings become explicit failed trials; nonfinite corrections are not accepted. No dense full-domain Jacobian or residual-per-column construction is used. Condition diagnostics estimate the one-norm condition of the principal-row-scaled constrained Jacobian and its inverse using direct solves; these are estimates, not exact singular values or uniqueness proofs.

CPU Newton uses physical-residual merit, direction-preserving damping and a 50 m global maximum step. Weak Newton steps can fall back to independently computed frozen/direct Picard directions; Newton is retried after physical RMS contracts. Picard uses 0.7 initial relaxation and componentwise 50 m clipping. The search recomputes the complete nonlinear state, prefers merit descent and can allow bounded 1.10 norm recovery. These choices are explicitly recorded reference controls, not changes to production GPU globalization. Pure Newton and earlier reference policies also ran and failed; their failure labels are retained. Reference caps are 6,000, separately from the GPU 2,000 cap. The extended n96 reference additionally reports generic stagnation when two 100-iteration windows show less than 0.5% improvement in their best physical RMS.

Every returned head is independently re-evaluated with the original gate:

```
finite state AND head-equivalent RMS <= 1e-6 m
AND (physical cell RMS <= 1e-12 m3/day OR cell residual L1/gross throughput <= 1e-8).
```

The head scale is the seven-point principal diagonal plus open contact diagonals. Throughput accumulates absolute source and separate exterior inflow/outflow before cancellation, with the existing float64.tiny denominator treatment. A small head update, signed domain balance, linear residual, or optimizer flag cannot establish nonlinear success. Tighter polishing has distinct controls and retains any failed stricter stopping label.

## MF6 version and representations

The executable is the existing `/bin/modflow/mf6`, version 6.6.2 (12 May 2025), SHA-256 `21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176`. It was not built here; its exact binary build source is unavailable. Official source tag 6.6.2 resolves to `7cb7958940186ecfdcd3ea9e6a563cc4609581c9`. Relevant unmodified official source files are archived with the evidence.

The [6.6.2 NPF input documentation](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-npf.html), [DIS documentation](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-dis.html), [DRN documentation](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-drn.html) and [WEL documentation](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/gwf-wel.html) were consulted. Source confirmation uses [gwf-npf](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/GroundWaterFlow/gwf-npf.f90), [conductance utilities](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/ModelUtilities/GwfConductanceUtils.f90), [XT3D](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/ModelUtilities/Xt3dInterface.f90) and [DIS connection vectors](https://github.com/MODFLOW-ORG/modflow6/blob/6.6.2/src/Model/Discretization/Dis.f90). Documentation, source-derived statements and measured results are distinguished below.

| Item | DarcyWarp | Executed MF6 representations | Consequence |
|---|---|---|---|
| Geometry | Nonparallel terrain mapping, shared-interface metric-defect fluxes | DIS with identical top/bottom samples, plan dimensions, index mapping and geometric midpoint elevations; TPFA or XT3D | Volumes match the supplied midpoint thickness representation; shared-face and connection equations differ. Supplying TOP/BOT alone is not equivalence. |
| Tensor frame | Physical Cartesian diagonal K | Same Cartesian diagonal K; no geological rotation or transformed-tensor double application | ANGLE1/2/3 explicitly zero in the closest runs. |
| Horizontal connection geometry | Fixed terrain-centre gradient/metric reconstruction | Source: XT3D uses DIS connection vectors; omitting ANGLE2 enables NOZEE. Explicit zero retains z. With partial saturation, horizontal endpoints use bottom+.5*sat*T; vertical endpoints use full-thickness midpoints | A head-dependent connection-geometry difference remains; this is not an exact physical head mapping. The omission experiment measures its effect with unchanged K. |
| Horizontal saturation | Harmonic endpoint K*clip(h-bottom,.01,T) | Newton upstream weighting and quadratic saturation, epsilon=1e-6; XT3D's wetted face-area construction also differs | MF6 cannot impose the original 0.01 m floor through these native options. |
| Vertical flux | Saturation-independent physical Kz plus head-dependent transformed slope/metric terms | TPFA default full-thickness series Kz; XT3D uses full map area for vertical faces and its own geometric reconstruction | Flat vertical series case matches; sloping operators are not identical. |
| Sources | Original per-cell volumetric Q | WEL SOURCE package, same intended cell and sign; no automatic reduction or RCH allocation | Full FP64 text precision is required. Realized budget allocations are checked individually. |
| Fixed heads | Original coastal cell unknowns constrained to prescribed h | Same cell-centre CHD data after exact row reversal | These are the original prescribed cells, not an invented shoreline-face head. |
| Outlets | Individual saturation-dependent terrain trace, centroid/gradient terms, max(raw q,0) | Native DRN control, or separately labelled custom exact-algebra contact feedback | Native DRN is not the original contact. |
| Time | Steady, zero storage | One steady solve, no STO, no transient warmup | Feedback repeats steady solves; it does not add storage. |

Source clarification: although general NPF documentation mentions initial heads for default vertical conductance, `calc_initial_sat` and `calc_condsat` in this version use full thickness when THICKSTRT flags are absent. THICKSTRT is off, ICELLTYPE is nonnegative, VARIABLECV/PERCHED/REWET are off. The initial-guess experiments therefore do not deliberately change the physical operator. The close final-head comparisons are measured evidence for the two tested 64-cell initial guesses, not a uniqueness proof.

Native DRN rejects the original 2 m stage below the topsoil cell bottom (5.7 m) when that cell is convertible; `gwf-drn.f90` checks this explicitly. The rejected original deck is retained. A separately labelled native control makes those three owning cells confined and assigns independently derived full-natural C. That changes the formulation and is not presented as the original case. Neither elevation nor conductance is fitted to GPU solutions.

The custom-coupled experiment keeps all cells convertible and uses a separate CONTACT WEL package with `-q_contact`. Each outer recomputes the original terrain contact algebra independently at MF6's indexed heads, under-relaxes those three fluxes by 0.5, and reruns steady MF6. Its tolerance is 1e-8 m³/day per contact plus native success and independent conservation. Source Q never changes. This establishes the stated **custom algebraic coupling**, not exact equivalence of native physical contact geometry: MF6's horizontal saturated-centroid reconstruction remains different. Native and custom results are never pooled.

IMS settings are recorded in every deck: BICGSTAB, COMPLEX, outer/inner head-change tolerances 1e-9 m, STRICT linear residual close 1e-7 m³/day, 2,000 inner iterations, DBD relaxation and 3,000 or 4,000 native outer caps as labelled. A separate n32 attempt changes globalization to SIMPLE gamma=.5 with cap 8,000. MF6 change/residual tolerances are not DarcyWarp's final physical gate. Native success requires normal termination and exit code zero; conservation, source allocation and finite nonsentinel heads are independently required.

The [versioned IMS specification](https://modflow6.readthedocs.io/en/6.6.2/_mf6io/sln-ims.html) defines DVCLOSE using maximum absolute nodal change, not RMS. STRICT RCLOSE uses an infinity norm and requires both head and flow criteria at the first inner iteration. COMPLEX supplies DBD theta=.8, kappa=.0001, gamma=momentum=0; backtracking count=20, tolerance=1.05, reduction=.1, residual limit=.002; ILUT levels=5, drop=.0001, orthogonalization interval=2, scaling/reordering=NONE. The deck explicitly overrides linear relaxation to .97. These iteration settings and their resolved native listing records are preserved; they do not redefine the independent physical gate.

Binary extraction uses verified GRB IA/JA connectivity. FLOW-JA-FACE off-diagonals are flow **into the row cell**; diagonal residual slots are never counted as physical flows. Pairs are reciprocal. Package node numbers are one-based; WEL source/contact identity comes from destination `paknam2`, not `paknam`. Each internal face is reconstructed once. Sources, CHD and individual outlet exchanges remain separate until gross accumulation. DATA-SAT and raw binary files are retained, including sentinel values.

`mf6_tpfa_reference.py` independently evaluates the exact supported Newton TPFA upstream/saturation/full-Kz equations at a supplied head, with no MF6 execution or update. Its fluxes match reported final native flows to approximately 1e-9 m³/day, consistent with the native iterative tolerance; independently recomputed relative residuals are around 1e-10. This avoids mistaking the last frozen linear residual for nonlinear convergence. An equivalent independent fixed-head XT3D nonlinear residual evaluator was not built; its final reported fluxes and physical cell balances are checked, with that limitation explicit.

## Comparisons, geometry controls and remaining limits

Same-equation CPU/GPU checks retain existing strict FP64 tolerances; new smooth Jv checks require relative error below 2e-7 before roundoff. A 1e-4 m head difference is a predeclared investigation flag, not a model-selection rule. Final physical tolerances are unchanged. MF6 conservation uses physical RMS 1e-12 OR L1/gross 1e-8, together with exact source allocation and native convergence. No convenient percentage was selected after seeing the results.

Head comparisons exclude fixed cells and separate wet-both, dry-both, wet/dry mismatch and thinly saturated subsets. Water-table diagnostics use the existing first resolvable partially saturated layer, distinguish dry/potentiometric/artesian columns, and are not maximum-head plots or independent vadose-zone validation. A layer-selection change can make that diagnostic jump despite a smaller cell-head difference.

MF6 faces and terrain faces are not directly interchangeable. Cross-formulation tables compare integrated transfer between corresponding geological units and complete common Cartesian map planes, plus individual contacts and coastal exchange. Geological-interface-normal transfer is not labelled Cartesian vertical discharge. Saved native and terrain face arrays support future finer physical mapping; they are not declared pointwise flux parity.

The original fixed-dx ladder is not refinement. Separate fixed-domain tests use the same 320 m square and analytic nonparallel geology while reducing dx. An initial fixed-cell-centre control has moving numerical boundary locations and is retained only as a control. The final refinement uses physical no-flow lateral planes and top/bottom boundary surfaces at fixed physical locations, with unchanged integrated projected boundary area. CPU/GPU use natural terrain contacts; MF6 uses independently derived physical Kz projected-area GHB on its stepped surface. For h=z, the analytical target is zero lateral flux and downward Kz*map area. This isolates geometry consistency and boundary reconstruction; it does not certify full-case unconfined discretisation convergence.

Same-head geometry/floor term removal and upstream/contact comparisons quantify local contributions and their interactions. They are not full solutions at a different floor. The small wet/dry/contact ladder includes a separately labelled altered-floor full solve. Unresolved full-case geometry/saturation/centroid interactions, physical branch selection and uniqueness require further work; the validation report gives exact next experiments. No performance speedup, MF6 truth status, independent physical-model validation, or promotion of Newton follows from these tests.

## Additional discriminating findings

A separately declared flat control isolates a structural mechanism. One free upper cell has bottom=0, top=4 m, a horizontal fixed neighbour at H=2 m, a lower fixed neighbour at -1 m, source=0.8 m³/day, Kx=1 and Kz=4 m/day on a unit plan grid. The unchanged vertical series conductance is 1 m²/day. For .01<h<4, the horizontal harmonic conductance is 4h/(h+2), giving

```
4h/(h+2)*(h-2) + (h+1) = .8
5h² - 5.8h + .4 = 0.
```

Both wet roots, 0.0736404439531 and 1.086359556047 m, are above the floor. The constant-floor branch adds h=-0.157073170732 m. Direct independent CPU and GPU Newton solves from perturbed analytical seeds reproduce all three. The constrained Jacobian changes sign (+1.0199,-2.4419,+1.6406). This proves nonuniqueness for this **control**, and shows why neither large effort budgets nor accurate linear solves nor smoothing only the floor kink can establish a unique answer in general. It does not prove nonuniqueness or physical stability of every full case. MF6 TPFA's upstream law instead gives 3h-3=.8, or h=1.266666666667 m; its measured solution agrees. No conductance was fitted to that answer.

The nonzero fully saturated affine control also qualifies earlier accuracy wording. With varying thickness, the existing horizontal formula uses harmonic half-cell thickness, while the physical midpoint face area from linearly reconstructed interfaces uses arithmetic thickness. These coincide for constant thickness. The old affine tests verify the **harmonic discrete formula**, plus exact vertical metric cancellation; they do not prove exact physical flux through every variable-thickness face. A common-face test shows the normalised area discrepancy quarters under refinement. For a uniform-K affine field with both horizontal gradient components, the CPU/GPU discrete head error decreases from 1.744e-6 to 5.092e-7 to 1.280e-7 m on the control ladder; the analytical physical-face balance is about 1e-16 m³/day. This is discretisation error, not failure to solve the discrete equations. No conductance arithmetic is changed here. The separate fixed-physical-boundary h=z study remains the strict boundary-preserving refinement experiment.

The n32 XT3D connection-elevation control changes only explicit ANGLE2/3 zero to omission, leaving the physical tensor unrotated. Native MF6 then converges and the independently recomputed original contact fluxes remain exactly zero. This supports a geometry/nonlinear interaction in the closest model's failure, not a repair of the intended physical model. Source inspection suggests testing the full derivative of its head-dependent connection geometry; that derivative has not been independently verified and is an open hypothesis, not a confirmed MF6 implementation defect.

CPU usage without CUDA:

```python
from pathlib import Path
from DARCY_WARP_PACKAGE.validation.terrain_case import TerrainCase
from DARCY_WARP_PACKAGE.validation.terrain_cpu import CPUControls, solve_cpu

case = TerrainCase.load(directory=Path("path/to/canonical/case"))
head, result = solve_cpu(case=case, controls=CPUControls(method="newton"))
```

The package API is reusable logic. Standalone experiment configuration is in each script's exact execution guard; `execute_terrain_validation_programme.py` is the fresh, cache-checked campaign entry point. Native heads, DATA-SAT, fluxes, free-cell reconstructions and masks are independently extracted into `native_final_fields` and `analysis`. Missing/mismatched manifests, corrupted run outputs and native failure flags cannot become passing cache hits.
