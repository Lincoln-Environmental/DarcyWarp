NPF 6.6.2: 250 x 250 x 5 local baseline
======================================

Scope and reproduction
----------------------

The self-contained Git baseline is
``baselines/npf662_250x250x5_rch150_dx100_v1/``. Each variant contains
``inputs.npz``, direct input arrays in ``inputs/``, accepted MF6 arrays in
``mf6/``, and exact native evidence in a lossless ``native.tar.gz`` archive.
Reports, complete numerical histories, measured source snapshots and a
SHA256 manifest are included. Verify the saved baseline without rerunning
either solver with ``python working_tests/archive_npf662_baseline.py``.

Source baseline: ``867b79e14dea83cfed7b4898264e366a1d05c971``. No governing
operator or final numerical acceptance criterion is changed. The public
``solver='kcycle'`` selector is required by the pinned API, but the actual
layered backend is **device Newton / diagonal-preconditioned GMRES**. The
separate seven-point K-cycle experiment is linear algebra on A0, and is not
an unconfined Newton speedup.

Run from the repository root::

  DARCY_FLOAT=float64 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    PYTHONPATH=. /home/patrickdurney/miniconda3/envs/darcywarp/bin/python \
    working_tests/benchmark_npf662_baseline.py

All controls are in the exact ``if __name__ == "__main__":`` block. Choose a
NEW output directory before repeating; existing evidence is never replaced.
``frozen_settings.json`` and exact ``inputs.npz`` are written before any
solve. ``source_snapshot/`` contains the source used, including the driver;
``manifest.json`` hashes evidence files after completion. Kernel cache files
are excluded from the evidence manifest. Input hashes are numerical hashes
of the canonical arrays, independent of ZIP container timestamps. The manifest
is tamper-evident local evidence; this is not write-once hardware storage.

The physical model is fixed independently of timings. Domain dimensions are
25 km by 25 km, with 100 m square cells (the user's final cell-size choice).
Model top rises from 8 m to 20 m,
with a 1 m wave. Layers are 24, 27, 30, 33, 36 m thick with independent
2 m sinusoidal variations and layer phase offsets. Every thickness is
strictly positive. The six interfaces are authoritative; there is no terrain
coordinate transformation and no slope cross term. These are conventional
DIS same-layer connections, not geometric overlap or geological-axis fluxes.

Seed 6622505 generates bounded log conductivity perturbations. Layer Kx
scales are 12, 8, 16, 6, 10 m/day, multiplied by exp(smooth wave + bounded
noise). Ky/Kx varies from 0.7 to 0.9; Kz/Kx from 0.125 to 0.175. These fields
are diagonal in physical x/y/z. The west plane (all five layers, x index 0)
is the ONLY prescribed-head boundary, at 0 m (fixed cell centres at x=50 m).
All other exterior faces are
no-flow. Native **RCH** recharge is 150 mm/year on a 365-day year
(150/1000/365 m/day), with a +/-20% smooth spatial modulation.
``IRCH=1`` (native one-based numbering) and ``FIXED_CELL`` explicitly
allocate it to the top layer, x indices 1..249 only. The CHD column has zero
recharge. The equivalent DarcyWarp volume source is recharge times the
10,000 m2 plan area. There is no WEL package in the final native deck and no
redistribution to other layers. Total source is 255,821.917808 m3/day in
62,250 recharge cells. The cold head is 0..2 m rising linearly inland, identical in
both variants and wet above every top-layer bottom. The confined variant has
ICELLTYPE=0 everywhere; the unconfined variant has ICELLTYPE=1 in layer 0
only, with layers 1..4 confined. Thus the source, materials, geometry and
boundary condition are shared while saturation physics is explicitly distinct.

Native MF6 uses DIS/NPF/IC/CHD/RCH/OC, Newton under-relaxation for the
convertible variant, no XT3D, no VARIABLECV, PERCHED, THICKSTRT or REWET,
and no auto-flow reduction. Horizontal conductance is the harmonic
full-thickness conductance multiplied by upstream MF6 saturation when either
endpoint is convertible. Vertical conductance remains the harmonic
full-thickness series resistance. MF6's existing quadratic endpoint smoothing
(epsilon 1e-6) is retained; no artificial conductance saturation floor is added.

Arrays and native evidence
--------------------------

Arrays use C order (layer top-to-bottom, y south-to-north, x west-to-east).
Cell arrays have shape (5,250,250); interfaces (6,250,250). Top and bottom
are the upper/lower elevation of EACH cell. ``centres.npy`` is centre z;
centre x/y are (index+0.5)*100 m. All physical inputs are FP64, masks and
cell types integers. Length is metres; time is days; K is m/day; source,
face flux, reservoir exchange and water residual are m3/day. Positive
source is injection. MF6 DIS row order is north-to-south, so native adapters
reverse y, including source node mapping. Native files are never reordered
or edited; only decoded arrays use canonical order.

``qx`` has shape (5,250,249), positive west-to-east. ``qy`` has shape
(5,249,250), positive south-to-north. ``qz`` has shape (4,250,250), positive
from upper to lower layer (downward, opposite physical elevation z).
Each internal face is stored exactly once. Incidence is +q outward at the
first endpoint and -q at the second endpoint. A positive CHD reservoir
exchange means external injection into the fixed cell. Including that
exchange, the cell residual is source + reservoir injection - outward
internal divergence. The free-cell residual does not include a fictitious
fixed-cell equation. Both arrays are saved.

``inputs.npz`` is the complete canonical model schema, loadable by
``load_npf662_case``; ``arrays/*.npy`` also exposes each input and derived
geometry directly. ``mf6/`` retains its exact input deck, executable SHA256,
version string, stdout, HEAD/CBC/GRB, model and simulation listings, outer
and inner CSVs, decoded ``result.npz`` and independent summary. Each successful
DarcyWarp run retains heads, saturation, three face flux arrays, cell budgets,
source allocation, boundary exchange, complete iteration/candidate histories,
per-run settings and timing/memory measurements. MF6 failures remain visible;
a process success flag never establishes independent acceptance by itself.

The pinned executable must report ``mf6: 6.6.2 05/12/2025`` and SHA256
``21efe1534afd1d9f19227938ff70cca4b2f216d326f7d73073b4d8ebe1ed7176``.
The final native reference uses COMPLEX/BICGSTAB with outer head tolerance
1e-6 m, inner head tolerance 1e-8 m and STRICT residual tolerance 1e-7.
It uses ILUT fill 8, drop tolerance 1e-4, relaxation factor zero, RCM ordering,
diagonal scaling, a maximum 500 outer and 2000 inner iterations and a 300 s
process cap. The accepted confined reference uses IMS DBD and zero periodic
residual replacements; unconfined uses IMS NONE and eight periodic residual
replacements. Exact per-variant settings are in the native summaries.
These are solver settings, not a
relaxation of independent physical acceptance. The full effective IMS
settings are saved; the shared native runner's old defaults remain intact.

The final 100 m/150 mm-year case reaches above-top heads in 59,054 of 62,500
columns. It is a convertible-top benchmark with most columns in the confined
regime. Its maximum head is about 82.904 m. Surface discharge/runoff is not
represented by the prescribed single-coast/no-flow model. These results are
reported without changing geometry, source strength or the equations to
force a different saturation regime.

Independent acceptance
----------------------

The unchanged NPF reference requires finite active heads, prescribed heads
within 1e-10 m, a grounded conducting topology, head-equivalent cell-residual
RMS <=1e-6 m, AND either physical residual RMS <=1e-12 m3/day OR
throughput-normalized cell-imbalance L1 <=1e-8. Throughput counts absolute
free sources plus gross free-to-CHD exchange before cancellation. It does
not permit cell errors to cancel in a single domain budget. Native decoding
also checks actual RCH allocation, reciprocal FLOW-JA-FACE entries, the
canonical edge set, native saturation, and native-flux cell conservation.
The large independent physical audit uses NumPy face-incidence sums without
assembling a flow matrix. Only an unweighted graph is built to check
grounding. Existing native/device reference internals retain their original
sparse assembly; no large direct factorization is performed. The physical
conductances are constructed independently of production face arrays.

Timing boundaries and caveats
-----------------------------

Explicit kernel module compilation/loading is measured in a private empty
cache, separately from timed executions. The first complete cold-head run
then includes the first workspace, Krylov arrays, specialized kernels and
restart graphs. Three subsequent complete runs reconstruct the model and
workspace from the SAME saved cold head, using compiled kernels. They do
not reuse the previous solution. ``solve_only_seconds`` is the existing
backend timer, excluding workspace initialization and the final host
independent audit. ``public_solve_wall_seconds`` includes reference setup,
workspace allocation/transfers, graph creation, solver work and that audit.
``end_to_end_seconds`` includes NPZ load and host model construction as well.
It excludes interpreter/import startup, explicit compilation, report writing
and post-run duplicate validation. Initial compilation must be added when
comparing a cold start that includes it. Timed medians use only independently
accepted runs, with successful sample counts visible; all failures are retained.

The benchmark probe measures synchronized host-to-device neighbor array
creation and workspace field upload, including packing, Python API and
synchronization costs. Workspace initialization is a superset of these
costs; do not add it to the public solve wall a second time. GPU memory is
sampled device-wide every 100 ms, including display/context usage. Warp pool
high-water values are process cumulative and are reported with pre-run
baselines, never mislabeled as resettable per-run peaks. The sample peak
can miss a sub-100-ms transient. K-cycle hierarchy arrays also have exact
unique device allocation byte counts. Measurements include host launch and
scalar-readback costs; they are not pure GPU event times.

The MF6 process timer measures subprocess wall time, including executable
startup and native I/O; native listing elapsed time and iteration CSVs are
also retained. Write/run/decode time is separately recorded. There is one
native solve per variant, so no MF6 timing median is asserted. These times
are not directly interchangeable with a warmed GPU inner-loop timer.

K-cycle isolation and future integration
----------------------------------------

The linear experiment uses the exact confined case geometry, active/fixed
topology, materials and full-saturated A0. A smooth deterministic head field
with the same CHD values is prescribed analytically; its RHS is constructed
using the independent physical face incidence. The original physical RCH
recharge is NOT the manufactured forcing. Both the linear inputs and known
head are separately archived. Production A0 face conductances must first
agree with the independent ones to relative tolerance 1e-14. Final linear
solutions receive the same independent physical acceptance tests, plus
comparison against the manufactured head. A common absolute residual target
of 1e-10 m3/day, zero relative stopping tolerance and no head-change stopping
gate is used for every configuration, with a 100-cycle cap. Failed budget
runs are not ranked as times-to-solution.

The existing K-cycle behavior is preserved. Requested levels 3..8 are
mapped through its actual min_coarse_n=2 rule and deduplicated by actual
dimensions. Horizontal factors are two where the rounded-up coarse dimension
can meet the minimum; nz remains five. Base Chebyshev uses two pre and two
post point sweeps, spectral interval [0.05,1.95], and ten coarse relaxation
iterations. Each coarse iteration applies the pre sequence (two sweeps),
not a direct solve. After the level sweep, only level 5 is compared using
vertical line and combined point/line smoothing. Vertical-only uses the
pre/post sequence LENGTH as its line-sweep count; combined mode adds one
line sweep after each point sequence and one after each coarse sequence.
This distinction comes from the existing implementation and is recorded.

An optional observation hook synchronizes around each whole cycle and
records the existing residual check every cycle. It does not alter smoothers,
coarse operators, Krylov corrections or stopping rules. The observer-disabled
run is a control of the same unchanged algorithm. Recursive K-cycles make
two child calls at each level; deep hierarchies can have substantial
launch/scalar-synchronization work even when coarse arrays are tiny.
Block-average restriction, bilinear prolongation and aggregated coarse
faces/springs are approximate, not exact Galerkin operators.

For later integration, let D0 be the full-saturated principal diagonal and
S=D0^-1 on free rows. Newton solves S*J*delta = S*R. An approximate inverse
of A0 must receive the PHYSICAL residual D0*v when given a scaled Krylov
vector v: P_scaled(v) = approximate_A0_inverse(D0*v). The correction is
already in head units and must not receive a second row scaling. Set
homogeneous CHD correction values, zero fixed/inactive RHS, and preserve
coarse Dirichlet springs. Validate this scaling independently with exact
linear manufactured residuals before coupling the nonlinear solver.

This K-cycle computes alpha=(r1 dot z2)/(z2 dot A z2) from the current
residual; even with fixed sweep counts its map is generally nonlinear in the
right-hand side, and not an ordinary fixed matrix inverse. The current
right-preconditioned GMRES assumes one fixed linear mapping and applies it
to combinations of basis vectors. A safe next integration requires flexible
GMRES that retains EACH preconditioned basis vector and combines those
vectors for the correction. Alternatively, demonstrate a separately defined
fixed linear cycle; that would be a different algorithm and is outside this
milestone. No Newton preconditioner integration is implemented here.

A future persistent hierarchy owner should hold host levels, device faces,
masks, springs, diagonals, smoother workspaces and restriction/prolongation
buffers for a model lifetime. Reset RHS/corrections per application without
rebuilding. Reuse A0 across Newton iterates since full-saturated geometry and
K are unchanged. Invalidate on shape, interfaces, spacing, topology, fixed
mask, device or precision change; refresh coefficients and coarse arrays on
material change. Head/source/CHD-value changes alone do not require topology
reconstruction. Define one owner and deterministic close semantics; never
rebuild inside Jv or each preconditioner application. The current public
standalone function constructs/releases levels per solve, and is not that
persistent interface yet.
