"""Cartesian/terrain timings and controlled 900 m relief dry-floor sensitivity."""
from pathlib import Path
import argparse
import gc
import json
import os
import platform
import subprocess
from time import perf_counter

import numpy as np
import warp as wp

from DARCY_WARP_PACKAGE.warped_darcy_3d import WarpDarcySolver3D


def make_case(*, nx, ny, interfaces, dx, relief, basement, shallow, floor, physics):
    y,x = np.mgrid[:ny,:nx]
    xn = x/max(nx-1,1)
    yn = y/max(ny-1,1)
    surface = interfaces[-1]+relief*(xn+physics["dem_wave_fraction"]*np.sin(np.pi*xn)*np.sin(2*np.pi*yn))
    shape = (len(interfaces)-1,ny,nx)
    active = np.ones((ny,nx),dtype=bool)
    if shallow:
        active = ((xn-.5)**2/physics["footprint_radius"]**2+(yn-.5)**2/physics["footprint_radius"]**2 < 1)
    west = np.zeros_like(active)
    east = np.zeros_like(active)
    west[:,1:] = active[:,:-1]
    east[:,:-1] = active[:,1:]
    boundary = active&(~west|~east)
    fixed = np.broadcast_to(boundary,shape).astype(np.int32).copy()
    k = np.full(shape,physics["conductivity"])
    k[-1] *= basement
    values = np.broadcast_to(surface[None]-(physics["shallow_depth"] if shallow else physics["boundary_depth"]+physics["head_drop"]*xn[None]),shape).copy()
    initial = np.broadcast_to(surface[None]-(physics["shallow_depth"] if shallow else physics["initial_depth"]),shape).copy()
    source = np.zeros(shape)
    source[0] = physics["recharge"]*dx**2
    source[0,ny//2,nx//2] -= physics["abstraction"]
    source[:,~active] = 0
    return dict(surface=surface,shape=shape,active=active,fixed=fixed,k=k,
                values=values,initial=initial,source=source,interfaces=interfaces,floor=floor)


def run_case(*, case, geometry, unconfined, transient, dt, dx, repeats, device, controls, physics, verify_sparse):
    shape = case["shape"]
    nz,ny,nx = shape
    options = dict(controls)
    options.update(unconfined=unconfined,unconfined_min_sat=case["floor"])
    if not unconfined:
        options.pop("unconfined_min_sat")
    if transient:
        options.update(transient=True,dt=dt,head_prev=case["initial"])
        options.update(dict(sy=physics["sy"],ss=physics["ss"]) if unconfined else dict(storage_coeff=physics["ss"]))
    wp.synchronize_device(device=device)
    memory_before = wp.get_mempool_used_mem_current(device=device)
    build_start = perf_counter()
    solver = WarpDarcySolver3D(nx=nx,ny=ny,nz=nz,dx=dx,dy=dx,
        dz=float(np.diff(case["interfaces"])[0]),device=device,implementation="fast")
    if geometry == "terrain":
        solver.build_from_terrain(surface_elevation=case["surface"],zeta_faces=case["interfaces"],
            active_columns=case["active"],kx_field=case["k"],ky_field=.7*case["k"],kz_field=.1*case["k"],
            bc_mask=case["fixed"],bc_values=case["values"],rhs=case["source"],initial_head=case["initial"])
    else:
        active = np.broadcast_to(case["active"],shape).astype(np.int32)
        solver.build_from_K_fields(kx_field=case["k"],ky_field=.7*case["k"],kz_field=.1*case["k"],
            active=active,bc_mask=case["fixed"],bc_values=case["values"],rhs=case["source"],initial_head=case["initial"])
        if unconfined:
            options.update(zbot_field=case["surface"][None]-case["interfaces"][1:,None,None],chebyshev_enabled=False)
        options.pop("krylov_restart",None)
        if not unconfined:
            for name in ("unconfined_head_tol","final_head_residual_tol","unconfined_max_picard_iter"):
                options.pop(name,None)
    build_seconds = perf_counter()-build_start
    rows = []
    with solver:
        for repeat in range(repeats):
            wp.synchronize_device(device=device)
            start = perf_counter()
            head,info = solver.solve(**options)
            wp.synchronize_device(device=device)
            seconds = perf_counter()-start
            row = dict(repeat=repeat,seconds=seconds,geometry_build_seconds=build_seconds,
                converged=info["converged"],iterations=info.get("krylov_iterations",info.get("n_cycles_used")),
                mg_cycles=info.get("multigrid_cycles",sum(entry.get("inner_iterations",0) for entry in info.get("outer_history",[]))),
                outer=info.get("n_outer_iterations",info.get("outer_iterations")),
                workspace_reused=info.get("workspace_reused"),graph_reused=info.get("cuda_graph_reused"),
                graph_captures=info.get("cuda_graph_captures"),graph_replays=info.get("cuda_graph_replays"),
                operator_refreshes=info.get("operator_refreshes"),operator_refresh_seconds=info.get("operator_refresh_seconds"),
                physical_residual_rms=info["budget"]["cell_residual_rms"],
                physical_residual_max=info["budget"]["cell_residual_max"],
                head_residual_rms=info.get("final_head_residual_rms",info.get("final_physical_head_residual_rms")),
                domain_imbalance=info["budget"].get("domain_imbalance",info["budget"].get("imbalance")),relative_cell_imbalance=info["budget"]["relative_cell_imbalance"],
                throughput=info["budget"]["throughput"],boundary_net=info["budget"]["boundary_net"],
                gpu_memory_delta=wp.get_mempool_used_mem_current(device=device)-memory_before,
                gpu_memory_current=wp.get_mempool_used_mem_current(device=device),
                gpu_memory_process_high=wp.get_mempool_used_mem_high(device=device),
                dry=info.get("dry_cell_approximation"))
            rows.append(row)
            print(geometry,unconfined,transient,repeat,seconds,info["converged"],row["iterations"],flush=True)
        verification = None
        if verify_sparse and geometry == "terrain":
            verification = verify_frozen_physical_system(case=case,head=head,dx=dx,
                unconfined=unconfined,transient=transient,dt=dt,physics=physics,
                head_tolerance=physics["sparse_head_tolerance"])
        return dict(rows=rows,head=head.tolist(),independent_sparse=verification,relief=float(np.ptp(case["surface"])),
                    total_depth=float(case["interfaces"][-1]),active_columns=int(np.sum(case["active"])))


def run_benchmarks(*, nx, ny, dx, repeats, device, controls, output, representative_only, physics, verify_sparse):
    wp.init()
    results = {}
    if not representative_only:
        for geometry in ("cartesian","terrain"):
            for name,unconfined,transient,basement in (("confined_steady",False,False,1),
                ("confined_transient",False,True,1),("unconfined_steady",True,False,1),
                ("unconfined_transient",True,True,1),("layered_transient",True,True,physics["layered_contrast"])):
                case = make_case(nx=nx,ny=ny,interfaces=physics["regular_interfaces"],dx=dx,
                    relief=0 if geometry == "cartesian" else physics["regular_relief"],basement=basement,shallow=False,floor=physics["default_floor"],physics=physics)
                results[f"{geometry}_{name}"] = run_case(case=case,geometry=geometry,unconfined=unconfined,
                    transient=transient,dt=physics["regular_dt"],dx=dx,repeats=repeats,device=device,controls=controls,physics=physics,verify_sparse=verify_sparse)
                gc.collect()
                output.write_text(json.dumps(summarise_results(results=results,physics=physics,dx=dx,
                    controls=controls,device=device,repeats=repeats,nx=nx,ny=ny),indent=2))
    for floor in physics["floors"]:
        case = make_case(nx=nx,ny=ny,interfaces=physics["representative_interfaces"],dx=dx,
            relief=physics["representative_relief"],basement=physics["basement_contrast"],shallow=True,floor=floor,physics=physics)
        results[f"representative_floor_{floor}"] = run_case(case=case,geometry="terrain",unconfined=True,
            transient=True,dt=physics["representative_dt"],dx=dx,repeats=repeats,device=device,controls=controls,physics=physics,verify_sparse=verify_sparse)
        gc.collect()
        output.write_text(json.dumps(summarise_results(results=results,physics=physics,dx=dx,
            controls=controls,device=device,repeats=repeats,nx=nx,ny=ny),indent=2))


def verify_frozen_physical_system(*, case, head, dx, unconfined, transient, dt, physics, head_tolerance):
    """Independent sparse solve at final constitutive values; outside timing."""
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    from DARCY_WARP_PACKAGE.physics.terrain_grid_3d import TerrainGrid3D,TerrainMetrics3D
    from DARCY_WARP_PACKAGE.physics.terrain_reference_3d import TerrainReferenceOperator3D
    start = perf_counter()
    grid = TerrainGrid3D(surface_elevation=case["surface"],zeta_faces=case["interfaces"],
        dx=dx,dy=dx,active_columns=case["active"])
    operator = TerrainReferenceOperator3D(metrics=TerrainMetrics3D(grid=grid),
        kx=case["k"],ky=.7*case["k"],kz=.1*case["k"],head=head if unconfined else None,min_sat=case["floor"])
    matrix,offset = operator.sparse_matrix()
    storage = np.zeros(grid.shape)
    previous = case["initial"]
    if transient:
        if unconfined:
            wet = np.clip(head-grid.bottoms,0,grid.thickness)
            old_wet = np.clip(previous-grid.bottoms,0,grid.thickness)
            excess = np.maximum(head-grid.tops,0)-np.maximum(previous-grid.tops,0)
            change = physics["sy"]*(wet-old_wet)+physics["ss"]*((wet-old_wet)*(wet+old_wet)/2+grid.thickness*excess)
            tangent = physics["sy"]*((head > grid.bottoms)&(head < grid.tops))+physics["ss"]*wet
            storage = np.divide(change,head-previous,out=tangent.copy(),where=head != previous)*dx**2/dt
        else:
            storage = physics["ss"]*grid.volumes/dt
    matrix += sparse.diags(storage.ravel())
    datum = grid.surface_elevation.flat[0]
    known = (case["values"]-datum).ravel()
    free = ((grid.active != 0)&(case["fixed"] == 0)).ravel()
    rhs = (case["source"]+storage*(previous-datum)).ravel()
    fixed = case["fixed"].ravel() != 0
    independent = spsolve(matrix[free][:,free],rhs[free]-matrix[free][:,fixed] @ known[fixed])+datum
    difference = independent-head.ravel()[free]
    maximum = float(np.max(np.abs(difference)))
    if maximum > head_tolerance:
        raise RuntimeError(f"independent frozen sparse head error {maximum:.3g} m exceeds {head_tolerance:.3g}")
    return dict(head_max_error=maximum,head_rms_error=float(np.sqrt(np.mean(difference**2))),
                tolerance=head_tolerance,verification_seconds=perf_counter()-start)


def summarise_results(*, results, physics, dx, controls, device, repeats, nx, ny):
    """Compact diagnostics and head/water-table floor sensitivity."""
    summary = {name:{key:value for key,value in result.items() if key != "head"}
               for name,result in results.items()}
    reference_name = f'representative_floor_{physics["default_floor"]}'
    if reference_name in results:
        reference = np.asarray(results[reference_name]["head"])
        nz,ny,nx = reference.shape
        case = make_case(nx=nx,ny=ny,interfaces=physics["representative_interfaces"],dx=dx,
            relief=physics["representative_relief"],basement=physics["basement_contrast"],
            shallow=True,floor=physics["default_floor"],physics=physics)
        active = np.broadcast_to(case["active"],reference.shape)
        free = active&(case["fixed"] == 0)
        reference_table = reference[0]
        comparisons = {}
        recharge_volume = physics["recharge"]*dx**2*np.sum(free[0])
        for floor in physics["floors"]:
            name = f"representative_floor_{floor}"
            if name not in results:
                continue
            head = np.asarray(results[name]["head"])
            difference = head-reference
            rows = results[name]["rows"]
            dry = rows[-1]["dry"]
            comparisons[str(floor)] = dict(head_max_difference=float(np.max(np.abs(difference[free]))),
                head_rms_difference=float(np.sqrt(np.mean(difference[free]**2))),
                water_table_max_difference=float(np.max(np.abs(head[0]-reference_table)[free[0]])),
                surface_exchange_head_max=float(np.max((head[0]-case["surface"])[free[0]])),
                boundary_net_inflow_change=rows[-1]["boundary_net"]-results[reference_name]["rows"][-1]["boundary_net"],
                net_outward_discharge_change=results[reference_name]["rows"][-1]["boundary_net"]-rows[-1]["boundary_net"],
                horizontal_floor_to_gross_recharge=dry["horizontal_floor_excess_flow"]/recharge_volume,
                horizontal_floor_to_imposed_abstraction=dry["horizontal_floor_excess_flow"]/physics["abstraction"],
                vertical_floor_to_throughput=dry["vertical_floor_excess_flow"]/rows[-1]["throughput"])
        summary["floor_sensitivity"] = comparisons
    summary["configuration"] = {key:(value.tolist() if isinstance(value,np.ndarray) else value)
                                for key,value in physics.items()}
    summary["configuration"]["dx"] = dx
    summary["configuration"].update(nx=nx,ny=ny,repeats=repeats,device=device,controls=controls)
    summary["provenance"] = dict(branch="dev",core_commit=subprocess.check_output(
        ["git","rev-parse","HEAD"],text=True).strip(),python=platform.python_version(),warp=wp.__version__,
        device_name=wp.get_device(device).name,boundary_net_convention="positive inflow",
        independent_verification="SciPy frozen constitutive system, excluded from timed solves",
        other_gpu_processes=other_gpu_processes())
    return summary


def other_gpu_processes():
    """Record possible timing contention without managing other workloads."""
    try:
        result = subprocess.run(["nvidia-smi","--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    except (OSError,subprocess.CalledProcessError):
        return {"available":False,"processes":[]}
    processes = []
    for line in result.stdout.splitlines():
        values = line.split(",")
        if len(values) == 2 and values[0].strip().isdigit() and int(values[0]) != os.getpid():
            processes.append(dict(pid=int(values[0]),memory_mib=values[1].strip()))
    return {"available":True,"processes":processes}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx",type=int,default=64)
    parser.add_argument("--ny",type=int,default=64)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--representative-only",action="store_true")
    args = parser.parse_args()
    dx = 25.
    repeats = 3
    device = "cuda:0"
    verify_sparse = True
    physics = dict(conductivity=10.,recharge=1e-4,abstraction=.02,sy=.15,ss=1e-4,
        dem_wave_fraction=.04,footprint_radius=.55,shallow_depth=.295,
        boundary_depth=5.,head_drop=3.,initial_depth=6.,layered_contrast=1e-6,
        default_floor=.01,regular_relief=20.,representative_relief=900.,
        basement_contrast=1e-8,regular_dt=7.,representative_dt=.01,
        regular_interfaces=np.arange(5)*10.,representative_interfaces=np.array([0,.3,1,3,5,15,25]),
        floors=(.1,.01,.001),sparse_head_tolerance=2e-7)
    controls = dict(max_cycles=600,max_levels=6,krylov_restart=12,
        rel_tol=1e-10,abs_tol_min=1e-10,dh_rms_tol=1e-7,
        unconfined_head_tol=1e-9,final_head_residual_tol=1e-9,
        unconfined_max_picard_iter=900,final_budget_relative_tol=1e-8)
    run_benchmarks(nx=args.nx,ny=args.ny,dx=dx,repeats=repeats,device=device,
        controls=controls,output=args.out,representative_only=args.representative_only,physics=physics,verify_sparse=verify_sparse)
