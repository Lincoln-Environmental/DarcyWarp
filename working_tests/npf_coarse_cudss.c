/* cuDSS 0.8 bridge for the optional Warp coarse-solve benchmark. */
#include <cudss.h>
#include <stdlib.h>

typedef struct {
    cudssHandle_t handle;
    cudssConfig_t config;
    cudssData_t data;
    cudssMatrix_t matrix, rhs, solution;
    size_t allocated, peak;
} CoarseSolver;

static int allocate_device(void *context, void **ptr, size_t size, cudaStream_t stream)
{
    CoarseSolver *solver = context;
    cudaError_t error = cudaMallocAsync(ptr, size, stream);
    if (error == cudaSuccess) {
        solver->allocated += size;
        if (solver->allocated > solver->peak)
            solver->peak = solver->allocated;
    }
    return error;
}

static int free_device(void *context, void *ptr, size_t size, cudaStream_t stream)
{
    CoarseSolver *solver = context;
    cudaError_t error = cudaFreeAsync(ptr, stream);
    if (error == cudaSuccess)
        solver->allocated -= size;
    return error;
}

void coarse_destroy(CoarseSolver *solver)
{
    if (!solver)
        return;
    if (solver->solution)
        cudssMatrixDestroy(solver->solution);
    if (solver->rhs)
        cudssMatrixDestroy(solver->rhs);
    if (solver->matrix)
        cudssMatrixDestroy(solver->matrix);
    if (solver->data)
        cudssDataDestroy(solver->handle, solver->data);
    if (solver->config)
        cudssConfigDestroy(solver->config);
    if (solver->handle)
        cudssDestroy(solver->handle);
    free(solver);
}

int coarse_create(CoarseSolver **result, cudaStream_t stream, int64_t rows, int64_t nnz,
                  void *offsets, void *columns, void *values, void *rhs, void *solution)
{
    CoarseSolver *solver = calloc(1, sizeof(*solver));
    if (!solver)
        return -1;
    const cudssDeviceMemHandler_t handler = {
        .ctx = solver, .device_alloc = allocate_device, .device_free = free_device,
        .name = "DarcyWarp coarse bench"
    };
    int status;
#define CHECK(call) do { status = (call); if (status) goto failed; } while (0)
    CHECK(cudssCreate(&solver->handle));
    CHECK(cudssSetStream(solver->handle, stream));
    CHECK(cudssSetDeviceMemHandler(solver->handle, &handler));
    CHECK(cudssConfigCreate(&solver->config));
    CHECK(cudssDataCreate(solver->handle, &solver->data));
    CHECK(cudssMatrixCreateCsr(&solver->matrix, rows, rows, nnz, offsets, NULL, columns, values,
                              CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F, CUDSS_MTYPE_GENERAL,
                              CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    CHECK(cudssMatrixCreateDn(&solver->rhs, rows, 1, rows, rhs, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    CHECK(cudssMatrixCreateDn(&solver->solution, rows, 1, rows, solution,
                              CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
#undef CHECK
    *result = solver;
    return 0;

failed:
    coarse_destroy(solver);
    return status;
}

static int execute_phase(CoarseSolver *solver, int phase)
{
    return cudssExecute(solver->handle, phase, solver->config, solver->data,
                        solver->matrix, solver->solution, solver->rhs);
}

int coarse_analyze(CoarseSolver *solver)
{
    return execute_phase(solver, CUDSS_PHASE_ANALYSIS);
}

int coarse_factor(CoarseSolver *solver, int refactor)
{
    return execute_phase(solver, refactor ? CUDSS_PHASE_REFACTORIZATION : CUDSS_PHASE_FACTORIZATION);
}

int coarse_solve(CoarseSolver *solver)
{
    return execute_phase(solver, CUDSS_PHASE_SOLVE);
}

int coarse_get_info(CoarseSolver *solver, int *info)
{
    size_t written = 0;
    return cudssDataGet(solver->handle, solver->data, CUDSS_DATA_INFO, info, sizeof(*info), &written);
}

size_t coarse_peak_bytes(CoarseSolver *solver)
{
    return solver->peak;
}
