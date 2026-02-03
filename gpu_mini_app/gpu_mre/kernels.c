#include "kernels.h"
#include <omp.h>
#include <stdio.h>

static int g_selected_device = 0;
static int g_num_devices = 0;

int mre_init(MPI_Comm comm, MREContext* ctx) {
    int err;

    ctx->comm = comm;

    /* Get MPI rank and size */
    err = MPI_Comm_rank(comm, &ctx->rank);
    if (err != MPI_SUCCESS) return err;

    err = MPI_Comm_size(comm, &ctx->size);
    if (err != MPI_SUCCESS) return err;

    /* Get number of GPU devices */
#ifdef CPU_ONLY_MODE
    g_num_devices = 0;
    g_selected_device = -1;
#else
    g_num_devices = omp_get_num_devices();

    /* Select device based on rank */
    if (g_num_devices > 0) {
        g_selected_device = ctx->rank % g_num_devices;
        omp_set_default_device(g_selected_device);
    } else {
        g_selected_device = -1;
    }
#endif

    ctx->num_devices = g_num_devices;
    ctx->selected_device = g_selected_device;

    printf("Rank %d/%d: num_devices=%d, selected_device=%d\n",
           ctx->rank, ctx->size, g_num_devices, g_selected_device);
    fflush(stdout);

    return MPI_SUCCESS;
}

int mre_finalize(MREContext* ctx) {
    /* Nothing to clean up for now */
    ctx->comm = MPI_COMM_NULL;
    return MPI_SUCCESS;
}

int mre_saxpy(int n, double alpha, double* x, double* y) {
#ifdef CPU_ONLY_MODE
    /* CPU-only path with standard OpenMP */
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + y[i];
    }
#else
    /* GPU offloading path */
    #pragma omp target teams loop map(to: x[0:n]) map(tofrom: y[0:n])
    for (int i = 0; i < n; i++) {
        y[i] = alpha * x[i] + y[i];
    }
#endif
    return 0;
}

int mre_get_num_devices(void) {
    return g_num_devices;
}

int mre_get_selected_device(void) {
    return g_selected_device;
}

int mre_barrier(MREContext* ctx) {
    return MPI_Barrier(ctx->comm);
}
