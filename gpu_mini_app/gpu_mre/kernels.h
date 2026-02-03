#ifndef KERNELS_H
#define KERNELS_H

#include <mpi.h>

typedef struct {
    int rank;
    int size;
    int num_devices;
    int selected_device;
    MPI_Comm comm;
} MREContext;

/* Initialize the MRE context with MPI communicator */
int mre_init(MPI_Comm comm, MREContext* ctx);

/* Finalize/cleanup */
int mre_finalize(MREContext* ctx);

/* SAXPY: y = alpha * x + y */
int mre_saxpy(int n, double alpha, double* x, double* y);

/* Get device info */
int mre_get_num_devices(void);
int mre_get_selected_device(void);

/* MPI barrier */
int mre_barrier(MREContext* ctx);

#endif /* KERNELS_H */
