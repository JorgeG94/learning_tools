#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

int main(int argc, char** argv) {
    int ierr, rank, nprocs;
    int i, iter;
    int n = 1000000;
    const int niter = 10000;
    double a = 2.5;
    double *x, *y;
    double t1, t2;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    
    // Allocate arrays
    x = (double*) malloc(n * sizeof(double));
    y = (double*) malloc(n * sizeof(double));
    
    // Initialize arrays
    for (i = 0; i < n; i++) {
        x[i] = (double)(i + 1);
        y[i] = (double)(i + 1) * 0.5;
    }
    
    if (rank == 0) {
        printf("Starting DAXPY iterations on %d ranks\n", nprocs);
    }
    
    // Map data to device
    #pragma omp target enter data map(to: x[0:n], y[0:n])
    
    // Barrier to synchronize before timing
    MPI_Barrier(MPI_COMM_WORLD);
    
    t1 = MPI_Wtime();
    
    // Perform niter DAXPY operations
    for (iter = 0; iter < niter; iter++) {
        #pragma omp target teams distribute parallel for
        for (i = 0; i < n; i++) {
            y[i] = a * x[i] + y[i];
        }
    }
    
    t2 = MPI_Wtime();
    
    // Copy data back from device
    #pragma omp target update from(y[0:n])
    #pragma omp target exit data map(delete: x[0:n], y[0:n])
    
    // Print timing from rank 0
    if (rank == 0) {
        printf("Time (s) = %.6f\n", t2 - t1);
        printf("First y value = %.4e\n", y[0]);
    }
    
    // Clean up
    free(x);
    free(y);
    
    MPI_Finalize();
    
    return 0;
}
