#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

struct domain {
    int nrows;
    int ncols;
    double *a;
    double *b;
    double *c;
    double alpha;
};


void daxpy2d_gpu(struct domain *D) {
    int n = D->nrows * D->ncols;
    double alpha = D->alpha;
    
    double *a = D->a;
    double *b = D->b;
    double *c = D->c;

    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; i++) {
        c[i] = alpha * a[i] + b[i];
    }

    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + alpha*b[i];
    }
}

// we will need a similar function to initialize the domain. Most of it will get initialized via transfers, because we do not have a way to init on the GPU effectively
void init_arrays_gpu(struct domain *D) {
    int n = D->nrows * D->ncols;
    double *a = D->a;
    double *b = D->b;
    double *c = D->c;

    #pragma omp target teams distribute parallel for
    for (int i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }
}


int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s N\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    if (N <= 0) {
        fprintf(stderr, "Error: N must be positive\n");
        return 1;
    }

    size_t n = (size_t)N * N;
    printf("Array size: %d x %d = %zu elements (%.2f MB per array)\n", 
           N, N, n, (double)(n * sizeof(double)) / (1024 * 1024));

    // idea is to mimic the numpy init in ANUGA
    double* a = NULL;
    double* b = NULL;
    double* c = NULL;
    a = (double *)malloc(n*sizeof(double));
    b = (double *)malloc(n*sizeof(double));
    c = (double *)malloc(n*sizeof(double)); 

    //parallel init
    #pragma omp parallel for 
    for(int i = 0; i < n; i++){
      a[i] = 1.0;
      b[i] = 2.0;
      c[i] = 0.0;
    }

    struct domain D;
    D.nrows = N;
    D.ncols = N;
    // equivalent to get_python_domain_pointers
    D.a = a;
    D.b = b;
    D.c = c;
    // this will get transferred automatically
    D.alpha = 3.0;

    if (!D.a || !D.b || !D.c) {
        fprintf(stderr, "Error: malloc failed\n");
        return 1;
    }

    double t_alloc, t_init, t_compute, t_transfer;

    // 1. Allocate on GPU using struct members
    double t0 = omp_get_wtime();
    #pragma omp target enter data map(to: D.a[0:n], D.b[0:n], D.c[0:n])
    t_alloc = omp_get_wtime() - t0;

    // 2. Initialize arrays on GPU
    //t0 = omp_get_wtime();
    //init_arrays_gpu(&D);
    //t_init = omp_get_wtime() - t0;
    t_init = 0.0;

    // 3. Run kernel iterations
    int niter = 1000;
    t0 = omp_get_wtime();
    for (int i = 0; i < niter; i++) {
        daxpy2d_gpu(&D);
    }
    t_compute = omp_get_wtime() - t0;

    // 4. Transfer only c back to host
    t0 = omp_get_wtime();
    #pragma omp target exit data map(from: D.c[0:n]) map(delete: D.a[0:n], D.b[0:n])
    t_transfer = omp_get_wtime() - t0;

    double t_memory_total = t_alloc + t_transfer;

    // Print timing results
    printf("\nTiming breakdown:\n");
    printf("  GPU alloc/transfer:     %8.4f ms\n", t_alloc * 1000);
    //printf("  GPU init:      %8.4f ms\n", t_init * 1000);
    printf("  Compute (%3d): %8.4f ms (%.4f ms/iter)\n", 
           niter, t_compute * 1000, t_compute * 1000 / niter);
    printf("  Transfer back: %8.4f ms\n", t_transfer * 1000);
    printf("  Total memory ops: %8.4f ms\n", t_memory_total* 1000);
    printf("  --------------------------------\n");
    printf("  Total:         %8.4f ms\n", 
           (t_alloc + t_init + t_compute + t_transfer) * 1000);
    printf(" Percentage of memory/total     %8.4f % \n", 100.0*(t_memory_total / (t_alloc + t_init + t_compute + t_transfer))); 

    // Bandwidth estimate
    double bytes_per_iter = 3.0 * n * sizeof(double);
    double bandwidth = (niter * bytes_per_iter) / t_compute / 1e9;
    printf("\nEffective bandwidth: %.2f GB/s\n", bandwidth);

    // Verify
    int errors = 0;
    double expected = D.alpha * 1.0 + 2.0;
    expected = D.alpha*2.0 + 1.0;
    for (size_t i = 0; i < n && errors < 5; i++) {
        if (D.c[i] != expected) {
            fprintf(stderr, "Error at c[%zu] = %f (expected %f)\n", i, D.c[i], expected);
            errors++;
        }
    }
    if (errors == 0) {
        printf("Verification passed! c[0] = %.1f, c[n-1] = %.1f\n", D.c[0], D.c[n-1]);
    }

    free(D.a);
    free(D.b);
    free(D.c);
    return 0;
}
