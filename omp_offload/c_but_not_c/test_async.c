// test_async.c - Test what actually overlaps with OpenMP target nowait
// Comparing: compute+compute overlap vs compute+transfer overlap

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

#define N 500
#define M (2 * (N-1) * (N-1))  // ~500K elements

int main() {
    printf("Testing OpenMP target async behavior\n");
    printf("Array size: %d elements (%.2f MB each)\n\n", M, (double)(M*sizeof(double))/(1024*1024));

    double *a = (double*)malloc(M * sizeof(double));
    double *b = (double*)malloc(M * sizeof(double));
    double *c = (double*)malloc(M * sizeof(double));
    double *d = (double*)malloc(M * sizeof(double));

    // Initialize
    for (int i = 0; i < M; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
        d[i] = 0.0;
    }

    // Map to GPU
    #pragma omp target enter data map(to: a[0:M], b[0:M], c[0:M], d[0:M])

    //=================================================================
    // TEST 1: Sequential compute kernels (baseline)
    //=================================================================
    printf("TEST 1: Sequential compute kernels\n");
    double t0 = omp_get_wtime();

    #pragma omp target teams distribute parallel for
    for (int i = 0; i < M; i++) {
        c[i] = sin(a[i]) * cos(b[i]) + sqrt(a[i] + b[i]);
    }

    #pragma omp target teams distribute parallel for
    for (int i = 0; i < M; i++) {
        d[i] = cos(a[i]) * sin(b[i]) + sqrt(a[i] * b[i]);
    }

    double t_seq = omp_get_wtime() - t0;
    printf("  Time: %.4f ms\n\n", t_seq * 1000);

    //=================================================================
    // TEST 2: Parallel compute kernels with nowait
    //=================================================================
    printf("TEST 2: Parallel compute kernels (nowait)\n");
    t0 = omp_get_wtime();

    #pragma omp target teams distribute parallel for nowait
    for (int i = 0; i < M; i++) {
        c[i] = sin(a[i]) * cos(b[i]) + sqrt(a[i] + b[i]);
    }

    #pragma omp target teams distribute parallel for nowait
    for (int i = 0; i < M; i++) {
        d[i] = cos(a[i]) * sin(b[i]) + sqrt(a[i] * b[i]);
    }

    #pragma omp taskwait

    double t_par = omp_get_wtime() - t0;
    printf("  Time: %.4f ms\n", t_par * 1000);
    printf("  Speedup: %.2fx %s\n\n", t_seq/t_par,
           t_par < t_seq * 0.7 ? "(OVERLAP!)" : "(no overlap)");

    //=================================================================
    // TEST 3: Sequential transfer
    //=================================================================
    printf("TEST 3: Sequential transfers\n");
    t0 = omp_get_wtime();

    #pragma omp target update from(c[0:M])
    #pragma omp target update from(d[0:M])

    double t_seq_xfer = omp_get_wtime() - t0;
    printf("  Time: %.4f ms\n\n", t_seq_xfer * 1000);

    //=================================================================
    // TEST 4: Parallel transfers with nowait
    //=================================================================
    printf("TEST 4: Parallel transfers (nowait)\n");
    t0 = omp_get_wtime();

    #pragma omp target update from(c[0:M]) nowait
    #pragma omp target update from(d[0:M]) nowait

    #pragma omp taskwait

    double t_par_xfer = omp_get_wtime() - t0;
    printf("  Time: %.4f ms\n", t_par_xfer * 1000);
    printf("  Speedup: %.2fx %s\n\n", t_seq_xfer/t_par_xfer,
           t_par_xfer < t_seq_xfer * 0.7 ? "(OVERLAP!)" : "(no overlap)");

    //=================================================================
    // TEST 5: Compute + Transfer overlap attempt
    //=================================================================
    printf("TEST 5: Compute while transferring (the real test)\n");

    // First do compute
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < M; i++) {
        c[i] = sin(a[i]) * cos(b[i]);
    }

    t0 = omp_get_wtime();

    // Start transfer of c (nowait)
    #pragma omp target update from(c[0:M]) nowait

    // While c transfers, compute d
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < M; i++) {
        d[i] = cos(a[i]) * sin(b[i]) + sqrt(a[i] * b[i]);
    }

    #pragma omp taskwait

    double t_overlap = omp_get_wtime() - t0;

    // Compare to sequential: transfer then compute
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < M; i++) {
        c[i] = sin(a[i]) * cos(b[i]);
    }

    double t1 = omp_get_wtime();
    #pragma omp target update from(c[0:M])
    double t_xfer_only = omp_get_wtime() - t1;

    t1 = omp_get_wtime();
    #pragma omp target teams distribute parallel for
    for (int i = 0; i < M; i++) {
        d[i] = cos(a[i]) * sin(b[i]) + sqrt(a[i] * b[i]);
    }
    double t_compute_only = omp_get_wtime() - t1;

    double t_sequential = t_xfer_only + t_compute_only;

    printf("  Transfer only: %.4f ms\n", t_xfer_only * 1000);
    printf("  Compute only:  %.4f ms\n", t_compute_only * 1000);
    printf("  Sequential:    %.4f ms (sum)\n", t_sequential * 1000);
    printf("  Overlapped:    %.4f ms\n", t_overlap * 1000);
    printf("  Speedup: %.2fx %s\n\n", t_sequential/t_overlap,
           t_overlap < t_sequential * 0.8 ? "(OVERLAP!)" : "(no overlap)");

    //=================================================================
    // TEST 6: Using depend clauses
    //=================================================================
    printf("TEST 6: Using depend clauses\n");

    #pragma omp target teams distribute parallel for
    for (int i = 0; i < M; i++) {
        c[i] = sin(a[i]) * cos(b[i]);
    }

    t0 = omp_get_wtime();

    #pragma omp target update from(c[0:M]) nowait depend(out: c)

    #pragma omp target teams distribute parallel for nowait depend(out: d)
    for (int i = 0; i < M; i++) {
        d[i] = cos(a[i]) * sin(b[i]) + sqrt(a[i] * b[i]);
    }

    #pragma omp taskwait

    double t_depend = omp_get_wtime() - t0;
    printf("  Time: %.4f ms\n", t_depend * 1000);
    printf("  vs sequential: %.2fx %s\n\n", t_sequential/t_depend,
           t_depend < t_sequential * 0.8 ? "(OVERLAP!)" : "(no overlap)");

    // Cleanup
    #pragma omp target exit data map(delete: a[0:M], b[0:M], c[0:M], d[0:M])

    free(a); free(b); free(c); free(d);

    printf("Summary:\n");
    printf("  If TEST 2 shows speedup but TEST 4/5/6 don't,\n");
    printf("  then compute overlaps but transfers don't.\n");
    printf("  This is a known limitation of OpenMP target.\n");

    return 0;
}
