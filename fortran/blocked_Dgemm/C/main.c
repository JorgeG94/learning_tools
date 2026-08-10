#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

void blocked_dgemm(double *A, double *B, double *C, int Ndim, int Mdim, int Pdim);

int main(int argc, char *argv[]) {
  int N = 1024;
  
  if (argc >= 2) {
    N = atoi(argv[1]);
  }
  
  printf("N = %d\n", N);
  
  // Allocate matrices
  double *A = (double*)malloc(N * N * sizeof(double));
  double *B = (double*)malloc(N * N * sizeof(double));
  double *C = (double*)malloc(N * N * sizeof(double));
  
  // Initialize
  for (int i = 0; i < N*N; i++) {
    A[i] = 1.0;
    B[i] = 1.0;
    C[i] = 0.0;
  }
  
  // Time the computation
  double t0 = omp_get_wtime();
  blocked_dgemm(A, B, C, N, N, N);
  double t1 = omp_get_wtime();
  double dt = t1 - t0;
  
  // Calculate checksum
  double sum = 0.0;
  for (int i = 0; i < N*N; i++) {
    sum += C[i];
  }
  
  double flops = 2.0 * (double)N * (double)N * (double)N;
  double gflops = flops / dt / 1.0e9;
  
  printf("Checksum(C) = %e\n", sum);
  printf("Time = %.4f s\n", dt);
  printf("Performance = %.2f GFLOP/s\n", gflops);
  
  free(A);
  free(B);
  free(C);
  
  return 0;
}
