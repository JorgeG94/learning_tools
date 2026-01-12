// test_async_cuda.cu - Test CUDA async with explicit streams
// This should actually overlap compute and transfers!

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define N 500
#define M (2 * (N-1) * (N-1))  // ~500K elements

// Compute kernel - does some work
__global__ void compute_kernel(double *c, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = sin(a[i]) * cos(b[i]) + sqrt(a[i] + b[i]);
    }
}

__global__ void compute_kernel2(double *d, const double *a, const double *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        d[i] = cos(a[i]) * sin(b[i]) + sqrt(a[i] * b[i]);
    }
}

double get_time() {
    cudaDeviceSynchronize();
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main() {
    printf("Testing CUDA async with explicit streams\n");
    printf("Array size: %d elements (%.2f MB each)\n\n", M, (double)(M*sizeof(double))/(1024*1024));

    // Host arrays (pinned for async transfers!)
    double *h_a, *h_b, *h_c, *h_d;
    CHECK_CUDA(cudaMallocHost(&h_a, M * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_b, M * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_c, M * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_d, M * sizeof(double)));

    // Device arrays
    double *d_a, *d_b, *d_c, *d_d;
    CHECK_CUDA(cudaMalloc(&d_a, M * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_b, M * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_c, M * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_d, M * sizeof(double)));

    // Initialize
    for (int i = 0; i < M; i++) {
        h_a[i] = 1.0;
        h_b[i] = 2.0;
        h_c[i] = 0.0;
        h_d[i] = 0.0;
    }

    // Copy to device
    CHECK_CUDA(cudaMemcpy(d_a, h_a, M * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, M * sizeof(double), cudaMemcpyHostToDevice));

    int blockSize = 256;
    int numBlocks = (M + blockSize - 1) / blockSize;

    // Create streams
    cudaStream_t stream1, stream2;
    CHECK_CUDA(cudaStreamCreate(&stream1));
    CHECK_CUDA(cudaStreamCreate(&stream2));

    //=================================================================
    // TEST 1: Sequential compute kernels (baseline)
    //=================================================================
    printf("TEST 1: Sequential compute kernels (default stream)\n");
    double t0 = get_time();

    compute_kernel<<<numBlocks, blockSize>>>(d_c, d_a, d_b, M);
    compute_kernel2<<<numBlocks, blockSize>>>(d_d, d_a, d_b, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_seq = get_time() - t0;
    printf("  Time: %.4f ms\n\n", t_seq * 1000);

    //=================================================================
    // TEST 2: Parallel compute kernels (different streams)
    //=================================================================
    printf("TEST 2: Parallel compute kernels (two streams)\n");
    t0 = get_time();

    compute_kernel<<<numBlocks, blockSize, 0, stream1>>>(d_c, d_a, d_b, M);
    compute_kernel2<<<numBlocks, blockSize, 0, stream2>>>(d_d, d_a, d_b, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_par = get_time() - t0;
    printf("  Time: %.4f ms\n", t_par * 1000);
    printf("  Speedup: %.2fx %s\n\n", t_seq/t_par,
           t_par < t_seq * 0.7 ? "(OVERLAP!)" : "(no overlap)");

    //=================================================================
    // TEST 3: Sequential transfers (baseline)
    //=================================================================
    printf("TEST 3: Sequential transfers (default stream)\n");
    t0 = get_time();

    CHECK_CUDA(cudaMemcpy(h_c, d_c, M * sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_d, d_d, M * sizeof(double), cudaMemcpyDeviceToHost));

    double t_seq_xfer = get_time() - t0;
    printf("  Time: %.4f ms\n\n", t_seq_xfer * 1000);

    //=================================================================
    // TEST 4: Parallel transfers (different streams)
    //=================================================================
    printf("TEST 4: Parallel transfers (two streams) - NOTE: PCIe is single lane\n");
    t0 = get_time();

    CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, M * sizeof(double), cudaMemcpyDeviceToHost, stream1));
    CHECK_CUDA(cudaMemcpyAsync(h_d, d_d, M * sizeof(double), cudaMemcpyDeviceToHost, stream2));
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_par_xfer = get_time() - t0;
    printf("  Time: %.4f ms\n", t_par_xfer * 1000);
    printf("  Speedup: %.2fx %s\n", t_seq_xfer/t_par_xfer,
           t_par_xfer < t_seq_xfer * 0.7 ? "(OVERLAP!)" : "(no overlap - expected, PCIe serializes)");
    printf("\n");

    //=================================================================
    // TEST 5: Compute + Transfer overlap (THE REAL TEST!)
    //=================================================================
    printf("TEST 5: Compute on stream1 while transferring on stream2\n");

    // First compute c
    compute_kernel<<<numBlocks, blockSize>>>(d_c, d_a, d_b, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    t0 = get_time();

    // Transfer c on stream1, compute d on stream2 simultaneously
    CHECK_CUDA(cudaMemcpyAsync(h_c, d_c, M * sizeof(double), cudaMemcpyDeviceToHost, stream1));
    compute_kernel2<<<numBlocks, blockSize, 0, stream2>>>(d_d, d_a, d_b, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    double t_overlap = get_time() - t0;

    // Measure individual times for comparison
    compute_kernel<<<numBlocks, blockSize>>>(d_c, d_a, d_b, M);
    CHECK_CUDA(cudaDeviceSynchronize());

    double t1 = get_time();
    CHECK_CUDA(cudaMemcpy(h_c, d_c, M * sizeof(double), cudaMemcpyDeviceToHost));
    double t_xfer_only = get_time() - t1;

    t1 = get_time();
    compute_kernel2<<<numBlocks, blockSize>>>(d_d, d_a, d_b, M);
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_compute_only = get_time() - t1;

    double t_sequential = t_xfer_only + t_compute_only;

    printf("  Transfer only: %.4f ms\n", t_xfer_only * 1000);
    printf("  Compute only:  %.4f ms\n", t_compute_only * 1000);
    printf("  Sequential:    %.4f ms (sum)\n", t_sequential * 1000);
    printf("  Overlapped:    %.4f ms\n", t_overlap * 1000);
    printf("  Speedup: %.2fx %s\n\n", t_sequential/t_overlap,
           t_overlap < t_sequential * 0.8 ? "(OVERLAP!)" : "(no overlap)");

    //=================================================================
    // TEST 6: Multiple iterations with overlap pattern
    //=================================================================
    printf("TEST 6: Realistic pattern - 10 iterations with output every 2\n");

    int niter = 10;
    int yieldstep = 2;

    // Sequential version
    t0 = get_time();
    for (int iter = 0; iter < niter; iter++) {
        compute_kernel<<<numBlocks, blockSize>>>(d_c, d_a, d_b, M);
        CHECK_CUDA(cudaDeviceSynchronize());

        if ((iter + 1) % yieldstep == 0) {
            CHECK_CUDA(cudaMemcpy(h_c, d_c, M * sizeof(double), cudaMemcpyDeviceToHost));
        }
    }
    double t_seq_loop = get_time() - t0;

    // Overlapped version using double buffering
    double *d_c2;
    CHECK_CUDA(cudaMalloc(&d_c2, M * sizeof(double)));
    double *d_current = d_c;
    double *d_other = d_c2;
    int transfer_pending = 0;

    t0 = get_time();
    for (int iter = 0; iter < niter; iter++) {
        // Compute on current buffer using stream2
        compute_kernel<<<numBlocks, blockSize, 0, stream2>>>(d_current, d_a, d_b, M);

        if ((iter + 1) % yieldstep == 0) {
            // Wait for previous transfer if any
            if (transfer_pending) {
                CHECK_CUDA(cudaStreamSynchronize(stream1));
                transfer_pending = 0;
            }

            // Wait for compute to finish before starting transfer
            CHECK_CUDA(cudaStreamSynchronize(stream2));

            // Start async transfer on stream1
            CHECK_CUDA(cudaMemcpyAsync(h_c, d_current, M * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream1));
            transfer_pending = 1;

            // Swap buffers
            double *tmp = d_current;
            d_current = d_other;
            d_other = tmp;
        }
    }
    // Wait for final transfer
    CHECK_CUDA(cudaDeviceSynchronize());
    double t_overlap_loop = get_time() - t0;

    printf("  Sequential loop: %.4f ms\n", t_seq_loop * 1000);
    printf("  Overlapped loop: %.4f ms\n", t_overlap_loop * 1000);
    printf("  Speedup: %.2fx %s\n\n", t_seq_loop/t_overlap_loop,
           t_overlap_loop < t_seq_loop * 0.9 ? "(OVERLAP!)" : "(minimal benefit)");

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(stream1));
    CHECK_CUDA(cudaStreamDestroy(stream2));
    CHECK_CUDA(cudaFree(d_a));
    CHECK_CUDA(cudaFree(d_b));
    CHECK_CUDA(cudaFree(d_c));
    CHECK_CUDA(cudaFree(d_c2));
    CHECK_CUDA(cudaFree(d_d));
    CHECK_CUDA(cudaFreeHost(h_a));
    CHECK_CUDA(cudaFreeHost(h_b));
    CHECK_CUDA(cudaFreeHost(h_c));
    CHECK_CUDA(cudaFreeHost(h_d));

    printf("Summary:\n");
    printf("  - Compute+compute overlap: Should work (GPU has multiple SMs)\n");
    printf("  - Transfer+transfer: Won't overlap (single PCIe bus)\n");
    printf("  - Compute+transfer: SHOULD overlap (different hardware units!)\n");
    printf("  - Key: Use cudaMallocHost for pinned memory + separate streams\n");

    return 0;
}
