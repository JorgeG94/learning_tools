// sw_cuda_async.cu - Shallow water solver with TRUE async overlap
// Computes multiple iterations while previous yieldstep transfer is in flight
// This is what OpenMP target SHOULD do but doesn't...

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Device pointers stored globally for kernel access
struct device_domain {
    int n;              // number of elements
    double *stage;
    double *xmom;
    double *ymom;
    double *bed;
    double *height;
    double *stage_update;
    double *xmom_update;
    double *ymom_update;
    double *stage_edge;
    double *height_edge;
    int *neighbours;
    double *edgelengths;
    double *normals;
    double *areas;
    double *radii;
    double *max_speed;
    // Derived quantities for verbose output
    double *xvel;
    double *yvel;
    double *speed;
    double *froude;
    double g;
    double epsilon;
    double min_height;
};

//=============================================================================
// CUDA Kernels
//=============================================================================

__global__ void extrapolate_kernel(double *stage, double *height,
                                   double *stage_edge, double *height_edge, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        double s = stage[k];
        double h = height[k];
        stage_edge[3*k + 0] = s;
        stage_edge[3*k + 1] = s;
        stage_edge[3*k + 2] = s;
        height_edge[3*k + 0] = h;
        height_edge[3*k + 1] = h;
        height_edge[3*k + 2] = h;
    }
}

__global__ void flux_kernel(double *xmom, double *ymom, double *height,
                            double *height_edge, double *stage_update,
                            double *xmom_update, double *ymom_update,
                            double *max_speed_arr,
                            int *neighbours, double *edgelengths,
                            double *normals, double *areas, double *radii,
                            double g, double epsilon, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        double stage_accum = 0.0, xmom_accum = 0.0, ymom_accum = 0.0;
        double speed_max = 0.0;
        double uh_k = xmom[k], vh_k = ymom[k];

        for (int i = 0; i < 3; i++) {
            int ki = 3*k + i;
            int nb = neighbours[ki];
            double h_left = height_edge[ki];
            double h_right, uh_right, vh_right;

            if (nb >= 0) {
                h_right = height[nb];
                uh_right = xmom[nb];
                vh_right = ymom[nb];
            } else {
                h_right = h_left;
                uh_right = -uh_k;
                vh_right = -vh_k;
            }

            double el = edgelengths[ki];
            double nx = normals[6*k + 2*i];
            double ny = normals[6*k + 2*i + 1];

            double c_left = sqrt(g * fmax(h_left, 0.0));
            double c_right = sqrt(g * fmax(h_right, 0.0));
            double c_max = fmax(c_left, c_right);

            stage_accum += c_max * (h_left - h_right) * el;
            xmom_accum += c_max * (uh_k - uh_right) * el * nx;
            ymom_accum += c_max * (vh_k - vh_right) * el * ny;

            if (c_max > epsilon) speed_max = fmax(speed_max, c_max);
        }

        double inv_area = 1.0 / areas[k];
        stage_update[k] = stage_accum * inv_area;
        xmom_update[k] = xmom_accum * inv_area;
        ymom_update[k] = ymom_accum * inv_area;
        max_speed_arr[k] = speed_max;
    }
}

__global__ void protect_kernel(double *stage, double *bed, double *xmom,
                               double *ymom, double *height,
                               double min_height, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        double hc = stage[k] - bed[k];
        if (hc < min_height) {
            xmom[k] = 0.0;
            ymom[k] = 0.0;
            if (hc <= 0.0 && stage[k] < bed[k]) {
                stage[k] = bed[k];
            }
        }
        height[k] = fmax(stage[k] - bed[k], 0.0);
    }
}

__global__ void update_kernel(double *stage, double *xmom, double *ymom,
                              double *stage_update, double *xmom_update,
                              double *ymom_update, double dt, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        stage[k] += dt * stage_update[k];
        xmom[k] += dt * xmom_update[k];
        ymom[k] += dt * ymom_update[k];
    }
}

// Compute derived quantities (velocity, speed, froude) for verbose output
__global__ void derived_kernel(double *xmom, double *ymom, double *height,
                               double *xvel, double *yvel, double *speed_arr,
                               double *froude, double g, double epsilon, int n) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k < n) {
        double h = height[k];
        if (h > epsilon) {
            double u = xmom[k] / h;
            double v = ymom[k] / h;
            double spd = sqrt(u*u + v*v);
            double c = sqrt(g * h);
            xvel[k] = u;
            yvel[k] = v;
            speed_arr[k] = spd;
            froude[k] = spd / c;
        } else {
            xvel[k] = 0.0;
            yvel[k] = 0.0;
            speed_arr[k] = 0.0;
            froude[k] = 0.0;
        }
    }
}

//=============================================================================
// Host code
//=============================================================================

void print_progress(int current, int total, double elapsed, int yields) {
    int bar_width = 40;
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);

    printf("\r  [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %3d%% (%d/%d) %.2fs [%d yields]",
           (int)(progress * 100), current, total, elapsed, yields);
    fflush(stdout);
}

void generate_mesh(int *neighbours, double *edgelengths, double *normals,
                   double *areas, double *radii, int n, int grid_size) {
    int nx = grid_size;
    int ny = grid_size;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);
    double area = 0.5 * dx * dy;
    double edgelen = dx;
    double radius = area / (1.5 * edgelen);

    for (int k = 0; k < n; k++) {
        areas[k] = area;
        radii[k] = radius;
        for (int i = 0; i < 3; i++) edgelengths[3*k + i] = edgelen;
        normals[6*k + 0] = 1.0;  normals[6*k + 1] = 0.0;
        normals[6*k + 2] = 0.0;  normals[6*k + 3] = 1.0;
        normals[6*k + 4] = -0.707; normals[6*k + 5] = -0.707;

        int cell = k / 2, tri = k % 2;
        int cx = cell % (nx - 1), cy = cell / (nx - 1);

        if (tri == 0) {
            neighbours[3*k + 0] = k + 1;
            neighbours[3*k + 1] = (cy > 0) ? 2 * ((cy-1)*(nx-1) + cx) + 1 : -1;
            neighbours[3*k + 2] = (cx > 0) ? 2 * (cy*(nx-1) + (cx-1)) + 1 : -1;
        } else {
            neighbours[3*k + 0] = k - 1;
            neighbours[3*k + 1] = (cy < ny-2) ? 2 * ((cy+1)*(nx-1) + cx) : -1;
            neighbours[3*k + 2] = (cx < nx-2) ? 2 * (cy*(nx-1) + (cx+1)) : -1;
        }
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr, "Usage: %s N [niter] [yieldstep] [async]\n", argv[0]);
        fprintf(stderr, "  async: 0=sync, 1=async (default: 1)\n");
        return 1;
    }

    int grid_size = atoi(argv[1]);
    int niter = (argc >= 3) ? atoi(argv[2]) : 1000;
    int yieldstep = (argc >= 4) ? atoi(argv[3]) : 100;
    int use_async = (argc >= 5) ? atoi(argv[4]) : 1;

    int n = 2 * (grid_size - 1) * (grid_size - 1);

    double mb_per_array = (double)(n * sizeof(double)) / (1024 * 1024);
    // Verbose output like sw_verbose.c:
    // Centroid: stage, height, xmom, ymom, stage_update, xmom_update, ymom_update, max_speed (8)
    // Derived: xvel, yvel, speed, froude (4)
    // Edge: stage_edge, height_edge (2 × 3n)
    // Total: 12 centroid + 2 edge = 12n + 6n = 18n elements
    double output_mb = 12 * mb_per_array + 2 * 3 * mb_per_array;

    printf("=== SW_CUDA_ASYNC: %s mode (VERBOSE output) ===\n", use_async ? "ASYNC" : "SYNC");
    printf("Grid: %dx%d, Elements: %d (%.2f MB/array)\n", grid_size, grid_size, n, mb_per_array);
    printf("Iterations: %d, Yieldstep: %d\n\n", niter, yieldstep);

    printf("Arrays transferred at each yieldstep (matching sw_verbose):\n");
    printf("  Centroid arrays (n elements each):\n");
    printf("    - stage, height, xmom, ymom       : 4 × %.2f MB = %.2f MB\n", mb_per_array, 4*mb_per_array);
    printf("    - stage_update, xmom_update, ymom_update, max_speed: 4 × %.2f MB = %.2f MB\n", mb_per_array, 4*mb_per_array);
    printf("    - xvel, yvel, speed, froude       : 4 × %.2f MB = %.2f MB\n", mb_per_array, 4*mb_per_array);
    printf("  Edge arrays (3n elements each):\n");
    printf("    - stage_edge, height_edge         : 2 × %.2f MB = %.2f MB\n", 3*mb_per_array, 6*mb_per_array);
    printf("  -----------------------------------------\n");
    printf("  TOTAL per yieldstep:                  %.2f MB\n\n", output_mb);

    // Allocate pinned host memory (required for async!)
    double *h_stage, *h_height, *h_xmom, *h_ymom, *h_bed;
    double *h_stage_edge, *h_height_edge;
    double *h_stage_update, *h_xmom_update, *h_ymom_update, *h_max_speed;
    double *h_xvel, *h_yvel, *h_speed, *h_froude;
    int *h_neighbours;
    double *h_edgelengths, *h_normals, *h_areas, *h_radii;

    CHECK_CUDA(cudaMallocHost(&h_stage, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_height, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_xmom, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_ymom, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_bed, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_stage_edge, 3 * n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_height_edge, 3 * n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_neighbours, 3 * n * sizeof(int)));
    CHECK_CUDA(cudaMallocHost(&h_edgelengths, 3 * n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_normals, 6 * n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_areas, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_radii, n * sizeof(double)));
    // Verbose output arrays
    CHECK_CUDA(cudaMallocHost(&h_stage_update, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_xmom_update, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_ymom_update, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_max_speed, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_xvel, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_yvel, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_speed, n * sizeof(double)));
    CHECK_CUDA(cudaMallocHost(&h_froude, n * sizeof(double)));

    // Initialize host data
    generate_mesh(h_neighbours, h_edgelengths, h_normals, h_areas, h_radii, n, grid_size);
    for (int k = 0; k < n; k++) {
        h_stage[k] = 1.0;
        h_height[k] = 1.0;
        h_xmom[k] = 0.0;
        h_ymom[k] = 0.0;
        h_bed[k] = 0.0;
    }

    // Allocate device memory
    device_domain D;
    D.n = n;
    D.g = 9.81;
    D.epsilon = 1e-12;
    D.min_height = 1e-6;

    CHECK_CUDA(cudaMalloc(&D.stage, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.xmom, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.ymom, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.bed, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.height, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.stage_update, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.xmom_update, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.ymom_update, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.stage_edge, 3 * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.height_edge, 3 * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.neighbours, 3 * n * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&D.edgelengths, 3 * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.normals, 6 * n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.areas, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.radii, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.max_speed, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.xvel, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.yvel, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.speed, n * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&D.froude, n * sizeof(double)));

    // Copy to device
    CHECK_CUDA(cudaMemcpy(D.stage, h_stage, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.xmom, h_xmom, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.ymom, h_ymom, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.bed, h_bed, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.height, h_height, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.neighbours, h_neighbours, 3 * n * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.edgelengths, h_edgelengths, 3 * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.normals, h_normals, 6 * n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.areas, h_areas, n * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(D.radii, h_radii, n * sizeof(double), cudaMemcpyHostToDevice));

    // Create streams
    cudaStream_t compute_stream, transfer_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));
    CHECK_CUDA(cudaStreamCreate(&transfer_stream));

    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;

    double dt = 0.001;
    int yield_count = 0;
    int transfer_pending = 0;
    int iters_during_transfer = 0;
    int total_iters_overlapped = 0;

    printf("Running %s...\n", use_async ? "with async overlap" : "synchronously");

    cudaDeviceSynchronize();
    double t0 = omp_get_wtime();

    for (int iter = 0; iter < niter; iter++) {
        // If async and transfer pending, check if we need to wait
        if (use_async && transfer_pending && ((iter + 1) % yieldstep == 0)) {
            // Wait for previous transfer before starting new one
            CHECK_CUDA(cudaStreamSynchronize(transfer_stream));
            total_iters_overlapped += iters_during_transfer;
            transfer_pending = 0;
            iters_during_transfer = 0;
        }

        // Run compute kernels on compute_stream
        extrapolate_kernel<<<numBlocks, blockSize, 0, compute_stream>>>(
            D.stage, D.height, D.stage_edge, D.height_edge, n);

        flux_kernel<<<numBlocks, blockSize, 0, compute_stream>>>(
            D.xmom, D.ymom, D.height, D.height_edge,
            D.stage_update, D.xmom_update, D.ymom_update, D.max_speed,
            D.neighbours, D.edgelengths, D.normals, D.areas, D.radii,
            D.g, D.epsilon, n);

        protect_kernel<<<numBlocks, blockSize, 0, compute_stream>>>(
            D.stage, D.bed, D.xmom, D.ymom, D.height, D.min_height, n);

        update_kernel<<<numBlocks, blockSize, 0, compute_stream>>>(
            D.stage, D.xmom, D.ymom, D.stage_update, D.xmom_update,
            D.ymom_update, dt, n);

        // Count iterations during transfer
        if (transfer_pending) {
            iters_during_transfer++;
        }

        // Yieldstep output
        if ((iter + 1) % yieldstep == 0) {
            yield_count++;

            // Wait for compute to finish before transferring its results
            CHECK_CUDA(cudaStreamSynchronize(compute_stream));

            // Compute derived quantities before transfer
            derived_kernel<<<numBlocks, blockSize, 0, compute_stream>>>(
                D.xmom, D.ymom, D.height, D.xvel, D.yvel, D.speed, D.froude,
                D.g, D.epsilon, n);
            CHECK_CUDA(cudaStreamSynchronize(compute_stream));

            if (use_async) {
                // Start async transfers on transfer_stream - VERBOSE output (14 arrays)
                // Primary quantities (4)
                CHECK_CUDA(cudaMemcpyAsync(h_stage, D.stage, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_height, D.height, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_xmom, D.xmom, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_ymom, D.ymom, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                // Update/flux arrays (4)
                CHECK_CUDA(cudaMemcpyAsync(h_stage_update, D.stage_update, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_xmom_update, D.xmom_update, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_ymom_update, D.ymom_update, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_max_speed, D.max_speed, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                // Derived quantities (4)
                CHECK_CUDA(cudaMemcpyAsync(h_xvel, D.xvel, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_yvel, D.yvel, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_speed, D.speed, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_froude, D.froude, n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                // Edge arrays (2 × 3n)
                CHECK_CUDA(cudaMemcpyAsync(h_stage_edge, D.stage_edge, 3 * n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                CHECK_CUDA(cudaMemcpyAsync(h_height_edge, D.height_edge, 3 * n * sizeof(double),
                                           cudaMemcpyDeviceToHost, transfer_stream));
                transfer_pending = 1;
            } else {
                // Synchronous transfers - VERBOSE output (14 arrays)
                CHECK_CUDA(cudaMemcpy(h_stage, D.stage, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_height, D.height, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_xmom, D.xmom, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_ymom, D.ymom, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_stage_update, D.stage_update, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_xmom_update, D.xmom_update, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_ymom_update, D.ymom_update, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_max_speed, D.max_speed, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_xvel, D.xvel, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_yvel, D.yvel, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_speed, D.speed, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_froude, D.froude, n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_stage_edge, D.stage_edge, 3 * n * sizeof(double), cudaMemcpyDeviceToHost));
                CHECK_CUDA(cudaMemcpy(h_height_edge, D.height_edge, 3 * n * sizeof(double), cudaMemcpyDeviceToHost));
            }
        }

        if ((iter + 1) % 100 == 0 || iter == niter - 1) {
            print_progress(iter + 1, niter, omp_get_wtime() - t0, yield_count);
        }
    }

    // Wait for final transfer
    if (transfer_pending) {
        CHECK_CUDA(cudaStreamSynchronize(transfer_stream));
        total_iters_overlapped += iters_during_transfer;
    }

    CHECK_CUDA(cudaDeviceSynchronize());
    double t_total = omp_get_wtime() - t0;
    printf("\n\n");

    printf("Results:\n");
    printf("  Total time: %.4f ms (%.4f ms/iter)\n", t_total * 1000, t_total * 1000 / niter);
    printf("  Yields: %d\n", yield_count);
    if (use_async) {
        printf("  Iterations overlapped with transfers: %d / %d (%.1f%%)\n",
               total_iters_overlapped, niter, 100.0 * total_iters_overlapped / niter);
        printf("  Average iters per transfer: %.1f\n",
               yield_count > 0 ? (double)total_iters_overlapped / yield_count : 0);
    }

    // Verify
    double sum = 0;
    for (int k = 0; k < n; k++) sum += h_stage[k];
    printf("  Avg stage: %.6f\n", sum / n);

    // Cleanup
    CHECK_CUDA(cudaStreamDestroy(compute_stream));
    CHECK_CUDA(cudaStreamDestroy(transfer_stream));

    CHECK_CUDA(cudaFree(D.stage)); CHECK_CUDA(cudaFree(D.xmom));
    CHECK_CUDA(cudaFree(D.ymom)); CHECK_CUDA(cudaFree(D.bed));
    CHECK_CUDA(cudaFree(D.height)); CHECK_CUDA(cudaFree(D.stage_update));
    CHECK_CUDA(cudaFree(D.xmom_update)); CHECK_CUDA(cudaFree(D.ymom_update));
    CHECK_CUDA(cudaFree(D.stage_edge)); CHECK_CUDA(cudaFree(D.height_edge));
    CHECK_CUDA(cudaFree(D.neighbours)); CHECK_CUDA(cudaFree(D.edgelengths));
    CHECK_CUDA(cudaFree(D.normals)); CHECK_CUDA(cudaFree(D.areas));
    CHECK_CUDA(cudaFree(D.radii)); CHECK_CUDA(cudaFree(D.max_speed));
    CHECK_CUDA(cudaFree(D.xvel)); CHECK_CUDA(cudaFree(D.yvel));
    CHECK_CUDA(cudaFree(D.speed)); CHECK_CUDA(cudaFree(D.froude));

    CHECK_CUDA(cudaFreeHost(h_stage)); CHECK_CUDA(cudaFreeHost(h_height));
    CHECK_CUDA(cudaFreeHost(h_xmom)); CHECK_CUDA(cudaFreeHost(h_ymom));
    CHECK_CUDA(cudaFreeHost(h_bed)); CHECK_CUDA(cudaFreeHost(h_stage_edge));
    CHECK_CUDA(cudaFreeHost(h_height_edge)); CHECK_CUDA(cudaFreeHost(h_neighbours));
    CHECK_CUDA(cudaFreeHost(h_edgelengths)); CHECK_CUDA(cudaFreeHost(h_normals));
    CHECK_CUDA(cudaFreeHost(h_areas)); CHECK_CUDA(cudaFreeHost(h_radii));
    CHECK_CUDA(cudaFreeHost(h_stage_update)); CHECK_CUDA(cudaFreeHost(h_xmom_update));
    CHECK_CUDA(cudaFreeHost(h_ymom_update)); CHECK_CUDA(cudaFreeHost(h_max_speed));
    CHECK_CUDA(cudaFreeHost(h_xvel)); CHECK_CUDA(cudaFreeHost(h_yvel));
    CHECK_CUDA(cudaFreeHost(h_speed)); CHECK_CUDA(cudaFreeHost(h_froude));

    return 0;
}
