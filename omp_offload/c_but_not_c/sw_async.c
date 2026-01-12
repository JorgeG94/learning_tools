// sw_async.c - Shallow water mini app with ASYNC transfers using OpenMP tasks
// Uses double buffering to overlap GPU compute with PCIe transfers
// While GPU computes iteration N, we transfer output from iteration N-1

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

struct domain {
    int number_of_elements;

    double *stage_centroid_values;
    double *xmom_centroid_values;
    double *ymom_centroid_values;
    double *bed_centroid_values;
    double *height_centroid_values;

    double *stage_explicit_update;
    double *xmom_explicit_update;
    double *ymom_explicit_update;

    double *stage_edge_values;
    double *height_edge_values;

    int *neighbours;
    double *edgelengths;
    double *normals;
    double *areas;
    double *radii;
    double *max_speed;

    double g;
    double epsilon;
    double minimum_allowed_height;
};

// Double-buffered output structure
struct output_buffer {
    double *stage;
    double *height;
    double *xmom;
    double *ymom;
    double *xvel;
    double *yvel;
    double *speed;
    double *froude;
    double *stage_update;
    double *xmom_update;
    double *ymom_update;
    double *stage_edge;
    double *height_edge;
    double *max_speed_elem;

    double time;
    double total_mass;
    double max_speed_scalar;
    int ready;  // Flag: 1 if data is ready to be processed
};

static void print_progress(int current, int total, double elapsed, int yields) {
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

static void alloc_output_buffer(struct output_buffer *buf, int n) {
    buf->stage = (double *)malloc(n * sizeof(double));
    buf->height = (double *)malloc(n * sizeof(double));
    buf->xmom = (double *)malloc(n * sizeof(double));
    buf->ymom = (double *)malloc(n * sizeof(double));
    buf->xvel = (double *)malloc(n * sizeof(double));
    buf->yvel = (double *)malloc(n * sizeof(double));
    buf->speed = (double *)malloc(n * sizeof(double));
    buf->froude = (double *)malloc(n * sizeof(double));
    buf->stage_update = (double *)malloc(n * sizeof(double));
    buf->xmom_update = (double *)malloc(n * sizeof(double));
    buf->ymom_update = (double *)malloc(n * sizeof(double));
    buf->stage_edge = (double *)malloc(3 * n * sizeof(double));
    buf->height_edge = (double *)malloc(3 * n * sizeof(double));
    buf->max_speed_elem = (double *)malloc(n * sizeof(double));
    buf->ready = 0;
}

static void free_output_buffer(struct output_buffer *buf) {
    free(buf->stage); free(buf->height); free(buf->xmom); free(buf->ymom);
    free(buf->xvel); free(buf->yvel); free(buf->speed); free(buf->froude);
    free(buf->stage_update); free(buf->xmom_update); free(buf->ymom_update);
    free(buf->stage_edge); free(buf->height_edge); free(buf->max_speed_elem);
}

static int generate_mesh(struct domain *D, int grid_size) {
    int nx = grid_size;
    int ny = grid_size;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);

    int num_elements = 2 * (nx - 1) * (ny - 1);
    D->number_of_elements = num_elements;

    double area = 0.5 * dx * dy;
    double edgelen = dx;
    double radius = area / (1.5 * edgelen);

    #pragma omp parallel for
    for (int k = 0; k < num_elements; k++) {
        D->areas[k] = area;
        D->radii[k] = radius;

        for (int i = 0; i < 3; i++) {
            D->edgelengths[3*k + i] = edgelen;
        }

        D->normals[6*k + 0] = 1.0;  D->normals[6*k + 1] = 0.0;
        D->normals[6*k + 2] = 0.0;  D->normals[6*k + 3] = 1.0;
        D->normals[6*k + 4] = -0.707; D->normals[6*k + 5] = -0.707;
    }

    #pragma omp parallel for
    for (int k = 0; k < num_elements; k++) {
        int cell = k / 2;
        int tri_in_cell = k % 2;
        int cell_x = cell % (nx - 1);
        int cell_y = cell / (nx - 1);

        if (tri_in_cell == 0) {
            D->neighbours[3*k + 0] = k + 1;
            D->neighbours[3*k + 1] = (cell_y > 0) ? 2 * ((cell_y - 1) * (nx - 1) + cell_x) + 1 : -1;
            D->neighbours[3*k + 2] = (cell_x > 0) ? 2 * (cell_y * (nx - 1) + (cell_x - 1)) + 1 : -1;
        } else {
            D->neighbours[3*k + 0] = k - 1;
            D->neighbours[3*k + 1] = (cell_y < ny - 2) ? 2 * ((cell_y + 1) * (nx - 1) + cell_x) : -1;
            D->neighbours[3*k + 2] = (cell_x < nx - 2) ? 2 * (cell_y * (nx - 1) + (cell_x + 1)) : -1;
        }
    }

    return num_elements;
}

static void init_quantities(struct domain *D) {
    int n = D->number_of_elements;

    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        D->bed_centroid_values[k] = 0.0;
        D->stage_centroid_values[k] = 1.0;
        D->height_centroid_values[k] = 1.0;
        D->xmom_centroid_values[k] = 0.0;
        D->ymom_centroid_values[k] = 0.0;
        D->stage_explicit_update[k] = 0.0;
        D->xmom_explicit_update[k] = 0.0;
        D->ymom_explicit_update[k] = 0.0;
        D->max_speed[k] = 0.0;

        for (int i = 0; i < 3; i++) {
            D->stage_edge_values[3*k + i] = 1.0;
            D->height_edge_values[3*k + i] = 1.0;
        }
    }
}

// GPU kernels (same as before, no map clauses)
static double compute_fluxes_gpu(struct domain *D) {
    int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;
    double local_timestep = 1.0e+100;

    double *xmom_centroid = D->xmom_centroid_values;
    double *ymom_centroid = D->ymom_centroid_values;
    double *height_centroid = D->height_centroid_values;
    double *height_edge = D->height_edge_values;
    double *stage_update = D->stage_explicit_update;
    double *xmom_update = D->xmom_explicit_update;
    double *ymom_update = D->ymom_explicit_update;
    int *neighbours = D->neighbours;
    double *edgelengths = D->edgelengths;
    double *normals = D->normals;
    double *areas = D->areas;
    double *radii = D->radii;
    double *max_speed = D->max_speed;

    #pragma omp target teams distribute parallel for reduction(min: local_timestep)
    for (int k = 0; k < n; k++) {
        double stage_accum = 0.0, xmom_accum = 0.0, ymom_accum = 0.0, speed_max = 0.0;
        double uh_k = xmom_centroid[k], vh_k = ymom_centroid[k];

        for (int i = 0; i < 3; i++) {
            int ki = 3*k + i;
            int nb = neighbours[ki];
            double h_left = height_edge[ki];
            double h_right, uh_right, vh_right;

            if (nb >= 0) {
                h_right = height_centroid[nb];
                uh_right = xmom_centroid[nb];
                vh_right = ymom_centroid[nb];
            } else {
                h_right = h_left;
                uh_right = -uh_k;
                vh_right = -vh_k;
            }

            double edgelen = edgelengths[ki];
            double nx = normals[6*k + 2*i];
            double ny = normals[6*k + 2*i + 1];
            double c_left = sqrt(g * fmax(h_left, 0.0));
            double c_right = sqrt(g * fmax(h_right, 0.0));
            double c_max = fmax(c_left, c_right);

            stage_accum += c_max * (h_left - h_right) * edgelen;
            xmom_accum += c_max * (uh_k - uh_right) * edgelen * nx;
            ymom_accum += c_max * (vh_k - vh_right) * edgelen * ny;
            if (c_max > epsilon) speed_max = fmax(speed_max, c_max);
        }

        double inv_area = 1.0 / areas[k];
        stage_update[k] = stage_accum * inv_area;
        xmom_update[k] = xmom_accum * inv_area;
        ymom_update[k] = ymom_accum * inv_area;
        max_speed[k] = speed_max;

        if (speed_max > epsilon) {
            local_timestep = fmin(local_timestep, radii[k] / speed_max);
        }
    }
    return local_timestep;
}

static void protect_gpu(struct domain *D) {
    int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;

    double *stage_centroid = D->stage_centroid_values;
    double *bed_centroid = D->bed_centroid_values;
    double *xmom_centroid = D->xmom_centroid_values;
    double *ymom_centroid = D->ymom_centroid_values;
    double *height_centroid = D->height_centroid_values;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        double hc = stage_centroid[k] - bed_centroid[k];
        if (hc < minimum_allowed_height) {
            xmom_centroid[k] = 0.0;
            ymom_centroid[k] = 0.0;
            if (hc <= 0.0 && stage_centroid[k] < bed_centroid[k]) {
                stage_centroid[k] = bed_centroid[k];
            }
        }
        height_centroid[k] = fmax(stage_centroid[k] - bed_centroid[k], 0.0);
    }
}

static void update_gpu(struct domain *D, double dt) {
    int n = D->number_of_elements;

    double *stage_centroid = D->stage_centroid_values;
    double *xmom_centroid = D->xmom_centroid_values;
    double *ymom_centroid = D->ymom_centroid_values;
    double *stage_update = D->stage_explicit_update;
    double *xmom_update = D->xmom_explicit_update;
    double *ymom_update = D->ymom_explicit_update;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        stage_centroid[k] += dt * stage_update[k];
        xmom_centroid[k] += dt * xmom_update[k];
        ymom_centroid[k] += dt * ymom_update[k];
    }
}

static void extrapolate_to_edges_gpu(struct domain *D) {
    int n = D->number_of_elements;

    double *stage_centroid = D->stage_centroid_values;
    double *height_centroid = D->height_centroid_values;
    double *stage_edge = D->stage_edge_values;
    double *height_edge = D->height_edge_values;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < 3; i++) {
            int ki = 3*k + i;
            stage_edge[ki] = stage_centroid[k];
            height_edge[ki] = height_centroid[k];
        }
    }
}

// ASYNC transfer using nowait - initiates transfer without blocking
static void start_async_transfer(struct domain *D, struct output_buffer *buf,
                                  double sim_time, int n) {
    // These transfers happen asynchronously - GPU can continue computing
    // The nowait clause makes them non-blocking

    #pragma omp target update from(D->stage_centroid_values[0:n]) nowait
    #pragma omp target update from(D->height_centroid_values[0:n]) nowait
    #pragma omp target update from(D->xmom_centroid_values[0:n]) nowait
    #pragma omp target update from(D->ymom_centroid_values[0:n]) nowait
    #pragma omp target update from(D->stage_explicit_update[0:n]) nowait
    #pragma omp target update from(D->xmom_explicit_update[0:n]) nowait
    #pragma omp target update from(D->ymom_explicit_update[0:n]) nowait
    #pragma omp target update from(D->stage_edge_values[0:3*n]) nowait
    #pragma omp target update from(D->height_edge_values[0:3*n]) nowait
    #pragma omp target update from(D->max_speed[0:n]) nowait

    buf->time = sim_time;
    buf->ready = 0;  // Not ready yet - transfers in flight
}

// Wait for async transfers to complete and compute derived quantities
static void finish_async_transfer(struct domain *D, struct output_buffer *buf, int n) {
    // Wait for all async transfers to complete
    #pragma omp taskwait

    // Now copy to buffer and compute derived quantities on host
    double g = D->g;
    double epsilon = D->epsilon;
    double total_mass = 0.0;
    double max_spd = 0.0;

    #pragma omp parallel for reduction(+:total_mass) reduction(max:max_spd)
    for (int k = 0; k < n; k++) {
        buf->stage[k] = D->stage_centroid_values[k];
        buf->height[k] = D->height_centroid_values[k];
        buf->xmom[k] = D->xmom_centroid_values[k];
        buf->ymom[k] = D->ymom_centroid_values[k];
        buf->stage_update[k] = D->stage_explicit_update[k];
        buf->xmom_update[k] = D->xmom_explicit_update[k];
        buf->ymom_update[k] = D->ymom_explicit_update[k];
        buf->max_speed_elem[k] = D->max_speed[k];

        // Compute derived quantities
        double h = buf->height[k];
        if (h > epsilon) {
            buf->xvel[k] = buf->xmom[k] / h;
            buf->yvel[k] = buf->ymom[k] / h;
            buf->speed[k] = sqrt(buf->xvel[k]*buf->xvel[k] + buf->yvel[k]*buf->yvel[k]);
            buf->froude[k] = buf->speed[k] / sqrt(g * h);
        } else {
            buf->xvel[k] = buf->yvel[k] = buf->speed[k] = buf->froude[k] = 0.0;
        }

        total_mass += h * D->areas[k];
        max_spd = fmax(max_spd, buf->max_speed_elem[k]);
    }

    // Copy edge values
    for (int k = 0; k < 3*n; k++) {
        buf->stage_edge[k] = D->stage_edge_values[k];
        buf->height_edge[k] = D->height_edge_values[k];
    }

    buf->total_mass = total_mass;
    buf->max_speed_scalar = max_spd;
    buf->ready = 1;
}

// Synchronous transfer for comparison
static void sync_transfer(struct domain *D, struct output_buffer *buf,
                          double sim_time, int n, double *t_accum) {
    double t0 = omp_get_wtime();

    #pragma omp target update from(D->stage_centroid_values[0:n])
    #pragma omp target update from(D->height_centroid_values[0:n])
    #pragma omp target update from(D->xmom_centroid_values[0:n])
    #pragma omp target update from(D->ymom_centroid_values[0:n])
    #pragma omp target update from(D->stage_explicit_update[0:n])
    #pragma omp target update from(D->xmom_explicit_update[0:n])
    #pragma omp target update from(D->ymom_explicit_update[0:n])
    #pragma omp target update from(D->stage_edge_values[0:3*n])
    #pragma omp target update from(D->height_edge_values[0:3*n])
    #pragma omp target update from(D->max_speed[0:n])

    // Copy and compute derived
    double g = D->g;
    double epsilon = D->epsilon;
    double total_mass = 0.0, max_spd = 0.0;

    #pragma omp parallel for reduction(+:total_mass) reduction(max:max_spd)
    for (int k = 0; k < n; k++) {
        buf->stage[k] = D->stage_centroid_values[k];
        buf->height[k] = D->height_centroid_values[k];
        buf->xmom[k] = D->xmom_centroid_values[k];
        buf->ymom[k] = D->ymom_centroid_values[k];
        buf->stage_update[k] = D->stage_explicit_update[k];
        buf->xmom_update[k] = D->xmom_explicit_update[k];
        buf->ymom_update[k] = D->ymom_explicit_update[k];
        buf->max_speed_elem[k] = D->max_speed[k];

        double h = buf->height[k];
        if (h > epsilon) {
            buf->xvel[k] = buf->xmom[k] / h;
            buf->yvel[k] = buf->ymom[k] / h;
            buf->speed[k] = sqrt(buf->xvel[k]*buf->xvel[k] + buf->yvel[k]*buf->yvel[k]);
            buf->froude[k] = buf->speed[k] / sqrt(g * h);
        } else {
            buf->xvel[k] = buf->yvel[k] = buf->speed[k] = buf->froude[k] = 0.0;
        }

        total_mass += h * D->areas[k];
        max_spd = fmax(max_spd, buf->max_speed_elem[k]);
    }

    for (int k = 0; k < 3*n; k++) {
        buf->stage_edge[k] = D->stage_edge_values[k];
        buf->height_edge[k] = D->height_edge_values[k];
    }

    buf->time = sim_time;
    buf->total_mass = total_mass;
    buf->max_speed_scalar = max_spd;
    buf->ready = 1;

    *t_accum += omp_get_wtime() - t0;
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 5) {
        fprintf(stderr, "Usage: %s N [niter] [yieldstep] [async]\n", argv[0]);
        fprintf(stderr, "  N         = grid size\n");
        fprintf(stderr, "  niter     = iterations (default: 1000)\n");
        fprintf(stderr, "  yieldstep = output interval (default: 100)\n");
        fprintf(stderr, "  async     = 0=sync, 1=async (default: 1)\n");
        return 1;
    }

    int grid_size = atoi(argv[1]);
    int niter = (argc >= 3) ? atoi(argv[2]) : 1000;
    int yieldstep = (argc >= 4) ? atoi(argv[3]) : 100;
    int use_async = (argc >= 5) ? atoi(argv[4]) : 1;

    if (grid_size < 3) {
        fprintf(stderr, "Error: Grid size must be at least 3\n");
        return 1;
    }

    int n = 2 * (grid_size - 1) * (grid_size - 1);
    int num_yields = niter / yieldstep;

    double mb_per_array = (double)(n * sizeof(double)) / (1024 * 1024);
    double total_mb = 10 * mb_per_array + 2 * 3 * mb_per_array;  // 10 centroid + 2 edge arrays

    printf("=== SW_ASYNC: %s transfers ===\n", use_async ? "ASYNC" : "SYNC");
    printf("Grid: %dx%d, Elements: %d\n", grid_size, grid_size, n);
    printf("Iterations: %d, Yieldstep: %d (yields: %d)\n", niter, yieldstep, num_yields);
    printf("Transfer per yield: %.2f MB\n\n", total_mb);

    // Allocate domain
    struct domain D;
    D.g = 9.81;
    D.epsilon = 1.0e-12;
    D.minimum_allowed_height = 1.0e-6;

    D.stage_centroid_values = (double *)malloc(n * sizeof(double));
    D.xmom_centroid_values = (double *)malloc(n * sizeof(double));
    D.ymom_centroid_values = (double *)malloc(n * sizeof(double));
    D.bed_centroid_values = (double *)malloc(n * sizeof(double));
    D.height_centroid_values = (double *)malloc(n * sizeof(double));
    D.stage_explicit_update = (double *)malloc(n * sizeof(double));
    D.xmom_explicit_update = (double *)malloc(n * sizeof(double));
    D.ymom_explicit_update = (double *)malloc(n * sizeof(double));
    D.stage_edge_values = (double *)malloc(3 * n * sizeof(double));
    D.height_edge_values = (double *)malloc(3 * n * sizeof(double));
    D.neighbours = (int *)malloc(3 * n * sizeof(int));
    D.edgelengths = (double *)malloc(3 * n * sizeof(double));
    D.normals = (double *)malloc(6 * n * sizeof(double));
    D.areas = (double *)malloc(n * sizeof(double));
    D.radii = (double *)malloc(n * sizeof(double));
    D.max_speed = (double *)malloc(n * sizeof(double));

    // Double buffers for async
    struct output_buffer buf[2];
    alloc_output_buffer(&buf[0], n);
    alloc_output_buffer(&buf[1], n);
    int current_buf = 0;
    int pending_transfer = 0;  // Is there a transfer in flight?

    // Initialize
    generate_mesh(&D, grid_size);
    init_quantities(&D);

    // Transfer to GPU
    printf("Transferring to GPU...\n");
    double t0 = omp_get_wtime();
    #pragma omp target enter data \
        map(to: D.stage_centroid_values[0:n], D.xmom_centroid_values[0:n], \
                D.ymom_centroid_values[0:n], D.bed_centroid_values[0:n], \
                D.height_centroid_values[0:n], D.stage_explicit_update[0:n], \
                D.xmom_explicit_update[0:n], D.ymom_explicit_update[0:n], \
                D.stage_edge_values[0:3*n], D.height_edge_values[0:3*n], \
                D.neighbours[0:3*n], D.edgelengths[0:3*n], D.normals[0:6*n], \
                D.areas[0:n], D.radii[0:n], D.max_speed[0:n])
    double t_to_gpu = omp_get_wtime() - t0;

    // Main loop
    double dt = 0.001;
    double sim_time = 0.0;
    double t_transfer = 0.0;
    double t_wait = 0.0;
    int yield_count = 0;

    printf("Computing (%s mode):\n", use_async ? "async" : "sync");

    t0 = omp_get_wtime();

    for (int iter = 0; iter < niter; iter++) {
        // If async transfer pending, wait for it before this yield's transfer
        if (use_async && pending_transfer && ((iter + 1) % yieldstep == 0)) {
            double tw0 = omp_get_wtime();
            finish_async_transfer(&D, &buf[1 - current_buf], n);
            t_wait += omp_get_wtime() - tw0;
            pending_transfer = 0;
            // Here you would process buf[1 - current_buf] (write to file, etc.)
        }

        // Run compute kernels
        extrapolate_to_edges_gpu(&D);
        compute_fluxes_gpu(&D);
        protect_gpu(&D);
        update_gpu(&D, dt);
        sim_time += dt;

        // Yieldstep: initiate transfer
        if ((iter + 1) % yieldstep == 0) {
            yield_count++;

            if (use_async) {
                // Start async transfer to current buffer
                double tt0 = omp_get_wtime();
                start_async_transfer(&D, &buf[current_buf], sim_time, n);
                t_transfer += omp_get_wtime() - tt0;
                pending_transfer = 1;
                current_buf = 1 - current_buf;  // Swap buffers
            } else {
                // Synchronous transfer
                sync_transfer(&D, &buf[current_buf], sim_time, n, &t_transfer);
            }
        }

        if ((iter + 1) % 100 == 0 || iter == niter - 1) {
            print_progress(iter + 1, niter, omp_get_wtime() - t0, yield_count);
        }
    }

    // Wait for final pending transfer
    if (use_async && pending_transfer) {
        double tw0 = omp_get_wtime();
        finish_async_transfer(&D, &buf[1 - current_buf], n);
        t_wait += omp_get_wtime() - tw0;
    }

    double t_compute_total = omp_get_wtime() - t0;
    printf("\n\n");

    // Cleanup GPU
    #pragma omp target exit data \
        map(delete: D.stage_centroid_values[0:n], D.xmom_centroid_values[0:n], \
                    D.ymom_centroid_values[0:n], D.bed_centroid_values[0:n], \
                    D.height_centroid_values[0:n], D.stage_explicit_update[0:n], \
                    D.xmom_explicit_update[0:n], D.ymom_explicit_update[0:n], \
                    D.stage_edge_values[0:3*n], D.height_edge_values[0:3*n], \
                    D.neighbours[0:3*n], D.edgelengths[0:3*n], D.normals[0:6*n], \
                    D.areas[0:n], D.radii[0:n], D.max_speed[0:n])

    // Results
    printf("Timing breakdown:\n");
    printf("  Transfer to GPU:     %8.4f ms\n", t_to_gpu * 1000);
    printf("  Total loop time:     %8.4f ms\n", t_compute_total * 1000);
    if (use_async) {
        printf("  Async transfer init: %8.4f ms\n", t_transfer * 1000);
        printf("  Async wait time:     %8.4f ms\n", t_wait * 1000);
        printf("  Transfer overhead:   %8.4f ms (init + wait)\n", (t_transfer + t_wait) * 1000);
    } else {
        printf("  Sync transfer time:  %8.4f ms\n", t_transfer * 1000);
    }
    printf("  Per-yield transfer:  %8.4f ms\n",
           (t_transfer + t_wait) * 1000 / (yield_count > 0 ? yield_count : 1));
    printf("  --------------------------------\n");
    double t_pure_compute = t_compute_total - t_transfer - t_wait;
    printf("  Pure compute:        %8.4f ms (%.4f ms/iter)\n",
           t_pure_compute * 1000, t_pure_compute * 1000 / niter);

    // Get last valid buffer
    int last_buf = use_async ? (1 - current_buf) : current_buf;
    if (buf[last_buf].ready) {
        printf("\nLast output (t=%.4f): mass=%.6f, max_speed=%.4f\n",
               buf[last_buf].time, buf[last_buf].total_mass, buf[last_buf].max_speed_scalar);
    }

    // Cleanup
    free(D.stage_centroid_values); free(D.xmom_centroid_values);
    free(D.ymom_centroid_values); free(D.bed_centroid_values);
    free(D.height_centroid_values); free(D.stage_explicit_update);
    free(D.xmom_explicit_update); free(D.ymom_explicit_update);
    free(D.stage_edge_values); free(D.height_edge_values);
    free(D.neighbours); free(D.edgelengths); free(D.normals);
    free(D.areas); free(D.radii); free(D.max_speed);
    free_output_buffer(&buf[0]);
    free_output_buffer(&buf[1]);

    return 0;
}
