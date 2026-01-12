// sw_yieldstep.c - Shallow water mini app with PERIODIC data transfers
// Data is transferred to GPU once, kernels run, but every "yieldstep"
// iterations we transfer back quantities needed for output/visualization.
// This represents realistic simulation patterns.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

struct domain {
    int number_of_elements;

    // Centroid values (per triangle)
    double *stage_centroid_values;    // Water surface elevation (h + z)
    double *xmom_centroid_values;     // x-momentum (h * u)
    double *ymom_centroid_values;     // y-momentum (h * v)
    double *bed_centroid_values;      // Bed elevation (z) - static
    double *height_centroid_values;   // Water depth (h = stage - bed)

    // Explicit update arrays
    double *stage_explicit_update;
    double *xmom_explicit_update;
    double *ymom_explicit_update;

    // Edge values (3 per triangle)
    double *stage_edge_values;
    double *height_edge_values;

    // Mesh connectivity
    int *neighbours;
    double *edgelengths;
    double *normals;
    double *areas;
    double *radii;
    double *max_speed;

    // Constants
    double g;
    double epsilon;
    double minimum_allowed_height;
};

// Quantities we transfer back at each yieldstep
struct output_quantities {
    double *stage;      // Water surface elevation
    double *height;     // Water depth
    double *xvel;       // x-velocity (computed from xmom/height)
    double *yvel;       // y-velocity (computed from ymom/height)
    double time;        // Current simulation time
    double total_mass;  // For conservation check
    double max_speed;   // For CFL info
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

// Compute fluxes - NO map clauses
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

    #pragma omp target teams distribute parallel for \
        reduction(min: local_timestep)
    for (int k = 0; k < n; k++) {
        double stage_accum = 0.0;
        double xmom_accum = 0.0;
        double ymom_accum = 0.0;
        double speed_max = 0.0;

        double uh_k = xmom_centroid[k];
        double vh_k = ymom_centroid[k];

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

            double h_diff = h_left - h_right;
            stage_accum += c_max * h_diff * edgelen;
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

static double protect_gpu(struct domain *D) {
    int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;
    double mass_error = 0.0;

    double *stage_centroid = D->stage_centroid_values;
    double *bed_centroid = D->bed_centroid_values;
    double *xmom_centroid = D->xmom_centroid_values;
    double *ymom_centroid = D->ymom_centroid_values;
    double *height_centroid = D->height_centroid_values;
    double *areas = D->areas;

    #pragma omp target teams distribute parallel for \
        reduction(+: mass_error)
    for (int k = 0; k < n; k++) {
        double hc = stage_centroid[k] - bed_centroid[k];

        if (hc < minimum_allowed_height) {
            xmom_centroid[k] = 0.0;
            ymom_centroid[k] = 0.0;

            if (hc <= 0.0) {
                double bmin = bed_centroid[k];
                if (stage_centroid[k] < bmin) {
                    mass_error += (bmin - stage_centroid[k]) * areas[k];
                    stage_centroid[k] = bmin;
                }
            }
        }
        height_centroid[k] = fmax(stage_centroid[k] - bed_centroid[k], 0.0);
    }

    return mass_error;
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

// Compute velocities on GPU (derived quantity)
static void compute_velocities_gpu(struct domain *D, double *xvel, double *yvel) {
    int n = D->number_of_elements;
    double epsilon = D->epsilon;

    double *xmom = D->xmom_centroid_values;
    double *ymom = D->ymom_centroid_values;
    double *height = D->height_centroid_values;

    #pragma omp target teams distribute parallel for \
        map(from: xvel[0:n], yvel[0:n])
    for (int k = 0; k < n; k++) {
        double h = height[k];
        if (h > epsilon) {
            xvel[k] = xmom[k] / h;
            yvel[k] = ymom[k] / h;
        } else {
            xvel[k] = 0.0;
            yvel[k] = 0.0;
        }
    }
}

// Compute diagnostics on GPU (reductions - very cheap to transfer back)
static void compute_diagnostics_gpu(struct domain *D, double *total_mass, double *max_spd) {
    int n = D->number_of_elements;
    double mass = 0.0;
    double spd = 0.0;

    double *height = D->height_centroid_values;
    double *areas = D->areas;
    double *max_speed = D->max_speed;

    #pragma omp target teams distribute parallel for \
        reduction(+: mass) reduction(max: spd)
    for (int k = 0; k < n; k++) {
        mass += height[k] * areas[k];
        spd = fmax(spd, max_speed[k]);
    }

    *total_mass = mass;
    *max_spd = spd;
}

// Transfer output quantities from GPU to host
// This is what happens at each "yieldstep" in ANUGA
static void transfer_output_quantities(struct domain *D, struct output_quantities *out,
                                       double sim_time, double *t_transfer_accum) {
    int n = D->number_of_elements;
    double t0 = omp_get_wtime();

    // Transfer stage and height (the primary quantities for visualization)
    #pragma omp target update from(D->stage_centroid_values[0:n])
    #pragma omp target update from(D->height_centroid_values[0:n])

    // Copy to output struct (in real code, you'd write to file here)
    for (int k = 0; k < n; k++) {
        out->stage[k] = D->stage_centroid_values[k];
        out->height[k] = D->height_centroid_values[k];
    }

    // Compute and transfer velocities (derived quantities)
    compute_velocities_gpu(D, out->xvel, out->yvel);

    // Compute diagnostics (reductions on GPU, only scalars come back)
    compute_diagnostics_gpu(D, &out->total_mass, &out->max_speed);

    out->time = sim_time;

    *t_transfer_accum += omp_get_wtime() - t0;
}

int main(int argc, char *argv[]) {
    if (argc < 2 || argc > 4) {
        fprintf(stderr, "Usage: %s N [niter] [yieldstep]\n", argv[0]);
        fprintf(stderr, "  N         = grid size (creates 2*(N-1)^2 triangles)\n");
        fprintf(stderr, "  niter     = number of iterations (default: 1000)\n");
        fprintf(stderr, "  yieldstep = transfer output every N iterations (default: 100)\n");
        return 1;
    }

    int grid_size = atoi(argv[1]);
    int niter = (argc >= 3) ? atoi(argv[2]) : 1000;
    int yieldstep = (argc >= 4) ? atoi(argv[3]) : 100;

    if (grid_size < 3) {
        fprintf(stderr, "Error: Grid size must be at least 3\n");
        return 1;
    }

    int n = 2 * (grid_size - 1) * (grid_size - 1);
    int num_yields = niter / yieldstep;

    printf("=== SW_YIELDSTEP: Periodic output transfers ===\n");
    printf("Grid size: %d x %d\n", grid_size, grid_size);
    printf("Number of triangular elements: %d\n", n);
    printf("Memory per centroid array: %.2f MB\n",
           (double)(n * sizeof(double)) / (1024 * 1024));
    printf("Iterations: %d, Yieldstep: %d (yields: %d)\n\n", niter, yieldstep, num_yields);

    printf("Output quantities transferred at each yieldstep:\n");
    printf("  - stage (%.2f MB)\n", (double)(n * sizeof(double)) / (1024 * 1024));
    printf("  - height (%.2f MB)\n", (double)(n * sizeof(double)) / (1024 * 1024));
    printf("  - xvel, yvel (%.2f MB each, computed on GPU)\n", (double)(n * sizeof(double)) / (1024 * 1024));
    printf("  - total_mass, max_speed (scalars, from reductions)\n");
    printf("  Total per yieldstep: %.2f MB\n\n", (double)(4 * n * sizeof(double)) / (1024 * 1024));

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

    // Allocate output quantities
    struct output_quantities out;
    out.stage = (double *)malloc(n * sizeof(double));
    out.height = (double *)malloc(n * sizeof(double));
    out.xvel = (double *)malloc(n * sizeof(double));
    out.yvel = (double *)malloc(n * sizeof(double));

    if (!D.stage_centroid_values || !D.neighbours || !out.stage) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }

    // Initialize on host
    double t0 = omp_get_wtime();
    generate_mesh(&D, grid_size);
    init_quantities(&D);
    double t_init = omp_get_wtime() - t0;

    // Transfer ALL data to GPU once
    printf("Transferring data to GPU...\n");
    t0 = omp_get_wtime();
    #pragma omp target enter data \
        map(to: D.stage_centroid_values[0:n], \
                D.xmom_centroid_values[0:n], \
                D.ymom_centroid_values[0:n], \
                D.bed_centroid_values[0:n], \
                D.height_centroid_values[0:n], \
                D.stage_explicit_update[0:n], \
                D.xmom_explicit_update[0:n], \
                D.ymom_explicit_update[0:n], \
                D.stage_edge_values[0:3*n], \
                D.height_edge_values[0:3*n], \
                D.neighbours[0:3*n], \
                D.edgelengths[0:3*n], \
                D.normals[0:6*n], \
                D.areas[0:n], \
                D.radii[0:n], \
                D.max_speed[0:n])
    double t_to_gpu = omp_get_wtime() - t0;

    // Run iterations with periodic yieldstep transfers
    double dt = 0.001;
    double sim_time = 0.0;
    double t_yield_transfer = 0.0;  // Accumulated yieldstep transfer time
    int yield_count = 0;

    printf("Computing with yieldstep=%d:\n", yieldstep);

    t0 = omp_get_wtime();
    for (int iter = 0; iter < niter; iter++) {
        extrapolate_to_edges_gpu(&D);
        compute_fluxes_gpu(&D);
        protect_gpu(&D);
        update_gpu(&D, dt);
        sim_time += dt;

        // Yieldstep: transfer output quantities
        if ((iter + 1) % yieldstep == 0) {
            transfer_output_quantities(&D, &out, sim_time, &t_yield_transfer);
            yield_count++;

            // In real code, you'd write to file or process the data here
            // printf("  Yield %d: t=%.4f, mass=%.6f, max_speed=%.4f\n",
            //        yield_count, out.time, out.total_mass, out.max_speed);
        }

        if ((iter + 1) % 100 == 0 || iter == niter - 1) {
            print_progress(iter + 1, niter, omp_get_wtime() - t0, yield_count);
        }
    }
    double t_compute_total = omp_get_wtime() - t0;
    double t_compute_pure = t_compute_total - t_yield_transfer;
    printf("\n\n");

    // Final transfer (anything not already transferred)
    t0 = omp_get_wtime();
    #pragma omp target exit data \
        map(from: D.stage_centroid_values[0:n]) \
        map(delete: D.xmom_centroid_values[0:n], \
                    D.ymom_centroid_values[0:n], \
                    D.bed_centroid_values[0:n], \
                    D.height_centroid_values[0:n], \
                    D.stage_explicit_update[0:n], \
                    D.xmom_explicit_update[0:n], \
                    D.ymom_explicit_update[0:n], \
                    D.stage_edge_values[0:3*n], \
                    D.height_edge_values[0:3*n], \
                    D.neighbours[0:3*n], \
                    D.edgelengths[0:3*n], \
                    D.normals[0:6*n], \
                    D.areas[0:n], \
                    D.radii[0:n], \
                    D.max_speed[0:n])
    double t_from_gpu = omp_get_wtime() - t0;

    double t_transfer_total = t_to_gpu + t_yield_transfer + t_from_gpu;
    double t_total = t_init + t_to_gpu + t_compute_total + t_from_gpu;

    // Timing results
    printf("Timing breakdown:\n");
    printf("  Host init:           %8.4f ms\n", t_init * 1000);
    printf("  Transfer to GPU:     %8.4f ms\n", t_to_gpu * 1000);
    printf("  Pure compute:        %8.4f ms (%.4f ms/iter)\n",
           t_compute_pure * 1000, t_compute_pure * 1000 / niter);
    printf("  Yieldstep transfers: %8.4f ms (%d yields, %.4f ms/yield)\n",
           t_yield_transfer * 1000, yield_count,
           yield_count > 0 ? t_yield_transfer * 1000 / yield_count : 0.0);
    printf("  Transfer from GPU:   %8.4f ms\n", t_from_gpu * 1000);
    printf("  --------------------------------\n");
    printf("  Total:               %8.4f ms\n", t_total * 1000);
    printf("  Transfer/Total:      %8.4f %%\n", 100.0 * t_transfer_total / t_total);
    printf("  Yieldstep overhead:  %8.4f %% of compute time\n",
           100.0 * t_yield_transfer / t_compute_pure);

    // Bandwidth estimate
    double bytes_per_iter = (double)n * sizeof(double) * 32;
    double bandwidth = (niter * bytes_per_iter) / t_compute_pure / 1e9;
    printf("\nEffective GPU memory bandwidth: %.2f GB/s\n", bandwidth);

    // Last output quantities
    printf("\nLast yieldstep output (t=%.4f):\n", out.time);
    printf("  Total mass: %.6f\n", out.total_mass);
    printf("  Max speed:  %.6f\n", out.max_speed);
    printf("  stage[0] = %.6f, stage[n-1] = %.6f\n", out.stage[0], out.stage[n-1]);

    // Cleanup
    free(D.stage_centroid_values);
    free(D.xmom_centroid_values);
    free(D.ymom_centroid_values);
    free(D.bed_centroid_values);
    free(D.height_centroid_values);
    free(D.stage_explicit_update);
    free(D.xmom_explicit_update);
    free(D.ymom_explicit_update);
    free(D.stage_edge_values);
    free(D.height_edge_values);
    free(D.neighbours);
    free(D.edgelengths);
    free(D.normals);
    free(D.areas);
    free(D.radii);
    free(D.max_speed);
    free(out.stage);
    free(out.height);
    free(out.xvel);
    free(out.yvel);

    return 0;
}
