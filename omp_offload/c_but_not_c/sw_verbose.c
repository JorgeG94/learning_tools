// sw_verbose.c - Shallow water mini app with VERBOSE output transfers
// Transfers many arrays at each yieldstep to simulate:
// - Full visualization dumps
// - Debugging with all intermediate quantities
// - Model coupling requiring full state
// This creates enough transfer overhead to potentially benefit from async

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

struct domain {
    int number_of_elements;

    // Centroid values (per triangle)
    double *stage_centroid_values;
    double *xmom_centroid_values;
    double *ymom_centroid_values;
    double *bed_centroid_values;
    double *height_centroid_values;

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

// Verbose output - transfers MANY arrays
struct verbose_output {
    // Primary quantities
    double *stage;
    double *height;
    double *xmom;
    double *ymom;

    // Derived quantities
    double *xvel;
    double *yvel;
    double *speed;           // sqrt(xvel^2 + yvel^2)
    double *froude;          // speed / sqrt(g*h)

    // Flux/update info (useful for debugging)
    double *stage_update;
    double *xmom_update;
    double *ymom_update;

    // Edge values (3x size)
    double *stage_edge;
    double *height_edge;

    // Per-element diagnostics
    double *max_speed_elem;

    // Scalars
    double time;
    double total_mass;
    double total_momentum_x;
    double total_momentum_y;
    double max_speed;
    double max_froude;
    double min_height;
    double max_height;
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

// Compute derived quantities on GPU before transfer
static void compute_derived_quantities_gpu(struct domain *D, struct verbose_output *out) {
    int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;

    double *xmom = D->xmom_centroid_values;
    double *ymom = D->ymom_centroid_values;
    double *height = D->height_centroid_values;
    double *max_speed_elem = D->max_speed;

    double *xvel = out->xvel;
    double *yvel = out->yvel;
    double *speed = out->speed;
    double *froude = out->froude;

    // Compute velocity, speed, froude on GPU and transfer
    #pragma omp target teams distribute parallel for \
        map(from: xvel[0:n], yvel[0:n], speed[0:n], froude[0:n])
    for (int k = 0; k < n; k++) {
        double h = height[k];
        if (h > epsilon) {
            double u = xmom[k] / h;
            double v = ymom[k] / h;
            double spd = sqrt(u*u + v*v);
            double c = sqrt(g * h);

            xvel[k] = u;
            yvel[k] = v;
            speed[k] = spd;
            froude[k] = spd / c;
        } else {
            xvel[k] = 0.0;
            yvel[k] = 0.0;
            speed[k] = 0.0;
            froude[k] = 0.0;
        }
    }
}

// Compute all scalar diagnostics via reductions
static void compute_diagnostics_gpu(struct domain *D, struct verbose_output *out) {
    int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;

    double *height = D->height_centroid_values;
    double *xmom = D->xmom_centroid_values;
    double *ymom = D->ymom_centroid_values;
    double *areas = D->areas;
    double *max_speed_arr = D->max_speed;

    double total_mass = 0.0;
    double total_xmom = 0.0;
    double total_ymom = 0.0;
    double max_spd = 0.0;
    double max_fr = 0.0;
    double min_h = 1e100;
    double max_h = 0.0;

    #pragma omp target teams distribute parallel for \
        reduction(+: total_mass, total_xmom, total_ymom) \
        reduction(max: max_spd, max_fr, max_h) \
        reduction(min: min_h)
    for (int k = 0; k < n; k++) {
        double h = height[k];
        double area = areas[k];

        total_mass += h * area;
        total_xmom += xmom[k] * area;
        total_ymom += ymom[k] * area;

        max_spd = fmax(max_spd, max_speed_arr[k]);

        if (h > epsilon) {
            double u = xmom[k] / h;
            double v = ymom[k] / h;
            double spd = sqrt(u*u + v*v);
            double c = sqrt(g * h);
            max_fr = fmax(max_fr, spd / c);
        }

        min_h = fmin(min_h, h);
        max_h = fmax(max_h, h);
    }

    out->total_mass = total_mass;
    out->total_momentum_x = total_xmom;
    out->total_momentum_y = total_ymom;
    out->max_speed = max_spd;
    out->max_froude = max_fr;
    out->min_height = min_h;
    out->max_height = max_h;
}

// VERBOSE transfer - transfers MANY arrays
static void transfer_verbose_output(struct domain *D, struct verbose_output *out,
                                    double sim_time, double *t_transfer_accum) {
    int n = D->number_of_elements;
    double t0 = omp_get_wtime();

    // Transfer primary quantities (4 arrays)
    #pragma omp target update from(D->stage_centroid_values[0:n])
    #pragma omp target update from(D->height_centroid_values[0:n])
    #pragma omp target update from(D->xmom_centroid_values[0:n])
    #pragma omp target update from(D->ymom_centroid_values[0:n])

    // Copy to output struct
    for (int k = 0; k < n; k++) {
        out->stage[k] = D->stage_centroid_values[k];
        out->height[k] = D->height_centroid_values[k];
        out->xmom[k] = D->xmom_centroid_values[k];
        out->ymom[k] = D->ymom_centroid_values[k];
    }

    // Transfer update/flux arrays (3 arrays) - useful for debugging
    #pragma omp target update from(D->stage_explicit_update[0:n])
    #pragma omp target update from(D->xmom_explicit_update[0:n])
    #pragma omp target update from(D->ymom_explicit_update[0:n])

    for (int k = 0; k < n; k++) {
        out->stage_update[k] = D->stage_explicit_update[k];
        out->xmom_update[k] = D->xmom_explicit_update[k];
        out->ymom_update[k] = D->ymom_explicit_update[k];
    }

    // Transfer edge values (2 arrays × 3 = 6n values)
    #pragma omp target update from(D->stage_edge_values[0:3*n])
    #pragma omp target update from(D->height_edge_values[0:3*n])

    for (int k = 0; k < 3*n; k++) {
        out->stage_edge[k] = D->stage_edge_values[k];
        out->height_edge[k] = D->height_edge_values[k];
    }

    // Transfer max_speed per element
    #pragma omp target update from(D->max_speed[0:n])
    for (int k = 0; k < n; k++) {
        out->max_speed_elem[k] = D->max_speed[k];
    }

    // Compute and transfer derived quantities (4 arrays computed on GPU)
    compute_derived_quantities_gpu(D, out);

    // Compute scalar diagnostics (reductions - cheap)
    compute_diagnostics_gpu(D, out);

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

    double mb_per_array = (double)(n * sizeof(double)) / (1024 * 1024);
    double mb_per_edge_array = (double)(3 * n * sizeof(double)) / (1024 * 1024);

    printf("=== SW_VERBOSE: Heavy output transfers ===\n");
    printf("Grid size: %d x %d\n", grid_size, grid_size);
    printf("Number of triangular elements: %d\n", n);
    printf("Iterations: %d, Yieldstep: %d (yields: %d)\n\n", niter, yieldstep, num_yields);

    printf("Arrays transferred at each yieldstep:\n");
    printf("  Centroid arrays (n elements each):\n");
    printf("    - stage, height, xmom, ymom      : 4 × %.2f MB = %.2f MB\n", mb_per_array, 4*mb_per_array);
    printf("    - stage_update, xmom_update, ymom_update: 3 × %.2f MB = %.2f MB\n", mb_per_array, 3*mb_per_array);
    printf("    - xvel, yvel, speed, froude      : 4 × %.2f MB = %.2f MB\n", mb_per_array, 4*mb_per_array);
    printf("    - max_speed_elem                 : 1 × %.2f MB = %.2f MB\n", mb_per_array, mb_per_array);
    printf("  Edge arrays (3n elements each):\n");
    printf("    - stage_edge, height_edge        : 2 × %.2f MB = %.2f MB\n", mb_per_edge_array, 2*mb_per_edge_array);
    printf("  -----------------------------------------\n");
    double total_mb = 12 * mb_per_array + 2 * mb_per_edge_array;
    printf("  TOTAL per yieldstep:                 %.2f MB\n\n", total_mb);

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

    // Allocate verbose output
    struct verbose_output out;
    out.stage = (double *)malloc(n * sizeof(double));
    out.height = (double *)malloc(n * sizeof(double));
    out.xmom = (double *)malloc(n * sizeof(double));
    out.ymom = (double *)malloc(n * sizeof(double));
    out.xvel = (double *)malloc(n * sizeof(double));
    out.yvel = (double *)malloc(n * sizeof(double));
    out.speed = (double *)malloc(n * sizeof(double));
    out.froude = (double *)malloc(n * sizeof(double));
    out.stage_update = (double *)malloc(n * sizeof(double));
    out.xmom_update = (double *)malloc(n * sizeof(double));
    out.ymom_update = (double *)malloc(n * sizeof(double));
    out.stage_edge = (double *)malloc(3 * n * sizeof(double));
    out.height_edge = (double *)malloc(3 * n * sizeof(double));
    out.max_speed_elem = (double *)malloc(n * sizeof(double));

    if (!D.stage_centroid_values || !D.neighbours || !out.stage) {
        fprintf(stderr, "Error: Memory allocation failed\n");
        return 1;
    }

    // Initialize on host
    double t0 = omp_get_wtime();
    generate_mesh(&D, grid_size);
    init_quantities(&D);
    double t_init = omp_get_wtime() - t0;

    // Transfer to GPU
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

    // Run iterations
    double dt = 0.001;
    double sim_time = 0.0;
    double t_yield_transfer = 0.0;
    int yield_count = 0;

    printf("Computing with VERBOSE output every %d iterations:\n", yieldstep);

    t0 = omp_get_wtime();
    for (int iter = 0; iter < niter; iter++) {
        extrapolate_to_edges_gpu(&D);
        compute_fluxes_gpu(&D);
        protect_gpu(&D);
        update_gpu(&D, dt);
        sim_time += dt;

        // Verbose yieldstep transfer
        if ((iter + 1) % yieldstep == 0) {
            transfer_verbose_output(&D, &out, sim_time, &t_yield_transfer);
            yield_count++;
        }

        if ((iter + 1) % 100 == 0 || iter == niter - 1) {
            print_progress(iter + 1, niter, omp_get_wtime() - t0, yield_count);
        }
    }
    double t_compute_total = omp_get_wtime() - t0;
    double t_compute_pure = t_compute_total - t_yield_transfer;
    printf("\n\n");

    // Final cleanup
    t0 = omp_get_wtime();
    #pragma omp target exit data \
        map(delete: D.stage_centroid_values[0:n], \
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
    double t_from_gpu = omp_get_wtime() - t0;

    double t_transfer_total = t_to_gpu + t_yield_transfer + t_from_gpu;
    double t_total = t_init + t_to_gpu + t_compute_total + t_from_gpu;

    // Results
    printf("Timing breakdown:\n");
    printf("  Host init:           %8.4f ms\n", t_init * 1000);
    printf("  Transfer to GPU:     %8.4f ms\n", t_to_gpu * 1000);
    printf("  Pure compute:        %8.4f ms (%.4f ms/iter)\n",
           t_compute_pure * 1000, t_compute_pure * 1000 / niter);
    printf("  Yieldstep transfers: %8.4f ms (%d yields, %.4f ms/yield)\n",
           t_yield_transfer * 1000, yield_count,
           yield_count > 0 ? t_yield_transfer * 1000 / yield_count : 0.0);
    printf("  Cleanup:             %8.4f ms\n", t_from_gpu * 1000);
    printf("  --------------------------------\n");
    printf("  Total:               %8.4f ms\n", t_total * 1000);
    printf("  Transfer/Total:      %8.4f %%\n", 100.0 * t_transfer_total / t_total);
    printf("  Yieldstep overhead:  %8.4f %% of compute time\n",
           100.0 * t_yield_transfer / t_compute_pure);

    // Data transfer rate
    double yield_data_mb = total_mb * yield_count;
    double yield_bandwidth = yield_data_mb / (t_yield_transfer) / 1000.0;  // GB/s
    printf("\nYieldstep transfer stats:\n");
    printf("  Total data transferred: %.2f MB (%d yields × %.2f MB)\n",
           yield_data_mb, yield_count, total_mb);
    printf("  Effective PCIe bandwidth: %.2f GB/s\n", yield_bandwidth);

    // Bandwidth estimate
    double bytes_per_iter = (double)n * sizeof(double) * 32;
    double bandwidth = (niter * bytes_per_iter) / t_compute_pure / 1e9;
    printf("  GPU memory bandwidth: %.2f GB/s\n", bandwidth);

    // Last output
    printf("\nLast yieldstep diagnostics (t=%.4f):\n", out.time);
    printf("  Total mass: %.6f (conservation check)\n", out.total_mass);
    printf("  Max speed: %.6f, Max Froude: %.6f\n", out.max_speed, out.max_froude);
    printf("  Height range: [%.6f, %.6f]\n", out.min_height, out.max_height);

    // Cleanup
    free(D.stage_centroid_values); free(D.xmom_centroid_values);
    free(D.ymom_centroid_values); free(D.bed_centroid_values);
    free(D.height_centroid_values); free(D.stage_explicit_update);
    free(D.xmom_explicit_update); free(D.ymom_explicit_update);
    free(D.stage_edge_values); free(D.height_edge_values);
    free(D.neighbours); free(D.edgelengths); free(D.normals);
    free(D.areas); free(D.radii); free(D.max_speed);
    free(out.stage); free(out.height); free(out.xmom); free(out.ymom);
    free(out.xvel); free(out.yvel); free(out.speed); free(out.froude);
    free(out.stage_update); free(out.xmom_update); free(out.ymom_update);
    free(out.stage_edge); free(out.height_edge); free(out.max_speed_elem);

    return 0;
}
