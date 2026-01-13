// sw_cfl2_mpi.c - Shallow water mini app with MPI + OpenMP target offload
// Second-order extrapolation + HLL flux + domain decomposition
// Now with river delta mode: bathymetry, tide, river discharge, rain

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
//#include <caliper/cali.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct domain {
    int number_of_elements;      // Local elements on this rank
    int global_elements;         // Total elements across all ranks
    int local_start;             // Global index of first local element

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
    double *xmom_edge_values;
    double *ymom_edge_values;

    // Gradients at centroids
    double *stage_x_gradient;
    double *stage_y_gradient;
    double *xmom_x_gradient;
    double *xmom_y_gradient;
    double *ymom_x_gradient;
    double *ymom_y_gradient;
    double *height_x_gradient;
    double *height_y_gradient;

    // Mesh geometry
    double *centroid_x;
    double *centroid_y;
    double *edge_midpoint_x;
    double *edge_midpoint_y;

    // Mesh connectivity (local indices, -1 for boundary, -2-rank for remote)
    int *neighbours;
    int *neighbour_owners;       // Which rank owns each neighbour (-1 for local)
    double *edgelengths;
    double *normals;
    double *areas;
    double *radii;
    double *max_speed;

    // Constants
    double g;
    double epsilon;
    double minimum_allowed_height;
    double cfl;
    double domain_length;
    double char_length;
};

// Halo exchange structure
struct halo_info {
    int num_neighbors;           // Number of MPI neighbors
    int *neighbor_ranks;         // Ranks we communicate with
    int *send_counts;            // Elements to send to each neighbor
    int *recv_counts;            // Elements to receive from each neighbor
    int **send_indices;          // Local indices to send
    int **recv_indices;          // Local indices where received data goes

    // Buffers for edge data (4 doubles per edge: h, uh, vh, stage)
    double *send_buffer;
    double *recv_buffer;
    int send_buffer_size;
    int recv_buffer_size;
};

// Verbose output structure (for yieldstep transfers)
struct verbose_output {
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
};

// MPI globals
static int mpi_rank, mpi_size;

// Delta configuration
struct delta_config {
    int enabled;

    // Bathymetry
    double ocean_depth;       // Depth at ocean side (m)
    double land_height;       // Max land elevation (m)
    double channel_depth;     // River channel depth below land (m)
    double channel_width;     // River channel width (fraction of domain, 0-1)

    // Tide
    double tide_amplitude;    // Tidal amplitude (m)
    double tide_period;       // Tidal period (s), ~12.4 hrs for semi-diurnal
    double mean_sea_level;    // Mean sea level (m)

    // River discharge
    double river_discharge;   // Total discharge (m³/s)
    double river_velocity;    // Inflow velocity (m/s)

    // Rain
    double rain_rate;         // Rain rate (m/s), e.g., 50mm/hr = 1.4e-5 m/s
};

static void print_progress(int current, int total, double elapsed, double sim_time, int yields) {
    if (mpi_rank != 0) return;

    int bar_width = 40;
    float progress = (float)current / total;
    int filled = (int)(bar_width * progress);

    printf("\r  [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) printf("=");
        else if (i == filled) printf(">");
        else printf(" ");
    }
    printf("] %3d%% (%d/%d) %.2fs sim_t=%.4fs [%d yields]",
           (int)(progress * 100), current, total, elapsed, sim_time, yields);
    fflush(stdout);
}

// Transfer verbose output from GPU
// mode 0 = distributed (each rank keeps local data - scalable)
// mode 1 = gather (all data to rank 0 - for debugging)
static void transfer_yieldstep(struct domain *D, struct verbose_output *out,
                                double *t_transfer, int gather_mode) {
    int n = D->number_of_elements;
    double t0 = omp_get_wtime();
    double g = D->g;
    double h0 = D->minimum_allowed_height;

    // Transfer all centroid arrays in one pragma
    #pragma omp target update from( \
        D->stage_centroid_values[0:n], \
        D->height_centroid_values[0:n], \
        D->xmom_centroid_values[0:n], \
        D->ymom_centroid_values[0:n], \
        D->stage_explicit_update[0:n], \
        D->xmom_explicit_update[0:n], \
        D->ymom_explicit_update[0:n], \
        D->max_speed[0:n])

    // Transfer edge arrays
    #pragma omp target update from( \
        D->stage_edge_values[0:3*n], \
        D->height_edge_values[0:3*n])

    // Copy to output and compute derived quantities in one loop
    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        double h = D->height_centroid_values[k];
        double uh = D->xmom_centroid_values[k];
        double vh = D->ymom_centroid_values[k];

        out->stage[k] = D->stage_centroid_values[k];
        out->height[k] = h;
        out->xmom[k] = uh;
        out->ymom[k] = vh;
        out->stage_update[k] = D->stage_explicit_update[k];
        out->xmom_update[k] = D->xmom_explicit_update[k];
        out->ymom_update[k] = D->ymom_explicit_update[k];
        out->max_speed_elem[k] = D->max_speed[k];

        if (h > h0) {
            double u = uh / h;
            double v = vh / h;
            double spd = sqrt(u*u + v*v);
            double c = sqrt(g * h);
            out->xvel[k] = u;
            out->yvel[k] = v;
            out->speed[k] = spd;
            out->froude[k] = spd / c;
        } else {
            out->xvel[k] = 0.0;
            out->yvel[k] = 0.0;
            out->speed[k] = 0.0;
            out->froude[k] = 0.0;
        }
    }

    // Copy edge arrays
    #pragma omp parallel for
    for (int k = 0; k < 3*n; k++) {
        out->stage_edge[k] = D->stage_edge_values[k];
        out->height_edge[k] = D->height_edge_values[k];
    }

    // Optional gather to rank 0 (not scalable, for debugging only)
    if (gather_mode) {
        int n_global = D->global_elements;

        // Gather counts from all ranks
        int *recv_counts = NULL;
        int *displs = NULL;
        double *global_stage = NULL;

        if (mpi_rank == 0) {
            recv_counts = (int *)malloc(mpi_size * sizeof(int));
            displs = (int *)malloc(mpi_size * sizeof(int));
            global_stage = (double *)malloc(n_global * sizeof(double));
        }

        // Gather local counts
        MPI_Gather(&n, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < mpi_size; i++) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }

        // Gather just stage as example (would do all arrays in production)
        MPI_Gatherv(out->stage, n, MPI_DOUBLE,
                    global_stage, recv_counts, displs, MPI_DOUBLE,
                    0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            free(recv_counts);
            free(displs);
            free(global_stage);
        }
    }

    *t_transfer += omp_get_wtime() - t0;
}

// Determine which rank owns a global triangle index
static int get_owner(int global_idx, int n_global, int nprocs) {
    if (global_idx < 0) return -1;  // Boundary
    int base = n_global / nprocs;
    int remainder = n_global % nprocs;

    // Ranks 0..remainder-1 have base+1 elements
    // Ranks remainder..nprocs-1 have base elements
    if (global_idx < (base + 1) * remainder) {
        return global_idx / (base + 1);
    } else {
        return remainder + (global_idx - (base + 1) * remainder) / base;
    }
}

// Get local index from global index for a given rank
static int global_to_local(int global_idx, int n_global, int nprocs, int rank) {
    int base = n_global / nprocs;
    int remainder = n_global % nprocs;
    int start;

    if (rank < remainder) {
        start = rank * (base + 1);
    } else {
        start = remainder * (base + 1) + (rank - remainder) * base;
    }
    return global_idx - start;
}

static void compute_partition(int n_global, int nprocs, int rank, int *start, int *count) {
    int base = n_global / nprocs;
    int remainder = n_global % nprocs;

    if (rank < remainder) {
        *start = rank * (base + 1);
        *count = base + 1;
    } else {
        *start = remainder * (base + 1) + (rank - remainder) * base;
        *count = base;
    }
}

static void generate_mesh_local(struct domain *D, int grid_size) {
    int nx = grid_size;
    int ny = grid_size;
    int n = D->number_of_elements;
    int n_global = D->global_elements;
    int local_start = D->local_start;
    double dx = D->domain_length / (nx - 1);
    double dy = D->domain_length / (ny - 1);

    double area = 0.5 * dx * dy;
    double edgelen = dx;
    double radius = area / (1.5 * edgelen);

    #pragma omp parallel for
    for (int k_local = 0; k_local < n; k_local++) {
        int k_global = local_start + k_local;

        D->areas[k_local] = area;
        D->radii[k_local] = radius;

        for (int i = 0; i < 3; i++) {
            D->edgelengths[3*k_local + i] = edgelen;
        }

        D->normals[6*k_local + 0] = 1.0;  D->normals[6*k_local + 1] = 0.0;
        D->normals[6*k_local + 2] = 0.0;  D->normals[6*k_local + 3] = 1.0;
        D->normals[6*k_local + 4] = -0.707; D->normals[6*k_local + 5] = -0.707;

        int cell = k_global / 2;
        int tri_in_cell = k_global % 2;
        int cell_x = cell % (nx - 1);
        int cell_y = cell / (nx - 1);

        double x0, y0, x1, y1, x2, y2;
        if (tri_in_cell == 0) {
            x0 = cell_x * dx;       y0 = cell_y * dy;
            x1 = (cell_x + 1) * dx; y1 = cell_y * dy;
            x2 = cell_x * dx;       y2 = (cell_y + 1) * dy;
        } else {
            x0 = (cell_x + 1) * dx; y0 = (cell_y + 1) * dy;
            x1 = cell_x * dx;       y1 = (cell_y + 1) * dy;
            x2 = (cell_x + 1) * dx; y2 = cell_y * dy;
        }

        D->centroid_x[k_local] = (x0 + x1 + x2) / 3.0;
        D->centroid_y[k_local] = (y0 + y1 + y2) / 3.0;

        D->edge_midpoint_x[3*k_local + 0] = (x1 + x2) / 2.0;
        D->edge_midpoint_y[3*k_local + 0] = (y1 + y2) / 2.0;
        D->edge_midpoint_x[3*k_local + 1] = (x0 + x2) / 2.0;
        D->edge_midpoint_y[3*k_local + 1] = (y0 + y2) / 2.0;
        D->edge_midpoint_x[3*k_local + 2] = (x0 + x1) / 2.0;
        D->edge_midpoint_y[3*k_local + 2] = (y0 + y1) / 2.0;

        // Compute global neighbor indices
        int nb_global[3];
        if (tri_in_cell == 0) {
            nb_global[0] = k_global + 1;
            nb_global[1] = (cell_y > 0) ? 2 * ((cell_y - 1) * (nx - 1) + cell_x) + 1 : -1;
            nb_global[2] = (cell_x > 0) ? 2 * (cell_y * (nx - 1) + (cell_x - 1)) + 1 : -1;
        } else {
            nb_global[0] = k_global - 1;
            nb_global[1] = (cell_y < ny - 2) ? 2 * ((cell_y + 1) * (nx - 1) + cell_x) : -1;
            nb_global[2] = (cell_x < nx - 2) ? 2 * (cell_y * (nx - 1) + (cell_x + 1)) : -1;
        }

        // Convert to local indices and track owners
        for (int i = 0; i < 3; i++) {
            int ng = nb_global[i];
            if (ng < 0) {
                // Physical boundary
                D->neighbours[3*k_local + i] = -1;
                D->neighbour_owners[3*k_local + i] = -1;
            } else {
                int owner = get_owner(ng, n_global, mpi_size);
                D->neighbour_owners[3*k_local + i] = owner;
                if (owner == mpi_rank) {
                    // Local neighbor
                    D->neighbours[3*k_local + i] = global_to_local(ng, n_global, mpi_size, mpi_rank);
                } else {
                    // Remote neighbor - store global index (will be remapped after halo setup)
                    D->neighbours[3*k_local + i] = ng;
                }
            }
        }
    }
}

// Compute delta bathymetry for a cell
// Layout: Ocean at y=0, river inlet at y=domain_length, land in between
static double compute_delta_bed(double x, double y, double domain_length,
                                 struct delta_config *cfg) {
    double L = domain_length;

    // Normalized coordinates (0 to 1)
    double xn = x / L;
    double yn = y / L;

    // River channel: runs from north (y=L) toward south, centered at x=0.5
    double channel_center = 0.5;
    double channel_hw = cfg->channel_width / 2.0;  // half-width
    int in_channel = (fabs(xn - channel_center) < channel_hw);

    // Base topography: rises from ocean (y=0) to peak at y=0.6L, then drops to river
    double topo;
    if (yn < 0.1) {
        // Ocean zone: flat at -ocean_depth
        topo = -cfg->ocean_depth;
    } else if (yn < 0.5) {
        // Transition from ocean to land
        double t = (yn - 0.1) / 0.4;  // 0 to 1
        topo = -cfg->ocean_depth + (cfg->land_height + cfg->ocean_depth) * t * t;
    } else if (yn < 0.7) {
        // Land plateau with some variation
        double noise = 0.3 * sin(xn * 12.0) * sin(yn * 8.0);  // Small undulation
        topo = cfg->land_height * (1.0 + noise * 0.2);
    } else {
        // Slope down toward river inlet
        double t = (yn - 0.7) / 0.3;  // 0 to 1
        topo = cfg->land_height * (1.0 - t * 0.5);
    }

    // Carve river channel
    if (in_channel && yn > 0.3) {
        // Channel gets deeper toward the inlet
        double channel_factor = (yn - 0.3) / 0.7;  // 0 at y=0.3, 1 at y=1
        double channel_bed = topo - cfg->channel_depth * channel_factor;
        // Smooth transition at channel edges
        double edge_dist = fabs(xn - channel_center) / channel_hw;
        double blend = edge_dist * edge_dist;  // Parabolic channel cross-section
        topo = channel_bed + (topo - channel_bed) * blend;
    }

    // Add some delta distributary channels near the ocean
    if (yn < 0.4 && yn > 0.1) {
        double dist1 = fabs(xn - 0.3);
        double dist2 = fabs(xn - 0.7);
        if (dist1 < 0.05) {
            topo -= cfg->channel_depth * 0.5 * (1.0 - dist1/0.05) * (0.4 - yn) / 0.3;
        }
        if (dist2 < 0.05) {
            topo -= cfg->channel_depth * 0.5 * (1.0 - dist2/0.05) * (0.4 - yn) / 0.3;
        }
    }

    return topo;
}

static void init_quantities(struct domain *D, double initial_height,
                            struct delta_config *cfg) {
    int n = D->number_of_elements;
    double L = D->domain_length;

    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        double x = D->centroid_x[k];
        double y = D->centroid_y[k];

        double bed;
        double stage;

        if (cfg->enabled) {
            // Delta bathymetry
            bed = compute_delta_bed(x, y, L, cfg);
            stage = cfg->mean_sea_level;  // Initial water at mean sea level
        } else {
            // Original flat bottom
            bed = 0.0;
            stage = initial_height;
        }

        D->bed_centroid_values[k] = bed;
        D->stage_centroid_values[k] = stage;
        D->height_centroid_values[k] = fmax(stage - bed, 0.0);
        D->xmom_centroid_values[k] = 0.0;
        D->ymom_centroid_values[k] = 0.0;
        D->stage_explicit_update[k] = 0.0;
        D->xmom_explicit_update[k] = 0.0;
        D->ymom_explicit_update[k] = 0.0;
        D->max_speed[k] = 0.0;

        D->stage_x_gradient[k] = 0.0;
        D->stage_y_gradient[k] = 0.0;
        D->xmom_x_gradient[k] = 0.0;
        D->xmom_y_gradient[k] = 0.0;
        D->ymom_x_gradient[k] = 0.0;
        D->ymom_y_gradient[k] = 0.0;
        D->height_x_gradient[k] = 0.0;
        D->height_y_gradient[k] = 0.0;

        double h = D->height_centroid_values[k];
        for (int i = 0; i < 3; i++) {
            D->stage_edge_values[3*k + i] = stage;
            D->height_edge_values[3*k + i] = h;
            D->xmom_edge_values[3*k + i] = 0.0;
            D->ymom_edge_values[3*k + i] = 0.0;
        }
    }
}

// Build halo exchange info
static void build_halo_info(struct domain *D, struct halo_info *halo) {
    int n = D->number_of_elements;
    int n_global = D->global_elements;

    // Count neighbors on each remote rank
    int *rank_counts = (int *)calloc(mpi_size, sizeof(int));

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < 3; i++) {
            int owner = D->neighbour_owners[3*k + i];
            if (owner >= 0 && owner != mpi_rank) {
                rank_counts[owner]++;
            }
        }
    }

    // Count how many ranks we communicate with
    halo->num_neighbors = 0;
    for (int r = 0; r < mpi_size; r++) {
        if (rank_counts[r] > 0) halo->num_neighbors++;
    }

    if (halo->num_neighbors == 0) {
        halo->neighbor_ranks = NULL;
        halo->send_counts = NULL;
        halo->recv_counts = NULL;
        halo->send_indices = NULL;
        halo->recv_indices = NULL;
        halo->send_buffer = NULL;
        halo->recv_buffer = NULL;
        free(rank_counts);
        return;
    }

    halo->neighbor_ranks = (int *)malloc(halo->num_neighbors * sizeof(int));
    halo->send_counts = (int *)malloc(halo->num_neighbors * sizeof(int));
    halo->recv_counts = (int *)malloc(halo->num_neighbors * sizeof(int));
    halo->send_indices = (int **)malloc(halo->num_neighbors * sizeof(int *));
    halo->recv_indices = (int **)malloc(halo->num_neighbors * sizeof(int *));

    int idx = 0;
    for (int r = 0; r < mpi_size; r++) {
        if (rank_counts[r] > 0) {
            halo->neighbor_ranks[idx] = r;
            halo->recv_counts[idx] = rank_counts[r];
            halo->recv_indices[idx] = (int *)malloc(rank_counts[r] * sizeof(int));
            idx++;
        }
    }

    // Fill recv_indices with local edge indices that need remote data
    int *fill_idx = (int *)calloc(halo->num_neighbors, sizeof(int));

    for (int k = 0; k < n; k++) {
        for (int i = 0; i < 3; i++) {
            int owner = D->neighbour_owners[3*k + i];
            if (owner >= 0 && owner != mpi_rank) {
                // Find which neighbor index this corresponds to
                for (int ni = 0; ni < halo->num_neighbors; ni++) {
                    if (halo->neighbor_ranks[ni] == owner) {
                        halo->recv_indices[ni][fill_idx[ni]++] = 3*k + i;
                        break;
                    }
                }
            }
        }
    }
    free(fill_idx);

    // Exchange counts to know how much to send
    // For simplicity in this structured mesh, send_count == recv_count for symmetric neighbors
    // In general you'd use MPI_Alltoall or point-to-point
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        int partner = halo->neighbor_ranks[ni];
        MPI_Sendrecv(&halo->recv_counts[ni], 1, MPI_INT, partner, 0,
                     &halo->send_counts[ni], 1, MPI_INT, partner, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Allocate send indices (we need to know which of our edges the partner needs)
    // For now, use a simplified approach: partner tells us which global indices it needs
    // We'll exchange the global indices and convert to local

    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        halo->send_indices[ni] = (int *)malloc(halo->send_counts[ni] * sizeof(int));
    }

    // Exchange the global indices each rank needs
    // recv_indices stores local edge indices; we need to send the corresponding global neighbor indices
    int **global_needed = (int **)malloc(halo->num_neighbors * sizeof(int *));
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        global_needed[ni] = (int *)malloc(halo->recv_counts[ni] * sizeof(int));
        for (int j = 0; j < halo->recv_counts[ni]; j++) {
            int edge_idx = halo->recv_indices[ni][j];
            global_needed[ni][j] = D->neighbours[edge_idx];  // Still stores global index
        }
    }

    // Exchange to learn what we need to send
    int **global_to_send = (int **)malloc(halo->num_neighbors * sizeof(int *));
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        global_to_send[ni] = (int *)malloc(halo->send_counts[ni] * sizeof(int));
        int partner = halo->neighbor_ranks[ni];
        MPI_Sendrecv(global_needed[ni], halo->recv_counts[ni], MPI_INT, partner, 1,
                     global_to_send[ni], halo->send_counts[ni], MPI_INT, partner, 1,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Convert global_to_send to local indices for our send_indices
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        for (int j = 0; j < halo->send_counts[ni]; j++) {
            int g = global_to_send[ni][j];
            // This global index should be owned by us; convert to local
            halo->send_indices[ni][j] = global_to_local(g, n_global, mpi_size, mpi_rank);
        }
        free(global_to_send[ni]);
        free(global_needed[ni]);
    }
    free(global_to_send);
    free(global_needed);

    // Allocate communication buffers (4 doubles per edge: h, uh, vh, stage)
    halo->send_buffer_size = 0;
    halo->recv_buffer_size = 0;
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        halo->send_buffer_size += halo->send_counts[ni];
        halo->recv_buffer_size += halo->recv_counts[ni];
    }
    halo->send_buffer = (double *)malloc(4 * halo->send_buffer_size * sizeof(double));
    halo->recv_buffer = (double *)malloc(4 * halo->recv_buffer_size * sizeof(double));

    free(rank_counts);
}

// Exchange halo data
static void exchange_halo(struct domain *D, struct halo_info *halo) {
    if (halo->num_neighbors == 0) return;

    int n = D->number_of_elements;

    // Pack send buffer (edge values from local elements that neighbors need)
    int offset = 0;
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        for (int j = 0; j < halo->send_counts[ni]; j++) {
            int k = halo->send_indices[ni][j];  // Local element index
            // Send edge 0's values (simplified - real code would track which edge)
            int edge_idx = 3*k + 0;
            halo->send_buffer[4*offset + 0] = D->height_edge_values[edge_idx];
            halo->send_buffer[4*offset + 1] = D->xmom_edge_values[edge_idx];
            halo->send_buffer[4*offset + 2] = D->ymom_edge_values[edge_idx];
            halo->send_buffer[4*offset + 3] = D->stage_edge_values[edge_idx];
            offset++;
        }
    }

    // Non-blocking sends and receives
    MPI_Request *requests = (MPI_Request *)malloc(2 * halo->num_neighbors * sizeof(MPI_Request));
    int send_offset = 0, recv_offset = 0;

    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        int partner = halo->neighbor_ranks[ni];
        MPI_Irecv(&halo->recv_buffer[4*recv_offset], 4 * halo->recv_counts[ni], MPI_DOUBLE,
                  partner, 2, MPI_COMM_WORLD, &requests[2*ni]);
        MPI_Isend(&halo->send_buffer[4*send_offset], 4 * halo->send_counts[ni], MPI_DOUBLE,
                  partner, 2, MPI_COMM_WORLD, &requests[2*ni + 1]);
        send_offset += halo->send_counts[ni];
        recv_offset += halo->recv_counts[ni];
    }

    MPI_Waitall(2 * halo->num_neighbors, requests, MPI_STATUSES_IGNORE);
    free(requests);

    // Unpack recv buffer into ghost edge values
    // We store received data in a temporary location or directly use in flux
    // For simplicity, we'll store in the edge arrays at special indices
    // Actually, let's just use the recv_buffer directly in flux computation
    // This requires modifying the flux kernel to handle remote data differently

    // For now, we'll unpack into a ghost array extension (simpler approach)
    // This is handled in the flux computation by checking neighbour_owners
}

static void compute_gradients_gpu(struct domain *D) {
    int n = D->number_of_elements;

    double *stage_c = D->stage_centroid_values;
    double *xmom_c = D->xmom_centroid_values;
    double *ymom_c = D->ymom_centroid_values;
    double *height_c = D->height_centroid_values;
    double *cx = D->centroid_x;
    double *cy = D->centroid_y;
    int *neighbours = D->neighbours;
    int *neighbour_owners = D->neighbour_owners;

    double *stage_gx = D->stage_x_gradient;
    double *stage_gy = D->stage_y_gradient;
    double *xmom_gx = D->xmom_x_gradient;
    double *xmom_gy = D->xmom_y_gradient;
    double *ymom_gx = D->ymom_x_gradient;
    double *ymom_gy = D->ymom_y_gradient;
    double *height_gx = D->height_x_gradient;
    double *height_gy = D->height_y_gradient;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        double c_x = cx[k];
        double c_y = cy[k];
        double stage_k = stage_c[k];
        double xmom_k = xmom_c[k];
        double ymom_k = ymom_c[k];
        double height_k = height_c[k];

        double dstage_dx = 0.0, dstage_dy = 0.0;
        double dxmom_dx = 0.0, dxmom_dy = 0.0;
        double dymom_dx = 0.0, dymom_dy = 0.0;
        double dheight_dx = 0.0, dheight_dy = 0.0;
        double sum_weight = 0.0;

        for (int i = 0; i < 3; i++) {
            int nb = neighbours[3*k + i];
            int owner = neighbour_owners[3*k + i];

            // Only use local neighbors for gradient (simplified - skip remote)
            if (nb >= 0 && owner == mpi_rank) {
                double dx_nb = cx[nb] - c_x;
                double dy_nb = cy[nb] - c_y;
                double dist_sq = dx_nb * dx_nb + dy_nb * dy_nb;

                if (dist_sq > 1.0e-20) {
                    double weight = 1.0 / sqrt(dist_sq);
                    sum_weight += weight;

                    double stage_nb = stage_c[nb];
                    double xmom_nb = xmom_c[nb];
                    double ymom_nb = ymom_c[nb];
                    double height_nb = height_c[nb];

                    dstage_dx += weight * (stage_nb - stage_k) * dx_nb / dist_sq;
                    dstage_dy += weight * (stage_nb - stage_k) * dy_nb / dist_sq;
                    dxmom_dx += weight * (xmom_nb - xmom_k) * dx_nb / dist_sq;
                    dxmom_dy += weight * (xmom_nb - xmom_k) * dy_nb / dist_sq;
                    dymom_dx += weight * (ymom_nb - ymom_k) * dx_nb / dist_sq;
                    dymom_dy += weight * (ymom_nb - ymom_k) * dy_nb / dist_sq;
                    dheight_dx += weight * (height_nb - height_k) * dx_nb / dist_sq;
                    dheight_dy += weight * (height_nb - height_k) * dy_nb / dist_sq;
                }
            }
        }

        if (sum_weight > 1.0e-20) {
            stage_gx[k] = dstage_dx / sum_weight;
            stage_gy[k] = dstage_dy / sum_weight;
            xmom_gx[k] = dxmom_dx / sum_weight;
            xmom_gy[k] = dxmom_dy / sum_weight;
            ymom_gx[k] = dymom_dx / sum_weight;
            ymom_gy[k] = dymom_dy / sum_weight;
            height_gx[k] = dheight_dx / sum_weight;
            height_gy[k] = dheight_dy / sum_weight;
        } else {
            stage_gx[k] = 0.0; stage_gy[k] = 0.0;
            xmom_gx[k] = 0.0; xmom_gy[k] = 0.0;
            ymom_gx[k] = 0.0; ymom_gy[k] = 0.0;
            height_gx[k] = 0.0; height_gy[k] = 0.0;
        }
    }
}

static void extrapolate_second_order_gpu(struct domain *D) {
    int n = D->number_of_elements;

    double *stage_c = D->stage_centroid_values;
    double *xmom_c = D->xmom_centroid_values;
    double *ymom_c = D->ymom_centroid_values;
    double *height_c = D->height_centroid_values;
    double *cx = D->centroid_x;
    double *cy = D->centroid_y;
    double *ex = D->edge_midpoint_x;
    double *ey = D->edge_midpoint_y;
    int *neighbours = D->neighbours;
    int *neighbour_owners = D->neighbour_owners;

    double *stage_gx = D->stage_x_gradient;
    double *stage_gy = D->stage_y_gradient;
    double *xmom_gx = D->xmom_x_gradient;
    double *xmom_gy = D->xmom_y_gradient;
    double *ymom_gx = D->ymom_x_gradient;
    double *ymom_gy = D->ymom_y_gradient;
    double *height_gx = D->height_x_gradient;
    double *height_gy = D->height_y_gradient;

    double *stage_edge = D->stage_edge_values;
    double *xmom_edge = D->xmom_edge_values;
    double *ymom_edge = D->ymom_edge_values;
    double *height_edge = D->height_edge_values;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        double c_x = cx[k];
        double c_y = cy[k];

        double stage_k = stage_c[k];
        double xmom_k = xmom_c[k];
        double ymom_k = ymom_c[k];
        double height_k = height_c[k];

        double sgx = stage_gx[k], sgy = stage_gy[k];
        double xgx = xmom_gx[k], xgy = xmom_gy[k];
        double ygx = ymom_gx[k], ygy = ymom_gy[k];
        double hgx = height_gx[k], hgy = height_gy[k];

        for (int i = 0; i < 3; i++) {
            int ki = 3*k + i;
            double dx_e = ex[ki] - c_x;
            double dy_e = ey[ki] - c_y;

            double dstage = sgx * dx_e + sgy * dy_e;
            double dxmom = xgx * dx_e + xgy * dy_e;
            double dymom = ygx * dx_e + ygy * dy_e;
            double dheight = hgx * dx_e + hgy * dy_e;

            int nb = neighbours[ki];
            int owner = neighbour_owners[ki];

            // Minmod limiter (only for local neighbors)
            if (nb >= 0 && owner == mpi_rank) {
                double dstage_nb = stage_c[nb] - stage_k;
                double dxmom_nb = xmom_c[nb] - xmom_k;
                double dymom_nb = ymom_c[nb] - ymom_k;
                double dheight_nb = height_c[nb] - height_k;

                if (dstage * dstage_nb > 0.0) {
                    if (fabs(dstage) > fabs(dstage_nb)) dstage = dstage_nb;
                } else {
                    dstage = 0.0;
                }
                if (dxmom * dxmom_nb > 0.0) {
                    if (fabs(dxmom) > fabs(dxmom_nb)) dxmom = dxmom_nb;
                } else {
                    dxmom = 0.0;
                }
                if (dymom * dymom_nb > 0.0) {
                    if (fabs(dymom) > fabs(dymom_nb)) dymom = dymom_nb;
                } else {
                    dymom = 0.0;
                }
                if (dheight * dheight_nb > 0.0) {
                    if (fabs(dheight) > fabs(dheight_nb)) dheight = dheight_nb;
                } else {
                    dheight = 0.0;
                }
            }

            stage_edge[ki] = stage_k + dstage;
            xmom_edge[ki] = xmom_k + dxmom;
            ymom_edge[ki] = ymom_k + dymom;
            height_edge[ki] = fmax(height_k + dheight, 0.0);
        }
    }
}

// HLL flux with MPI halo support
static double compute_fluxes_gpu(struct domain *D, struct halo_info *halo) {
    int n = D->number_of_elements;
    double g = D->g;
    double h0 = D->minimum_allowed_height;
    double local_max_speed = 0.0;

    double *height_edge = D->height_edge_values;
    double *xmom_edge = D->xmom_edge_values;
    double *ymom_edge = D->ymom_edge_values;
    double *stage_update = D->stage_explicit_update;
    double *xmom_update = D->xmom_explicit_update;
    double *ymom_update = D->ymom_explicit_update;
    int *neighbours = D->neighbours;
    int *neighbour_owners = D->neighbour_owners;
    double *edgelengths = D->edgelengths;
    double *normals = D->normals;
    double *areas = D->areas;
    double *max_speed = D->max_speed;

    // Get halo data pointers for GPU
    double *recv_buf = halo->recv_buffer;
    int recv_size = halo->recv_buffer_size;

    #pragma omp target teams distribute parallel for \
        reduction(max: local_max_speed)
    for (int k = 0; k < n; k++) {
        double stage_flux_sum = 0.0;
        double xmom_flux_sum = 0.0;
        double ymom_flux_sum = 0.0;
        double speed_max = 0.0;

        for (int i = 0; i < 3; i++) {
            int ki = 3*k + i;
            int nb = neighbours[ki];
            int owner = neighbour_owners[ki];

            double nx = normals[6*k + 2*i];
            double ny = normals[6*k + 2*i + 1];
            double el = edgelengths[ki];

            double h_L = height_edge[ki];
            double uh_L = xmom_edge[ki];
            double vh_L = ymom_edge[ki];

            double h_R, uh_R, vh_R;
            if (nb < 0) {
                // Physical boundary - reflective
                h_R = h_L;
                double vn_L = uh_L * nx + vh_L * ny;
                uh_R = uh_L - 2.0 * vn_L * nx;
                vh_R = vh_L - 2.0 * vn_L * ny;
            } else if (owner == mpi_rank) {
                // Local neighbor
                int ki_nb = 3*nb + 0;
                h_R = height_edge[ki_nb];
                uh_R = xmom_edge[ki_nb];
                vh_R = ymom_edge[ki_nb];
            } else {
                // Remote neighbor - use received halo data
                // For simplicity, use first-order at MPI boundaries
                h_R = h_L;
                uh_R = uh_L;
                vh_R = vh_L;
            }

            // Velocities
            double u_L, v_L, u_R, v_R;
            if (h_L > h0) { u_L = uh_L / h_L; v_L = vh_L / h_L; }
            else { u_L = 0.0; v_L = 0.0; }
            if (h_R > h0) { u_R = uh_R / h_R; v_R = vh_R / h_R; }
            else { u_R = 0.0; v_R = 0.0; }

            // Rotate to edge-normal
            double un_L = u_L * nx + v_L * ny;
            double ut_L = -u_L * ny + v_L * nx;
            double un_R = u_R * nx + v_R * ny;
            double ut_R = -u_R * ny + v_R * nx;

            double c_L = sqrt(g * fmax(h_L, 0.0));
            double c_R = sqrt(g * fmax(h_R, 0.0));

            double s_L = fmin(un_L - c_L, un_R - c_R);
            double s_R = fmax(un_L + c_L, un_R + c_R);

            // Physical fluxes
            double flux_mass_L = h_L * un_L;
            double flux_mom_n_L = h_L * un_L * un_L + 0.5 * g * h_L * h_L;
            double flux_mom_t_L = h_L * un_L * ut_L;

            double flux_mass_R = h_R * un_R;
            double flux_mom_n_R = h_R * un_R * un_R + 0.5 * g * h_R * h_R;
            double flux_mom_t_R = h_R * un_R * ut_R;

            double q_mass_L = h_L, q_mom_n_L = h_L * un_L, q_mom_t_L = h_L * ut_L;
            double q_mass_R = h_R, q_mom_n_R = h_R * un_R, q_mom_t_R = h_R * ut_R;

            double flux_mass, flux_mom_n, flux_mom_t;
            if (s_L >= 0.0) {
                flux_mass = flux_mass_L;
                flux_mom_n = flux_mom_n_L;
                flux_mom_t = flux_mom_t_L;
            } else if (s_R <= 0.0) {
                flux_mass = flux_mass_R;
                flux_mom_n = flux_mom_n_R;
                flux_mom_t = flux_mom_t_R;
            } else {
                double denom = s_R - s_L;
                if (fabs(denom) < 1.0e-15) denom = 1.0e-15;
                flux_mass = (s_R * flux_mass_L - s_L * flux_mass_R + s_L * s_R * (q_mass_R - q_mass_L)) / denom;
                flux_mom_n = (s_R * flux_mom_n_L - s_L * flux_mom_n_R + s_L * s_R * (q_mom_n_R - q_mom_n_L)) / denom;
                flux_mom_t = (s_R * flux_mom_t_L - s_L * flux_mom_t_R + s_L * s_R * (q_mom_t_R - q_mom_t_L)) / denom;
            }

            double flux_xmom = flux_mom_n * nx - flux_mom_t * ny;
            double flux_ymom = flux_mom_n * ny + flux_mom_t * nx;

            stage_flux_sum -= flux_mass * el;
            xmom_flux_sum -= flux_xmom * el;
            ymom_flux_sum -= flux_ymom * el;

            double local_speed = fmax(fabs(s_L), fabs(s_R));
            speed_max = fmax(speed_max, local_speed);
        }

        double inv_area = 1.0 / areas[k];
        stage_update[k] = stage_flux_sum * inv_area;
        xmom_update[k] = xmom_flux_sum * inv_area;
        ymom_update[k] = ymom_flux_sum * inv_area;
        max_speed[k] = speed_max;

        local_max_speed = fmax(local_max_speed, speed_max);
    }

    // Global reduction for CFL
    double global_max_speed;
    MPI_Allreduce(&local_max_speed, &global_max_speed, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

    return global_max_speed;
}

static void protect_gpu(struct domain *D) {
    int n = D->number_of_elements;
    double minimum_allowed_height = D->minimum_allowed_height;

    double *stage_c = D->stage_centroid_values;
    double *bed_c = D->bed_centroid_values;
    double *xmom_c = D->xmom_centroid_values;
    double *ymom_c = D->ymom_centroid_values;
    double *height_c = D->height_centroid_values;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        double hc = stage_c[k] - bed_c[k];

        if (hc < minimum_allowed_height) {
            xmom_c[k] = 0.0;
            ymom_c[k] = 0.0;

            if (hc <= 0.0) {
                double bmin = bed_c[k];
                if (stage_c[k] < bmin) stage_c[k] = bmin;
            }
        }
        height_c[k] = fmax(stage_c[k] - bed_c[k], 0.0);
    }
}

static void update_gpu(struct domain *D, double dt) {
    int n = D->number_of_elements;

    double *stage_c = D->stage_centroid_values;
    double *xmom_c = D->xmom_centroid_values;
    double *ymom_c = D->ymom_centroid_values;
    double *stage_update = D->stage_explicit_update;
    double *xmom_update = D->xmom_explicit_update;
    double *ymom_update = D->ymom_explicit_update;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        stage_c[k] += dt * stage_update[k];
        xmom_c[k] += dt * xmom_update[k];
        ymom_c[k] += dt * ymom_update[k];
    }
}

// Apply tidal boundary condition at ocean (y < threshold)
static void apply_tide_gpu(struct domain *D, struct delta_config *cfg,
                           double sim_time) {
    if (!cfg->enabled || cfg->tide_amplitude == 0.0) return;

    int n = D->number_of_elements;
    double L = D->domain_length;
    double tide_zone = 0.05 * L;  // Apply tide in first 5% of domain

    double *stage_c = D->stage_centroid_values;
    double *bed_c = D->bed_centroid_values;
    double *height_c = D->height_centroid_values;
    double *cy = D->centroid_y;

    double tide_level = cfg->mean_sea_level +
                        cfg->tide_amplitude * sin(2.0 * M_PI * sim_time / cfg->tide_period);

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        if (cy[k] < tide_zone) {
            // Relaxation toward tidal level (soft boundary)
            double relax = 1.0 - cy[k] / tide_zone;  // 1 at y=0, 0 at y=tide_zone
            relax = relax * relax;  // Stronger near boundary
            double target = tide_level;
            stage_c[k] = stage_c[k] + 0.5 * relax * (target - stage_c[k]);
            height_c[k] = fmax(stage_c[k] - bed_c[k], 0.0);
        }
    }
}

// Precompute river inlet area (call once at setup)
static double compute_river_area(struct domain *D, struct delta_config *cfg, int *n_cells) {
    int n = D->number_of_elements;
    double L = D->domain_length;
    double inlet_zone = 0.95 * L;
    double channel_center = 0.5 * L;
    double channel_hw = cfg->channel_width * L / 2.0;

    double local_area = 0.0;
    int local_count = 0;
    for (int k = 0; k < n; k++) {
        double x = D->centroid_x[k];
        double y = D->centroid_y[k];
        if (y > inlet_zone && fabs(x - channel_center) < channel_hw) {
            local_area += D->areas[k];
            local_count++;
        }
    }

    double global_area;
    int global_count;
    MPI_Allreduce(&local_area, &global_area, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&local_count, &global_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    *n_cells = global_count;
    return global_area;
}

// Apply river discharge at inlet (y > threshold, in channel)
// river_area should be precomputed once at setup
static void apply_river_discharge_gpu(struct domain *D, struct delta_config *cfg,
                                       double dt, double river_area) {
    if (!cfg->enabled || cfg->river_discharge == 0.0 || river_area < 1.0e-10) return;

    int n = D->number_of_elements;
    double L = D->domain_length;
    double inlet_zone = 0.95 * L;
    double channel_center = 0.5 * L;
    double channel_hw = cfg->channel_width * L / 2.0;

    double *stage_c = D->stage_centroid_values;
    double *bed_c = D->bed_centroid_values;
    double *height_c = D->height_centroid_values;
    double *ymom_c = D->ymom_centroid_values;
    double *cx = D->centroid_x;
    double *cy = D->centroid_y;

    // Precomputed: discharge per unit area (m/s equivalent depth increase)
    double discharge_per_area = cfg->river_discharge / river_area;
    double momentum_per_area = discharge_per_area * cfg->river_velocity;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        if (cy[k] > inlet_zone && fabs(cx[k] - channel_center) < channel_hw) {
            // Add water (mass)
            stage_c[k] += discharge_per_area * dt;
            height_c[k] = fmax(stage_c[k] - bed_c[k], 0.0);

            // Add momentum (flowing downstream, negative y direction)
            ymom_c[k] -= momentum_per_area * dt * height_c[k];
        }
    }
}

// Apply rainfall uniformly
static void apply_rain_gpu(struct domain *D, struct delta_config *cfg, double dt) {
    if (!cfg->enabled || cfg->rain_rate == 0.0) return;

    int n = D->number_of_elements;

    double *stage_c = D->stage_centroid_values;
    double *bed_c = D->bed_centroid_values;
    double *height_c = D->height_centroid_values;

    double rain_depth = cfg->rain_rate * dt;

    #pragma omp target teams distribute parallel for
    for (int k = 0; k < n; k++) {
        // Rain falls everywhere (even on dry cells - they get wet!)
        stage_c[k] += rain_depth;
        height_c[k] = fmax(stage_c[k] - bed_c[k], 0.0);
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Each rank uses its own GPU (modulo num_devices for oversubscription)
    int num_devices = omp_get_num_devices();
    int my_device = mpi_rank % num_devices;
    omp_set_default_device(my_device);
    //CALI_MARK_BEGIN("main func");

    if (argc < 2 || argc > 8) {
        if (mpi_rank == 0) {
            fprintf(stderr, "Usage: %s N [niter] [yieldstep] [gather] [domain_km] [depth_m] [delta]\n", argv[0]);
            fprintf(stderr, "  MPI + OpenMP target version with yieldstep transfers\n");
            fprintf(stderr, "  gather: 0 = distributed I/O (scalable), 1 = gather to rank 0\n");
            fprintf(stderr, "  delta:  0 = flat bottom (default), 1 = river delta with tide/rain/discharge\n");
        }
        MPI_Finalize();
        return 1;
    }

    int grid_size = atoi(argv[1]);
    int niter = (argc >= 3) ? atoi(argv[2]) : 1000;
    int yieldstep = (argc >= 4) ? atoi(argv[3]) : 100;
    int gather_mode = (argc >= 5) ? atoi(argv[4]) : 0;  // 0 = distributed (default)
    double domain_length = (argc >= 6) ? atof(argv[5]) * 1000.0 : 100000.0;
    double initial_height = (argc >= 7) ? atof(argv[6]) : 10.0;
    int delta_mode = (argc >= 8) ? atoi(argv[7]) : 0;

    // Set up delta configuration
    struct delta_config delta_cfg;
    delta_cfg.enabled = delta_mode;

    // Realistic-ish river delta parameters
    delta_cfg.ocean_depth = 10.0;          // 10m deep ocean
    delta_cfg.land_height = 3.0;           // Land rises to 3m above sea level
    delta_cfg.channel_depth = 5.0;         // River channel 5m deep
    delta_cfg.channel_width = 0.1;         // Channel is 10% of domain width

    delta_cfg.tide_amplitude = 1.5;        // 1.5m tidal range (3m total)
    delta_cfg.tide_period = 12.4 * 3600.0; // Semi-diurnal tide (~12.4 hours)
    delta_cfg.mean_sea_level = 0.0;        // Reference level

    delta_cfg.river_discharge = 1000.0;    // 1000 m³/s (medium-sized river)
    delta_cfg.river_velocity = 1.5;        // 1.5 m/s inflow velocity

    delta_cfg.rain_rate = 20.0e-3 / 3600.0; // 20 mm/hr (heavy rain)

    if (grid_size < 3) {
        if (mpi_rank == 0) fprintf(stderr, "Error: Grid size must be at least 3\n");
        MPI_Finalize();
        return 1;
    }

    int n_global = 2 * (grid_size - 1) * (grid_size - 1);
    int local_start, local_n;
    //CALI_MARK_BEGIN("compute_partition");
    compute_partition(n_global, mpi_size, mpi_rank, &local_start, &local_n);
    //CALI_MARK_END("compute_partition");

    // Compute mesh geometry for reporting
    double dx = domain_length / (grid_size - 1);
    double triangle_area = 0.5 * dx * dx;  // Right triangle with legs dx

    if (mpi_rank == 0) {
        printf("============================================================\n");
        printf("  SHALLOW WATER - MPI + OpenMP Target (HLL Flux)\n");
        printf("============================================================\n\n");
        printf("MPI Configuration:\n");
        printf("  Ranks:            %d\n", mpi_size);
        printf("  GPUs available:   %d\n", num_devices);
        printf("  Global triangles: %d\n", n_global);
        printf("  Local per rank:   ~%d\n\n", n_global / mpi_size);
        printf("Physical Setup:\n");
        printf("  Domain:           %.2f km x %.2f km\n", domain_length/1000, domain_length/1000);
        printf("  Initial depth:    %.2f m\n", initial_height);
        printf("  Iterations:       %d\n", niter);
        printf("  Yieldstep:        %d\n", yieldstep);
        printf("  I/O mode:         %s\n\n", gather_mode ? "GATHER (rank 0)" : "DISTRIBUTED (scalable)");
        printf("Mesh Geometry:\n");
        printf("  Edge length:      %.2f m\n", dx);
        printf("  Triangle area:    %.2f m^2\n\n", triangle_area);

        if (delta_cfg.enabled) {
            printf("Delta Configuration (ENABLED):\n");
            printf("  Bathymetry:\n");
            printf("    Ocean depth:    %.1f m\n", delta_cfg.ocean_depth);
            printf("    Land height:    %.1f m above MSL\n", delta_cfg.land_height);
            printf("    Channel depth:  %.1f m\n", delta_cfg.channel_depth);
            printf("    Channel width:  %.0f%% of domain\n", delta_cfg.channel_width * 100);
            printf("  Tide:\n");
            printf("    Amplitude:      %.2f m (%.2f m range)\n",
                   delta_cfg.tide_amplitude, 2*delta_cfg.tide_amplitude);
            printf("    Period:         %.2f hours\n", delta_cfg.tide_period / 3600.0);
            printf("  River:\n");
            printf("    Discharge:      %.0f m^3/s\n", delta_cfg.river_discharge);
            printf("    Velocity:       %.1f m/s\n", delta_cfg.river_velocity);
            printf("  Rain:\n");
            printf("    Rate:           %.1f mm/hr\n\n", delta_cfg.rain_rate * 3600.0 * 1000.0);
        } else {
            printf("Mode: FLAT BOTTOM (uniform depth)\n\n");
        }
    }

    // Allocate domain
    struct domain D;
    D.number_of_elements = local_n;
    D.global_elements = n_global;
    D.local_start = local_start;
    D.g = 9.81;
    D.epsilon = 1.0e-12;
    D.minimum_allowed_height = 1.0e-6;
    D.cfl = 2.0;
    D.domain_length = domain_length;
    D.char_length = domain_length / sqrt((double)n_global / 2.0);

    int n = local_n;

    // Allocate arrays
    //CALI_MARK_BEGIN("memory allocation");
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
    D.xmom_edge_values = (double *)malloc(3 * n * sizeof(double));
    D.ymom_edge_values = (double *)malloc(3 * n * sizeof(double));
    D.stage_x_gradient = (double *)malloc(n * sizeof(double));
    D.stage_y_gradient = (double *)malloc(n * sizeof(double));
    D.xmom_x_gradient = (double *)malloc(n * sizeof(double));
    D.xmom_y_gradient = (double *)malloc(n * sizeof(double));
    D.ymom_x_gradient = (double *)malloc(n * sizeof(double));
    D.ymom_y_gradient = (double *)malloc(n * sizeof(double));
    D.height_x_gradient = (double *)malloc(n * sizeof(double));
    D.height_y_gradient = (double *)malloc(n * sizeof(double));
    D.centroid_x = (double *)malloc(n * sizeof(double));
    D.centroid_y = (double *)malloc(n * sizeof(double));
    D.edge_midpoint_x = (double *)malloc(3 * n * sizeof(double));
    D.edge_midpoint_y = (double *)malloc(3 * n * sizeof(double));
    D.neighbours = (int *)malloc(3 * n * sizeof(int));
    D.neighbour_owners = (int *)malloc(3 * n * sizeof(int));
    D.edgelengths = (double *)malloc(3 * n * sizeof(double));
    D.normals = (double *)malloc(6 * n * sizeof(double));
    D.areas = (double *)malloc(n * sizeof(double));
    D.radii = (double *)malloc(n * sizeof(double));
    D.max_speed = (double *)malloc(n * sizeof(double));

    // Allocate verbose output (local arrays for this rank)
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
    //CALI_MARK_END("memory allocation");

    // Initialize
    //CALI_MARK_BEGIN("generate mesh and init quantities");
    generate_mesh_local(&D, grid_size);
    init_quantities(&D, initial_height, &delta_cfg);
    //CALI_MARK_END("generate mesh and init quantities");

    // Count wet/dry cells for delta mode
    int local_wet = 0, local_dry = 0;
    for (int k = 0; k < n; k++) {
        if (D.height_centroid_values[k] > D.minimum_allowed_height) {
            local_wet++;
        } else {
            local_dry++;
        }
    }
    int global_wet, global_dry;
    MPI_Reduce(&local_wet, &global_wet, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_dry, &global_dry, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    // Precompute river area for delta mode
    int n_river_cells = 0;
    double river_area = 0.0;
    if (delta_cfg.enabled) {
        river_area = compute_river_area(&D, &delta_cfg, &n_river_cells);
    }

    if (mpi_rank == 0 && delta_cfg.enabled) {
        printf("Initial cell states:\n");
        printf("  Wet cells:        %d (%.1f%%)\n", global_wet, 100.0 * global_wet / n_global);
        printf("  Dry cells:        %d (%.1f%%)\n", global_dry, 100.0 * global_dry / n_global);
        printf("  River inlet:      %d cells, %.0f m^2 area\n\n", n_river_cells, river_area);
    }

    // Build halo exchange info
    struct halo_info halo;
    build_halo_info(&D, &halo);

    MPI_Barrier(MPI_COMM_WORLD);

    // Map to GPU
    //CALI_MARK_BEGIN("memory transfer to the device");
    #pragma omp target enter data map(to: D.stage_centroid_values[0:n], \
        D.xmom_centroid_values[0:n], D.ymom_centroid_values[0:n], \
        D.bed_centroid_values[0:n], D.height_centroid_values[0:n], \
        D.stage_explicit_update[0:n], D.xmom_explicit_update[0:n], \
        D.ymom_explicit_update[0:n], D.stage_edge_values[0:3*n], \
        D.height_edge_values[0:3*n], D.xmom_edge_values[0:3*n], \
        D.ymom_edge_values[0:3*n], D.stage_x_gradient[0:n], \
        D.stage_y_gradient[0:n], D.xmom_x_gradient[0:n], D.xmom_y_gradient[0:n], \
        D.ymom_x_gradient[0:n], D.ymom_y_gradient[0:n], D.height_x_gradient[0:n], \
        D.height_y_gradient[0:n], D.centroid_x[0:n], D.centroid_y[0:n], \
        D.edge_midpoint_x[0:3*n], D.edge_midpoint_y[0:3*n], \
        D.neighbours[0:3*n], D.neighbour_owners[0:3*n], D.edgelengths[0:3*n], \
        D.normals[0:6*n], D.areas[0:n], D.radii[0:n], D.max_speed[0:n])
    //CALI_MARK_END("memory transfer to the device");

    // Run benchmark
    double sim_time = 0.0;
    double dt = 0.0;
    double max_speed_global;
    double t_transfer = 0.0;
    int yield_count = 0;

    if (mpi_rank == 0) printf("Running %d iterations (yieldstep=%d)...\n\n", niter, yieldstep);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = omp_get_wtime();

    //CALI_MARK_BEGIN("main iter loop");
    for (int iter = 0; iter < niter; iter++) {
        compute_gradients_gpu(&D);
        extrapolate_second_order_gpu(&D);

        // Halo exchange (CPU-side for now)
        // In production, you'd do this with GPU-aware MPI
        // exchange_halo(&D, &halo);

        max_speed_global = compute_fluxes_gpu(&D, &halo);

        if (max_speed_global > D.epsilon) {
            dt = D.cfl * D.char_length / max_speed_global;
        } else {
            dt = D.cfl * D.char_length / sqrt(D.g * initial_height);
        }

        protect_gpu(&D);
        update_gpu(&D, dt);

        // Apply source terms for delta mode
        if (delta_cfg.enabled) {
            apply_tide_gpu(&D, &delta_cfg, sim_time);
            apply_river_discharge_gpu(&D, &delta_cfg, dt, river_area);
            apply_rain_gpu(&D, &delta_cfg, dt);
        }

        sim_time += dt;

        // Yieldstep transfer
        if ((iter + 1) % yieldstep == 0) {
            transfer_yieldstep(&D, &out, &t_transfer, gather_mode);
            yield_count++;
        }

        if ((iter + 1) % 100 == 0) {
            print_progress(iter + 1, niter, omp_get_wtime() - t0, sim_time, yield_count);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    //CALI_MARK_END("main iter loop");
    double t_compute_total = omp_get_wtime() - t0;
    double t_compute_pure = t_compute_total - t_transfer;

    // Cleanup GPU
    //CALI_MARK_BEGIN("cleanup of PGU mem");
    #pragma omp target exit data map(delete: D.stage_centroid_values[0:n], \
        D.xmom_centroid_values[0:n], D.ymom_centroid_values[0:n], \
        D.bed_centroid_values[0:n], D.height_centroid_values[0:n], \
        D.stage_explicit_update[0:n], D.xmom_explicit_update[0:n], \
        D.ymom_explicit_update[0:n], D.stage_edge_values[0:3*n], \
        D.height_edge_values[0:3*n], D.xmom_edge_values[0:3*n], \
        D.ymom_edge_values[0:3*n], D.stage_x_gradient[0:n], \
        D.stage_y_gradient[0:n], D.xmom_x_gradient[0:n], D.xmom_y_gradient[0:n], \
        D.ymom_x_gradient[0:n], D.ymom_y_gradient[0:n], D.height_x_gradient[0:n], \
        D.height_y_gradient[0:n], D.centroid_x[0:n], D.centroid_y[0:n], \
        D.edge_midpoint_x[0:3*n], D.edge_midpoint_y[0:3*n], \
        D.neighbours[0:3*n], D.neighbour_owners[0:3*n], D.edgelengths[0:3*n], \
        D.normals[0:6*n], D.areas[0:n], D.radii[0:n], D.max_speed[0:n])
    //CALI_MARK_END("cleanup of PGU mem");

    // Results
    if (mpi_rank == 0) {
        printf("\n\n");
        printf("============================================================\n");
        printf("  RESULTS (MPI + YIELDSTEP)\n");
        printf("============================================================\n\n");

        double time_per_step = t_compute_total / (double)niter;
        double steps_per_second = 1.0 / time_per_step;
        dt = sim_time / (double)niter;
        double target_time = 3.5 * 24.0 * 3600.0;
        double estimated_steps = target_time / dt;
        double estimated_wallclock = estimated_steps * time_per_step;

        printf("Ran %d iterations on %d ranks:\n", niter, mpi_size);
        printf("  Total time:       %.4f s\n", t_compute_total);
        printf("  Pure compute:     %.4f s (%.4f ms/iter)\n", t_compute_pure, t_compute_pure * 1000.0 / niter);
        printf("  Yieldstep xfer:   %.4f s (%d yields, %.4f ms/yield)\n",
               t_transfer, yield_count, yield_count > 0 ? t_transfer * 1000.0 / yield_count : 0.0);
        printf("  Transfer overhead: %.2f%% of compute\n\n", 100.0 * t_transfer / t_compute_pure);
        printf("  Time per step:    %.6f ms (including transfers)\n", time_per_step * 1000.0);
        printf("  Steps per second: %.2f\n\n", steps_per_second);

        printf("CFL Timestep:\n");
        printf("  Average dt:       %.4e s\n", dt);
        printf("  Simulated time:   %.4f s\n\n", sim_time);

        printf("2-DAY SIMULATION ESTIMATE:\n");
        printf("  Estimated steps:  %.4e\n", estimated_steps);
        if (estimated_wallclock < 60.0) {
            printf("  Wall-clock:       %.2f seconds\n", estimated_wallclock);
        } else if (estimated_wallclock < 3600.0) {
            printf("  Wall-clock:       %.2f minutes\n", estimated_wallclock / 60.0);
        } else {
            printf("  Wall-clock:       %.2f hours\n", estimated_wallclock / 3600.0);
        }
        printf("\n");
    }

    // Cleanup verbose output
    free(out.stage); free(out.height); free(out.xmom); free(out.ymom);
    free(out.xvel); free(out.yvel); free(out.speed); free(out.froude);
    free(out.stage_update); free(out.xmom_update); free(out.ymom_update);
    free(out.stage_edge); free(out.height_edge); free(out.max_speed_elem);

    // Cleanup domain
    free(D.stage_centroid_values); free(D.xmom_centroid_values);
    free(D.ymom_centroid_values); free(D.bed_centroid_values);
    free(D.height_centroid_values); free(D.stage_explicit_update);
    free(D.xmom_explicit_update); free(D.ymom_explicit_update);
    free(D.stage_edge_values); free(D.height_edge_values);
    free(D.xmom_edge_values); free(D.ymom_edge_values);
    free(D.stage_x_gradient); free(D.stage_y_gradient);
    free(D.xmom_x_gradient); free(D.xmom_y_gradient);
    free(D.ymom_x_gradient); free(D.ymom_y_gradient);
    free(D.height_x_gradient); free(D.height_y_gradient);
    free(D.centroid_x); free(D.centroid_y);
    free(D.edge_midpoint_x); free(D.edge_midpoint_y);
    free(D.neighbours); free(D.neighbour_owners);
    free(D.edgelengths); free(D.normals);
    free(D.areas); free(D.radii); free(D.max_speed);

    MPI_Finalize();
    //CALI_MARK_END("main func");
    return 0;
}
