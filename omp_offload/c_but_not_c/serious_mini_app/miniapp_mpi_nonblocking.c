// sw_cfl2_mpi_nonblocking.c - Shallow water mini app with MPI + OpenMP target offload
// Non-blocking MPI version with communication/computation overlap
// Second-order extrapolation + HLL flux + domain decomposition

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <mpi.h>
#include <caliper/cali.h>

struct domain {
    int number_of_elements;      // Local elements on this rank
    int global_elements;         // Total elements across all ranks
    int local_start;             // Global index of first local element

    // Element classification for overlap
    int num_interior;            // Elements with only local neighbors
    int num_boundary;            // Elements with remote neighbors
    int *interior_indices;       // Indices of interior elements
    int *boundary_indices;       // Indices of boundary elements

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

// Halo exchange structure with persistent requests
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

    // Non-blocking request arrays
    MPI_Request *send_requests;
    MPI_Request *recv_requests;

    // Flag to track if exchange is in progress
    int exchange_in_progress;
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

    // Optional gather to rank 0 using non-blocking ops
    if (gather_mode) {
        int n_global = D->global_elements;

        int *recv_counts = NULL;
        int *displs = NULL;
        double *global_stage = NULL;

        if (mpi_rank == 0) {
            recv_counts = (int *)malloc(mpi_size * sizeof(int));
            displs = (int *)malloc(mpi_size * sizeof(int));
            global_stage = (double *)malloc(n_global * sizeof(double));
        }

        MPI_Gather(&n, 1, MPI_INT, recv_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (mpi_rank == 0) {
            displs[0] = 0;
            for (int i = 1; i < mpi_size; i++) {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }

        // Non-blocking gather
        MPI_Request gather_req;
        MPI_Igatherv(out->stage, n, MPI_DOUBLE,
                     global_stage, recv_counts, displs, MPI_DOUBLE,
                     0, MPI_COMM_WORLD, &gather_req);

        // Could do other work here while gather is in progress
        MPI_Wait(&gather_req, MPI_STATUS_IGNORE);

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

// Classify elements as interior or boundary for overlap
static void classify_elements(struct domain *D) {
    int n = D->number_of_elements;

    // First pass: count interior and boundary elements
    int num_interior = 0;
    int num_boundary = 0;

    for (int k = 0; k < n; k++) {
        int has_remote = 0;
        for (int i = 0; i < 3; i++) {
            int owner = D->neighbour_owners[3*k + i];
            if (owner >= 0 && owner != mpi_rank) {
                has_remote = 1;
                break;
            }
        }
        if (has_remote) {
            num_boundary++;
        } else {
            num_interior++;
        }
    }

    D->num_interior = num_interior;
    D->num_boundary = num_boundary;
    D->interior_indices = (int *)malloc(num_interior * sizeof(int));
    D->boundary_indices = (int *)malloc(num_boundary * sizeof(int));

    // Second pass: fill index arrays
    int int_idx = 0;
    int bnd_idx = 0;
    for (int k = 0; k < n; k++) {
        int has_remote = 0;
        for (int i = 0; i < 3; i++) {
            int owner = D->neighbour_owners[3*k + i];
            if (owner >= 0 && owner != mpi_rank) {
                has_remote = 1;
                break;
            }
        }
        if (has_remote) {
            D->boundary_indices[bnd_idx++] = k;
        } else {
            D->interior_indices[int_idx++] = k;
        }
    }

    if (mpi_rank == 0) {
        printf("Element classification:\n");
        printf("  Interior: %d (%.1f%%)\n", num_interior, 100.0 * num_interior / n);
        printf("  Boundary: %d (%.1f%%)\n\n", num_boundary, 100.0 * num_boundary / n);
    }
}

static void init_quantities(struct domain *D, double initial_height) {
    int n = D->number_of_elements;

    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        D->bed_centroid_values[k] = 0.0;
        D->stage_centroid_values[k] = initial_height;
        D->height_centroid_values[k] = initial_height;
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

        for (int i = 0; i < 3; i++) {
            D->stage_edge_values[3*k + i] = initial_height;
            D->height_edge_values[3*k + i] = initial_height;
            D->xmom_edge_values[3*k + i] = 0.0;
            D->ymom_edge_values[3*k + i] = 0.0;
        }
    }
}

// Build halo exchange info with non-blocking support
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

    halo->exchange_in_progress = 0;

    if (halo->num_neighbors == 0) {
        halo->neighbor_ranks = NULL;
        halo->send_counts = NULL;
        halo->recv_counts = NULL;
        halo->send_indices = NULL;
        halo->recv_indices = NULL;
        halo->send_buffer = NULL;
        halo->recv_buffer = NULL;
        halo->send_requests = NULL;
        halo->recv_requests = NULL;
        free(rank_counts);
        return;
    }

    halo->neighbor_ranks = (int *)malloc(halo->num_neighbors * sizeof(int));
    halo->send_counts = (int *)malloc(halo->num_neighbors * sizeof(int));
    halo->recv_counts = (int *)malloc(halo->num_neighbors * sizeof(int));
    halo->send_indices = (int **)malloc(halo->num_neighbors * sizeof(int *));
    halo->recv_indices = (int **)malloc(halo->num_neighbors * sizeof(int *));
    halo->send_requests = (MPI_Request *)malloc(halo->num_neighbors * sizeof(MPI_Request));
    halo->recv_requests = (MPI_Request *)malloc(halo->num_neighbors * sizeof(MPI_Request));

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

    // Exchange counts using non-blocking ops
    MPI_Request *count_reqs = (MPI_Request *)malloc(2 * halo->num_neighbors * sizeof(MPI_Request));
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        int partner = halo->neighbor_ranks[ni];
        MPI_Irecv(&halo->send_counts[ni], 1, MPI_INT, partner, 0,
                  MPI_COMM_WORLD, &count_reqs[2*ni]);
        MPI_Isend(&halo->recv_counts[ni], 1, MPI_INT, partner, 0,
                  MPI_COMM_WORLD, &count_reqs[2*ni + 1]);
    }
    MPI_Waitall(2 * halo->num_neighbors, count_reqs, MPI_STATUSES_IGNORE);
    free(count_reqs);

    // Allocate send indices
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        halo->send_indices[ni] = (int *)malloc(halo->send_counts[ni] * sizeof(int));
    }

    // Exchange the global indices each rank needs
    int **global_needed = (int **)malloc(halo->num_neighbors * sizeof(int *));
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        global_needed[ni] = (int *)malloc(halo->recv_counts[ni] * sizeof(int));
        for (int j = 0; j < halo->recv_counts[ni]; j++) {
            int edge_idx = halo->recv_indices[ni][j];
            global_needed[ni][j] = D->neighbours[edge_idx];
        }
    }

    int **global_to_send = (int **)malloc(halo->num_neighbors * sizeof(int *));
    MPI_Request *idx_reqs = (MPI_Request *)malloc(2 * halo->num_neighbors * sizeof(MPI_Request));

    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        global_to_send[ni] = (int *)malloc(halo->send_counts[ni] * sizeof(int));
        int partner = halo->neighbor_ranks[ni];
        MPI_Irecv(global_to_send[ni], halo->send_counts[ni], MPI_INT, partner, 1,
                  MPI_COMM_WORLD, &idx_reqs[2*ni]);
        MPI_Isend(global_needed[ni], halo->recv_counts[ni], MPI_INT, partner, 1,
                  MPI_COMM_WORLD, &idx_reqs[2*ni + 1]);
    }
    MPI_Waitall(2 * halo->num_neighbors, idx_reqs, MPI_STATUSES_IGNORE);
    free(idx_reqs);

    // Convert global_to_send to local indices for our send_indices
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        for (int j = 0; j < halo->send_counts[ni]; j++) {
            int g = global_to_send[ni][j];
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

// Start non-blocking halo exchange (post receives and sends)
static void start_halo_exchange(struct domain *D, struct halo_info *halo) {
    if (halo->num_neighbors == 0) return;

    // Pack send buffer
    int offset = 0;
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        for (int j = 0; j < halo->send_counts[ni]; j++) {
            int k = halo->send_indices[ni][j];
            int edge_idx = 3*k + 0;
            halo->send_buffer[4*offset + 0] = D->height_edge_values[edge_idx];
            halo->send_buffer[4*offset + 1] = D->xmom_edge_values[edge_idx];
            halo->send_buffer[4*offset + 2] = D->ymom_edge_values[edge_idx];
            halo->send_buffer[4*offset + 3] = D->stage_edge_values[edge_idx];
            offset++;
        }
    }

    // Post all receives first (good MPI practice)
    int recv_offset = 0;
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        int partner = halo->neighbor_ranks[ni];
        MPI_Irecv(&halo->recv_buffer[4*recv_offset], 4 * halo->recv_counts[ni], MPI_DOUBLE,
                  partner, 2, MPI_COMM_WORLD, &halo->recv_requests[ni]);
        recv_offset += halo->recv_counts[ni];
    }

    // Then post all sends
    int send_offset = 0;
    for (int ni = 0; ni < halo->num_neighbors; ni++) {
        int partner = halo->neighbor_ranks[ni];
        MPI_Isend(&halo->send_buffer[4*send_offset], 4 * halo->send_counts[ni], MPI_DOUBLE,
                  partner, 2, MPI_COMM_WORLD, &halo->send_requests[ni]);
        send_offset += halo->send_counts[ni];
    }

    halo->exchange_in_progress = 1;
}

// Complete non-blocking halo exchange
static void finish_halo_exchange(struct halo_info *halo) {
    if (halo->num_neighbors == 0 || !halo->exchange_in_progress) return;

    MPI_Waitall(halo->num_neighbors, halo->recv_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(halo->num_neighbors, halo->send_requests, MPI_STATUSES_IGNORE);

    halo->exchange_in_progress = 0;
}

// Test if receives have completed (for more fine-grained overlap)
static int test_halo_recv_complete(struct halo_info *halo) {
    if (halo->num_neighbors == 0) return 1;

    int flag;
    MPI_Testall(halo->num_neighbors, halo->recv_requests, &flag, MPI_STATUSES_IGNORE);
    return flag;
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

            // Only use local neighbors for gradient
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

// HLL flux kernel for a subset of elements
static void compute_fluxes_subset_gpu(struct domain *D, int *indices, int count,
                                       double *local_max_speed_ptr) {
    double g = D->g;
    double h0 = D->minimum_allowed_height;
    double local_max_speed = *local_max_speed_ptr;

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

    #pragma omp target teams distribute parallel for \
        map(to: indices[0:count]) reduction(max: local_max_speed)
    for (int idx = 0; idx < count; idx++) {
        int k = indices[idx];
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
                h_R = h_L;
                double vn_L = uh_L * nx + vh_L * ny;
                uh_R = uh_L - 2.0 * vn_L * nx;
                vh_R = vh_L - 2.0 * vn_L * ny;
            } else if (owner == mpi_rank) {
                int ki_nb = 3*nb + 0;
                h_R = height_edge[ki_nb];
                uh_R = xmom_edge[ki_nb];
                vh_R = ymom_edge[ki_nb];
            } else {
                h_R = h_L;
                uh_R = uh_L;
                vh_R = vh_L;
            }

            double u_L, v_L, u_R, v_R;
            if (h_L > h0) { u_L = uh_L / h_L; v_L = vh_L / h_L; }
            else { u_L = 0.0; v_L = 0.0; }
            if (h_R > h0) { u_R = uh_R / h_R; v_R = vh_R / h_R; }
            else { u_R = 0.0; v_R = 0.0; }

            double un_L = u_L * nx + v_L * ny;
            double ut_L = -u_L * ny + v_L * nx;
            double un_R = u_R * nx + v_R * ny;
            double ut_R = -u_R * ny + v_R * nx;

            double c_L = sqrt(g * fmax(h_L, 0.0));
            double c_R = sqrt(g * fmax(h_R, 0.0));

            double s_L = fmin(un_L - c_L, un_R - c_R);
            double s_R = fmax(un_L + c_L, un_R + c_R);

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

    *local_max_speed_ptr = local_max_speed;
}

// Main flux computation with overlap
static double compute_fluxes_overlapped(struct domain *D, struct halo_info *halo) {
    double local_max_speed = 0.0;

    // If we have neighbors, use overlapped approach
    if (halo->num_neighbors > 0 && D->num_interior > 0) {
        // Start halo exchange for edge values
        // First, transfer edge values from GPU to CPU for packing
        #pragma omp target update from(D->height_edge_values[0:3*D->number_of_elements], \
                                        D->xmom_edge_values[0:3*D->number_of_elements], \
                                        D->ymom_edge_values[0:3*D->number_of_elements], \
                                        D->stage_edge_values[0:3*D->number_of_elements])

        // Start non-blocking halo exchange
        start_halo_exchange(D, halo);

        // Compute interior elements while halo is in flight
        compute_fluxes_subset_gpu(D, D->interior_indices, D->num_interior, &local_max_speed);

        // Wait for halo exchange to complete
        finish_halo_exchange(halo);

        // Now compute boundary elements
        compute_fluxes_subset_gpu(D, D->boundary_indices, D->num_boundary, &local_max_speed);
    } else {
        // No MPI neighbors or all elements are boundary - compute all at once
        int n = D->number_of_elements;
        double g = D->g;
        double h0 = D->minimum_allowed_height;

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
                    h_R = h_L;
                    double vn_L = uh_L * nx + vh_L * ny;
                    uh_R = uh_L - 2.0 * vn_L * nx;
                    vh_R = vh_L - 2.0 * vn_L * ny;
                } else if (owner == mpi_rank) {
                    int ki_nb = 3*nb + 0;
                    h_R = height_edge[ki_nb];
                    uh_R = xmom_edge[ki_nb];
                    vh_R = ymom_edge[ki_nb];
                } else {
                    h_R = h_L;
                    uh_R = uh_L;
                    vh_R = vh_L;
                }

                double u_L, v_L, u_R, v_R;
                if (h_L > h0) { u_L = uh_L / h_L; v_L = vh_L / h_L; }
                else { u_L = 0.0; v_L = 0.0; }
                if (h_R > h0) { u_R = uh_R / h_R; v_R = vh_R / h_R; }
                else { u_R = 0.0; v_R = 0.0; }

                double un_L = u_L * nx + v_L * ny;
                double ut_L = -u_L * ny + v_L * nx;
                double un_R = u_R * nx + v_R * ny;
                double ut_R = -u_R * ny + v_R * nx;

                double c_L = sqrt(g * fmax(h_L, 0.0));
                double c_R = sqrt(g * fmax(h_R, 0.0));

                double s_L = fmin(un_L - c_L, un_R - c_R);
                double s_R = fmax(un_L + c_L, un_R + c_R);

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
    }

    // Non-blocking global reduction for CFL
    double global_max_speed;
    MPI_Request reduce_req;
    MPI_Iallreduce(&local_max_speed, &global_max_speed, 1, MPI_DOUBLE, MPI_MAX,
                   MPI_COMM_WORLD, &reduce_req);

    // Could do other work here while reduction is in progress
    // For now, just wait
    MPI_Wait(&reduce_req, MPI_STATUS_IGNORE);

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

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

    // Each rank uses its own GPU
    int num_devices = omp_get_num_devices();
    int my_device = mpi_rank % num_devices;
    omp_set_default_device(my_device);
    CALI_MARK_BEGIN("main func");

    if (argc < 2 || argc > 7) {
        if (mpi_rank == 0) {
            fprintf(stderr, "Usage: %s N [niter] [yieldstep] [gather] [domain_km] [depth_m]\n", argv[0]);
            fprintf(stderr, "  MPI + OpenMP target version with NON-BLOCKING communication\n");
            fprintf(stderr, "  gather: 0 = distributed I/O (scalable), 1 = gather to rank 0\n");
        }
        MPI_Finalize();
        return 1;
    }

    int grid_size = atoi(argv[1]);
    int niter = (argc >= 3) ? atoi(argv[2]) : 1000;
    int yieldstep = (argc >= 4) ? atoi(argv[3]) : 100;
    int gather_mode = (argc >= 5) ? atoi(argv[4]) : 0;
    double domain_length = (argc >= 6) ? atof(argv[5]) * 1000.0 : 100000.0;
    double initial_height = (argc >= 7) ? atof(argv[6]) : 10.0;

    if (grid_size < 3) {
        if (mpi_rank == 0) fprintf(stderr, "Error: Grid size must be at least 3\n");
        MPI_Finalize();
        return 1;
    }

    int n_global = 2 * (grid_size - 1) * (grid_size - 1);
    int local_start, local_n;
    compute_partition(n_global, mpi_size, mpi_rank, &local_start, &local_n);

    if (mpi_rank == 0) {
        printf("============================================================\n");
        printf("  SHALLOW WATER - MPI + OpenMP Target (NON-BLOCKING)\n");
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
    }

    // Allocate domain
    struct domain D;
    D.number_of_elements = local_n;
    D.global_elements = n_global;
    D.local_start = local_start;
    D.g = 9.81;
    D.epsilon = 1.0e-12;
    D.minimum_allowed_height = 1.0e-6;
    D.cfl = 0.9;
    D.domain_length = domain_length;
    D.char_length = domain_length / sqrt((double)n_global / 2.0);

    int n = local_n;

    // Allocate arrays
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

    // Initialize
    generate_mesh_local(&D, grid_size);
    init_quantities(&D, initial_height);

    // Build halo exchange info
    struct halo_info halo;
    build_halo_info(&D, &halo);

    // Classify elements for overlap
    classify_elements(&D);

    MPI_Barrier(MPI_COMM_WORLD);

    // Map to GPU (including interior/boundary index arrays)
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

    // Run benchmark
    double sim_time = 0.0;
    double dt = 0.0;
    double max_speed_global;
    double t_transfer = 0.0;
    int yield_count = 0;

    if (mpi_rank == 0) printf("Running %d iterations (yieldstep=%d)...\n\n", niter, yieldstep);

    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = omp_get_wtime();

    for (int iter = 0; iter < niter; iter++) {
        compute_gradients_gpu(&D);
        extrapolate_second_order_gpu(&D);

        // Overlapped flux computation with non-blocking halo exchange
        max_speed_global = compute_fluxes_overlapped(&D, &halo);

        if (max_speed_global > D.epsilon) {
            dt = D.cfl * D.char_length / max_speed_global;
        } else {
            dt = D.cfl * D.char_length / sqrt(D.g * initial_height);
        }

        protect_gpu(&D);
        update_gpu(&D, dt);
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
    double t_compute_total = omp_get_wtime() - t0;
    double t_compute_pure = t_compute_total - t_transfer;

    // Cleanup GPU
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

    // Results
    if (mpi_rank == 0) {
        printf("\n\n");
        printf("============================================================\n");
        printf("  RESULTS (NON-BLOCKING MPI + YIELDSTEP)\n");
        printf("============================================================\n\n");

        double time_per_step = t_compute_total / (double)niter;
        double steps_per_second = 1.0 / time_per_step;
        dt = sim_time / (double)niter;
        double target_time = 2.0 * 24.0 * 3600.0;
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
    free(D.interior_indices); free(D.boundary_indices);

    // Cleanup halo
    if (halo.num_neighbors > 0) {
        for (int ni = 0; ni < halo.num_neighbors; ni++) {
            free(halo.send_indices[ni]);
            free(halo.recv_indices[ni]);
        }
        free(halo.neighbor_ranks);
        free(halo.send_counts);
        free(halo.recv_counts);
        free(halo.send_indices);
        free(halo.recv_indices);
        free(halo.send_buffer);
        free(halo.recv_buffer);
        free(halo.send_requests);
        free(halo.recv_requests);
    }

    MPI_Finalize();
    CALI_MARK_END("main func");
    return 0;
}
