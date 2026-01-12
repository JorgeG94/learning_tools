// sw_cfl2.c - Shallow water mini app with SECOND-ORDER extrapolation
// Includes gradient reconstruction with minmod limiter
// This is much closer to real ANUGA computational patterns

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

    // Edge values (3 per triangle) - now computed via 2nd order extrapolation
    double *stage_edge_values;
    double *height_edge_values;
    double *xmom_edge_values;
    double *ymom_edge_values;

    // Gradients at centroids (for 2nd order reconstruction)
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

    // CFL parameters
    double cfl;
    double domain_length;
    double char_length;
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

static void print_progress(int current, int total, double elapsed, double sim_time, int yields) {
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
static void transfer_yieldstep(struct domain *D, struct verbose_output *out, double *t_transfer) {
    int n = D->number_of_elements;
    double t0 = omp_get_wtime();

    // Transfer primary quantities (4 arrays)
    #pragma omp target update from(D->stage_centroid_values[0:n])
    #pragma omp target update from(D->height_centroid_values[0:n])
    #pragma omp target update from(D->xmom_centroid_values[0:n])
    #pragma omp target update from(D->ymom_centroid_values[0:n])

    // Copy to output
    for (int k = 0; k < n; k++) {
        out->stage[k] = D->stage_centroid_values[k];
        out->height[k] = D->height_centroid_values[k];
        out->xmom[k] = D->xmom_centroid_values[k];
        out->ymom[k] = D->ymom_centroid_values[k];
    }

    // Transfer update arrays (3 arrays)
    #pragma omp target update from(D->stage_explicit_update[0:n])
    #pragma omp target update from(D->xmom_explicit_update[0:n])
    #pragma omp target update from(D->ymom_explicit_update[0:n])

    for (int k = 0; k < n; k++) {
        out->stage_update[k] = D->stage_explicit_update[k];
        out->xmom_update[k] = D->xmom_explicit_update[k];
        out->ymom_update[k] = D->ymom_explicit_update[k];
    }

    // Transfer edge arrays (2 arrays × 3n)
    #pragma omp target update from(D->stage_edge_values[0:3*n])
    #pragma omp target update from(D->height_edge_values[0:3*n])

    for (int k = 0; k < 3*n; k++) {
        out->stage_edge[k] = D->stage_edge_values[k];
        out->height_edge[k] = D->height_edge_values[k];
    }

    // Transfer max_speed
    #pragma omp target update from(D->max_speed[0:n])
    for (int k = 0; k < n; k++) {
        out->max_speed_elem[k] = D->max_speed[k];
    }

    // Compute derived quantities on host
    double g = D->g;
    double eps = D->epsilon;
    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        double h = out->height[k];
        if (h > eps) {
            double u = out->xmom[k] / h;
            double v = out->ymom[k] / h;
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

    *t_transfer += omp_get_wtime() - t0;
}

static void generate_mesh(struct domain *D, int grid_size) {
    int nx = grid_size;
    int ny = grid_size;
    int n = D->number_of_elements;
    double dx = D->domain_length / (nx - 1);
    double dy = D->domain_length / (ny - 1);

    double area = 0.5 * dx * dy;
    double edgelen = dx;
    double radius = area / (1.5 * edgelen);

    #pragma omp parallel for
    for (int k = 0; k < n; k++) {
        D->areas[k] = area;
        D->radii[k] = radius;

        for (int i = 0; i < 3; i++) {
            D->edgelengths[3*k + i] = edgelen;
        }

        D->normals[6*k + 0] = 1.0;  D->normals[6*k + 1] = 0.0;
        D->normals[6*k + 2] = 0.0;  D->normals[6*k + 3] = 1.0;
        D->normals[6*k + 4] = -0.707; D->normals[6*k + 5] = -0.707;

        int cell = k / 2;
        int tri_in_cell = k % 2;
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

        // Centroid
        D->centroid_x[k] = (x0 + x1 + x2) / 3.0;
        D->centroid_y[k] = (y0 + y1 + y2) / 3.0;

        // Edge midpoints
        D->edge_midpoint_x[3*k + 0] = (x1 + x2) / 2.0;
        D->edge_midpoint_y[3*k + 0] = (y1 + y2) / 2.0;
        D->edge_midpoint_x[3*k + 1] = (x0 + x2) / 2.0;
        D->edge_midpoint_y[3*k + 1] = (y0 + y2) / 2.0;
        D->edge_midpoint_x[3*k + 2] = (x0 + x1) / 2.0;
        D->edge_midpoint_y[3*k + 2] = (y0 + y1) / 2.0;

        // Neighbours
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

        // Gradients
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

// Compute gradients using weighted least-squares
static void compute_gradients_gpu(struct domain *D) {
    int n = D->number_of_elements;

    double *stage_c = D->stage_centroid_values;
    double *xmom_c = D->xmom_centroid_values;
    double *ymom_c = D->ymom_centroid_values;
    double *height_c = D->height_centroid_values;
    double *cx = D->centroid_x;
    double *cy = D->centroid_y;
    int *neighbours = D->neighbours;

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
            if (nb >= 0) {
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

// Second-order extrapolation with minmod limiter
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

            // Unlimited extrapolation
            double dstage = sgx * dx_e + sgy * dy_e;
            double dxmom = xgx * dx_e + xgy * dy_e;
            double dymom = ygx * dx_e + ygy * dy_e;
            double dheight = hgx * dx_e + hgy * dy_e;

            // Apply minmod limiter
            int nb = neighbours[ki];
            if (nb >= 0) {
                double dstage_nb = stage_c[nb] - stage_k;
                double dxmom_nb = xmom_c[nb] - xmom_k;
                double dymom_nb = ymom_c[nb] - ymom_k;
                double dheight_nb = height_c[nb] - height_k;

                // Minmod
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

// HLL flux computation with proper rotation and physical fluxes
// This is much closer to real ANUGA's flux computation
static double compute_fluxes_gpu(struct domain *D) {
    int n = D->number_of_elements;
    double g = D->g;
    double epsilon = D->epsilon;
    double h0 = D->minimum_allowed_height;
    double global_max_speed = 0.0;

    double *stage_edge = D->stage_edge_values;
    double *height_edge = D->height_edge_values;
    double *xmom_edge = D->xmom_edge_values;
    double *ymom_edge = D->ymom_edge_values;
    double *bed_c = D->bed_centroid_values;
    double *stage_update = D->stage_explicit_update;
    double *xmom_update = D->xmom_explicit_update;
    double *ymom_update = D->ymom_explicit_update;
    int *neighbours = D->neighbours;
    double *edgelengths = D->edgelengths;
    double *normals = D->normals;
    double *areas = D->areas;
    double *max_speed = D->max_speed;

    #pragma omp target teams distribute parallel for \
        reduction(max: global_max_speed)
    for (int k = 0; k < n; k++) {
        double stage_flux_sum = 0.0;
        double xmom_flux_sum = 0.0;
        double ymom_flux_sum = 0.0;
        double speed_max = 0.0;

        for (int i = 0; i < 3; i++) {
            int ki = 3*k + i;
            int nb = neighbours[ki];

            // Edge normal (outward pointing)
            double nx = normals[6*k + 2*i];
            double ny = normals[6*k + 2*i + 1];
            double el = edgelengths[ki];

            // Left state (this element)
            double h_L = height_edge[ki];
            double uh_L = xmom_edge[ki];
            double vh_L = ymom_edge[ki];

            // Right state (neighbor or boundary)
            double h_R, uh_R, vh_R;
            if (nb >= 0) {
                // Find matching edge on neighbor (simplified: use edge 0)
                int ki_nb = 3*nb + 0;
                h_R = height_edge[ki_nb];
                uh_R = xmom_edge[ki_nb];
                vh_R = ymom_edge[ki_nb];
            } else {
                // Reflective boundary: reflect normal momentum
                h_R = h_L;
                // Reflect velocity: v_R = v_L - 2*(v_L.n)*n
                double vn_L = (uh_L * nx + vh_L * ny);
                uh_R = uh_L - 2.0 * vn_L * nx;
                vh_R = vh_L - 2.0 * vn_L * ny;
            }

            // Compute velocities (with desingularization)
            double u_L, v_L, u_R, v_R;
            if (h_L > h0) {
                u_L = uh_L / h_L;
                v_L = vh_L / h_L;
            } else {
                u_L = 0.0;
                v_L = 0.0;
            }
            if (h_R > h0) {
                u_R = uh_R / h_R;
                v_R = vh_R / h_R;
            } else {
                u_R = 0.0;
                v_R = 0.0;
            }

            // Rotate to edge-normal coordinates
            // un = u*nx + v*ny (normal velocity)
            // ut = -u*ny + v*nx (tangential velocity)
            double un_L = u_L * nx + v_L * ny;
            double ut_L = -u_L * ny + v_L * nx;
            double un_R = u_R * nx + v_R * ny;
            double ut_R = -u_R * ny + v_R * nx;

            // Wave speeds
            double c_L = sqrt(g * fmax(h_L, 0.0));
            double c_R = sqrt(g * fmax(h_R, 0.0));

            // HLL wave speed estimates (Einfeldt)
            double s_L = fmin(un_L - c_L, un_R - c_R);
            double s_R = fmax(un_L + c_L, un_R + c_R);

            // Physical fluxes in normal direction
            // F = [h*un, h*un^2 + 0.5*g*h^2, h*un*ut]
            double flux_mass_L = h_L * un_L;
            double flux_mom_n_L = h_L * un_L * un_L + 0.5 * g * h_L * h_L;
            double flux_mom_t_L = h_L * un_L * ut_L;

            double flux_mass_R = h_R * un_R;
            double flux_mom_n_R = h_R * un_R * un_R + 0.5 * g * h_R * h_R;
            double flux_mom_t_R = h_R * un_R * ut_R;

            // Conserved variables
            double q_mass_L = h_L;
            double q_mom_n_L = h_L * un_L;
            double q_mom_t_L = h_L * ut_L;

            double q_mass_R = h_R;
            double q_mom_n_R = h_R * un_R;
            double q_mom_t_R = h_R * ut_R;

            // HLL flux
            double flux_mass, flux_mom_n, flux_mom_t;

            if (s_L >= 0.0) {
                // Supersonic flow to the right - use left flux
                flux_mass = flux_mass_L;
                flux_mom_n = flux_mom_n_L;
                flux_mom_t = flux_mom_t_L;
            } else if (s_R <= 0.0) {
                // Supersonic flow to the left - use right flux
                flux_mass = flux_mass_R;
                flux_mom_n = flux_mom_n_R;
                flux_mom_t = flux_mom_t_R;
            } else {
                // Subsonic - HLL average
                double denom = s_R - s_L;
                if (fabs(denom) < 1.0e-15) denom = 1.0e-15;

                flux_mass = (s_R * flux_mass_L - s_L * flux_mass_R
                           + s_L * s_R * (q_mass_R - q_mass_L)) / denom;
                flux_mom_n = (s_R * flux_mom_n_L - s_L * flux_mom_n_R
                            + s_L * s_R * (q_mom_n_R - q_mom_n_L)) / denom;
                flux_mom_t = (s_R * flux_mom_t_L - s_L * flux_mom_t_R
                            + s_L * s_R * (q_mom_t_R - q_mom_t_L)) / denom;
            }

            // Rotate momentum flux back to (x,y) coordinates
            // flux_x = flux_n * nx - flux_t * ny
            // flux_y = flux_n * ny + flux_t * nx
            double flux_xmom = flux_mom_n * nx - flux_mom_t * ny;
            double flux_ymom = flux_mom_n * ny + flux_mom_t * nx;

            // Accumulate (negative because outward normal)
            stage_flux_sum -= flux_mass * el;
            xmom_flux_sum -= flux_xmom * el;
            ymom_flux_sum -= flux_ymom * el;

            // Track max wave speed for CFL
            double local_speed = fmax(fabs(s_L), fabs(s_R));
            speed_max = fmax(speed_max, local_speed);
        }

        // Divide by area to get update rate
        double inv_area = 1.0 / areas[k];
        stage_update[k] = stage_flux_sum * inv_area;
        xmom_update[k] = xmom_flux_sum * inv_area;
        ymom_update[k] = ymom_flux_sum * inv_area;
        max_speed[k] = speed_max;

        global_max_speed = fmax(global_max_speed, speed_max);
    }

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
                if (stage_c[k] < bmin) {
                    stage_c[k] = bmin;
                }
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
    if (argc < 2 || argc > 6) {
        fprintf(stderr, "Usage: %s N [niter] [yieldstep] [domain_length_km] [initial_height_m]\n", argv[0]);
        fprintf(stderr, "  N                = grid size (creates 2*(N-1)^2 triangles)\n");
        fprintf(stderr, "  niter            = number of iterations (default: 1000)\n");
        fprintf(stderr, "  yieldstep        = transfer output every N iters (default: 100)\n");
        fprintf(stderr, "  domain_length_km = physical domain size in km (default: 100)\n");
        fprintf(stderr, "  initial_height_m = initial water depth in meters (default: 10)\n");
        fprintf(stderr, "\n  SECOND-ORDER extrapolation with minmod limiter + yieldstep transfers\n");
        return 1;
    }

    int grid_size = atoi(argv[1]);
    int niter = (argc >= 3) ? atoi(argv[2]) : 1000;
    int yieldstep = (argc >= 4) ? atoi(argv[3]) : 100;
    double domain_length = (argc >= 5) ? atof(argv[4]) * 1000.0 : 100000.0;
    double initial_height = (argc >= 6) ? atof(argv[5]) : 10.0;

    if (grid_size < 3) {
        fprintf(stderr, "Error: Grid size must be at least 3\n");
        return 1;
    }

    int n = 2 * (grid_size - 1) * (grid_size - 1);
    double target_time = 2.0 * 24.0 * 3600.0;  // 2 days

    printf("============================================================\n");
    printf("  SHALLOW WATER CFL TIMING - SECOND ORDER (C version)\n");
    printf("============================================================\n\n");

    printf("Physical Setup:\n");
    printf("  Domain size:      %.2f km x %.2f km\n", domain_length/1000.0, domain_length/1000.0);
    printf("  Initial depth:    %.2f m\n", initial_height);
    printf("  Wave speed:       %.2f m/s\n\n", sqrt(9.81 * initial_height));

    printf("Mesh:\n");
    printf("  Grid:             %d x %d\n", grid_size, grid_size);
    printf("  Triangles:        %d\n\n", n);

    printf("Reconstruction:\n");
    printf("  Order:            SECOND (gradient + minmod limiter)\n");
    printf("  Yieldstep:        %d (output every %d iterations)\n\n", yieldstep, yieldstep);

    // Calculate output size per yieldstep
    double mb_per_array = (double)(n * sizeof(double)) / (1024.0 * 1024.0);
    double mb_per_edge = (double)(3 * n * sizeof(double)) / (1024.0 * 1024.0);
    // 12 centroid arrays + 2 edge arrays
    double output_mb = 12.0 * mb_per_array + 2.0 * mb_per_edge;
    printf("Output per yieldstep: %.2f MB (12 centroid + 2 edge arrays)\n\n", output_mb);

    // Allocate domain
    struct domain D;
    D.number_of_elements = n;
    D.g = 9.81;
    D.epsilon = 1.0e-12;
    D.minimum_allowed_height = 1.0e-6;
    D.cfl = 0.9;
    D.domain_length = domain_length;
    D.char_length = domain_length / sqrt((double)n / 2.0);

    printf("  Element size:     %.4f m\n\n", D.char_length);

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
    double t0 = omp_get_wtime();
    generate_mesh(&D, grid_size);
    init_quantities(&D, initial_height);
    double t_init = omp_get_wtime() - t0;

    // Map to GPU
    t0 = omp_get_wtime();
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
        D.neighbours[0:3*n], D.edgelengths[0:3*n], D.normals[0:6*n], \
        D.areas[0:n], D.radii[0:n], D.max_speed[0:n])
    double t_to_gpu = omp_get_wtime() - t0;

    // Run benchmark
    double sim_time = 0.0;
    double dt = 0.0;
    double max_speed_global;
    double t_transfer = 0.0;
    int yield_count = 0;

    printf("Running %d iterations (2nd order + yieldstep output)...\n\n", niter);

    t0 = omp_get_wtime();
    for (int iter = 0; iter < niter; iter++) {
        // 1. Compute gradients
        compute_gradients_gpu(&D);

        // 2. Second-order extrapolation with limiter
        extrapolate_second_order_gpu(&D);

        // 3. Compute fluxes
        max_speed_global = compute_fluxes_gpu(&D);

        // CFL timestep
        if (max_speed_global > D.epsilon) {
            dt = D.cfl * D.char_length / max_speed_global;
        } else {
            dt = D.cfl * D.char_length / sqrt(D.g * initial_height);
        }

        // 4. Protect
        protect_gpu(&D);

        // 5. Update
        update_gpu(&D, dt);
        sim_time += dt;

        // 6. Yieldstep transfer
        if ((iter + 1) % yieldstep == 0) {
            transfer_yieldstep(&D, &out, &t_transfer);
            yield_count++;
        }

        if ((iter + 1) % 100 == 0) {
            print_progress(iter + 1, niter, omp_get_wtime() - t0, sim_time, yield_count);
        }
    }
    double t_compute_total = omp_get_wtime() - t0;
    double t_compute_pure = t_compute_total - t_transfer;
    printf("\n\n");

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
        D.neighbours[0:3*n], D.edgelengths[0:3*n], D.normals[0:6*n], \
        D.areas[0:n], D.radii[0:n], D.max_speed[0:n])

    // Results
    double time_per_step = t_compute_total / (double)niter;
    double steps_per_second = 1.0 / time_per_step;
    dt = sim_time / (double)niter;
    double estimated_steps = target_time / dt;
    double estimated_wallclock = estimated_steps * time_per_step;

    printf("============================================================\n");
    printf("  BENCHMARK RESULTS (SECOND ORDER + YIELDSTEP)\n");
    printf("============================================================\n\n");

    printf("Ran %d iterations:\n", niter);
    printf("  Total time:       %.4f s\n", t_compute_total);
    printf("  Pure compute:     %.4f s (%.4f ms/iter)\n", t_compute_pure, t_compute_pure * 1000.0 / niter);
    printf("  Yieldstep xfer:   %.4f s (%d yields, %.4f ms/yield)\n",
           t_transfer, yield_count, yield_count > 0 ? t_transfer * 1000.0 / yield_count : 0.0);
    printf("  Transfer overhead: %.2f%% of compute time\n\n", 100.0 * t_transfer / t_compute_pure);
    printf("  Time per step:    %.6f ms (including transfers)\n", time_per_step * 1000.0);
    printf("  Steps per second: %.2f\n\n", steps_per_second);

    printf("CFL Timestep:\n");
    printf("  Average dt:       %.4e s\n", dt);
    printf("  Simulated time:   %.4f s\n\n", sim_time);

    printf("============================================================\n");
    printf("  2-DAY SIMULATION ESTIMATE\n");
    printf("============================================================\n\n");

    printf("  Estimated steps:  %.4e\n\n", estimated_steps);
    printf("  Estimated wall-clock time:\n");

    if (estimated_wallclock < 60.0) {
        printf("                    %.2f seconds\n", estimated_wallclock);
    } else if (estimated_wallclock < 3600.0) {
        printf("                    %.2f minutes\n", estimated_wallclock / 60.0);
    } else if (estimated_wallclock < 86400.0) {
        printf("                    %.2f hours\n", estimated_wallclock / 3600.0);
    } else {
        printf("                    %.2f days\n", estimated_wallclock / 86400.0);
    }
    printf("\n");

    if (estimated_wallclock < 3600.0) {
        printf("  EXCELLENT - Under 1 hour\n");
    } else if (estimated_wallclock < 86400.0) {
        printf("  GOOD - Under 1 day\n");
    } else if (estimated_wallclock < 7.0 * 86400.0) {
        printf("  ACCEPTABLE - Under 1 week\n");
    } else {
        printf("  CHALLENGING - More than 1 week\n");
    }
    printf("\n");

    // Cleanup
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
    free(D.neighbours); free(D.edgelengths); free(D.normals);
    free(D.areas); free(D.radii); free(D.max_speed);

    // Cleanup verbose output
    free(out.stage); free(out.height); free(out.xmom); free(out.ymom);
    free(out.xvel); free(out.yvel); free(out.speed); free(out.froude);
    free(out.stage_update); free(out.xmom_update); free(out.ymom_update);
    free(out.stage_edge); free(out.height_edge); free(out.max_speed_elem);

    return 0;
}
