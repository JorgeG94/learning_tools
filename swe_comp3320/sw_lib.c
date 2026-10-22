// sw_lib.c -- the shallow-water model as a small shared library for Python
// Data lives on the device for the whole run (target enter data); Python asks
// for eta only when it wants to draw a frame (target update from). That copy
// is the MOVE question in action: draw less often, run faster.
//
// GPU (NVIDIA): nvc -O3 -mp=gpu -gpu=nordc -shared -fPIC sw_lib.c -o libsw.so
// CPU (any):    gcc -O3 -fopenmp -shared -fPIC sw_lib.c -lm -o libsw.so
#include <math.h>
#include <stdlib.h>
#include <omp.h>

#define E(i,j) eta[(size_t)(j)*nx + (i)]
#define U(i,j) u[(size_t)(j)*(nx+1) + (i)]
#define V(i,j) v[(size_t)(j)*nx + (i)]

static int nx, ny;
static double a, b;
static double *eta, *u, *v;

// Set up an n x n grid with a Gaussian bump at (cx, cy) (fractions of the box)
int sw_init(int n, double cx, double cy) {
  nx = ny = n;
  const double dx = 1000.0, g = 9.81, H = 100.0;
  const double dt = 0.5 * dx / sqrt(g * H) / sqrt(2.0);
  a = g * dt / dx;  b = H * dt / dx;
  eta = malloc((size_t)nx * ny * sizeof *eta);
  u = calloc((size_t)(nx + 1) * ny, sizeof *u);
  v = calloc((size_t)nx * (ny + 1), sizeof *v);
  if (!eta || !u || !v) return -1;
  const double s = 8.0 * dx;
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++) {
      double x = (i + 0.5) * dx - cx * nx * dx, y = (j + 0.5) * dx - cy * ny * dx;
      E(i,j) = exp(-(x * x + y * y) / (2.0 * s * s));
    }
  #pragma omp target enter data map(to: eta[0:nx*ny], u[0:(nx+1)*ny], v[0:nx*(ny+1)])
  return 0;
}

// Add another drop (on the host copy, then push it to the device)
void sw_drop(double cx, double cy, double height) {
  #pragma omp target update from(eta[0:nx*ny])
  const double dx = 1000.0, s = 8.0 * dx;
  for (int j = 0; j < ny; j++)
    for (int i = 0; i < nx; i++) {
      double x = (i + 0.5) * dx - cx * nx * dx, y = (j + 0.5) * dx - cy * ny * dx;
      E(i,j) += height * exp(-(x * x + y * y) / (2.0 * s * s));
    }
  #pragma omp target update to(eta[0:nx*ny])
}

// Advance nsteps time steps entirely on the device
void sw_step(int nsteps) {
  for (int n = 0; n < nsteps; n++) {
    #pragma omp target teams distribute parallel for collapse(2)
    for (int j = 0; j < ny; j++)
      for (int i = 1; i < nx; i++)
        U(i,j) -= a * (E(i,j) - E(i-1,j));
    #pragma omp target teams distribute parallel for collapse(2)
    for (int j = 1; j < ny; j++)
      for (int i = 0; i < nx; i++)
        V(i,j) -= a * (E(i,j) - E(i,j-1));
    #pragma omp target teams distribute parallel for collapse(2)
    for (int j = 0; j < ny; j++)
      for (int i = 0; i < nx; i++)
        E(i,j) -= b * (U(i+1,j) - U(i,j) + V(i,j+1) - V(i,j));
  }
}

// Copy eta device -> host into Python's buffer (row-major ny x nx); returns sum(eta)
double sw_get_eta(double *out) {
  #pragma omp target update from(eta[0:nx*ny])
  double m = 0.0;
  for (size_t k = 0; k < (size_t)nx * ny; k++) { out[k] = eta[k]; m += eta[k]; }
  return m;
}

// 1 if the kernels run on the host (no GPU / no offload build), 0 on a device
int sw_on_host(void) {
  int h = 1;
  #pragma omp target map(from: h)
  h = omp_is_initial_device();
  return h;
}

double sw_dt(void) { return 0.5 * 1000.0 / sqrt(9.81 * 100.0) / sqrt(2.0); }

void sw_finalize(void) {
  #pragma omp target exit data map(delete: eta[0:nx*ny], u[0:(nx+1)*ny], v[0:nx*(ny+1)])
  free(eta); free(u); free(v);
}
