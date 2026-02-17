#include <cuda_runtime.h>

extern "C" void launch_mega_kernel(
    double *a1, double *b1, double *c1, double va1, double vb1, double alpha1, int n1,
    double *a2, double *b2, double *c2, double va2, double vb2, double alpha2, int n2,
    double *a3, double *b3, double *c3, double va3, double vb3, double alpha3, int n3,
    double *a4, double *b4, double *c4, double va4, double vb4, double alpha4, int n4,
    double *a5, double *b5, double *c5, double va5, double vb5, double alpha5, int n5,
    double *a6, double *b6, double *c6, double va6, double vb6, double alpha6, int n6,
    int nmax, int max_it);

__global__ void mega_kernel(
    double *a1, double *b1, double *c1, double va1, double vb1, double alpha1, int n1,
    double *a2, double *b2, double *c2, double va2, double vb2, double alpha2, int n2,
    double *a3, double *b3, double *c3, double va3, double vb3, double alpha3, int n3,
    double *a4, double *b4, double *c4, double va4, double vb4, double alpha4, int n4,
    double *a5, double *b5, double *c5, double va5, double vb5, double alpha5, int n5,
    double *a6, double *b6, double *c6, double va6, double vb6, double alpha6, int n6,
    int nmax, int max_it)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nmax) return;

    if (idx < n1) { a1[idx] = va1; b1[idx] = vb1; c1[idx] = 0.0; }
    if (idx < n2) { a2[idx] = va2; b2[idx] = vb2; c2[idx] = 0.0; }
    if (idx < n3) { a3[idx] = va3; b3[idx] = vb3; c3[idx] = 0.0; }
    if (idx < n4) { a4[idx] = va4; b4[idx] = vb4; c4[idx] = 0.0; }
    if (idx < n5) { a5[idx] = va5; b5[idx] = vb5; c5[idx] = 0.0; }
    if (idx < n6) { a6[idx] = va6; b6[idx] = vb6; c6[idx] = 0.0; }

    for (int it = 0; it < max_it; it++) {
        if (idx < n1) c1[idx] = alpha1 * a1[idx] + b1[idx];
        if (idx < n2) c2[idx] = alpha2 * a2[idx] + b2[idx];
        if (idx < n3) c3[idx] = alpha3 * a3[idx] + b3[idx];
        if (idx < n4) c4[idx] = alpha4 * a4[idx] + b4[idx];
        if (idx < n5) c5[idx] = alpha5 * a5[idx] + b5[idx];
        if (idx < n6) c6[idx] = alpha6 * a6[idx] + b6[idx];
    }
}

extern "C" void launch_mega_kernel(
    double *a1, double *b1, double *c1, double va1, double vb1, double alpha1, int n1,
    double *a2, double *b2, double *c2, double va2, double vb2, double alpha2, int n2,
    double *a3, double *b3, double *c3, double va3, double vb3, double alpha3, int n3,
    double *a4, double *b4, double *c4, double va4, double vb4, double alpha4, int n4,
    double *a5, double *b5, double *c5, double va5, double vb5, double alpha5, int n5,
    double *a6, double *b6, double *c6, double va6, double vb6, double alpha6, int n6,
    int nmax, int max_it)
{
    int blockSize = 256;
    int gridSize = (nmax + blockSize - 1) / blockSize;
    mega_kernel<<<gridSize, blockSize>>>(
        a1, b1, c1, va1, vb1, alpha1, n1,
        a2, b2, c2, va2, vb2, alpha2, n2,
        a3, b3, c3, va3, vb3, alpha3, n3,
        a4, b4, c4, va4, vb4, alpha4, n4,
        a5, b5, c5, va5, vb5, alpha5, n5,
        a6, b6, c6, va6, vb6, alpha6, n6,
        nmax, max_it);
    cudaDeviceSynchronize();
}
