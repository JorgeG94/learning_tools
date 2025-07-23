#include "model.h"
#include <stdio.h>

int main() {
    Model m;
    size_t n = 10;

    model_init(&m, n);

    printf("== Serial computation ==\n");
    model_compute_serial(&m);
    for (size_t i = 0; i < n; ++i)
        printf("out[%zu] = %f\n", i, m.out[i]);

    printf("\n== OpenMP parallel computation ==\n");
    model_compute_omp(&m);
    for (size_t i = 0; i < n; ++i)
        printf("out[%zu] = %f\n", i, m.out[i]);

    printf("\n== OpenMP parallel computation GPU ==\n");
    model_compute_omp_gpu(&m);
    for (size_t i = 0; i < n; ++i)
        printf("out[%zu] = %f\n", i, m.out[i]);
    model_free(&m);

    return 0;
}

