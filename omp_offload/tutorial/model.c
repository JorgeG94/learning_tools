#include "model.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

void model_init(Model *m, size_t n) {
    m->n = n;
    m->a = malloc(n * sizeof(double));
    m->b = malloc(n * sizeof(double));
    m->out = malloc(n * sizeof(double));

    for (size_t i = 0; i < n; ++i) {
        m->a[i] = i * 1.0;
        m->b[i] = 2.0 + i;
        m->out[i] = 0.0;
    }
}

void model_free(Model *m) {
    free(m->a);
    free(m->b);
    free(m->out);
}

void model_compute_serial(Model *m) {
    for (size_t i = 0; i < m->n; ++i) {
        m->out[i] = m->a[i] * m->b[i] + m->a[i];
    }
}

void model_compute_omp(Model *m) {
#pragma omp parallel for
    for (size_t i = 0; i < m->n; ++i) {
        m->out[i] = m->a[i] * m->b[i] + m->a[i];
    }
}

void model_compute_omp_gpu(Model *m) {
    size_t n = m->n;

    #pragma omp target teams distribute parallel for \
    map(to: m[0:1], m->a[0:n], m->b[0:n]) map(from: m->out[0:n])
    for (size_t i = 0; i < n; ++i) {
        m->out[i] = m->a[i] * m->b[i] + m->a[i];
    }
}

