#ifndef MODEL_H
#define MODEL_H

#include <stddef.h>

typedef struct {
    double *a;
    double *b;
    double *out;
    size_t n;
} Model;

void model_init(Model *m, size_t n);
void model_free(Model *m);

void model_compute_serial(Model *m);
void model_compute_omp(Model *m);
void model_compute_omp_gpu(Model *m);

#endif

