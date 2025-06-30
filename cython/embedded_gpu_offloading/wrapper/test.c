#include <stdio.h>
#include <omp.h>

void test_function(){

    int n_reps = 10;
    int n_elements = 12000;  // Adjust as needed (e.g. 10000 x 10000 = 100M elements)
    printf("doing a big ass saxpy with %d elements \n", n_elements);
    float alpha = 2.5f;

    size_t n_total = (size_t)n_elements * n_elements;

    float *x = (float *)malloc(sizeof(float) * n_total);
    float *y = (float *)malloc(sizeof(float) * n_total);
    float *out = (float *)malloc(sizeof(float) * n_total);

    if (!x || !y || !out) {
        fprintf(stderr, "Allocation failed!\n");
        exit(1);
    }

    // Initialize
    for (size_t i = 0; i < n_total; ++i) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }
for(int rep = 0; rep < n_reps; rep++){
printf("Doing rep %d of %d \n", rep, n_reps);
    // Offload SAXPY: out[i] = alpha * x[i] + y[i]
    #pragma omp target teams distribute parallel for map(to: x[:n_total], y[:n_total]) map(from: out[:n_total])
    for (size_t i = 0; i < n_total; ++i) {
        out[i] = alpha * x[i] + y[i];
    }
  }

    // Check a few results
    for (int i = 0; i < 5; ++i) {
        printf("out[%d] = %f\n", i, out[i]);
    }

    free(x);
    free(y);
    free(out);

} 

