#include <mpi.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define ARRAY_SIZE 1000000
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

int main(int argc, char** argv) {
    int rank, size;
    
    // Initialize MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size != 2) {
        if (rank == 0) {
            printf("This program requires exactly 2 MPI processes\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Set GPU device based on rank
    CUDA_CHECK(cudaSetDevice(rank));
    
    // Allocate GPU memory
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, ARRAY_SIZE * sizeof(float)));
    
    if (rank == 0) {
        printf("=== CUDA-Aware MPI Example ===\n");
        printf("Sending data directly from GPU memory to GPU memory\n");
        
        // Initialize data on GPU 0
        float *h_data = (float*)malloc(ARRAY_SIZE * sizeof(float));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            h_data[i] = (float)i;
        }
        CUDA_CHECK(cudaMemcpy(d_data, h_data, ARRAY_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
        free(h_data);
        
        double start_time = MPI_Wtime();
        
        // CUDA-Aware MPI: Send GPU memory directly
        MPI_Send(d_data, ARRAY_SIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        
        double end_time = MPI_Wtime();
        printf("Rank 0: Sent %d floats directly from GPU memory\n", ARRAY_SIZE);
        printf("Rank 0: Time taken: %f seconds\n", end_time - start_time);
        
    } else if (rank == 1) {
        double start_time = MPI_Wtime();
        
        // CUDA-Aware MPI: Receive directly into GPU memory
        MPI_Recv(d_data, ARRAY_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, 
                 MPI_STATUS_IGNORE);
        
        double end_time = MPI_Wtime();
        
        // Verify data (check first and last elements)
        float first, last;
        CUDA_CHECK(cudaMemcpy(&first, d_data, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last, d_data + ARRAY_SIZE - 1, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        printf("Rank 1: Received %d floats directly into GPU memory\n", ARRAY_SIZE);
        printf("Rank 1: First element: %f, Last element: %f\n", first, last);
        printf("Rank 1: Time taken: %f seconds\n", end_time - start_time);
        printf("Rank 1: Data verification: %s\n", 
               (first == 0.0f && last == (float)(ARRAY_SIZE-1)) ? "PASSED" : "FAILED");
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    
    MPI_Finalize();
    return 0;
}
