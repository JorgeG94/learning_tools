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
    
    // Allocate host memory for staging (required for non-CUDA-aware MPI)
    float *h_buffer = (float*)malloc(ARRAY_SIZE * sizeof(float));
    
    if (rank == 0) {
        printf("=== Non-CUDA-Aware MPI Example ===\n");
        printf("Using host memory staging for GPU-to-GPU communication\n");
        
        // Initialize data on GPU 0
        float *h_init = (float*)malloc(ARRAY_SIZE * sizeof(float));
        for (int i = 0; i < ARRAY_SIZE; i++) {
            h_init[i] = (float)i;
        }
        CUDA_CHECK(cudaMemcpy(d_data, h_init, ARRAY_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
        free(h_init);
        
        double start_time = MPI_Wtime();
        
        // Non-CUDA-Aware MPI: Must copy to host first, then send
        printf("Rank 0: Copying data from GPU to host...\n");
        CUDA_CHECK(cudaMemcpy(h_buffer, d_data, ARRAY_SIZE * sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        printf("Rank 0: Sending data from host memory...\n");
        MPI_Send(h_buffer, ARRAY_SIZE, MPI_FLOAT, 1, 0, MPI_COMM_WORLD);
        
        double end_time = MPI_Wtime();
        printf("Rank 0: Completed GPU->Host->Network transfer\n");
        printf("Rank 0: Time taken: %f seconds\n", end_time - start_time);
        
    } else if (rank == 1) {
        double start_time = MPI_Wtime();
        
        // Non-CUDA-Aware MPI: Receive into host memory first
        printf("Rank 1: Receiving data into host memory...\n");
        MPI_Recv(h_buffer, ARRAY_SIZE, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, 
                 MPI_STATUS_IGNORE);
        
        printf("Rank 1: Copying data from host to GPU...\n");
        CUDA_CHECK(cudaMemcpy(d_data, h_buffer, ARRAY_SIZE * sizeof(float), 
                             cudaMemcpyHostToDevice));
        
        double end_time = MPI_Wtime();
        
        // Verify data (check first and last elements)
        float first, last;
        CUDA_CHECK(cudaMemcpy(&first, d_data, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&last, d_data + ARRAY_SIZE - 1, sizeof(float), 
                             cudaMemcpyDeviceToHost));
        
        printf("Rank 1: Completed Network->Host->GPU transfer\n");
        printf("Rank 1: First element: %f, Last element: %f\n", first, last);
        printf("Rank 1: Time taken: %f seconds\n", end_time - start_time);
        printf("Rank 1: Data verification: %s\n", 
               (first == 0.0f && last == (float)(ARRAY_SIZE-1)) ? "PASSED" : "FAILED");
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_data));
    free(h_buffer);
    
    MPI_Finalize();
    return 0;
}
