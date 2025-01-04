#include <stdio.h>
#include <cuda.h>
#include <cuda_runtimes.h>
#include <sys/time.h>
#include "cuda_common.cuh"

typedef int EL_TYPE;

// this goes to the GPU
// This is CUDA c code
__global__ void cuda_vector_add_simple(EL_TYPE *OUT, EL_TYPE *A, EL_TYPE *B, int N){
    
    // N threads are identified and assigned a unique identifier for each
    // We are telling which item each thread should prcess (same as its identifier)
    // First thread --> i=0
    // Second thread --> i=1 ...
    // Single instruction multiple thread
    // Means: Same instruction is executed and value of variable may be different
    int i = threadIdx.x;

    // For all threads the if statement is True, instuctions inside the if are executed
    // For all threads the if statement is False, STILL ENTER THE LOOP because controller
    // is the same for all threads (so same instruction has to be executed) so they cannot not enter the loop
    // but no operation is done inside the loop i.e. idle
    if (i < N){
        OUT[i] = A[i] + B[i];
    }
}


void test_vector_add(int N){
    EL_TYPE *A, *B, *OUT;
    EL_TYPE *d_A, *d_B, *d_OUT;

    // Allocate the vectors on the host device
    A = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);
    B = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);
    OUT = (EL_TYPE *)malloc(sizeof(EL_TYPE) * N);
    
    // Initialize the vectors
    for (int i = 0; i < N; i++){
        A[i] = rand() % 100;
        B[i] = rand() % 100;
    }

    // Allocate device memory for a
    CUDA_CHECK(cudaMalloc((void **)&d_A, sizeof(EL_TYPE) * N));
    CUDA_CHECK(cudaMalloc((void **)&d_B, sizeof(EL_TYPE) * N));
    CUDA_CHECK(cudaMalloc((void **)&d_OUT, sizeof(EL_TYPE) * N));

    // Transfoer the vectors to the device (GPU)
    CUDA_CHECK(cudaMemcpy(d_A, A, sizeof(EL_TYPE) * N, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, B, sizeof(EL_TYPE) * N, cudaMemcpyHostToDevice));

    // Call the kernel
    CUDA_CHECK(cudaEventCreate(&start_kernel));
    CUDA_CHECK(cudaEventCreate(&stop_kernel));

    CUDA_CHECK(cudaEventRecord(start_kernel));

    // Launch the kernel 
    // i.e. launch a program that the GPU should excecute in parallel across threads/cores
    // 1 tells how many blocks we have
    // N tells how many threads we have for each block
    cuda_vector_add_simple<<<1, N>>>(d_OUT, d_A, d_B, N);
    CUDA_CHECK(cudaEventRecord(stop_kernel));
}