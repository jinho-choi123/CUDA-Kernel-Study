/**
* Example of General Matrix Multiplication (GEMM)
* Version 0: Optimized with Shared Memory
* NOTE: MMA for alpha * A(M x K) @ B(K x N) + beta * C(M x N) = C'(M x N)
*/

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cu"
#include <stdio.h>

#define M 4099
#define K 4099
#define N 4099
#define ENABLE_VERIFICATION 0

#define SMEM_CACHE_ELEM_NUM_PER_BLOCK 16 * 16

// number of datas to print
#define PRINT_NUM 0

// Shared Memory Caches for A and B
__shared__ float smem_cache_A[SMEM_CACHE_ELEM_NUM_PER_BLOCK];
__shared__ float smem_cache_B[SMEM_CACHE_ELEM_NUM_PER_BLOCK];

// GEMM kernel
__global__ void gemmV0(float *A, float *B, float *C, float alpha, float beta) {
    // Thread Layout
    // Block: (16, 16) threads
    // Grid: (M + 15) / 16, (N + 15) / 16) blocks

    int rowC = blockDim.y * blockIdx.y + threadIdx.y;
    int colC = blockDim.x * blockIdx.x + threadIdx.x;

    // Row and column index in the block
    int rowBlock = threadIdx.y;
    int colBlock = threadIdx.x;

    // zero the shared memory cache
    smem_cache_A[rowBlock * 16 + colBlock] = 0.0f;
    smem_cache_B[rowBlock * 16 + colBlock] = 0.0f;
    __syncthreads();

    if (rowC >= M || colC >= N) {
        // index out of bound
        return;
    }

    float dotProduct = 0.0f;

    for (int offset = 0; offset < K - (K % 16); offset += 16) {
        // Load the sub-matrix of A and B into shared memory
        smem_cache_A[rowBlock * 16 + colBlock] = A[rowC * K + offset + colBlock];
        smem_cache_B[rowBlock * 16 + colBlock] = B[(rowBlock + offset) * N + colC];

        // wait for all the threads in the block to load the sub-matrix of A and B into shared memory
        __syncthreads();

        // Perform the matmul sub-matrix of A(16 x 16) @ sub-matrix of B(16 x 16) = sub-matrix of C(16 x 16)
        for (int i = 0; i < 16; i++) {
            dotProduct += smem_cache_A[rowBlock * 16 + i] * smem_cache_B[i * 16 + colBlock];
        }
        __syncthreads();
    }

    // Calculate the left-over elements when K is not a multiple of 16
    if (K % 16 != 0) {
        int offset = K - (K % 16);

        if ((offset + colBlock >= K) || (offset + rowBlock >= K)) {
            // index out of bound. store zero to shared memory
            smem_cache_A[rowBlock * 16 + colBlock] = 0.0f;
            smem_cache_B[rowBlock * 16 + colBlock] = 0.0f;
        }
        else {
            smem_cache_A[rowBlock * 16 + colBlock] = A[rowC * K + offset + colBlock];
            smem_cache_B[rowBlock * 16 + colBlock] = B[(rowBlock + offset) * N + colC];
        }

        __syncthreads();

        // Perform the matmul sub-matrix of A(16 x 16) @ sub-matrix of B(16 x 16) = sub-matrix of C(16 x 16)
        for (int i = 0; i < 16; i++) {
            dotProduct += smem_cache_A[rowBlock * 16 + i] * smem_cache_B[i * 16 + colBlock];
        }
        __syncthreads();
    }
    // store the result
    // NOTE: caching C into shared memory is not necessary for performance. So we directly use the global memory here.
    C[rowC * N + colC] = dotProduct * alpha + beta * C[rowC * N + colC];
}

int main(void) {
    int result = 0;

    float *a, *b, *c, *goldenC;

    cudaEvent_t start, end;

    int memSizeA = sizeof(float) * M * K;
    int memSizeB = sizeof(float) * K * N;
    int memSizeC = sizeof(float) * M * N;

    // initialize host data
    a = (float *)calloc(M * K, sizeof(float));
    b = (float *)calloc(K * N, sizeof(float));
    c = (float *)calloc(M * N, sizeof(float));
    goldenC = (float *)calloc(M * N, sizeof(float));

    for (int i = 0; i < M * K; i++) {
         a[i] = (float)(rand() % 100);
         // a[i] = 1.0f;
    }
    for (int i = 0; i < K * N; i++) {
        b[i] = (float)(rand() % 100);
        // b[i] = 1.0f;
    }

    // Calculate the golden value for c
    if (ENABLE_VERIFICATION) {
        printf("Calculating golden value for c...\n");
        for (int row = 0; row < M; row++) {
            for (int col = 0; col < N; col++) {
                goldenC[row * N + col] = 0.0f;
                for (int offset = 0; offset < K; offset++) {
                    goldenC[row * N + col] += a[row * K + offset] * b[offset * N + col];
                }
            }
        }
        printf("Golden value for c calculated\n");
    }

    // Allocate device memory
    float *devA, *devB, *devC;
    CUDA_CHECK(cudaMalloc(&devA, memSizeA));

    CUDA_CHECK(cudaMalloc(&devB, memSizeB));
    CUDA_CHECK(cudaMalloc(&devC, memSizeC));
    CUDA_CHECK(cudaMemset(devC, 0, memSizeC));

    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(devA, a, memSizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(devB, b, memSizeB, cudaMemcpyHostToDevice));

    // Profiling the kernel
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    // Launch Kernel
    printf("Launching kernel...\n");
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((M + 15) / 16, (N + 15) / 16);
    CUDA_CHECK(cudaEventRecord(start));
    gemmV0<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC, 1.0f, 0.0f);
    CUDA_CHECK(cudaEventRecord(end));

    // Wait for the kernel to end
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Kernel launch completed\n");

    float elapsedTime;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, end));
    printf("Kernel execution time: %f ms\n", elapsedTime);

    // Copy result from device to host
    CUDA_CHECK(cudaMemcpy(c, devC, memSizeC, cudaMemcpyDeviceToHost));

    // Print the result
    printf("Result: ");
    for (int i = 0; i < PRINT_NUM; i++) {
        printf("%.1f ", c[i]);
    }
    printf("\n");

    // Compare the result
    if (ENABLE_VERIFICATION) {
        printf("Comparing result with golden value...\n");
    for (int i = 0; i < M * N; i++) {
        if (c[i] != goldenC[i]) {
            printf("Error at index %d: c[%d] = %.1f, goldenC[%d] = %.1f\n", i, i, c[i], i, goldenC[i]);
            result = -1;
            }
        }
        printf("Comparison completed\n");
    }

    // Free device memory
    CUDA_CHECK(cudaFree(devA));
    CUDA_CHECK(cudaFree(devB));
    CUDA_CHECK(cudaFree(devC));

    // Free host memory
    free(a);
    free(b);
    free(c);
    free(goldenC);

    return result;
}
