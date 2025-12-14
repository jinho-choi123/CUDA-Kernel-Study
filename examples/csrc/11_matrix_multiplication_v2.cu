/**
 * Example of Matrix Multiplication
 * Version 1: Optimized with Shared Memory
 * NOTE: MMA for A(M x K) @ B(K x N) = C(M x N)
 * Shared Memory is used to store the sub-matrix of A and B.
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cu"
#include <stdio.h>

#define M 4096
#define K 4096
#define N 4096
#define SMEM_CACHE_ELEM_NUM_PER_BLOCK                                          \
  1024 // we are going to use (32, 32) threads in a block
#define ENABLE_VERIFICATION 0

// number of datas to print
#define PRINT_NUM 0

// Shared Memory Caches for A and B
__shared__ float smem_cache_A[SMEM_CACHE_ELEM_NUM_PER_BLOCK];
__shared__ float smem_cache_B[SMEM_CACHE_ELEM_NUM_PER_BLOCK];

// Matrix Multiplication Kernel using 2D Grid-layout and 2D Block-layout
__global__ void matrixMultiplicationV2(float *A, float *B, float *C) {
  // Thread Layout
  // Block: (32, 32) threads
  // Grid: (M / 32, N / 32) blocks
  int rowC = blockDim.y * blockIdx.y + threadIdx.y;
  int colC = blockDim.x * blockIdx.x + threadIdx.x;

  int rowBlock = threadIdx.y;
  int colBlock = threadIdx.x;

  if (rowC >= M || colC >= N) {
    // index out of bound
    return;
  }

  // ASSUME the block size is (32, 32) threads
  float dotProduct = 0.0f;
  for (int offset = 0; offset < K; offset += 32) {
    // load the sub-matrix of A into shared memory
    smem_cache_A[rowBlock * 32 + colBlock] = A[rowC * K + colBlock + offset];
    smem_cache_B[rowBlock * 32 + colBlock] = B[(rowBlock + offset) * N + colC];

    // wait for all threads to load the sub-matrix of A and B into shared memory
    __syncthreads();

    // perform the matrix multiplication
    for (int i = 0; i < 32; i++) {
      dotProduct +=
          smem_cache_A[rowBlock * 32 + i] * smem_cache_B[i * 32 + colBlock];
    }

    __syncthreads();
  }

  // store the result
  C[rowC * N + colC] = dotProduct;
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
    // a[i] = (float)(rand() % 100);
    a[i] = 1.0f;
  }
  for (int i = 0; i < K * N; i++) {
    // b[i] = (float)(rand() % 100);
    b[i] = 1.0f;
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

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(devA, a, memSizeA, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(devB, b, memSizeB, cudaMemcpyHostToDevice));

  // Profiling the kernel
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&end));

  // Launch Kernel
  printf("Launching kernel...\n");
  dim3 threadsPerBlock(32, 32);
  dim3 blocksPerGrid((M + 31) / 32, (N + 31) / 32);
  CUDA_CHECK(cudaEventRecord(start));
  matrixMultiplicationV2<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC);
  CUDA_CHECK(cudaEventRecord(end));

  // Wait for the kernel to end
  CUDA_CHECK(cudaDeviceSynchronize());
  printf("Kernel launch completed\n");

  float elapsedTime;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, end));
  printf("Time taken for matrix multiplication kernel: %f ms\n", elapsedTime);

  // Copy result from device to host
  CUDA_CHECK(cudaMemcpy(c, devC, memSizeC, cudaMemcpyDeviceToHost));
  printf("Result copied from device to host\n");

  // Print the result
  printf("Matrix A: ");
  for (int i = 0; i < PRINT_NUM; i++) {
    printf("%.1f ", a[i]);
  }
  printf("\n");

  printf("Matrix B: ");
  for (int i = 0; i < PRINT_NUM; i++) {
    printf("%.1f ", b[i]);
  }
  printf("\n");

  printf("Matrix C: ");
  for (int i = 0; i < PRINT_NUM; i++) {
    printf("%.1f ", c[i]);
  }
  printf("\n");

  // Compare the result
  if (ENABLE_VERIFICATION) {
    for (int i = 0; i < M * N; i++) {
      if (c[i] != goldenC[i]) {
        printf("Error at index %d: c[%d] = %.1f, goldenC[%d] = %.1f\n", i, i,
               c[i], i, goldenC[i]);
        result = -1;
      }
    }
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
