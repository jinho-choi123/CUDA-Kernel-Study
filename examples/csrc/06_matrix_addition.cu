/**
 * Example of Large Matrix Addition
 * NOTE: NUM_ROW > 1024, NUM_COL > 1024
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cu"
#include <stdio.h>

#define NUM_ROW 2048
#define NUM_COL 2048

// number of datas to print
#define PRINT_NUM 32

__global__ void matrixAddition(const float *A, const float *B, float *C,
                               int numElem) {
  // Thread Layout
  // Block: (1024,) threads
  // Grid: (NUM_ROW * NUM_COL / 1024 + 1,) blocks
  int gThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;

  if (gThreadIdx >= numElem) {
    // index out of bound
    return;
  }

  C[gThreadIdx] = A[gThreadIdx] + B[gThreadIdx];
}

int main(void) {
  int result = 0;

  float *a, *b, *c, *goldenC;

  int memSize = sizeof(float) * NUM_ROW * NUM_COL;

  // initialize host data
  a = (float *)calloc(NUM_ROW * NUM_COL, sizeof(float));
  b = (float *)calloc(NUM_ROW * NUM_COL, sizeof(float));
  c = (float *)calloc(NUM_ROW * NUM_COL, sizeof(float));
  goldenC = (float *)calloc(NUM_ROW * NUM_COL, sizeof(float));

  for (int i = 0; i < NUM_ROW * NUM_COL; i++) {
    a[i] = (float)(rand() % 100);
    b[i] = (float)(rand() % 100);
  }

  // Calculate the golden value for c
  for (int i = 0; i < NUM_ROW * NUM_COL; i++) {
    goldenC[i] = a[i] + b[i];
  }

  // Allocate device memory
  float *devA, *devB, *devC;
  CUDA_CHECK(cudaMalloc(&devA, memSize));
  CUDA_CHECK(cudaMalloc(&devB, memSize));
  CUDA_CHECK(cudaMalloc(&devC, memSize));

  // Copy data from host to device
  CUDA_CHECK(cudaMemcpy(devA, a, memSize, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(devB, b, memSize, cudaMemcpyHostToDevice));

  // Launch Kernel
  int threadsPerBlock = 1024;
  int blocksPerGrid = (NUM_ROW * NUM_COL / threadsPerBlock) + 1;

  // Launch Matrix Addition Kernel
  matrixAddition<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC,
                                                     NUM_ROW * NUM_COL);

  // Wait for the kernel to end
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy result from device to host
  CUDA_CHECK(cudaMemcpy(c, devC, memSize, cudaMemcpyDeviceToHost));

  // Print the result
  printf("Vector A: ");
  for (int i = 0; i < PRINT_NUM; i++) {
    printf("%.1f ", a[i]);
  }
  printf("\n");

  printf("Vector B: ");
  for (int i = 0; i < PRINT_NUM; i++) {
    printf("%.1f ", b[i]);
  }
  printf("\n");

  printf("Vector C: ");
  for (int i = 0; i < PRINT_NUM; i++) {
    printf("%.1f ", c[i]);
  }
  printf("\n");

  // Compare the result
  for (int i = 0; i < NUM_ROW * NUM_COL; i++) {
    if (c[i] != goldenC[i]) {
      printf("Error at index %d\n", i);
      result = -1;
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
