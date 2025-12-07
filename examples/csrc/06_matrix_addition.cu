/**
 * Example of Large Matrix Addition
 * NOTE: NUM_ROW > 1024, NUM_COL > 1024
 */

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "utils.cu"
#include <stdio.h>

#define NUM_ROW 4096
#define NUM_COL 4096

// number of datas to print
#define PRINT_NUM 32

// Matrix Addition Kernel using 1D Grid-layout and 1D Block-layout
__global__ void matrixAddition1(const float *A, const float *B, float *C,
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

// Matrix Addition Kernel using 2D Grid-layout and 2D Block-layout
__global__ void matrixAddition2(const float *A, const float *B, float *C,
                                int numElem) {
  // Thread Layout
  // Block: (32, 32) threads
  // Grid: (NUM_ROW / 32, NUM_COL / 32) blocks
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;

  if (col >= NUM_COL || row >= NUM_ROW) {
    // index out of bound
    return;
  }

  C[row * NUM_COL + col] = A[row * NUM_COL + col] + B[row * NUM_COL + col];
}

// Matrix Addition Kernel using 2D Grid-layout and 1D Block-layout
__global__ void matrixAddition3(const float *A, const float *B, float *C,
                                int numElem) {
  // Thread Layout
  // Block: (1024, 1) threads
  // Grid: (NUM_ROW / 1024, NUM_COL) blocks
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockIdx.y;

  if (col >= NUM_COL || row >= NUM_ROW) {
    // index out of bound
    return;
  }

  C[row * NUM_COL + col] = A[row * NUM_COL + col] + B[row * NUM_COL + col];
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

  // Launch Matrix Addition Kernel
  // You can choose any of the following kernels to run

  // 1D Grid-layout and 1D Block-layout
  // START: 1D Grid-layout and 1D Block-layout
  // int threadsPerBlock = 1024;
  // int blocksPerGrid =
  //     ((NUM_ROW * NUM_COL + threadsPerBlock - 1) / threadsPerBlock) + 1;
  // matrixAddition1<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC,
  //                                                     NUM_ROW * NUM_COL);
  // END: 1D Grid-layout and 1D Block-layout

  // 2D Grid-layout and 2D Block-layout
  // START: 2D Grid-layout and 2D Block-layout
  // dim3 threadsPerBlock(32, 32);
  // dim3 blocksPerGrid((NUM_ROW + 31) / 32, (NUM_COL + 31) / 32);
  // matrixAddition2<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC,
  //                                                     NUM_ROW * NUM_COL);
  // END: 2D Grid-layout and 2D Block-layout

  // 2D Grid-layout and 1D Block-layout
  // START: 2D Grid-layout and 1D Block-layout
  dim3 threadsPerBlock(1024, 1);
  dim3 blocksPerGrid((NUM_ROW + 1023) / 1024, NUM_COL);
  matrixAddition3<<<blocksPerGrid, threadsPerBlock>>>(devA, devB, devC,
                                                      NUM_ROW * NUM_COL);
  // END: 2D Grid-layout and 1D Block-layout
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
